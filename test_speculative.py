"""Forced acceptance/rejection branches plus actual miniature Transformer pairs."""

import unittest
from types import SimpleNamespace

from speculative_decoding import crop_cache, speculative_decode

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from decoding import greedy_decode


class TransitionModel(torch.nn.Module):
    """Controlled causal LM with real cache tensors, not a mocked decoder."""
    def __init__(self, rule):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, max_position_embeddings=128, pad_token_id=0, eos_token_id=31)
        self.rule = rule
        self.calls = []
        self.eval()

    def forward(self, input_ids, attention_mask, position_ids, past_key_values=None,
                use_cache=True, logits_to_keep=1):
        old = past_key_values.get_seq_length() if past_key_values is not None else 0
        if attention_mask.shape[1] != old + input_ids.shape[1]:
            raise ValueError('Mask must include all past and current tokens.')
        expected = torch.arange(old, old + input_ids.shape[1]).unsqueeze(0)
        if not torch.equal(position_ids.cpu(), expected):
            raise ValueError('Incorrect absolute positions.')
        prior = (past_key_values.layers[0].keys.flatten().tolist()
                 if old else [])
        self.calls.append({'past': prior, 'input': input_ids.flatten().tolist()})
        if use_cache:
            states = input_ids.reshape(1, 1, -1, 1).float()
            past_key_values.update(states, states.clone(), 0)
        values = input_ids[:, -logits_to_keep:] if logits_to_keep else input_ids
        logits = torch.full((*values.shape, self.config.vocab_size), -100.0)
        for index, token in enumerate(values[0].tolist()):
            logits[0, index, self.rule(token)] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values if use_cache else None)


class SpeculativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        return torch.tensor([[0, 1]], dtype=torch.long)

    def target(self):
        return TransitionModel(lambda token: (token + 1) % 32)

    def test_all_candidates_accepted_bonus_tokens_and_target_forward_reduction(self):
        target, draft = self.target(), self.target()
        result = speculative_decode(target, draft, self.inputs(), max_new_tokens=13,
                                    draft_length=4, eos_token_id=None, trace=True)
        self.assertEqual(result.token_ids, list(range(2, 15)))
        self.assertEqual(result.accepted_tokens, result.proposed_tokens)
        self.assertGreater(result.bonus_tokens, 0)
        self.assertLess(result.target_forward_calls, len(result.token_ids))
        self.assertEqual(result.target_cache_tokens, 14)
        self.assertEqual(result.draft_cache_tokens, 14)
        self.assertEqual(target.calls[0]['input'], [0, 1, 2, 3, 4, 5])
        self.assertTrue(all(r['target_cache_after'] == r['committed_tokens'] - 1
                            == r['draft_cache_after'] for r in result.trace))

    def test_zero_acceptance_correction_and_progress(self):
        target, draft = self.target(), TransitionModel(lambda token: 0)
        result = speculative_decode(target, draft, self.inputs(), max_new_tokens=9,
                                    draft_length=4, eos_token_id=None, trace=True)
        self.assertEqual(result.token_ids, list(range(2, 11)))
        self.assertEqual(result.accepted_tokens, 0)
        self.assertEqual(result.rejection_rounds, 9)
        self.assertEqual(result.target_forward_calls, 9)

    def test_middle_rejection_discards_unaccepted_suffix_and_synchronizes_caches(self):
        target = self.target()
        draft = TransitionModel(lambda token: 20 if token == 3 else (token + 1) % 32)
        result = speculative_decode(target, draft, self.inputs(), max_new_tokens=11,
                                    draft_length=4, eos_token_id=None, trace=True)
        self.assertEqual(result.token_ids, list(range(2, 13)))
        self.assertEqual(result.trace[0]['accepted'], 2)
        self.assertTrue(result.trace[0]['rejected'])
        for model in (target, draft):
            self.assertTrue(all(20 not in call['past'] for call in model.calls))
        self.assertEqual(result.target_cache_tokens, result.draft_cache_tokens)

    def test_eos_acceptance_and_rejection_correction_stop_without_extra_tokens(self):
        for draft in (self.target(), TransitionModel(lambda token: 0)):
            result = speculative_decode(self.target(), draft, self.inputs(), max_new_tokens=10,
                                        draft_length=4, eos_token_id=2)
            self.assertEqual(result.token_ids, [2])
            self.assertTrue(result.stopped_on_eos)
            self.assertEqual(result.target_cache_tokens, 2)
            self.assertEqual(result.draft_cache_tokens, 2)
        result = speculative_decode(self.target(), self.target(), self.inputs(), max_new_tokens=10,
                                    draft_length=2, eos_token_id=4)
        self.assertEqual(result.token_ids, [2, 3, 4])  # EOS arrives as target bonus.
        self.assertTrue(result.stopped_on_eos)

    def test_remaining_budget_and_no_cache_leak_across_calls(self):
        target, draft = self.target(), self.target()
        inputs = self.inputs()
        saved = inputs.clone()
        for budget in (1, 2, 3, 7):
            result = speculative_decode(target, draft, inputs, max_new_tokens=budget,
                                        draft_length=8, eos_token_id=None)
            self.assertEqual(result.token_ids, list(range(2, 2 + budget)))
            self.assertEqual(result.target_cache_tokens, 2 + budget - 1)
            self.assertEqual(result.draft_cache_tokens, 2 + budget - 1)
        self.assertTrue(torch.equal(inputs, saved))

    def test_real_qwen_pairs_match_ordinary_greedy_across_lengths_and_draft_sizes(self):
        def tiny(seed):
            torch.manual_seed(seed)
            config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=128, pad_token_id=0, eos_token_id=63)
            config._attn_implementation = 'sdpa'
            return Qwen3ForCausalLM(config).double().eval()
        target, draft = tiny(31), tiny(77)
        for length, budget in ((3, 1), (7, 9), (17, 15)):
            inputs = torch.arange(1, length + 1).unsqueeze(0)
            expected = greedy_decode(target, inputs, max_new_tokens=budget, use_cache=True,
                                     eos_token_id=None).token_ids
            for draft_length in (1, 2, 4):
                with self.subTest(length=length, budget=budget, draft_length=draft_length):
                    actual = speculative_decode(target, draft, inputs, max_new_tokens=budget,
                        draft_length=draft_length, eos_token_id=None, measure=True)
                    self.assertEqual(actual.token_ids, expected)
                    self.assertGreater(actual.total_seconds, 0)
                    self.assertIsNone(actual.peak_allocated_gib)

    def test_bad_inputs_and_unsupported_pair_fail(self):
        target, draft = self.target(), self.target()
        for values in ({'max_new_tokens': 0}, {'max_new_tokens': 1.5},
                       {'max_new_tokens': 4, 'draft_length': 0}, {'max_new_tokens': 4, 'draft_length': True}):
            with self.assertRaises(ValueError):
                speculative_decode(target, draft, self.inputs(), **values)
        with self.assertRaises(ValueError):
            speculative_decode(target, draft, self.inputs().float(), max_new_tokens=4)
        with self.assertRaises(ValueError):
            speculative_decode(target, draft, self.inputs().repeat(2, 1), max_new_tokens=4)
        draft.config.vocab_size = 99
        with self.assertRaises(ValueError):
            speculative_decode(target, draft, self.inputs(), max_new_tokens=4)


if __name__ == '__main__':
    unittest.main()
