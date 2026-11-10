"""Actual miniature Qwen3 forwards; no downloads or clinical claims."""

import unittest

import torch
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from decoding import first_divergence, greedy_decode


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, pad_token_id=0,
            eos_token_id=63, bos_token_id=1, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        cls.model = Qwen3ForCausalLM(config).eval()

    def prompt(self, length=7):
        return (torch.arange(length, dtype=torch.long).reshape(1, -1) % 60) + 1

    def test_three_lengths_match_uncached_and_independent_hf_reference(self):
        for length, budget in ((3, 1), (9, 5), (23, 11)):
            with self.subTest(length=length, budget=budget):
                inputs = self.prompt(length)
                cached = greedy_decode(self.model, inputs, max_new_tokens=budget,
                                       use_cache=True, eos_token_id=None)
                reference = greedy_decode(self.model, inputs, max_new_tokens=budget,
                                          use_cache=False, eos_token_id=None)
                config = GenerationConfig(do_sample=False, num_beams=1,
                    max_new_tokens=budget, eos_token_id=None, pad_token_id=0,
                    use_cache=True)
                with torch.inference_mode():
                    hf = self.model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                             generation_config=config)
                self.assertEqual(cached.token_ids, reference.token_ids)
                self.assertEqual(cached.token_ids, hf[0, length:].tolist())
                self.assertEqual(len(cached.token_ids), budget)
                self.assertFalse(cached.stopped_on_eos)
                self.assertEqual(cached.decode_steps, budget - 1)

    def test_actual_forward_cache_growth_positions_and_mask_lengths(self):
        calls = []

        def observe(module, args, kwargs):
            cache = kwargs.get("past_key_values")
            calls.append({"input": kwargs["input_ids"].shape[1],
                "positions": kwargs["position_ids"].tolist()[0],
                "mask": kwargs["attention_mask"].shape[1],
                "past": cache.get_seq_length() if cache is not None else 0,
                "cache": kwargs["use_cache"]})

        handle = self.model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            cached = greedy_decode(self.model, self.prompt(), max_new_tokens=4,
                use_cache=True, eos_token_id=None, trace=True)
        finally:
            handle.remove()
        self.assertEqual([x["input"] for x in calls], [7, 1, 1, 1])
        self.assertEqual([x["positions"] for x in calls], [list(range(7)), [7], [8], [9]])
        self.assertEqual([x["mask"] for x in calls], [7, 8, 9, 10])
        self.assertEqual([x["past"] for x in calls], [0, 7, 8, 9])
        self.assertEqual([x["cache_tokens"] for x in cached.trace], [7, 8, 9, 10])
        self.assertEqual(cached.cache_tokens, 10)
        # Two layers, K and V, two KV heads, ten tokens, head dimension eight, FP32.
        self.assertEqual(cached.cache_bytes, 2 * 2 * 2 * 10 * 8 * 4)
        uncached = greedy_decode(self.model, self.prompt(), max_new_tokens=4,
            use_cache=False, eos_token_id=None, trace=True)
        self.assertEqual([x["input_tokens"] for x in uncached.trace], [7, 8, 9, 10])
        self.assertEqual(uncached.cache_bytes, 0)
        self.assertEqual(uncached.cache_tokens, 0)

    def test_eos_is_returned_and_stops_even_at_first_token(self):
        first = greedy_decode(self.model, self.prompt(), max_new_tokens=1,
                              use_cache=False, eos_token_id=None).token_ids[0]
        for cached in (False, True):
            result = greedy_decode(self.model, self.prompt(), max_new_tokens=9,
                use_cache=cached, eos_token_id=[first, 63], measure=True)
            self.assertEqual(result.token_ids, [first])
            self.assertTrue(result.stopped_on_eos)
            self.assertEqual(result.decode_steps, 0)
            self.assertEqual(result.decode_seconds, 0.0)
            self.assertGreater(result.prefill_seconds, 0)
            self.assertEqual(result.total_seconds, result.prefill_seconds)
            self.assertIsNone(result.peak_allocated_gib)

    def test_cache_is_private_and_inputs_unchanged_across_calls(self):
        inputs = self.prompt()
        before = inputs.clone()
        first = greedy_decode(self.model, inputs, max_new_tokens=6,
                              use_cache=True, eos_token_id=None)
        greedy_decode(self.model, self.prompt(15), max_new_tokens=3,
                      use_cache=True, eos_token_id=None)
        again = greedy_decode(self.model, inputs, max_new_tokens=6,
                              use_cache=True, eos_token_id=None)
        self.assertEqual(first.token_ids, again.token_ids)
        self.assertTrue(torch.equal(before, inputs))

    def test_bad_shape_mask_dtype_or_budget_fails_explicitly(self):
        invalid = [self.prompt()[0], torch.ones((2, 3), dtype=torch.long),
                   torch.empty((1, 0), dtype=torch.long), self.prompt().float()]
        for inputs in invalid:
            with self.assertRaises(ValueError):
                greedy_decode(self.model, inputs, max_new_tokens=2, use_cache=True)
        for budget in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                greedy_decode(self.model, self.prompt(), max_new_tokens=budget, use_cache=True)
        with self.assertRaises(ValueError):
            greedy_decode(self.model, self.prompt(), max_new_tokens=2,
                          use_cache=True, attention_mask=torch.zeros_like(self.prompt()))
        with self.assertRaises(ValueError):
            greedy_decode(self.model, self.prompt(125), max_new_tokens=8, use_cache=True)

    def test_first_divergence_includes_prefix_length_mismatch(self):
        self.assertIsNone(first_divergence([1, 2], [1, 2]))
        self.assertEqual(first_divergence([1, 2], [1, 3]), 1)
        self.assertEqual(first_divergence([1], [1, 2]), 1)


if __name__ == "__main__":
    unittest.main()
