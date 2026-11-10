"""Benchmark checks exercise real tiny-model paths and aggregate saved trials."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from benchmark_kv_cache import controlled_cases, compare_paths, summarize_trials
import test_decoding


class KVBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_decoding.DecodingTests.setUpClass()
        cls.model = test_decoding.DecodingTests.model

    def test_math_backend_controls_all_paths_and_restores_backend_flags(self):
        from benchmark_kv_cache import attention_backend
        flags = lambda: (torch.backends.cuda.flash_sdp_enabled(),
                         torch.backends.cuda.mem_efficient_sdp_enabled(),
                         torch.backends.cuda.math_sdp_enabled())
        before = flags()
        with attention_backend('math'):
            self.assertEqual(flags(), (False, False, True))
            result = compare_paths(self.model, torch.tensor([[1, 3, 5, 7]]),
                                   budget=6, eos_token_id=None)
            self.assertTrue(result['passed'])
        self.assertEqual(flags(), before)
        with self.assertRaises(ValueError):
            attention_backend('unsupported')

    def test_controlled_cases_cross_lengths_without_early_eos_or_false_clinical_claims(self):
        cases = controlled_cases([1, 4, 9], [3, 7, 12], [1, 2, 5])
        self.assertEqual(len(cases), 9)
        case = next(c for c in cases if c['input_tokens'] == 7 and c['output_budget'] == 2)
        self.assertEqual(case['token_ids'], [1, 4, 9, 1, 4, 9, 1])
        self.assertEqual(case['kind'], 'fixed_token_stress')
        self.assertIsNone(case['eos_token_id'])
        with self.assertRaises(ValueError):
            controlled_cases([], [3], [2])

    def test_real_cached_uncached_and_hf_comparison_retains_tokens_and_trace(self):
        inputs = torch.tensor([[1, 3, 5, 7, 9]])
        result = compare_paths(self.model, inputs, budget=5, eos_token_id=None)
        self.assertTrue(result['passed'])
        self.assertEqual(result['cached']['token_ids'], result['hf_token_ids'])
        self.assertEqual(len(result['cached']['trace']), 5)
        self.assertIsNone(result['first_divergence']['cached_vs_uncached'])

    def test_fixed_work_reference_does_not_restore_the_models_default_eos(self):
        inputs = torch.tensor([[1, 3, 5, 7, 9]])
        from decoding import greedy_decode
        first = greedy_decode(self.model, inputs, max_new_tokens=1,
                              use_cache=False, eos_token_id=None).token_ids[0]
        original = self.model.generation_config.eos_token_id
        self.model.generation_config.eos_token_id = first
        try:
            result = compare_paths(self.model, inputs, budget=5, eos_token_id=None)
            self.assertEqual(len(result['hf_token_ids']), 5)
            self.assertTrue(result['passed'])
        finally:
            self.model.generation_config.eos_token_id = original

    def trial(self, mode, repeat, seconds, tokens=None):
        return {'case_id': 'p3-o2', 'mode': mode, 'repeat': repeat,
                'result': {'token_ids': [4, 5] if tokens is None else tokens,
                    'prefill_seconds': seconds / 4, 'decode_seconds': seconds * 3 / 4,
                    'total_seconds': seconds, 'decode_steps': 1,
                    'peak_allocated_gib': None, 'peak_reserved_gib': None,
                    'initial_allocated_gib': None, 'cache_bytes': 0, 'cache_tokens': 0}}

    def test_summary_uses_all_repeats_and_marks_missing_work_incomplete(self):
        case = controlled_cases([1], [3], [2])[0]
        trials = [self.trial('cached', i, t) for i, t in enumerate([1, 2, 3])]
        trials += [self.trial('uncached', i, t) for i, t in enumerate([3, 4, 5])]
        summary = summarize_trials([case], trials, repeats=3)
        self.assertTrue(summary['complete'])
        group = summary['cases'][0]
        self.assertTrue(group['tokens_equal'])
        self.assertEqual(group['modes']['cached']['total_seconds_median'], 2)
        self.assertEqual(group['modes']['cached']['total_seconds_min'], 1)
        self.assertEqual(group['modes']['cached']['total_seconds_max'], 3)
        self.assertEqual(group['total_speedup_uncached_over_cached'], 2)
        self.assertFalse(summarize_trials([case], trials[:-1], repeats=3)['complete'])

    def test_mismatch_or_short_stress_output_cannot_be_reported_as_pass(self):
        case = controlled_cases([1], [3], [2])[0]
        wrong = [self.trial('cached', 0, 1, [4]), self.trial('uncached', 0, 2)]
        result = summarize_trials([case], wrong, repeats=1)
        self.assertTrue(result['complete'])
        self.assertFalse(result['cases'][0]['tokens_equal'])
        self.assertFalse(result['cases'][0]['fixed_length_satisfied'])
        self.assertFalse(result['passed'])

    def test_duplicate_or_unexpected_trials_are_rejected_on_resume(self):
        case = controlled_cases([1], [3], [2])[0]
        duplicate = self.trial('cached', 0, 1)
        with self.assertRaises(ValueError):
            summarize_trials([case], [duplicate, duplicate], repeats=1)
        wrong = {**duplicate, 'case_id': 'unknown'}
        with self.assertRaises(ValueError):
            summarize_trials([case], [wrong], repeats=1)

    def test_timed_tokens_must_also_match_the_untimed_reference(self):
        case = controlled_cases([1], [3], [2])[0]
        trials = [self.trial('cached', 0, 1), self.trial('uncached', 0, 2)]
        trials[0]['matches_reference'] = False
        self.assertFalse(summarize_trials([case], trials, repeats=1)['passed'])

    def test_missing_clinical_correctness_cannot_claim_complete(self):
        from kv_benchmark_report import combine_results
        summary = {'complete': True, 'passed': True}
        result = combine_results(summary, [{'case_id': 'stress', 'passed': True}],
                                 expected_ids={'stress', 'clinical'})
        self.assertFalse(result['complete'])
        self.assertFalse(result['passed'])

    def test_interruption_before_checkpoint_replace_preserves_previous_trials(self):
        from benchmark_kv_cache import save_trials
        import json
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trials.jsonl'
            previous = [self.trial('cached', 0, 1)]
            save_trials(path, previous)
            self.assertEqual([json.loads(l) for l in path.read_text().splitlines()], previous)
            following = previous + [self.trial('uncached', 0, 2)]
            # Simulate interruption after writing the temporary file, before commit.
            with patch.object(Path, 'replace', side_effect=OSError('simulated interrupted commit')):
                with self.assertRaises(OSError):
                    save_trials(path, following)
            self.assertEqual([json.loads(l) for l in path.read_text().splitlines()], previous)
            save_trials(path, following)
            self.assertEqual([json.loads(l) for l in path.read_text().splitlines()], following)


if __name__ == '__main__':
    unittest.main()
