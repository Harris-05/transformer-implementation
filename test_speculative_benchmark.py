import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from speculative_report import summarize_trials


class SpeculativeBenchmarkTests(unittest.TestCase):
    def test_saved_failed_preflight_is_never_overwritten_by_a_restart(self):
        from benchmark_speculative import experiment
        from test_speculative import TransitionModel
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            path = output / 'preflight.json'
            saved = [{'case_id': 'sample', 'passed': False}]
            path.write_text(json.dumps(saved), encoding='utf-8')
            original = path.read_bytes()
            model = TransitionModel(lambda token: 1)
            with patch('benchmark_speculative.preflight', return_value={
                    'case_id': 'sample', 'passed': True, 'greedy_token_ids': [1, 1]}) as check:
                with self.assertRaises(RuntimeError):
                    experiment(model, model, [{'case_id': 'sample', 'token_ids': [1],
                               'kind': 'fixed_token_stress', 'output_budget': 2, 'eos_token_id': None}],
                               {'draft_lengths': [2], 'repetitions': 1, 'warmups': 0}, output, 'test')
                check.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def cases(self):
        return [{'case_id': 'sample', 'kind': 'fixed_token_stress', 'output_budget': 2}]

    def trial(self, mode, repeat=0, seconds=1, equal=True):
        result = {'token_ids': [3, 4] if equal else [3, 5], 'total_seconds': seconds,
            'initial_allocated_gib': None, 'peak_allocated_gib': None, 'peak_reserved_gib': None,
            'target_forward_calls': 2 if mode == 'greedy' else 1, 'draft_forward_calls': 0 if mode == 'greedy' else 3}
        if mode != 'greedy':
            result.update(proposed_tokens=4, accepted_tokens=2)
        return {'case_id': 'sample', 'mode': mode, 'repeat': repeat,
                'matches_greedy': equal, 'result': result}

    def test_summary_retains_slowdown_acceptance_and_unmeasured_memory(self):
        trials = [self.trial('greedy', seconds=1), self.trial('speculative_2', seconds=2)]
        result = summarize_trials(self.cases(), trials, [2], 1)
        self.assertTrue(result['complete'])
        self.assertTrue(result['passed'])
        row = result['cases'][0]['modes']['speculative_2']
        self.assertEqual(row['speedup_over_greedy'], 0.5)
        self.assertEqual(row['draft_acceptance_rate'], 0.5)
        self.assertEqual(row['target_forward_calls_median'], 1)
        self.assertIsNone(row['peak_allocated_gib_max'])

    def test_missing_trials_or_mismatch_cannot_pass_or_show_speedup(self):
        incomplete = summarize_trials(self.cases(), [self.trial('greedy')], [2], 1)
        self.assertFalse(incomplete['complete'])
        failed = summarize_trials(self.cases(), [self.trial('greedy'), self.trial('speculative_2', equal=False)], [2], 1)
        self.assertFalse(failed['passed'])
        self.assertIsNone(failed['cases'][0]['modes']['speculative_2']['speedup_over_greedy'])

    def test_duplicate_wrong_mode_or_invalid_time_is_rejected(self):
        trial = self.trial('greedy')
        for trials in ([trial, trial], [self.trial('unknown')], [self.trial('greedy', seconds=0)]):
            with self.assertRaises(ValueError):
                summarize_trials(self.cases(), trials, [2], 1)


if __name__ == '__main__':
    unittest.main()
