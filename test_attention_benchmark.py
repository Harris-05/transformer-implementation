import unittest
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmark_attention import aggregate_measurements, error_metrics, score_workspace_bytes


class AttentionBenchmarkTests(unittest.TestCase):
    def config(self):
        return json.loads(Path(__file__).with_name('configs').joinpath('attention_config.json').read_text(encoding='utf-8'))

    def test_empty_duplicate_or_bad_config_cannot_claim_success(self):
        from benchmark_attention import validate_config
        for changes in ({'lengths': []}, {'lengths': [8, 8]}, {'lengths': [0]},
                        {'causal_modes': []}, {'causal_modes': [1]}, {'causal_modes': [True, True]},
                        {'warmups': -1}, {'repetitions': 0}, {'atol': float('nan')}):
            with self.assertRaises(ValueError):
                validate_config({**self.config(), **changes})

    def test_complete_metrics_can_regenerate_missing_reports_without_remeasurement(self):
        import benchmark_attention
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'configs').mkdir()
            config = {**self.config(), 'lengths': [4], 'causal_modes': [True],
                      'batch': 1, 'heads': 1, 'head_dim': 2, 'repetitions': 1, 'warmups': 0}
            (root / 'configs/attention_config.json').write_text(json.dumps(config), encoding='utf-8')
            for name in ('attention.py', 'benchmark_attention.py', 'attention_benchmark_report.py'):
                shutil.copyfile(Path(__file__).with_name(name), root / name)
            output = root / 'output'
            arguments = SimpleNamespace(root=root, device='cpu', output=output)
            benchmark_attention.run(arguments)
            original = (output / 'metrics.json').read_bytes()
            (output / 'summary.md').unlink()
            with patch.object(benchmark_attention, 'run_case', side_effect=AssertionError('must not remeasure')):
                benchmark_attention.run(arguments)
            self.assertTrue((output / 'summary.md').is_file())
            self.assertEqual((output / 'metrics.json').read_bytes(), original)

    def records(self):
        return [{'method': name, 'repeat': repeat, 'seconds': seconds,
                 'initial_allocated_bytes': None, 'peak_allocated_bytes': None,
                 'peak_reserved_bytes': None}
                for name, samples in [('reference', [1, 2, 3]), ('tiled', [2, 4, 6]), ('sdpa_auto', [0.5, 1, 1.5])]
                for repeat, seconds in enumerate(samples)]

    def test_repeated_summary_retains_slowdown_ranges_and_missing_memory(self):
        result = aggregate_measurements(self.records(), repetitions=3)
        self.assertTrue(result['complete'])
        self.assertEqual(result['methods']['reference']['median_seconds'], 2)
        self.assertEqual(result['methods']['tiled']['min_seconds'], 2)
        self.assertEqual(result['methods']['tiled']['max_seconds'], 6)
        self.assertIsNone(result['methods']['tiled']['peak_allocated_bytes'])
        self.assertEqual(result['reference_over_tiled_speedup'], 0.5)
        self.assertFalse(aggregate_measurements(self.records()[:-1], repetitions=3)['complete'])

    def test_duplicate_unexpected_or_invalid_measurements_fail(self):
        rows = self.records()
        for wrong in (rows + [rows[0]], [{**rows[0], 'method': 'unknown'}],
                      [{**rows[0], 'seconds': float('nan')}], [{**rows[0], 'seconds': 0}]):
            with self.assertRaises(ValueError):
                aggregate_measurements(wrong, repetitions=3)

    def test_error_metrics_report_real_errors_and_nonfinite_outputs_fail(self):
        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        values = error_metrics(torch.tensor([1.125, 1.75]), expected, atol=0.01, rtol=0.01)
        self.assertFalse(values['passed'])
        self.assertEqual(values['max_absolute_error'], 0.25)
        self.assertEqual(values['mean_absolute_error'], 0.1875)
        bad = error_metrics(torch.tensor([float('nan'), 2.0]), expected, atol=1, rtol=1)
        self.assertFalse(bad['passed'])
        self.assertFalse(bad['finite'])
        self.assertIsNone(bad['max_absolute_error'])

    def test_theoretical_workspace_is_separate_from_peak_memory(self):
        values = score_workspace_bytes(batch=1, heads=2, queries=128, keys=256,
                                       block_q=16, block_k=32, element_bytes=4)
        self.assertEqual(values['dense_score_tensor_bytes'], 262144)
        self.assertEqual(values['largest_tiled_score_tensor_bytes'], 4096)
        self.assertEqual(values['kind'], 'theoretical_score_tensor_only')


if __name__ == '__main__':
    unittest.main()
