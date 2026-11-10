"""Hand-checked evaluation cases; no model download or GPU required."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import baseline
except ModuleNotFoundError:
    baseline = None


class BaselineTests(unittest.TestCase):
    def require(self, name):
        function = getattr(baseline, name, None)
        self.assertTrue(callable(function), f"baseline.{name} must be implemented")
        return function

    def test_strict_json_does_not_repair_markdown_or_missing_fields(self):
        parse = self.require("parse_prediction")
        for text in ('```json\n{"label":"included","sentence_ids":[0]}\n```',
                     '{"label":"included"}', '[1]', 'not JSON',
                     '{"label":"included","sentence_ids":[true]}'):
            with self.subTest(text=text):
                result = parse(text, "inclusion", "0. Age 58.")
                self.assertFalse(result["output_valid"])
                self.assertEqual(result["label"], "__invalid__")

    def test_invalid_sentence_ids_count_as_retrieval_false_positives(self):
        result = self.require("parse_prediction")(
            '{"label":"included","sentence_ids":[0,99]}', "inclusion", "0. Age 58.")
        self.assertEqual(result["label"], "included")
        self.assertEqual(result["sentence_ids"], [0, 99])
        self.assertEqual(result["invalid_sentence_ids"], [99])
        self.assertFalse(result["output_valid"])
        scores = self.require("evidence_metrics")([([0], [0, 99])])
        self.assertEqual(scores["micro"]["precision"], 0.5)
        self.assertAlmostEqual(scores["micro"]["f1"], 2 / 3)

    def test_wrong_criterion_label_is_invalid_but_evidence_is_scored_separately(self):
        result = self.require("parse_prediction")(
            '{"label":"excluded","sentence_ids":[0]}', "inclusion", "0. Age 58.")
        self.assertEqual(result["label"], "__invalid__")
        self.assertEqual(result["sentence_ids"], [0])
        self.assertFalse(result["output_valid"])

    def test_all_empty_evidence_is_a_perfect_match(self):
        result = self.require("evidence_metrics")([([], []), ([], [])])
        self.assertEqual(result["micro"], {"precision": 1.0, "recall": 1.0, "f1": 1.0})
        self.assertEqual(result["macro"]["f1"], 1.0)
        self.assertIsNone(result["nonempty_gold_macro"])

    def test_evidence_micro_and_macro_include_empty_and_missing_cases(self):
        result = self.require("evidence_metrics")([([0, 1], [1, 2]), ([], []), ([3], [])])
        self.assertAlmostEqual(result["micro"]["precision"], 1 / 2)
        self.assertAlmostEqual(result["micro"]["recall"], 1 / 3)
        self.assertAlmostEqual(result["micro"]["f1"], 2 / 5)
        self.assertAlmostEqual(result["macro"]["f1"], 1 / 2)
        self.assertAlmostEqual(result["nonempty_gold_macro"]["f1"], 1 / 4)
        self.assertEqual(result["empty_gold_records"], 1)

    def test_invalid_label_predictions_remain_in_accuracy_and_confusion_matrix(self):
        result = self.require("classification_metrics")(
            ["included", "included", "not included"],
            ["included", "__invalid__", "not included"],
            ["included", "not included", "not enough information", "not applicable"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 5 / 12)
        self.assertEqual(result["confusion_matrix"][0][-1], 1)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 3)

    def test_resume_rejects_changed_settings_duplicate_or_foreign_annotations(self):
        validate = self.require("validate_progress")
        row = {"annotation_id": 10, "run_signature": "same"}
        validate([row], {10, 20}, "same")
        for rows, allowed, signature in [([row], {10}, "different"),
                                         ([row, row], {10}, "same"),
                                         ([row], {20}, "same")]:
            with self.subTest(rows=rows, allowed=allowed), self.assertRaises(ValueError):
                validate(rows, allowed, signature)

    def test_report_marks_partial_runs_and_writes_confusion_artifacts(self):
        row = {"criterion_type": "inclusion", "gold": {"label": "included", "sentence_ids": [0]},
               "prediction": {"label": "included", "sentence_ids": [0], "schema_valid": True,
                              "output_valid": True, "invalid_sentence_ids": []},
               "generation_seconds": 2.0, "output_tokens": 10,
               "peak_allocated_gib": 3.0, "peak_reserved_gib": 4.0}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            metrics = self.require("write_reports")([row], output, expected=206)
            self.assertFalse(metrics["complete"])
            self.assertEqual(metrics["evaluated_records"], 1)
            self.assertEqual(metrics["generation"]["tokens_per_second"], 5.0)
            self.assertTrue((output / "confusion_overall.png").is_file())
            self.assertTrue((output / "confusion_inclusion.csv").is_file())
            self.assertEqual((output / "errors.jsonl").read_text(), "")
            self.assertEqual(json.loads((output / "metrics.json").read_text())["expected_records"], 206)


if __name__ == "__main__":
    unittest.main()
