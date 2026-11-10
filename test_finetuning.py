"""Checks for assistant-only supervision and unambiguous training padding."""

import unittest

try:
    import finetune
except ModuleNotFoundError:
    finetune = None


class FinetuningTests(unittest.TestCase):
    def require(self, name):
        function = getattr(finetune, name, None)
        self.assertTrue(callable(function), f"finetune.{name} must be implemented")
        return function

    def test_only_completion_tokens_receive_loss(self):
        mask = self.require("mask_completion")
        value = mask([10, 11, 12], [10, 11, 12, 20, 21], 8)
        self.assertEqual(value["input_ids"], [10, 11, 12, 20, 21])
        self.assertEqual(value["labels"], [-100, -100, -100, 20, 21])
        self.assertEqual(value["attention_mask"], [1, 1, 1, 1, 1])

    def test_bad_prefix_empty_target_and_overlength_fail_without_truncation(self):
        mask = self.require("mask_completion")
        for prompt, full, maximum in [([1, 2], [1, 3, 4], 8), ([1, 2], [1, 2], 8),
                                      ([1], [1, 2, 3], 2), ([], [1, 2], 8)]:
            with self.subTest(prompt=prompt, full=full), self.assertRaises(ValueError):
                mask(prompt, full, maximum)

    def test_padding_is_masked_and_never_changes_completion_targets(self):
        mask = self.require("mask_completion")
        pad = self.require("pad_examples")
        features = [mask([1, 2], [1, 2, 3, 9], 8), mask([4], [4, 5], 8)]
        batch = pad(features, pad_token_id=9)
        self.assertEqual(batch["input_ids"], [[1, 2, 3, 9], [4, 5, 9, 9]])
        self.assertEqual(batch["attention_mask"], [[1, 1, 1, 1], [1, 1, 0, 0]])
        self.assertEqual(batch["labels"], [[-100, -100, 3, 9], [-100, 5, -100, -100]])

    def test_training_rejects_validation_patients_and_wrong_category_labels(self):
        check = self.require("validate_training_record")
        import json
        from prepare_model_inputs import build_messages
        inputs = {"note": "0. Age 58.", "criterion_type": "inclusion", "criterion_text": "Age 18 to 65"}
        row = {"metadata": {"annotation_id": 1, "patient_id": "train"},
               "prompt": build_messages(inputs),
               "completion": [{"role": "assistant", "content": json.dumps({"label": "included", "sentence_ids": [0]})}]}
        check(row, {"train"})
        with self.assertRaises(ValueError):
            check(row, {"different-patient"})
        row["completion"][0]["content"] = '{"label":"excluded","sentence_ids":[0]}'
        with self.assertRaises(ValueError):
            check(row, {"train"})

    def test_adapter_evaluator_reproduces_the_original_baseline_scores(self):
        import json
        from pathlib import Path
        from evaluate_adapter import metric_groups
        root = Path(__file__).resolve().parent / "results/baseline_validation"
        rows = [json.loads(line) for line in (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        original = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metric_groups(rows), original["groups"])


if __name__ == "__main__":
    unittest.main()
