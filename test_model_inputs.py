"""Checks for prompt leakage, response validity, and context budgeting."""

import copy
import json
import unittest

try:
    import prepare_model_inputs as prep
except ModuleNotFoundError:
    prep = None


def example():
    return {
        "input": {"note": "0. Age is 58.\n1. No diabetes.",
                  "criterion_type": "inclusion", "criterion_text": "Age 18 to 65"},
        "target": {"label": "included", "sentence_ids": [0]},
        "metadata": {"annotation_id": 1, "patient_id": "development", "trial_id": "trial", "trial_title": "Example"},
    }


class ModelInputTests(unittest.TestCase):
    def require(self, name):
        function = getattr(prep, name, None)
        self.assertTrue(callable(function), f"prepare_model_inputs.{name} must be implemented")
        return function

    def test_prompt_is_independent_of_targets_and_metadata(self):
        build = self.require("build_messages")
        record = example()
        messages = build(record["input"])
        record["target"] = {"label": "ANSWER_LEAK", "sentence_ids": [999]}
        record["metadata"]["trial_title"] = "ANSWER_LEAK"
        self.assertEqual(messages, build(record["input"]))
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(json.loads(messages[-1]["content"]), example()["input"])
        self.assertNotIn("ANSWER_LEAK", json.dumps(messages))

    def test_unexpected_input_fields_and_blank_criteria_are_rejected(self):
        build = self.require("build_messages")
        inputs = example()["input"]
        with self.assertRaises(ValueError):
            build({**inputs, "expert_eligibility": "included"})
        with self.assertRaises(ValueError):
            build({**inputs, "criterion_text": " "})

    def test_response_validates_all_applicable_labels_and_empty_evidence(self):
        validate = self.require("validate_response")
        for kind, labels in {
            "inclusion": ["included", "not included", "not enough information", "not applicable"],
            "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
        }.items():
            for label in labels:
                with self.subTest(kind=kind, label=label):
                    value = {"label": label, "sentence_ids": []}
                    self.assertEqual(validate(value, kind, "0. Age is 58."), value)

    def test_response_rejects_wrong_type_labels_and_invalid_citations(self):
        validate = self.require("validate_response")
        invalid = [
            {"label": "excluded", "sentence_ids": [0]},
            {"label": "included", "sentence_ids": [9]},
            {"label": "included", "sentence_ids": [True]},
            {"label": "included", "sentence_ids": [0, 0]},
            {"label": "included", "sentence_ids": "[0]"},
            {"label": "included", "sentence_ids": [0], "explanation": "extra"},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(value, "inclusion", "0. Age is 58.")

    def test_context_budget_reserves_generation_space(self):
        recommend = self.require("recommend_limits")
        rows = [{"prompt_tokens": 520, "completion_tokens": 31, "total_tokens": 551},
                {"prompt_tokens": 300, "completion_tokens": 42, "total_tokens": 342}]
        result = recommend(rows)
        self.assertEqual(result["max_sequence_length"], 1024)
        self.assertEqual(result["max_new_tokens"], 128)
        self.assertEqual(result["truncated_records"], 0)
        with self.assertRaises(ValueError):
            recommend(rows, requested_length=512)

    def test_long_completions_are_not_given_an_inadequate_generation_budget(self):
        result = self.require("recommend_limits")([
            {"prompt_tokens": 100, "completion_tokens": 180, "total_tokens": 280}])
        self.assertGreaterEqual(result["max_new_tokens"], 180)
        self.assertGreaterEqual(result["max_sequence_length"], 100 + result["max_new_tokens"])

    def test_development_validation_rejects_held_out_patients_and_duplicate_ids(self):
        validate = self.require("validate_records")
        validate([example()], {"development"})
        with self.assertRaises(ValueError):
            validate([example()], {"another-patient"})
        with self.assertRaises(ValueError):
            validate([example(), copy.deepcopy(example())], {"development"})


if __name__ == "__main__":
    unittest.main()
