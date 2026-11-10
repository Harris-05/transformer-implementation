"""Safety checks for splitting, evidence parsing, and training exports."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import eda


def fixture():
    return pd.DataFrame([
        {
            "annotation_id": i, "patient_id": f"p{i:02d}",
            "note": "0. The patient is 58 years old.\n1. No prior treatment.",
            "trial_id": "trial", "trial_title": "Example",
            "criterion_type": "inclusion", "criterion_text": "Age 18 to 65",
            "expert_eligibility": "included", "expert_sentences": "[0]",
            "gpt4_explanation": "ANSWER_LEAK", "gpt4_eligibility": "ANSWER_LEAK",
            "gpt4_sentences": "ANSWER_LEAK", "explanation_correctness": "ANSWER_LEAK",
            "training": False,
        }
        for i in range(53)
    ])


class DatasetAuditTests(unittest.TestCase):
    def require(self, name):
        function = getattr(eda, name, None)
        self.assertTrue(callable(function), f"eda.{name} must be implemented")
        return function

    def test_split_is_order_independent_and_keeps_patient_annotations_together(self):
        split = self.require("make_splits")
        frame = fixture()
        frame = pd.concat([frame, frame.iloc[[0]].assign(annotation_id=100)])
        groups = split(frame)
        self.assertEqual(groups, split(frame.sample(frac=1, random_state=7)))
        self.assertEqual([len(groups[k]) for k in ("train", "validation", "test")], [37, 8, 8])
        self.assertEqual([groups[k][0] for k in ("train", "validation", "test")], ["p27", "p06", "p14"])
        all_ids = [p for group in groups.values() for p in group]
        self.assertEqual(len(set(all_ids)), 53)
        self.assertEqual(set(all_ids), set(frame.patient_id))

    def test_split_rejects_missing_patients_or_unexpected_population(self):
        split = self.require("make_splits")
        with self.assertRaises(ValueError):
            split(fixture().iloc[:-1])
        frame = fixture()
        frame.loc[0, "patient_id"] = None
        with self.assertRaises(ValueError):
            split(frame)

    def test_sentence_parser_preserves_decimal_measurements_and_original_ids(self):
        parse = self.require("parse_note")
        self.assertEqual(parse("0. Mass is 2.5 cm.\n1. Age is 58."),
                         {0: "Mass is 2.5 cm.", 1: "Age is 58."})
        for value in ("unnumbered text", "0. First\n0. Duplicate", "0. First\n2. Gap"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value)

    def test_evidence_parser_distinguishes_empty_evidence_from_bad_data(self):
        parse = self.require("parse_evidence")
        self.assertEqual(parse("[]"), [])
        self.assertEqual(parse("[0, 2]"), [0, 2])
        for value in (None, "", "[True]", "[-1]", "[0, 0]", "['0']", "__import__('os')"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value)

    def test_audit_flags_bad_citations_but_allows_empty_evidence(self):
        audit = self.require("audit_split")
        frame = fixture().iloc[:3].copy()
        frame.loc[0, "expert_sentences"] = "[]"
        frame.loc[1, "expert_sentences"] = "[99]"
        frame.loc[2, "expert_eligibility"] = "excluded"
        report = audit(frame)
        self.assertEqual(report["empty_evidence_rows"], 1)
        self.assertEqual({e["annotation_id"] for e in report["errors"]}, {1, 2})

    def test_exports_exclude_previous_answers_and_all_test_patients(self):
        export = self.require("export_records")
        groups = self.require("make_splits")(fixture())
        frame = fixture()
        # Held-out data must never reach the development export or parser.
        frame.loc[frame.patient_id.isin(groups["test"]), "note"] = "TEST_SECRET"
        with tempfile.TemporaryDirectory() as temp:
            counts = export(frame, groups, Path(temp))
            self.assertEqual(counts, {"train": 37, "validation": 8})
            self.assertFalse((Path(temp) / "test.jsonl").exists())
            content = (Path(temp) / "train.jsonl").read_text(encoding="utf-8")
            content += (Path(temp) / "validation.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("ANSWER_LEAK", content)
            self.assertNotIn("TEST_SECRET", content)
            for line in content.splitlines():
                record = json.loads(line)
                self.assertEqual(set(record["input"]), {"note", "criterion_type", "criterion_text"})
                self.assertEqual(set(record["target"]), {"label", "sentence_ids"})
                self.assertNotIn(record["metadata"]["patient_id"], groups["test"])

    def test_training_record_without_criterion_is_explicitly_quarantined(self):
        frame = fixture()
        groups = self.require("make_splits")(frame)
        patient = groups["train"][0]
        frame.loc[frame.patient_id.eq(patient), "criterion_text"] = None
        with tempfile.TemporaryDirectory() as temp:
            counts = self.require("export_records")(frame, groups, Path(temp))
            self.assertEqual(counts, {"train": 36, "validation": 8})
            quarantine = json.loads((Path(temp) / "quarantined_train.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(quarantine["record"]["patient_id"], patient)
            self.assertIsNone(quarantine["record"]["criterion_text"])
            self.assertTrue(any("criterion_text" in reason for reason in quarantine["issues"]))
            self.assertEqual(len(groups["train"]), 37)

    def test_invalid_validation_examples_are_not_silently_removed(self):
        frame = fixture()
        groups = self.require("make_splits")(frame)
        frame.loc[frame.patient_id.eq(groups["validation"][0]), "criterion_text"] = None
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                self.require("export_records")(frame, groups, Path(temp))
            self.assertFalse((Path(temp) / "validation.jsonl").exists())

    def test_all_invalid_notes_still_produce_json_serializable_audit(self):
        frame = fixture().iloc[:1].copy()
        frame["note"] = None
        report = self.require("audit_split")(frame)
        self.assertTrue(report["errors"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
