"""Audit TrialGPT development data and prepare leakage-safe training records.

Run: python eda.py
Only patient IDs and annotation counts are used from the held-out test group.
Original files and labels are never modified. Invalid training records are
explicitly quarantined; validation records are never silently removed.
"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow


ROOT = Path(__file__).resolve().parent
INPUT_COLUMNS = ["note", "criterion_type", "criterion_text"]
TARGET_COLUMNS = ["expert_eligibility", "expert_sentences"]
METADATA_COLUMNS = ["annotation_id", "patient_id", "trial_id", "trial_title"]
LABELS = {
    "inclusion": {"included", "not included", "not enough information", "not applicable"},
    "exclusion": {"excluded", "not excluded", "not enough information", "not applicable"},
}


def is_blank(value):
    return value is None or (isinstance(value, str) and not value.strip()) or bool(pd.isna(value))


def make_splits(data):
    """Apply the assignment's sort -> default_rng(42) shuffle -> 37/8/8 rule."""
    if data["patient_id"].map(is_blank).any():
        raise ValueError("Patient IDs cannot be missing or blank.")
    if not data["patient_id"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Patient IDs must be strings; do not silently coerce IDs.")
    patients = np.array(sorted(data["patient_id"].unique()))
    if len(patients) != 53:
        raise ValueError(f"Expected 53 patients; found {len(patients)}. Check the source file.")
    np.random.default_rng(42).shuffle(patients)
    return {"train": patients[:37].tolist(), "validation": patients[37:45].tolist(),
            "test": patients[45:].tolist()}


def parse_note(note):
    """Read the dataset's numbered lines without re-segmenting medical text."""
    if not isinstance(note, str) or not note.strip():
        raise ValueError("Missing patient note.")
    matches = list(re.finditer(r"(?m)^[ \t]*(\d+)\.\s+", note))
    if not matches or note[:matches[0].start()].strip():
        raise ValueError("Expected a patient note numbered from sentence 0.")
    ids = [int(match.group(1)) for match in matches]
    if ids != list(range(len(ids))):
        raise ValueError("Sentence IDs must be unique and consecutive, starting at 0.")
    sentences = {
        sentence_id: note[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(note)].strip()
        for i, (sentence_id, match) in enumerate(zip(ids, matches))
    }
    if not all(sentences.values()):
        raise ValueError("A numbered patient sentence is empty.")
    return sentences


def parse_evidence(value):
    """Parse IDs safely; [] is valid, a missing or malformed value is not."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Evidence must be a string containing a list; [] means no evidence.")
    try:
        ids = ast.literal_eval(value)
    except (ValueError, SyntaxError) as error:
        raise ValueError("Evidence is not a valid literal list.") from error
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Evidence must contain only nonnegative integer sentence IDs.")
    if len(ids) != len(set(ids)):
        raise ValueError("Evidence has duplicate sentence IDs.")
    return ids


def length_summary(values):
    series = pd.Series(values, dtype="float64")
    if series.empty:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {"min": float(series.min()), "median": float(series.median()),
            "p95": float(series.quantile(0.95)), "max": float(series.max())}


def audit_split(frame):
    """Inspect training/validation only; preserve every anomaly for review."""
    columns = INPUT_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS
    missing = {column: int(frame[column].map(is_blank).sum()) for column in columns}
    errors, evidence_counts, sentence_counts, examples = [], [], [], []
    example_categories = set()
    empty_evidence = 0
    for row in frame.to_dict("records"):
        row_errors = []
        for column in columns:
            if is_blank(row[column]):
                row_errors.append(f"Missing or blank {column}.")
        if row["criterion_type"] not in LABELS:
            row_errors.append("Unknown criterion type.")
        elif row["expert_eligibility"] not in LABELS[row["criterion_type"]]:
            row_errors.append("Eligibility label is invalid for this criterion type.")
        try:
            sentences = parse_note(row["note"])
            ids = parse_evidence(row["expert_sentences"])
            invalid = sorted(set(ids) - set(sentences))
            if invalid:
                raise ValueError(f"Evidence references nonexistent sentence IDs: {invalid}.")
            empty_evidence += not ids
            evidence_counts.append(len(ids))
            sentence_counts.append(len(sentences))
            category = (row["criterion_type"], row["expert_eligibility"])
            if category not in example_categories:
                example_categories.add(category)
                examples.append({
                    "annotation_id": row["annotation_id"], "patient_id": row["patient_id"],
                    "criterion_type": row["criterion_type"], "criterion_text": row["criterion_text"],
                    "label": row["expert_eligibility"], "sentence_ids": ids,
                    "evidence_text": [sentences[i] for i in ids],
                })
        except ValueError as error:
            row_errors.append(str(error))
        errors.extend({"annotation_id": row["annotation_id"], "issue": issue} for issue in row_errors)

    key = ["patient_id", "trial_id", "criterion_type", "criterion_text"]
    duplicates = frame.loc[frame.duplicated(key, keep=False)]
    conflicts = []
    for _, group in duplicates.groupby(key, dropna=False):
        if group["expert_eligibility"].nunique(dropna=False) > 1 or group["expert_sentences"].nunique(dropna=False) > 1:
            conflicts.append(group["annotation_id"].tolist())
    inconsistent = frame.groupby("patient_id")["note"].nunique(dropna=False)
    inconsistent = inconsistent[inconsistent > 1].index.tolist()
    counts = frame.groupby(["criterion_type", "expert_eligibility"]).size()
    return {
        "rows": len(frame), "patients": int(frame.patient_id.nunique()),
        "trials": int(frame.trial_id.nunique()), "missing_or_blank": missing,
        "label_counts": [{"criterion_type": kind, "label": label, "count": int(count)}
                         for (kind, label), count in counts.items()],
        "empty_evidence_rows": int(empty_evidence),
        "duplicate_criterion_rows": len(duplicates),
        "repeated_criterion_annotation_ids": duplicates["annotation_id"].tolist(),
        "conflicting_annotation_groups": conflicts, "inconsistent_patient_notes": inconsistent,
        "lengths": {
            "note_characters_per_annotation": length_summary(frame.note.fillna("").str.len()),
            "note_words_per_unique_patient_note": length_summary(frame.note.dropna().drop_duplicates().str.split().str.len()),
            "criterion_characters": length_summary(frame.criterion_text.fillna("").str.len()),
            "patient_sentence_count_per_valid_annotation": length_summary(sentence_counts),
            "evidence_sentence_count_per_valid_annotation": length_summary(evidence_counts),
        },
        "examples": examples, "errors": errors,
    }


def export_records(data, splits, directory):
    """Allowlist inputs/targets/metadata; never export held-out examples here."""
    frames = {name: data.loc[data.patient_id.isin(splits[name])]
              for name in ("train", "validation")}
    audits = {name: audit_split(frame) for name, frame in frames.items()}
    if audits["validation"]["errors"]:
        raise ValueError("Validation has invalid records. Review them; do not silently filter evaluation cases.")
    training_issues = {}
    for error in audits["train"]["errors"]:
        training_issues.setdefault(error["annotation_id"], []).append(error["issue"])
    quarantine = []
    for row in frames["train"].loc[frames["train"].annotation_id.isin(training_issues)].to_dict("records"):
        quarantine.append(json.dumps({
            "record": {key: None if is_blank(row[key]) else row[key]
                       for key in INPUT_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS},
            "issues": training_issues[row["annotation_id"]],
        }, ensure_ascii=False))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "quarantined_train.jsonl").write_text(
        "\n".join(quarantine) + ("\n" if quarantine else ""), encoding="utf-8")
    counts = {}
    for split in ("train", "validation"):
        frame = frames[split]
        if split == "train":
            frame = frame.loc[~frame.annotation_id.isin(training_issues)]
        lines = []
        for row in frame.sort_values("annotation_id").to_dict("records"):
            record = {
                "input": {key: row[key] for key in INPUT_COLUMNS},
                "target": {"label": row["expert_eligibility"],
                           "sentence_ids": parse_evidence(row["expert_sentences"])},
                "metadata": {key: row[key] for key in METADATA_COLUMNS},
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        (directory / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[split] = len(lines)
    return counts


def markdown_report(report):
    lines = ["# TrialGPT dataset audit", "", f"Status: {report['status']}", "",
             f"Source SHA-256: `{report['source_sha256']}`", "",
             f"Rows: {report['rows']}; patients: {report['patients']}; columns: {len(report['schema'])}.", "",
             "## Fixed patient split", "", "| Split | Patients | Annotations |",
             "|---|---:|---:|"]
    for name, group in report["split_summary"].items():
        lines.append(f"| {name} | {group['patients']} | {group['annotations']} |")
    lines += ["", "Test notes, labels, and evidence were not explored or exported.", "",
              "## Development data", ""]
    for name, audit in report["development"].items():
        lines += [f"### {name}", "", "| Criterion type | Expert label | Count |", "|---|---|---:|"]
        for item in audit["label_counts"]:
            lines.append(f"| {item['criterion_type']} | {item['label']} | {item['count']} |")
        lines += ["", f"Missing/blank cells: {sum(audit['missing_or_blank'].values())}.",
                  f"Empty evidence lists: {audit['empty_evidence_rows']} (retained as valid).",
                  f"Rows with repeated patient/trial/criterion keys: {audit['duplicate_criterion_rows']}.",
                  f"Conflicting duplicate groups: {len(audit['conflicting_annotation_groups'])}.",
                  f"Patients with inconsistent notes: {len(audit['inconsistent_patient_notes'])}.",
                  f"Validation errors: {len(audit['errors'])}.", ""]
        lines += ["| Length measure | Min | Median | P95 | Max |", "|---|---:|---:|---:|---:|"]
        for measure, stats in audit["lengths"].items():
            values = ["n/a" if value is None else f"{value:.1f}" for value in stats.values()]
            lines.append(f"| {measure} | " + " | ".join(values) + " |")
        lines += [""]
        for error in audit["errors"]:
            lines.append(f"- Annotation {error['annotation_id']}: {error['issue']}")
    lines += ["", "## Interpretation and next steps", "",
              "- Preserve the original labels, sentence numbering, and empty evidence lists.",
              "- Rare labels can make accuracy misleading; use macro-F1 and per-class metrics.",
              "- Repeated patient notes across different criteria are expected; the split groups patients.",
              "- Repeated trial criteria across different patients are expected in this benchmark.",
              "- Review duplicate conflicts manually; this script never edits the source or relabels records.",
              f"- Training quarantine: {report['quarantined_training_ids']}. These IDs stay in the split manifest but are excluded from training exports with recorded reasons.",
              f"- Exported development rows: {report['exports']}. Validation is never silently filtered.",
              "- GPT-4 outputs, explanation correctness, and the existing training flag are excluded from exports.",
              "- Lengths here are characters/words, not tokens. Select a tokenizer before choosing context limits.",
              "- Next: select the model, build a prompt, measure token lengths/evidence truncation, and run a validation baseline.",
              "- Full machine-readable details and evidence examples are in data_audit.json.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "train-00000-of-00001.parquet")
    parser.add_argument("--output", type=Path, default=ROOT, help="Base directory for reports and prepared data")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}")
    data = pd.read_parquet(args.data)
    required = INPUT_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS
    missing = sorted(set(required) - set(data.columns))
    if missing:
        parser.error(f"Missing required columns: {missing}")
    if data.annotation_id.map(is_blank).any() or data.annotation_id.duplicated().any():
        parser.error("Annotation IDs must be present and unique.")
    try:
        splits = make_splits(data)
    except ValueError as error:
        parser.error(str(error))
    manifest = {
        "source_file": args.data.name,
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "seed": 42, "algorithm": "Sort unique patient IDs, shuffle with NumPy default_rng(42), slice 37/8/8.",
        "numpy_version": np.__version__,
        "splits": {
            name: [{"patient_id": patient, "annotation_count": int(data.patient_id.eq(patient).sum())}
                   for patient in patients]
            for name, patients in splits.items()
        },
    }
    split_directory = args.output / "data" / "splits"
    split_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = split_directory / "split_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        parser.error("Existing split manifest differs. Use a separate --output directory to investigate.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    development = {name: audit_split(data.loc[data.patient_id.isin(splits[name])])
                   for name in ("train", "validation")}
    errors = len(development["validation"]["errors"])
    quarantined = sorted({error["annotation_id"] for error in development["train"]["errors"]})
    warnings = any(audit["duplicate_criterion_rows"] or audit["inconsistent_patient_notes"]
                   for audit in development.values()) or len(data) != 1015 or bool(quarantined)
    report = {
        "status": "FAILED" if errors else "REVIEW WARNINGS" if warnings else "PASSED",
        "source_sha256": manifest["source_sha256"], "rows": len(data),
        "expected_rows": 1015, "patients": int(data.patient_id.nunique()),
        "schema": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__},
        "split_summary": {name: {"patients": len(patients), "annotations": int(data.patient_id.isin(patients).sum())}
                          for name, patients in splits.items()},
        "development": development, "test_content_inspected": False,
        "quarantined_training_ids": quarantined,
        "exports": export_records(data, splits, args.output / "data" / "processed") if not errors else {},
    }
    reports_directory = args.output / "reports"
    reports_directory.mkdir(parents=True, exist_ok=True)
    (reports_directory / "data_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (reports_directory / "data_audit.md").write_text(markdown_report(report), encoding="utf-8")
    print(f"Audit: {report['status']} | {len(data)} annotations | {report['patients']} patients")
    for name, group in report["split_summary"].items():
        print(f"  {name}: {group['patients']} patients, {group['annotations']} annotations")
    for name, audit in development.items():
        print(f"  {name}: {len(audit['errors'])} errors, {audit['empty_evidence_rows']} empty evidence lists, "
              f"{audit['duplicate_criterion_rows']} duplicate-key rows")
    print(f"Prepared rows: {report['exports']}; quarantined training IDs: {quarantined}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {reports_directory / 'data_audit.md'}")
    print("Test content remains held out. Token length checks await model/tokenizer selection.")
    if errors:
        raise SystemExit("Dataset validation failed. Review the report before preparing training inputs.")


if __name__ == "__main__":
    main()
