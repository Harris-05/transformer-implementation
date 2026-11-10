"""Validation baseline and common eligibility/evidence metrics.

GPU run: python baseline.py --output results/baseline_validation
Local check: python baseline.py --validate-only
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from eda import LABELS, parse_note
from prepare_model_inputs import MODEL_ID, SYSTEM_PROMPT, build_messages, sha256, validate_response


ROOT = Path(__file__).resolve().parent
INVALID = "__invalid__"
ALL_LABELS = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]


def parse_prediction(text, criterion_type, note):
    """Strict format parsing; evidence retrieval and label validity are distinct."""
    result = {"label": INVALID, "sentence_ids": [], "schema_valid": False,
              "output_valid": False, "invalid_sentence_ids": [], "error": None}
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or set(value) != {"label", "sentence_ids"}:
            raise ValueError("Expected exactly label and sentence_ids in a JSON object.")
        ids = value["sentence_ids"]
        if not isinstance(value["label"], str) or not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("Wrong label or sentence_ids types.")
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sentence IDs.")
        result["schema_valid"] = True
        result["sentence_ids"] = ids
        if value["label"] in LABELS[criterion_type]:
            result["label"] = value["label"]
        result["invalid_sentence_ids"] = sorted(set(ids) - set(parse_note(note)))
        result["output_valid"] = result["label"] != INVALID and not result["invalid_sentence_ids"]
        if not result["output_valid"]:
            result["error"] = "Invalid criterion label or nonexistent sentence citation."
    except (json.JSONDecodeError, ValueError) as error:
        result["error"] = str(error)
    return result


def prf(tp, fp, fn):
    if tp + fp + fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn)}


def evidence_metrics(pairs):
    scores, nonempty, totals = [], [], [0, 0, 0]
    empty_gold = 0
    for gold_ids, predicted_ids in pairs:
        gold, predicted = set(gold_ids), set(predicted_ids)
        counts = [len(gold & predicted), len(predicted - gold), len(gold - predicted)]
        totals = [a + b for a, b in zip(totals, counts)]
        score = prf(*counts)
        scores.append(score)
        if gold:
            nonempty.append(score)
        else:
            empty_gold += 1
    if not scores:
        raise ValueError("Cannot score an empty evaluation set.")
    def average(items):
        return {key: statistics.fmean(item[key] for item in items) for key in ("precision", "recall", "f1")}
    return {"micro": prf(*totals), "macro": average(scores),
            "nonempty_gold_macro": average(nonempty) if nonempty else None,
            "empty_gold_records": empty_gold, "nonempty_gold_records": len(nonempty),
            "tp": totals[0], "fp": totals[1], "fn": totals[2]}


def classification_metrics(gold, predicted, labels):
    # Invalid predictions stay in the evaluation denominator and confusion matrix.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    report = classification_report(gold, predicted, labels=labels, output_dict=True, zero_division=0)
    matrix_labels = labels + [INVALID]
    return {"accuracy": float(accuracy_score(gold, predicted)),
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {label: report[label] for label in labels},
            "confusion_labels": matrix_labels,
            "confusion_matrix": confusion_matrix(gold, predicted, labels=matrix_labels).tolist()}


def validate_progress(records, allowed_ids, signature):
    seen = set()
    for record in records:
        annotation_id = record["annotation_id"]
        if annotation_id not in allowed_ids or annotation_id in seen:
            raise ValueError("Checkpoint has duplicate or unexpected annotation IDs.")
        if record["run_signature"] != signature:
            raise ValueError("Checkpoint settings differ; use a new output folder.")
        seen.add(annotation_id)


def load_validation(root):
    """Load only the fixed validation group; cross-check prompts and targets."""
    config = json.loads((root / "configs/model_input_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "data/splits/split_manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "configs/tokenizer_lock.json").read_text(encoding="utf-8"))
    if config["model_id"] != MODEL_ID or config["revision"] != lock["revision"]:
        raise ValueError("Model configuration and tokenizer revision disagree.")
    if config["system_prompt_sha256"] != hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest():
        raise ValueError("Prompt differs from the recorded preparation configuration.")
    path = root / "data/model_ready/validation.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    allowed = {item["patient_id"] for item in manifest["splits"]["validation"]}
    forbidden = {item["patient_id"] for name in ("train", "test") for item in manifest["splits"][name]}
    expected = {item["patient_id"]: item["annotation_count"] for item in manifest["splits"]["validation"]}
    counts = dict.fromkeys(allowed, 0)
    seen = set()
    for record in records:
        metadata = record["metadata"]
        if metadata["patient_id"] not in allowed or metadata["patient_id"] in forbidden:
            raise ValueError("A non-validation patient reached the baseline.")
        if metadata["annotation_id"] in seen:
            raise ValueError("Duplicate annotation ID in validation input.")
        seen.add(metadata["annotation_id"])
        counts[metadata["patient_id"]] += 1
        if [item["role"] for item in record["prompt"]] != ["system", "user"]:
            raise ValueError("Unexpected prompt roles.")
        inputs = json.loads(record["prompt"][1]["content"])
        if build_messages(inputs) != record["prompt"]:
            raise ValueError("Prepared prompt does not match the frozen template.")
        if len(record["completion"]) != 1 or record["completion"][0]["role"] != "assistant":
            raise ValueError("Expected one separate expert completion.")
        validate_response(json.loads(record["completion"][0]["content"]), inputs["criterion_type"], inputs["note"])
    if counts != expected:
        raise ValueError("Validation patient counts differ from the split manifest.")
    return config, sorted(records, key=lambda record: record["metadata"]["annotation_id"]), sha256(path)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_reports(predictions, output, expected):
    """Write overall and criterion-type metrics, retaining every evaluated row."""
    groups = {}
    for name in ("overall", "inclusion", "exclusion"):
        selected = predictions if name == "overall" else [row for row in predictions if row["criterion_type"] == name]
        if not selected:
            continue
        labels = ALL_LABELS if name == "overall" else [label for label in ALL_LABELS if label in LABELS[name]]
        groups[name] = {
            "records": len(selected),
            "classification": classification_metrics([row["gold"]["label"] for row in selected],
                                                     [row["prediction"]["label"] for row in selected], labels),
            "evidence": evidence_metrics([(row["gold"]["sentence_ids"], row["prediction"]["sentence_ids"]) for row in selected]),
            "schema_valid_rate": statistics.fmean(row["prediction"]["schema_valid"] for row in selected),
            "output_valid_rate": statistics.fmean(row["prediction"]["output_valid"] for row in selected),
            "invalid_citation_records": sum(bool(row["prediction"]["invalid_sentence_ids"]) for row in selected),
        }
    total_seconds = sum(row["generation_seconds"] for row in predictions)
    metrics = {"complete": len(predictions) == expected, "expected_records": expected,
               "evaluated_records": len(predictions), "groups": groups,
               "generation": {"seconds": total_seconds,
                              "output_tokens": sum(row["output_tokens"] for row in predictions),
                              "tokens_per_second": sum(row["output_tokens"] for row in predictions) / total_seconds if total_seconds else None,
                              "max_peak_allocated_gib": max(row["peak_allocated_gib"] for row in predictions),
                              "max_peak_reserved_gib": max(row["peak_reserved_gib"] for row in predictions)}}
    save_json(output / "metrics.json", metrics)
    lines = ["# Original-model validation baseline", "",
             f"Evaluated **{len(predictions)}/{expected}** validation records. Complete: **{metrics['complete']}**.", "",
             "| Group | N | Accuracy | Macro-F1 | Evidence micro-F1 | Evidence macro-F1 | Valid output |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    for name, group in groups.items():
        classification, evidence = group["classification"], group["evidence"]
        lines.append(f"| {name} | {group['records']} | {classification['accuracy']:.4f} | {classification['macro_f1']:.4f} | "
                     f"{evidence['micro']['f1']:.4f} | {evidence['macro']['f1']:.4f} | {group['output_valid_rate']:.4f} |")
        with (output / f"confusion_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["gold/predicted", *classification["confusion_labels"]])
            writer.writerows([label, *row] for label, row in zip(classification["confusion_labels"], classification["confusion_matrix"]))
        # ConfusionMatrixDisplay expects an array rather than a Python list.
        import numpy as np
        figure, axis = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(np.array(classification["confusion_matrix"]),
                               display_labels=classification["confusion_labels"]).plot(ax=axis, xticks_rotation=45, colorbar=False)
        axis.set_title(f"Original model: {name} validation")
        figure.tight_layout()
        figure.savefig(output / f"confusion_{name}.png", dpi=160)
        plt.close(figure)
    lines += ["", "## Scoring policy", "",
              "- Macro-F1 averages the fixed applicable label set, including zero-support classes; __invalid__ predictions remain errors.",
              "- Confusion matrices include an explicit invalid-prediction column. Per-class precision/recall/F1 are in metrics.json.",
              "- Responses must be JSON with exactly label and sentence_ids; markdown and malformed responses are not repaired.",
              "- Structurally malformed responses produce an invalid label and empty evidence; no example is dropped.",
              "- With valid structure, label and evidence are evaluated separately. Nonexistent integer citations count as evidence false positives.",
              "- Both gold and predicted evidence empty: precision=recall=F1=1. With only one set empty: all three are 0.",
              "- Evidence micro scores pool sentence-set TP/FP/FN; macro scores average examples. Nonempty-gold-only macro scores are also reported.",
              "- Timing covers synchronized generation calls only, excluding download, model loading, warm-up, and file writes.",
              "- This is a 4-bit original-model baseline. Compare the fine-tuned model under the same quantization, prompt, and decoding settings.",
              "- Validation results are for development. This run does not evaluate the held-out test group.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    errors = [row for row in predictions if row["prediction"]["label"] != row["gold"]["label"]
              or set(row["prediction"]["sentence_ids"]) != set(row["gold"]["sentence_ids"])
              or not row["prediction"]["output_valid"]]
    (output / "errors.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in errors) + ("\n" if errors else ""), encoding="utf-8")
    return metrics


def run(args):
    config, records, data_hash = load_validation(args.root)
    print(f"Verified {len(records)} validation records; model: {config['model_id']} @ {config['revision']}", flush=True)
    if args.validate_only:
        print("Validation bundle is consistent. No model weights loaded.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {"model": config, "validation_sha256": data_hash,
                "source_sha256": {name: sha256(args.root / name) for name in ("baseline.py", "prepare_model_inputs.py", "eda.py")},
                "quantization": {"bits": 4, "type": "nf4", "double_quant": True, "compute_dtype": "float16"},
                "attention": "sdpa", "seed": 42, "batch_size": 1, "adapters": None}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output / "run_manifest.json"
    predictions_path = args.output / "predictions.jsonl"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_signature"] != signature:
            raise ValueError("Existing run uses different data/code/settings. Choose a new output folder.")
    else:
        if predictions_path.exists():
            raise ValueError("Predictions exist without a provenance manifest. Choose a new output folder.")
        manifest = {"run_signature": signature, "settings": settings, "runtimes": []}
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()] if predictions_path.exists() else []
    validate_progress(predictions, {row["metadata"]["annotation_id"] for row in records}, signature)
    completed = {row["annotation_id"] for row in predictions}
    chosen = records[:args.limit] if args.limit else records
    pending = [row for row in chosen if row["metadata"]["annotation_id"] not in completed]
    if args.metrics_only or not pending:
        if not predictions:
            raise ValueError("No saved predictions to score.")
        write_reports(predictions, args.output, len(records))
        print(f"Scored {len(predictions)}/{len(records)} saved predictions.")
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Colab GPU: Runtime > Change runtime type > GPU. This run does not fall back to CPU.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tokenizers", "huggingface-hub", "bitsandbytes", "accelerate", "scikit-learn")}
    if manifest["runtimes"] and manifest["runtimes"][0]["versions"] != versions:
        raise ValueError("Package versions differ from the saved run. Restore them or choose a new output folder.")
    runtime = {"id": uuid.uuid4().hex, "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "versions": versions, "gpu": torch.cuda.get_device_name(0),
               "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
               "cuda": torch.version.cuda}
    manifest["runtimes"].append(runtime)
    save_json(manifest_path, manifest)
    (args.output / "environment.txt").write_text("\n".join(sorted(
        f"{distribution.metadata['Name']}=={distribution.version}" for distribution in importlib.metadata.distributions())) + "\n", encoding="utf-8")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    print(f"GPU: {runtime['gpu']} ({runtime['gpu_memory_gib']:.1f} GiB). Loading 4-bit weights; the first download is several GB.", flush=True)
    # Official loading API: https://huggingface.co/docs/transformers/quantization/bitsandbytes
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], trust_remote_code=False)
    lock = json.loads((args.root / "configs/tokenizer_lock.json").read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download
    for name, digest in lock["file_sha256"].items():
        cached = Path(hf_hub_download(config["model_id"], name, revision=config["revision"]))
        if sha256(cached) != digest:
            raise ValueError(f"Tokenizer/configuration file differs from the recorded preparation: {name}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["revision"], quantization_config=quantization,
        device_map={"": 0}, dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=False)
    model.eval()
    generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
                                  eos_token_id=config["eos_token_id"], pad_token_id=config["pad_token_id"])

    def encode(record):
        # The model receives prompt messages only, never the expert completion.
        inputs = tokenizer.apply_chat_template(record["prompt"], tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True, truncation=False)
        tokens = inputs["input_ids"].shape[-1]
        if tokens != record["token_counts"]["prompt_tokens"] or tokens + config["max_new_tokens"] > config["max_sequence_length"]:
            raise ValueError("Tokenization changed or the input exceeds its recorded context budget.")
        return inputs.to("cuda:0")

    with torch.inference_mode():
        model.generate(**encode(pending[0]), generation_config=generation, max_new_tokens=8)
        torch.cuda.synchronize()
    print(f"Resuming: {len(predictions)} saved; {len(pending)} pending in this invocation.", flush=True)
    for record in pending:
        inputs = encode(record)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=generation, max_new_tokens=config["max_new_tokens"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        output_ids = generated[0, inputs["input_ids"].shape[-1]:].tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        clinical_inputs = json.loads(record["prompt"][1]["content"])
        prediction = parse_prediction(text, clinical_inputs["criterion_type"], clinical_inputs["note"])
        note_sentences = parse_note(clinical_inputs["note"])
        result = {"annotation_id": record["metadata"]["annotation_id"], "metadata": record["metadata"],
                  "run_signature": signature, "runtime_id": runtime["id"], **clinical_inputs,
                  "gold": json.loads(record["completion"][0]["content"]), "raw_response": text,
                  "prediction": prediction,
                  "predicted_evidence_text": {str(i): note_sentences[i] for i in prediction["sentence_ids"] if i in note_sentences},
                  "output_tokens": len(output_ids), "hit_generation_limit": len(output_ids) == config["max_new_tokens"] and (not output_ids or output_ids[-1] != config["eos_token_id"]),
                  "generation_seconds": seconds, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        with predictions_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        predictions.append(result)
        print(f"[{len(predictions)}/{len(records)}] annotation {result['annotation_id']} | "
              f"label={prediction['label']} | valid={prediction['output_valid']} | {seconds:.2f}s", flush=True)
    metrics = write_reports(predictions, args.output, len(records))
    overall = metrics["groups"]["overall"]
    print(f"Complete: {metrics['complete']} | accuracy={overall['classification']['accuracy']:.4f} | "
          f"macro-F1={overall['classification']['macro_f1']:.4f} | evidence micro-F1={overall['evidence']['micro']['f1']:.4f}", flush=True)
    print(f"Saved results: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/baseline_validation")
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N smoke run; 0 evaluates all validation rows")
    parser.add_argument("--validate-only", action="store_true", help="Validate prepared data without importing/loading the model")
    parser.add_argument("--metrics-only", action="store_true", help="Score saved predictions without a GPU")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative.")
    run(args)


if __name__ == "__main__":
    main()
