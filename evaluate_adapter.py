"""Evaluate a selected QLoRA adapter using the frozen baseline protocol."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path

from baseline import ALL_LABELS, classification_metrics, evidence_metrics, load_validation, parse_prediction, save_json, validate_progress
from eda import LABELS, parse_note
from prepare_model_inputs import load_tokenizer, sha256


ROOT = Path(__file__).resolve().parent


def metric_groups(predictions):
    groups = {}
    for name in ("overall", "inclusion", "exclusion"):
        rows = predictions if name == "overall" else [row for row in predictions if row["criterion_type"] == name]
        if not rows:
            continue
        labels = ALL_LABELS if name == "overall" else [label for label in ALL_LABELS if label in LABELS[name]]
        groups[name] = {
            "records": len(rows),
            "classification": classification_metrics([row["gold"]["label"] for row in rows],
                                                     [row["prediction"]["label"] for row in rows], labels),
            "evidence": evidence_metrics([(row["gold"]["sentence_ids"], row["prediction"]["sentence_ids"]) for row in rows]),
            "schema_valid_rate": statistics.fmean(row["prediction"]["schema_valid"] for row in rows),
            "output_valid_rate": statistics.fmean(row["prediction"]["output_valid"] for row in rows),
            "invalid_citation_records": sum(bool(row["prediction"]["invalid_sentence_ids"]) for row in rows),
        }
    return groups


def compare_groups(before, after):
    changes = {}
    for name, group in after.items():
        old = before[name]
        values = {
            "accuracy": (old["classification"]["accuracy"], group["classification"]["accuracy"]),
            "macro_f1": (old["classification"]["macro_f1"], group["classification"]["macro_f1"]),
            "evidence_micro_f1": (old["evidence"]["micro"]["f1"], group["evidence"]["micro"]["f1"]),
            "output_valid_rate": (old["output_valid_rate"], group["output_valid_rate"]),
        }
        changes[name] = {key: {"before": a, "after": b, "delta": b - a} for key, (a, b) in values.items()}
    return changes


def write_reports(predictions, output, original, expected):
    metrics = {"complete": len(predictions) == expected, "expected_records": expected,
               "evaluated_records": len(predictions), "groups": metric_groups(predictions)}
    seconds = sum(row["generation_seconds"] for row in predictions)
    tokens = sum(row["output_tokens"] for row in predictions)
    metrics["generation"] = {"seconds": seconds, "output_tokens": tokens,
                             "tokens_per_second": tokens / seconds if seconds else None,
                             "max_peak_allocated_gib": max(row["peak_allocated_gib"] for row in predictions),
                             "max_peak_reserved_gib": max(row["peak_reserved_gib"] for row in predictions)}
    comparison = {"complete": metrics["complete"], "groups": compare_groups(original["groups"], metrics["groups"])}
    save_json(output / "metrics.json", metrics)
    save_json(output / "comparison.json", comparison)
    lines = ["# Fine-tuned model: validation comparison", "",
             f"Completed: **{metrics['complete']}**; evaluated **{len(predictions)}/{expected}** records.", "",
             "| Group | Metric | Original | Fine-tuned | Change |", "|---|---|---:|---:|---:|"]
    for name, changes in comparison["groups"].items():
        for metric, values in changes.items():
            lines.append(f"| {name} | {metric} | {values['before']:.4f} | {values['after']:.4f} | {values['delta']:+.4f} |")
    lines += ["", "Values and changes are on a 0-1 scale. Negative changes are retained.",
              "The original 4-bit baseline and the fine-tuned model use the same prompt, validation examples, quantization and greedy decoding.",
              "Metrics use the original baseline's scoring functions, applicable label sets and empty-evidence policy.",
              "Invalid responses stay in all evaluation denominators. No output is repaired.",
              "Per-class results, evidence scores restricted to nonempty gold, and confusion matrices are in metrics.json.",
              "This is validation evaluation; the held-out test group remains reserved.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay
    for name, group in metrics["groups"].items():
        values = group["classification"]
        with (output / f"confusion_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["gold/predicted", *values["confusion_labels"]])
            writer.writerows([label, *row] for label, row in zip(values["confusion_labels"], values["confusion_matrix"]))
        figure, axis = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(np.array(values["confusion_matrix"]), display_labels=values["confusion_labels"]).plot(
            ax=axis, xticks_rotation=45, colorbar=False)
        axis.set_title(f"Fine-tuned model: {name} validation")
        figure.tight_layout()
        figure.savefig(output / f"confusion_{name}.png", dpi=160)
        plt.close(figure)
    errors = [row for row in predictions if row["gold"]["label"] != row["prediction"]["label"]
              or set(row["gold"]["sentence_ids"]) != set(row["prediction"]["sentence_ids"])
              or not row["prediction"]["output_valid"]]
    (output / "errors.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in errors) + ("\n" if errors else ""), encoding="utf-8")
    return metrics


def run(args):
    config, records, data_hash = load_validation(args.root)
    training = json.loads((args.training_output / "training_summary.json").read_text(encoding="utf-8"))
    training_manifest = json.loads((args.training_output / "training_manifest.json").read_text(encoding="utf-8"))
    baseline_manifest = json.loads((args.root / "results/baseline_validation/run_manifest.json").read_text(encoding="utf-8"))
    original = json.loads((args.root / "results/baseline_validation/metrics.json").read_text(encoding="utf-8"))
    if not training["complete"] or not original["complete"] or original["evaluated_records"] != len(records):
        raise ValueError("Complete training and an original baseline are required.")
    if training["signature"] != training_manifest["signature"]:
        raise ValueError("Training summary and manifest belong to different runs.")
    if config != training_manifest["identity"]["model"] or config != baseline_manifest["settings"]["model"]:
        raise ValueError("Model/prompt/generation configuration differs from the original baseline or training.")
    if baseline_manifest["settings"]["validation_sha256"] != data_hash or training_manifest["identity"]["data_sha256"]["validation"] != data_hash:
        raise ValueError("Validation examples differ from the baseline or training selection data.")
    if sha256(args.root / "baseline.py") != baseline_manifest["settings"]["source_sha256"]["baseline.py"]:
        raise ValueError("The frozen baseline scoring code has changed.")
    adapter = args.training_output / "adapter"
    for name, digest in training["adapter_sha256"].items():
        if sha256(adapter / name) != digest:
            raise ValueError(f"Selected adapter file changed: {name}")
    identity = {"model": config, "validation_sha256": data_hash, "training_signature": training["signature"],
                "adapter_sha256": training["adapter_sha256"], "evaluator_sha256": sha256(Path(__file__)),
                "scoring_sha256": sha256(args.root / "baseline.py")}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.training_output / "validation"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, predictions_path = output / "run_manifest.json", output / "predictions.jsonl"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_signature"] != signature:
            raise ValueError("Saved adapter evaluation has a different signature.")
    else:
        if predictions_path.exists():
            raise ValueError("Predictions exist without a matching evaluation manifest.")
        manifest = {"run_signature": signature, "identity": identity, "runtimes": []}
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()] if predictions_path.exists() else []
    validate_progress(predictions, {row["metadata"]["annotation_id"] for row in records}, signature)
    done = {row["annotation_id"] for row in predictions}
    pending = [row for row in records if row["metadata"]["annotation_id"] not in done]
    if pending:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Colab GPU for adapter evaluation.")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "tokenizers", "huggingface-hub", "scikit-learn")}
        if manifest["runtimes"] and versions != manifest["runtimes"][0]["versions"]:
            raise ValueError("Software versions changed during a resumed evaluation.")
        runtime = {"versions": versions, "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                   "baseline_shared_versions_match": all(value == versions.get(name, value)
                       for name, value in baseline_manifest["runtimes"][0]["versions"].items())}
        manifest["runtimes"].append(runtime)
        save_json(manifest_path, manifest)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        tokenizer, _ = load_tokenizer(args.root)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                          bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["revision"],
            quantization_config=quantization, device_map={"": 0}, dtype=torch.float16,
            attn_implementation="sdpa", trust_remote_code=False)
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        model.eval()
        generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
                                      eos_token_id=config["eos_token_id"], pad_token_id=config["pad_token_id"])

        def encode(row):
            values = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, truncation=False)
            length = values["input_ids"].shape[-1]
            if length != row["token_counts"]["prompt_tokens"] or length + config["max_new_tokens"] > config["max_sequence_length"]:
                raise ValueError("Prompt tokenization differs or exceeds the context budget.")
            return values.to("cuda:0")

        with torch.inference_mode():
            model.generate(**encode(pending[0]), generation_config=generation, max_new_tokens=8)
            torch.cuda.synchronize()
        for row in pending:
            inputs = encode(row)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                values = model.generate(**inputs, generation_config=generation, max_new_tokens=config["max_new_tokens"])
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            tokens = values[0, inputs["input_ids"].shape[-1]:].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            clinical = json.loads(row["prompt"][1]["content"])
            prediction = parse_prediction(text, clinical["criterion_type"], clinical["note"])
            sentences = parse_note(clinical["note"])
            result = {"annotation_id": row["metadata"]["annotation_id"], "metadata": row["metadata"],
                      "run_signature": signature, **clinical, "gold": json.loads(row["completion"][0]["content"]),
                      "raw_response": text, "prediction": prediction,
                      "predicted_evidence_text": {str(i): sentences[i] for i in prediction["sentence_ids"] if i in sentences},
                      "output_tokens": len(tokens), "generation_seconds": seconds,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            with predictions_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            predictions.append(result)
            print(f"[{len(predictions)}/{len(records)}] annotation {result['annotation_id']} | {prediction['label']}", flush=True)
    metrics = write_reports(predictions, output, original, len(records))
    shared_versions_match = all(runtime["baseline_shared_versions_match"] for runtime in manifest["runtimes"])
    comparison_path = output / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["baseline_shared_versions_match"] = shared_versions_match
    save_json(comparison_path, comparison)
    if not shared_versions_match:
        print("Software versions differ from the original baseline; rerun the original model under this environment for a controlled comparison.", flush=True)
    print("Adapter validation complete:", metrics["complete"], "Results:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--training-output", type=Path, default=ROOT / "results/qlora_v1")
    run(parser.parse_args())
