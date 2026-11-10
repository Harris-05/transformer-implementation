"""QLoRA training with completion-only loss and fixed patient groups.

Local data check: python finetune.py --prepare-only --offline
Colab GPU training: python finetune.py --output /path/to/persistent/run
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from baseline import load_validation, save_json
from prepare_model_inputs import build_messages, load_tokenizer, sha256, validate_response


ROOT = Path(__file__).resolve().parent


def mask_completion(prompt_ids, full_ids, maximum):
    if not prompt_ids or len(full_ids) <= len(prompt_ids) or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Training tokens must contain the exact prompt followed by a nonempty completion.")
    if len(full_ids) > maximum:
        raise ValueError("Training example exceeds the context limit; no truncation is permitted.")
    return {"input_ids": list(full_ids), "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])}


def pad_examples(features, pad_token_id):
    maximum = max(len(item["input_ids"]) for item in features)
    batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for feature in features:
        padding = maximum - len(feature["input_ids"])
        for key, fill in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)):
            batch[key].append(feature[key] + [fill] * padding)
    return batch


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in pad_examples(features, self.pad_token_id).items()}


def validate_training_record(record, allowed_patients):
    if record["metadata"]["patient_id"] not in allowed_patients:
        raise ValueError("A patient outside the training split reached supervised training.")
    if [item["role"] for item in record["prompt"]] != ["system", "user"]:
        raise ValueError("Training prompt has unexpected roles.")
    inputs = json.loads(record["prompt"][1]["content"])
    if record["prompt"] != build_messages(inputs):
        raise ValueError("The frozen prompt was changed.")
    if len(record["completion"]) != 1 or record["completion"][0]["role"] != "assistant":
        raise ValueError("Expected a separate expert assistant completion.")
    validate_response(json.loads(record["completion"][0]["content"]), inputs["criterion_type"], inputs["note"])


def load_training_data(root):
    config, validation, _ = load_validation(root)
    manifest = json.loads((root / "data/splits/split_manifest.json").read_text(encoding="utf-8"))
    allowed = {item["patient_id"] for item in manifest["splits"]["train"]}
    forbidden = {item["patient_id"] for name in ("validation", "test") for item in manifest["splits"][name]}
    if allowed & forbidden:
        raise ValueError("Patient split overlap.")
    quarantined = [json.loads(line)["record"] for line in
                   (root / "data/processed/quarantined_train.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {item["patient_id"]: item["annotation_count"] for item in manifest["splits"]["train"]}
    excluded_ids = set()
    for row in quarantined:
        if row["patient_id"] not in allowed or row["annotation_id"] in excluded_ids:
            raise ValueError("Invalid training quarantine manifest.")
        expected[row["patient_id"]] -= 1
        excluded_ids.add(row["annotation_id"])
    train = [json.loads(line) for line in (root / "data/model_ready/train.jsonl").read_text(encoding="utf-8").splitlines()]
    seen, counts = set(), dict.fromkeys(allowed, 0)
    validation_ids = {row["metadata"]["annotation_id"] for row in validation}
    for row in train:
        validate_training_record(row, allowed)
        annotation_id = row["metadata"]["annotation_id"]
        if annotation_id in seen | validation_ids | excluded_ids:
            raise ValueError("Duplicate, validation, or quarantined annotation in training.")
        seen.add(annotation_id)
        counts[row["metadata"]["patient_id"]] += 1
    if counts != expected:
        raise ValueError("Training patient counts differ from the split manifest minus the documented quarantine.")
    return config, {"train": train, "validation": validation}


def encode_record(record, tokenizer, maximum):
    prompt = tokenizer.apply_chat_template(record["prompt"], tokenize=True, add_generation_prompt=True,
                                           return_dict=False, truncation=False)
    complete = tokenizer.apply_chat_template(record["prompt"] + record["completion"], tokenize=True,
                                             add_generation_prompt=False, return_dict=False, truncation=False)
    if len(prompt) != record["token_counts"]["prompt_tokens"] or len(complete) != record["token_counts"]["total_tokens"]:
        raise ValueError("Actual token lengths differ from the frozen preparation.")
    return mask_completion(prompt, complete, maximum)


def training_arguments(output, settings, train_count, cpu=False):
    """Use the installed Transformers API; warmup_steps replaces older warmup_ratio."""
    from transformers import TrainingArguments
    updates_per_epoch = math.ceil(train_count / (settings["per_device_train_batch_size"] * settings["gradient_accumulation_steps"]))
    total_updates = math.ceil(updates_per_epoch * settings["num_train_epochs"])
    return TrainingArguments(
        output_dir=str(output), num_train_epochs=settings["num_train_epochs"],
        learning_rate=settings["learning_rate"],
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        per_device_eval_batch_size=settings["per_device_eval_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        warmup_steps=math.ceil(total_updates * settings["warmup_ratio"]),
        weight_decay=settings["weight_decay"], max_grad_norm=settings["max_grad_norm"],
        eval_strategy="epoch", save_strategy="epoch", logging_steps=10, logging_first_step=True,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, prediction_loss_only=True, eval_accumulation_steps=1,
        gradient_checkpointing=settings["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}, fp16=not cpu, bf16=False,
        optim="adamw_torch", report_to="none", remove_unused_columns=False, label_names=["labels"],
        seed=settings["seed"], data_seed=settings["seed"], dataloader_num_workers=0,
        dataloader_pin_memory=False, use_cpu=cpu, push_to_hub=False)


def train(args, config, development, tokenizer, encoded):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Colab GPU before training.")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback, Trainer, TrainerCallback, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    settings = json.loads((args.root / "configs/qlora_config.json").read_text(encoding="utf-8"))
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "tokenizers")}
    identity = {"model": config, "qlora": settings, "versions": versions,
                "data_sha256": {name: sha256(args.root / "data/model_ready" / f"{name}.jsonl") for name in development},
                "source_sha256": {name: sha256(args.root / name) for name in ("finetune.py", "prepare_model_inputs.py", "eda.py")}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "training_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError("Training data/code/settings/versions changed. Use a new output folder.")
    else:
        if any(args.output.iterdir()):
            raise ValueError("Training folder contains files but no matching manifest. Use an empty folder.")
        save_json(manifest_path, {"signature": signature, "identity": identity,
                                 "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                                 "train_records": len(development["train"]), "validation_records": len(development["validation"])})
    if (args.output / "training_summary.json").exists():
        print("Training already completed. The selected adapter is in", args.output / "adapter", flush=True)
        return
    set_seed(settings["seed"])
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["revision"],
        quantization_config=quantization, device_map={"": 0}, dtype=torch.float16,
        attn_implementation="sdpa", trust_remote_code=False)
    model.config.use_cache = False
    # Official PEFT recipe: https://huggingface.co/docs/peft/developer_guides/quantization
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=settings["lora_r"], lora_alpha=settings["lora_alpha"],
                      lora_dropout=settings["lora_dropout"], target_modules=settings["target_modules"],
                      bias="none", task_type="CAUSAL_LM", revision=config["revision"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class PersistLogs(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            if logs:
                with (args.output / "training_log.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())

    checkpoint_dir = args.output / "checkpoints"
    trainer = Trainer(model=model, args=training_arguments(checkpoint_dir, settings, len(encoded["train"])),
                      train_dataset=encoded["train"], eval_dataset=encoded["validation"],
                      data_collator=CompletionCollator(tokenizer.pad_token_id), processing_class=tokenizer,
                      callbacks=[PersistLogs(), EarlyStoppingCallback(early_stopping_patience=settings["early_stopping_patience"])])
    last_checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    print("Resume checkpoint:", last_checkpoint or "new training run", flush=True)
    torch.cuda.reset_peak_memory_stats()
    outcome = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Keep the epoch history separate from a re-evaluation of the reloaded best model.
    history = list(trainer.state.log_history)
    selected = trainer.evaluate()
    adapter = args.output / "adapter"
    trainer.model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    trainer.save_state()
    save_json(args.output / "training_history.json", history)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 4))
    for key, label in (("loss", "Training completion loss"), ("eval_loss", "Validation completion loss")):
        points = [(row["step"], row[key]) for row in history if key in row]
        if points:
            axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=label)
    axis.set(xlabel="Optimizer step", ylabel="Cross-entropy loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "training_curves.png", dpi=160)
    plt.close(figure)
    save_json(args.output / "training_summary.json", {
        "complete": True, "signature": signature, "selection": settings["checkpoint_selection"],
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_validation_loss": trainer.state.best_metric,
        "selected_validation": selected, "train_metrics": outcome.metrics,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "adapter_sha256": {path.name: sha256(path) for path in adapter.iterdir() if path.is_file()}})
    print("Training completed. Selected adapter:", adapter, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/qlora_v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use the previously downloaded tokenizer")
    args = parser.parse_args()
    config, development = load_training_data(args.root)
    tokenizer, _ = load_tokenizer(args.root, offline=args.offline)
    encoded = {name: [encode_record(row, tokenizer, config["max_sequence_length"]) for row in rows]
               for name, rows in development.items()}
    check = {name: {"records": len(rows), "max_tokens": max(len(row["input_ids"]) for row in rows),
                    "supervised_tokens": sum(sum(label != -100 for label in row["labels"]) for row in rows)}
             for name, rows in encoded.items()}
    print("Completion-only data check:", json.dumps(check), flush=True)
    if args.prepare_only:
        (args.root / "reports").mkdir(parents=True, exist_ok=True)
        save_json(args.root / "reports/finetuning_data_check.json", {"groups": check, "truncated_records": 0, "model_weights_loaded": False})
        return
    train(args, config, development, tokenizer, encoded)


if __name__ == "__main__":
    main()
