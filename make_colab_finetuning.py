"""Package QLoRA training, adapter evaluation, and the frozen baseline."""

import hashlib
import json
import zipfile
from pathlib import Path

from finetune import load_training_data
from make_colab_baseline import cell


ROOT = Path(__file__).resolve().parent
FILES = (
    "finetune.py", "evaluate_adapter.py", "verify_training.py", "test_finetuning.py",
    "baseline.py", "prepare_model_inputs.py", "eda.py",
    "requirements-colab-finetuning.txt", "requirements-colab-baseline.txt",
    "requirements-model-inputs.txt", "requirements-data.txt",
    "configs/qlora_config.json", "configs/model_input_config.json", "configs/tokenizer_lock.json",
    "data/splits/split_manifest.json", "data/model_ready/train.jsonl", "data/model_ready/validation.jsonl",
    "data/processed/quarantined_train.jsonl",
    "results/baseline_validation/metrics.json", "results/baseline_validation/predictions.jsonl",
    "results/baseline_validation/run_manifest.json",
)


def build(root=ROOT):
    config, development = load_training_data(root)
    manifest = {"model_id": config["model_id"], "revision": config["revision"],
                "counts": {name: len(rows) for name, rows in development.items()},
                "contains_test_content": False,
                "file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}}
    with zipfile.ZipFile(root / "colab_finetuning_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root / name, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    contents = [
        ("markdown", """
            # QLoRA fine-tuning and validation comparison

            Select **Runtime > Change runtime type > GPU** and run the cells in order.
            Upload `colab_finetuning_bundle.zip` when asked. This trains on 602 expert-annotated
            examples and uses 206 validation examples for checkpoint selection and comparison.
            No held-out test records are included.

            The original baseline and prompt are frozen. Some exclusion-label disagreements reflect
            annotation/information-policy differences; labels are retained without inventing rules
            that turn missing information into eligibility. Improvement is measured, not assumed.

            The first run uses 4-bit NF4 base weights with FP16 computation and rank-16 LoRA adapters.
            Maximum three epochs; completion-only loss; effective training batch size eight.
            Gradient checkpointing and batch size one reduce memory. Actual GPU training has to run
            here; local CPU checks do not prove the full model fits or improves on the T4.
        """),
        ("markdown", """
            ## 1. Check GPU and upload the training bundle
        """),
        ("code", f"""
            import hashlib, io, json, zipfile
            from pathlib import Path
            import torch
            from google.colab import files

            if not torch.cuda.is_available():
                raise RuntimeError("Enable a GPU using Runtime > Change runtime type, then reconnect.")
            print("GPU:", torch.cuda.get_device_name(0))
            print("VRAM (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
            print("Upload colab_finetuning_bundle.zip")
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError("Upload exactly one ZIP bundle.")
            PROJECT = Path("/content/trialgpt_finetuning")
            expected_files = set({FILES!r})
            with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
                if set(archive.namelist()) != expected_files | {{"bundle_manifest.json"}}:
                    raise ValueError("Unexpected bundle file list.")
                manifest = json.loads(archive.read("bundle_manifest.json"))
                if set(manifest["file_sha256"]) != expected_files:
                    raise ValueError("Unexpected manifest file list.")
                for name in expected_files:
                    if hashlib.sha256(archive.read(name)).hexdigest() != manifest["file_sha256"][name]:
                        raise ValueError(f"Checksum mismatch: {{name}}")
                PROJECT.mkdir(parents=True, exist_ok=True)
                archive.extractall(PROJECT)
            print("Verified bundle counts:", manifest["counts"])
        """),
        ("markdown", """
            ## 2. Install and run preparation/integration checks

            The tiny CPU check verifies actual Trainer/PEFT adapter updates on a random miniature
            model. It downloads no model weights and produces no clinical results. The preparation
            check verifies all real example lengths and completion masks using the selected tokenizer.
            Setup removes Colab's optional torchao package: this experiment uses bitsandbytes, and an
            incompatible preinstalled torchao can prevent PEFT from attaching adapters.
        """),
        ("code", """
            import subprocess, sys

            def run_checked(arguments):
                # Print child output through the notebook kernel so errors remain visible.
                with subprocess.Popen(arguments, cwd=PROJECT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                    return_code = process.wait()
                if return_code:
                    raise RuntimeError(f"Command failed with exit code {return_code}. The full error is printed above: {arguments}")

            # PEFT probes optional backends even when the selected backend is bitsandbytes.
            # https://github.com/huggingface/peft/blob/v0.21.2/src/peft/import_utils.py
            run_checked([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
            run_checked([sys.executable, "-m", "pip", "install", "--quiet", "-r",
                         str(PROJECT / "requirements-colab-finetuning.txt")])
            run_checked([sys.executable, "-m", "unittest", "-v", "test_finetuning"])
            run_checked([sys.executable, "-u", "verify_training.py"])
            run_checked([sys.executable, "-u", "finetune.py", "--prepare-only"])
        """),
        ("markdown", """
            ## 3. Connect Drive for adapters and checkpoints

            Authorize Drive access. Use the same directory when resuming this exact experiment.
            Base-model weights remain in Colab's temporary cache. Adapter and optimizer checkpoints
            are saved after each epoch; an interrupted partial epoch may have to be repeated.
        """),
        ("code", """
            from google.colab import drive

            drive.mount("/content/drive")
            RUN = Path("/content/drive/MyDrive/TrialGPT_Assignment2/qlora_v1")
            RUN.mkdir(parents=True, exist_ok=True)
            print("Training output:", RUN)
        """),
        ("markdown", """
            ## 4. Train and save the selected adapter

            Let this cell finish. It loads the same pinned base checkpoint, freezes its base weights,
            and trains LoRA adapters with prompt/padding labels masked out. It evaluates completion
            loss each epoch and reloads the saved checkpoint with the lowest validation loss.
            Early stopping is enabled. Training can take substantially longer than the baseline.

            Re-running this cell resumes a saved checkpoint with matching settings. After successful
            completion it reports that the selected adapter already exists rather than retraining.
        """),
        ("code", """
            run_checked([sys.executable, "-u", "finetune.py", "--output", str(RUN)])
            print((RUN / "training_summary.json").read_text())
        """),
        ("markdown", """
            ## 5. Evaluate the selected adapter on the same validation examples

            This runs in a fresh process so training activations and optimizer memory are released.
            It reloads the original 4-bit base plus the selected adapter and uses the same prompt,
            generation allowance, greedy decoding, and scoring functions as the original baseline.
            All 206 cases must finish; predictions are saved individually and can be resumed.
        """),
        ("code", """
            run_checked([sys.executable, "-u", "evaluate_adapter.py", "--training-output", str(RUN)])
            print((RUN / "validation/summary.md").read_text())
        """),
        ("markdown", """
            ## 6. Inspect curves and download the adapter/results

            Keep the downloaded ZIP. It contains the selected adapter, logs, training curves, and
            validation comparison. Optimizer checkpoints stay on Drive and are excluded from the
            download to reduce its size. Share `validation/comparison.json`, `validation/metrics.json`,
            and `training_summary.json` before selecting further experiments. Scores may improve or
            worsen; the report retains both outcomes.
        """),
        ("code", """
            from IPython.display import Image, display

            display(Image(filename=str(RUN / "training_curves.png")))
            display(Image(filename=str(RUN / "validation/confusion_overall.png")))
            print((RUN / "validation/summary.md").read_text())
            download = Path("/content/finetuning_results.zip")
            with zipfile.ZipFile(download, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in RUN.rglob("*"):
                    relative = path.relative_to(RUN)
                    if path.is_file() and "checkpoints" not in relative.parts:
                        archive.write(path, relative.as_posix())
            files.download(str(download))
        """),
    ]
    cells = [cell(kind, source, i) for i, (kind, source) in enumerate(contents)]
    for i, value in enumerate(cells):
        value["id"] = f"finetuning-{i}"
    notebook = {"cells": cells, "metadata": {"colab": {"name": "colab_finetuning.ipynb", "provenance": []},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    (root / "colab_finetuning.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print("Created colab_finetuning.ipynb and colab_finetuning_bundle.zip.")
    print("Bundle:", manifest["counts"], "No held-out test content.")


if __name__ == "__main__":
    build()
