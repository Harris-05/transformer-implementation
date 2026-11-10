"""Build the Colab notebook and a validation-only upload bundle."""

import hashlib
import json
import textwrap
import zipfile
from pathlib import Path

from baseline import load_validation


ROOT = Path(__file__).resolve().parent
FILES = (
    "baseline.py", "prepare_model_inputs.py", "eda.py", "test_baseline.py",
    "requirements-colab-baseline.txt", "requirements-model-inputs.txt", "requirements-data.txt",
    "configs/model_input_config.json", "configs/tokenizer_lock.json",
    "data/splits/split_manifest.json", "data/model_ready/validation.jsonl",
)


def cell(kind, source, number):
    value = {"cell_type": kind, "id": f"baseline-{number}", "metadata": {},
             "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build(root=ROOT):
    config, records, _ = load_validation(root)
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}
    manifest = {"model_id": config["model_id"], "revision": config["revision"],
                "validation_records": len(records), "contains_test_content": False, "file_sha256": hashes}
    bundle = root / "colab_baseline_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root / name, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")

    cells = [
        ("markdown", """
            # Qwen3-4B original-model validation baseline

            Run the cells from top to bottom. First select **Runtime > Change runtime type > GPU**
            (a free T4 is suitable when available). Upload `colab_baseline_bundle.zip` when asked.

            This uses the 206 validation examples only. The original model is loaded in 4-bit NF4
            precision with FP16 computation and greedy decoding. No training or test evaluation
            occurs. The model weights download is several GB; no paid API or access token is required.

            Local validation/metric tests have been run. The actual CUDA model run happens in this
            notebook. Results are saved to your Google Drive after each example so you can resume.
        """),
        ("markdown", """
            ## 1. Check the GPU and upload the prepared bundle

            Run this cell and choose the ZIP file from your laptop. It verifies the file list and
            checksums, then extracts the prepared project into `/content/trialgpt_baseline`.
        """),
        ("code", f"""
            import hashlib, io, json, zipfile
            from pathlib import Path
            import torch
            from google.colab import files

            if not torch.cuda.is_available():
                raise RuntimeError("Choose Runtime > Change runtime type > GPU, then reconnect and run this cell.")
            print("GPU:", torch.cuda.get_device_name(0))
            print("VRAM (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
            print("Upload colab_baseline_bundle.zip")
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError("Upload exactly one baseline bundle ZIP.")
            PROJECT = Path("/content/trialgpt_baseline")
            expected_files = set({FILES!r})
            with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
                if set(archive.namelist()) != expected_files | {{"bundle_manifest.json"}}:
                    raise ValueError("Unexpected files in the bundle. Rebuild it with make_colab_baseline.py.")
                bundle_manifest = json.loads(archive.read("bundle_manifest.json"))
                if set(bundle_manifest["file_sha256"]) != expected_files:
                    raise ValueError("Bundle manifest has a different file list.")
                for name in expected_files:
                    if hashlib.sha256(archive.read(name)).hexdigest() != bundle_manifest["file_sha256"][name]:
                        raise ValueError(f"Bundle checksum mismatch: {{name}}")
                PROJECT.mkdir(parents=True, exist_ok=True)
                archive.extractall(PROJECT)
            print("Verified and extracted", bundle_manifest["validation_records"], "validation records.")
        """),
        ("markdown", """
            ## 2. Install packages and check the data/evaluator

            This uses the pinned preparation dependencies plus Accelerate and bitsandbytes. It keeps
            Colab's CUDA-enabled PyTorch installation. The checks below do not download model weights.
        """),
        ("code", """
            import subprocess, sys

            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r",
                            str(PROJECT / "requirements-colab-baseline.txt")], check=True)
            subprocess.run([sys.executable, "-m", "unittest", "-v", "test_baseline"], cwd=PROJECT, check=True)
            subprocess.run([sys.executable, "-u", "baseline.py", "--validate-only"], cwd=PROJECT, check=True)
        """),
        ("markdown", """
            ## 3. Connect Drive for progress saving

            Authorize your Google Drive when prompted. The output directory below stores predictions,
            metrics, error cases, environment details, and confusion matrices. Model weights stay in
            Colab's temporary cache. Keep the same directory when resuming this exact run.
        """),
        ("code", """
            from google.colab import drive

            drive.mount("/content/drive")
            OUTPUT = Path("/content/drive/MyDrive/TrialGPT_Assignment2/baseline_validation_v1")
            OUTPUT.mkdir(parents=True, exist_ok=True)
            print("Results and progress:", OUTPUT)
        """),
        ("markdown", """
            ## 4. Run the original model on all 206 validation examples

            Wait for this cell to finish. It downloads and loads the pinned model, performs a short
            warm-up, then prints one progress line per example. Every prediction is saved immediately.
            Re-running the cell resumes matching saved work. Changed code, data, package versions, or
            settings are not mixed into a previous run. If the runtime disconnects, reconnect to a GPU
            and rerun the notebook with the same bundle and Drive directory.

            This baseline uses the library's normal greedy generation. The assignment's custom
            cache-aware and speculative decoders will be implemented separately later.
        """),
        ("code", """
            subprocess.run([sys.executable, "-u", "baseline.py", "--output", str(OUTPUT)],
                           cwd=PROJECT, check=True)
        """),
        ("markdown", """
            ## 5. Inspect and download the results

            `complete: true` and `evaluated_records: 206` indicate the complete validation baseline.
            Accuracy, macro-F1, class details, evidence scores, and format validity are reported.
            Both-empty evidence sets receive F1=1; the report also isolates nonempty gold cases.
            Malformed responses remain errors and are never removed from evaluation.

            Download the ZIP and keep it for the report. Share `metrics.json` and `summary.md` to
            interpret the results before starting fine-tuning.
        """),
        ("code", """
            import shutil
            from IPython.display import Image, display

            metrics = json.loads((OUTPUT / "metrics.json").read_text())
            print("Complete:", metrics["complete"], "| Evaluated:", metrics["evaluated_records"])
            print((OUTPUT / "summary.md").read_text())
            display(Image(filename=str(OUTPUT / "confusion_overall.png")))
            archive_path = shutil.make_archive("/content/baseline_results", "zip", root_dir=OUTPUT)
            files.download(archive_path)
        """),
    ]
    notebook = {"cells": [cell(kind, source, i) for i, (kind, source) in enumerate(cells)],
                "metadata": {"colab": {"name": "colab_baseline.ipynb", "provenance": []},
                             "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU"},
                "nbformat": 4, "nbformat_minor": 5}
    (root / "colab_baseline.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(f"Created colab_baseline.ipynb and {bundle.name} ({bundle.stat().st_size / 1024:.1f} KiB).")
    print(f"Bundle: {len(records)} validation rows; no training rows, test content, or model weights.")


if __name__ == "__main__":
    build()
