# Copy this entire file into ONE Colab cell after notebook setup steps 1 and 2.
# This replaces steps 3 and 4: upload results, restore adapter, run benchmark.

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import torch
from google.colab import drive, files

PROJECT = Path("/content/trialgpt_kv_cache")
RUN = Path("/content/drive/MyDrive/TrialGPT_Assignment2/kv_cache_v1")
TRAINING = Path("/content/restored_finetuning")

if not (PROJECT / "benchmark_kv_cache.py").is_file():
    raise FileNotFoundError("Run notebook setup steps 1 and 2 first to load the code bundle and dependencies.")
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > GPU before continuing.")

if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")

print("Select finetuning_results.zip (the downloaded fine-tuning results).")
uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload exactly one file: finetuning_results.zip.")

required = {
    "training_summary.json",
    "training_manifest.json",
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
}

with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
    if not required.issubset(archive.namelist()):
        raise ValueError("This ZIP does not contain the expected fine-tuning results. Select finetuning_results.zip.")
    destination = TRAINING.resolve()
    for member in archive.infolist():
        if not (destination / member.filename).resolve().is_relative_to(destination):
            raise ValueError("The ZIP contains a path outside the extraction folder.")
    TRAINING.mkdir(parents=True, exist_ok=True)
    archive.extractall(TRAINING)

del uploaded
print("Restored adapter:", TRAINING / "adapter")
print("Benchmark results will be saved to:", RUN)


def run_checked(arguments):
    with subprocess.Popen(
        arguments, cwd=PROJECT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"Command failed with exit code {exit_code}. Read the full error above.")


run_checked([
    sys.executable, "-u", "benchmark_kv_cache.py",
    "--training-output", str(TRAINING), "--prepare-only",
])

run_checked([
    sys.executable, "-u", "benchmark_kv_cache.py",
    "--training-output", str(TRAINING), "--output", str(RUN),
])

print((RUN / "summary.md").read_text(encoding="utf-8"))
