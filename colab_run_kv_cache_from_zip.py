# Paste into one Colab cell after notebook setup steps 1 and 2.
# Reads the ZIP already uploaded through Colab's Files sidebar.

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import torch
from google.colab import drive

PROJECT = Path("/content/trialgpt_kv_cache")
ZIP_PATH = Path("/content/finetuning_results.zip")
TRAINING = Path("/content/restored_finetuning")
RUN = Path("/content/drive/MyDrive/TrialGPT_Assignment2/kv_cache_v1")

if not ZIP_PATH.is_file():
    raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")
if not (PROJECT / "benchmark_kv_cache.py").is_file():
    raise FileNotFoundError("Run notebook setup steps 1 and 2 first.")
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > GPU.")

with zipfile.ZipFile(ZIP_PATH) as archive:
    summary = json.loads(archive.read("training_summary.json"))
    needed = ["training_summary.json", "training_manifest.json"]
    needed += [f"adapter/{name}" for name in summary["adapter_sha256"]]
    destination = TRAINING.resolve()
    for name in needed:
        if not (destination / name).resolve().is_relative_to(destination):
            raise ValueError("Invalid extraction path in ZIP.")
        archive.getinfo(name)
    TRAINING.mkdir(parents=True, exist_ok=True)
    for name in needed:
        archive.extract(name, TRAINING)

print("Restored adapter from:", ZIP_PATH)

if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")


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
