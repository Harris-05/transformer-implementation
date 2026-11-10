# Copy this entire file into one Colab cell after notebook setup steps 1 and 2.
# Place the manually uploaded training files in /content/restored_finetuning/.

import json
import subprocess
import sys
from pathlib import Path

import torch
from google.colab import drive

PROJECT = Path("/content/trialgpt_kv_cache")
TRAINING = Path("/content/restored_finetuning")
RUN = Path("/content/drive/MyDrive/TrialGPT_Assignment2/kv_cache_v1")

if not (PROJECT / "benchmark_kv_cache.py").is_file():
    raise FileNotFoundError("Run notebook setup steps 1 and 2 first.")
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > GPU.")

required = {
    "training_summary.json",
    "training_manifest.json",
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
}
summary_path = TRAINING / "training_summary.json"
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required.update(f"adapter/{name}" for name in summary["adapter_sha256"])

missing = [str(TRAINING / name) for name in sorted(required)
           if not (TRAINING / name).is_file()]
if missing:
    raise FileNotFoundError("Place the missing files in these locations:\n" + "\n".join(missing))

print("Using uploaded adapter:", TRAINING / "adapter")
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
