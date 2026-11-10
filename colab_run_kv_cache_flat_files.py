# Copy this entire file into ONE Colab cell after notebook setup steps 1 and 2.
# All uploaded training files are read directly from /content, without a ZIP.

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from google.colab import drive

PROJECT = Path("/content/trialgpt_kv_cache")
SOURCE = Path("/content")
RUN = Path("/content/drive/MyDrive/TrialGPT_Assignment2/kv_cache_v1")

if not (PROJECT / "benchmark_kv_cache.py").is_file():
    raise FileNotFoundError("Run notebook setup steps 1 and 2 first.")
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > GPU.")

for name in ("training_summary.json", "training_manifest.json"):
    if not (SOURCE / name).is_file():
        raise FileNotFoundError(f"Missing uploaded file: {SOURCE / name}")

summary = json.loads((SOURCE / "training_summary.json").read_text(encoding="utf-8"))
adapter_files = list(summary["adapter_sha256"])
for name in adapter_files:
    if Path(name).name != name:
        raise ValueError(f"Expected an adapter filename, got: {name}")
missing = [str(SOURCE / name) for name in adapter_files if not (SOURCE / name).is_file()]
if missing:
    raise FileNotFoundError("Missing uploaded files:\n" + "\n".join(missing))

# The benchmark requires an adapter subfolder internally. Links refer to the
# existing /content files; they do not duplicate the model or move your uploads.
TRAINING = Path(tempfile.mkdtemp(prefix="kv_training_links_", dir=SOURCE))
(TRAINING / "adapter").mkdir()
for name in ("training_summary.json", "training_manifest.json"):
    (TRAINING / name).symlink_to(SOURCE / name)
for name in adapter_files:
    (TRAINING / "adapter" / name).symlink_to(SOURCE / name)

print("Using training files already uploaded to /content.")
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
