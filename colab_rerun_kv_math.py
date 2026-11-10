# Copy this entire file into ONE new Colab cell in your current runtime.
# Uses the existing PROJECT/TRAINING files; reruns inference with math SDPA.
# Original kv_cache_v1 results are preserved.

import hashlib
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/content/trialgpt_kv_cache")
RUN = Path("/content/drive/MyDrive/TrialGPT_Assignment2/kv_cache_math_v1")
if "TRAINING" not in globals() or not (Path(TRAINING) / "training_summary.json").is_file():
    raise FileNotFoundError("Use this cell in the same runtime after your uploaded-files benchmark cell.")

script = PROJECT / "benchmark_kv_cache.py"
ORIGINAL_SHA256 = 'cdf0a8d7f28d98ce0005bcd5df138aac878a743dde5993c5c4c7e69f71e52fca'
UPDATED_SHA256 = 'f1000abc7ae04a0c6fc85c902f9f777e98116a7affd54618a171f9d59e77e92e'
# Exact reviewed line edits; hashes prevent applying them to another version.
EDITS = [
    (3, 3, 'from contextlib import nullcontext\n'),
    (12, 12, 'from torch.nn.attention import SDPBackend, sdpa_kernel\n'),
    (21, 21, '\n\ndef attention_backend(name):\n    """Control all forwards, HF checks and timings with one recorded backend."""\n    if name == \'automatic\':\n        return nullcontext()\n    if name == \'math\':\n        return sdpa_kernel(backends=[SDPBackend.MATH])\n    raise ValueError(f\'Unsupported SDPA backend: {name}\')\n'),
    (173, 173, "    settings['sdpa_backend'] = args.sdpa_backend\n"),
    (215, 216, '    print(f"SDPA backend: {settings[\'sdpa_backend\']}", flush=True)\n    with attention_backend(settings[\'sdpa_backend\']):\n        experiment(model_loader(), cases, clinical, settings, output, signature, smoke=args.smoke)\n'),
    (225, 225, "    parser.add_argument('--sdpa-backend', choices=('automatic', 'math'), default='automatic',\n                        help='Apply the same backend to manual decoding, HF checks, warmups and timed trials.')\n"),
]

current = script.read_bytes()
digest = hashlib.sha256(current).hexdigest()
if digest == ORIGINAL_SHA256:
    lines = current.decode("utf-8").splitlines(keepends=True)
    for start, end, replacement in reversed(EDITS):
        lines[start:end] = [replacement]
    updated = "".join(lines).encode("utf-8")
    if hashlib.sha256(updated).hexdigest() != UPDATED_SHA256:
        raise ValueError("Backend update verification failed; the original script was not changed.")
    backup = script.with_name("benchmark_kv_cache_before_math.py")
    if not backup.exists():
        backup.write_bytes(current)
    script.write_bytes(updated)
elif digest != UPDATED_SHA256:
    raise ValueError("The benchmark source differs from the reviewed version. Upload the updated benchmark_kv_cache.py.")

print("Backend: math SDPA for both decoding paths and HF checks.")
print("New results:", RUN)
arguments = [sys.executable, "-u", str(script), "--training-output", str(TRAINING),
             "--output", str(RUN), "--sdpa-backend", "math"]
with subprocess.Popen(arguments, cwd=PROJECT, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                      errors="replace", bufsize=1) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    exit_code = process.wait()
if exit_code:
    raise RuntimeError("The math-SDPA benchmark did not pass. Inspect the saved diagnostics and error above.")
print((RUN / "summary.md").read_text(encoding="utf-8"))
