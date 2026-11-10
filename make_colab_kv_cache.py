"""Package Task 3 code and development prompts; reuse the adapter on Drive."""

import hashlib
import json
from pathlib import Path
import zipfile

from baseline import load_validation
from make_colab_baseline import cell


ROOT = Path(__file__).resolve().parent
FILES = (
    'decoding.py', 'benchmark_kv_cache.py', 'kv_benchmark_report.py',
    'test_decoding.py', 'test_kv_benchmark.py', 'KV_CACHE.md',
    'baseline.py', 'prepare_model_inputs.py', 'eda.py',
    'requirements-colab-finetuning.txt', 'requirements-colab-baseline.txt',
    'requirements-model-inputs.txt', 'requirements-data.txt',
    'configs/model_input_config.json', 'configs/tokenizer_lock.json',
    'configs/kv_cache_config.json', 'data/splits/split_manifest.json',
    'data/model_ready/validation.jsonl',
)


def build(root=ROOT):
    config, records, _ = load_validation(root)
    manifest = {'model_id': config['model_id'], 'revision': config['revision'],
        'validation_records': len(records), 'contains_test_content': False,
        'contains_adapter_weights': False,
        'file_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}}
    with zipfile.ZipFile(root / 'colab_kv_cache_bundle.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root / name, name)
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2) + '\n')
    contents = [
        ('markdown', '''
            # Task 3: explicit KV-cached greedy decoding

            Choose **Runtime > Change runtime type > GPU**, then run cells in order.
            Upload `colab_kv_cache_bundle.zip`. The selected adapter must already exist
            on Drive under `TrialGPT_Assignment2/qlora_v1/adapter` from fine-tuning.
            This notebook does not train. Model weights may download if the runtime is new.

            Two manual forward-pass decoders are compared against HF generation.
            The 3 x 3 stress matrix uses input lengths 128/256/512 and output lengths
            16/32/64, with three timed repetitions per path (54 trials).
            Three complete validation prompts separately check EOS stopping.
            Stress prompts are artificial fixed-token workloads, not accuracy examples.
            No held-out test records are included. A token mismatch remains a failure.
        '''),
        ('markdown', '## 1. Check GPU and upload the code bundle'),
        ('code', f'''
            import hashlib, io, json, zipfile
            from pathlib import Path
            import torch
            from google.colab import files

            if not torch.cuda.is_available():
                raise RuntimeError("Select a GPU runtime before continuing.")
            print("GPU:", torch.cuda.get_device_name(0))
            print("Upload colab_kv_cache_bundle.zip")
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError("Upload exactly one code bundle.")
            PROJECT = Path("/content/trialgpt_kv_cache")
            expected = set({FILES!r})
            with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
                if len(archive.namelist()) != len(expected) + 1 or set(archive.namelist()) != expected | {{"bundle_manifest.json"}}:
                    raise ValueError("Unexpected bundle contents.")
                manifest = json.loads(archive.read("bundle_manifest.json"))
                if set(manifest["file_sha256"]) != expected:
                    raise ValueError("Unexpected bundle manifest.")
                for name in expected:
                    if hashlib.sha256(archive.read(name)).hexdigest() != manifest["file_sha256"][name]:
                        raise ValueError(f"Checksum mismatch: {{name}}")
                PROJECT.mkdir(parents=True, exist_ok=True)
                archive.extractall(PROJECT)
            print("Code bundle verified; held-out test content:", manifest["contains_test_content"])
        '''),
        ('markdown', '''
            ## 2. Install the pinned environment and check the decoder

            Setup removes the unused optional torchao backend, as in the working
            fine-tuning notebook. Tests run real miniature Qwen3 forwards on CPU.
            Their timings do not measure the full selected model.
        '''),
        ('code', '''
            import subprocess, sys

            def run_checked(arguments):
                with subprocess.Popen(arguments, cwd=PROJECT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                    code = process.wait()
                if code:
                    raise RuntimeError(f"Command failed ({code}); full output is above: {arguments}")

            run_checked([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
            run_checked([sys.executable, "-m", "pip", "install", "--quiet", "-r",
                         str(PROJECT / "requirements-colab-finetuning.txt")])
            run_checked([sys.executable, "-m", "unittest", "-v", "test_decoding", "test_kv_benchmark"])
        '''),
        ('markdown', '## 3. Connect Drive and locate the selected adapter'),
        ('code', '''
            from google.colab import drive

            drive.mount("/content/drive")
            TRAINING = Path("/content/drive/MyDrive/TrialGPT_Assignment2/qlora_v1")
            RUN = Path("/content/drive/MyDrive/TrialGPT_Assignment2/kv_cache_v1")
            for relative in ("training_summary.json", "training_manifest.json", "adapter/adapter_model.safetensors"):
                if not (TRAINING / relative).is_file():
                    raise FileNotFoundError(f"Missing {TRAINING / relative}. Use the Drive folder from your completed fine-tuning run.")
            run_checked([sys.executable, "-u", "benchmark_kv_cache.py",
                         "--training-output", str(TRAINING), "--prepare-only"])
        '''),
        ('markdown', '''
            ## 4. Run correctness checks and repeated GPU measurements

            Let this finish. Model loading and correctness tests are outside timed trials.
            Prefill and decode measurements synchronize CUDA. Both paths use the same
            adapter, 4-bit weights, SDPA backend, tokens and greedy rule. Each case/path
            is warmed; path order alternates across repetitions. Raw trials are saved on
            Drive immediately. Rerunning with identical settings resumes missing trials.

            If the cell reports a token mismatch, inspect `correctness.json` and
            `trials.jsonl`. Continue to the download cell to share diagnostics.
            Do not edit or discard mismatches to claim equivalence.
        '''),
        ('code', '''
            run_checked([sys.executable, "-u", "benchmark_kv_cache.py",
                         "--training-output", str(TRAINING), "--output", str(RUN)])
            print((RUN / "summary.md").read_text())
        '''),
        ('markdown', '## 5. Inspect and download the benchmark results'),
        ('code', '''
            from IPython.display import Image, display

            if (RUN / "summary.md").exists():
                print((RUN / "summary.md").read_text())
            if (RUN / "timings.png").exists():
                display(Image(filename=str(RUN / "timings.png")))
            if not RUN.exists():
                raise FileNotFoundError("No benchmark output folder exists yet.")
            download = Path("/content/kv_cache_results.zip")
            with zipfile.ZipFile(download, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in RUN.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(RUN).as_posix())
            files.download(str(download))
        '''),
    ]
    cells = [cell(kind, source, index) for index, (kind, source) in enumerate(contents)]
    for index, value in enumerate(cells):
        value['id'] = f'kv-cache-{index}'
    notebook = {'cells': cells, 'metadata': {'colab': {'name': 'colab_kv_cache.ipynb', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}
    (root / 'colab_kv_cache.ipynb').write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    print('Created colab_kv_cache.ipynb and colab_kv_cache_bundle.zip. No weights, training rows or test content included.')


if __name__ == '__main__':
    build()
