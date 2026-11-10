"""Self-contained Task 5 notebook; reuse adapter files without upload widgets."""

import hashlib
import json
from pathlib import Path

from make_colab_baseline import cell


ROOT = Path(__file__).resolve().parent
FILES = (
    'speculative_decoding.py', 'speculative_inputs.py', 'speculative_report.py', 'benchmark_speculative.py',
    'test_speculative.py', 'test_speculative_inputs.py', 'test_speculative_benchmark.py',
    'decoding.py', 'benchmark_kv_cache.py', 'kv_benchmark_report.py',
    'baseline.py', 'prepare_model_inputs.py', 'eda.py', 'SPECULATIVE.md',
    'configs/speculative_config.json', 'configs/draft_tokenizer_lock.json',
    'configs/model_input_config.json', 'configs/tokenizer_lock.json',
    'data/model_ready/validation.jsonl', 'data/splits/split_manifest.json',
    'requirements-colab-finetuning.txt', 'requirements-colab-baseline.txt',
    'requirements-model-inputs.txt', 'requirements-data.txt',
)


def build(root=ROOT):
    # Text-mode reads normalize CRLF and break the frozen training data hashes.
    sources = {name: (root / name).read_bytes().decode('utf-8') for name in FILES}
    hashes = {name: hashlib.sha256(source.encode('utf-8')).hexdigest() for name, source in sources.items()}
    contents = [
        ('markdown', '''
            # Task 5: manual greedy speculative decoding

            Target: your fine-tuned **Qwen3-4B-Instruct-2507** adapter. Draft:
            **Qwen3-0.6B**. Both revisions and tokenizer assets are pinned.
            This notebook performs inference, not training. It embeds code and
            development prompts: no new KV/speculative code bundle needs uploading.

            Select **Runtime > Change runtime type > GPU**, then run steps in order.
            To avoid reuploading your adapter in a fresh runtime, first run
            `colab_save_adapter_to_drive.py` in the old runtime where its files exist.
            Step 3 finds that Drive backup automatically. Existing training folders,
            flat `/content` files and `/content/finetuning_results.zip` also work.
            No cell opens an upload widget.

            The controlled target uses 4-bit weights with FP32 computation and math
            SDPA in both ordinary and speculative paths; the draft uses FP16.
            These results are separate from the earlier FP16 KV/classification runs.
            Full model weights may download into this runtime's cache.
            Preflight checks token equality before spending time on the 54 timed
            trials. Numerical mismatches remain failures; speedup is not assumed.
        '''),
        ('markdown', '## 1. Restore and verify embedded code and development inputs'),
        ('code', f'''
            import hashlib, json
            from pathlib import Path

            PROJECT = Path("/content/trialgpt_speculative")
            SOURCES = {sources!r}
            EXPECTED_SHA256 = {hashes!r}
            for name, source in SOURCES.items():
                data = source.encode("utf-8")
                if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256[name]:
                    raise ValueError(f"Embedded checksum mismatch: {{name}}")
                path = PROJECT / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            print("Verified", len(SOURCES), "files. No code/data upload needed; no held-out test examples included.")
        '''),
        ('markdown', '## 2. Check GPU, install the pinned dependencies and run tiny-model tests'),
        ('code', '''
            import subprocess, sys
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Select a GPU runtime before continuing.")
            print("GPU:", torch.cuda.get_device_name(0))

            def run_checked(arguments):
                with subprocess.Popen(arguments, cwd=PROJECT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                      errors="replace", bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                    code = process.wait()
                if code:
                    raise RuntimeError(f"Command failed ({code}); full output is printed above.")

            run_checked([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
            run_checked([sys.executable, "-m", "pip", "install", "--quiet", "-r",
                         str(PROJECT / "requirements-colab-finetuning.txt")])
            run_checked([sys.executable, "-m", "unittest", "-v", "test_speculative",
                         "test_speculative_inputs", "test_speculative_benchmark"])
        '''),
        ('markdown', '''
            ## 3. Locate the existing adapter and verify tokenizer compatibility

            Authorize Drive access if requested. The default backup is
            `MyDrive/TrialGPT_Assignment2/selected_adapter`. This step can also reuse
            files already in the runtime, including all eight files placed directly
            in `/content`. If none are found, upload the results ZIP using Colab's
            Files sidebar, then rerun this cell. No model weights are loaded here.
        '''),
        ('code', '''
            import tempfile, zipfile
            from google.colab import drive

            if not Path("/content/drive/MyDrive").is_dir():
                drive.mount("/content/drive")
            candidates = []
            if globals().get("TRAINING") is not None:
                candidates.append(Path(TRAINING))
            candidates += [Path("/content/drive/MyDrive/TrialGPT_Assignment2/selected_adapter"),
                           Path("/content/drive/MyDrive/TrialGPT_Assignment2/qlora_v1"),
                           Path("/content/restored_finetuning")]
            TRAINING = next((p for p in candidates if (p / "training_summary.json").is_file()
                             and (p / "training_manifest.json").is_file()
                             and (p / "adapter/adapter_model.safetensors").is_file()), None)
            if TRAINING is None and Path("/content/training_summary.json").is_file():
                source = Path("/content")
                summary = json.loads((source / "training_summary.json").read_text())
                names = list(summary["adapter_sha256"])
                if any(Path(name).name != name for name in names):
                    raise ValueError("Expected plain adapter filenames.")
                required = ["training_summary.json", "training_manifest.json", *names]
                missing = [name for name in required if not (source / name).is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing files in /content: {missing}")
                TRAINING = Path(tempfile.mkdtemp(prefix="speculative_training_", dir="/content"))
                (TRAINING / "adapter").mkdir()
                for name in ("training_summary.json", "training_manifest.json"):
                    (TRAINING / name).symlink_to(source / name)
                for name in names:
                    (TRAINING / "adapter" / name).symlink_to(source / name)
            if TRAINING is None and Path("/content/finetuning_results.zip").is_file():
                TRAINING = Path(tempfile.mkdtemp(prefix="speculative_training_", dir="/content"))
                with zipfile.ZipFile("/content/finetuning_results.zip") as archive:
                    summary = json.loads(archive.read("training_summary.json"))
                    names = ["training_summary.json", "training_manifest.json"]
                    names += [f"adapter/{name}" for name in summary["adapter_sha256"]]
                    for name in names:
                        if not (TRAINING / name).resolve().is_relative_to(TRAINING.resolve()):
                            raise ValueError("Invalid ZIP path.")
                        archive.getinfo(name)
                    for name in names:
                        archive.extract(name, TRAINING)
            if TRAINING is None:
                raise FileNotFoundError("No adapter found. Save it to Drive from the old runtime, or place finetuning_results.zip in /content using the Files sidebar.")
            RUN = Path("/content/drive/MyDrive/TrialGPT_Assignment2/speculative_v1")
            print("Selected training files:", TRAINING)
            run_checked([sys.executable, "-u", "benchmark_speculative.py",
                         "--training-output", str(TRAINING), "--prepare-only"])
        '''),
        ('markdown', '''
            ## 4. Run preflight and the actual paired-model benchmark

            This loads both models, checks six development prompts against ordinary
            target greedy and HF, then measures three modes with three repetitions.
            Latency includes drafting/verification/rollback. Both models remain
            resident in every mode, including ordinary greedy, and memory is labelled
            accordingly. Results persist on Drive. A failed preflight is preserved;
            use a new RUN folder for any intentionally changed experiment.
        '''),
        ('code', '''
            run_checked([sys.executable, "-u", "benchmark_speculative.py",
                         "--training-output", str(TRAINING), "--output", str(RUN)])
            print((RUN / "summary.md").read_text(encoding="utf-8"))
        '''),
        ('markdown', '## 5. Download speculative_results.zip (also useful if preflight failed)'),
        ('code', '''
            from google.colab import files
            from IPython.display import Image, display

            if (RUN / "summary.md").exists():
                print((RUN / "summary.md").read_text(encoding="utf-8"))
            if (RUN / "timings.png").exists():
                display(Image(filename=str(RUN / "timings.png")))
            if not RUN.is_dir():
                raise FileNotFoundError("No benchmark output folder exists yet.")
            download = Path("/content/speculative_results.zip")
            with zipfile.ZipFile(download, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in RUN.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(RUN).as_posix())
            files.download(str(download))
        '''),
    ]
    cells = [cell(kind, source, index) for index, (kind, source) in enumerate(contents)]
    for index, value in enumerate(cells):
        value['id'] = f'speculative-{index}'
    notebook = {'cells': cells, 'metadata': {'colab': {'name': 'colab_speculative.ipynb', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}
    path = root / 'colab_speculative.ipynb'
    path.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    print('Created colab_speculative.ipynb with', len(FILES), 'embedded files; no code bundle or upload widget.')


if __name__ == '__main__':
    build()
