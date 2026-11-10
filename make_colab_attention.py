"""Embed Task 4 source in a self-contained notebook; no runtime file upload."""

import hashlib
import json
from pathlib import Path

from make_colab_baseline import cell


ROOT = Path(__file__).resolve().parent
FILES = ('attention.py', 'benchmark_attention.py', 'attention_benchmark_report.py',
         'test_attention.py', 'test_attention_benchmark.py', 'configs/attention_config.json', 'ATTENTION.md')


def build(root=ROOT):
    sources = {name: (root / name).read_text(encoding='utf-8') for name in FILES}
    hashes = {name: hashlib.sha256(value.encode('utf-8')).hexdigest() for name, value in sources.items()}
    contents = [
        ('markdown', '''
            # Task 4: tiled attention with online softmax

            This notebook contains its source code. No ZIP, dataset, adapter or model
            weights need uploading. It runs standalone attention correctness tests and
            performance measurements; no training takes place.

            Choose **Runtime > Change runtime type > GPU** for measured GPU memory.
            CPU also works, with CPU memory reported as unmeasured. Run all three steps
            in order and download the result ZIP. Optimized SDPA may be faster than
            the educational tiled implementation; observed slowdowns are retained.
        '''),
        ('markdown', '## 1. Restore and verify the embedded source files'),
        ('code', f'''
            import hashlib
            from pathlib import Path

            PROJECT = Path("/content/trialgpt_attention")
            SOURCES = {sources!r}
            EXPECTED_SHA256 = {hashes!r}
            for name, source in SOURCES.items():
                data = source.encode("utf-8")
                if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256[name]:
                    raise ValueError(f"Embedded source checksum mismatch: {{name}}")
                target = PROJECT / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            print("Verified", len(SOURCES), "embedded files. No uploads required.")
        '''),
        ('markdown', '''
            ## 2. Run correctness tests and the attention benchmark

            Inputs are controlled random tensors, not patient records. The experiment
            compares full and causal attention at lengths 128, 256 and 512, using two
            warmups and five repetitions per method. It measures numerical error against
            FP64, runtime and GPU memory where available. It does not rerun KV decoding.
        '''),
        ('code', '''
            import importlib.util
            import subprocess
            import sys
            import torch

            def run_checked(arguments):
                with subprocess.Popen(arguments, cwd=PROJECT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                    code = process.wait()
                if code:
                    raise RuntimeError(f"Command failed ({code}); the full error is printed above.")

            if importlib.util.find_spec("matplotlib") is None:
                run_checked([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib==3.10.7"])
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print("Benchmark device:", DEVICE)
            if DEVICE == "cuda":
                print("GPU:", torch.cuda.get_device_name(0))
            RUN = Path("/content/attention_results")
            run_checked([sys.executable, "-m", "unittest", "-v", "test_attention", "test_attention_benchmark"])
            run_checked([sys.executable, "-u", "benchmark_attention.py", "--device", DEVICE, "--output", str(RUN)])
            print((RUN / "summary.md").read_text(encoding="utf-8"))
        '''),
        ('markdown', '## 3. Inspect the plot and download attention_results.zip'),
        ('code', '''
            import zipfile
            from google.colab import files
            from IPython.display import Image, display

            if (RUN / "attention_comparison.png").is_file():
                display(Image(filename=str(RUN / "attention_comparison.png")))
            if not RUN.is_dir():
                raise FileNotFoundError("The benchmark has not created its output folder yet.")
            download = Path("/content/attention_results.zip")
            with zipfile.ZipFile(download, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in RUN.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(RUN).as_posix())
            files.download(str(download))
        '''),
    ]
    cells = [cell(kind, source, index) for index, (kind, source) in enumerate(contents)]
    for index, value in enumerate(cells):
        value['id'] = f'attention-{index}'
    notebook = {'cells': cells, 'metadata': {'colab': {'name': 'colab_attention.ipynb', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}
    destination = root / 'colab_attention.ipynb'
    destination.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    print('Created colab_attention.ipynb with', len(FILES), 'embedded source files; no runtime uploads needed.')


if __name__ == '__main__':
    build()
