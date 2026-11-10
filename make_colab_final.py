"""Embed exact source/data bytes for final evaluation; never inspect held-out rows."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from make_colab_baseline import cell

ROOT=Path(__file__).resolve().parent
FILES=('evaluate_final.py','test_final_evaluation.py','evaluate_adapter.py','baseline.py','prepare_model_inputs.py','eda.py',
       'configs/model_input_config.json','configs/tokenizer_lock.json','data/splits/split_manifest.json',
       'requirements-colab-finetuning.txt','requirements-colab-baseline.txt','requirements-model-inputs.txt','requirements-data.txt',
       'train-00000-of-00001.parquet')


def build():
    buffer=io.BytesIO()
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:archive.writestr(name,(ROOT/name).read_bytes())
    encoded=base64.b64encode(buffer.getvalue()).decode()
    restore=f'''import base64, hashlib, io, json, subprocess, sys, zipfile
from pathlib import Path
FINAL_PROJECT = Path("/content/trialgpt_final_evaluation")
PAYLOAD = {encoded!r}
EXPECTED = {hashes!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    for name, digest in EXPECTED.items():
        data = archive.read(name)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Embedded checksum differs: " + name)
        destination = FINAL_PROJECT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
def final_run(arguments):
    with subprocess.Popen(arguments, cwd=FINAL_PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        for line in process.stdout: print(line, end="", flush=True)
        code = process.wait()
    if code: raise RuntimeError("Final evaluation failed; read the output above. Partial predictions are preserved.")
print("Restored final-evaluation code and original dataset with exact checksums; no test predictions made yet.")
'''
    run='''import torch
from google.colab import drive
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime. This evaluates both models; no training occurs.")
if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")
candidates = []
if globals().get("TRAINING") is not None:
    candidates.append(Path(TRAINING))
candidates += [Path("/content/drive/MyDrive/TrialGPT_Assignment2/selected_adapter"),
               Path("/content/drive/MyDrive/TrialGPT_Assignment2/qlora_v1")]
FINAL_TRAINING = next((p for p in candidates if (p / "training_manifest.json").is_file()
                       and (p / "adapter/adapter_model.safetensors").is_file()), None)
if FINAL_TRAINING is None:
    raise FileNotFoundError("No selected adapter found. Keep TRAINING from the speculative notebook or save the adapter to Drive.")
FINAL_RUN = Path("/content/drive/MyDrive/TrialGPT_Assignment2/final_test_v1")
final_run([sys.executable, "-m", "unittest", "-v", "test_final_evaluation"])
final_run([sys.executable, "-u", "evaluate_final.py", "--training-output", str(FINAL_TRAINING), "--output", str(FINAL_RUN)])
print((FINAL_RUN / "summary.md").read_text())
'''
    download='''from google.colab import files
download = Path("/content/final_test_results.zip")
with zipfile.ZipFile(download, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in FINAL_RUN.rglob("*"):
        if path.is_file(): archive.write(path, path.relative_to(FINAL_RUN).as_posix())
files.download(str(download))
'''
    (ROOT/'colab_run_final_evaluation.py').write_text('# Copy this entire file into ONE cell in the existing speculative Colab runtime.\n# Uses already installed dependencies and the selected adapter; no retraining.\n'+restore+'\n'+run+'\n'+download,encoding='utf-8')
    cells=[('markdown','# Final held-out before/after evaluation\n\nRun after model/prompt choices are fixed. This is inference only. It evaluates original and selected-adapter models on the prescribed eight test patients, with the same FP16/NF4 protocol and no tuning. The original dataset bytes are embedded; no upload widget. Download final_test_results.zip and return it for the final report.'),
           ('code',restore),('markdown','## Install pinned dependencies (needed in a fresh runtime)'),
           ('code','final_run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])\nfinal_run([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(FINAL_PROJECT / "requirements-colab-finetuning.txt")])\n'),
           ('markdown','## Run the fixed final comparison'),('code',run),('markdown','## Download results'),('code',download)]
    notebook={'cells':[cell(kind,text,i) for i,(kind,text) in enumerate(cells)],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
    (ROOT/'colab_final_evaluation.ipynb').write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
    print('Created final notebook and one-cell current-runtime helper with',len(FILES),'exact-byte assets.')


if __name__=='__main__':build()
