"""Build one copyable GPU integration check cell; no model training or uploads."""
import base64
import hashlib
import io
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
FILES=('app_smoke.py','app.py','app_model.py','screening.py','decoding.py','benchmark_kv_cache.py','kv_benchmark_report.py',
       'speculative_decoding.py','speculative_inputs.py','baseline.py','prepare_model_inputs.py','eda.py',
       'configs/model_input_config.json','configs/tokenizer_lock.json','configs/draft_tokenizer_lock.json',
       'web/index.html','web/app.js','web/style.css','web/instrument-sans.ttf','web/instrument-serif.ttf')


def build():
    buffer=io.BytesIO();hashes={}
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED)as archive:
        for name in FILES:
            data=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();archive.writestr(name,data)
    code=f'''# Copy into ONE new cell in the current GPU Colab runtime. No training.
import base64,hashlib,io,json,subprocess,sys,zipfile
from pathlib import Path
from google.colab import files
APP_PROJECT=Path("/content/trialscreen_application")
PAYLOAD={base64.b64encode(buffer.getvalue()).decode()!r}
EXPECTED={hashes!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD)))as archive:
    for name,digest in EXPECTED.items():
        data=archive.read(name)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError("Embedded file differs: "+name)
        path=APP_PROJECT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
APP_TRAINING=globals().get("FINAL_TRAINING") or globals().get("TRAINING")
if APP_TRAINING is None:raise RuntimeError("Keep the selected adapter path from the previous notebooks.")
APP_RUN=Path("/content/drive/MyDrive/TrialGPT_Assignment2/application_check_v1")
with subprocess.Popen([sys.executable,"-u","app_smoke.py","--training-output",str(APP_TRAINING),"--output",str(APP_RUN)],cwd=APP_PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)as process:
    for line in process.stdout:print(line,end="",flush=True)
    exit_code=process.wait()
download=Path("/content/application_results.zip")
with zipfile.ZipFile(download,"w",compression=zipfile.ZIP_DEFLATED)as archive:
    for path in APP_RUN.rglob("*"):
        if path.is_file():archive.write(path,path.relative_to(APP_RUN).as_posix())
files.download(str(download))
if exit_code:raise RuntimeError("Application verification failed. Diagnostics were preserved and downloaded above; inspect the printed error.")
'''
    (ROOT/'colab_check_application.py').write_text(code,encoding='utf-8')
    restore=code[:code.index('APP_RUN=')]
    browser=restore+'''# Keep this cell's process running while recording the browser demonstration.
import time,urllib.request
from google.colab.output import eval_js
from IPython.display import HTML,display
APP_URL=eval_js("google.colab.kernel.proxyPort(8765)")
if globals().get("APP_PROCESS") is not None and APP_PROCESS.poll() is None:
    raise RuntimeError("The application is already running. Use the browser link printed previously.")
APP_LOG=(APP_PROJECT/"application.log").open("w")
APP_PROCESS=subprocess.Popen([sys.executable,"-u","app.py","--no-browser","--device","cuda",
    "--port","8765","--browser-origin",APP_URL,"--training-output",str(APP_TRAINING)],
    cwd=APP_PROJECT,stdout=APP_LOG,stderr=subprocess.STDOUT)
for attempt in range(60):
    if APP_PROCESS.poll() is not None:
        raise RuntimeError((APP_PROJECT/"application.log").read_text())
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/api/status",timeout=2)as response:
            if response.status==200:break
    except Exception:time.sleep(1)
else:raise RuntimeError("Server has not started. Read /content/trialscreen_application/application.log.")
display(HTML('<p><a href="'+APP_URL+'" target="_blank" rel="noopener">Open TrialScreen on the Colab GPU</a></p>'))
print("Open that link in a NEW TAB, click Load model, then record the app. No training occurs.")
print("Keep this runtime connected. This temporary authenticated proxy is for your recording, not a permanent public deployment.")
'''
    (ROOT/'colab_start_application.py').write_text(browser,encoding='utf-8')
    print('Created one-cell full application verification helper.')


if __name__=='__main__':build()
