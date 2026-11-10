"""Exercise the actual full-model application HTTP workflow on a GPU host."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading
import time
from urllib.request import Request,urlopen

from app import create_server
from app_model import ROOT


def save_verification(output,checks,identity):
    passed=bool(checks)and all(checks.values())
    (output/'verification.json').write_text(json.dumps({'checks':checks,'passed':passed,'identity':identity},indent=2),encoding='utf-8')
    if not passed:
        raise RuntimeError('Full-model application verification failed. The checks and supporting artifacts are preserved.')


def run(training,output):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('This full-model check requires the existing Colab GPU.')
    output.mkdir(parents=True,exist_ok=True)
    server=create_server(port=0,training=training,device='cuda')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    record={'patient_reference':'Unseen synthetic application check','trial_name':'Application integration check',
        'note':'0. The patient is 52 years old.\n1. The patient has type 2 diabetes treated with metformin.\n2. The record states no history of heart failure.\n3. No recent HbA1c measurement is provided.',
        'criteria':[{'type':'inclusion','text':'Age at least 18 years.'},
                    {'type':'inclusion','text':'HbA1c below 8% within three months.'},
                    {'type':'exclusion','text':'History of heart failure.'}]}
    def get(path):
        with urlopen(url+path,timeout=120)as response:return json.load(response)
    def job(path,value):
        with urlopen(Request(url+path,data=json.dumps(value).encode(),headers={'Content-Type':'application/json'}),timeout=120)as response:job_id=json.load(response)['job_id']
        previous=None
        while True:
            result=get('/api/jobs/'+job_id)
            if result['message']!=previous:print(result['message'],flush=True);previous=result['message']
            if result['state']=='failed':raise RuntimeError(result['message'])
            if result['state']=='complete':return job_id,result['result']
            time.sleep(1)
    try:
        screen_id,report=job('/api/screen',record)
        (output/'screening_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        with urlopen(url+'/api/jobs/'+screen_id+'/export')as response:html=response.read()
        (output/'screening_report.html').write_bytes(html)
        _,benchmark=job('/api/benchmark',{**record,'modes':['cached','uncached']})
        (output/'live_benchmark.json').write_text(json.dumps(benchmark,indent=2),encoding='utf-8')
        status=get('/api/status')
        checks={'http_screening_complete':len(report['results'])==3,
            'html_export_contains_notice':b'not a clinical enrollment decision'in html,
            'all_outputs_valid':all(row['prediction']['output_valid']for row in report['results']),
            'live_benchmark_equal':all(row['tokens_equal']for row in benchmark['summaries']),
            'loaded_full_model':status['loaded']and status['provenance']['device']=='cuda:0'}
        identity={'at':datetime.now(timezone.utc).isoformat(),'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
            'versions':{name:importlib.metadata.version(name)for name in ('torch','transformers','peft','bitsandbytes')},
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in ('app.py','app_model.py','screening.py','decoding.py','prepare_model_inputs.py','baseline.py')},
            'provenance':status['provenance']}
        print('Application checks:',checks,flush=True)
        save_verification(output,checks,identity)
    finally:
        server.shutdown();server.server_close();server.jobs.executor.shutdown(wait=True);thread.join()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-output',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.training_output,args.output)
