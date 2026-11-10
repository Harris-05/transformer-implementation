"""Create reviewable source/results and complete local submission archives."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent


def digest(path):
    result=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def build():
    destination=ROOT/'submission';destination.mkdir(exist_ok=True)
    files={p for suffix in ('*.py','*.md','*.cmd','*.txt','*.ipynb','*.parquet')for p in ROOT.glob(suffix)if p.is_file()}
    files.add(ROOT/'.gitignore')
    extensions={'.py','.md','.txt','.json','.jsonl','.csv','.png','.jpg','.tex','.pdf','.ttf','.html','.css','.js','.parquet'}
    for folder in ('configs','data','reports','report','web','docs','results'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file()and path.suffix.lower()in extensions and not {'adapter','checkpoints','__pycache__'}.intersection(path.relative_to(ROOT).parts):
                files.add(path)
    checks=ROOT/'artifacts/app_browser_final/workflow_checks.json'
    if checks.exists():files.add(checks)
    manifest={'author':'Muhammad Harris Azhar','student_id':'i232535',
        'contents':{p.relative_to(ROOT).as_posix():digest(p)for p in sorted(files)},
        'held_out_test_complete':True,'held_out_test_records_per_model':206,
        'test_accuracy_after':0.7427184466019418,
        'source_archive_excludes_model_weights':True,
        'repository_url':'https://github.com/Harris-05/transformer-implementation',
        'demo_url':'https://youtu.be/-zL3ECuJL1A',
        'remaining_user_actions':['Sync repository files with final source archive','Submit report, source/video links and actual included adapter','Confirm full-size application-host run']}
    source=destination/'i232535_Assignment2_Source.zip'
    with zipfile.ZipFile(source,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as archive:
        for path in sorted(files):archive.write(path,path.relative_to(ROOT).as_posix())
        archive.writestr('SUBMISSION_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    weights=ROOT/'finetuning_results.zip'
    bundle=destination/'i232535_Assignment2_Submission.zip'
    components={source.name:digest(source),'TrialScreen_Report.pdf':digest(ROOT/'report/TrialScreen_Report.pdf'),
                weights.name:digest(weights)}
    instructions='''Muhammad Harris Azhar — i232535 — Generative AI Assignment 2

Open TrialScreen_Report.pdf for the IEEE report.
Extract i232535_Assignment2_Source.zip for source, LaTeX, results, split and startup instructions.
finetuning_results.zip contains the actual selected adapter and its training/validation artifacts.
The base/draft weights are downloaded by pinned model IDs; they are not duplicated here.

Read README.md, APPLICATION.md and SUBMISSION_CHECKLIST.md in the source archive.
Repository: https://github.com/Harris-05/transformer-implementation
Demo video: https://youtu.be/-zL3ECuJL1A
Sync the repository with the final source archive and submit these links with the
report and included adapter. Do not upload this weights ZIP as
an ordinary GitHub file. The report includes verified final held-out results.
'''
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_STORED)as archive:
        archive.write(source,source.name);archive.write(ROOT/'report/TrialScreen_Report.pdf','TrialScreen_Report.pdf')
        archive.write(weights,weights.name)
        archive.writestr('READ_ME_FIRST.txt',instructions)
        archive.writestr('SHA256.json',json.dumps(components,indent=2)+'\n')
    (destination/'SHA256.json').write_text(json.dumps({source.name:digest(source),bundle.name:digest(bundle)},indent=2)+'\n')
    with zipfile.ZipFile(source)as archive:
        assert archive.testzip()is None
        for name,expected in manifest['contents'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==expected,name
    with zipfile.ZipFile(bundle)as archive:
        assert archive.testzip()is None
    print('Source/results archive:',source,'bytes:',source.stat().st_size,'files:',len(files))
    print('Submission archive including selected adapter:',bundle,'bytes:',bundle.stat().st_size)
    print('Verified archive CRCs and every source manifest checksum.')


if __name__=='__main__':build()
