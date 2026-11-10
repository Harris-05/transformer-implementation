# Run in your CURRENT Colab runtime if adapter files exist only under /content.
# Copies the existing files to Drive once; no training or model loading occurs.

import hashlib
import json
from pathlib import Path
import shutil
from google.colab import drive

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')

candidates = [Path('/content'), Path('/content/restored_finetuning')]
if globals().get('TRAINING') is not None:
    candidates.insert(0, Path(TRAINING))
SOURCE = next((p for p in candidates if (p / 'training_summary.json').is_file()
               and (p / 'training_manifest.json').is_file()), None)
if SOURCE is None:
    raise FileNotFoundError('No uploaded training files found in this runtime.')
summary = json.loads((SOURCE / 'training_summary.json').read_text(encoding='utf-8'))
manifest = json.loads((SOURCE / 'training_manifest.json').read_text(encoding='utf-8'))
if not summary['complete'] or summary['signature'] != manifest['signature']:
    raise ValueError('Training metadata is incomplete or mismatched.')

DESTINATION = Path('/content/drive/MyDrive/TrialGPT_Assignment2/selected_adapter')
copies = [(SOURCE / name, DESTINATION / name, None)
          for name in ('training_summary.json', 'training_manifest.json')]
for name, digest in summary['adapter_sha256'].items():
    if Path(name).name != name:
        raise ValueError('Expected a plain adapter filename.')
    source = SOURCE / 'adapter' / name
    if not source.is_file():
        source = SOURCE / name
    copies.append((source, DESTINATION / 'adapter' / name, digest))

for source, destination, expected in copies:
    if not source.is_file():
        raise FileNotFoundError(f'Missing source file: {source}')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'Source checksum mismatch: {source.name}')
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Drive backup contains a different file: {destination}. Use a new DESTINATION folder.')

for source, destination, _ in copies:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)

print('Adapter preserved on Drive:', DESTINATION)
print('The speculative notebook will find this folder automatically.')
