# Copy this entire file into ONE Colab cell after the failed speculative step 3.
# Restores only line endings that reproduce the exact training-manifest checksum.

import hashlib
import json
from pathlib import Path
import sys


def repair_validation_file(path, expected_sha256):
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() == expected_sha256:
        return False
    unix = original.replace(b'\r\n', b'\n')
    for candidate in (unix, unix.replace(b'\n', b'\r\n')):
        if hashlib.sha256(candidate).hexdigest() == expected_sha256:
            path.write_bytes(candidate)
            return True
    raise ValueError(
        'Validation content differs beyond line endings. No file was changed; '
        'check that the adapter and validation bundle belong to the same run.'
    )


if __name__ == '__main__':
    if globals().get('PROJECT') is None or globals().get('TRAINING') is None:
        raise RuntimeError('Run notebook steps 1-3 first so PROJECT and TRAINING are selected.')
    manifest = json.loads((Path(TRAINING) / 'training_manifest.json').read_text(encoding='utf-8'))
    expected = manifest['identity']['data_sha256']['validation']
    changed = repair_validation_file(Path(PROJECT) / 'data/model_ready/validation.jsonl', expected)
    print('Restored original line endings.' if changed else 'Validation bytes already match.')
    print('Exact training validation SHA256:', expected)
    run_checked([
        sys.executable, '-u', 'benchmark_speculative.py',
        '--training-output', str(TRAINING), '--prepare-only',
    ])
    print('Preparation passed. Continue with notebook step 4.')
