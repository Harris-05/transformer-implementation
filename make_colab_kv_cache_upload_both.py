"""Build a fresh-runtime notebook that uploads code and fine-tuning ZIPs."""

import json
from pathlib import Path

from make_colab_baseline import cell
from make_colab_kv_cache import build as build_code_bundle


ROOT = Path(__file__).resolve().parent


def build(root=ROOT):
    build_code_bundle(root)
    original = json.loads((root / 'colab_kv_cache.ipynb').read_text(encoding='utf-8'))
    combined = (root / 'colab_upload_and_run_kv_cache.py').read_text(encoding='utf-8')
    contents = [
        ('markdown', '''
            # KV-cache benchmark: upload both ZIP bundles

            This notebook works in a fresh Colab GPU runtime. Keep these two files
            on your computer:

            1. `colab_kv_cache_bundle.zip` — decoding code, dependencies and development prompts.
            2. `finetuning_results.zip` — your completed training results and selected adapter.

            Select **Runtime > Change runtime type > GPU**, then run the four steps
            below in order. Step 1 asks for the code ZIP. Step 3 asks for the fine-tuning
            ZIP, extracts the adapter and starts the benchmark. No files from an earlier
            Colab session or pre-existing training folder on Drive are required.

            Drive is used to save new benchmark outputs. The experiment reuses your
            trained adapter and compares cached/uncached greedy generation with an HF
            reference: nine stress cases, 54 timed trials and three full clinical prompts.
            Model weights may download into this runtime. Held-out test examples are excluded.
        '''),
        ('markdown', '''
            ## 1. Upload the KV-cache code bundle

            Select **only `colab_kv_cache_bundle.zip`** in this upload dialog.
            This cell verifies its contents and extracts the project automatically.
        '''),
        ('code', original['cells'][2]['source']),
        ('markdown', '''
            ## 2. Install dependencies and check the decoder

            Let installation and all decoder tests finish before continuing.
            These miniature CPU checks do not measure full-model GPU performance.
        '''),
        ('code', original['cells'][4]['source']),
        ('markdown', '''
            ## 3. Upload the fine-tuning results and run the benchmark

            Select **only `finetuning_results.zip`** in this upload dialog.
            Authorize Drive access if requested. The cell extracts the selected adapter,
            verifies preparation and runs the benchmark with that adapter.

            New results are saved under `MyDrive/TrialGPT_Assignment2/kv_cache_v1`.
            Let the cell finish. If it reports a mismatch or another failure, the last
            cell can still download the available diagnostics. An existing run with
            different settings requires a different `RUN` folder.
        '''),
        ('code', combined),
        ('markdown', '''
            ## 4. Download the results

            Run this cell to inspect the summary/plot and download `kv_cache_results.zip`.
            Return that ZIP for analysis, including if the benchmark reported a mismatch.
        '''),
        ('code', original['cells'][-1]['source']),
    ]
    notebook = {**original, 'cells': [cell(kind, source, index)
                for index, (kind, source) in enumerate(contents)]}
    for index, value in enumerate(notebook['cells']):
        value['id'] = f'kv-cache-upload-both-{index}'
    notebook['metadata']['colab']['name'] = 'colab_kv_cache_upload_both.ipynb'
    destination = root / 'colab_kv_cache_upload_both.ipynb'
    destination.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    print('Created colab_kv_cache_upload_both.ipynb: upload code ZIP, then fine-tuning results ZIP.')


if __name__ == '__main__':
    build()
