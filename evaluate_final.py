"""One final paired evaluation on the frozen held-out patients; no tuning here."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from baseline import parse_prediction, save_json
from eda import INPUT_COLUMNS, METADATA_COLUMNS, TARGET_COLUMNS, parse_evidence, parse_note
from evaluate_adapter import metric_groups, compare_groups
from prepare_model_inputs import build_messages, load_tokenizer, sha256, validate_response


ROOT = Path(__file__).resolve().parent


def build_test_records(frame, allowed, tokenizer, config):
    """Called only by the final evaluation command; fail on any invalid row."""
    counts = frame['patient_id'].value_counts().to_dict()
    if counts != allowed or frame['annotation_id'].duplicated().any():
        raise ValueError('Test patient identities, counts or annotation uniqueness differ.')
    rows = []
    for record in frame.to_dict(orient='records'):
        inputs = {key: record[key] for key in INPUT_COLUMNS}
        prompt = build_messages(inputs)
        gold = {'label': record['expert_eligibility'], 'sentence_ids': parse_evidence(record['expert_sentences'])}
        validate_response(gold, inputs['criterion_type'], inputs['note'])
        tokens = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(tokens) + config['max_new_tokens'] > config['max_sequence_length']:
            raise ValueError(f'Test annotation {record["annotation_id"]} exceeds the frozen context budget; no truncation is permitted.')
        rows.append({'annotation_id': int(record['annotation_id']),
            'metadata': {key: int(record[key]) if key == 'annotation_id' else record[key] for key in METADATA_COLUMNS},
            **inputs, 'prompt': prompt, 'gold': gold, 'prompt_tokens': len(tokens)})
    return rows


def validate_saved_predictions(saved, records, signature):
    by_id = {row['annotation_id']: row for row in records}
    seen = set()
    for row in saved:
        key = row['annotation_id']
        if key not in by_id or key in seen or row['run_signature'] != signature:
            raise ValueError('Saved predictions have duplicate/unexpected IDs or a changed signature.')
        if any(row.get(name) != value for name, value in by_id[key].items()):
            raise ValueError('Saved prediction input, prompt, metadata or expert target differs.')
        seen.add(key)


def write_final_reports(output, before, after, expected):
    groups = {name: metric_groups(rows) for name, rows in (('before', before), ('after', after)) if rows}
    complete = len(before) == len(after) == expected
    metrics = {'split': 'held_out_test', 'complete': complete, 'expected_records_per_model': expected,
               'evaluated_records': {'before': len(before), 'after': len(after)}, 'groups': groups}
    if len(groups) == 2:
        metrics['comparison'] = compare_groups(groups['before'], groups['after'])
    save_json(output / 'metrics.json', metrics)
    lines = ['# Final held-out test evaluation', '', f'Complete: **{complete}**. Expected {expected} records per model.', '',
             '| Group | Model | Accuracy | Macro F1 | Evidence micro P/R/F1 | Evidence macro F1 | Valid |',
             '|---|---|---:|---:|---|---:|---:|']
    for mode, mode_groups in groups.items():
        for name, group in mode_groups.items():
            cls, evidence = group['classification'], group['evidence']
            micro = evidence['micro']
            lines.append(f'| {name} | {mode} | {cls["accuracy"]:.4f} | {cls["macro_f1"]:.4f} | {micro["precision"]:.4f}/{micro["recall"]:.4f}/{micro["f1"]:.4f} | {evidence["macro"]["f1"]:.4f} | {group["output_valid_rate"]:.4f} |')
    lines += ['', 'Same frozen prompt, original expert categories, 4-bit NF4/double quantization, FP16 compute, automatic SDPA and greedy decoding for both phases.',
              'The original phase disables the selected LoRA adapter without changing the frozen base weights. The after phase enables the selected adapter.',
              'Invalid responses stay in every denominator. No response repair or truncation. Both evidence sets empty scores one; one empty scores zero. Nonempty-gold-only macro scores remain in metrics.json.',
              'No tuning or checkpoint selection uses these final test scores.', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    if not groups:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay
    for mode, mode_groups in groups.items():
        for name, group in mode_groups.items():
            cls = group['classification']
            fig, ax = plt.subplots(figsize=(9, 7))
            ConfusionMatrixDisplay(np.asarray(cls['confusion_matrix']), display_labels=cls['confusion_labels']).plot(ax=ax, xticks_rotation=45, colorbar=False)
            ax.set_title(f'Held-out test: {mode}, {name}')
            fig.tight_layout(); fig.savefig(output / f'confusion_{mode}_{name}.png', dpi=160); plt.close(fig)


def run(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run final evaluation on a Colab GPU. This command does not train a model.')
    import pandas as pd
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
    config = json.loads((args.root / 'configs/model_input_config.json').read_text())
    split = json.loads((args.root / 'data/splits/split_manifest.json').read_text())
    training = json.loads((args.training_output / 'training_summary.json').read_text())
    manifest = json.loads((args.training_output / 'training_manifest.json').read_text())
    dataset = args.dataset or args.root / split['source_file']
    if sha256(dataset) != split['source_sha256'] or not training['complete'] or training['signature'] != manifest['signature'] or manifest['identity']['model'] != config:
        raise ValueError('Dataset, selected training run or frozen model/prompt differs.')
    for name, digest in training['adapter_sha256'].items():
        if Path(name).name != name or sha256(args.training_output / 'adapter' / name) != digest:
            raise ValueError('Selected adapter checksum differs.')
    tokenizer, _ = load_tokenizer(args.root)
    # Read only permitted inputs/targets/metadata, never prior GPT predictions.
    frame = pd.read_parquet(dataset, columns=INPUT_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS)
    allowed = {p['patient_id']: p['annotation_count'] for p in split['splits']['test']}
    records = build_test_records(frame.loc[frame.patient_id.isin(allowed)], allowed, tokenizer, config)
    versions = {name: importlib.metadata.version(name) for name in ('torch','transformers','peft','bitsandbytes','accelerate','tokenizers')}
    identity = {'model': config, 'training_signature': training['signature'], 'adapter_sha256': training['adapter_sha256'],
        'dataset_sha256': split['source_sha256'], 'test_patient_counts': allowed,
        'records_sha256': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        'source_sha256': {name: sha256(args.root / name) for name in ('evaluate_final.py','baseline.py','evaluate_adapter.py','prepare_model_inputs.py','eda.py')},
        'versions': versions, 'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'protocol': 'NF4_double_quant_FP16_compute_automatic_SDPA_greedy_paired_adapter_disabled_enabled'}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['run_signature'] != signature:
            raise ValueError('Output belongs to another final experiment; use a new directory.')
    else:
        if any((output / f'predictions_{name}.jsonl').exists() for name in ('before','after')):
            raise ValueError('Predictions exist without a signed manifest.')
        save_json(manifest_path, {'run_signature': signature, 'identity': identity, 'started_at': datetime.now(timezone.utc).isoformat()})
    saved = {}
    for name in ('before','after'):
        path = output / f'predictions_{name}.jsonl'
        saved[name] = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        validate_saved_predictions(saved[name], records, signature)
    if all(len(value) == len(records) for value in saved.values()):
        write_final_reports(output, saved['before'], saved['after'], len(records)); return
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    quantization = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(config['model_id'],revision=config['revision'],quantization_config=quantization,
        device_map={'':0},dtype=torch.float16,attn_implementation='sdpa',trust_remote_code=False)
    model = PeftModel.from_pretrained(base,args.training_output/'adapter',is_trainable=False).eval()
    generation = GenerationConfig(do_sample=False,num_beams=1,use_cache=True,max_new_tokens=config['max_new_tokens'],
        eos_token_id=config['eos_token_id'],pad_token_id=config['pad_token_id'],temperature=1.0,top_p=1.0,top_k=50)
    for name in ('before','after'):
        done = {row['annotation_id'] for row in saved[name]}
        pending = [row for row in records if row['annotation_id'] not in done]
        if not pending:continue
        with model.disable_adapter() if name=='before' else nullcontext():
            def encode(row):
                return tokenizer.apply_chat_template(row['prompt'], tokenize=True,add_generation_prompt=True,
                    return_tensors='pt',return_dict=True,truncation=False).to('cuda:0')
            with torch.inference_mode():model.generate(**encode(pending[0]),generation_config=generation)
            torch.cuda.synchronize()
            for row in pending:
                inputs=encode(row);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                with torch.inference_mode():values=model.generate(**inputs,generation_config=generation)
                torch.cuda.synchronize();seconds=time.perf_counter()-started
                ids=values[0,inputs['input_ids'].shape[-1]:].tolist();raw=tokenizer.decode(ids,skip_special_tokens=True)
                prediction=parse_prediction(raw,row['criterion_type'],row['note']);sentences=parse_note(row['note'])
                result={**row,'run_signature':signature,'raw_response':raw,'prediction':prediction,
                    'predicted_evidence_text':{str(i):sentences[i] for i in prediction['sentence_ids'] if i in sentences},
                    'output_tokens':len(ids),'generation_seconds':seconds,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
                with (output/f'predictions_{name}.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                saved[name].append(result)
                print(f'{name} [{len(saved[name])}/{len(records)}] annotation {row["annotation_id"]}',flush=True)
    write_final_reports(output,saved['before'],saved['after'],len(records))
    print((output/'summary.md').read_text(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--training-output',type=Path,default=ROOT/'results/finetuning_qlora_v1')
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'results/final_test')
    run(parser.parse_args())
