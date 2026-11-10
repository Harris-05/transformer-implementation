"""Task 3: token correctness and repeated KV-cache benchmarks on development data."""

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from transformers import GenerationConfig

from baseline import load_validation, save_json
from decoding import diagnose_divergence, first_divergence, greedy_decode
from kv_benchmark_report import summarize_trials, write_report
from prepare_model_inputs import load_tokenizer, sha256


ROOT = Path(__file__).resolve().parent


def controlled_cases(seed_tokens, input_lengths, output_lengths):
    if not seed_tokens or any(type(n) is not int or n < 1 for n in [*input_lengths, *output_lengths]):
        raise ValueError('A nonempty seed and positive integer lengths are required.')
    return [{'case_id': f'p{p}-o{n}', 'kind': 'fixed_token_stress',
             'input_tokens': p, 'output_budget': n, 'eos_token_id': None,
             'token_ids': (seed_tokens * ((p + len(seed_tokens) - 1) // len(seed_tokens)))[:p]}
            for p in input_lengths for n in output_lengths]


def compare_paths(model, inputs, *, budget, eos_token_id):
    cached = greedy_decode(model, inputs, max_new_tokens=budget, use_cache=True,
                           eos_token_id=eos_token_id, trace=True)
    uncached = greedy_decode(model, inputs, max_new_tokens=budget, use_cache=False,
                             eos_token_id=eos_token_id, trace=True)
    generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        # In 5.15.1, None inherits the model's EOS; an empty list disables it.
        max_new_tokens=budget, eos_token_id=[] if eos_token_id is None else eos_token_id,
        temperature=1.0, top_p=1.0, top_k=50,
        pad_token_id=model.config.pad_token_id)
    with torch.inference_mode():
        hf = model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=generation)
    tokens = hf[0, inputs.shape[1]:].tolist()
    differences = {'cached_vs_uncached': first_divergence(cached.token_ids, uncached.token_ids),
        'cached_vs_hf': first_divergence(cached.token_ids, tokens),
        'uncached_vs_hf': first_divergence(uncached.token_ids, tokens)}
    # Includes identical-prefix diagnostics for either manual-path or HF mismatch.
    indices = [value for value in differences.values() if value is not None]
    diagnostic = diagnose_divergence(model, inputs, uncached.token_ids[:min(indices)]) if indices else None
    return {'passed': not indices, 'first_divergence': differences,
            'cached': asdict(cached), 'uncached': asdict(uncached),
            'hf_token_ids': tokens, 'logit_diagnostic': diagnostic}


def prepare_cases(root, training_output, settings):
    config, records, validation_hash = load_validation(root)
    training = json.loads((training_output / 'training_summary.json').read_text(encoding='utf-8'))
    training_manifest = json.loads((training_output / 'training_manifest.json').read_text(encoding='utf-8'))
    if not training['complete'] or training['signature'] != training_manifest['signature']:
        raise ValueError('A completed, matching training run is required.')
    if training_manifest['identity']['model'] != config or training_manifest['identity']['data_sha256']['validation'] != validation_hash:
        raise ValueError('Training model/prompt or validation data differs.')
    for name, digest in training['adapter_sha256'].items():
        if sha256(training_output / 'adapter' / name) != digest:
            raise ValueError(f'Adapter checksum mismatch: {name}')
    tokenizer, _ = load_tokenizer(root)
    ordered = sorted(records, key=lambda r: (r['token_counts']['prompt_tokens'], r['metadata']['annotation_id']))
    selected = [ordered[index] for index in (0, len(ordered) // 2, len(ordered) - 1)]
    clinical = []
    for row in selected:
        tokens = tokenizer.apply_chat_template(row['prompt'], tokenize=True,
                                               add_generation_prompt=True, return_dict=False)
        if len(tokens) != row['token_counts']['prompt_tokens'] or len(tokens) + config['max_new_tokens'] > config['max_sequence_length']:
            raise ValueError('Tokenization changed or exceeds frozen context budget.')
        clinical.append({'case_id': f"clinical-{row['metadata']['annotation_id']}",
            'kind': 'clinical_development', 'annotation_id': row['metadata']['annotation_id'],
            'patient_id': row['metadata']['patient_id'], 'input_tokens': len(tokens),
            'output_budget': config['max_new_tokens'], 'eos_token_id': config['eos_token_id'],
            'token_ids': tokens})
    cases = controlled_cases(clinical[-1]['token_ids'], settings['input_lengths'], settings['output_lengths'])
    if any(c['input_tokens'] + c['output_budget'] > config['max_sequence_length'] for c in cases):
        raise ValueError('Stress case exceeds the documented context budget.')
    return config, training, validation_hash, cases, clinical


def load_model(config, training_output, attention):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['revision'],
        quantization_config=quantization, device_map={'': 0}, dtype=torch.float16,
        attn_implementation=attention, trust_remote_code=False)
    return PeftModel.from_pretrained(base, training_output / 'adapter', is_trainable=False).eval()


def save_trials(path, trials):
    """Commit a complete snapshot; an interrupted write leaves prior trials intact."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for trial in trials:
            stream.write(json.dumps(trial, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def experiment(model, cases, clinical, settings, output, signature, *, smoke=False):
    device = next(model.parameters()).device
    path = output / 'trials.jsonl'
    trials = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if any(trial['run_signature'] != signature for trial in trials):
        raise ValueError('Saved trials belong to different settings.')
    summarize_trials(cases, trials, repeats=settings['repetitions'])
    done = {(t['case_id'], t['mode'], t['repeat']) for t in trials}
    correctness_path = output / 'correctness.json'
    correctness = json.loads(correctness_path.read_text(encoding='utf-8')) if correctness_path.exists() else []
    allowed = {c['case_id'] for c in cases + clinical}
    if len({r['case_id'] for r in correctness}) != len(correctness) or any(r['case_id'] not in allowed for r in correctness):
        raise ValueError('Unexpected or duplicate correctness results.')
    for case in cases + clinical:
        inputs = torch.tensor([case['token_ids']], device=device, dtype=torch.long)
        if not any(r['case_id'] == case['case_id'] for r in correctness):
            comparison = compare_paths(model, inputs, budget=case['output_budget'], eos_token_id=case['eos_token_id'])
            correctness.append({**{k: v for k, v in case.items() if k != 'token_ids'}, **comparison})
            save_json(correctness_path, correctness)
            print(f"Correctness {case['case_id']}: {comparison['passed']}", flush=True)
        if case['kind'] != 'fixed_token_stress':
            continue
        # Warm every case/path after correctness so timed trials do not include cold kernels.
        pending = [(mode, repeat) for repeat in range(settings['repetitions'])
                   for mode in ('cached', 'uncached') if (case['case_id'], mode, repeat) not in done]
        if not pending:
            continue
        for mode in ('cached', 'uncached'):
            for _ in range(settings['warmups_per_case_and_mode']):
                greedy_decode(model, inputs, max_new_tokens=case['output_budget'],
                              use_cache=mode == 'cached', eos_token_id=None)
        for repeat in range(settings['repetitions']):
            order = ('cached', 'uncached') if repeat % 2 == 0 else ('uncached', 'cached')
            for mode in order:
                if (case['case_id'], mode, repeat) in done:
                    continue
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                result = greedy_decode(model, inputs, max_new_tokens=case['output_budget'],
                    use_cache=mode == 'cached', eos_token_id=None, measure=True)
                reference = next(r for r in correctness if r['case_id'] == case['case_id'])
                mismatch = first_divergence(result.token_ids, reference['hf_token_ids'])
                trial = {'case_id': case['case_id'], 'mode': mode, 'repeat': repeat,
                         'run_signature': signature, 'result': asdict(result),
                         'matches_reference': mismatch is None, 'reference_first_divergence': mismatch}
                if mismatch is not None:
                    trial['logit_diagnostic'] = diagnose_divergence(model, inputs, result.token_ids[:mismatch])
                trials.append(trial)
                save_trials(path, trials)
                print(f"{case['case_id']} {mode} repetition {repeat + 1}: {result.total_seconds:.3f}s", flush=True)
        summary = summarize_trials(cases, trials, repeats=settings['repetitions'])
        write_report(output, summary, correctness, expected_ids=allowed, smoke=smoke)
    summary = summarize_trials(cases, trials, repeats=settings['repetitions'])
    payload = write_report(output, summary, correctness, expected_ids=allowed, smoke=smoke)
    save_json(output / 'cases.json', {'stress': cases, 'clinical': clinical})
    if len(correctness) != len(cases + clinical) or not payload['passed']:
        raise RuntimeError('Benchmark incomplete or token mismatch detected. Inspect saved metrics and correctness diagnostics.')
    print(f"Complete: {payload['complete']}; correctness passed: {payload['passed']}; results: {output}", flush=True)


def run(args):
    settings = json.loads((args.root / 'configs/kv_cache_config.json').read_text(encoding='utf-8'))
    torch.manual_seed(settings['seed'])
    if args.smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(2)
        settings.update(input_lengths=[8, 16, 32], output_lengths=[4, 8, 12], repetitions=2)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, pad_token_id=0, eos_token_id=63, bos_token_id=1)
        config._attn_implementation = settings['attention_implementation']
        cases = controlled_cases([1, 3, 5, 7, 9], settings['input_lengths'], settings['output_lengths'])
        clinical = []
        identity = {'model': config.to_dict(), 'settings': settings, 'experiment': 'miniature_cpu_smoke'}
        model_loader = lambda: Qwen3ForCausalLM(config).eval()
    else:
        config, training, validation_hash, cases, clinical = prepare_cases(args.root, args.training_output, settings)
        if args.prepare_only:
            print(f'Prepared {len(cases)} stress cases and {len(clinical)} full clinical prompts; no test records or model weights loaded.')
            return
        if not torch.cuda.is_available():
            raise RuntimeError('Use a Colab GPU for the full 4-bit adapter benchmark, or --smoke for a tiny CPU check.')
        torch.cuda.manual_seed_all(settings['seed'])
        identity = {'model': config, 'settings': settings, 'validation_sha256': validation_hash,
            'training_signature': training['signature'], 'adapter_sha256': training['adapter_sha256'],
            'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
            'gpu_total_memory': torch.cuda.get_device_properties(0).total_memory}
        model_loader = lambda: load_model(config, args.training_output, settings['attention_implementation'])
    identity['versions'] = {name: importlib.metadata.version(name) for name in
        (('torch', 'transformers') if args.smoke else ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'tokenizers'))}
    identity['source_sha256'] = {name: sha256(args.root / name) for name in
        ('decoding.py', 'benchmark_kv_cache.py', 'kv_benchmark_report.py', 'prepare_model_inputs.py', 'baseline.py')}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.output or args.root / 'results' / ('kv_cache_cpu_smoke' if args.smoke else 'kv_cache_v1')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'run_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['run_signature'] != signature:
            raise ValueError('Output contains a different run; choose a new --output directory.')
    elif (output / 'trials.jsonl').exists() or (output / 'correctness.json').exists():
        raise ValueError('Saved work exists without a matching manifest.')
    else:
        save_json(manifest_path, {'run_signature': signature, 'identity': identity})
    experiment(model_loader(), cases, clinical, settings, output, signature, smoke=args.smoke)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--training-output', type=Path, default=ROOT / 'results/finetuning_qlora_v1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
