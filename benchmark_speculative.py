"""Manual greedy speculative decoding: preflight first, then paired measurements."""

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import GenerationConfig

from baseline import save_json
from benchmark_kv_cache import attention_backend, save_trials
from decoding import first_divergence, greedy_decode
from prepare_model_inputs import sha256
from speculative_decoding import speculative_decode
from speculative_inputs import load_pair, prepare
from speculative_report import summarize_trials, write_reports


ROOT = Path(__file__).resolve().parent


def preflight(target, draft, inputs, case, draft_lengths):
    ordinary = greedy_decode(target, inputs, max_new_tokens=case['output_budget'],
                             use_cache=True, eos_token_id=case['eos_token_id'])
    generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        max_new_tokens=case['output_budget'], eos_token_id=[] if case['eos_token_id'] is None else case['eos_token_id'],
        pad_token_id=target.config.pad_token_id, temperature=1.0, top_p=1.0, top_k=50)
    with torch.inference_mode():
        hf = target.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=generation)
    hf_ids = hf[0, inputs.shape[1]:].tolist()
    checks = {}
    for k in draft_lengths:
        result = speculative_decode(target, draft, inputs, max_new_tokens=case['output_budget'],
            draft_length=k, eos_token_id=case['eos_token_id'], trace=True)
        difference = first_divergence(ordinary.token_ids, result.token_ids)
        checks[str(k)] = {'result': asdict(result), 'first_divergence': difference, 'passed': difference is None}
    hf_difference = first_divergence(ordinary.token_ids, hf_ids)
    return {'case_id': case['case_id'], 'greedy_token_ids': ordinary.token_ids, 'hf_token_ids': hf_ids,
            'greedy_vs_hf_first_divergence': hf_difference, 'speculative': checks,
            'passed': hf_difference is None and all(c['passed'] for c in checks.values())}


def experiment(target, draft, cases, settings, output, signature, smoke=False):
    device = next(target.parameters()).device
    check_path = output / 'preflight.json'
    checks = json.loads(check_path.read_text(encoding='utf-8')) if check_path.exists() else []
    allowed = {case['case_id'] for case in cases}
    if len({check['case_id'] for check in checks}) != len(checks) or any(check['case_id'] not in allowed for check in checks):
        raise ValueError('Saved preflight contains unexpected or duplicate cases.')
    if any(not check['passed'] for check in checks):
        raise RuntimeError('This folder contains a failed preflight. Its evidence is preserved; use a new output folder.')
    for case in cases:
        if any(check['case_id'] == case['case_id'] for check in checks):
            continue
        inputs = torch.tensor([case['token_ids']], dtype=torch.long, device=device)
        check = preflight(target, draft, inputs, case, settings['draft_lengths'])
        checks.append(check); save_json(check_path, checks)
        print('Preflight', case['case_id'], check['passed'], flush=True)
    # Restore the deterministic case order after an interrupted partial preflight.
    by_id = {check['case_id']: check for check in checks}
    checks = [by_id[case['case_id']] for case in cases]
    if not all(c['passed'] for c in checks):
        summary = summarize_trials(cases, [], settings['draft_lengths'], settings['repetitions'])
        write_reports(output, summary, checks, smoke=smoke)
        raise RuntimeError('Token mismatch in preflight. Saved diagnostics; no timed trials were run.')
    path = output / 'trials.jsonl'
    trials = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if any(t['run_signature'] != signature for t in trials):
        raise ValueError('Saved trials belong to another experiment.')
    summarize_trials(cases, trials, settings['draft_lengths'], settings['repetitions'])
    references = {check['case_id']: check['greedy_token_ids'] for check in checks}
    for trial in trials:
        difference = first_divergence(trial['result']['token_ids'], references[trial['case_id']])
        if difference != trial['first_divergence'] or trial['matches_greedy'] != (difference is None):
            raise ValueError('Saved trial no longer matches the current greedy reference. Use a new output folder.')
    done = {(t['case_id'], t['mode'], t['repeat']) for t in trials}
    modes = ['greedy', *[f'speculative_{k}' for k in settings['draft_lengths']]]
    for case, check in zip(cases, checks):
        inputs = torch.tensor([case['token_ids']], dtype=torch.long, device=device)

        def generate(mode, measure):
            if mode == 'greedy':
                value = asdict(greedy_decode(target, inputs, max_new_tokens=case['output_budget'],
                    use_cache=True, eos_token_id=case['eos_token_id'], measure=measure))
                value.update(target_forward_calls=len(value['token_ids']), draft_forward_calls=0)
                return value
            return asdict(speculative_decode(target, draft, inputs, max_new_tokens=case['output_budget'],
                draft_length=int(mode.split('_')[1]), eos_token_id=case['eos_token_id'], measure=measure))

        if all((case['case_id'], mode, repeat) in done for mode in modes for repeat in range(settings['repetitions'])):
            continue
        for mode in modes:
            for _ in range(settings['warmups']):
                generate(mode, False)
        for repeat in range(settings['repetitions']):
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in order:
                if (case['case_id'], mode, repeat) in done:
                    continue
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device); torch.cuda.empty_cache()
                result = generate(mode, True)
                difference = first_divergence(result['token_ids'], check['greedy_token_ids'])
                trials.append({'case_id': case['case_id'], 'mode': mode, 'repeat': repeat,
                    'run_signature': signature, 'matches_greedy': difference is None,
                    'first_divergence': difference, 'result': result})
                save_trials(path, trials)
                print(case['case_id'], mode, repeat + 1, f"{result['total_seconds']:.3f}s", 'equal:', difference is None, flush=True)
    summary = summarize_trials(cases, trials, settings['draft_lengths'], settings['repetitions'])
    payload = write_reports(output, summary, checks, smoke=smoke)
    if not payload['passed']:
        raise RuntimeError('Recorded speculative trials contain a mismatch or incomplete workload.')
    print('Complete:', payload['complete'], 'passed:', payload['passed'], 'results:', output, flush=True)


def run(args):
    torch.manual_seed(42); torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    if args.smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        def tiny(seed):
            torch.manual_seed(seed)
            config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                max_position_embeddings=128, pad_token_id=0, eos_token_id=63)
            config._attn_implementation = 'sdpa'
            return Qwen3ForCausalLM(config).double().eval()
        settings = {'draft_lengths': [2, 4], 'repetitions': 2, 'warmups': 1, 'seed': 42}
        cases = [{'case_id': f'tiny-{p}-{n}', 'kind': 'fixed_token_stress', 'output_budget': n,
                  'eos_token_id': None, 'token_ids': list(range(1, p + 1))}
                 for p, n in ((3, 1), (7, 9), (17, 15))]
        identity = {'settings': settings, 'cases': cases, 'experiment': 'miniature_cpu_smoke_fp64'}
        loader = lambda: (tiny(31), tiny(77))
    else:
        prepared = prepare(args.root, args.training_output, offline=args.offline)
        if args.prepare_only:
            (args.root / 'reports').mkdir(parents=True, exist_ok=True)
            save_json(args.root / 'reports/speculative_preparation.json', prepared)
            print('Prepared', len(prepared['cases']), 'cases; compatible tokenizer tokens:',
                  prepared['tokenizer_compatibility']['token_count'], 'No model weights loaded.', flush=True)
            return
        if not torch.cuda.is_available():
            raise RuntimeError('Use a Colab GPU for the real pair, or --smoke for a miniature CPU experiment.')
        settings, cases = prepared['settings'], prepared['cases']
        identity = {key: prepared[key] for key in ('model', 'settings', 'validation_sha256',
                    'tokenizer_compatibility', 'draft_tokenizer_lock', 'target_tokenizer_lock')}
        identity.update(training_signature=prepared['training']['signature'],
            adapter_sha256=prepared['training']['adapter_sha256'], gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
        loader = lambda: load_pair(prepared, args.training_output)
    identity['versions'] = {name: importlib.metadata.version(name) for name in
        (('torch', 'transformers') if args.smoke else ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'tokenizers'))}
    identity['source_sha256'] = {name: sha256(args.root / name) for name in
        ('speculative_decoding.py', 'speculative_inputs.py', 'speculative_report.py', 'benchmark_speculative.py',
         'decoding.py', 'benchmark_kv_cache.py', 'baseline.py', 'prepare_model_inputs.py')}
    identity['float32_matmul_precision'] = torch.get_float32_matmul_precision()
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.output or args.root / 'results' / ('speculative_cpu_smoke' if args.smoke else 'speculative_v1')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))['run_signature'] != signature:
            raise ValueError('Output has another configuration, source or runtime. Choose a new --output directory.')
    elif any((output / name).exists() for name in ('preflight.json', 'trials.jsonl')):
        raise ValueError('Saved work exists without a matching manifest.')
    else:
        save_json(manifest_path, {'run_signature': signature, 'identity': identity})
    save_json(output / 'cases.json', cases)
    with attention_backend('math'):
        target, draft = loader()
        experiment(target, draft, cases, settings, output, signature, smoke=args.smoke)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--training-output', type=Path, default=ROOT / 'results/finetuning_qlora_v1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
