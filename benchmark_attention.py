"""Task 4 controlled attention correctness, timing and GPU-memory experiment."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
import torch.nn.functional as F

from attention import reference_attention, tiled_attention
from attention_benchmark_report import METHODS, aggregate_measurements, score_workspace_bytes, write_reports


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def error_metrics(actual, expected, *, atol, rtol):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    if actual.shape != expected.shape:
        raise ValueError('Attention output shape differs from the oracle.')
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {'finite': False, 'passed': False, 'max_absolute_error': None,
                'mean_absolute_error': None, 'max_relative_error': None}
    difference = (actual - expected).abs()
    return {'finite': True, 'passed': bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
            'max_absolute_error': difference.max().item(), 'mean_absolute_error': difference.mean().item(),
            'max_relative_error': (difference / expected.abs().clamp_min(1e-12)).max().item()}


@torch.inference_mode()
def sdpa_attention(q, k, v, *, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=0.0)


def measure(function, device):
    gc.collect()
    gpu = device.type == 'cuda'
    if gpu:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    initial = torch.cuda.memory_allocated(device) if gpu else None
    started = time.perf_counter()
    output = function()
    if gpu:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    result = {'seconds': seconds, 'initial_allocated_bytes': initial,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if gpu else None,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if gpu else None}
    del output
    return result


def run_case(config, length, causal, device, seed):
    generator = torch.Generator().manual_seed(seed)
    dtype = getattr(torch, config['dtype'])
    shape = (config['batch'], config['heads'], length, config['head_dim'])
    # Generate once on CPU, then transfer; every implementation gets these tensors.
    host = tuple(torch.randn(shape, generator=generator, dtype=dtype) for _ in range(3))
    oracle = reference_attention(*(x.double() for x in host), causal=causal)
    q, k, v = (x.to(device) for x in host)
    functions = {
        'reference': lambda: reference_attention(q, k, v, causal=causal),
        'tiled': lambda: tiled_attention(q, k, v, causal=causal,
            block_q=config['block_q'], block_k=config['block_k']),
        'sdpa_auto': lambda: sdpa_attention(q, k, v, causal=causal),
    }
    numerical = {name: error_metrics(function(), oracle, atol=config['atol'], rtol=config['rtol'])
                 for name, function in functions.items()}
    for function in functions.values():
        for _ in range(config['warmups']):
            function()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    records = []
    for repeat in range(config['repetitions']):
        order = METHODS[repeat % len(METHODS):] + METHODS[:repeat % len(METHODS)]
        for name in order:
            records.append({'method': name, 'repeat': repeat, **measure(functions[name], device)})
    summary = aggregate_measurements(records, repetitions=config['repetitions'])
    passed = summary['complete'] and all(value['passed'] for value in numerical.values())
    if not passed:
        summary['reference_over_tiled_speedup'] = None
    return {'case_id': f"n{length}-{'causal' if causal else 'full'}", 'length': length,
            'causal': causal, 'seed': seed, 'shape': list(shape), 'passed': passed,
            'numerical': numerical, 'timing': summary, 'measurements': records,
            'score_workspace': score_workspace_bytes(batch=config['batch'], heads=config['heads'],
                queries=length, keys=length, block_q=config['block_q'], block_k=config['block_k'],
                element_bytes=8 if dtype == torch.float64 else 4)}


def validate_config(config):
    for key in ('batch', 'heads', 'head_dim', 'block_q', 'block_k', 'repetitions', 'cpu_threads'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    lengths = config['lengths']
    if not isinstance(lengths, list) or not lengths or any(type(n) is not int or n < 1 for n in lengths) or len(set(lengths)) != len(lengths):
        raise ValueError('lengths must be a nonempty list of distinct positive integers.')
    modes = config['causal_modes']
    if not isinstance(modes, list) or not modes or any(type(m) is not bool for m in modes) or len(set(modes)) != len(modes):
        raise ValueError('causal_modes must be a nonempty list of distinct bool values.')
    if type(config['warmups']) is not int or config['warmups'] < 0:
        raise ValueError('warmups must be a nonnegative integer.')
    for key in ('atol', 'rtol'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('Numerical tolerances must be finite nonnegative numbers.')
    if config['dtype'] not in ('float32', 'float64', 'float16', 'bfloat16'):
        raise ValueError('Unsupported benchmark dtype.')


def run(args):
    config = json.loads((args.root / 'configs/attention_config.json').read_text(encoding='utf-8'))
    validate_config(config)
    device = torch.device('cuda:0' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select a Colab GPU or use --device cpu.')
    torch.set_num_threads(config['cpu_threads'])
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(config['seed'])
    identity = {'config': config, 'device': str(device), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'platform': platform.platform(),
                'processor': platform.processor(), 'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'source_sha256': {name: hashlib.sha256((args.root / name).read_bytes()).hexdigest()
                    for name in ('attention.py', 'benchmark_attention.py', 'attention_benchmark_report.py')}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.output or args.root / 'results' / f'attention_{device.type}'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, metrics_path = output / 'run_manifest.json', output / 'metrics.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['signature'] != signature:
            raise ValueError('Output belongs to a different experiment. Choose a new --output directory.')
        if metrics_path.exists():
            prior = json.loads(metrics_path.read_text(encoding='utf-8'))
            if prior['complete']:
                write_reports(output, prior)
                print((output / 'summary.md').read_text(encoding='utf-8'))
                if not prior['passed']:
                    raise RuntimeError('This completed experiment has numerical failures; its artifacts were preserved.')
                return
    write_json(manifest_path, {'signature': signature, 'identity': identity})
    payload = {'complete': False, 'passed': False, 'device': str(device), 'signature': signature, 'cases': []}
    write_json(metrics_path, payload)
    for index, (length, causal) in enumerate((n, causal) for n in config['lengths'] for causal in config['causal_modes']):
        case = run_case(config, length, causal, device, config['seed'] + index)
        payload['cases'].append(case)
        write_json(metrics_path, payload)
        print(case['case_id'], 'passed:', case['passed'],
              'tiled max abs error:', case['numerical']['tiled']['max_absolute_error'], flush=True)
    payload['complete'] = len(payload['cases']) == len(config['lengths']) * len(config['causal_modes'])
    payload['passed'] = payload['complete'] and all(case['passed'] for case in payload['cases'])
    write_json(metrics_path, payload)
    write_reports(output, payload)
    print('Results:', output, flush=True)
    if not payload['passed']:
        raise RuntimeError('Attention experiment retained one or more numerical/completeness failures.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    run(parser.parse_args())
