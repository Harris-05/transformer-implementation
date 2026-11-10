"""Attention timing aggregation and scientific artifacts; no model dependencies."""

import csv
import math
import statistics


METHODS = ('reference', 'tiled', 'sdpa_auto')


def aggregate_measurements(records, *, repetitions):
    expected = {(method, repeat) for method in METHODS for repeat in range(repetitions)}
    seen = set()
    for record in records:
        key = (record['method'], record['repeat'])
        if key not in expected or key in seen or not math.isfinite(record['seconds']) or record['seconds'] <= 0:
            raise ValueError('Duplicate, unexpected or invalid timing measurement.')
        seen.add(key)
    groups = {}
    for method in METHODS:
        rows = [r for r in records if r['method'] == method]
        if not rows:
            continue
        seconds = [r['seconds'] for r in rows]
        groups[method] = {'repetitions': len(rows), 'median_seconds': statistics.median(seconds),
                          'min_seconds': min(seconds), 'max_seconds': max(seconds)}
        for key in ('initial_allocated_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'):
            values = [r[key] for r in rows if r[key] is not None]
            groups[method][key] = max(values) if values else None
    complete = seen == expected
    return {'complete': complete, 'methods': groups,
            'reference_over_tiled_speedup': groups['reference']['median_seconds'] / groups['tiled']['median_seconds']
            if complete else None}


def score_workspace_bytes(*, batch, heads, queries, keys, block_q, block_k, element_bytes):
    return {'kind': 'theoretical_score_tensor_only',
            'dense_score_tensor_bytes': batch * heads * queries * keys * element_bytes,
            'largest_tiled_score_tensor_bytes': batch * heads * min(block_q, queries) * min(block_k, keys) * element_bytes}


def write_reports(output, payload):
    rows = payload['cases']
    fields = ['case_id', 'method', 'passed', 'max_absolute_error', 'mean_absolute_error',
              'median_seconds', 'min_seconds', 'max_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes']
    with (output / 'benchmark.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for case in rows:
            for method, values in case['timing']['methods'].items():
                writer.writerow({'case_id': case['case_id'], 'method': method,
                                 **case['numerical'][method], **values})
    lines = ['# Attention experiment', '',
        f"Device: **{payload['device']}**; complete: **{payload['complete']}**; numerical checks passed: **{payload['passed']}**.", '',
        '| Case | Reference ms | Tiled ms | SDPA auto ms | Tiled max abs error | Reference / tiled |',
        '|---|---:|---:|---:|---:|---:|']
    for case in rows:
        timing = case['timing']['methods']
        values = [timing[m]['median_seconds'] * 1000 for m in METHODS]
        error = case['numerical']['tiled']['max_absolute_error']
        ratio = case['timing']['reference_over_tiled_speedup']
        error_text = f'{error:.3g}' if error is not None else 'nonfinite'
        ratio_text = f'{ratio:.3f}x' if ratio is not None else 'unavailable'
        lines.append(f"| {case['case_id']} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {error_text} | {ratio_text} |")
    lines += ['', 'Ratios below 1 mean the tiled implementation was slower. Every method uses the',
        'same inputs and dtype. Times are medians; CSV/JSON retain min/max ranges and raw',
        'repetitions. CUDA measurements synchronize before/after; warmups and transfers',
        'are outside timing. Error is measured against a dense FP64 CPU oracle.', '',
        'Custom implementations accumulate FP16/BF16 in FP32 and preserve FP64 inputs.',
        'SDPA auto means PyTorch selected the backend; it is not a claim that a particular',
        'fused kernel ran. This is standalone attention on controlled tensors, not model',
        'accuracy, a training experiment, or a measured end-to-end Qwen speedup.', '',
        'CPU peak tensor memory is not measured (null). Score-workspace sizes are',
        'theoretical tensor sizes, not measured peak memory. GPU peaks include input',
        'tensors and returned output plus intermediate allocations. Python loop overhead,',
        'launches, temporary allocations and repeated tile operations can outweigh savings.', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, offset in zip(METHODS, (-0.25, 0, 0.25)):
        values = [r['timing']['methods'][method]['median_seconds'] * 1000 for r in rows]
        lows = [(r['timing']['methods'][method]['median_seconds'] - r['timing']['methods'][method]['min_seconds']) * 1000 for r in rows]
        highs = [(r['timing']['methods'][method]['max_seconds'] - r['timing']['methods'][method]['median_seconds']) * 1000 for r in rows]
        axes[0].bar([i + offset for i in range(len(rows))], values, width=0.25,
                    label=method, yerr=[lows, highs], capsize=2)
        errors = [r['numerical'][method]['max_absolute_error'] for r in rows]
        axes[1].plot(range(len(rows)), [max(e, 1e-16) if e is not None else float('nan') for e in errors],
                     marker='o', label=method)
    for axis in axes:
        axis.set_xticks(range(len(rows)), [r['case_id'] for r in rows], rotation=45)
        axis.legend()
    axes[0].set_ylabel('Milliseconds (median; range bars)')
    axes[0].set_title(f"Attention runtime: {payload['device']}")
    axes[1].set_ylabel('Maximum absolute error vs FP64 oracle')
    axes[1].set_yscale('log')
    figure.tight_layout()
    figure.savefig(output / 'attention_comparison.png', dpi=160)
    plt.close(figure)
