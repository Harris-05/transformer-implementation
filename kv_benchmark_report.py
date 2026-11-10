"""Validate and summarize raw KV-cache benchmark repetitions."""

import csv
import json
import statistics


def summarize_trials(cases, trials, *, repeats):
    expected = {(case['case_id'], mode, repeat) for case in cases
                for mode in ('cached', 'uncached') for repeat in range(repeats)}
    seen = set()
    for trial in trials:
        key = (trial['case_id'], trial['mode'], trial['repeat'])
        if key not in expected or key in seen:
            raise ValueError('Duplicate or unexpected benchmark trial.')
        seen.add(key)
    rows = []
    for case in cases:
        selected = [t for t in trials if t['case_id'] == case['case_id']]
        modes = {}
        for mode in ('cached', 'uncached'):
            values = [t['result'] for t in selected if t['mode'] == mode]
            if not values:
                continue
            stats = {'repetitions': len(values)}
            for key in ('prefill_seconds', 'decode_seconds', 'total_seconds'):
                samples = [value[key] for value in values]
                stats.update({key + '_median': statistics.median(samples),
                              key + '_min': min(samples), key + '_max': max(samples)})
            stats['output_tokens_per_second_median'] = statistics.median(
                len(v['token_ids']) / v['total_seconds'] for v in values)
            rates = [v['decode_steps'] / v['decode_seconds'] for v in values if v['decode_seconds'] > 0]
            stats['decode_tokens_per_second_median'] = statistics.median(rates) if rates else None
            for key in ('initial_allocated_gib', 'peak_allocated_gib', 'peak_reserved_gib', 'cache_bytes', 'cache_tokens'):
                samples = [v[key] for v in values if v[key] is not None]
                stats[key + '_max'] = max(samples) if samples else None
            modes[mode] = stats
        sequences = [t['result']['token_ids'] for t in selected]
        full = len(selected) == repeats * 2
        equal = full and all(sequence == sequences[0] for sequence in sequences)
        fixed = full and all(len(sequence) == case['output_budget'] for sequence in sequences)
        matches_reference = full and all(t.get('matches_reference', True) for t in selected)
        speedup = (modes['uncached']['total_seconds_median'] / modes['cached']['total_seconds_median']
                   if len(modes) == 2 and equal and fixed and matches_reference else None)
        rows.append({'case_id': case['case_id'], 'input_tokens': case['input_tokens'],
            'output_budget': case['output_budget'], 'complete': full, 'tokens_equal': equal,
            'fixed_length_satisfied': fixed, 'modes': modes,
            'matches_reference': matches_reference,
            'total_speedup_uncached_over_cached': speedup})
    complete = seen == expected
    return {'complete': complete, 'passed': complete and all(r['tokens_equal'] and r['fixed_length_satisfied'] and r['matches_reference'] for r in rows),
            'expected_trials': len(expected), 'recorded_trials': len(trials), 'cases': rows}


def combine_results(summary, correctness, *, expected_ids):
    complete = (summary['complete'] and len(correctness) == len(expected_ids)
                and {item['case_id'] for item in correctness} == set(expected_ids))
    return {**summary, 'timing_complete': summary['complete'], 'complete': complete,
            'passed': complete and summary['passed'] and all(item['passed'] for item in correctness)}


def write_report(output, summary, correctness, *, expected_ids, smoke=False):
    # Local import keeps pure summary tests independent of plotting dependencies.
    from baseline import save_json
    payload = {**combine_results(summary, correctness, expected_ids=expected_ids), 'correctness': correctness,
               'experiment': 'miniature_cpu_smoke' if smoke else 'selected_4bit_adapter_gpu'}
    save_json(output / 'metrics.json', payload)
    fields = ['case_id', 'mode', 'repetitions', 'prefill_seconds_median',
              'decode_seconds_median', 'total_seconds_median',
              'output_tokens_per_second_median', 'decode_tokens_per_second_median',
              'peak_allocated_gib_max', 'peak_reserved_gib_max', 'cache_bytes_max']
    with (output / 'benchmark.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for case in summary['cases']:
            for mode, values in case['modes'].items():
                writer.writerow({'case_id': case['case_id'], 'mode': mode, **values})
    lines = ['# KV-cache benchmark', '',
        f"Experiment: **{payload['experiment']}**; completed: **{payload['complete']}**; all correctness checks passed: **{payload['passed']}**.", '',
        f"Recorded {summary['recorded_trials']}/{summary['expected_trials']} timed trials.", '',
        '| Input tokens | Output tokens | Uncached total (s) | Cached total (s) | Speedup | Tokens equal |',
        '|---:|---:|---:|---:|---:|---|']
    for case in summary['cases']:
        def seconds(mode):
            value = case['modes'].get(mode, {}).get('total_seconds_median')
            return f'{value:.4f}' if value is not None else 'pending'
        speedup = case['total_speedup_uncached_over_cached']
        ratio = f'{speedup:.2f}x' if speedup is not None else 'unavailable'
        lines.append(f"| {case['input_tokens']} | {case['output_budget']} | {seconds('uncached')} | {seconds('cached')} | {ratio} | {case['tokens_equal']} |")
    lines += ['', 'Times are medians of repeated synchronized measurements. Prefill includes the first forward',
        'and first greedy token; decode covers the remaining N-1 tokens. Total tokens/s uses N / total time.',
        'Raw trials include every token, timing, cache size and memory peak. Peaks include model weights;',
        'reserved memory is affected by allocator history. The two paths alternate order across repetitions.',
        '', 'Fixed-token stress prompts use cropped/repeated development token IDs, ignore EOS, and are not',
        'clinical predictions or accuracy measurements. Complete clinical prompts separately stop at EOS.',
        'A mismatch is retained with its first differing position and logit diagnostics. No output is repaired.',
        'No speedup is reported for unequal or incomplete fixed-token workloads.',
        '', 'GPU trials use one warm-up per case/path by default, with fresh caches in every call.',
        'Tokenization, loading, CPU output transfer, plotting and correctness tracing are outside timing.',
        'The held-out test set is not used. CPU smoke timings do not establish full-model GPU performance.', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    if not summary['complete']:
        return payload
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mode, offset in (('uncached', -0.18), ('cached', 0.18)):
        positions = [i + offset for i in range(len(summary['cases']))]
        for axis, metric, title in ((axes[0], 'total_seconds', 'Total generation seconds (median)'),
                                     (axes[1], 'decode_seconds', 'Decode seconds (median)')):
            centers = [c['modes'][mode][metric + '_median'] for c in summary['cases']]
            errors = [[center - c['modes'][mode][metric + '_min'] for center, c in zip(centers, summary['cases'])],
                      [c['modes'][mode][metric + '_max'] - center for center, c in zip(centers, summary['cases'])]]
            axis.bar(positions, centers, width=0.36, label=mode, yerr=errors, capsize=2)
            axis.set_title(title)
            axis.set_xticks(range(len(summary['cases'])), [c['case_id'] for c in summary['cases']], rotation=45)
            axis.legend()
    figure.suptitle(payload['experiment'])
    figure.tight_layout()
    figure.savefig(output / 'timings.png', dpi=160)
    plt.close(figure)
    return payload
