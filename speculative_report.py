"""Strict speculative-decoding trial aggregation and report artifacts."""

import csv
import math
import statistics


def summarize_trials(cases, trials, draft_lengths, repetitions):
    modes = ['greedy', *[f'speculative_{k}' for k in draft_lengths]]
    expected = {(case['case_id'], mode, repeat) for case in cases
                for mode in modes for repeat in range(repetitions)}
    seen = set()
    for trial in trials:
        key = (trial['case_id'], trial['mode'], trial['repeat'])
        seconds = trial['result']['total_seconds']
        if key not in expected or key in seen or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Duplicate, unexpected or invalid speculative trial.')
        seen.add(key)
    summaries = []
    for case in cases:
        selected = [t for t in trials if t['case_id'] == case['case_id']]
        baselines = [t for t in selected if t['mode'] == 'greedy']
        baseline_tokens = baselines[0]['result']['token_ids'] if baselines else None
        groups = {}
        for mode in modes:
            rows = [t for t in selected if t['mode'] == mode]
            if not rows:
                continue
            values = [t['result'] for t in rows]
            tokens_equal = baseline_tokens is not None and all(t['matches_greedy'] and
                t['result']['token_ids'] == baseline_tokens for t in rows)
            fixed = case['kind'] != 'fixed_token_stress' or all(len(v['token_ids']) == case['output_budget'] for v in values)
            seconds = [v['total_seconds'] for v in values]
            result = {'repetitions': len(rows), 'tokens_equal': tokens_equal, 'fixed_budget_satisfied': fixed,
                'median_seconds': statistics.median(seconds), 'min_seconds': min(seconds), 'max_seconds': max(seconds),
                'tokens_per_second_median': statistics.median(len(v['token_ids']) / v['total_seconds'] for v in values),
                'target_forward_calls_median': statistics.median(v['target_forward_calls'] for v in values),
                'draft_forward_calls_median': statistics.median(v['draft_forward_calls'] for v in values)}
            proposed = sum(v.get('proposed_tokens', 0) for v in values)
            accepted = sum(v.get('accepted_tokens', 0) for v in values)
            result['draft_acceptance_rate'] = accepted / proposed if proposed else None
            for key in ('initial_allocated_gib', 'peak_allocated_gib', 'peak_reserved_gib'):
                samples = [v[key] for v in values if v[key] is not None]
                result[key + '_max'] = max(samples) if samples else None
            groups[mode] = result
        complete = len(selected) == len(modes) * repetitions
        passed = complete and all(g['tokens_equal'] and g['fixed_budget_satisfied'] for g in groups.values())
        for name, group in groups.items():
            group['speedup_over_greedy'] = (groups['greedy']['median_seconds'] / group['median_seconds']
                if name != 'greedy' and passed else None)
        summaries.append({'case_id': case['case_id'], 'complete': complete, 'passed': passed, 'modes': groups})
    complete = seen == expected
    return {'complete': complete, 'passed': complete and all(c['passed'] for c in summaries),
            'expected_trials': len(expected), 'recorded_trials': len(trials), 'cases': summaries}


def write_reports(output, summary, checks, *, smoke=False):
    from baseline import save_json
    passed = summary['passed'] and len(checks) == len(summary['cases']) and all(c['passed'] for c in checks)
    payload = {**summary, 'passed': passed, 'preflight': checks,
               'experiment': 'miniature_cpu_smoke' if smoke else 'qwen3_4b_adapter_fp32_target_math_sdpa',
               'memory_protocol': 'Both models resident in every timed mode; GPU peaks include both models.'}
    save_json(output / 'metrics.json', payload)
    fields = ['case_id', 'mode', 'tokens_equal', 'median_seconds', 'min_seconds', 'max_seconds',
              'tokens_per_second_median', 'speedup_over_greedy', 'draft_acceptance_rate',
              'target_forward_calls_median', 'draft_forward_calls_median', 'peak_allocated_gib_max']
    with (output / 'benchmark.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for case in summary['cases']:
            for mode, values in case['modes'].items():
                writer.writerow({'case_id': case['case_id'], 'mode': mode, **values})
    lines = ['# Greedy speculative decoding', '',
        f"Experiment: **{payload['experiment']}**; timed work complete: **{summary['complete']}**; all checks passed: **{passed}**.", '',
        f"Recorded {summary['recorded_trials']}/{summary['expected_trials']} timed trials.", '',
        '| Case | Mode | Median seconds | Tokens/s | Speedup | Draft acceptance | Equal tokens |',
        '|---|---|---:|---:|---:|---:|---|']
    for case in summary['cases']:
        for mode, value in case['modes'].items():
            speed = value['speedup_over_greedy']; acceptance = value['draft_acceptance_rate']
            speed_text = f'{speed:.3f}x' if speed is not None else '—'
            acceptance_text = f'{acceptance:.3f}' if acceptance is not None else '—'
            lines.append(f"| {case['case_id']} | {mode} | {value['median_seconds']:.4f} | {value['tokens_per_second_median']:.2f} | {speed_text} | {acceptance_text} | {value['tokens_equal']} |")
    precision = ('CPU smoke uses random miniature FP64 target and draft models; it is not a full-model GPU result.'
                 if smoke else 'FP32 target computation and math SDPA are shared by ordinary and speculative generation. '
                 'Target weights remain 4-bit NF4; draft is FP16. This is a separate controlled experiment '
                 'from the earlier FP16 classification/KV runs.')
    lines += ['', precision, '',
        'All target/draft model weights remain resident in every mode. GPU memory peaks',
        'therefore include both models, even ordinary greedy. CPU GPU-memory values are null.',
        'Timings include drafting, target verification, Python comparisons and cache rollback;',
        'loading, tokenization, warmup, initial buffer allocation and final CPU output transfer',
        'are excluded. CUDA timing is synchronized; min/max and every trial are retained.', '',
        'Acceptance is accepted matching-prefix draft tokens / all proposed draft tokens.',
        'Target bonus/correction tokens are excluded from that numerator. Slowdowns are',
        'retained. A mismatch is never repaired and suppresses validated speedup.', '',
        'Preflight checks ordinary target greedy against HF and speculative against ordinary',
        'before any timed trials. Controlled stress cases ignore EOS; full clinical prompts',
        'use normal EOS stopping. Held-out test records are not used.', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    if summary['complete']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        modes = list(summary['cases'][0]['modes'])
        figure, axis = plt.subplots(figsize=(10, 4))
        width = 0.8 / len(modes)
        for index, mode in enumerate(modes):
            positions = [i + (index - (len(modes) - 1) / 2) * width for i in range(len(summary['cases']))]
            axis.bar(positions, [c['modes'][mode]['median_seconds'] for c in summary['cases']], width=width, label=mode)
        axis.set_xticks(range(len(summary['cases'])), [c['case_id'] for c in summary['cases']], rotation=30)
        axis.set_ylabel('Median generation seconds')
        axis.set_title(payload['experiment']); axis.legend(); figure.tight_layout()
        figure.savefig(output / 'timings.png', dpi=160); plt.close(figure)
    return payload
