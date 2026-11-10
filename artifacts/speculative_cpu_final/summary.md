# Greedy speculative decoding

Experiment: **miniature_cpu_smoke**; timed work complete: **True**; all checks passed: **True**.

Recorded 18/18 timed trials.

| Case | Mode | Median seconds | Tokens/s | Speedup | Draft acceptance | Equal tokens |
|---|---|---:|---:|---:|---:|---|
| tiny-3-1 | greedy | 0.0439 | 59.23 | — | — | True |
| tiny-3-1 | speculative_2 | 0.0250 | 41.33 | 1.756x | 0.000 | True |
| tiny-3-1 | speculative_4 | 0.0331 | 30.24 | 1.328x | 0.000 | True |
| tiny-7-9 | greedy | 0.0793 | 113.45 | — | — | True |
| tiny-7-9 | speculative_2 | 0.2762 | 32.71 | 0.287x | 0.000 | True |
| tiny-7-9 | speculative_4 | 0.4773 | 19.21 | 0.166x | 0.000 | True |
| tiny-17-15 | greedy | 0.2528 | 72.25 | — | — | True |
| tiny-17-15 | speculative_2 | 0.5835 | 27.71 | 0.433x | 0.000 | True |
| tiny-17-15 | speculative_4 | 0.8621 | 17.90 | 0.293x | 0.000 | True |

CPU smoke uses random miniature FP64 target and draft models; it is not a full-model GPU result.

All target/draft model weights remain resident in every mode. GPU memory peaks
therefore include both models, even ordinary greedy. CPU GPU-memory values are null.
Timings include drafting, target verification, Python comparisons and cache rollback;
loading, tokenization, warmup, initial buffer allocation and final CPU output transfer
are excluded. CUDA timing is synchronized; min/max and every trial are retained.

Acceptance is accepted matching-prefix draft tokens / all proposed draft tokens.
Target bonus/correction tokens are excluded from that numerator. Slowdowns are
retained. A mismatch is never repaired and suppresses validated speedup.

Preflight checks ordinary target greedy against HF and speculative against ordinary
before any timed trials. Controlled stress cases ignore EOS; full clinical prompts
use normal EOS stopping. Held-out test records are not used.
