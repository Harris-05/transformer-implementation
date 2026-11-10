# Greedy speculative decoding

Experiment: **qwen3_4b_adapter_fp32_target_math_sdpa**; timed work complete: **True**; all checks passed: **True**.

Recorded 54/54 timed trials.

| Case | Mode | Median seconds | Tokens/s | Speedup | Draft acceptance | Equal tokens |
|---|---|---:|---:|---:|---:|---|
| p128-o16 | greedy | 2.8710 | 5.57 | — | — | True |
| p128-o16 | speculative_2 | 2.1812 | 7.34 | 1.316x | 0.833 | True |
| p128-o16 | speculative_4 | 2.1094 | 7.59 | 1.361x | 0.812 | True |
| p256-o32 | greedy | 4.8140 | 6.65 | — | — | True |
| p256-o32 | speculative_2 | 5.5359 | 5.78 | 0.870x | 0.342 | True |
| p256-o32 | speculative_4 | 7.3826 | 4.33 | 0.652x | 0.270 | True |
| p512-o64 | greedy | 9.6746 | 6.62 | — | — | True |
| p512-o64 | speculative_2 | 9.0020 | 7.11 | 1.075x | 0.765 | True |
| p512-o64 | speculative_4 | 10.1862 | 6.28 | 0.950x | 0.562 | True |
| clinical-938 | greedy | 3.4959 | 4.58 | — | — | True |
| clinical-938 | speculative_2 | 4.0603 | 3.94 | 0.861x | 0.368 | True |
| clinical-938 | speculative_4 | 4.4311 | 3.61 | 0.789x | 0.310 | True |
| clinical-6 | greedy | 3.3606 | 4.17 | — | — | True |
| clinical-6 | speculative_2 | 3.0747 | 4.55 | 1.093x | 0.615 | True |
| clinical-6 | speculative_4 | 3.4764 | 4.03 | 0.967x | 0.429 | True |
| clinical-243 | greedy | 3.8976 | 4.11 | — | — | True |
| clinical-243 | speculative_2 | 4.2995 | 3.72 | 0.907x | 0.368 | True |
| clinical-243 | speculative_4 | 4.7716 | 3.35 | 0.817x | 0.310 | True |

FP32 target computation and math SDPA are shared by ordinary and speculative generation. Target weights remain 4-bit NF4; draft is FP16. This is a separate controlled experiment from the earlier FP16 classification/KV runs.

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
