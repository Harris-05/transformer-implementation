# KV-cache benchmark

Experiment: **miniature_cpu_smoke**; completed: **True**; all correctness checks passed: **True**.

Recorded 36/36 timed trials.

| Input tokens | Output tokens | Uncached total (s) | Cached total (s) | Speedup | Tokens equal |
|---:|---:|---:|---:|---:|---|
| 8 | 4 | 0.0133 | 0.0179 | 0.75x | True |
| 8 | 8 | 0.0270 | 0.0211 | 1.28x | True |
| 8 | 12 | 0.0282 | 0.0422 | 0.67x | True |
| 16 | 4 | 0.0112 | 0.0108 | 1.04x | True |
| 16 | 8 | 0.0197 | 0.0172 | 1.14x | True |
| 16 | 12 | 0.0257 | 0.0240 | 1.07x | True |
| 32 | 4 | 0.0228 | 0.0229 | 0.99x | True |
| 32 | 8 | 0.0245 | 0.0229 | 1.07x | True |
| 32 | 12 | 0.0615 | 0.0376 | 1.63x | True |

Times are medians of repeated synchronized measurements. Prefill includes the first forward
and first greedy token; decode covers the remaining N-1 tokens. Total tokens/s uses N / total time.
Raw trials include every token, timing, cache size and memory peak. Peaks include model weights;
reserved memory is affected by allocator history. The two paths alternate order across repetitions.

Fixed-token stress prompts use cropped/repeated development token IDs, ignore EOS, and are not
clinical predictions or accuracy measurements. Complete clinical prompts separately stop at EOS.
A mismatch is retained with its first differing position and logit diagnostics. No output is repaired.
No speedup is reported for unequal or incomplete fixed-token workloads.

GPU trials use one warm-up per case/path by default, with fresh caches in every call.
Tokenization, loading, CPU output transfer, plotting and correctness tracing are outside timing.
The held-out test set is not used. CPU smoke timings do not establish full-model GPU performance.
