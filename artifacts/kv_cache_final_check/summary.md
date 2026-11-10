# KV-cache benchmark

Experiment: **miniature_cpu_smoke**; completed: **True**; all correctness checks passed: **True**.

Recorded 36/36 timed trials.

| Input tokens | Output tokens | Uncached total (s) | Cached total (s) | Speedup | Tokens equal |
|---:|---:|---:|---:|---:|---|
| 8 | 4 | 0.0103 | 0.0103 | 1.00x | True |
| 8 | 8 | 0.0806 | 0.0528 | 1.53x | True |
| 8 | 12 | 0.0285 | 0.0840 | 0.34x | True |
| 16 | 4 | 0.0159 | 0.0145 | 1.10x | True |
| 16 | 8 | 0.0215 | 0.0243 | 0.88x | True |
| 16 | 12 | 0.0370 | 0.0399 | 0.93x | True |
| 32 | 4 | 0.0120 | 0.0137 | 0.88x | True |
| 32 | 8 | 0.0184 | 0.0173 | 1.06x | True |
| 32 | 12 | 0.0296 | 0.0302 | 0.98x | True |

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
