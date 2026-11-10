# KV-cache benchmark

Experiment: **miniature_cpu_smoke**; completed: **True**; all correctness checks passed: **True**.

Recorded 36/36 timed trials.

| Input tokens | Output tokens | Uncached total (s) | Cached total (s) | Speedup | Tokens equal |
|---:|---:|---:|---:|---:|---|
| 8 | 4 | 0.0371 | 0.0764 | 0.49x | True |
| 8 | 8 | 0.0486 | 0.0497 | 0.98x | True |
| 8 | 12 | 0.0855 | 0.0865 | 0.99x | True |
| 16 | 4 | 0.0264 | 0.0310 | 0.85x | True |
| 16 | 8 | 0.0657 | 0.0396 | 1.66x | True |
| 16 | 12 | 0.1875 | 0.1355 | 1.38x | True |
| 32 | 4 | 0.0914 | 0.0621 | 1.47x | True |
| 32 | 8 | 0.1811 | 0.0571 | 3.17x | True |
| 32 | 12 | 0.0735 | 0.0606 | 1.21x | True |

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
