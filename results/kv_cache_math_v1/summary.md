# KV-cache benchmark

Experiment: **selected_4bit_adapter_gpu**; completed: **True**; all correctness checks passed: **False**.

Recorded 54/54 timed trials.

| Input tokens | Output tokens | Uncached total (s) | Cached total (s) | Speedup | Tokens equal |
|---:|---:|---:|---:|---:|---|
| 128 | 16 | 3.2494 | 2.2908 | 1.42x | True |
| 128 | 32 | 7.1798 | 4.9625 | 1.45x | True |
| 128 | 64 | 15.9552 | 9.2676 | 1.72x | True |
| 256 | 16 | 5.8830 | 2.4207 | 2.43x | True |
| 256 | 32 | 12.4834 | 4.7097 | 2.65x | True |
| 256 | 64 | 24.4015 | 9.7462 | 2.50x | True |
| 512 | 16 | 9.9124 | 2.8475 | 3.48x | True |
| 512 | 32 | 20.7754 | 4.9782 | 4.17x | True |
| 512 | 64 | 42.6393 | 10.0655 | unavailable | False |

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
