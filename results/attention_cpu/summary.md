# Attention experiment

Device: **cpu**; complete: **True**; numerical checks passed: **True**.

| Case | Reference ms | Tiled ms | SDPA auto ms | Tiled max abs error | Reference / tiled |
|---|---:|---:|---:|---:|---:|
| n128-full | 1.885 | 5.705 | 1.243 | 4.34e-07 | 0.330x |
| n128-causal | 2.868 | 4.143 | 0.928 | 5.27e-07 | 0.692x |
| n256-full | 3.872 | 16.509 | 2.962 | 3.33e-07 | 0.235x |
| n256-causal | 7.421 | 15.897 | 3.132 | 4.45e-07 | 0.467x |
| n512-full | 15.056 | 66.686 | 8.877 | 2.82e-07 | 0.226x |
| n512-causal | 23.324 | 43.309 | 9.281 | 6.91e-07 | 0.539x |

Ratios below 1 mean the tiled implementation was slower. Every method uses the
same inputs and dtype. Times are medians; CSV/JSON retain min/max ranges and raw
repetitions. CUDA measurements synchronize before/after; warmups and transfers
are outside timing. Error is measured against a dense FP64 CPU oracle.

Custom implementations accumulate FP16/BF16 in FP32 and preserve FP64 inputs.
SDPA auto means PyTorch selected the backend; it is not a claim that a particular
fused kernel ran. This is standalone attention on controlled tensors, not model
accuracy, a training experiment, or a measured end-to-end Qwen speedup.

CPU peak tensor memory is not measured (null). Score-workspace sizes are
theoretical tensor sizes, not measured peak memory. GPU peaks include input
tensors and returned output plus intermediate allocations. Python loop overhead,
launches, temporary allocations and repeated tile operations can outweigh savings.
