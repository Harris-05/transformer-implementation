# Attention experiment

Device: **cpu**; complete: **True**; numerical checks passed: **True**.

| Case | Reference ms | Tiled ms | SDPA auto ms | Tiled max abs error | Reference / tiled |
|---|---:|---:|---:|---:|---:|
| n128-full | 1.820 | 3.220 | 0.694 | 4.34e-07 | 0.565x |
| n128-causal | 1.375 | 2.639 | 0.920 | 5.27e-07 | 0.521x |
| n256-full | 3.373 | 16.227 | 1.932 | 3.33e-07 | 0.208x |
| n256-causal | 7.617 | 13.738 | 3.796 | 4.45e-07 | 0.554x |
| n512-full | 13.500 | 39.385 | 7.238 | 2.82e-07 | 0.343x |
| n512-causal | 14.958 | 31.892 | 5.326 | 6.91e-07 | 0.469x |

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
