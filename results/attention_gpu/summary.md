# Attention experiment

Device: **cuda**; complete: **True**; numerical checks passed: **True**.

| Case | Reference ms | Tiled ms | SDPA auto ms | Tiled max abs error | Reference / tiled |
|---|---:|---:|---:|---:|---:|
| n128-full | 0.876 | 2.709 | 0.292 | 4.34e-07 | 0.323x |
| n128-causal | 1.045 | 2.583 | 0.299 | 5.66e-07 | 0.405x |
| n256-full | 1.406 | 7.888 | 0.324 | 3.63e-07 | 0.178x |
| n256-causal | 1.329 | 6.719 | 0.330 | 4.98e-07 | 0.198x |
| n512-full | 1.415 | 28.808 | 0.673 | 2.82e-07 | 0.049x |
| n512-causal | 1.584 | 22.205 | 0.677 | 6.91e-07 | 0.071x |

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
