# Attention experiment

Device: **cpu**; complete: **True**; numerical checks passed: **True**.

| Case | Reference ms | Tiled ms | SDPA auto ms | Tiled max abs error | Reference / tiled |
|---|---:|---:|---:|---:|---:|
| n128-full | 1.531 | 3.831 | 0.701 | 4.34e-07 | 0.400x |
| n128-causal | 2.410 | 3.067 | 0.927 | 5.27e-07 | 0.785x |
| n256-full | 2.610 | 8.057 | 1.552 | 3.33e-07 | 0.324x |
| n256-causal | 6.140 | 12.129 | 2.691 | 4.45e-07 | 0.506x |
| n512-full | 9.286 | 33.141 | 4.243 | 2.82e-07 | 0.280x |
| n512-causal | 11.013 | 24.436 | 3.541 | 6.91e-07 | 0.451x |

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
