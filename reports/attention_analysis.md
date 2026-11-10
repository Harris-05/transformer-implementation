# Attention implementation and CPU experiment

Task 4's dense reference and tiled online-softmax forward paths are implemented.
They are tested as standalone attention operators; the selected Qwen model's
attention has not been replaced. The received Colab GPU results were verified
and preserved in `results/attention_gpu/` alongside the original ZIP.

## Verification

All **59 project tests pass**. The 14 attention/benchmark tests include literal
hand-calculated values, independent PyTorch math SDPA, causal leakage checks,
partial tile boundaries, initially/all-masked rows, negative/rectangular query
offsets, GQA, large FP16 magnitudes, noncontiguous tensors and FP64 preservation.
The later-larger-maximum fixture specifically checks old-accumulator rescaling.
Invalid/empty/duplicated configuration and interrupted-report recovery are covered.
Completed metrics can regenerate reports without rerunning timed measurements.

The final local CPU experiment contains six controlled cases and **90 timed
calls** (three methods, five repetitions each). Numerical outputs are compared
with a dense FP64 CPU oracle, at `atol=2e-5, rtol=2e-4` for the FP32 experiment.
Every method/case passes. Maximum tiled absolute error is **6.9103e-7**, below
the configured absolute tolerance. JSON retains mean and relative errors too.
The run manifest records source hashes, configuration and environment; the
notebook's seven embedded sources also match their recorded checksums.

## Measured CPU runtime

Primary artifacts: `results/attention_cpu_final/`. Measurements use a shared
Windows laptop, two CPU threads, two warmups, five repetitions, batch one,
four heads, head dimension 64, FP32 inputs and 64x64 tiles. Compare within this
run; these numbers do not predict GPU or end-to-end model performance.

| Case | Dense median ms | Tiled median ms | SDPA auto median ms | Dense / tiled |
|---|---:|---:|---:|---:|
| 128 full | 1.820 | 3.220 | 0.694 | 0.565x |
| 128 causal | 1.375 | 2.639 | 0.920 | 0.521x |
| 256 full | 3.373 | 16.227 | 1.932 | 0.208x |
| 256 causal | 7.617 | 13.738 | 3.796 | 0.554x |
| 512 full | 13.500 | 39.385 | 7.238 | 0.343x |
| 512 causal | 14.958 | 31.892 | 5.326 | 0.469x |

The educational tiled path is slower in all six CPU cases. Python loops,
repeated small operations and allocation overhead outweigh its smaller score
workspace here. PyTorch automatic SDPA is faster in these measurements; no
specific fused backend was identified. Raw timings and min/max ranges are
retained in metrics.json and benchmark.csv, with a comparison PNG for the report.

## Memory scope

For length 512, the dense FP32 score tensor is **4 MiB**; the largest 64x64
tile score tensor is **64 KiB**. This is a theoretical score-tensor comparison,
not a measured 64x reduction in peak process memory. Input/output/accumulator
buffers, conversions and simultaneous intermediate tensors also require memory.
CPU native tensor peak memory was not measured; its fields are null.

## Verified Tesla T4 GPU experiment

The uploaded `attention_results.zip` contains a complete, passing experiment:
six cases and **90 timed calls**. Independently verified the signed identity,
current source/configuration hashes, case shapes/seeds, uniqueness/completeness
of timing records, every median/range/speed ratio, and all recorded allocated/
reserved memory aggregates. Reported numerical errors are finite and below the
configured absolute tolerance. Maximum tiled absolute error is **6.9123e-7**.
No GPU computation was rerun during this artifact verification; output tensors
are not saved in the archive, so exact CUDA numerical errors were not recomputed.

Environment: Tesla T4, PyTorch 2.14.1+cu130, CUDA 13.0, FP32 inputs/accumulation,
highest FP32 matmul precision, four heads and head dimension 64. Shapes, seeds,
tile sizes, masks, warmups and repetitions match the documented configuration.

| Case | Dense median ms | Tiled median ms | SDPA auto median ms | Dense peak MiB | Tiled peak MiB |
|---|---:|---:|---:|---:|---:|
| 128 full | 0.876 | 2.709 | 0.292 | 9.258 | 9.005 |
| 128 causal | 1.045 | 2.583 | 0.299 | 9.258 | 9.005 |
| 256 full | 1.406 | 7.888 | 0.324 | 11.891 | 9.505 |
| 256 causal | 1.329 | 6.719 | 0.330 | 11.891 | 9.505 |
| 512 full | 1.415 | 28.808 | 0.673 | 21.656 | 10.505 |
| 512 causal | 1.584 | 22.205 | 0.677 | 21.656 | 10.505 |

Peak values are the maximum PyTorch-allocated bytes across the five repetitions,
converted to MiB. At length 512, the tiled path uses approximately **51.5% less
peak allocated memory** than the dense reference in both mask modes. This is an
observed total allocator-peak reduction, distinct from the theoretical 64x score-
tensor ratio. Initial allocated memory and reserved peaks remain in raw records.

The tiled implementation is slower in every GPU case, while automatic SDPA is
fastest in this experiment. The explicit Python tile loop performs many small
operations and launches; it does not provide the fused memory scheduling of an
optimized kernel. These results demonstrate numerical agreement and a memory/
runtime trade-off, not a speed improvement. They do not establish an end-to-end
Qwen speedup or identify which SDPA auto kernel was selected.

Task 4 implementation, validation and the documented CPU/GPU experiment are
complete. Preserve these results for the report and browser benchmark view.
