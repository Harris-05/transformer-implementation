# Task 4: tiled attention with online softmax

## What is implemented

`attention.py` supplies two inference-only implementations:

- `reference_attention(q, k, v, causal=True, query_offset=0)` explicitly creates
  scaled scores, applies a causal mask, normalizes with a stable softmax, and
  computes weighted values.
- `tiled_attention(..., block_q=64, block_k=64)` processes query/key blocks using
  running maxima, normalizers and unnormalized weighted-value accumulators.
  It does not call a fused attention API or allocate a full score matrix when
  tiles are smaller than the sequence.

Inputs are finite tensors `[batch, heads, tokens, head_dim]` sharing device and
dtype. Q/K feature dimensions match; V may have a different feature dimension.
Q heads may be an integer multiple of KV heads for grouped-query attention.
Query i has absolute position `i + query_offset`; keys have positions `0..S-1`.
For a cached last-token query over S keys, pass `query_offset=S-1`. The default
offset is zero, including for rectangular shapes; it is not inferred silently.
Fully masked rows return zeros. FP16/BF16 inputs accumulate in FP32; FP64 is
preserved. Output dtype equals the input dtype. No dropout, bias, padding mask,
backward implementation or nonfinite-input guarantee is supplied.

## Algorithm and numerical stability

For query block Q and key/value block K,V, form `S = Q K^T / sqrt(d)` and mask
future keys to negative infinity. Each query row maintains running maximum m,
denominator l, and value numerator a, initialized to `-inf`, zero and zero:

```text
m_new = max(m, rowmax(S))
alpha = exp(m - m_new)
P = exp(S - m_new)
l_new = alpha * l + rowsum(P)
a_new = alpha * a + P @ V
m, l, a = m_new, l_new, a_new
output = a / l
```

When both the old and new maxima are `-inf`, subtraction would produce NaN.
The implementation substitutes a safe zero maximum for that calculation,
making masked probabilities/rescaling zero and retaining a zero accumulator.
Final zero denominators use one for division, yielding zero output.
When a later tile has a larger maximum, rescaling both old l and old a is
essential; otherwise probabilities from different tiles use incompatible scales.
Each partial final tile uses its actual dimensions. Entire future-only key tiles
are skipped, while boundary tiles receive elementwise causal masking.

The dense score tensor has `B * Hq * L * S` elements. The largest tiled score
tensor has at most `B * Hq * block_q * block_k` elements, plus linear input,
output, conversion and accumulator storage. GQA expansion happens per key/value
tile. Several tile intermediates may coexist, so one tile's byte count is not
total peak memory. Python/PyTorch allocations are not guaranteed to stay in
on-chip SRAM; this educational implementation is not a custom fused CUDA kernel.

## Validation

Unit tests include literal hand-calculated attention values; independent PyTorch
math-SDPA comparisons; causal future-token changes; negative offsets and all-masked
rows; rectangular cached queries; GQA; noncontiguous inputs; partial/unit/large
tiles; later larger maxima; high FP16 magnitudes; FP32/FP64/BF16/FP16; and bad
shapes, devices and dtypes. No model weights or dataset are needed.

The benchmark compares each implementation with a dense FP64 CPU oracle using
the exact same input values. Default FP32 tolerances are `atol=2e-5, rtol=2e-4`.
Maximum/mean absolute error, maximum relative error, finiteness and pass flags
are saved for every case/method. Relative error can be large near zero even
when absolute error is small; the declared absolute-plus-relative tolerance
determines passing. Library SDPA is a comparator, not used inside custom code.

## Performance experiment

`configs/attention_config.json` defines lengths 128/256/512, full and causal
attention, batch one, four heads, head dimension 64, FP32, tiles 64x64, two
warmups, five repetitions and two CPU threads. There are six cases, three methods
and **90 timed calls**. Methods rotate order across repetitions. Inputs are
generated once per case on CPU with a recorded seed, then transferred outside
timing. Highest FP32 matmul precision is selected and recorded.

Methods: dense reference, tiled online softmax, and `sdpa_auto` (PyTorch's selected
backend, without claiming any specific fused kernel). CUDA calls synchronize
before and after timing. Garbage collection, allocator cache clearing, warmup,
CPU-oracle work, device transfers, reporting and plotting are outside timing.
Kernel launches and temporary allocations inside each attention function count.
Clearing unused GPU allocator blocks gives a warmed-kernel, cleared-pool protocol;
it is not an end-to-end serving benchmark.

GPU allocated/reserved peaks include resident input tensors, output and temporary
allocations. Initial allocated bytes are recorded. Native CPU tensor peak memory
is **not measured**, so CPU memory fields are null. The separately labelled
score-workspace estimates are analytical tensor sizes, not measured process RAM
or a claim of total peak savings. Ratios below one are slowdowns and are retained.

Python loops, repeated operations, launches and allocation overhead can make the
educational tiled path slower than dense operations or optimized SDPA. Standalone
attention results do not establish an end-to-end Qwen improvement. This module
does not replace the selected model's attention implementation or alter its
clinical outputs; the later browser benchmark view can expose the experiment.

## Run locally or in Colab

```powershell
python -m unittest -v test_attention test_attention_benchmark
python benchmark_attention.py --device cpu --output results/attention_cpu_final
python make_colab_attention.py
```

Open `colab_attention.ipynb` in Colab and run its three steps. Source files are
embedded and checksummed in the notebook: **no ZIP, dataset, adapter or model
upload is required**. It uses a GPU when available, otherwise CPU, and downloads
`attention_results.zip` at the end. It does not train or load Qwen weights.
The result ZIP contains manifest/configuration, numerical results, raw repeated
timings, CSV, summary and a plot. Return it for measured GPU analysis.

The output folder refuses a different source/configuration/environment identity.
A matching completed experiment regenerates its reports and is displayed without
repeating measurements; recorded failures
are not replaced by a rerun. An interrupted incomplete run restarts the small
experiment. Choose a new output directory to intentionally run another version.

## Sources

- [FlashAttention: exact attention with tiling and online normalization](https://arxiv.org/html/2205.14135v2)
- [Milakov and Gimelshein: online softmax normalizer](https://arxiv.org/html/1805.02867v2)
- [PyTorch SDPA scaling, masks and backend behavior](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)

The equations follow these primary sources. The paper's optimized-kernel speedups
are not attributed to this Python implementation.

## Completed results

Verified CPU results are in `results/attention_cpu_final/`; verified Tesla T4 GPU
results are in `results/attention_gpu/`. Both six-case experiments passed, with
90 timed calls each. Largest tiled absolute errors versus FP64 were approximately
6.91e-7 on both devices. The tiled path is slower in all measured cases; at length
512 it uses about 51.5% less peak allocated GPU memory than the dense reference.
Full measured tables, verification limits and interpretation are recorded in
`reports/attention_analysis.md`. Task 4 is complete; no model retraining is needed.
