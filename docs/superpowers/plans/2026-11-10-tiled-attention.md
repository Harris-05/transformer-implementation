# Tiled Attention Implementation Plan

**Goal:** Complete Task 4's explicit dense/tiled attention implementations,
numerical checks and reproducible performance comparison.

**Spec:** `GenAI_Assignment#2.pdf`, Task 4 / rubric E1-E3. User requested moving
on after closing the KV-cache numerical investigation with its limitation retained.

**Architecture:** Two independent inference-only PyTorch forward implementations
operate on `[batch, heads, tokens, head_dim]` Q/K/V tensors. A standalone runner
compares them with PyTorch SDPA on seeded controlled tensors and saves errors,
repeated timings and measured GPU memory. A self-contained Colab notebook embeds
the small source files and requires no dataset, adapter or ZIP upload.

**Approach:** Dense scores with stable row softmax are the reference. Query/key
tiling with per-row running maxima, sums and unnormalized weighted values avoids
the full score matrix. FP16/BF16 inputs accumulate in FP32; FP64 stays FP64.
This is an educational PyTorch algorithm, with actual speed measured rather than
inferred from the fused FlashAttention implementation in the cited paper.

## Constraints and decisions

- No further KV-cache reruns. Preserve the failed GPU equality flag and original
  artifacts. Verify the uploaded math-SDPA results independently before using them.
- Preserve frozen model preparation, baseline, fine-tuning and decoder sources.
- Attention supports causal/full masks, absolute query offsets for rectangular
  cached-query shapes, grouped-query heads and a distinct value head dimension.
- Fully masked rows return zeros. Invalid shapes/dtypes/devices/tiles fail clearly.
- No `scaled_dot_product_attention` call inside either custom implementation;
  library attention is an independent validation/performance comparator only.
- Benchmark CPU locally, GPU when available. Never label theoretical score-tile
  sizes as measured process/peak memory. CPU native tensor peak memory is null.
- Fixed seed, explicit shapes/dtype/threads, two warmups, five timed repetitions,
  synchronized CUDA measurements. All repetitions and numerical errors retained.

## Tasks

- [x] Verify and preserve the received KV archive, write its measured analysis.
- [x] Write failing `test_attention.py`: hand-calculated values; future-token
  masking; awkward tile sizes; all-masked rows; offsets; GQA; high-magnitude input;
  low precision and FP64; independent PyTorch math SDPA agreement; invalid inputs.
- [x] Implement `attention.py:reference_attention` and `tiled_attention`; run tests.
- [x] Implement `benchmark_attention.py` and `configs/attention_config.json`:
  correctness checks, raw repeated timings, summary JSON/CSV/PNG and source/runtime
  provenance. Write meaningful aggregation/completeness tests before that logic.
- [x] Run the complete local CPU experiment. Create `ATTENTION.md` and a generated
  self-contained `colab_attention.ipynb` through `make_colab_attention.py`.
- [x] Validate notebooks, run full tests, obtain a fresh final review, and record
  results/limitations. Return the next concrete Colab step without retraining.

## Review focus

Rescaling old accumulators when a later tile has a larger maximum; masking within
boundary tiles; initially/all-masked rows producing NaN; preserving FP64; causal
offsets in cached rectangular shapes; conflating score-workspace estimates with
measured memory; calling unsupported/failing benchmark checks a pass.

## Execution notes

Implement inline in the existing assignment directory. The enclosing Git root is
the user's home directory: no broad status, branching or unrelated commits.
The established assignment/pipeline and explicit request authorize implementation;
do not ask for redundant design approval. Perform a fresh review at the end.

## Verification ledger

- Uploaded math KV archive verified independently: code/data/adapter identity,
  all 54 raw trials, token comparisons and every aggregate metric reproduced.
  Preserved in `results/kv_cache_math_v1`; residual failure index 43 retained.
  The additional `kv_cache.json` belongs to another model and is not mixed in.
- Watched attention and benchmark tests fail before implementing their modules.
  All **59** project tests pass, including eight mathematical attention tests and
  six benchmark/report tests. Existing temporary-folder tests ran with Windows
  sandbox escalation; no tests or failure flags were disabled.
- Review found a completed-metrics/missing-report recovery defect and a config
  validation gap. Both reproduced with failing regressions, then fixed. Saved
  metrics regenerate missing reports without repeating measurements. Empty or
  duplicated workloads cannot pass as a complete experiment.
- Final CPU experiment: six cases, all numerical checks pass, all **90** timed
  calls saved in `results/attention_cpu_final`. Largest tiled absolute error vs
  FP64 is 6.910331324938568e-7. Tiled runtime is slower in all six CPU cases;
  slowdowns retained. CPU peak memory is unmeasured, not fabricated.
- Self-contained Colab notebook schema, all three code cells and all seven
  embedded source hashes verified. Fresh final review reports no blockers.
- Actual GPU attention timings/memory remain pending the user-run notebook.
  No adapter, tokenizer, patient data or model weights need uploading for it.

## GPU closure

Received `attention_results.zip`: six cases, 90 unique timed calls, complete and
passing. Verified run/source/config identity and reproduced every timing and
memory aggregate. All reported numerical errors are finite and below tolerance;
maximum tiled absolute error is 6.91226735405337e-7. Raw artifacts are preserved
in `results/attention_gpu/`. GPU numerical outputs were not recomputed locally.

Tesla T4 measured dense peak allocation at length 512 is 21.65625 MiB, compared
with tiled 10.5048828125 MiB (about 51.5% lower). The tiled implementation is
slower in every CPU and GPU case; this limitation is retained. Task 4 is complete
with actual GPU evidence and no speedup claim. Next pipeline stage: greedy
speculative decoding with a compatible smaller draft model.
