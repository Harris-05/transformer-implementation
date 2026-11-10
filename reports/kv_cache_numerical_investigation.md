# KV-cache GPU mismatch: preliminary investigation

Based on the user's pasted Colab output; the full GPU results archive has not
yet been independently inspected. Original results must remain unchanged.

## Observations

- All 54 timed trials finished. Six of nine stress cases and all three full
  clinical correctness cases passed.
- The three 256-token stress cases diverge at output index 5 (the sixth token).
  They share the same input and first five generated tokens, so this is the
  same early divergence at three output budgets, not three distinct triggers.
- Manual cached generation matches the HF cached reference for these cases.
  The uncached path differs from both at index 5.
- At the common 261-token prefix, uncached logits favor token 8720 (15.7109375)
  over token 3255 (15.6953125), a margin of 0.015625.
- Cached logits for both tokens equal 15.7265625. The largest observed absolute
  logit difference between paths is 0.0390625, larger than the uncached margin.

PyTorch argmax selects the first maximum in vocabulary order, so these diagnostic
logits imply token **3255** for the cached tie and **8720** for the uncached path.
The topk listing starts with 8720 in both paths; this listing is not evidence
that argmax selects the same token when scores tie.

## Interpretation and limits

These observations strongly suggest a numerical near-tie changing the greedy
choice. They do not establish which operation caused the logit discrepancy, or
prove that cache indexing/masking is correct under every condition. CPU tests,
clinical checks and HF agreement support the implementation but cannot replace
the failed GPU equality checks. Do not repair tokens, change tie-breaking rules,
loosen equality, drop cases or mark this run as passing.

The subsequent controlled experiment repeated only p256-o16 with automatic SDPA and
forced math SDPA on the same loaded 4-bit/FP16 adapter. Both manual paths and HF
use the same selected backend in each variant. This isolates backend choice while
retaining the quantization and compute dtype. Math attention can still leave
differences from quantized linear operations or other shape-dependent arithmetic;
success was not guaranteed. No FP32-compute control was performed.

`colab_check_kv_numerics.py` provides a copyable Colab cell. It saves a separate
`numerical_diagnostic.json`, preserving original metrics/trials and recording
model/data identity, actual tokens, backend variants and runtime details. These
are correctness diagnostics, not speed benchmarks or a substitute for the final
required equivalence checks. The focused Colab outcome is recorded below.

## Focused Colab control result

The user subsequently ran `colab_check_kv_numerics.py` on the same selected
adapter and p256-o16 prompt. Automatic SDPA reproduced the mismatch at output
index 5: cached/HF selected 3255, uncached selected 8720. With forced math SDPA,
cached, uncached and HF agreed on all 16 generated tokens, including token 3255
at index 5. The original scores and diagnostic remained unchanged.

This intervention isolates attention-backend choice as a trigger of the observed
case's discrepancy, consistent with its numerical near-tie. It does not identify
the exact fused kernel or prove agreement on the other workloads. A full 54-trial
math-SDPA rerun plus all clinical checks is required before claiming that this
backend satisfies the complete GPU token-equivalence experiment.

The benchmark now supports `--sdpa-backend math` and fingerprints that backend
choice. Automatic selection remains available and is the default. The one-cell
`colab_rerun_kv_math.py` performs the controlled run in a separate output folder.
No token repairs or test exclusions have been introduced.

Local verification of the backend option: all **45** unit tests passed; a complete
math-SDPA miniature CPU smoke run passed all nine cases and 36 timed trials.
The signed run manifest records `settings.sdpa_backend: math`. These checks do
not establish full-model GPU equivalence. Independent review found no remaining
blockers in backend scope, run provenance or the verified one-cell source update.

## Full math-SDPA run and closure (November 10)

The user supplied the full math-SDPA log. All **54 timed trials finished**.
Correctness passed for **8/9 stress cases** and **3/3 complete clinical prompts**.
The previously failing 256-token cases now pass at every budget. The p512-o64
stress case fails token equivalence under math SDPA, although p512-o16 and
p512-o32 pass. The subsequently uploaded archive locates the remaining divergence
at output index 43 (the 44th token). Do not assume that the earlier p256 numerical
tie also explains this case. The final RuntimeError reflects the retained correctness failure rather
than a failure to finish the timed trials.

At the user's explicit request, further reruns and numerical investigation stop
here, and the KV-cache implementation/experiment stage is **closed with a known
limitation**. This is not a claim of universal GPU token agreement or full
satisfaction of the assignment's equality criterion. Keep all failure flags,
case records and original results intact. Do not report a validated speedup for
the unequal p512-o64 workload; retain its raw timings as observations only.

The report should distinguish automatic SDPA (6/9 stress checks passed) from
math SDPA (8/9 passed), retain both timing protocols, and state that both runs
passed the three clinical checks. The focused control explains the original
p256 discrepancy's backend sensitivity; the residual p512-o64 discrepancy
remains unresolved.

The subsequently uploaded `kv_cache_results.zip` contains the math-SDPA run
only. Its source/data/adapter identity, all trial tokens and every aggregate
metric were independently verified and preserved in `results/kv_cache_math_v1`.
Measured timings and memory are in `reports/kv_cache_analysis.md`. Original
automatic-SDPA and focused-control results remain based on pasted logs; their
archives were not included in this upload. `colab_download_kv_results.py` can
export both Drive folders and the available numerical diagnostic together.

## Official sources

- [PyTorch argmax tie semantics](https://docs.pytorch.org/docs/2.15/generated/torch.argmax.html)
- [Numerical differences between batched/sliced operations](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html#batched-computations-or-slice-computations)
- [SDPA backends and math precision](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)
