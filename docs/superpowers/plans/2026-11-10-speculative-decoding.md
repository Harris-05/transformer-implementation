# Greedy Speculative Decoding Implementation Plan

**Goal:** Complete Task 5 with explicit drafting, batched target verification,
prefix acceptance, rejection correction, cache synchronization and measured
comparison against ordinary target greedy decoding.

**Spec:** `GenAI_Assignment#2.pdf`, Task 5 / rubric F1-F3. User explicitly asked
to start this stage after verified completion of tiled attention.

**Architecture:** A new decoder owns separate draft/target DynamicCaches. A pinned
draft-tokenizer loader verifies the configured fast tokenizers, token-ID mappings
and special IDs. A runner reuses the selected adapter and existing validation
prompts, saves preflight correctness before timing, and writes all trial outcomes.
A self-contained Colab notebook embeds code/data and locates existing adapter
files or an uploaded results ZIP without an upload widget.

## Decisions

- Draft: `Qwen/Qwen3-0.6B`, pinned revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0. Only tokenizer/config
  assets downloaded locally; no full weights. Its configured fast tokenizer and
  target's fast tokenizer were independently checked: identical backend JSON,
  vocabulary map (151669 entries), and EOS/pad/BOS IDs. Their model vocab sizes
  both equal 151936. Raw merges.txt differs; it is not used as compatibility proof.
- Target remains the selected Qwen3-4B adapter. For this controlled experiment,
  use 4-bit NF4/double-quantized target weights with **FP32** compute and math SDPA
  in both ordinary and speculative paths. Draft uses FP16/math SDPA. This controls
  sensitivity observed in the earlier FP16 KV study; it does not overwrite those
  results or claim classification equivalence to the saved FP16 baseline.
- Prefer parallel target verification over tokenwise verification: it demonstrates
  the intended reduction in expensive target forwards. Tokenwise verification
  would give up that mechanism and is not the default.
- No token repair, relaxed equality or silent fallback. Every measured output
  must match ordinary target greedy; mismatches suppress speedup and remain failures.
- At a round boundary both caches describe committed tokens **except the last**.
  Draft the next K tokens, verify the pending suffix plus candidates together,
  accept only the matching prefix and emit target correction/bonus. Crop caches
  with the pinned API's negative removal count. If necessary, process the final
  accepted draft token to synchronize the draft cache before the next round.
- EOS and remaining-token budget constrain proposals/emissions. A round emits
  at least one token. Cache state must not retain rejected candidates.
- Tests use actual tiny Qwen3 pairs plus deterministic causal models with real
  DynamicCache storage to force all-accept, zero-accept and middle rejection.
- Comparison uses three full clinical prompts and three fixed-work stress cases,
  draft lengths 2/4, warm-up and three repetitions. Keep all models resident for
  both paths and label memory accordingly. GPU measurement still requires Colab.
- Run full preflight first to avoid spending GPU time timing unequal outputs.

## Tasks

- [x] Write failing decoder/cache/EOS/budget/isolation tests, then implement
  `speculative_decoding.py` and verify real tiny-model greedy equivalence.
- [x] Implement `speculative_inputs.py`, draft asset lock/compatibility report,
  immutable pair configuration and offline preparation checks.
- [x] Implement `benchmark_speculative.py`, report aggregation and meaningful
  tests for missing/unequal trials, acceptance and speed/memory summaries.
- [x] Build `colab_speculative.ipynb` with embedded code and validation inputs,
  automatic existing-file/ZIP adapter discovery, actual paired-model load,
  preflight checks, repeated benchmark and downloadable artifacts.
- [x] Run full suite and CPU smoke, validate notebook/locks/data isolation,
  perform fresh final review, document sources/limitations and next user action.

## Verification and handoff

- Full existing suite: 71 tests passed (`artifacts/speculative_full_suite.log`).
  Two subsequent notebook retry/backup regressions failed for the discovered
  `Path(None)` bug, then passed after both guards were corrected: 73 verified tests
  across those runs. No decoder or benchmark source changed after the full suite.
- Final miniature CPU smoke: 18/18 timed trials complete, strict equality against
  ordinary greedy, HF preflight passed; recorded source hashes match final files.
- Offline actual tokenizer/adapter preparation: six cases, 151669 compatible
  token mappings, both model vocabulary sizes 151936, pinned asset hashes verified.
- Regenerated notebook: schema and all five code cells valid; 24 embedded files
  match sources/hashes; all 206 examples belong only to the eight validation
  patients. Frozen EDA/preparation/baseline/training/KV decoder hashes preserved.
- Fresh final review found failed-preflight overwrite and missing-file retry
  issues. Both were reproduced with regressions, corrected and verified.
- **Completed:** returned `speculative_results.zip` independently verified and
  preserved in `results/speculative_v1`. Six preflight cases and 54/54 timings
  pass exact equality; measured ratios 0.652–1.361x. All aggregates/source identities
  recomputed. See `reports/speculative_analysis.md`.

## Constraints

Do not change the frozen baseline, model-input preparation, fine-tuning, KV decoder
or existing results. Do not rerun the closed KV investigation. Patient test group
stays reserved. Existing enclosing Git root is the user's home: no broad git
operations or unrelated commits. Follow established direct implementation and
one fresh final review; no redundant user approval pause.
