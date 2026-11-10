# Independently verified final held-out assessment

`final_test_results.zip` is preserved without changes under `results/final_test`.
All 206 predictions per model belong to exactly the prescribed eight held-out
patients. Reconstructed the original test inputs and expert targets only for
this final verification, after model/prompt/checkpoint choices were fixed.
Verified original dataset checksum, canonical record hash, selected adapter and
training signature, all evaluator/scorer source hashes, prediction signatures,
input/gold metadata, strict raw-response re-parsing and extracted evidence text.
Recomputed every classification/evidence group and before/after delta exactly.
No GPU inference or tuning was performed during artifact verification.

| Group | N | Before accuracy | After accuracy | Before macro-F1 | After macro-F1 | Before evidence F1 | After evidence F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 206 | 0.4078 | 0.7427 | 0.3492 | 0.4733 | 0.4571 | 0.5714 |
| Inclusion | 75 | 0.7067 | 0.7600 | 0.5815 | 0.5802 | 0.5524 | 0.6847 |
| Exclusion | 131 | 0.2366 | 0.7328 | 0.1905 | 0.3508 | 0.1714 | 0.2222 |

Both phases use NF4/double quantization, FP16 computation, automatic SDPA,
the frozen prompt and greedy decoding. The original phase disables the LoRA
adapter on the same frozen base; the adapted phase enables the selected adapter.
Both phases have 100% output validity, which is distinct from clinical correctness.

Correct classifications increase from 84 to 153, a 33.50 percentage-point gain.
Much of it comes from not excluded: 2/91 → 89/91 correct. Not included falls
3/7 → 1/7, not applicable falls 11/15 → 3/15, and included stays 20/27. Inclusion
macro-F1 slightly decreases despite increased inclusion accuracy. There are **zero
gold excluded examples** in this test group, preventing an excluded-sensitivity
estimate. Fixed macro-F1 retains that zero-support class; it is not silently removed.

Adapted evidence precision is 0.7925 and recall 0.4468: 42 gold citations retrieved,
11 false positives and 52 false negatives. Evidence macro-F1 is 0.8343, but 145
of 206 records have empty gold evidence. Restricted to 61 nonempty-gold records,
macro-F1 is 0.5553. Exclusion evidence recall remains only 0.1250.

Eight final error cases in `reports/final_test_error_cases.json` cover negative
inclusion mistakes, unsupported resolution of missing information, not-applicable
mistakes and incomplete evidence despite matching categories. Their analysis is
reporting after final evaluation, not feedback for another tuning cycle.
