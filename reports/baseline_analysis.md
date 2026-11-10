# Original-model validation baseline: analysis

Date: November 7, 2026.

## Verification

The attached `baseline_results.zip` contains a completed run of all 206 fixed
validation annotations. All annotation IDs are unique and match the prepared
validation file. Patient metadata, inputs, expert targets, validation-file hash,
run signatures, and the three recorded source-file hashes match this workspace.
Independent calculations reproduce the reported accuracy, macro-F1, and evidence
micro-F1 for the overall, inclusion, and exclusion groups.

The original artifacts are preserved without modification in
`results/baseline_validation/`. The attached ZIP is retained as well.

## Results

| Group | Correct / total | Accuracy | Classification macro-F1 | Evidence micro-F1 |
|---|---:|---:|---:|---:|
| Overall | 77 / 206 | 37.38% | 0.2924 | 0.5258 |
| Inclusion | 45 / 66 | 68.18% | 0.4115 | 0.6446 |
| Exclusion | 32 / 140 | 22.86% | 0.1693 | 0.3288 |

All 206 responses had the expected JSON schema. Two exclusion responses used the
inclusion-only label `not included`, producing 204/206 valid overall outputs
(99.03%). No nonexistent sentence citations or generation-limit hits were
reported. Correct citation IDs do not establish that the cited text supports the
decision; evidence retrieval still needs improvement.

## Main error pattern

Of 101 expert `not excluded` cases, only eight received that prediction. The model
instead returned `not enough information` for 58, `not applicable` for 33, and an
invalid criterion label for two. This group accounts for most classification
errors. It never predicted `excluded`, although two validation examples have that
gold label; the small support limits what can be concluded about this class.

Representative annotation 5 concerns hypertension above a specified threshold.
The note mentions hypertension but gives no blood-pressure measurement. The gold
label is `not excluded` with empty evidence, while the model returns
`not enough information` with empty evidence. Similar inspected disagreements
concern conditions not mentioned in a note.

This suggests a difference between the conservative prompt's information rules
and some benchmark annotations, rather than a formatting problem alone. It does
not prove that every model disagreement is defensible or that a larger model is
needed. Review the annotation definitions and document ambiguous examples before
attributing all errors to model capacity. Do not relabel expert targets or teach
a blanket rule equating missing information with eligibility.

## Interpreting evidence and imbalance

Overall evidence macro-F1 is 0.7224, but 116/206 records have no gold evidence and
both-empty sets receive a perfect score under the documented policy. On the 90
records with nonempty gold evidence, macro-F1 is only 0.4425. Report that figure
alongside micro-F1 0.5258 to avoid overstating retrieval quality.

Always predicting the single most frequent validation label would yield 49.03%
accuracy. Predicting each criterion type's majority label would yield 66.02%.
These are descriptive majority-label references calculated on validation, not
trained systems or held-out test results. The model's lower overall accuracy
reinforces the need for adaptation and per-class analysis.

## Measured compute

The run used the pinned Qwen3-4B-Instruct-2507 checkpoint on a Tesla T4 with 4-bit
NF4 weights, FP16 computation, greedy decoding, and no adapters. Synchronized
generation calls totaled 332.03 seconds, with 11.41 output tokens/second. Peak
PyTorch allocated memory was 2.68 GiB and reserved memory was 2.77 GiB; these are
not measurements of all GPU memory usage. Download, model loading, warm-up, and
file writes are excluded from generation time. Training memory remains unmeasured.

## Next task

Preserve this original-model baseline. Review the exclusion-label interpretation,
then prepare QLoRA fine-tuning on the 602 usable training examples. Evaluate using
the same validation examples, prompt, quantization, and decoding settings. Any
intentional prompt change needs a separately recorded baseline. Keep the final
test set reserved.
