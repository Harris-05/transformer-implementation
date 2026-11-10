# Exclusion-label review before fine-tuning

The original validation baseline remains frozen. Its main disagreement is the
expert `not excluded` class: 8/101 correct, with 58 predictions of insufficient
information and 33 of not applicable.

## Reference interpretation

The [original TrialGPT matching prompt](https://github.com/ncbi-nlp/TrialGPT/blob/main/trialgpt_matching/TrialGPT.py)
allows contextual assumptions when a medically important finding would normally
appear in a note. Our assignment prohibits automatically treating absent evidence
as proof of eligibility or ineligibility, so our frozen prompt uses a more
conservative information rule. This difference plausibly explains some benchmark
disagreements. It does not establish that every original prediction is correct.

## Representative validation errors

| Annotation | Criterion or issue | Expert label | Baseline label | Observation |
|---|---|---|---|---|
| 0 | Chronic stable angina and imaging requirement | not enough information | not included | The model turns uncertainty into a negative inclusion judgment and misses annotated evidence. |
| 5 | Hypertension above a specified threshold | not excluded | not enough information | The note reports hypertension but no numerical pressure; this illustrates the information-policy tension. |
| 13 | Type-1 diabetes | not excluded | not applicable | The patient denies diabetes; evidence sentence 4 supports a negative exclusion finding. |
| 15 | Pregnant and nursing women | not enough information | not applicable | The model selects inapplicability without the expert's support. |
| 24 | Ischemic disease and PCI timing/premise | not applicable | not enough information | A conditional criterion's applicability needs to be distinguished from uncertainty. |
| 41 | Respiratory distress | excluded | not excluded | The model cites the same sentence as the expert but assigns the opposite label; localization alone is insufficient. |
| 252 | Uterine size threshold | not excluded | invalid: not included | The response uses an inclusion-only category for an exclusion criterion. |
| 263 | Uterine pathology other than leiomyoma | excluded | not enough information | The model misses the expert evidence sentence and the positive exclusion judgment. |

These eight distinct error examples come only from the permitted validation set.
The detailed notes, criterion text, and predictions remain in
`results/baseline_validation/predictions.jsonl`; the expert annotations are not
rewritten based on this review.

## Training decision

Proceed with supervised adaptation to the original expert labels and evidence
IDs. Keep the exact prompt, inclusion/exclusion vocabularies, quantization, and
greedy decoding for the before/after comparison. Train on the 602 usable training
records, use validation for checkpoint selection, and keep test cases reserved.
Do not create an automatic missing-information-to-`not excluded` rule. Report
annotation ambiguity and the prompt-policy tension as limitations even if the
fine-tuned benchmark scores improve.
