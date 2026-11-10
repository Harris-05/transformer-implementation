# Original-model validation baseline

Evaluated **206/206** validation records. Complete: **True**.

| Group | N | Accuracy | Macro-F1 | Evidence micro-F1 | Evidence macro-F1 | Valid output |
|---|---:|---:|---:|---:|---:|---:|
| overall | 206 | 0.3738 | 0.2924 | 0.5258 | 0.7224 | 0.9903 |
| inclusion | 66 | 0.6818 | 0.4115 | 0.6446 | 0.6438 | 1.0000 |
| exclusion | 140 | 0.2286 | 0.1693 | 0.3288 | 0.7595 | 0.9857 |

## Scoring policy

- Macro-F1 averages the fixed applicable label set, including zero-support classes; __invalid__ predictions remain errors.
- Confusion matrices include an explicit invalid-prediction column. Per-class precision/recall/F1 are in metrics.json.
- Responses must be JSON with exactly label and sentence_ids; markdown and malformed responses are not repaired.
- Structurally malformed responses produce an invalid label and empty evidence; no example is dropped.
- With valid structure, label and evidence are evaluated separately. Nonexistent integer citations count as evidence false positives.
- Both gold and predicted evidence empty: precision=recall=F1=1. With only one set empty: all three are 0.
- Evidence micro scores pool sentence-set TP/FP/FN; macro scores average examples. Nonempty-gold-only macro scores are also reported.
- Timing covers synchronized generation calls only, excluding download, model loading, warm-up, and file writes.
- This is a 4-bit original-model baseline. Compare the fine-tuned model under the same quantization, prompt, and decoding settings.
- Validation results are for development. This run does not evaluate the held-out test group.
