# Final held-out test evaluation

Complete: **True**. Expected 206 records per model.

| Group | Model | Accuracy | Macro F1 | Evidence micro P/R/F1 | Evidence macro F1 | Valid |
|---|---|---:|---:|---|---:|---:|
| overall | before | 0.4078 | 0.3492 | 0.6957/0.3404/0.4571 | 0.8127 | 1.0000 |
| inclusion | before | 0.7067 | 0.5815 | 0.6744/0.4677/0.5524 | 0.6668 | 1.0000 |
| exclusion | before | 0.2366 | 0.1905 | 1.0000/0.0938/0.1714 | 0.8962 | 1.0000 |
| overall | after | 0.7427 | 0.4733 | 0.7925/0.4468/0.5714 | 0.8343 | 1.0000 |
| inclusion | after | 0.7600 | 0.5802 | 0.7755/0.6129/0.6847 | 0.7130 | 1.0000 |
| exclusion | after | 0.7328 | 0.3508 | 1.0000/0.1250/0.2222 | 0.9038 | 1.0000 |

Same frozen prompt, original expert categories, 4-bit NF4/double quantization, FP16 compute, automatic SDPA and greedy decoding for both phases.
The original phase disables the selected LoRA adapter without changing the frozen base weights. The after phase enables the selected adapter.
Invalid responses stay in every denominator. No response repair or truncation. Both evidence sets empty scores one; one empty scores zero. Nonempty-gold-only macro scores remain in metrics.json.
No tuning or checkpoint selection uses these final test scores.
