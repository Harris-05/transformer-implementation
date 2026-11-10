# Fine-tuned model: validation comparison

Completed: **True**; evaluated **206/206** records.

| Group | Metric | Original | Fine-tuned | Change |
|---|---|---:|---:|---:|
| overall | accuracy | 0.3738 | 0.7524 | +0.3786 |
| overall | macro_f1 | 0.2924 | 0.4438 | +0.1513 |
| overall | evidence_micro_f1 | 0.5258 | 0.6154 | +0.0896 |
| overall | output_valid_rate | 0.9903 | 1.0000 | +0.0097 |
| inclusion | accuracy | 0.6818 | 0.7424 | +0.0606 |
| inclusion | macro_f1 | 0.4115 | 0.3857 | -0.0258 |
| inclusion | evidence_micro_f1 | 0.6446 | 0.7288 | +0.0842 |
| inclusion | output_valid_rate | 1.0000 | 1.0000 | +0.0000 |
| exclusion | accuracy | 0.2286 | 0.7571 | +0.5286 |
| exclusion | macro_f1 | 0.1693 | 0.4192 | +0.2499 |
| exclusion | evidence_micro_f1 | 0.3288 | 0.4062 | +0.0775 |
| exclusion | output_valid_rate | 0.9857 | 1.0000 | +0.0143 |

Values and changes are on a 0-1 scale. Negative changes are retained.
The original 4-bit baseline and the fine-tuned model use the same prompt, validation examples, quantization and greedy decoding.
Metrics use the original baseline's scoring functions, applicable label sets and empty-evidence policy.
Invalid responses stay in all evaluation denominators. No output is repaired.
Per-class results, evidence scores restricted to nonempty gold, and confusion matrices are in metrics.json.
This is validation evaluation; the held-out test group remains reserved.
