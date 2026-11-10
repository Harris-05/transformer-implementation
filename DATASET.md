# Dataset audit and preparation

Run from this folder:

```powershell
python -m pip install -r requirements-data.txt
python eda.py
python -m unittest discover -v
```

The existing dataset is read without modification. Default paths are resolved
relative to `eda.py`, so the script also works when launched from another folder.
For a different location:

```powershell
python eda.py --data path/to/dataset.parquet --output path/to/artifacts
```

## Outputs

| File | Contents |
|---|---|
| `reports/data_audit.md` | Readable findings, label distributions, lengths, and next steps |
| `reports/data_audit.json` | Detailed checks, package versions, and one evidence example per development label/type |
| `data/splits/split_manifest.json` | Source SHA-256, split algorithm, patient IDs and annotation counts |
| `data/processed/train.jsonl` | Usable supervised training records |
| `data/processed/validation.jsonl` | All validation records, for baseline and model selection |
| `data/processed/quarantined_train.jsonl` | Invalid training records and explicit exclusion reasons |

The split is made by sorting the 53 unique patient IDs, shuffling with NumPy
`default_rng(42)`, and slicing 37/8/8. A differing existing manifest is never
overwritten. Re-running with the same file and package versions is deterministic.

## Findings for the downloaded file

- 1,015 annotations, 53 patients, 14 columns.
- Training: 37 patients, 603 source annotations, 602 usable annotations.
- Validation: 8 patients, all 206 annotations retained.
- Test: 8 patients, 206 annotations reserved; content is not explored or exported.
- Training annotation **883** has null `criterion_text`. It remains in the source
  and split manifest, but is quarantined from supervised training because the
  criterion cannot be reconstructed from permitted inputs. No label is invented.
- Training contains 349 valid empty evidence lists; validation contains 116.
  These examples are retained. Empty evidence is distinct from malformed evidence
  and is never interpreted as automatic eligibility or ineligibility.
- All development evidence IDs refer to existing numbered patient sentences.
- No repeated patient/trial/type/criterion keys or inconsistent patient notes were
  found within either development split.
- Labels are imbalanced: training includes only eight `excluded` examples and five
  inclusion `not applicable` examples. Use macro-F1 and per-class metrics.

The audit exits successfully with `REVIEW WARNINGS` for the documented training
quarantine. Validation errors stop preparation; validation examples are never
silently excluded to improve metrics. The script does not inspect test labels,
notes or evidence. Global checks are limited to schema, source identity, patient
IDs and annotation IDs/counts.

## Training-record contract

Each JSONL line has exactly three groups:

```json
{
  "input": {
    "note": "0. Patient is 58 years old.",
    "criterion_type": "inclusion",
    "criterion_text": "Age 18 to 65"
  },
  "target": {"label": "included", "sentence_ids": [0]},
  "metadata": {
    "annotation_id": 123,
    "patient_id": "example",
    "trial_id": "example-trial",
    "trial_title": "Example trial"
  }
}
```

Only `input` may be included in the model prompt. `target` is expert supervision;
`metadata` is for traceability. GPT-4 predictions/explanations, explanation
correctness, and the existing `training` flag are not exported. The dataset
does not provide an expert-written explanation target: do not reuse GPT-4
explanations as if they were expert rationale.

## Next step

Select a target model based on available hardware, then build its prompt/chat
template using only the three permitted inputs. Measure actual tokenizer lengths,
including the chat template and output allowance, and check whether any evidence
would be truncated. Run the original model on validation data to establish label,
evidence, and output-format baselines before fine-tuning. Keep test cases reserved
for the final fixed comparison.
