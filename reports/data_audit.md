# TrialGPT dataset audit

Status: REVIEW WARNINGS

Source SHA-256: `14c547184fac9b013cd17441fd028845b40150c76b022e58334f5494077e5234`

Rows: 1015; patients: 53; columns: 14.

## Fixed patient split

| Split | Patients | Annotations |
|---|---:|---:|
| train | 37 | 603 |
| validation | 8 | 206 |
| test | 8 | 206 |

Test notes, labels, and evidence were not explored or exported.

## Development data

### train

| Criterion type | Expert label | Count |
|---|---|---:|
| exclusion | excluded | 8 |
| exclusion | not applicable | 23 |
| exclusion | not enough information | 65 |
| exclusion | not excluded | 292 |
| inclusion | included | 88 |
| inclusion | not applicable | 5 |
| inclusion | not enough information | 109 |
| inclusion | not included | 13 |

Missing/blank cells: 1.
Empty evidence lists: 349 (retained as valid).
Rows with repeated patient/trial/criterion keys: 0.
Conflicting duplicate groups: 0.
Patients with inconsistent notes: 0.
Validation errors: 1.

| Length measure | Min | Median | P95 | Max |
|---|---:|---:|---:|---:|
| note_characters_per_annotation | 370.0 | 582.0 | 1441.0 | 1447.0 |
| note_words_per_unique_patient_note | 58.0 | 91.0 | 172.8 | 235.0 |
| criterion_characters | 0.0 | 57.0 | 220.0 | 969.0 |
| patient_sentence_count_per_valid_annotation | 3.0 | 6.0 | 12.0 | 14.0 |
| evidence_sentence_count_per_valid_annotation | 0.0 | 0.0 | 3.0 | 9.0 |

- Annotation 883: Missing or blank criterion_text.
### validation

| Criterion type | Expert label | Count |
|---|---|---:|
| exclusion | excluded | 2 |
| exclusion | not applicable | 11 |
| exclusion | not enough information | 26 |
| exclusion | not excluded | 101 |
| inclusion | included | 35 |
| inclusion | not applicable | 1 |
| inclusion | not enough information | 27 |
| inclusion | not included | 3 |

Missing/blank cells: 0.
Empty evidence lists: 116 (retained as valid).
Rows with repeated patient/trial/criterion keys: 0.
Conflicting duplicate groups: 0.
Patients with inconsistent notes: 0.
Validation errors: 0.

| Length measure | Min | Median | P95 | Max |
|---|---:|---:|---:|---:|
| note_characters_per_annotation | 518.0 | 756.0 | 824.0 | 824.0 |
| note_words_per_unique_patient_note | 86.0 | 112.0 | 127.6 | 129.0 |
| criterion_characters | 7.0 | 57.0 | 243.2 | 679.0 |
| patient_sentence_count_per_valid_annotation | 5.0 | 7.0 | 9.0 | 9.0 |
| evidence_sentence_count_per_valid_annotation | 0.0 | 0.0 | 2.0 | 4.0 |


## Interpretation and next steps

- Preserve the original labels, sentence numbering, and empty evidence lists.
- Rare labels can make accuracy misleading; use macro-F1 and per-class metrics.
- Repeated patient notes across different criteria are expected; the split groups patients.
- Repeated trial criteria across different patients are expected in this benchmark.
- Review duplicate conflicts manually; this script never edits the source or relabels records.
- Training quarantine: [883]. These IDs stay in the split manifest but are excluded from training exports with recorded reasons.
- Exported development rows: {'train': 602, 'validation': 206}. Validation is never silently filtered.
- GPT-4 outputs, explanation correctness, and the existing training flag are excluded from exports.
- Lengths here are characters/words, not tokens. Select a tokenizer before choosing context limits.
- Next: select the model, build a prompt, measure token lengths/evidence truncation, and run a validation baseline.
- Full machine-readable details and evidence examples are in data_audit.json.
