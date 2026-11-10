# QLoRA validation results

## Artifact verification

The uploaded `finetuning_results.zip` contains the selected adapter, training
history and manifests, and all 206 validation predictions. Original contents are
preserved in `results/finetuning_qlora_v1/`; the ZIP is retained.

Verified unique annotation IDs, patient metadata, input fields, expert targets,
raw-response parsing, run signatures, train/validation data hashes, source hashes,
and every recorded adapter-file hash. Recomputing all classification and evidence
groups with the frozen baseline scoring functions reproduces the saved metrics
and comparison. Extracted files match the archive byte for byte. The five
fine-tuning unit tests also pass. These checks do not rerun GPU inference.

The original and adapter runs use the same pinned Qwen3-4B-Instruct-2507 revision,
prompt, validation examples, 4-bit quantization and greedy decoding protocol.
The adapter evaluation manifest records matching shared dependency versions.
The held-out test group has not been evaluated.

## Comparison

| Group | Correct before / after | Accuracy before / after | Macro-F1 before / after | Evidence micro-F1 before / after |
|---|---:|---:|---:|---:|
| Overall (206) | 77 / 155 | 0.3738 / 0.7524 | 0.2924 / 0.4438 | 0.5258 / 0.6154 |
| Inclusion (66) | 45 / 49 | 0.6818 / 0.7424 | 0.4115 / 0.3857 | 0.6446 / 0.7288 |
| Exclusion (140) | 32 / 106 | 0.2286 / 0.7571 | 0.1693 / 0.4192 | 0.3288 / 0.4062 |

Overall accuracy improves by **37.86 percentage points**. All 206 adapted outputs
are valid, compared with 204 original outputs, with no nonexistent citations.
Accuracy also exceeds the existing criterion-type majority reference of 66.02%.

## Remaining weaknesses

- `not included`: 0/3 correct, compared with 1/3 originally. Its inclusion F1
  falls from 0.1667 to zero. This outweighs the improvement in inclusion
  `not enough information` F1 from 0.6667 to 0.7302 in the equal-class average;
  `included` F1 remains 0.8125. This explains the inclusion macro-F1 decrease.
- `excluded`: 0/2 correct; both are predicted `not excluded`. The model never
  predicts `excluded` in this validation run. These small supports limit what
  can be concluded about performance on these classes.
- Most classification improvement comes from `not excluded`: 95/101 correct,
  compared with 8/101 originally. Eighteen of 26 exclusion examples labelled
  `not enough information` are also predicted `not excluded`.
- Evidence precision is 0.8358, but recall is 0.4870: the model retrieves 56 gold
  sentence citations and misses 59. Exclusion evidence recall is only 0.2708.
  Nonempty-gold evidence macro-F1 improves from 0.4425 to 0.5685. Report it
  alongside overall evidence macro-F1 (0.7872), which includes 116 examples
  with empty gold evidence and awards a perfect score when both sets are empty.
- The conservative prompt versus expert annotation convention discussed in
  `reports/exclusion_label_review.md` remains relevant. Higher annotation agreement
  alone does not establish that the model follows the missing-information rule.

## Training and checkpoint selection

Training uses 602 examples, three epochs and 228 optimizer steps, with
33,030,144 trainable parameters (approximately 0.8145% of the model).
Recorded training runtime is 2,707.25 seconds (45.12 minutes), with peak GPU
allocated memory 6.03 GiB and reserved memory 9.73 GiB on a Tesla T4.

| Epoch | Validation completion loss |
|---|---:|
| 1 | 0.10681445 |
| 2 | 0.10694941 |
| 3 | 0.12950735 |

The selected checkpoint is **checkpoint-76, at the end of epoch 1**, chosen by
lowest validation completion loss. Later training loss decreases while validation
loss eventually rises, which is consistent with overfitting. The selected
adapter was reloaded before evaluation. The `epoch: 3.0` field in
`selected_validation` reflects the final Trainer state; it does not change which
checkpoint was selected. Selection was by completion loss, not classification F1.

The generation warnings did not prevent completion. Preserve these artifacts
and the frozen protocol for subsequent comparisons.

## Next implementation task

Implement manual greedy generation with and without a KV cache. Check token
agreement under the same prompt and decoding settings, then benchmark synchronized
generation time and memory on the same hardware. Use development examples and
keep the held-out test group reserved until the final system is fixed.
