# Submission checklist

Author: Muhammad Harris Azhar, i232535  
National University of Computer and Emerging Sciences

## Prepared locally

- [x] Original dataset, audit, source checksum and prescribed patient split.
- [x] Leakage-safe prompts/targets, no prior GPT outputs as input features.
- [x] Model comparison, pinned base/draft revisions and credible research sources.
- [x] Executable QLoRA, selected adapter, logs, completion-mask tests.
- [x] Before/after validation predictions, detailed metrics and confusion matrices.
- [x] Explicit cached/uncached decoding and honest residual mismatch analysis.
- [x] Dense/tiled online-softmax attention, GPU/CPU agreement and trade-off results.
- [x] Manual speculative algorithm, exact equality and 54 real GPU timings.
- [x] Local browser app: inputs, citations, explanations, preliminary rules,
  live comparisons, HTML/JSON export, launcher and documented startup.
- [x] Real miniature-model integration, API/export tests, desktop/mobile browser checks.
- [x] IEEE LaTeX report PDF/source, figures/tables, eight observed validation errors,
  limitations, source references and AI-use appendix.
- [x] Self-contained final-evaluation notebook and one-cell existing-runtime helper.
- [x] Five-to-seven-minute demo script and technical defense notes.

## Must finish before calling the submission complete

- [x] Final test ZIP independently verified and preserved under `results/final_test`:
  206 before + 206 after predictions from the reserved eight patients. Inputs,
  expert targets, signatures, adapter/source hashes and every metric recomputed.
  Accuracy 0.4078 → 0.7427, adapted macro-F1 0.4733 and evidence F1 0.5714.
  Detailed final tables, confusion matrix and eight held-out errors added to paper.
- [ ] Load the full-size selected model in the actual browser application on the
  intended host and run at least one unseen record. The current local verification
  used real miniature Qwen models; it does not establish full-size CPU fit/latency.
- [x] Repository URL supplied: https://github.com/Harris-05/transformer-implementation
  Confirm its files match the final source archive.
- [x] Actual selected adapter is included as `finetuning_results.zip` in the
  complete submission archive. If distributing source alone through GitHub,
  also supply a reachable hosted download link and test signed-out access.
- [x] Recorded demonstration URL supplied: https://youtu.be/-zL3ECuJL1A
  Confirm it is accessible to the evaluator and meets the 5–7 minute requirement.
- [ ] Check the PDF, source project, split, logs, adapter method and video link are
  all included in the actual Classroom submission. Attend the live demo.

## Links to supply (not fabricated)

| Item | Value |
|---|---|
| GitHub source repository | https://github.com/Harris-05/transformer-implementation |
| Report source project | `report/TrialScreen_Report.tex` in the repository and source archive |
| Selected adapter | Actual `finetuning_results.zip` included in the complete submission archive; add a hosted link if needed |
| Demo video | https://youtu.be/-zL3ECuJL1A |

## Important distinctions for the defense

Validation accuracy is 75.24%; held-out accuracy is 74.27%. The KV grid finished
timing but one correctness case failed.
Tiled attention saves memory but is slower and is not injected into Qwen.
Speculative generation is exact for all tested cases but not always faster.
Application FP32 computation is a different deployment protocol from saved FP16
classification. The overall screening outcome is a rule aggregate, not a labeled
clinical enrollment decision.
