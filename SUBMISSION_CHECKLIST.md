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

- [ ] Run `colab_run_final_evaluation.py` in the current GPU runtime and download
  `final_test_results.zip`. Restore verified artifacts under `results/final_test`,
  regenerate the paper, and check that all final test tables replace the pending
  paragraph. Keep validation and test results separate; do not tune afterward.
- [ ] Load the full-size selected model in the actual browser application on the
  intended host and run at least one unseen record. The current local verification
  used real miniature Qwen models; it does not establish full-size CPU fit/latency.
- [ ] Publish the source in your assignment GitHub repository and insert its URL.
- [ ] Host `finetuning_results.zip` or the selected adapter files at a reachable
  download URL. Test access from a signed-out/incognito window.
- [ ] Record the required 5–7 minute application demo, host it, and insert its URL.
- [ ] Check the PDF, source project, split, logs, adapter method and video link are
  all included in the actual Classroom submission. Attend the live demo.

## Links to supply (not fabricated)

| Item | Value |
|---|---|
| GitHub source repository | Pending publication / user-provided URL |
| Report source project | `report/TrialScreen_Report.tex` in that repository |
| Selected adapter download | Pending reachable hosted URL; local artifact supplied |
| Demo video | Pending recording and video URL |

## Important distinctions for the defense

Validation accuracy is 75.24%; no held-out accuracy is claimed until final
evaluation completes. The KV grid finished timing but one correctness case failed.
Tiled attention saves memory but is slower and is not injected into Qwen.
Speculative generation is exact for all tested cases but not always faster.
Application FP32 computation is a different deployment protocol from saved FP16
classification. The overall screening outcome is a rule aggregate, not a labeled
clinical enrollment decision.
