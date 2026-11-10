# TrialScreen — Generative AI Assignment 2

**Muhammad Harris Azhar · i232535**  
National University of Computer and Emerging Sciences

Criterion-level clinical trial screening with a fine-tuned
`Qwen/Qwen3-4B-Instruct-2507`, exact patient-sentence citations, explicit cached and
uncached generation, tiled online-softmax attention and greedy speculative
decoding with `Qwen/Qwen3-0.6B`.

**Educational screening prototype; not a clinical enrollment decision.**

## Open the application

On Windows, run `Setup_TrialScreen_CPU.cmd` once if dependencies are missing,
then double-click **`Start_TrialScreen.cmd`**. The browser opens
`http://127.0.0.1:8765`. Enter a patient note and individual criteria, run the
selected model, inspect original cited sentences, export HTML/JSON, or open
Inference lab for a live decoding comparison. No notebook is needed by the
application's end user.

The selected adapter is already restored under `results/finetuning_qlora_v1`
in this workspace. Source-only distributions need the separately supplied
`finetuning_results.zip`; see `APPLICATION.md` for its exact layout.
Base/draft weights download automatically from pinned Hugging Face revisions
on first actual use. Opening the UI does not load weights.

**Resource requirement:** CPU inference uses unquantized FP32, approximately
16 GB of base weights alone; allow additional RAM for loading, caches and the
optional draft. A 24–32 GB host is a more realistic starting point. Full-size CPU
latency/fit have not been measured here. On a compatible CUDA host, install a
CUDA PyTorch build plus `requirements-app.txt`, then use `python app.py --device cuda`.
The CUDA loader uses NF4 weights with FP32 compute. See
[APPLICATION.md](APPLICATION.md) for startup, protocols, limits and recovery.

## Report and observed results

- [IEEE report PDF](report/TrialScreen_Report.pdf)
- [LaTeX source](report/TrialScreen_Report.tex)
- [Model comparison](model_comparison.md)
- [Data audit and split](DATASET.md)

| Experiment | Verified outcome |
|---|---|
| Original → adapted validation accuracy | 0.3738 → 0.7524, same 206 records |
| Original → adapted validation macro-F1 | 0.2924 → 0.4438 |
| Original → adapted evidence micro-F1 | 0.5258 → 0.6154 |
| KV math-SDPA | 54 timings complete; 8/9 stress and 3/3 clinical cases equal; one residual mismatch retained |
| Matching KV workloads | 1.42–4.17× cached speedup |
| Tiled attention | 6/6 shapes/masks pass; peak reduction approximately 51.5% at length 512, with slower runtime |
| Speculative decoding | 6/6 preflight cases and 54/54 timings pass exact equality; speed ratios 0.652–1.361× |
| Final held-out test | Pending the separate final GPU evaluation; validation is never relabeled as test |

Full-model Colab experiments are verified from their returned raw artifacts.
The browser/API and actual miniature Qwen forward integration are verified
locally. Full-size model loading/inference through the application server still
needs verification on the intended demo host. That distinction is explicit in
the report.

## Reproduce the pipeline

Use Python 3.13/3.14, install PyTorch for the chosen host, then the relevant
pinned requirements. The Colab notebooks preserve the runtime's CUDA PyTorch.
Optional incompatible `torchao` is removed in Colab because this project uses
bitsandbytes, not TorchAO.

1. `python eda.py` — audit source and fixed 37/8/8 patient split; quarantine
   training annotation 883 with missing criterion text.
2. `python prepare_model_inputs.py` — prepare development prompts and completion
   targets, lock tokenizer assets, verify no truncation.
3. `colab_baseline.ipynb` — original-model validation baseline.
4. `colab_finetuning.ipynb` — completion-only QLoRA; select the best validation-loss
   checkpoint and evaluate with the unchanged scoring functions.
5. `colab_kv_cache.ipynb` / upload-both notebook — explicit cache experiments.
   Preserve the documented numerical mismatch; do not change equality to pass.
6. `colab_attention.ipynb` — self-contained standalone attention study.
7. `colab_speculative.ipynb` — selected adapter plus compatible draft; preflight
   equality before any timing.
8. **Final evaluation:** after choices are fixed, copy all of
   `colab_run_final_evaluation.py` into one cell in the existing speculative
   runtime, or use `colab_final_evaluation.ipynb` for a new runtime. It opens the
   prescribed eight test patients only for final inference, evaluates original
   and adapted models under the same NF4/FP16 protocol, and downloads
   `final_test_results.zip`. It does not train either model.
9. Restore final artifacts to `results/final_test`, run `python assemble_report.py`,
   then compile the LaTeX twice from `report/`. Generation refuses incomplete
   test metrics. Keep all raw manifests/predictions and do not tune on test.

Source/config/data hashes, pinned revisions, all raw trials and failed checks
are retained. Notebook bundles preserve raw bytes, including original line
endings; identical text with a different byte checksum is not silently accepted.

## Tests

```text
python -m unittest discover -v
```

Tests cover patient leakage, input/target separation, strict metrics and parsing,
completion loss masks, actual miniature-model cache/speculative behavior,
rejection/bonus/EOS/cache cropping, causal/fully-masked/GQA tiled attention,
aggregation of missing/unequal trials, exact notebook restoration, adapter retry,
browser/API validation, report escaping and final-evaluation resume integrity.
They do not claim to prove clinical correctness or unmeasured hardware fit.

## Key files

| Component | Files |
|---|---|
| Data and prompt | `eda.py`, `prepare_model_inputs.py`, `configs/`, `data/splits/` |
| Adaptation/evaluation | `finetune.py`, `baseline.py`, `evaluate_adapter.py`, `evaluate_final.py` |
| Explicit KV decoding | `decoding.py`, `benchmark_kv_cache.py` |
| Educational attention | `attention.py`, `benchmark_attention.py` |
| Speculation | `speculative_decoding.py`, `speculative_inputs.py`, `benchmark_speculative.py` |
| Browser application | `app.py`, `app_model.py`, `screening.py`, `web/` |
| Reproducibility | `requirements*.txt`, `make_colab_*.py`, all manifests in `results/` |
| Paper assembly | `assemble_report.py`, `report/`, analyses under `reports/` |

See [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for publication links,
demo recording and remaining verification. Do not submit claims of completion
for pending test inference, full-size application-host verification, URLs or video.

## Attribution

Dataset: [NCBI TrialGPT Criterion Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations),
public domain. Base and draft model cards and Apache-2.0 licenses are linked in
`model_comparison.md` and `SPECULATIVE.md`. The original TrialGPT project is a
research reference, not the submitted system. Font licenses are in `web/`.
An OpenAI Codex assistant contributed implementation/debugging/report assistance;
the report includes an AI-use appendix. The student must understand and defend
the complete implementation and acknowledge reused libraries and sources.
