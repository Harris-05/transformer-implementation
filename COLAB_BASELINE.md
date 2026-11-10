# Run the original-model baseline in Colab

## Files to use

- `colab_baseline.ipynb`: upload this notebook to Colab.
- `colab_baseline_bundle.zip`: upload this ZIP when the notebook asks.

Both files are already generated. To rebuild them after changing project code:

```powershell
python make_colab_baseline.py
```

The bundle includes the validation prompts/targets, model and split configuration,
runner, evaluator tests, and dependency files. It contains 206 validation examples
and no training rows, test records, or model weights. The split manifest retains
patient IDs/counts for all groups solely to verify validation membership.

## Your steps

1. Open [Google Colab](https://colab.research.google.com/), choose **File > Upload
   notebook**, and select `colab_baseline.ipynb`.
2. Choose **Runtime > Change runtime type > GPU**. Use a free T4 if offered.
   Free GPU availability is variable; do not select a paid resource for this task.
3. Run the notebook cells in order. In the upload cell, select
   `colab_baseline_bundle.zip`. The next cell installs packages and runs checks.
4. Authorize Drive access in the Drive cell. Results are saved under
   `MyDrive/TrialGPT_Assignment2/baseline_validation_v1`.
5. Run the baseline cell and let it reach **206/206**. The first model-weight
   download is several GB. No training occurs and no API key is needed.
6. Run the last cell to inspect the results and download `baseline_results.zip`.
   Keep this ZIP and return `metrics.json`/`summary.md` for interpretation before
   fine-tuning.

This workspace has CPU-only PyTorch and no direct connection to your Colab
runtime. The actual GPU predictions and measured baseline scores must be produced
by running the notebook; local preparation and unit-test results are not a model
baseline.

## Configuration and reproducibility

- Target: `Qwen/Qwen3-4B-Instruct-2507` at the preparation's pinned commit.
- Quantization: 4-bit NF4, double quantization, FP16 computation, no adapters.
- Generation: greedy, one beam, ordinary library KV cache enabled, batch size 1,
  up to 128 new tokens; input plus generation reserve fits the 1,024-token budget.
- Attention: PyTorch SDPA; this does not implement the assignment's custom tiled
  attention or cache-aware decoder requirements.
- Data: fixed validation patients only. Expert targets are never passed to the
  model. Every real prompt is checked against its recorded token length.
- Provenance: model revision, prompt/data/code hashes, settings, actual package
  versions, GPU, CUDA, and runtime IDs are saved.

The notebook retains Colab's CUDA PyTorch installation and installs pinned
preparation/evaluation dependencies. Each new process uses those installed
versions. Keep these quantization, precision, prompt, and generation settings
consistent when evaluating the fine-tuned model.

## Outputs

| File | Contents |
|---|---|
| `predictions.jsonl` | Raw model responses, gold targets, parsed outputs, source evidence, token counts, generation timing and peak memory |
| `metrics.json` | Overall and inclusion/exclusion accuracy, macro-F1, per-class precision/recall/F1, evidence micro/macro scores, validity rates |
| `summary.md` | Readable metric table and scoring policy |
| `confusion_*.csv`, `confusion_*.png` | Confusion matrices with an invalid-prediction category |
| `errors.jsonl` | Label errors, evidence mismatches, and invalid outputs for later review |
| `run_manifest.json`, `environment.txt` | Exact settings, hashes, versions, and runtime information |

`complete: true` requires all 206 predictions. A smoke run using `--limit 5` is
marked incomplete and cannot be reported as the full baseline.

## How failures and evidence are scored

No predicted response is silently repaired. Non-JSON text, markdown-wrapped JSON,
wrong keys/types, or duplicate sentence IDs are schema failures: the label is
invalid and evidence is treated as empty. These rows remain in all denominators.

For correctly structured JSON, the category and evidence are scored separately.
A category from the wrong criterion type becomes invalid. Nonexistent integer
sentence IDs remain in the predicted set and count as evidence false positives;
they also make the overall output invalid. A correct label with incorrect
citations can therefore score well on classification and poorly on evidence.

Evidence is evaluated as sets of sentence IDs. Both gold and predicted sets empty
receive precision, recall and F1 of 1; when exactly one is empty, all three are 0.
Micro scores pool TP/FP/FN across examples; macro scores average per-example
scores. The report also includes macro scores restricted to nonempty gold sets
so frequent empty annotations do not obscure retrieval failures.

Classification macro-F1 uses the fixed applicable expert label vocabulary, with
zero for undefined class scores. Confusion matrices retain an invalid-prediction
column. Timing covers synchronized generation calls after a warm-up, excluding
download/loading and file writes. Runtime IDs identify resumed GPU sessions;
preserve comparable hardware when making performance comparisons.

## If Colab disconnects

Reconnect to a GPU and rerun the cells with the same bundle and Drive output
directory. The runner checks the saved run signature and resumes completed
annotation IDs without repeating them. A changed configuration, code/data hash,
or package version is rejected rather than mixed into an existing run. Use a new
output folder if you intentionally start a different experiment.

To regenerate reports from saved predictions without loading a model:

```powershell
python baseline.py --output path/to/downloaded/baseline_results --metrics-only
```

## Official references

- [Qwen checkpoint and usage](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Transformers bitsandbytes integration](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [Generation configuration](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [scikit-learn classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [Colab resource limitations](https://research.google.com/colaboratory/faq.html)
