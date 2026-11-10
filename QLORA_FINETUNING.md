# QLoRA fine-tuning: first controlled experiment

## Decision and label review

Use the selected Qwen3-4B-Instruct-2507 checkpoint with quantized base weights
and train LoRA adapters. Keep the original prompt and expert labels unchanged.
The [exclusion-label review](reports/exclusion_label_review.md) documents the
reference prompt's information-policy difference and eight validation errors.
Do not introduce an automatic rule that converts absent evidence into eligibility.

This experiment measures adaptation to the benchmark. It does not establish
clinical reliability, and improved scores do not resolve annotation ambiguities.

## Files and commands

The generated files are `colab_finetuning.ipynb` and
`colab_finetuning_bundle.zip`. Open the notebook in Colab, enable a GPU, and run
its cells in order. Upload the training bundle when prompted and connect Drive.
The notebook runs preparation and integration checks, training, adapter evaluation,
and result download. Its Drive folder is:

```text
MyDrive/TrialGPT_Assignment2/qlora_v1
```

The bundle contains 602 training and 206 validation examples, a documented
quarantined training record, frozen baseline artifacts, code, and configuration.
It includes no held-out test content. It downloads the public original model in
Colab and does not require a paid API.

To rebuild after code changes:

```powershell
python make_colab_finetuning.py
```

Local preparation check, with no model weights or GPU:

```powershell
python finetune.py --prepare-only --offline
```

The Colab notebook runs these GPU stages in separate processes:

```text
python finetune.py --output <Drive run directory>
python evaluate_adapter.py --training-output <same Drive run directory>
```

## Initial configuration

The exact parameters are recorded in `configs/qlora_config.json`.

| Setting | Initial value and reason |
|---|---|
| Base model | Same pinned Qwen checkpoint as the original baseline |
| Quantization | 4-bit NF4 with double quantization; FP16 computation for the T4 |
| Adapter | Rank 16, alpha 32, dropout 0.05; all linear layers except the output head through PEFT's `all-linear` policy |
| Training batch | One example per device, eight accumulation steps; effective batch eight on one GPU |
| Learning rate | 0.0001; a starting experiment, not an optimized value |
| Warm-up | 10% of planned optimizer updates, passed as `warmup_steps` for the installed Transformers API |
| Epochs | At most three; early stopping after two non-improving validation evaluations |
| Memory | Non-reentrant gradient checkpointing; 1,024-token working context; no truncation |
| Checkpoint selection | Lowest validation completion cross-entropy loss among saved epoch checkpoints |
| Optimizer | PyTorch AdamW; weight decay 0.01; gradient norm cap 1.0 |
| Seed | 42 |

No class labels are collapsed and no rare examples are discarded or oversampled.
The dataset is imbalanced, so report per-class scores and macro-F1 regardless of
the change in accuracy. Checkpoint selection uses validation loss, not test
performance or a claim that the selected checkpoint maximizes eligibility F1.

## Completion-only objective

For each example, tokenize the frozen prompt and the full prompt-plus-expert JSON
conversation with the original chat template. Verify that the prompt token IDs
are an exact prefix of the full sequence. Set all prompt labels to `-100`, keep
completion token IDs as labels, and also mask padding with `-100`.

The model therefore receives the full patient record and criterion, but loss is
calculated only on the expert response and its template ending tokens. The
collator pads on the right without replacing real completion/end tokens.
The preparation check covers all 808 development records. Original sentence
numbering and all expert-cited text remain in the input.

## What is checked locally

Unit checks cover prompt/completion masking, padding, invalid sequence prefixes,
overlength rejection, patient isolation, applicable labels, and reproduction of
the frozen baseline metrics by the adapter evaluator's shared scoring functions.

`verify_training.py` additionally constructs a tiny random Qwen3 model on CPU and
exercises the real Trainer/PEFT interfaces. It checks that adapter weights change,
base embeddings stay frozen, validation loss is finite, and checkpoint save/load
works. That smoke check downloads no pretrained weights and produces no clinical
performance results. The full 4-bit GPU training/memory path must run in Colab.

## Checkpoints, logs, and resuming

Checkpoint and optimizer states are saved at each epoch boundary on Drive. If a
session disconnects, rerun the notebook with the same bundle and Drive directory.
The trainer resumes the latest saved epoch checkpoint; work after that checkpoint
may be repeated. Data, code, settings, and version mismatches are rejected.

Raw training logs are appended throughout the run. Successful completion saves:

| File/directory | Contents |
|---|---|
| `adapter/` | Selected adapter weights/configuration and tokenizer files |
| `training_manifest.json` | Model identity, versions, data/code hashes, GPU and experiment configuration |
| `training_log.jsonl`, `training_history.json` | Loss, validation loss, learning-rate and progress records |
| `training_curves.png` | Training and validation completion-loss curves |
| `training_summary.json` | Selected checkpoint, loss, parameter counts, measured memory and adapter hashes |
| `checkpoints/` | Epoch checkpoints with optimizer/RNG state for resuming |
| `validation/` | All 206 adapter predictions, per-class/evidence metrics, confusion matrices, errors, and before/after comparison |

The download excludes optimizer checkpoints to reduce its size; they remain on
Drive. Keep the selected adapter, its hashes, training logs, and comparison data.

## Colab's optional torchao conflict

PEFT 0.21.2 probes optional adapter backends and raises if an installed torchao is
older than 0.16.0. Colab's preinstalled 0.10.0 can therefore block `get_peft_model`
even though this experiment uses bitsandbytes. The notebook removes the unused
torchao package before setup and keeps the integration check enabled. It does
not upgrade PyTorch or change the experiment's quantization settings.

In an already-open notebook with `run_checked` defined, recover with:

```python
run_checked([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
run_checked([sys.executable, "-u", "verify_training.py"])
run_checked([sys.executable, "-u", "finetune.py", "--prepare-only"])
```

Continue only after those checks pass. The reported pandas, pyarrow, and diffusers
package conflicts involve other preinstalled applications; they are separate
from the confirmed torchao traceback. Our child-process checks still need to pass.
Source: [PEFT's optional torchao version check](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/import_utils.py).

## Fair comparison

Adapter evaluation reloads the same original 4-bit model plus the selected adapter,
with unchanged prompt, context/generation budget, batch size, SDPA attention and
greedy decoding. The original scoring functions retain malformed predictions,
invalid citation false positives, and the same empty-evidence convention.

The evaluator checks the validation data hash, model configuration, frozen scoring
code, selected adapter hashes, and training manifest. It records software versions;
if shared versions differ from the baseline, a controlled original-model rerun is
needed before attributing score changes exclusively to training. Negative score
changes remain visible in `validation/comparison.json`.

After the run, return `training_summary.json`, `validation/comparison.json`, and
`validation/metrics.json`. Review the measured results before choosing another
training configuration. The final held-out test evaluation remains a later step.

## Official sources

- [PEFT quantization recipe](https://huggingface.co/docs/peft/developer_guides/quantization)
- [LoRA and all-linear targeting](https://huggingface.co/docs/peft/developer_guides/lora)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Trainer callbacks and early stopping](https://huggingface.co/docs/transformers/en/main_classes/callback)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
