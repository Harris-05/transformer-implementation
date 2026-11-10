# Tokenizer and prompt preparation

This step uses the selected Qwen tokenizer to prepare training/validation
conversations and measure their lengths. It does not load model weights,
generate predictions, or train a model. Run it on the laptop; no GPU is required.

## Run

From the project directory:

```powershell
python -m pip install -r requirements-model-inputs.txt
python prepare_model_inputs.py
```

The tokenizer is already downloaded in this workspace. To verify and reuse it
without network access:

```powershell
python prepare_model_inputs.py --offline
```

The first online run retrieves only the tokenizer and configuration files
(approximately 15.9 MB). `configs/tokenizer_lock.json` records the exact model
commit and each file's SHA-256. Subsequent runs verify these files; a new checkout
can download that same revision using the lock. No Hugging Face login is required
for these public files.

If preparation reports missing data, run `python eda.py` first. The original
parquet file, split manifest, development JSONL files, and quarantine record are
required so preparation can verify source identity, counts, and patient membership.

## What the script produces

| File | What to read or use |
|---|---|
| `reports/prompt_example.txt` | A real training example showing the exact formatted prompt and its separate expert target |
| `reports/tokenization_report.md` | Readable counts, context budget, and truncation findings |
| `reports/tokenization_report.json` | Detailed length distributions, versions, hashes, and token IDs |
| `reports/token_lengths.csv` | Per-example lengths for all 808 development records |
| `configs/tokenizer_lock.json` | Exact model revision and tokenizer-file hashes |
| `configs/model_input_config.json` | Context length, generation allowance, prompt hash, and greedy-decoding setting for the next stage |
| `data/model_ready/train.jsonl` | 602 conversational prompt/completion records |
| `data/model_ready/validation.jsonl` | 206 conversational prompt/completion records |

The `prompt` field contains system instructions and the three permitted inputs.
The `completion` field contains an assistant message with the expert target.
Metadata and measured lengths are separate. A baseline must receive **only the
prompt**; the completion is used for scoring or supervised training. During
fine-tuning, the eventual trainer should apply loss to the assistant completion,
masking the prompt tokens. This script checks the prompt/full-conversation token
prefix needed to implement that mask correctly; it does not implement training.

## Response format

```json
{"label": "included", "sentence_ids": [0]}
```

Inclusion labels are `included`, `not included`, `not enough information`, and
`not applicable`. Exclusion labels are `excluded`, `not excluded`,
`not enough information`, and `not applicable`.

Sentence IDs must be unique nonnegative integers present in the original note.
An empty list is valid. `validate_response` checks the structure, applicable
label set, and citation validity; it cannot prove clinical correctness or that
the cited text actually supports the prediction. Those require evaluation.

The target intentionally contains no generated explanation. Evidence text can be
extracted using the cited IDs, and grounded explanations will be handled in the
later screening pipeline. The dataset has no expert-written explanation target;
GPT-4 explanations are not reused as supervision.

## Current results and your next action

Preparation measured a maximum training prompt of 744 tokens and a maximum
validation prompt of 582. A **1,024-token context**, reserving **128 tokens for a
response**, fits all current development examples. No truncation was applied;
every cited development sentence is retained. These are input-format measurements,
not model quality results. Longer unseen inputs will need explicit handling.

Open `reports/prompt_example.txt` and identify the patient note, criterion type,
criterion text, and separate expert response. Then read the short report. The
next implementation task is the original model's validation baseline, preferably
in a GPU runtime; fine-tuning comes after that baseline is saved.

## Sources

- [Selected checkpoint](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Official chat-template guide](https://huggingface.co/docs/transformers/chat_templating)
- [Official Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download)

The script uses `apply_chat_template(tokenize=True)` to include the checkpoint's
control tokens without adding duplicate special tokens, and `snapshot_download`
with an explicit file allowlist and pinned commit.
