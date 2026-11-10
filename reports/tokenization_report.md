# Tokenizer and prompt preparation

Model: `Qwen/Qwen3-4B-Instruct-2507`
Revision: `cdbee75f17c01a7cc42f958dc650907174af0554`

| Split | Records | Longest prompt | Longest complete training conversation |
|---|---:|---:|---:|
| train | 602 | 744 | 771 |
| validation | 206 | 582 | 605 |

Working context: **1024 tokens**.
Generation allowance: **128 tokens**.
Largest prompt plus generation allowance: 872 tokens.

No records were truncated. All expert-cited development sentences remain in the full input notes.
Chat-template control tokens and completion-ending tokens are included in the counts.
For every example, the inference prompt's token IDs were checked to be an exact prefix of the complete training conversation.

The generation allowance is a cap, not the number of tokens the model must produce. Generation can stop earlier at EOS.
The working context budget is chosen from development lengths; it is not the model's architectural maximum.
An unseen note can be longer. Later inference must reject or explicitly handle over-budget inputs rather than silently truncate evidence.

Only tokenizer/configuration files were loaded. No model weights, predictions, training, or test-content inspection were used.
Next: run the original model's validation baseline in the selected compute environment.
