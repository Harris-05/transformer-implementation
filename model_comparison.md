# Model comparison and selection

**Decision date:** November 4, 2026  
**Selected target:** `Qwen/Qwen3-4B-Instruct-2507`  
**Status:** Selected for implementation; task-specific performance and hardware feasibility remain to be measured.

## The task and constraints

Given a numbered patient record, criterion type, and one eligibility criterion,
predict the appropriate expert eligibility category and supporting patient-sentence
IDs. The model needs medical-text comprehension, negation handling, numerical and
temporal comparisons, uncertainty handling, evidence localization, and consistent
structured output. It must run locally and expose autoregressive forward passes
for the assignment's custom KV-cache and speculative-decoding work.

Fine-tuning should be practical with free compute. Our current development data
contains 602 usable training examples and 206 validation examples. Training
annotation 883 is explicitly quarantined because its criterion text is missing.
The 206 test annotations remain reserved for the final evaluation. These counts
and the fixed patient split are documented in [the dataset audit](reports/data_audit.md).

The user's priorities are a balance of model capacity, training cost, and local
application feasibility. The selection below follows those priorities; it is
not a measured ranking of clinical accuracy.

## Candidate comparison

Parameter counts and licenses come from the linked official model cards.
Weight-memory figures are our estimates: parameters multiplied by two bytes for
FP16/BF16, expressed in decimal GB. They exclude activations, KV caches, gradients,
optimizer state, adapters, temporary buffers, and any draft model. They are not
total training or inference memory requirements.

| Candidate and official source | Parameters | Estimated 16-bit weights only | License | Task-relevant strengths | Tradeoff and selection outcome |
|---|---:|---:|---|---|---|
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3.09B | 6.2 GB | Qwen Research | Instruction-tuned causal decoder; documented JSON support; lower weight memory than the selected model. | Remains a credible lower-resource alternative. We prefer the selected model's updated post-training and somewhat larger capacity. This does not establish that the 3B model is worse at clinical reasoning. |
| **[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)** | **4.0B** | **8.0 GB** | **Apache 2.0** | Instruction-tuned causal decoder; updated text comprehension and instruction following; direct output without thinking blocks; grouped-query attention. | **Selected:** balances capacity and resource demand for this project. Quantized adapter training is the proposed starting approach; memory and clinical quality must be measured. |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3.21B | 6.4 GB | Llama 3.2 Community | Compact instruction-tuned decoder; retrieval and summarization are stated use cases, relevant starting abilities for evidence selection. | Useful alternative family with lower weight memory. Access conditions and a custom license add setup considerations. Its clinical eligibility performance is unmeasured here. |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7.61B | 15.2 GB | Apache 2.0 | Larger capacity with documented instruction following and structured-output support. | Not selected as the initial target because of its higher memory demand and likely training/inference burden. It may outperform smaller models, but that improvement has not been demonstrated on our data. |

## Why Qwen3-4B-Instruct-2507 is selected

1. **Resource balance.** Its approximate 8 GB of 16-bit weights is substantially
   below the 7B candidate's 15.2 GB, while providing more parameters than either
   3B candidate. This is a capacity/resource compromise, not proof of superior
   reasoning or a guarantee of faster generation.
2. **Relevant post-training.** The developer reports updated instruction
   following and text comprehension. Those abilities are relevant to applying a
   criterion to a patient note. We will verify eligibility and evidence quality
   ourselves rather than transfer general benchmark claims to this task.
3. **Direct responses.** This particular checkpoint supports non-thinking output
   only, simplifying our label-and-sentence-ID response format. It is different
   from the original `Qwen/Qwen3-4B` checkpoint with switchable thinking mode.
4. **Implementation fit.** Its causal Transformer architecture supports the
   forward-pass and cache work required by the assignment. Apache 2.0 licensing
   simplifies reproducible distribution of our implementation and adapters.

The decision is to use this checkpoint as our target model. No candidate has yet
been evaluated or fine-tuned in this project, and no accuracy, speed, or memory
result is being claimed. More parameters alone do not guarantee better medical
understanding. If measured resource requirements exceed the available hardware,
we will document and revisit the training configuration or model choice.

## Selected architecture

| Property | Specification |
|---|---|
| Exact model ID | `Qwen/Qwen3-4B-Instruct-2507` |
| Model type | Causal language model; pretraining and post-training |
| Parameters | 4.0B total; 3.6B non-embedding |
| Decoder implementation | `Qwen3ForCausalLM` |
| Transformer layers | 36 |
| Attention | Grouped-query attention: 32 query heads and 8 key/value heads |
| Hidden size | 2,560 |
| Head dimension | 128 |
| Vocabulary size in configuration | 151,936 |
| Positional encoding | RoPE |
| Supported native context | 262,144 tokens |
| Response mode | Non-thinking only; no thinking switch is needed |

Sources: [official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
and [official configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json).

The context limit is an architectural capability, not our planned training length.
The next step will measure tokenized prompts and targets, then select a manageable
length that preserves relevant patient evidence. Long contexts increase attention
and cache costs and are unnecessary merely because the model supports them.

## Proposed compute and adaptation approach

Use the laptop for dataset preparation, tokenizer inspection, and lightweight
checks. The currently inspected PyTorch installation is CPU-only; this does not
establish whether the physical laptop has a compatible GPU. Plan GPU fine-tuning
in free Colab when a suitable runtime is available.

Start by investigating **QLoRA**: load quantized base weights and train LoRA
adapters. This reduces the trainable parameter count and base-weight memory
compared with full fine-tuning. Hardware compatibility, compute dtype, sequence
length, batch size, adapter configuration, and peak memory still need a smoke
test. Quantizing weights does not eliminate activation or runtime overhead.
[Hugging Face's official PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
describes this approach.

Free Colab GPU availability, runtime duration, and resource limits are variable;
the selected model is a feasibility target, not a guaranteed fit in every free
session. Record the GPU and memory actually allocated and save checkpoints and
logs outside the temporary runtime. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

For a fair before/after comparison, keep the base checkpoint, prompt, precision or
quantization configuration, and decoding settings consistent. Record the exact
model revision and dependency versions when the model is downloaded.

## Inference considerations to retain

The target supports the required cache-aware decoder work. A smaller draft model
must be selected separately for speculative decoding. Being in the same family
does not prove tokenizer compatibility: verify token-to-ID mappings, special
tokens, and tokenization behavior. Measure the additional memory for both models
and their caches together. Speculative-decoding speedup is not assumed.

## Immediate next task: tokenizer and prompt preparation

Before training or downloading the full target weights:

1. Load the selected checkpoint's tokenizer and chat template.
2. Define a prompt using only `note`, `criterion_type`, and `criterion_text`.
   Keep GPT-4 fields and expert targets out of the prompt.
3. Define the supervised response as JSON, for example
   `{"label": "included", "sentence_ids": [0]}`. Preserve the applicable
   inclusion/exclusion labels, including insufficient-information and
   not-applicable outcomes.
4. Measure training/validation prompt and target token lengths, including template
   and special-token overhead. Select a working length and document any
   truncation and its effect on evidence.

After that, load the original model in the chosen environment and establish a
validation baseline before fine-tuning. Record macro-F1, per-class metrics,
evidence precision/recall/F1, output-format validity, and resource use. Reserve
the test set for the final fixed before/after evaluation.

## Reading for the report

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388): model-family training,
  architecture, and evaluation context. Use the exact checkpoint card for
  `2507`-specific behavior and configuration.
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115): background for the
  two Qwen2.5 alternatives.
- [LoRA](https://arxiv.org/abs/2106.09685): low-rank adapter training.
- [QLoRA](https://arxiv.org/abs/2305.14314): quantized base-model adapter training.

These sources support the design rationale. Our eventual report must distinguish
published results from measurements produced by this project.
