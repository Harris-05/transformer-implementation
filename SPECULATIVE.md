# Task 5: greedy speculative decoding

## Selected pair and resource choices

The target is the selected fine-tuned `Qwen/Qwen3-4B-Instruct-2507` adapter and
the original pinned base revision. The draft is `Qwen/Qwen3-0.6B`, pinned to
`c1899de289a04d12100db370d81485cdf75e47ca`, with Apache-2.0 licensing. A smaller
same-family model limits draft latency/memory and allows direct token-ID sharing.
The draft is not fine-tuned here. Acceptance and speed are measured, not assumed.

This experiment keeps target weights in 4-bit NF4/double quantization but uses
**FP32 target computation** and forced **math SDPA** for ordinary target greedy,
HF reference and speculative target verification. Draft computation is FP16.
The precision/backend control responds to the numerical sensitivity observed in
the earlier FP16 KV experiment. These are separate results; prior classification
and KV artifacts are unchanged. Extra FP32 unquantized storage and the second
model increase memory needs, which the actual Colab run must measure.

Only tokenizer/config assets were downloaded locally. Full target/draft weights
are loaded in Colab. No speedup or GPU-fit claim follows from tiny CPU tests.

## Tokenizer compatibility

`configs/draft_tokenizer_lock.json` pins hashes of all five draft assets. The
target assets retain their existing lock. `speculative_inputs.py` loads fast
tokenizers, verifies exact token-to-ID dictionaries, configured backend JSON,
output vocabulary sizes, and BOS/EOS/pad IDs. Compatibility is checked before
loading models. A wrong normalization rule can fail even with equal vocab IDs.

The selected actual fast backends and tokenizer.json files are identical. Both
configured vocabularies contain 151669 entries and both model output vocabularies
have 151936 rows; EOS=151645, pad=151643, BOS=None. Raw merges.txt files differ,
so their byte equality is not claimed or used as compatibility proof. Each is
independently checked against its own pinned hash. The **target chat template
alone** formats every prompt. The draft receives those exact token IDs and the
accepted target continuation; its thinking-mode chat template is never applied.

## Explicit algorithm

`speculative_decoding.py` makes ordinary model forward calls and manages two
fresh DynamicCaches. It does not call an assisted-generation API.

At the start of a normal round, both caches contain the committed sequence
except its last token. The first round starts with empty caches and the full
prompt as the pending prefix.

1. Feed the uncached committed suffix to the draft and greedily propose up to K
   tokens. Between proposals, advance only the draft cache. Stop proposing at
   EOS or the remaining output budget. The last proposal is not yet processed
   by the draft at this stage.
2. Forward the target once on its pending committed suffix plus the proposal
   block. The final K+1 logit rows predict K candidate tokens and a possible
   bonus. Positions remain absolute and masks include past plus current tokens.
3. Accept only the consecutive draft prefix equal to target argmax tokens.
   On the first rejection emit the target correction. If all match and budget
   permits, emit the target bonus. Accepted/corrected/bonus EOS stops generation.
4. Crop caches to committed length minus one so rejected suffixes disappear.
   In the all-accepted-plus-bonus case, process the final draft candidate once
   to bring the draft cache to the same prefix as the target. The bonus remains
   pending for the next round. Negative removal counts use the pinned crop API.

Every round emits at least one token, and final output never exceeds the budget.
Both final caches cover prompt+output minus the last token. Tests verify full
acceptance, bonus, zero acceptance, middle rejection, EOS correction and budgets.
Logical KV-byte counts refer to active cache tensor shapes; cropped views can
retain larger backing storage, which measured GPU allocator peaks include.

Parallel chunk verification changes computation shapes relative to single-token
greedy. Even with the same weights, finite-precision outputs can diverge near
ties. Strict token checks remain enabled; no token repair, alternative tie rule,
relaxed equality or silent greedy fallback is used to manufacture a pass.

## Experiment

Six development cases: the three complete clinical prompts already selected by
shortest/middle/longest length (412/478/582 tokens, normal EOS, max 128 output),
plus controlled stress inputs/budgets 128/16, 256/32 and 512/64. Stress inputs
ignore EOS and are not clinical accuracy measurements. Draft lengths are 2 and 4.
The held-out patient test group is never used.

Preflight compares ordinary target greedy with HF, then speculative K=2/K=4
with ordinary greedy, for every case **before any timing**. On mismatch, output
IDs, first difference and round cache traces are saved and timed work is aborted.
All measured repeats are compared with the same preflight reference too.

One warmup per mode/case and three repetitions give **54 timed trials** across
ordinary greedy, speculative-2 and speculative-4. Order rotates across repeats;
CUDA timing synchronizes. Loading, tokenization, initial buffers, warmup, plots
and final CPU output transfer are outside timing. Drafting, verification,
acceptance comparisons, necessary host/device synchronization and cache rollback
are inside total latency. Saved raw trials include output IDs, timing, cache
sizes, proposed/accepted counts and actual draft/target forward counts.

Acceptance rate is accepted matching-prefix draft tokens divided by all proposed
tokens; correction and bonus tokens are not accepted draft tokens. Output tokens/s
uses actual output length divided by total latency. Ratios below one are retained
as slowdowns. Reported validated speedup is suppressed for unequal/incomplete work.

Both model weights stay resident for **every** timed mode, including ordinary
greedy. GPU memory therefore includes the pair in all cases; it is not a
target-only deployment memory comparison. CPU GPU-memory fields are null.
Measured peaks, initial allocation and the extra draft cache are retained.

## Running

```powershell
python -m unittest -v test_speculative test_speculative_inputs test_speculative_benchmark
python benchmark_speculative.py --prepare-only --offline
python benchmark_speculative.py --smoke --output artifacts/speculative_cpu_final
python make_colab_speculative.py
```

Preparation verifies the actual adapter and compatibility without loading any
model weights. CPU smoke uses random miniature **FP64** Transformers and is
implementation evidence, not a Qwen GPU performance result. Ordinary greedy and
HF are independent references; no assistant_model library feature is used.

The self-contained `colab_speculative.ipynb` embeds code, pinned requirements,
locks and validation data; it needs no new code/data bundle. Its setup locates
an existing training folder, flat files under `/content`, a local uploaded
`finetuning_results.zip`, or the saved adapter on Drive. It never opens an upload
widget. To avoid reuploading weights when opening a fresh runtime, first run
`colab_save_adapter_to_drive.py` in the current runtime. The notebook finds the
resulting `MyDrive/TrialGPT_Assignment2/selected_adapter` folder automatically.

Run cells in order, let preparation and preflight finish, and download
`speculative_results.zip`. Base/draft weights may download into the runtime cache;
no training happens. Results persist on Drive in `speculative_v1`.
Different settings/source/versions require a new output directory. Matching
incomplete timed runs resume saved trials. A changed reference or a prior failed
preflight must be kept as a separate failed experiment, not overwritten.

## Sources

- [Leviathan et al., speculative decoding](https://arxiv.org/html/2211.17192v2)
- [Transformers 5.15.1 assisted decoding and greedy verification](https://huggingface.co/docs/transformers/v5.15.1/en/assisted_decoding)
- [Qwen3-0.6B model card and Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pinned draft architecture configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)
- [Pinned Transformers cache implementation](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/cache_utils.py)

This implementation uses deterministic greedy prefix matching, not the paper's
probabilistic rejection-sampling procedure. The paper's speedups are not claimed
for this Python implementation or for this model pair before measurement.
