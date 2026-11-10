# Task 3: KV-cached autoregressive generation

## Run in Colab

1. Upload `colab_kv_cache.ipynb` to Colab and select a GPU runtime.
2. Run the cells in order; upload `colab_kv_cache_bundle.zip` when prompted.
3. Mount the same Drive account used for fine-tuning. The default adapter folder
   is `MyDrive/TrialGPT_Assignment2/qlora_v1`. Change `TRAINING` if you used another.
4. Let the benchmark cell finish. It loads the selected adapter, checks token
   agreement and measures 54 timed trials. It does not train.
5. Run the last cell to download `kv_cache_results.zip`, including on a mismatch,
   and return that ZIP for analysis. Results remain on Drive in `kv_cache_v1`.

The 18-file code bundle has validation prompts and split metadata; it contains no
training examples, test examples, adapter weights or base weights. The split
manifest retains IDs/counts for all groups to enforce patient separation.
The training summary and each selected adapter file are checked against their
recorded hashes. The base model revision, tokenizer assets and prompt stay fixed.
Dependencies match the working fine-tuning notebook; the unused optional torchao
package is removed to avoid the already observed PEFT compatibility error.

An interrupted run resumes saved trials with the same source, model, adapter,
configuration and environment fingerprint. Trial logs are committed as complete
temporary-file snapshots with replacement, so an interrupted write keeps the
previous committed trials. Changed settings require a new output
folder. Incomplete or mismatching runs cannot claim a completed correctness pass.

## Explicit decoding flow

`decoding.py:greedy_decode` accepts one unpadded nonempty `[1, length]` long tensor.
Call `model.eval()` first. Padding, batch sizes above one, invalid budgets and
context overflow fail explicitly; tokens are never silently truncated.

- **Uncached:** every step forwards the whole prompt plus generated prefix with
  `use_cache=False`, absolute positions `0..length-1`, and no cache.
- **Cached prefill:** create a fresh `DynamicCache(config=model.config)`, forward
  the complete prompt and save the returned `past_key_values`.
- **Cached decoding:** feed only the last generated token, reuse the returned
  cache, advance its absolute position and extend the attention mask to cover
  both cached and current tokens. Check that the cache grew by one processed token.
- Both paths choose `argmax` over the last token logits and append that token.
  They stop at an explicit EOS ID or the generation budget. EOS is included in
  the returned IDs. No sampling, repairs, repetition penalties or beam search.
- Both request `logits_to_keep=1`, avoiding full-prefix vocabulary logits while
  leaving attention computation intact. Library `generate()` is used solely as
  an independent correctness reference, with an explicit GenerationConfig.

For a prompt of P tokens and N generated tokens, there are N forwards: the first
predicts output token one; N-1 further forwards predict the rest. The final output
token has not been fed back into the model, so the cache ends at **P+N-1** tokens.
An independent call always starts with an empty cache. This is important for the
later speculative decoder, where cache state must describe processed tokens.

### Positions, masks and cache storage

This implementation targets Transformers **5.15.1**. Its Qwen3 forward accepts
`position_ids`; no obsolete `cache_position` argument is assumed. Cached tokens
keep their original rotary positions. The current cached input receives position
P, then P+1, and so on; positions never restart at zero during decoding.

The supplied all-ones attention mask has shape `[1, past_length+current_length]`.
Qwen3 constructs its causal mask using that mask, absolute positions and past
cache length. Full-prefix forwards prevent looking ahead; a single cached query
can attend to the complete earlier prefix and itself. This delegates Q/K/V
projection and causal-mask construction to the model while implementing cache
ownership, reuse, input slicing, position progression and mask growth explicitly.

Each layer's dynamic K/V tensors grow along sequence length. Full attention uses
storage proportional to `2 * layers * batch * KV_heads * processed_tokens * head_dim
* bytes_per_element`. The eight KV heads in the real model reflect grouped-query
attention; cache memory is not computed using its 32 query heads. The report sums
actual K/V tensor storage, including element size. The 4-bit base weights do not
imply a 4-bit KV cache. Dynamic concatenation allocates transient storage and can
copy previous tensors; its overhead is part of the cached-path measurements.

## Correctness and numerical differences

The CPU tests instantiate real random miniature Qwen3 models, without downloads.
They compare manual cached, manual uncached and HF generation across three
input/output lengths; check actual forward positions/masks and cache growth;
exercise EOS on the first token, budget exhaustion and independent repeated calls.
They also test incomplete/mismatching benchmark reporting and invalid inputs.

On the selected GPU adapter, correctness checks cover all nine stress cases and
three complete validation prompts chosen deterministically by shortest, middle
and longest prompt length. Clinical prompts retain the full frozen chat template
and normal EOS stopping, with a 128-token maximum. They are development checks,
not a new classification evaluation or held-out test result.

FP16/quantized computations with different query shapes or attention kernels may
have slightly different logits. A close argmax can therefore change a token.
Mismatches are retained with the first differing output index, complete token
sequences and identical-prefix last-logit diagnostics (top IDs, top-two margins
and maximum absolute difference). Any mismatch fails the experiment; it is not
hidden by a tolerance or output repair. Investigate masks/positions/cache state
before attributing divergence to floating-point arithmetic. A fixed seed alone
does not prove deterministic GPU arithmetic. Token equality is checked on every
timed repeat against both paths and the untimed HF reference.

## Benchmark protocol

`configs/kv_cache_config.json` defines input lengths **128, 256, 512** crossed with
output budgets **16, 32, 64**, three repetitions and one warm-up per case/path.
Input token IDs come from cropping/repeating the longest selected development
prompt to an exact stress length. These artificial prompts deliberately ignore
EOS to measure equal work, even if a special token appears in generated output.
Do not interpret these continuations as clinical predictions or report accuracy
on them. Full clinical prompts are checked separately with EOS enabled.

Both paths share the loaded 4-bit NF4/double-quantized adapter, FP16 compute, SDPA
backend, batch size one, prompt IDs and output budget. Order alternates between
cached-first and uncached-first across repetitions. Each trial starts with fresh
cache state. CUDA synchronization brackets timing. Garbage collection and
`empty_cache()` happen outside timing before each trial; this measures a warmed
model/kernel with a cleared unused allocator pool, not a steady-state service.

- **Prefill:** first full-prompt forward, argmax and first output token.
- **Decode:** remaining N-1 forward/selection steps; zero if EOS is output first.
- **Total:** prefill plus decode. Output tokens/s is N divided by total time;
  decode-only tokens/s is (N-1) divided by decode time.
- **Memory:** reset peak statistics each trial, record initial allocated memory
  and peak allocated/reserved memory. Peaks include resident weights, buffers and
  any KV cache, not only the cache. Allocator history still affects reserved
  memory. CPU smoke results have null GPU memory values.

Model loading, tokenizer work, buffer allocation, final CPU token transfer,
correctness tracing, diagnostics, file writes and plots are outside timing.
Warm-up is outside measurements. Records preserve every repetition; summaries
give medians and min/max ranges. Speedup is uncached/cached total-time median and
is suppressed when a fixed workload is incomplete, unequal or mismatching.
GPU clocks, load, quantized linear kernels and dynamic cache copying may limit
speedup. No faster result is assumed.

## Files and local checks

`decoding.py` owns generation and logit diagnostics. `benchmark_kv_cache.py` owns
the experiment, adapter loading, reproducible cases and resumable trial logs.
`kv_benchmark_report.py` validates trial completeness and writes JSON/CSV/plots.

### Explicit math-SDPA comparison

The original GPU experiment passed six of nine stress cases and all three
clinical prompts. The 256-token stress input diverged at output index 5 for
all three budgets; cached generation matched HF, while uncached generation
preferred another token by a small logit margin. A focused Colab control on
that same p256-o16 input passed all three paths when math SDPA was forced.
This establishes backend sensitivity for that case; it does not yet establish
full-grid equality. Preserve both runs and the original numerical diagnostic.

Use the explicitly recorded backend option for the controlled full rerun:

```powershell
python benchmark_kv_cache.py --training-output results/finetuning_qlora_v1 --output results/kv_cache_math_v1 --sdpa-backend math
```

`--sdpa-backend automatic` retains automatic kernel selection (the default).
`math` forces `SDPBackend.MATH` for both manual paths, HF generation, diagnostics,
warm-up and every timed repetition. The backend name is part of the signed run
identity in `run_manifest.json`; changing it cannot resume another backend's run.
Weights, adapter, FP16 compute, prompts, EOS policy and strict equality checks
remain the same. Math-SDPA timings must be measured again and labelled by backend;
do not reuse automatic-backend timings or mark original mismatches as passing.

For the current Colab session, `colab_rerun_kv_math.py` is one copyable cell. It
applies a SHA-verified source update to the known existing benchmark version,
keeps a source backup, reuses the existing TRAINING directory, and writes the
new experiment to Drive under `TrialGPT_Assignment2/kv_cache_math_v1`. It accepts
the already updated source too, allowing a matching run to resume. No model
files need uploading again.

The completed math-SDPA GPU log subsequently showed 8/9 stress correctness cases
and all three clinical checks passing, with all 54 timed trials finished.
The p512-o64 case still fails equivalence. The user closed this stage with that
limitation documented and requested no more reruns. Keep the original failure
flags and both runs; details are in `reports/kv_cache_numerical_investigation.md`.
Use `colab_download_kv_results.py` to download both experiments for the report.

The source snapshot in `artifacts/kv_backend_update/benchmark_original.py` is
archival material for the reviewed update, not an additional runnable test.

```powershell
python -m unittest -v test_decoding test_kv_benchmark
python benchmark_kv_cache.py --prepare-only
python benchmark_kv_cache.py --smoke
python make_colab_kv_cache.py
```

`--smoke` runs the entire benchmark/report workflow on a random miniature CPU
model with shorter lengths, saving `results/kv_cache_cpu_smoke/`. Its results
verify implementation and artifact generation; they are not performance evidence
for the full model. The real GPU measurements remain pending until Colab runs.

GPU outputs: `run_manifest.json`, `cases.json`, `correctness.json`, `trials.jsonl`,
`metrics.json`, `benchmark.csv`, `summary.md`, `timings.png`. Clinical token IDs
and cache traces are saved in correctness.json. Version/source fingerprints and
adapter hashes are in run_manifest.json. No test content is loaded or evaluated.

## Official references

- [Transformers 5.15.1 cache explanation and explicit loop](https://huggingface.co/docs/transformers/v5.15.1/en/cache_explanation)
- [Qwen3 5.15.1 forward: positions, masks, cache and logits_to_keep](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/qwen3#transformers.Qwen3ForCausalLM.forward)
- [PyTorch CUDA synchronization](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.synchronize.html)
- [PyTorch peak allocated memory](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.memory.max_memory_allocated.html)
- [SDPA backend controls and numerical precision](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)

The installed Qwen3 model source was also inspected to verify how it creates
causal masks and grows its DynamicCache with explicit position_ids.
