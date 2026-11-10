# Verified KV-cache math-SDPA experiment

The uploaded `kv_cache_results.zip` contains the **math-SDPA** experiment only,
not both experiments. All eight original files were preserved without alteration
in `results/kv_cache_math_v1/`; the ZIP is retained. The automatic-SDPA outcome
and focused numerical diagnostic remain documented from the user's pasted logs;
those original archives were not present in this upload.

## Independent verification

Verified the signed run identity; current code hashes; fixed validation-file
hash; selected adapter/training signatures; all 54 unique trial records; token
comparisons against the stored HF references; all correctness flags; and all
reported timing medians, speedups and completeness fields. Recomputing the
aggregation reproduces `metrics.json` exactly. No GPU inference was rerun.

Hardware: Tesla T4. Environment: torch 2.14.1+cu130, Transformers 5.15.1,
PEFT 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0 and tokenizers 0.22.2.
The selected 4-bit Qwen3-4B adapter and FP16 compute remain unchanged.

## Results

All **54 timed trials** finished. Eight of nine stress correctness cases and
all three complete clinical cases passed. Manual cached generation matches HF
in all 12 correctness cases. The uncached path differs at output index **43**
(the 44th token) for p512-o64. The overall stored result remains `passed: false`.

| Input / output tokens | Cached median seconds | Uncached median seconds | Validated speedup |
|---|---:|---:|---:|
| 128 / 16 | 2.2908 | 3.2494 | 1.42x |
| 128 / 32 | 4.9625 | 7.1798 | 1.45x |
| 128 / 64 | 9.2676 | 15.9552 | 1.72x |
| 256 / 16 | 2.4207 | 5.8830 | 2.43x |
| 256 / 32 | 4.7097 | 12.4834 | 2.65x |
| 256 / 64 | 9.7462 | 24.4015 | 2.50x |
| 512 / 16 | 2.8475 | 9.9124 | 3.48x |
| 512 / 32 | 4.9782 | 20.7754 | 4.17x |
| 512 / 64 | 10.0655 | 42.6393 | Not reported: unequal tokens |

For matching workloads, observed total-time median speedups range from 1.42x to
4.17x. These are model/backend/hardware-specific measurements, not guarantees.
Allocated GPU peaks range from 2.6992 to 2.8143 GiB cached and 2.6836 to
2.7725 GiB uncached. They include resident weights and temporary tensors, not
only KV tensors. See raw records for prefill/decode timing, throughput, reserved
memory and actual cache bytes. Short output budgets may be dominated by prefill.

The first p256 mismatch was sensitive to attention backend choice. Its math
control passed, but the residual p512-o64 disagreement remains unresolved.
At the user's request the stage is closed with that limitation; no further
reruns, token repairs, exclusions or equality-threshold changes are requested.

The separately supplied `kv_cache.json` describes **Qwen2.5-1.5B-Instruct**, with
another experiment schema. It is retained in place and excluded from these
Qwen3-4B adapter results to avoid attributing another model's measurements to ours.
