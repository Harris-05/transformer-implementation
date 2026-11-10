# Verified speculative decoding experiment

The uploaded `speculative_results.zip` was preserved byte for byte in
`results/speculative_v1`. Verified the canonical manifest signature, all eight
source hashes, 54 unique trial identities, all six saved HF/ordinary/speculative
preflight token sequences, every trial against its greedy reference, and every
reported aggregate by recomputing `summarize_trials` from raw trials.
All six cases and all 54 timed trials pass exact token equality. No GPU forward
passes were repeated during this independent artifact verification.

Target: selected Qwen3-4B adapter with NF4/double-quantized weights, FP32 target
compute; draft: pinned Qwen3-0.6B in FP16. Math SDPA and highest FP32 matmul
precision apply throughout. Device: Tesla T4, CUDA 13.0, torch 2.14.1+cu130,
Transformers 5.15.1, PEFT 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0.
This controlled FP32-target study is separate from the earlier FP16 KV and
classification experiments. Both models are resident in every measured mode.

| Case | Greedy seconds | K=2 seconds | K=2 speedup | K=2 acceptance | K=4 seconds | K=4 speedup | K=4 acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128/16 stress | 2.871 | 2.181 | 1.316x | 83.3% | 2.109 | 1.361x | 81.3% |
| 256/32 stress | 4.814 | 5.536 | 0.870x | 34.2% | 7.383 | 0.652x | 27.0% |
| 512/64 stress | 9.675 | 9.002 | 1.075x | 76.5% | 10.186 | 0.950x | 56.2% |
| Clinical 938 | 3.496 | 4.060 | 0.861x | 36.8% | 4.431 | 0.789x | 31.0% |
| Clinical 6 | 3.361 | 3.075 | 1.093x | 61.5% | 3.476 | 0.967x | 42.9% |
| Clinical 243 | 3.898 | 4.300 | 0.907x | 36.8% | 4.772 | 0.817x | 31.0% |

The maximum observed speedup is 1.361x, but the minimum is 0.652x. K=2 is faster
in three of six cases; K=4 is faster only for the shortest stress workload.
These six prompts are a small development study, not a representative latency
distribution. Do not average ratios into a general production speedup claim.

Lower draft acceptance wastes proposals and increases draft work even as target
forward calls decrease. In the 256/32 case, greedy uses 32 target forwards;
K=4 uses 16 target forwards but 64 draft forwards and is slower overall. On the
shortest stress case K=4 reduces target calls to four and outperforms greedy.
Python acceptance logic, synchronization and rollback also contribute latency.
The measured outcomes support an acceptance/workload-dependent trade-off rather
than a guarantee that larger drafts or larger proposal blocks are faster.

The comparison retains synchronized generation latency, actual output tokens/s,
pooled accepted/proposed counts, actual target/draft forwards and allocator peaks.
Memory includes both resident models even for ordinary greedy. Consult raw data
for min/max, reserved memory and cache counts; this is not a target-only memory
comparison. Controlled stress ignores EOS; complete clinical prompts use EOS.

Task 5 implementation and its documented GPU experiment are complete. No test
patients were used by this optimization study.
