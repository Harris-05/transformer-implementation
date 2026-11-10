# Five-to-seven-minute application demonstration

Record the real browser application after verifying full-size model loading on
the intended host. A recording of notebook cells or only static plots does not
meet the standalone application requirement. Do not describe the synthetic input
button as a model result: it fills inputs only.

| Time | Show and explain |
|---|---|
| 0:00–0:40 | Name, problem, local browser interface and educational notice. Identify target, selected adapter and host. |
| 0:40–2:00 | Enter a new note and both criterion types. Run the actual model; show label, missing-information handling, citation IDs and source highlighting. |
| 2:00–2:40 | Discuss an insufficient-information or unsupported case. Show raw output validation; do not treat absence as a negative finding. |
| 2:40–3:15 | Export readable HTML and JSON. Open the HTML and show exact evidence and model/adapter/protocol provenance. |
| 3:15–4:25 | In Inference lab, run live cached/uncached comparison on the same input. Show equality, latency, throughput and memory scope. Optional draft mode can take longer; preserve any slowdown/mismatch. |
| 4:25–5:30 | Show paper's verified training/validation and final-test tables, attention trade-off and speculative acceptance results. Explain the KV residual mismatch. |
| 5:30–6:30 | Demonstrate empty-input recovery, explain fixed split/no leakage, acknowledge AI/libraries, and identify the repository/adapter download links. |

Before recording, use an input short enough for the frozen 1,024-token budget.
First weight downloads are not normal inference latency: complete loading before
the timed walkthrough, while retaining the recorded host/resource requirements.

## Questions to be ready to answer

1. **Why patient grouping?** Multiple annotations reuse a patient's note; a row
   split would leak patient-specific information across groups. Sort/shuffle seed
   42 and manifest counts make the required split reproducible.
2. **What is trained?** Low-rank adapters, not all base weights. Prompt/padding
   tokens have loss label -100; expert JSON completion tokens receive loss.
3. **Why epoch one?** Lowest validation completion loss selects checkpoint 76.
   Later loss rises, so the final trained step is not automatically best.
4. **Why can accuracy rise while inclusion macro-F1 falls?** Accuracy weights
   records; macro-F1 weights fixed classes equally. Rare not-included performance
   worsened while frequent categories improved.
5. **What does the KV cache hold?** Per-layer key/value tensors for processed
   prefix tokens. Absolute positions and past-plus-current masks remain correct;
   final emitted token is not forwarded, so cache length is prompt+output−1.
6. **Why a mismatch with greedy decoding?** Different computation shapes and
   finite precision can perturb close logits and change argmax. Math SDPA fixed
   one case but not the whole grid; the remaining failure is retained.
7. **What makes tiled softmax stable?** Running row maxima rescale both previous
   numerator and denominator before merging the next key/value block.
8. **Why is tiling slower?** Educational PyTorch loops incur repeated launches
   and allocations; memory savings do not imply fused-kernel speed.
9. **How is speculation verified?** Target logits are shifted to predict each
   proposal. Accept only the consecutive match prefix, correct the first reject,
   emit a bonus only when allowed, then crop/synchronize both caches.
10. **Why do fewer target forwards still lose?** Draft cost, rejected proposals,
    low acceptance and Python synchronization can exceed target savings.
11. **Why separate FP16 and FP32 studies?** Speculative verification used a
    precision control after near-tie sensitivity. Only identical protocols are
    compared in each experiment; old FP16 times are not the FP32 baseline.
12. **What is an explanation here?** A category description tied to original
    sentence text, not an invented generative chain. A valid citation still does
    not prove correct clinical interpretation.
13. **What does an empty evidence set mean?** Unknown, not applicable or missing
    support may occur. It is not automatically proof of inclusion or exclusion.
14. **What must you change for an unseen case?** Nothing in the model/prompt.
    Enter it through the app, validate the context budget and inspect raw output.
15. **What is not established?** Clinical safety, final enrollment, general
    production speedups or full-size laptop fit. Held-out metrics are now available;
    they do not establish clinical safety or universal performance.
