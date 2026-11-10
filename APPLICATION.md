# TrialScreen browser application

The application is implemented in `app.py`, `app_model.py`, `screening.py` and
`web/`. It loads the selected **Qwen3-4B-Instruct-2507 LoRA adapter**, not an API
model. Opening the browser does not download or load weights; Load model or Run
screening starts the real load. No training occurs.

## Windows startup

1. Use Python 3.13 or 3.14. For a CPU-only host, double-click
   `Setup_TrialScreen_CPU.cmd` once to create a local environment and install
   dependencies. This needs an internet connection and several GB of disk.
2. Restore the eight selected training/adapter files under
   `results/finetuning_qlora_v1`: the two metadata JSON files and the six files
   listed by `training_summary.json` under `adapter/`. The original supplied
   `finetuning_results.zip` contains them. This workspace already has them.
3. Double-click `Start_TrialScreen.cmd`. The browser opens
   `http://127.0.0.1:8765`. Keep the host process running while using it.
4. Enter a record, inclusion requirements and/or exclusion conditions. Use one
   statement/criterion per line. A synthetic example fills inputs only; it does
   not supply fake model results. Click Run screening for actual inference.
5. Inspect predictions and exact sentence quotes. Click a sentence ID to highlight
   its original text. Download HTML or JSON with Export report.
6. In Inference lab, use the current record's first criterion to run cached and
   uncached decoding, optionally with the Qwen3-0.6B draft. Download the raw live
   comparison including token sequences and all three repetitions.

## Host resources and protocols

CPU mode loads unquantized FP32 base weights: approximately **16 GB for weights
alone**, plus adapters, model-loading overhead, activations, caches and optional
draft. A machine with 24-32 GB RAM is a more realistic starting point; memory
use and usable latency have not been measured for the full CPU model. An 8 GB
laptop is unsuitable for this implementation's CPU loader. CPU execution may be
slow. Do not interpret the working UI as evidence that a host can fit the model.

On a CUDA host, install the appropriate PyTorch CUDA build and the pinned
requirements, then use `python app.py --device cuda`. The base uses NF4/double
quantization with FP32 computation and math SDPA, matching the controlled
speculative study; the draft uses FP16. Its measured experiment peaks are in
`results/speculative_v1`, including both models. Auto chooses CUDA when available.
An explicit custom adapter path is supported by `--training-output PATH`.

The application uses **FP32 target compute** for consistent live comparisons.
Saved classification evaluation uses NF4/FP16 with automatic SDPA. Consequently,
published classification scores are not asserted as application-protocol scores.
CPU unquantized FP32 is another deployment protocol. All live compared modes use
the same resident target; optional draft weights remain resident for all modes.

The laptop used to implement this app has CPU-only PyTorch. We tested actual tiny
Qwen3 forward integration, API behavior and real desktop/mobile browsers, but
have not loaded the full-size target in this local server. Verify full-size model
loading and an unseen case on the intended demonstration host before presenting.
The mandatory application is a local browser application; the supplied Colab
notebooks are for experiments, not a substitute for its browser workflow.

## Behavior and recovery

- Inputs consist only of note/type/criterion text and optional display references.
  Existing original sentence IDs are preserved. Unknown outcomes are not collapsed.
- At most 12 criteria and 24,000 note characters; actual token budget is checked
  with the frozen tokenizer. Overlength prompts fail without truncation, before
  any criteria are inferred. Shorten the input and retry through the browser.
- Malformed outputs and invalid citations remain visible. Explanations quote
  exact record text; the model is not asked to invent extra medical reasoning.
- The preliminary overall outcome is an explicit rule-based aggregate, not a
  labeled or clinically validated trial enrollment decision.
- Only one model operation runs at a time. Failure releases the job slot and is
  shown in the browser. First downloads can take several minutes.
- The last five job results remain in memory. Patient records are not logged or
  persisted automatically. Exported files contain the entered record: choose
  their storage location deliberately.
- Only loopback is served, with local Host/Origin checks, limited JSON bodies,
  a static path allowlist, escaped HTML and a restrictive content security policy.
  This is a local research server, not an internet-facing production deployment.

## Verified implementation checks

`python -m unittest -v test_screening test_app test_app_model` exercises input
numbering, evidence extraction, missing information, invalid responses, report
escaping, API validation, asynchronous export and a real miniature Qwen3 live
comparison. Browser evidence under `artifacts/app_browser_final/` verifies fonts,
empty-input recovery, example filling, navigation, console errors and horizontal
overflow at widths 1440, 768 and 390. No full-size model output is fabricated in
those browser checks.

Official implementation references: [Python HTTP server](https://docs.python.org/3/library/http.server.html),
[PEFT model loading](https://huggingface.co/docs/peft/en/package_reference/peft_model),
[Transformers bitsandbytes quantization](https://huggingface.co/docs/transformers/v5.15.1/en/quantization/bitsandbytes).
