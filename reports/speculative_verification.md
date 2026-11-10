# Speculative decoding: local verification and Colab handoff

Implementation and local validation are complete. Actual paired-model GPU
measurement remains pending in `colab_speculative.ipynb`.

- The full suite passed 71 tests. Two additional notebook retry/Drive backup
  regressions passed after reproducing and fixing `TRAINING=None` recovery:
  73 verified tests across those runs.
- `artifacts/speculative_cpu_final/metrics.json` records 18/18 completed miniature
  FP64 CPU trials with exact ordinary/speculative token equality. All three
  preflight cases also match the HF reference. Final source hashes match this run.
  Random miniature models establish implementation behavior, not full-model speed.
- `reports/speculative_preparation.json` verifies the real selected adapter and
  pinned tokenizers offline without loading weights. Configured fast tokenizer
  backends and all 151669 token mappings match; model vocabularies are 151936.
- The notebook schema, five executable cells and 24 embedded source hashes were
  checked. Embedded data contains only 206 validation records from eight patients;
  no held-out test examples or training records are included.
- Frozen EDA, model preparation, baseline, training and KV decoder hashes match
  the saved training/KV manifests. Earlier experiment artifacts remain unchanged.
- Fresh review found and resolved two concrete issues: reruns overwriting failed
  preflight evidence, and adapter discovery failing after an earlier missing-file
  attempt. Regression tests cover both recovery behaviors.

Before opening a fresh Colab runtime, run `colab_save_adapter_to_drive.py` in the
current runtime if the selected adapter exists only under `/content`. Then open
the new notebook, select GPU, run its five steps and return
`speculative_results.zip`. It reuses the trained adapter and performs inference
only. Code and development inputs are embedded; there is no upload widget.

The real experiment compares ordinary greedy and explicit speculative K=2/K=4
on six cases with three repetitions (54 timed trials), after strict preflight.
Both paths share the same 4-bit target with FP32 computation and math SDPA; the
draft uses FP16. Both models remain resident in every mode. Numerical failures,
slowdowns and memory overhead are retained. GPU speedup and fit are not yet known.

Machine-readable checks are in `reports/speculative_verification.json`.

## Colab step 3 export correction

The first actual Colab preparation exposed a byte-preservation bug in the
notebook builder. `read_text()` normalized the Windows validation file's 206
CRLF endings to LF. Earlier embedded-content checks compared normalized text
and therefore missed this boundary; local preparation used the original bytes.

The training checksum is
`0a6ad872b4b22bb5f69ad2f3db163dc6aff75f421cb71e199aee328ad881e2e3`;
the broken LF-only export hashed to
`757b24526302c082f6877b193eb41a184da64315555790c5fcabad606ff08718`.
The builder now decodes raw bytes directly and restores their exact bytes.
A regression executes the actual notebook restore cell and checks its output
against both the training checksum and the original file bytes. It failed
before this fix and passes afterward.

`colab_repair_speculative_cell3.py` repairs an existing runtime only when a
line-ending variant exactly matches the saved training checksum. Content
changes still fail without writing anything; the benchmark's integrity guard
is unchanged. All four notebook/repair tests passed after regeneration. The
previous 71-test run plus these four tests verify 75 tests across those runs.
