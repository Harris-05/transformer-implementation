# Final local delivery verification

- Full project regression: **88 tests passed**, `python -m unittest discover -v`.
  Includes actual miniature Qwen cache/speculative/app integration, strict
  patient/target separation, completion masks, attention, metrics, notebook byte
  restoration, API/job/export behavior and explicit proxy-origin rejection.
- Browser checks at **1440, 768 and 390 px**: fonts loaded, blank-input recovery,
  synthetic-input filling, correct criterion counts, tabs, no horizontal overflow
  and no console exceptions. Saved under `artifacts/app_browser_final/`.
- Fresh independent review reported no important blockers in the app, final
  evaluator or paper. The later explicit proxy-origin change has its own passing
  regression. No wildcard origin acceptance was introduced.
- Follow-up review found that the optional full-model verification helper saved
  failed checks but exited successfully. It now preserves the failed evidence,
  raises on any failed check and still permits downloading diagnostics. Two
  red/green regression tests cover that behavior; the full 88-test suite passes.
- Final held-out archive verified by `verify_final_results.py`: raw source and
  record hashes, eight test patient identities/counts, 206 predictions per phase,
  selected adapter/source identities, exact inputs/targets, strict raw parsing,
  evidence text and every recomputed metric/comparison.
- Speculative archive: 54/54 trials and six preflight cases pass exact equality;
  aggregate recomputation and source identities match. KV's remaining failed case
  is preserved. Attention's numerical agreement and measured slowdown remain.
- IEEE PDF rebuilt with actual final test results, all references resolved, no
  overfull boxes. Title/author and all held-out tables are present; no pending-test
  paragraph remains. Selected pages visually inspected after rendering.
- Packaging verifies ZIP CRCs and a SHA256 manifest for every source/result file.
  Source archive excludes large weights; the complete local archive includes the
  original selected adapter ZIP. No secret/environment/cache files are included.

The full-size browser-server GPU check is a separate user-run artifact, not
implied by tiny tests. The supplied GPU check/startup cells reuse the existing
adapter, expose actual app behavior and perform no training. Repository and demo
URLs were supplied by the user and inserted as clickable links in the PDF,
LaTeX, README and checklist. Their external content/accessibility was not verified
by the unavailable web fetch; no claim of reviewing the video is made. Syncing
the repository and actual submission remain user actions.
