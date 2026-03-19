---
name: Final Dashboard Audit
overview: Completed a read-only final audit of dashboard information integrity and end-to-end process flow. The plan below captures confirmed issues, risk prioritization, and a safe remediation sequence for your approval.
todos:
  - id: fix-batch-write-policy
    content: Prevent misleading duplicate prediction rows for unchanged batch_id writes in prediction service.
    status: pending
  - id: fix-gate-and-suppressed-semantics
    content: Preserve gate zeros and stop recomputing direction from prob when row is suppressed/unavailable.
    status: pending
  - id: fix-market-hours-and-loader-consistency
    content: Make market-hours fallback strict and ensure snapshot fallback loaders always apply column adaptation.
    status: pending
  - id: centralize-threshold-source
    content: Use one canonical threshold source across all dashboard panels/charts instead of scattered defaults.
    status: pending
  - id: final-verify
    content: Run synthetic boundary sanity checks and live smoke checks for dashboard consistency.
    status: pending
isProject: false
---

# Final Dashboard and Pipeline Audit

## Confirmed Status

- Core threshold consistency work is mostly in place (several prior issues are fixed).
- Remaining risks are concentrated in edge-case integrity, fallback semantics, and display/process consistency.

## Highest-Risk Problems (Fix First)

- **Duplicate prediction rows for same batch** in [prediction_service.py](/workspace/Final_production_model/prediction_service.py) (`batch_id` can repeat when file mtime changes without true new batch).
- **Gate fallback bug for suppressed rows** in [theta_dashboard_v4_modern.py](/workspace/Final_production_model/theta_dashboard_v4_modern.py): `0` gates can be coerced to `1.0` by `or 1.0` pattern.
- **Suppressed direction can be recomputed from prob** in multiple dashboard panels, causing semantic drift from true suppressed state.
- **Market-hours fallback leak** in [theta_dashboard_v4_modern.py](/workspace/Final_production_model/theta_dashboard_v4_modern.py): market-hours filter can return unfiltered rows when filtered result is empty.

## Medium-Risk Integrity/UX Problems

- Hardcoded threshold fallbacks (`0.36`) spread across panels instead of one canonical source.
- Snapshot fallback loaders do not consistently run column adaptation in all paths.
- Divergence chart docs/comments still partially describe old `0.5` semantics despite newer threshold-aware behavior.
- Missing-vs-zero display ambiguity in some summary stats (`0.0` may appear as real value).

## Low-Risk Cleanup

- Inconsistent/old comments for market-hour window in a few docstrings/comments.
- `run_all.py` does not expose prediction polling interval.
- Minor formatting/label clarity opportunities (e.g., per-strike vs aggregate GEX wording).

## Remediation Sequence

- Patch correctness bugs first (duplicate-write policy, gate parsing, suppressed direction handling, market-hours fallback behavior).
- Centralize threshold source and propagate it to all chart/panel consumers.
- Normalize snapshot loader adaptation across all fallback paths.
- Tighten display semantics for missing values (`N/A` vs `0`).
- Final pass: docstrings/comments + quick synthetic sanity tests for boundary cases.

## Verification After Fixes

- Synthetic checks for threshold boundary (`prob=0.34/0.36/0.40`) across all major panels.
- CSV integrity checks: unique-write policy and no misleading duplicate batch artifacts.
- Market-hours tests: out-of-session data should not reappear via fallback.
- Dashboard smoke test on live run: all status cards, divergence chart, and model panels aligned.

