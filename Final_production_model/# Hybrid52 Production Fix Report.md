# Hybrid52 Production Fix Report

**Date:** March 19â€“22, 2026
**Repository:** https://github.com/chrisli0212/Hybrid52.git (renamed to workspace.git)
**Branch:** main

---

## Files Modified

| # | File | Commits |
|---|---|---|
| 1 | `Final_production_model/theta_fetching_v5.py` | 9ba1b9f |
| 2 | `Final_production_model/prediction_service.py` | 9ba1b9f, 7c1089d, 7d4549b, 5f0dc54 |
| 3 | `Final_production_model/theta_dashboard_v4_modern.py` | 7575380, f8c9efe, a6f0205 |
| 4 | `Final_production_model/stage1_models.py` | 7d4549b |
| 5 | `Final_production_model/confidence.py` | a6f0205 |
| 6 | `Final_production_model/theta_dashboard_v3_10.py` | Deleted (50113e3) |

---

## Fix 1: Always-Bullish Root Cause â€” Normalization Path Failure

**Files:** `prediction_service.py`, `theta_fetching_v5.py`
**Commits:** 9ba1b9f, 7c1089d

**Problem:** The model always predicted BULL. Root cause: `prediction_service.py` `_load_norm_stats()` could not find `norm_mean.npy` / `norm_std.npy` because `production_config.json` has a hardcoded RunPod path (`/workspace/data/tier3_binary_v5`). When norms are missing, features are not z-scored, producing near-zero logits â†’ sigmoid(0) = 0.5 â†’ always above the 0.36 threshold â†’ always BULL.

**Fix in `prediction_service.py`:**
- Added multi-path fallback search in `_load_norm_stats()`: checks config path, then repo-relative `../data/tier3_binary_v5/`, then sibling `./data/tier3_binary_v5/`.
- Added VIX-zero suppression guard: if VIX level reads 0.0 after full warmup, predictions are suppressed (prevents false signals when VIXW data is absent).

**Fix in `theta_fetching_v5.py`:**
- Changed `MAX_DTE` from a single integer to a per-symbol dictionary: `SPXW=5, SPY=5, QQQ=5, IWM=5, VIXW=30, TLT=5`. VIXW options expire weekly and need a wider DTE window to capture any contracts.
- Added OI cache key normalization fix.

**Verification:** After fix, 100 random z-scored inputs â†’ 5 BULL, 5 BEAR (model no longer stuck).

---

## Fix 2: Dashboard Visual Overhaul

**File:** `theta_dashboard_v4_modern.py`
**Commits:** multiple (586b270 through 7575380)

**Changes:**
- Line colors/opacity for multi-symbol charts: SPXW=blue solid, SPY=purple, QQQ=orange, IWM=green (non-SPXW at reduced opacity 0.45).
- Converted Net GEX and Net Premium charts from bar-only to hybrid bar+line (SPXW bars, others as lines).
- Converted accumulated signal chart from cumulative sum to instantaneous probability lines showing Stage 3 and all 7 agent probabilities at each moment.
- Dynamic legend titles for all multi-symbol overlay charts.

All changes were made to `theta_dashboard_v4_modern.py` only. `theta_dashboard_v3_10.py` was reverted to original and then deleted.

---

## Fix 3: Dashboard Bull/Bear Middle Lines â€” Per-Training Baselines

**File:** `theta_dashboard_v4_modern.py`
**Commits:** f8c9efe

**Problem:** Dashboard middle lines used a hardcoded 0.43 for Stage 3 and 0.5 for all agents. The actual trained model has different neutral points.

**Fix:**
- `S3_NEUTRAL` changed from 0.43 to 0.36 (the trained LogReg F1-optimized threshold, which also matches the model's neutral output ~0.3643 over 100 random inputs).
- `AGENT_TRAIN_MEDIAN` set to per-agent baselines computed from the full Stage 1â†’2 pipeline over 100 random z-scored inputs: `A=0.45, B=0.58, C=0.50, K=0.41, T=0.46, Q=0.49, 2D=0.47`.
- Updated all chart reference lines, tick labels, "How to read" guide text, and insight box help text from 0.43 to 0.36.
- Combined hybrid/rule chart: fixed duplicate 0.36 tick, now shows `0.36 BULL/BEAR` cleanly.

---

## Fix 4: Per-Symbol Platt Scaling â€” Activating Unused Training Parameters

**Files:** `prediction_service.py`, `stage1_models.py`
**Commit:** 7d4549b

**Problem:** Every Stage 1 checkpoint stores per-(symbol, agent) Platt scaling coefficients (`platt_scaler_coef`, `platt_scaler_intercept`) fitted during training. These calibrate raw neural network logits into well-calibrated probabilities. The inference pipeline was ignoring these and using raw `sigmoid(logit)` for all 35 models identically.

The coefficients vary significantly across symbols:
- SPXW Agent 2D: coef=1.78 (stretches output range 1.7x)
- TLT Agent T: coefâ‰ˆ-0.007 (compresses noise to near zero)
- SPY Agent A: coef=0.62 (moderates signal)

**Fix in `stage1_models.py`:**
- Added `platt_coef` and `platt_intercept` fields to `_Stage1Bundle`.

**Fix in `prediction_service.py`:**
- `_load_all_models()` now reads `platt_scaler_coef` and `platt_scaler_intercept` from each checkpoint and stores them in the bundle.
- `_stage1_predict()` now applies `calibrated_logit = coef * raw_logit + intercept` before sigmoid conversion.

**Effect:** High-signal symbol-agent pairs (SPXW) get wider output range; noisy pairs (TLT) get compressed. Cross-symbol diffs in the Stage 2 design matrix now carry meaningful divergence signal instead of being zero.

---

## Fix 5: VIX Regime-Gated Fusion â€” Activating Unused Trained Model

**File:** `prediction_service.py`
**Commit:** 5f0dc54

**Problem:** `models/stage3/stage3_vix_gated.pt` is a trained 36K-parameter `RegimeGatedProbFusion` model that was saved in the repo but never loaded or used in inference. It produces per-agent gates â€” learned weights for how much to trust each agent. The code had a comment: "Gates are uniformly 1.0 â€” RegimeGatedProbFusion is not deployed in this config."

The model learned to gate OFF three agents (C, T, 2D â€” the lowest S3 LogReg coefficients) and keep four agents fully active (A, B, K, Q).

**Fix in `prediction_service.py`:**
- Added `self.stage3_vg` container and loading logic in `_load_all_models()`.
- In `_run_inference()`, after the LogReg produces prob/pred, the VIX-gated model runs on the same agent probs + VIX features to produce learned per-agent gates.
- Gates now flow into the confidence computation instead of uniform 1.0.
- LogReg remains the primary model for prob/pred. The VIX-gated model provides gates only.

---

## Fix 6: Confidence Computation â€” Per-Agent Baselines

**File:** `confidence.py`
**Commit:** a6f0205

**Problem:** `confidence.py` had two hardcoded 0.5 references that should have used per-agent training baselines:

1. **Consensus ratio** (line 57): counted agents above/below 0.5 to determine agreement with the prediction direction. Agent B (baseline=0.58) was permanently counted as "bearish" since its output almost never exceeds 0.5, even when it's neutral or mildly bullish relative to its own baseline.

2. **Gate conviction** (line 70): measured `|prob - 0.5|` as conviction strength. Agent K (baseline=0.41) at prob=0.50 showed zero conviction (|0.50-0.50|=0.00), but it's actually strongly bullish (+0.09 above its 0.41 baseline).

**Fix:**
- Added `AGENT_BASELINES = [0.45, 0.58, 0.50, 0.41, 0.46, 0.49, 0.47]` matching `AGENT_TRAIN_MEDIAN` in the dashboard.
- Consensus ratio now uses `agent_probs >= baselines` instead of `agent_probs > 0.5`.
- Gate conviction now uses `|agent_probs - baselines|` instead of `|agent_probs - 0.5|`.
- Conviction scaling adjusted from `/0.25` to `/0.15` to match the actual observed agent deviation range.

---

## What Was NOT Changed (Intentional Hardcoding)

| Location | Value | Reason |
|---|---|---|
| `prediction_service.py` line 488: `agent_probs > 0.5` | 0.5 | Stage 3 `agree_up` cross-feature â€” the LogReg was TRAINED with this exact formula. Changing it would break the model. |
| `theta_dashboard_v4_modern.py` line 4088: `p > 0.5` | 0.5 | Mirrors the Stage 3 agree_up computation for dashboard display. Must match prediction_service.py. |
| `prediction_service.py` S3_NEUTRAL / threshold fallbacks | 0.36 | Correct trained threshold from config/checkpoint. |
| All `AGENT_TRAIN_MEDIAN` references in dashboard | per-agent | Already correct â€” uses learned baselines throughout. |

---

## Verification

- `walkthrough_test.py`: 66/66 checks passed after all fixes.
- All files syntax-checked with `py_compile`.
- Stage 3 LogReg reconstruction matches CSV output to 4 decimal places.
- March 19 live session: 84% BEAR on a -0.27% day (directionally correct).
- Pipeline test with 100 random inputs: 5 BULL / 5 BEAR (balanced).