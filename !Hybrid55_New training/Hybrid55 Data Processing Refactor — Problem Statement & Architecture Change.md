# Hybrid55 Data Processing Refactor — Problem Statement & Architecture Change
## Executive Summary
This document explains the problems found in the original `master_extractor_v2.py` / `feature_config_v2.py` pipeline, why they caused training failure across 4 out of 7 agents, and exactly what was redesigned in the new per-agent extractor architecture under `hybrid55_preprocessing/agents/`. It is written as a handoff spec for an IDE-based AI coding agent to continue implementation.

***
## Part 1 — The Problems We Faced
### 1.1 Feature Dimensionality Was Inconsistent Across Four Files
The single most dangerous problem was that the feature count was defined in four different places and they all disagreed:

| File | Claimed Feature Count | Reality |
|---|---|---|
| `feature_config_v2.py` | `TOTAL_FEATURES = 311` | Hybrid55 target |
| `master_extractor_v2.py` docstring | `"Produces 286 flat features"` | Outdated Hybrid52 number |
| `__init___v2.py` docstring | `"327 features (270 + 57 Phase 1)"` | Wrong — Phase 1 is 55, not 57 |
| `get_feature_names()` docstring | `"Get all 286 historical-mode feature names"` | Direct contradiction inside `feature_config_v2.py` where `TOTAL_FEATURES = 311` |

The `assert idx == 270` checkpoint in `extract_flat_features()` passes correctly, but then OHLC (25 dims) is added, reaching 295. The `csv_derived.extract()` block appends 16 more after `nan_to_num()` — making the live output 311. However, `TOTAL_FEATURES = 311` was only added to `feature_config_v2.py`, and no agent's training code or `feature_subsets.py` was ever updated to reflect this.

**Effect:** Agents trained on 295-dim vectors and then fed 311-dim vectors at inference. Or vice versa. Shape mismatch = crash or silent index shift.

***
### 1.2 A Monolithic Extractor Means One Failure Corrupts All Agents
`MasterFeatureExtractor.extract_flat_features()` runs all 9 extractor calls sequentially, writing to a shared `features` array with a shared `idx` pointer:

```python
greek_features = self.greek_extractor.extract(greek_df)
features[idx:idx + 75] = greek_features
idx += 75

gamma_vanna = self.gamma_extractor.extract(greek_df)
...
```

There is **no isolation** between extractor blocks. If `gamma_extractor.extract()` raises an exception or returns the wrong length, the remaining 200+ features are either shifted by the wrong offset, partially overwritten, or the entire call raises and returns nothing. All 7 agents that consume the master vector are affected simultaneously — there is no way to tell from the error message which agent's feature was the original problem.

**Effect:** A single schema mismatch in the OHLC extractor (added in Hybrid55) caused a silent 25-feature misalignment in dims 286–310 for every agent. Training loss showed no alert because the shape was technically valid — just wrong.

***
### 1.3 The `GAMMA_EXPOSURE` Extractor Returns 50 Features But the Config Registers Only 30
`feature_config_v2.py` defines `GAMMA_EXPOSURE` with `num_features=30` and `VANNA_CHARM` with `num_features=20` — two separate groups that together total 50.

But `GammaExposureExtractor.extract()` returns a single concatenated array of 50 values. The master extractor slices it back apart using the config numbers:

```python
gamma_vanna = self.gamma_extractor.extract(greek_df)
n_gamma = FEATURE_GROUPS[FeatureGroup.GAMMA_EXPOSURE].num_features   # 30
n_vanna = FEATURE_GROUPS[FeatureGroup.VANNA_CHARM].num_features       # 20
features[idx:idx + n_gamma] = gamma_vanna[:n_gamma]
features[idx:idx + n_vanna] = gamma_vanna[n_gamma:n_gamma + n_vanna]
```

If either number changes in the config without changing the extractor (or vice versa), the slice is silently wrong. The individual extractors for Agent C and Agent K directly use the `gamma_exposure.py` extractor — a boundary change breaks them without any error.

***
### 1.4 `active_chain_filter.py` Was Never Called Before Extraction
`active_chain_filter.py` filters the options chain to liquid delta-range contracts (calls Δ 0.2–0.9, puts Δ −0.9–−0.2), removing deep-OTM and illiquid strikes before any greek aggregation.

However, `MasterFeatureExtractor.preprocess_greek_df()` only removes dead column names — it never calls `filter_active_chain()`. Every extractor downstream (GreekFeatureExtractor, GammaExposureExtractor, IVSurfaceExtractor) therefore processes the full chain including deep-OTM strikes with near-zero vega and zero open interest.

**Effect:** Delta-bucket aggregations in `greek_features.py` become diluted by these near-zero rows. The `deep_otm_gamma` and `deep_itm_delta` bucket features appear to have signal but are in fact averaging over hundreds of worthless strikes. This is a major driver of the 45%+ dead feature rate found in the Stage 1 dead-feature audit.

***
### 1.5 `csv_derived` Features Were Never NaN-Cleaned
In `extract()`, `nan_to_num()` is applied to `flat_features` before `csv_derived` is appended:

```python
flat_features = np.nan_to_num(flat_features, nan=0.0, ...)
csv_feats = self.csv_derived.extract(greek_df)
flat_features = np.concatenate([flat_features, csv_feats])   # NaN cleaning MISSED
```

This means dims 295–310 (`lambda_mean`, `dist_atm_mean`, `spread_pct_*`, `oi_mean`, etc.) pass raw NaN/inf values directly into training. Any row where `lambda` or `spread_pct` was unavailable from the CSV source propagated NaN into the model's input tensor.

***
### 1.6 `HISTORICAL_MODE` Flag Created a Training–Serving Dimension Mismatch
`HISTORICAL_MODE = True` in `feature_config_v2.py` causes `include_phase1 = include_phase1 and not HISTORICAL_MODE` to evaluate to `False` inside the master extractor's `__init__`.

This means:
- **Training (BigQuery historical data):** `HISTORICAL_MODE=True` → 295 flat features output
- **Live deployment:** `HISTORICAL_MODE=False` → 350 flat features output (55 Phase 1 features added)

The model trained on 295-dim vectors is fed 350-dim vectors at inference. All agents crash or produce random output. This is a classic **training–serving skew** problem.

***
### 1.7 Sequence Length Was 20 — Too Short for the 30-Minute Prediction Horizon
`CHAIN_2D_CONFIG["n_timesteps"] = 20` and `SEQ_LEN = 20` in the tier3 builder.

The prediction horizon was 30 minutes. The model was asked to predict a move it had less context than the duration of the event itself — a lookback/horizon ratio of 0.67×. Industry standard for intraday options models is a lookback of 2×–8× the prediction horizon; for 30-minute targets, a minimum of 60 bars and ideally 120 bars.

Additionally, Agent H's `index out of range in self` crash at Stage 1 inference was caused directly by this: the LSTM/attention layers were initialized with a different sequence length than `SEQ_LEN=20` provided.

**Resolved change:** `SEQ_LEN` was updated to **120** bars. Graceful padding is applied when fewer than 120 bars are available (use available bars, zero-pad to 120). The prediction target is **5 minutes** only (not 15 or 30 minutes).

***
### 1.8 Stage 3 Meta-Model Used the Wrong Agent Namespace
`infer_stage3_meta.py` imported from `hybrid51_utils`/`hybrid51_models` and defined `ALL_AGENTS = ['A','B','C','K','T','Q','2D']` — the old Hybrid51 lineup.

When Stage 3 runs after Hybrid55 Stage 1 agents, it loads the wrong checkpoint files and assembles a probability vector of the wrong shape. The entire ensemble fusion layer operates on stale Hybrid51 outputs.

***
## Part 2 — Root Cause of Degenerate Agent Outputs
The combined effect of the above issues produced the following Stage 1 result pattern (smoke window Sep 25 – Oct 15 2024, 15 trading days):

| Agent | Accuracy | AUC | Prob. Std | Diagnosis |
|---|---|---|---|---|
| Agent A | 49.1% | 0.543 | 0.0076 | Probability collapse |
| Agent B | 48.9% | 0.500 | 0.0119 | Probability collapse |
| Agent C | 51.7% | 0.515 | **0.0000** | Full collapse — constant output 0.528293 every sample |
| Agent K | 51.7% | 0.480 | 0.0039 | Degenerate always-UP prediction |
| Agent TQ | 52.1% | 0.515 | 0.0190 | Partial collapse, small variance |
| Agent H | — | — | — | **Hard crash:** `index out of range in self` |

**Agent C outputting exactly 0.528293 for every sample** is the clearest evidence: the model ignores its input entirely. This is the signature behaviour when a model receives near-zero variance inputs (dead features) — the network finds the optimal constant prediction is the class prior (label_up_ratio = 0.7246 in this smoke window).

The 72.46% accuracy achieved by Agents B, C, K, and 2D is trivially achieved by always predicting UP — `pred_pos_rate = 1.0`, `pred_neg_rate = 0.0` in all four cases. This is not a 72% model; it is a broken model that learned to ignore its features.

***
## Part 3 — What Was Changed: The Per-Agent Architecture
### 3.1 Design Principle
The refactored architecture separates three concerns that were previously entangled:

1. **Raw computation** (shared): `gamma_exposure.py`, `iv_surface.py`, `flow_volume.py`, etc. These files compute their feature blocks from DataFrames. They are never modified per-agent.
2. **Feature assembly** (per-agent): Each agent has its own `extractor.py` that calls only the shared extractors it needs, assembles the vector in the correct order, and asserts the final dimension.
3. **Feature validation** (per-agent): Each agent has its own `validator.py` that reports zero-rates per block, NaN counts, and raises alert logs.
### 3.2 New Directory Structure
```
!Hybrid55_New training/hybrid55_preprocessing/
│
├── extractors/                          ← SHARED (never duplicated)
│   ├── gamma_exposure.py
│   ├── iv_surface.py
│   ├── flow_volume.py
│   ├── greek_features.py
│   ├── cross_strike_time.py
│   ├── active_chain_filter.py          ← NOW called by every agent extractor
│   ├── data_validation.py
│   └── base_extractor.py              ← Abstract base with safe_extract() and _check_zeros()
│
├── agents/
│   ├── agent_a/
│   │   ├── feature_config.py          ← AGENT_A_DIM = 53, asserted at import time
│   │   ├── extractor.py               ← AgentAExtractor: calls shared, asserts shape
│   │   └── validator.py               ← validate_agent_a(): zero-rate, NaN per block
│   ├── agent_b/
│   │   ├── feature_config.py          ← AGENT_B_DIM = 311 (hist) / 366 (live)
│   │   ├── extractor.py
│   │   └── validator.py
│   ├── agent_c/
│   │   ├── feature_config.py          ← AGENT_C_DIM = 69 (Multi-Scale sequence)
│   │   ├── extractor.py
│   │   └── validator.py
│   ├── agent_k/
│   │   ├── feature_config.py          ← AGENT_K_DIM = 75 (pure Greek block)
│   │   ├── extractor.py
│   │   └── validator.py
│   ├── agent_tq/
│   │   ├── feature_config.py          ← AGENT_TQ_DIM = 70 (hist) / 95 (live)
│   │   ├── extractor.py
│   │   └── validator.py
│   ├── agent_h/
│   │   ├── feature_config.py          ← AGENT_H_SEQ_LEN = 120, AGENT_H_DIM = 165
│   │   ├── extractor.py               ← extract_sequence() with graceful padding
│   │   └── validator.py
│   └── agent_2d/
│       ├── feature_config.py          ← AGENT_2D_SHAPE = (5, 30, 120)
│       ├── extractor.py               ← CNN tensor builder, 120 timesteps
│       └── validator.py
│
└── master_extractor_v2.py             ← Becomes a thin router calling per-agent extractors
```
### 3.3 Key Safeguards Added in Each Agent
**Import-time assertion (feature_config.py):**
```python
AGENT_K_DIM = 75
# Enforced at import — crashes immediately if dims drift before any training loop
_EXPECTED = len(AGENT_K_FEATURE_NAMES)
assert _EXPECTED == AGENT_K_DIM, f"AGENT_K feature name list has {_EXPECTED} entries, expected {AGENT_K_DIM}"
```

**Per-extractor isolation (extractor.py):**
```python
def _safe_extract(self, fn, df, n_expected, group_name):
    try:
        result = fn(df)
        if len(result) != n_expected:
            self._alert(f"[SIZE MISMATCH] {group_name}: got {len(result)}, expected {n_expected}")
            return np.zeros(n_expected, dtype=np.float32)
        return result
    except Exception as e:
        self._alert(f"[EXTRACTOR FAIL] {group_name}: {e}")
        return np.zeros(n_expected, dtype=np.float32)
```

**Zero-field alert (validator.py):**
```python
def _check_zeros(self, block, group_name, threshold=0.5):
    zero_rate = (block == 0).mean()
    if zero_rate >= threshold:
        self._alert(f"[ZERO ALERT] {group_name}: {zero_rate:.1%} of features are zero")
```

**Active chain filter — now mandatory before any extraction:**
```python
def extract(self, greek_df, trade_df=None):
    active_df = filter_active_chain(greek_df)   # Always applied
    greek_block = self._safe_extract(self.greek_extractor.extract, active_df, 75, "GREEK")
    ...
```
### 3.4 Per-Agent Feature Dimensions
| Agent | Dim | Mode | Feature blocks used from master 311-dim |
|---|---|---|---|
| Agent A | 53 | EOD Theta snapshot | Standalone EOD features from `extract_agent_a_features.py` |
| Agent B | 311 (hist) / 366 (live) | 1-min intraday full | All 311 dims; +55 Phase 1 in live mode |
| Agent C | 69 | Multi-Scale sequence, 120 bars | `gamma_exposure[75:105]` + `vanna_charm[105:125]` + `iv_surface[125:141]` + `csv_dist_atm[273:276]` |
| Agent K | 75 | Greek specialist | `greek_by_strike[0:75]` — single contiguous slice |
| Agent TQ | 70 (hist) / 95 (live) | Trade & Quote flow | `flow_volume[150:180]` + `microstructure[180:200]` + `sentiment[250:270]` + Phase 1 (live) |
| Agent H | (120, 165) | LSTM sequence | 165-dim static per bar × 120 timesteps with graceful padding |
| Agent 2D | (5, 30, 120) | CNN tensor | Delta-binned chain × 120 timesteps |
### 3.5 Sequence Length: 20 → 120, Predict 5 Minutes
All sequence agents now use `SEQ_LEN = 120` (2 hours of 1-minute bars):

- If a trade date has fewer than 120 prior bars available (e.g., early morning), the agent uses however many bars are available and **zero-pads the remainder** from the front of the tensor (most-recent bars are always at the end).
- Prediction target is **5 minutes only**. The 15-minute and 30-minute targets are removed.
- `CHAIN_2D_CONFIG["n_timesteps"]` updated from 20 to 120.

Graceful padding example:
```python
def extract_sequence(self, snapshots: list):
    n_avail = len(snapshots)
    seq = np.zeros((SEQ_LEN, AGENT_H_FEAT_DIM), dtype=np.float32)
    start = SEQ_LEN - min(n_avail, SEQ_LEN)
    for i, snap in enumerate(snapshots[-SEQ_LEN:]):
        seq[start + i] = self._extract_one(snap)
    assert seq.shape == (SEQ_LEN, AGENT_H_FEAT_DIM), f"Shape mismatch: {seq.shape}"
    return seq
```

***
## Part 4 — What the IDE Coding Agent Needs to Do
The files in `!Hybrid55_New training/hybrid55_preprocessing/agents/` are the **skeleton**. The following concrete steps remain:
### Step 1 — Port Shared Extractors to `extractors/` Subfolder
Move (not copy) the existing files into the `extractors/` subfolder:
- `gamma_exposure.py` → `extractors/gamma_exposure.py`
- `iv_surface.py` → `extractors/iv_surface.py`
- `flow_volume.py` → `extractors/flow_volume.py`
- `greek_features.py` → `extractors/greek_features.py`
- `cross_strike_time.py` → `extractors/cross_strike_time.py`
- `active_chain_filter.py` → `extractors/active_chain_filter.py`
- `data_validation.py` → `extractors/data_validation.py`

Update all import paths in every `agents/agent_x/extractor.py` to point to `extractors.*`.
### Step 2 — Replace Stub Blocks in Agent B Extractor
Agent B's `extractor.py` has `_stub_block()` placeholders. Each stub must be replaced with a real call to the corresponding shared extractor:

```python
# Replace stub with real call:
greek_block = self._safe_extract(self.greek_extractor.extract, active_df, 75, "GREEK")
gamma_block = self._safe_extract(self.gamma_extractor.extract, active_df, 30, "GAMMA")
# etc.
```

The stub pattern is clearly marked with `# TODO: replace with shared extractor call` comments in the file.
### Step 3 — Fix `nan_to_num` Order in Master Extractor
In `master_extractor_v2.py`, move the `nan_to_num` call to **after** `csv_derived` is appended:

```python
# Before (wrong):
flat_features = np.nan_to_num(flat_features, nan=0.0, ...)
csv_feats = self.csv_derived.extract(greek_df)
flat_features = np.concatenate([flat_features, csv_feats])

# After (correct):
csv_feats = self.csv_derived.extract(greek_df)
flat_features = np.concatenate([flat_features, csv_feats])
flat_features = np.nan_to_num(flat_features, nan=0.0, posinf=0.0, neginf=0.0)
```
### Step 4 — Update `feature_config_v2.py` Docstrings
Fix all stale docstrings in `feature_config_v2.py` and `master_extractor_v2.py`:
- `get_feature_names()` docstring: change `"Get all 286 historical-mode feature names"` → `"Get all 311 feature names"`
- `MasterFeatureExtractor` class docstring: change `"Produces 286 flat features"` → `"Produces 311 flat features"`
- `__init___v2.py`: change `"327 features (270 + 57 Phase 1)"` → `"366 features (311 + 55 Phase 1)"`
### Step 5 — Update `feature_subsets.py`
In `config/feature_subsets.py`:
- Set `TOTAL_FEAT_DIM = 311`
- Add Agent TQ subset: dims 150–199 + 250–269 = 70 features
- Add Agent H subset: dims 0–164 = 165 features per timestep
### Step 6 — Update Stage 3 Meta-Model Namespace
In `scripts/stage3/infer_stage3_meta.py`:
- Replace all `hybrid51_utils` and `hybrid51_models` imports with `hybrid55_*` equivalents
- Update `ALL_AGENTS = ['A', 'B', 'C', 'K', 'TQ', 'H', '2D']` (note: `TQ` not `T`+`Q` separately)
### Step 7 — `HISTORICAL_MODE` Unification
Set `HISTORICAL_MODE = False` in `feature_config_v2.py` for both training (BigQuery) and deployment. The Phase 1 features should either be included in both environments or excluded in both. If historical BigQuery data does not have TQ columns needed by Phase 1, use Agent TQ's `historical_mode=True` flag to drop to 70 dims — but do not use the global `HISTORICAL_MODE` flag for this.

***
## Part 5 — Files Created (GitHub Location)
All agent skeleton files are committed to:

`!Hybrid55_New training/hybrid55_preprocessing/agents/`

| Agent | Files | Status |
|---|---|---|
| agent_a | `feature_config.py`, `extractor.py`, `validator.py` | ✅ Complete |
| agent_b | `feature_config.py`, `extractor.py`, `validator.py` | ⚠️ Stubs need replacement (Step 2) |
| agent_c | `feature_config.py`, `extractor.py`, `validator.py` | ✅ Skeleton complete |
| agent_k | `feature_config.py`, `extractor.py`, `validator.py` | ✅ Skeleton complete |
| agent_tq | `feature_config.py`, `extractor.py`, `validator.py` | ✅ Skeleton complete |
| agent_h | `feature_config.py`, `extractor.py`, `validator.py` | ✅ Skeleton complete |
| agent_2d | `feature_config.py`, `extractor.py`, `validator.py` | ✅ Skeleton complete |
| extractors/ | base files | ⚠️ Need move from root (Step 1) |

Note: Files were accidentally placed at the double-nested path `!Hybrid55_New training/!Hybrid55_New training/hybrid55_preprocessing/agents/` during creation. These need to be moved to the correct path `!Hybrid55_New training/hybrid55_preprocessing/agents/` — the double-nested directory should be deleted after the move is confirmed.

***
## Appendix — Feature Block Map (311 dims, Historical Mode)
| Dims | Group | Size | Agent(s) using |
|---|---|---|---|
| 0–74 | `greek_by_strike` | 75 | A, B, K, H |
| 75–104 | `gamma_exposure` | 30 | B, C, H |
| 105–124 | `vanna_charm` | 20 | B, C, H |
| 125–149 | `iv_surface` | 25 | B, C, H |
| 150–179 | `flow_volume` | 30 | B, TQ |
| 180–199 | `microstructure` | 20 | B, TQ |
| 200–219 | `walls_positioning` | 20 | B |
| 220–234 | `cross_strike` | 15 | B |
| 235–249 | `time_decay` | 15 | B |
| 250–269 | `sentiment_regime` | 20 | B, TQ |
| 270–285 | `csv_derived` | 16 | B, C (dims 273–275 only) |
| 286–310 | `ohlc_dynamics` | 25 | B |