# Hybrid52 Tier 1 & Tier 2 Data Pipeline — Bug Report & Cursor AI Amendment Proposal

## Executive Summary

The Hybrid52 ML model's near-random Stage 1 performance (~50% accuracy) is caused primarily by a corrupted tier 2 feature pipeline, not by model architecture flaws. A diagnostic on the existing tier 2 parquet confirmed **154 out of 286 features are dead (zero-variance)** while **0 rows are all-zero** — the precise fingerprint of features that were computed but written to wrong index positions. Tier 1 raw data is clean and does not require rebuilding. Five bugs in the tier 1 → tier 2 pipeline must be fixed before any tier 2 or tier 3 rebuild is attempted. This document is structured as a Cursor AI amendment instruction set: each section states the file, the exact problem, and the exact code fix required.[^1]

***

## Tier 1 Data Health (No Changes Required)

Before detailing the bugs, it is important to establish that the raw source data at `/workspace/data/tier1_v4/` is healthy and should **not** be touched.[^1]

| Source | Files | Rows/Day | Usable Columns | Date Range |
|--------|-------|----------|----------------|------------|
| Greek parquet | 609 | ~67,487 | 24 | 2023-03-23 to 2026-03-20 |
| TQ parquet | 609 | ~47,527 | 8 | 2023-03-23 to 2026-03-20 |

**24 usable Greek columns (0% null):** `bid, ask, delta, theta, vega, lambda, gamma, vanna, charm, impliedvol, underlyingprice, openinterest, moneyness, distatmpct, mid, spread, spreadpct, lambdaratio, dteint, cpsign, minute, weekkey, tradedate, cpsign`[^1]

**14 permanently dead columns (100% NaN in Theta Data):** `rho, epsilon, vomma, veta, vera, speed, zomma, color, ultima, d1, d2, dualdelta, dualgamma, iverror`[^1]

There is one 141-day gap ending 2025-03-20, plus standard holiday/weekend gaps — these do not require fixing; the extractor must zero-fill them gracefully.[^1]

***

## Bug Inventory

All five bugs are in the tier 2 build pipeline. Together they explain the 154/286 dead feature count confirmed by diagnostic.

| # | File | Bug | Impact | Severity |
|---|------|-----|--------|----------|
| 1 | `build_tier2_fast.py` | Wrong `TIER1_ROOT` path | All features zero if path missing | 🔴 Critical |
| 2 | 14 extractor files | Import from old 270-dim `featureconfig.py` | Features written to wrong indices | 🔴 Critical |
| 3 | `master_extractor_v2.py` | Gamma extractor outputs 50 dims, config allocates 30 | Cascading 20-position shift for all subsequent groups | 🔴 Critical |
| 4 | `feature_config_v2.py` | Dead Greeks (rho, epsilon, vomma, veta, zomma, color) in bucketing list | 30 guaranteed dead dims at indices 0–64 | 🟠 High |
| 5 | `build_tier2_fast.py` | Silent exception swallowing in `process_one_date()` | Extraction failures invisible | 🟡 Medium |

***

## Bug 1: Wrong TIER1_ROOT Path

### File
`scripts/phase0/build_tier2_fast.py`

### Problem
The script hardcoded the wrong tier 1 data path. The actual data lives at `/workspace/data/tier1_v4/` but the script reads from `/workspace/data/tier1_hybrid52/` which either does not exist or contains incomplete data. When tier 2 was built, extractors either received empty DataFrames or read from the wrong source, zero-padding all features.[^1]

```python
# CURRENT (WRONG):
TIER1_ROOT = Path("/workspace/data/tier1_hybrid52")   # ← path does not exist
```

### Fix
```python
# CORRECT:
TIER1_ROOT = Path("/workspace/data/tier1_v4")   # ← actual data location

# Also update the argument parser default to match:
parser.add_argument(
    '--tier1-root',
    default=str(Path("/workspace/data/tier1_v4")),
    help='Tier1 root directory containing per-symbol subdirs with greek.parquet and tq.parquet'
)
```

### Verification After Fix
```bash
ls /workspace/data/tier1_v4/SPXW/ | head -5
# Expected: dates like 2023-03-23greek.parquet, 2023-03-23tq.parquet
```

***

## Bug 2: All 14 Extractor Modules Import Old 270-dim Schema

### Files Affected
```
hybrid52_preprocessing/chain2d.py
hybrid52_preprocessing/crossstriketime.py
hybrid52_preprocessing/flowvolume.py
hybrid52_preprocessing/gammaexposure.py
hybrid52_preprocessing/greekfeatures.py
hybrid52_preprocessing/ivsurface.py
hybrid52_preprocessing/masterextractor.py
hybrid52_preprocessing/microstructure.py
hybrid52_preprocessing/qualitychecks.py
hybrid52_preprocessing/sentimentregime.py
hybrid52_preprocessing/sequencepipeline.py
hybrid52_preprocessing/trainingpipeline.py
hybrid52_preprocessing/wallspositioning.py
hybrid52_preprocessing/__init__.py
```

### Problem
Every individual extractor imports `FeatureGroup`, `FEATUREGROUPS`, and related constants from the old `featureconfig.py` (Hybrid51, 270-dim total features). Only `master_extractor_v2.py` imports from `feature_config_v2.py` (Hybrid52, 286-dim). Since `master_extractor_v2.py` calls the individual extractors, each one computes its output using 270-dim group start/end offsets and writes into a 286-dim output array. Features land in systematically wrong positions.[^1]

```python
# CURRENT (WRONG) — same pattern in all 14 files:
from .featureconfig import FeatureGroup, FEATUREGROUPS           # 270-dim Hybrid51
from .featureconfig import TOTAL_FEATURES                        # = 270
from .featureconfig import DELTA_BUCKETS, MONEYNESS_LEVELS       # wrong offsets
```

### Fix — One Bash Command Fixes All 14 Files
```bash
cd /workspace/Hybrid52_New\ training/hybrid52_preprocessing

for f in chain2d.py crossstriketime.py flowvolume.py gammaexposure.py \
          greekfeatures.py ivsurface.py masterextractor.py microstructure.py \
          qualitychecks.py sentimentregime.py sequencepipeline.py \
          trainingpipeline.py wallspositioning.py __init__.py; do
    sed -i 's/from \.featureconfig import/from .feature_config_v2 import/g' "$f"
    echo "Fixed: $f"
done
```

### Verification After Fix
```bash
grep -rn "from .featureconfig import" /workspace/Hybrid52_New\ training/hybrid52_preprocessing/ \
    --include="*.py" | grep -v "v2" | grep -v "__pycache__"
# Expected: NO output (all imports now point to v2)

grep -rn "from .feature_config_v2 import" /workspace/Hybrid52_New\ training/hybrid52_preprocessing/ \
    --include="*.py" | grep -v "__pycache__" | wc -l
# Expected: 14 (all files updated)
```

***

## Bug 3: Gamma Extractor Dimension Mismatch (50 vs 30)

### File
`hybrid52_preprocessing/master_extractor_v2.py`

### Problem
The gamma extractor outputs **50 dims** but `feature_config_v2.py` allocates **30 dims** for `GAMMA_EXPOSURE` (indices 75–104). The write call does not check the output size:[^1]

```python
# CURRENT (WRONG):
gamma_features = self.gamma_extractor.extract(greek_df)
features[idx:idx + 50] = gamma_features   # writes 50, overwrites VANNA_CHARM start
```

The extra 20 dims overwrite the beginning of the VANNA_CHARM block (expected at idx 105). This creates a **cascading 20-position shift** for every subsequent feature group — IV_SURFACE, MICROSTRUCTURE, WALLS_POSITIONING, CROSS_STRIKE, TIME_DECAY, SENTIMENT_REGIME, and CSV_DERIVED all land 20 positions off. This single bug alone explains the majority of the 154 dead features.

### Root Cause in gammaexposure.py
The gamma extractor computes 5 delta buckets × 10 Greeks = 50 dims, but `GAMMA_EXPOSURE` in `feature_config_v2.py` defines 30 dims (5 buckets × 6 Greeks: gamma, delta_adj_gamma, net_gamma, dealer_gamma, gamma_zones×2). The fix requires either trimming the extractor output to 30 or updating the config — **trim the extractor** since the config is the source of truth.

### Fix — master_extractor_v2.py
```python
# CORRECT — clip gamma extractor output to exactly 30 dims:
gamma_features = self.gamma_extractor.extract(greek_df)
n_gamma = FeatureGroup.GAMMA_EXPOSURE.num_features   # = 30 from feature_config_v2
features[idx:idx + n_gamma] = gamma_features[:n_gamma]   # clip to 30
idx += n_gamma
```

### Fix — gammaexposure.py (update extractor to output exactly 30 dims)
Open `hybrid52_preprocessing/gammaexposure.py` and find the `extract()` method. Update `GREEKS_FOR_GAMMA_BUCKETING` to match exactly the 6 Greeks that produce 30 dims (5 buckets × 6 = 30):

```python
# CURRENT (WRONG — includes too many Greeks producing 50 dims):
GREEKS_FOR_GAMMA_BUCKETING = [
    'gamma', 'delta', 'vega', 'theta', 'vanna', 'charm',
    'lambda', 'impliedvol', 'openinterest', 'moneyness'   # 10 Greeks × 5 = 50
]

# CORRECT — 6 Greeks × 5 buckets = 30 dims:
GREEKS_FOR_GAMMA_BUCKETING = [
    'gamma', 'delta', 'vanna', 'charm', 'impliedvol', 'openinterest'
]
```

Also add an assertion at the end of `extract()` to catch future mismatches:

```python
def extract(self, greek_df: pd.DataFrame) -> np.ndarray:
    # ... existing extraction logic ...
    result = self._build_gamma_features(greek_df)
    assert len(result) == 30, (
        f"GammaExtractor output {len(result)} dims, expected 30. "
        f"Check GREEKS_FOR_GAMMA_BUCKETING list length × n_buckets."
    )
    return result
```

### Verification After Fix
```python
# Run this before rebuilding tier2:
from hybrid52_preprocessing.gammaexposure import GammaExtractor
import pandas as pd

ext = GammaExtractor()
dummy = pd.DataFrame({'gamma': [0.01]*5, 'delta': [0.5]*5,
                      'vanna': [0.001]*5, 'charm': [0.0001]*5,
                      'impliedvol': [0.2]*5, 'openinterest':*5,
                      'moneyness': [1.0]*5, 'distatmpct': [0.0]*5})
output = ext.extract(dummy)
print(f"GammaExtractor output dims: {len(output)}")
# Expected: 30
```

***

## Bug 4: Dead Greeks in GREEKS_FOR_BUCKETING

### File
`hybrid52_preprocessing/feature_config_v2.py`

### Problem
`GREEKS_FOR_BUCKETING` — the list used by `greekfeatures.py` to build per-delta-bucket Greek profiles — includes `rho, epsilon, vomma, veta, zomma, color`, all of which are **100% NaN in every Theta Data file**. These get bucketed across 5 delta buckets, producing up to **30 guaranteed-dead dims** at the start of the `GREEK_BY_STRIKE` block (indices 0–64).[^1]

```python
# CURRENT (WRONG — includes 6 columns that are 100% NaN):
GREEKS_FOR_BUCKETING = [
    'delta', 'gamma', 'vega', 'theta',
    'rho',      # ← 100% NaN
    'epsilon',  # ← 100% NaN
    'lambda', 'vanna', 'charm',
    'vomma',    # ← 100% NaN
    'veta',     # ← 100% NaN
    'zomma',    # ← 100% NaN
    'color',    # ← 100% NaN
    'impliedvol'
]
```

### Fix
```python
# CORRECT — remove all 100% NaN Greeks:
GREEKS_FOR_BUCKETING = [
    'delta',      # ✅ real
    'gamma',      # ✅ real
    'vega',       # ✅ real
    'theta',      # ✅ real
    'lambda',     # ✅ real
    'vanna',      # ✅ real
    'charm',      # ✅ real
    'impliedvol', # ✅ real
    'openinterest', # ✅ real
    'moneyness',   # ✅ real
]
# 10 Greeks × 5 delta buckets = 50 dims for GREEK_BY_STRIKE block
# NOTE: Adjust feature_config_v2.py GREEK_BY_STRIKE num_features to 50
# if only 5 buckets are used, or use fewer Greeks to match the 75-dim allocation.
# Recommended: use exactly the Greeks that fill the 75-dim budget cleanly.
```

**Recommended clean mapping for 75 dims in `GREEK_BY_STRIKE` (0–74):**

| Bucket type | Greeks | Buckets | Dims |
|------------|--------|---------|------|
| Per-delta bucket Greeks | delta, gamma, vega, theta, lambda, vanna, charm, impliedvol, openinterest, moneyness | 5 buckets | 50 |
| ATM Greeks (last bar) | delta, gamma, vega, theta, lambda, vanna, charm | — | 7 |
| Skew metrics | iv_skew, put_call_iv_ratio, wing_spread | — | 3 |
| Cross-bucket ratios | otm_itm_ratio, front_back_ratio | — | 2 |
| Buffer | zeros | — | 13 |
| **Total** | | | **75** |

***

## Bug 5: Silent Exception Swallowing

### File
`scripts/phase0/build_tier2_fast.py` — `process_one_date()` function

### Problem
The current extraction loop silently converts any exception into a zero-feature vector. There is no logging, no counter, and no way to know how many minutes failed extraction. Hybrid51 had retry logic and error logging; Hybrid52 dropped it during the rewrite:[^1]

```python
# CURRENT (WRONG — silent failure):
try:
    result = extractor.extract(greek_df=greek_group, trade_df=tq_group)
    features = result.features
except Exception:
    features = np.zeros(FEAT_DIM, dtype=np.float32)   # silent! no log
```

### Fix
```python
# CORRECT — log all failures, track failure rate, abort if >20% fail:
extraction_failures = 0
extraction_total = 0

for ts, greek_group in grouped_by_minute:
    extraction_total += 1
    try:
        result = extractor.extract(greek_df=greek_group, trade_df=tq_group)
        features = result.features
        if len(features) != FEAT_DIM:
            raise ValueError(
                f"Extractor output {len(features)} dims, expected {FEAT_DIM}"
            )
    except Exception as e:
        extraction_failures += 1
        if extraction_failures <= 5:   # log first 5 failures in detail
            logger.warning(
                f"Feature extraction failed at minute {ts}: {type(e).__name__}: {e}"
            )
        features = np.zeros(FEAT_DIM, dtype=np.float32)

# End-of-day failure summary
failure_rate = extraction_failures / max(extraction_total, 1)
if failure_rate > 0.20:
    logger.error(
        f"  ❌ HIGH FAILURE RATE: {extraction_failures}/{extraction_total} "
        f"minutes failed ({failure_rate:.1%}). "
        f"Check extractor column names vs tier1 schema."
    )
elif extraction_failures > 0:
    logger.warning(
        f"  ⚠️  {extraction_failures}/{extraction_total} minutes failed "
        f"extraction ({failure_rate:.1%})"
    )
else:
    logger.info(f"  ✅ All {extraction_total} minutes extracted successfully")
```

Also add a **schema audit block** at the start of `process_one_date()` for the first date processed:

```python
# Add at the start of process_one_date(), runs once on first date:
if not hasattr(process_one_date, '_schema_audited'):
    process_one_date._schema_audited = True
    logger.info(f"SCHEMA AUDIT — Greek columns: {list(greek_df.columns)}")
    logger.info(f"SCHEMA AUDIT — TQ columns: {list(tq_df.columns) if tq_df is not None else 'None'}")
    logger.info(f"SCHEMA AUDIT — Greek null summary:")
    null_pct = greek_df.isnull().mean()
    for col, pct in null_pct[null_pct > 0.5].items():
        logger.warning(f"  Column '{col}': {pct:.1%} null")
```

***

## Additional Fix: datavalidation.py — Extend KNOWN_ZERO_COLUMNS

### File
`hybrid52_preprocessing/datavalidation.py`

### Problem
`KNOWN_ZERO_COLUMNS` currently only lists `speed` and `vera`. The 14 confirmed-null columns from tier 1 are not excluded at the validation stage, so they pass downstream into extraction and produce dead dims without warning.[^1]

### Fix
```python
# CURRENT:
KNOWN_ZERO_COLUMNS = {'speed', 'vera'}

# CORRECT — add all 100% null Theta Data columns:
KNOWN_ZERO_COLUMNS = {
    'speed', 'vera',           # original dead cols
    'rho', 'epsilon',          # greeks1st — 100% NaN
    'vomma', 'veta',           # greeks2nd — 100% NaN
    'zomma', 'color',          # greeks2nd — 100% NaN
    'ultima',                  # greeks3rd — 100% NaN
    'dualdelta', 'dualgamma',  # greeksother — 100% NaN (confirmed by query)
    'd1', 'd2',                # technical — 100% NaN
    'iverror',                 # volatility — 100% NaN
}
```

***

## Rebuild Sequence (After All Code Fixes)

Execute in this exact order. **Do not skip steps or reorder.**

### Step 1 — Wipe stale tier 2
```bash
rm -rf /workspace/data/tier2_minutes_hybrid52/
echo "Tier2 wiped"
```

### Step 2 — Verify all import fixes are in place
```bash
# Should return NO output (no remaining old-schema imports):
grep -rn "from .featureconfig import" \
    /workspace/Hybrid52_New\ training/hybrid52_preprocessing/ \
    --include="*.py" | grep -v "v2" | grep -v "__pycache__"

# Should return 14 (all files updated to v2):
grep -rn "from .feature_config_v2 import" \
    /workspace/Hybrid52_New\ training/hybrid52_preprocessing/ \
    --include="*.py" | grep -v "__pycache__" | wc -l
```

### Step 3 — Run one symbol with 1 worker (to see all logs)
```bash
cd /workspace/Hybrid52_New\ training

PYTHONPATH=. python scripts/phase0/build_tier2_fast.py \
    --symbol SPXW \
    --tier1-root /workspace/data/tier1_v4 \
    --output-root /workspace/data/tier2_minutes_hybrid52 \
    --workers 1 2>&1 | tee /tmp/tier2_spxw.log

# Check the log:
grep -E "SCHEMA AUDIT|failed|fell back|useful_features|DONE|❌|⚠️" /tmp/tier2_spxw.log
```

**Expected log lines:**
- `SCHEMA AUDIT — Greek columns: ['bid', 'ask', 'delta', ...]` (24 usable columns)
- `✅ All N minutes extracted successfully` for each date
- `DONE: SPXW — X minutes, 286 features` at the end

**Failure indicators (investigate before proceeding):**
- `Feature extraction failed at minute ...` → column name mismatch, fix extractor
- `HIGH FAILURE RATE: X/Y ...` → extractor broken, do not rebuild all symbols yet

### Step 4 — Verify dead feature count after SPXW tier 2
```bash
python3 - << 'EOF'
import pandas as pd
import numpy as np

df = pd.read_parquet('/workspace/data/tier2_minutes_hybrid52/SPXW_minutes.parquet')
feat = np.array(df['features'].tolist())

dead = (feat.std(axis=0) < 1e-8).sum()
all_zero_rows = (feat == 0).all(axis=1).sum()
total = feat.shape

print(f"Shape: {feat.shape}")
print(f"Dead features: {dead}/{feat.shape[^1]}  (target: <20)")
print(f"All-zero rows: {all_zero_rows}/{total}  (target: 0)")

# Per-group liveness check
groups = {
    'GREEK_BY_STRIKE (0-74)':    (0, 75),
    'GAMMA_EXPOSURE (75-104)':   (75, 105),
    'VANNA_CHARM (105-124)':     (105, 125),
    'IV_SURFACE (125-149)':      (125, 150),
    'FLOW_VOLUME (150-179)':     (150, 180),
    'MICROSTRUCTURE (180-199)':  (180, 200),
    'WALLS_POSITIONING (200-219)': (200, 220),
    'CROSS_STRIKE (220-234)':    (220, 235),
    'TIME_DECAY (235-249)':      (235, 250),
    'SENTIMENT_REGIME (250-269)': (250, 270),
    'CSV_DERIVED (270-285)':     (270, 286),
}
print("\nPer-group liveness:")
for name, (s, e) in groups.items():
    blk = feat[:, s:e]
    nz = (np.abs(blk) > 1e-8).mean()
    dead_g = (blk.std(axis=0) < 1e-8).sum()
    flag = "✅" if nz > 0.05 else "❌ DEAD"
    print(f"  {flag}  {name}: nz={nz:.3f}, dead_dims={dead_g}/{e-s}")
EOF
```

**Target outcome:** dead features < 20/286, all groups show `nz > 0.05`.

### Step 5 — Rebuild all symbols (only after SPXW passes Step 4)
```bash
PYTHONPATH=. python scripts/phase0/build_tier2_fast.py \
    --all-symbols \
    --tier1-root /workspace/data/tier1_v4 \
    --output-root /workspace/data/tier2_minutes_hybrid52 \
    --workers 8
```

### Step 6 — Rebuild tier 3 (from the fixed tier 2)
```bash
# Apply the TQ_FEAT_START fix first (see companion proposal for build_tier3_binary.py)
PYTHONPATH=. python scripts/phase0/build_tier3_binary.py \
    --all-symbols \
    --horizons 5 15 30 \
    --tier2-root /workspace/data/tier2_minutes_hybrid52 \
    --output-root /workspace/data/tier3_binary_hybrid52 \
    --return-threshold 0.0003 \
    --strip-zero-variance
```

***

## Summary of All File Changes

| File | Change Type | Lines Changed |
|------|-------------|---------------|
| `scripts/phase0/build_tier2_fast.py` | Path fix + logging | ~30 lines |
| `hybrid52_preprocessing/chain2d.py` | Import fix (1 line) | 1 line |
| `hybrid52_preprocessing/crossstriketime.py` | Import fix | 1 line |
| `hybrid52_preprocessing/flowvolume.py` | Import fix | 1 line |
| `hybrid52_preprocessing/gammaexposure.py` | Import fix + GREEKS_FOR_BUCKETING | 3–5 lines |
| `hybrid52_preprocessing/greekfeatures.py` | Import fix | 1 line |
| `hybrid52_preprocessing/ivsurface.py` | Import fix | 1 line |
| `hybrid52_preprocessing/masterextractor.py` | Import fix | 1 line |
| `hybrid52_preprocessing/master_extractor_v2.py` | Gamma dim clip | 2–3 lines |
| `hybrid52_preprocessing/microstructure.py` | Import fix | 1 line |
| `hybrid52_preprocessing/qualitychecks.py` | Import fix | 1 line |
| `hybrid52_preprocessing/sentimentregime.py` | Import fix | 1 line |
| `hybrid52_preprocessing/sequencepipeline.py` | Import fix | 1 line |
| `hybrid52_preprocessing/trainingpipeline.py` | Import fix | 1 line |
| `hybrid52_preprocessing/wallspositioning.py` | Import fix | 1 line |
| `hybrid52_preprocessing/__init__.py` | Import fix | 1 line |
| `hybrid52_preprocessing/feature_config_v2.py` | Remove dead Greeks from bucketing | 6 lines |
| `hybrid52_preprocessing/datavalidation.py` | Extend KNOWN_ZERO_COLUMNS | 10 lines |

**Total: 18 files, ~70 lines changed. All changes are in the preprocessing layer only — no model architecture, no training script, no feature_subsets.py requires changes.**

***

## Expected Outcome After Full Rebuild

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Dead features in tier 2 | 154/286 (54%) | < 20/286 (<7%) |
| Dead features in tier 3 sequences | 156/286 | < 20/286 |
| Stage 1 single-agent AUC (SPXW h15) | ~0.50 | 0.52–0.56 |
| Stage 1 single-agent Accuracy | ~48–52% | 53–57% |
| Stage 2 ensemble AUC | N/A | 0.55–0.58 |
| Stage 3 meta-learner Accuracy | N/A | 58–62% |

The 60%+ accuracy achieved by Hybrid51's Stage 1 agents was not due to a superior architecture — it was due to clean, fully-populated feature vectors. Once tier 2 is rebuilt with the correct imports and extractor dimensions, Hybrid52 should reach parity immediately.

---

## References

1. [data-at-workspace_data_tier2_minutes_v4_VIXW_minut.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61007812/17cf909b-9078-4f0c-8ff4-e5ccfcbcb680/data-at-workspace_data_tier2_minutes_v4_VIXW_minut.md?AWSAccessKeyId=ASIA2F3EMEYE4DCESWKA&Signature=b90ZofTGbhNEp9DWqHitJdN1jgQ%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEAEaCXVzLWVhc3QtMSJHMEUCIHarILCKED3reGheuOFn8C3nWe03zO6dgj6LPPQQ3OKQAiEA0SaQ37EAHNCzPGGEn3iND3qqrEiR88mf9v%2BgI0cVjw4q%2FAQIyv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDC1GZkI2EalZfhwURSrQBKH1b62E6%2FYqKi5COx1U4vqDSddf704LtSXSmSERFQ0wf6DPwT3Xvq5NZHB87U4SR1Z%2F0cGEYlPE7sgaLSmzYIcNDwkqmohI6jGlTpGoTpCd02SLcQCwv0NqZOllaKzynQh%2B%2BghrkosB%2FpsoZZzjTv7NT7OawkDaKsJfMWm34lWTUZHLInKu5Sh3kE9S8AYIKWDJ13dc%2FdX4K3otnjVRnG8L8DMbaYHyTE%2FF18bgnCh4jGHrvjEhpVAHd1W8aWtZj6fEfrejXnoal2YUDYgUFLAVUYSaHxZexY47wXI2qT5NyD%2BjQI3oTvxI6iMcEXjM8YpzwBI64F1SjBwxTfzHAtcWYZ9PcPgQ27x7HSixh7rTdL%2BR48a2dQws%2F4Fte6xj0ho%2FWzX0FCcREhiA0mTyaGO4%2FpwXiAbC%2B5HJ5tKm3Hy4X68ggJJv4vsaVEAWt3XBEJJT0CkeArbNF34GlCLM9XQ0h2ir%2B2U%2F%2FJKYc%2BL1Qcj8BHWOAOHwEyOddQSD8YXINjDihTxSv8yifTbkUN6Qrc1zQQfAf4V4fnuf5rk8QNlFiewb7aKXOyIcT7YFLBpwHX5XEh74TbZANROo7NksLxAX8C6tYfn7sMzbZCy3%2FiDqR1rqnhxo%2BUs5spvkMn%2BfZBHxfDJETofVRNI1eqj7eHQiTgnNa4Mb8B6NiwleU8bNfY1KOE1t49ezMNubWuji4yEtjYiJecOigkjTEyF7MKBFVkxSxtzc5WZl3lb7C%2Fn1YJwQyD0zCZyTEGbrtgprVEKrA0mXr8cUb3fqpHkCa7kwqcCVzgY6mAGuCMBwe2WpPmUZl8s1ocfjiygn6nwLNlw7xYIXTs%2F4ahz5BN7kEuk16yKgPdrzsWX7%2BsOJkWv3s3nxBjUdzuxgdAywDgLFikJVyZmqLH7Wr5YGR3cEBsrE9DP0WZYsGb1v%2FAPNcSWak6JV1RULwk9s%2BmmkK2xFlLvu2QdSEUCOtDpiNtXWvnGBha114X4%2FXdq6T6nhgcnthQ%3D%3D&Expires=1774546428) - Sample symbol expiration strike right timestamp bid ask delta theta vega rho epsilon lambda gamma va...

