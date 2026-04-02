# Hybrid52 Dead Feature Dims — Root Cause & Fix Instructions for Cursor AI
## Executive Summary
The Tier 2 parquet builder writes a **278-dim feature vector** per minute bar, but **145 out of 278 dims are permanently zero** (`Dead (0%): 145`). This causes all 6 active agents (A, B, C, K, T, Q) to receive large fractions of dead inputs, which degrades model training. The root cause is a **mismatch between what the data pipeline produces and what the agents were redesigned to expect**. The fix requires changes to 5 files with no agent architecture changes required.
### Diagnostic Results (from SPXW_minutes.parquet, 230,531 rows)
```
Feature matrix shape: (230531, 278)
Good dims (>=50% populated): 126
Sparse dims (<50% but >0%):    7
Dead dims (all zero):          145   ← 52% dead
```
### Agent Impact
| Agent | feat_dim | Dead dims | Dead % |
|-------|----------|-----------|--------|
| A | 56 | 27/56 | 48% |
| B | 36 | 24/36 | 67% |
| C | 37 | 21/37 | 57% |
| K | 56 | 27/56 | 48% |
| T | 25 | 15/25 | 60% |
| Q | 20 | 7/20 | 35% |
| 2D | 0 | 0 | ✅ OK |

***
## Root Cause
### Why the Dead Dims Exist
The pipeline has **two layers** that create zeros:

**Layer 1 — `build_duckdb_from_historical_csv.py`** deliberately inserts `NULL` for fields the historical CSV doesn't have:

```python
# /workspace/ Hybrid52_New training/scripts/phase0/build_duckdb_from_historical_csv.py
# Inside the INSERT INTO optionstradequote block:
NULL::VARCHAR AS condition,   # OPRA codes don't exist in historical CSV
NULL::DOUBLE AS bid_size,     # order book depth doesn't exist
NULL::DOUBLE AS ask_size,     # order book depth doesn't exist
COALESCE(volume, count, 0.0) AS size,  # aggregate only, not per-trade
COALESCE(vwap, close, mid) AS price    # proxy, not actual trade price
```

**Layer 2 — `build_tier2fast.py`** runs `MasterFeatureExtractor` with `include_phase1=True`, so all 14 extraction modules always run — including the 4 modules that need the above NULL data:

```
smart_money.py     → dims 270-284  → ALL ZERO (needs OPRA sweep codes)
volume_anomaly.py  → dims 285-296  → ALL ZERO (needs per-trade volume stream)
trade_conditions.py→ dims 297-306  → ALL ZERO (needs OPRA condition codes)
quote_pressure.py  → dims 307-324  → MOSTLY ZERO (needs bid_size/ask_size)
```

Additionally, dims in the **microstructure group (190–199)** are also near-zero because they require per-trade aggression data unavailable in the historical CSV.

The **agents were already amended** to exclude the worst zero dims by updating `AGENT_FEATURE_SUBSETS` ranges. However, the **tier2 builder was never updated** to stop generating those zero slots — so the parquet still has 278 dims, and the agent range mappings still land on dead indices despite the redesign.

***
## What Needs to Be Fixed
### Files to Change
| File | Path | What to Change |
|------|------|----------------|
| `build_duckdb_from_historical_csv.py` | `/workspace/ Hybrid52_New training/scripts/phase0/` | Pass through `bid_size`, `ask_size`, `spread`, `spread_pct`, `moneyness`, `dist_atm_pct` instead of NULLing them |
| `build_tier2fast.py` | `/workspace/ Hybrid52_New training/scripts/phase0/` | Set `include_phase1=False` to skip the 4 dead extractors |
| `feature_config_v2.py` | `/workspace/ Hybrid52_New training/hybrid52/preprocessing/` | Change `TOTAL_FEATURES = 325` → `TOTAL_FEATURES = 270` (or 278 if keeping CSV-derived dims) |
| `build_tier3binary.py` | `/workspace/ Hybrid52_New training/scripts/phase0/` | Update `FEAT_DIM`, disable `TQ_FEAT_START/END` dead-dim checks |
| `config/feature_subsets.py` | `/workspace/ Hybrid52_New training/` | Update `TOTAL_FEAT_DIM` and Agent Q's dead ranges |

***
## Fix 1 — `build_duckdb_from_historical_csv.py`
**File:** `/workspace/ Hybrid52_New training/scripts/phase0/build_duckdb_from_historical_csv.py`

Find the `INSERT INTO optionstrade_quote` block. Replace the NULL overrides with real column pass-throughs:

```python
# BEFORE (current broken code):
NULL::VARCHAR AS condition,
NULL::DOUBLE AS bid_size,
NULL::DOUBLE AS ask_size,
COALESCE(volume, count, 0.0) AS size,

# AFTER (fixed):
NULL::VARCHAR AS condition,                          # still no OPRA codes, that's fine
COALESCE(bid_size, 0.0) AS bid_size,                # ← REAL DATA: pass through from CSV
COALESCE(ask_size, 0.0) AS ask_size,                # ← REAL DATA: pass through from CSV
COALESCE(volume, count, 0.0) AS size,
```

Also add these enrichment columns to the `options_greek` table INSERT (they exist in the historical CSV but are thrown away):

```python
# Add to the Greek INSERT SELECT — these columns exist in the CSV:
COALESCE(moneyness, NULL) AS moneyness,
COALESCE(dist_atm_pct, NULL) AS dist_atm_pct,
COALESCE(spread_pct, NULL) AS spread_pct,
COALESCE(lambda_ratio, NULL) AS lambda_ratio,
```

**Why:** `quote_pressure.py` reads `bid_size`/`ask_size` to compute depth ratio and order book imbalance for dims 307–324. Without this fix, those dims stay zero even though the CSV has the data.

***
## Fix 2 — `build_tier2fast.py`
**File:** `/workspace/ Hybrid52_New training/scripts/phase0/build_tier2fast.py`

Find the `MasterFeatureExtractor` instantiation. Change `include_phase1`:

```python
# BEFORE:
extractor = MasterFeatureExtractor(
    include_chain_2d=False,
    include_phase1=True,    # ← This runs 4 extractors on NULL data → zeros
    ...
)

# AFTER:
extractor = MasterFeatureExtractor(
    include_chain_2d=False,
    include_phase1=False,   # ← Skip Smart Money, Volume Anomaly, Trade Conditions
    ...
)
```

Also update `FEAT_DIM` if it's hardcoded in this file:

```python
# BEFORE:
FEAT_DIM = 325

# AFTER:
FEAT_DIM = 278   # 270 base + 8 CSV-derived dims
```

**Why:** `include_phase1=True` forces dims 270–324 (55 dims) to always be computed, but 37 of them (270–306) are guaranteed zero with historical CSV data. Disabling them stops generating and storing noise.

***
## Fix 3 — `feature_config_v2.py`
**File:** `/workspace/ Hybrid52_New training/hybrid52/preprocessing/feature_config_v2.py`

```python
# BEFORE:
TOTAL_FEATURES = 325

# AFTER:
TOTAL_FEATURES = 278   # 270 real dims + 8 CSV-derived (lambda, dist_atm, spread_pct)

# ADD: Historical mode flag
HISTORICAL_MODE = True   # Set False when real OPRA trade-flow data becomes available
```

**Why:** Downstream scripts and agents read `TOTAL_FEATURES` to set input dimensions. Keeping it at 325 leaves ghost dims in the tensor layout.

***
## Fix 4 — `build_tier3binary.py`
**File:** `/workspace/ Hybrid52_New training/scripts/phase0/build_tier3binary.py`

Three patches needed:

**Patch A — Update FEAT_DIM:**
```python
# BEFORE:
FEAT_DIM = 325

# AFTER:
FEAT_DIM = 278
```

**Patch B — Disable dead TQ dim range check:**
```python
# BEFORE:
TQ_FEAT_START = 270
TQ_FEAT_END = 325

# AFTER:
TQ_FEAT_START = 0   # disabled — CSV-derived dims now occupy 270-277
TQ_FEAT_END = 0     # disabled
```

**Patch C — Update group_ranges dead-dim entries:**
```python
# BEFORE (last 4 entries in group_ranges):
'SMART_MONEY':      (270, 285),
'VOLUME_ANOMALY':   (285, 297),
'TRADE_CONDITIONS': (297, 307),
'QUOTE_PRESSURE':   (307, 325),

# AFTER:
'CSV_DERIVED':      (270, 278),   # 8 dims: lambda, dist_atm, spread_pct
# Remove SMART_MONEY, VOLUME_ANOMALY, TRADE_CONDITIONS, QUOTE_PRESSURE entirely
```

***
## Fix 5 — `config/feature_subsets.py`
**File:** `/workspace/ Hybrid52_New training/config/feature_subsets.py`

**Patch A — Update TOTAL_FEAT_DIM:**
```python
# BEFORE:
TOTAL_FEAT_DIM = 325   # or 278 if already patched

# AFTER:
TOTAL_FEAT_DIM = 278
```

**Patch B — Fix Agent Q's dead ranges:**

Agent Q currently references dims 307–313 and 317–319 (Quote Pressure group, all dead). Replace with real dims:

```python
# BEFORE (Agent Q ranges in AGENT_FEATURE_SUBSETS):
'Q': {
    'ranges': [
        (307, 313),   # ← DEAD: cvd/quote dynamics, all zero
        (180, 189),   # bid-ask spread — real
        (189, 192),   # quote intensity — near-zero
        (317, 319),   # ← DEAD: depth ratio, near-zero
    ],
    'feat_dim': 20
}

# AFTER:
'Q': {
    'ranges': [
        (125, 131),   # IV by moneyness — HIGH quality (replaces 307-313)
        (180, 189),   # bid-ask spread — REAL (now enhanced by Fix 1)
        (105, 107),   # vanna net exposure — HIGH quality (replaces 317-319)
        (189, 192),   # quote intensity — weak but non-zero
    ],
    'feat_dim': 20   # unchanged: 6 + 9 + 2 + 3 = 20
}
```

**Note:** Agents A, B, C, K, T, and 2D do **not** need range changes in `feature_subsets.py`. Their dims are all within 0–269 and will be unaffected by removing Phase 1.

***
## Fix 6 — Rebuild Tier 2 and Tier 3
After applying all 5 file fixes above, **delete the existing parquets and rebuild** from scratch:

```bash
cd "/workspace/ Hybrid52_New training"

# Delete stale tier2 data
rm /workspace/data/tier2_minutes_hybrid52/SPXW_minutes.parquet
rm /workspace/data/tier2_minutes_hybrid52/SPY_minutes.parquet
# Wait for QQQ to finish, then: rm QQQ_minutes.parquet

# Rebuild DuckDB with real bid_size/ask_size pass-through (Fix 1)
python3 scripts/phase0/build_duckdb_from_historical_csv.py --symbols SPXW SPY QQQ

# Rebuild Tier 2 (now include_phase1=False, FEAT_DIM=278)
python3 scripts/phase0/build_tier2fast.py --all-symbols --workers 4

# Rebuild Tier 3 training sequences
python3 scripts/phase0/build_tier3binary.py --all-symbols --horizons 5 15 30
```

***
## Validation Check
After rebuild, run this to confirm zero dims are eliminated:

```bash
cat > /tmp/validate_fix.py << 'EOF'
import numpy as np
import pandas as pd
from pathlib import Path

f = Path("/workspace/data/tier2_minutes_hybrid52/SPXW_minutes.parquet")
df = pd.read_parquet(f)
feat_arr = np.vstack(df['features'].values).astype(np.float64)
print(f"Feature matrix shape: {feat_arr.shape}")

dead = [d for d in range(feat_arr.shape[^1])
        if np.sum(np.isfinite(feat_arr[:,d]) & (feat_arr[:,d] != 0.0)) == 0]
print(f"Dead dims remaining: {len(dead)}")
print(f"Expected: 0 dead dims")
print("✅ PASS" if len(dead) == 0 else f"❌ FAIL — still {len(dead)} dead dims: {dead[:20]}")
EOF
cd "/workspace/ Hybrid52_New training" && python3 /tmp/validate_fix.py
```

***
## What Does NOT Change
- **All agent `.py` files** (agent_a.py, agent_b.py, agent_c.py, agent_k.py, agent_t.py, agent_2d.py, agent_vix.py) — no changes needed
- **`independent_agent.py`** — no changes needed
- **Agent 2D chain pipeline** — `CHAIN_INPUT_STRIKES=30`, `CHAIN_OUTPUT_STRIKES=20` are correct and untouched
- **VIX side track** — `build_tier3_vix.py` is independent and unaffected
- **Training logic** in `build_tier3binary.py` — only constants change, not logic

***
## Summary: What Was Wrong vs. What Gets Fixed
| Problem | Root Cause | Fix Applied |
|---------|-----------|-------------|
| 145/278 dims all zero | `include_phase1=True` runs 4 extractors on NULL data | Fix 2: Set `include_phase1=False` |
| `bid_size`/`ask_size` NULL | `build_duckdb` hardcodes `NULL::DOUBLE` | Fix 1: Pass through real CSV columns |
| Agent Q reads dead 307–319 | Old Phase 1 range still in `feature_subsets.py` | Fix 5: Remap to real dims 125–131, 105–107 |
| `TOTAL_FEATURES=325` stale | Never updated after agent redesign | Fix 3 + Fix 4: Set to 278 |
| Tier 2 parquets have stale data | Built before fixes applied | Fix 6: Delete and rebuild |

---

## References

1. [what-is-sequent-of-using-these-py-for-data-process.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61007812/e2b1e229-52a2-46d5-b012-6ea26f4aa591/what-is-sequent-of-using-these-py-for-data-process.md?AWSAccessKeyId=ASIA2F3EMEYE3KQRGGIJ&Signature=F3Pm4A0IXSTmY9KXvobS3Q%2FWnCY%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEOT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQC90FbHBNqMdPnekEp%2B9aryWaMW6pgDR8hRAG02Zo0AtQIhAKYlxXC1vSU4Okh6WbIoi6pXZrE79hESkqPGn7BP1HPFKvwECK3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgzbRh9UYv5tOJQsHU8q0ATZUOyPk7Tb9X7J2EqPOsXihVEiN6x8M9KGcQWZXzq3K977A4Oegd1b%2BjpxmzeZyA8Ifq5vxaIvzd4cKUW5IjYv9%2FgUkQFziIs%2BBjKrIDK5n6JOJ8tUJZ49LjWY%2Fo6EBZ0laDORss%2BBSpM4e%2FWx6eWObQBoDr5bLqOhI2SPvuDhAAklCIpNUIHmrelnT00hQn4us16mVXeUSXQSxbYJRND69kEpqqsNkNCr9gwd2KhQnjWsAljTCFrRx%2FvBKyk%2BBkAfxpC7eozhAHdaUfCKywRLpyV0qzcBwj5qIP5yi2iGv5SUbEK0GnJiCPsCA7PQXX%2FwnSM%2BcXY%2B6DZVAPSA4k1szeikt8paIjQ3uTtSbDX3sfoxWI77V4whqXZVeO6UtNlcIUcZQ4rFwYpqc3glCbnjxP7gCN80buDcnqR1EBbWiX7NUnJ%2F8kgLT3RF6O7WlEwOJvgppWx%2FyxHA%2FXyx%2FbnRB%2B%2F6%2FtZpexLRxzYOfw9dVO8sdYOuNnaH5NjSk5ClLk3Rq4slmjkADjhr3tQUXvNkEtDRz4LjLFFfYGrrId5Q%2Fv4W2tOUCzo1aXCL7JKcXsT7PDqhHtF1mp4tRo7CvMdmfJNbipXb2%2BRn%2FCp1kQE6nTG%2F1SABzHLOoJkEQPi6bt0Fj4yTXYolwPFAV9XtjxkPOLvN3iv%2ByH1ONnSaaEpOjFoMA5lib6ySae0LK0L866k8GhvFUzZOu06iBarxytt8jZD3P9KIIjPeEkNmhebx6c%2BBMfiFQ1zVBEiU0JSnGc7DV3OcjPbLGW6f4F%2BuCZloMMOVj84GOpcBCbEHrgi5SrmvXAbVjQFW6knTUyctBMtV9y5gmnzZUFBdoKVERsgtni0r5gNmLpQjElbD%2FztD5pOxLYSqB62APyxazVHl3Jm0r1DunPnCexZYwkvtKBcXZQu76MlODWPj6%2BuH1nwfo7hMzByP3arYQxb8mcLqiJWJ0L6rZgLBBio8LmCkOjXIhbPOmugjmOvvXkiemnXoyQ%3D%3D&Expires=1774442646)

