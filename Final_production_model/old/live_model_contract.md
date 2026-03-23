# Live Hybrid51 Online Feature Contract (from `/workspace/daily_data`)

## Purpose

Define how live batch CSVs are transformed into model-required tensors for the production Hybrid51 stack.

Required model inputs:
- Stage 1 per symbol: `X_seq` shape `(1, 20, 325)`
- Stage 1 Agent2D per symbol: `X_chain_2d` shape `(1, 5, 20, 20)`
- Stage 3: `X_vix` shape `(1, 10)`

Symbols for stage-1: `SPXW`, `SPY`, `QQQ`, `IWM`, `TLT`

Primary live sources:
- `/workspace/daily_data/theta_snapshot.csv`
- `/workspace/daily_data/theta_agg.csv` (or DTE-filtered variants already loaded by dashboard)

## Parity Mapping

### 1) Stage-1 flat sequence vector (325 dims)

| Group | Source | Status | Notes |
|---|---|---|---|
| Batch-level option flow/IV/microstructure | `theta_agg*.csv` (`spot`, `call_vol`, `put_vol`, `pc_ratio`, `net_gex`, `call_iv`, `put_iv`, `iv_skew`, `avg_spread_pct`, `bid_ask_imbalance`, `trade_aggression`, `avg_trade_size`, etc.) | Direct | Used as core live macro features |
| Strike-level greek/quote/trade summaries | `theta_snapshot.csv` (`delta`, `gamma`, `theta`, `vega`, `implied_vol`, `spread_pct`, `mid`, `oi`, `volume`, `count`, `moneyness`, `dist_atm_pct`, etc.) | Derivable | Aggregated into moments/quantiles to create stable per-batch features |
| Full training 325 schema exact ordering | Offline tier3 pipeline (not present in live fetcher output) | Missing exact parity | Live bridge uses deterministic proxy schema + fixed ordering to produce 325 dims |

Implementation policy:
- Build deterministic base feature vector from direct + derived live columns.
- Enforce fixed order and dtype (`float32`).
- Expand to 325 dims by stable deterministic tiling and bounded nonlinear transforms.
- Replace NaN/inf with 0 and clip extremes.

### 2) Agent2D chain tensor (`5 x 20 x 20`)

| Component | Source | Status | Notes |
|---|---|---|---|
| Greeks channels | `theta_snapshot.csv`: `delta`, `gamma`, `theta`, `vega`, `implied_vol` | Direct | Channel order fixed |
| Strike axis (20) | nearest-to-ATM strikes from snapshot | Derivable | Uses `dist_atm_pct`/`atm_strike`/`strike` |
| Time axis (20) | rolling per-batch history in memory | Derivable | One frame per batch; padded until warm |

### 3) VIX regime vector (10 dims)

| Feature | Source | Status |
|---|---|---|
| `vix_level` | agg row for symbol `VIXW` (`spot`) | Direct |
| `vix_pct_5m`, `vix_pct_15m`, `vix_pct_1h` | rolling `%` change of VIX spot | Derivable |
| `vix_zscore_15m` | rolling z-score of VIX spot | Derivable |
| `vix_percentile_1h` | rolling percentile rank of VIX spot | Derivable |
| `vix_term_slope` | fallback proxy from `iv_skew` and `call_iv`/`put_iv` | Approximate |
| `vvix_level` | not available in fetcher | Missing -> fallback to VIX volatility proxy |
| `vix_vix1d_spread` | not available in fetcher | Missing -> fallback synthetic spread proxy |
| `vix_hilo_range` | rolling high-low range ratio | Derivable |

## Data Quality and Fail-safe Contract

The bridge emits `quality` diagnostics used by inference gating:
- `feature_completeness`: fraction of non-missing source values before fallback
- `warmup_ready`: whether enough rolling history exists for sequence/2D/VIX context
- `symbols_ready`: symbols with valid tensors

Default suppression conditions:
- Any required stage-1 symbol tensor missing
- VIX feature vector invalid
- Completeness below configured threshold

In suppression mode, dashboard shows `insufficient data` and no actionable signal.
