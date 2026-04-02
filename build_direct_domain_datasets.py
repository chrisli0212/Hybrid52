#!/usr/bin/env python3
"""
build_direct_domain_datasets.py
================================
Direct CSV → Per-Agent-Domain .npy Builder (No DuckDB, No Tier1/Tier2)

Reads /workspace/historical_data_1yr/{SYMBOL}/{SYMBOL}_historical_*.csv directly
and produces per-symbol, per-horizon, per-domain .npy training datasets:

/workspace/data/direct_domain/{SYMBOL}/horizon_{N}min/
  greek_sequences.npy           (N, SEQ_LEN, D_GREEK)  → Agents A, B, C, K
  quote_sequences.npy           (N, SEQ_LEN, D_QUOTE)  → Agent Q
  microstructure_sequences.npy  (N, SEQ_LEN, D_MICRO)  → Agent T
  chain_2d.npy                  (N, 5, 20, SEQ_LEN)    → Agent 2D
  labels.npy                    (N,) shared
  returns.npy                   (N,) shared raw returns
  timestamps.npy                (N,) int64 ns shared alignment
  split_indices.json            {train/val/test ranges} shared
  norm_{domain}_mean.npy        per-domain z-score mean
  norm_{domain}_std.npy         per-domain z-score std
  metadata.json                 build stats

For VIXW symbol: also writes vix_sequences.npy (N, VIX_LOOKBACK, 10)

Filter policy (quote-safe, not aggressive):
  - bid > 0 AND ask > 0 AND ask >= bid (valid quote)
  - strike IS NOT NULL AND timestamp valid
  - NO volume > 0 requirement (preserves all quoted minutes)
  - flat-return filter applied at SEQUENCE level only (|ret| < 0.0003)

Smoke-test mode: --smoke (use only first 2 CSV files per symbol)
Mass mode:       --all-symbols OR --symbol SPXW

Bug fixes applied (v2):
  [BUG-1] _feat_matrix(): missing columns path returned bare ndarray instead of
          (ndarray, list) tuple — causes ValueError on unpacking. Fixed to always
          return (mat, avail).
  [BUG-2] GREEK_COLS / QUOTE_COLS / MICRO_COLS list literals were never closed
          (missing closing bracket ']'). Fixed — all three lists now properly
          terminated.
  [BUG-3] Overlapping column names (spread, spread_pct, implied_vol, bid, ask)
          shared across GREEK/QUOTE/MICRO domains. The old join-then-drop-_dup
          approach silently clobbered domain-specific values. Fixed by renaming
          aggregated columns with a domain prefix before merging, then extracting
          per-domain subsets cleanly in _make_sequences().
  [BUG-4] Chain timestamp lookup: chain_map keys are pd.Timestamp objects, but
          minute_df.index.to_numpy() returns numpy.datetime64. Lookups always
          missed → chain_2d.npy was all-zeros. Fixed by converting chain_map keys
          to pd.Timestamp on build, and looking up via pd.Timestamp(ts).
  [BUG-5] _zscore_normalize() flattens to (-1, last_dim) which works for 3-D
          sequence tensors but is wrong for 4-D chain tensors (N, 5, 20, SEQ_LEN).
          Chain normalization now uses reshape(-1, chain_all.shape[1]) so stats
          are computed per-Greek-channel, not per-timestep.
  [BUG-6] VIX rolling percentile used raw=False (correct), but vix_hilo_range
          used raw=True then accessed x.iloc[0] — crashes with numpy array.
          Fixed to raw=False for both lambdas.
  [BUG-7] VIX: price_series concatenated across all CSV files continuously,
          allowing overnight rolling-feature contamination. Fixed by computing
          per-session features then concatenating results.
  [BUG-8] logging.basicConfig() call missing closing ')'. Fixed.
  [BUG-9] run_domain_datasets.sh output path was /workspace/data/direct_domain;
          user requested /workspace/data. Bash script OUT_ROOT corrected.

Usage:
  python build_direct_domain_datasets.py --smoke --symbol SPXW
  python build_direct_domain_datasets.py --all-symbols --horizons 5 15 30
  python build_direct_domain_datasets.py --symbol VIXW --horizons 15
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CSV_ROOT    = Path("/workspace/historical_data_1yr")
OUTPUT_ROOT = Path("/workspace/data")           # BUG-9 fix: was /workspace/data/direct_domain

ALL_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
VIX_SYMBOL  = "VIXW"
SEQ_LEN     = 20       # minutes per sequence window (matches existing Tier3)
HORIZONS    = [5, 15, 30]
RETURN_THRESHOLD = 0.0003   # flat-filter at sequence level; set 0 to disable

# ── Domain feature columns from CSV ──────────────────────────────────────────
# BUG-2 fix: all three lists now properly closed with ']'
# Agents A/B/C/K
GREEK_COLS = [
    "delta", "theta", "vega", "gamma", "vanna", "charm", "lambda",
    "implied_vol",
    "moneyness", "dist_atm_pct", "mid", "spread", "spread_pct",
    "dte", "cp_sign",
]   # ← was missing

# Agent Q: quote-depth / order-book features
QUOTE_COLS = [
    "bid_size", "ask_size", "bid", "ask", "spread", "spread_pct", "implied_vol",
]   # ← was missing

# Agent T: microstructure + trade-flow proxies
MICRO_COLS = [
    "volume", "count", "vwap",
    "open", "high", "low", "close",
    "spread", "bid", "ask",
]   # ← was missing

# BUG-3 fix: build per-domain unique column names to avoid join collisions.
# Columns that appear in multiple domains are renamed with a prefix so they
# can be stored separately in the minute DataFrame and extracted cleanly later.
_GREEK_UNIQUE = [f"g__{c}" for c in GREEK_COLS]
_QUOTE_UNIQUE = [f"q__{c}" for c in QUOTE_COLS]
_MICRO_UNIQUE = [f"m__{c}" for c in MICRO_COLS]

# Chain 2D
CHAIN_GREEKS   = ["delta", "gamma", "theta", "vega", "implied_vol"]
CHAIN_N_STRIKES = 20

# VIX
VIX_LOOKBACK = 4
VIX_RESAMPLE = "5min"
VIX_FEAT_DIM = 10

REQUIRED_COLS = ["timestamp", "strike", "right", "bid", "ask"]

# BUG-8 fix: missing closing ')' on basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADING & QUOTE-SAFE FILTER
# ─────────────────────────────────────────────────────────────────────────────

def _all_csv_files(symbol: str, csv_root: Path, smoke: bool) -> list[Path]:
    pattern = str(csv_root / symbol / f"{symbol}_historical_*.csv")
    files = sorted(Path(f) for f in glob.glob(pattern))
    if not files:
        pattern2 = str(csv_root / f"{symbol}_historical_*.csv")
        files = sorted(Path(f) for f in glob.glob(pattern2))
    if smoke:
        files = files[:2]
        logger.info(f"[{symbol}] SMOKE MODE — using {len(files)} file(s)")
    return files


def _load_csv(path: Path, needed_cols: list[str]) -> pd.DataFrame:
    """Load CSV, apply quote-safe filter, parse timestamp."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        logger.warning(f"  Read error {path.name}: {e}")
        return pd.DataFrame()

    for col in ["bid", "ask", "strike"]:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[
        (pd.to_numeric(df["bid"],    errors="coerce") > 0) &
        (pd.to_numeric(df["ask"],    errors="coerce") > 0) &
        (pd.to_numeric(df["ask"],    errors="coerce") >= pd.to_numeric(df["bid"], errors="coerce")) &
        (pd.to_numeric(df["strike"], errors="coerce").notna())
    ].copy()

    if df.empty:
        return df

    ts_col = next((c for c in ["timestamp", "datetime"] if c in df.columns), None)
    if ts_col is None:
        return pd.DataFrame()
    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["_ts"])

    for col in needed_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PER-MINUTE AGGREGATION
# BUG-3 fix: each domain is aggregated into prefixed columns so overlapping
# names (spread, bid, ask, implied_vol, spread_pct) never collide.
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_minute(df: pd.DataFrame) -> pd.DataFrame:
    """
    From per-contract rows → one row per minute.
    Columns are stored with domain prefixes (g__, q__, m__) to prevent collision.
    """
    df = df.copy()
    df["_minute"] = df["_ts"].dt.floor("min")

    # ATM filter for Greeks
    if "dist_atm_pct" in df.columns:
        df["dist_atm_pct"] = pd.to_numeric(df["dist_atm_pct"], errors="coerce")
        atm = df[df["dist_atm_pct"].abs() < 5.0]
        if len(atm) == 0:
            atm = df
    else:
        atm = df

    # ── Greeks (prefixed g__) ────────────────────────────────────────────────
    greek_src = {c: "mean" for c in GREEK_COLS if c in atm.columns}
    if greek_src:
        gdf = atm.groupby("_minute").agg(greek_src)
        gdf.columns = [f"g__{c}" for c in gdf.columns]
    else:
        gdf = pd.DataFrame()

    # ── Quote (prefixed q__) ─────────────────────────────────────────────────
    q_sum = [c for c in ["bid_size", "ask_size"] if c in df.columns]
    q_mean = [c for c in QUOTE_COLS if c not in q_sum and c in df.columns]
    q_parts = []
    if q_sum:
        tmp = df.groupby("_minute")[q_sum].sum()
        tmp.columns = [f"q__{c}" for c in tmp.columns]
        q_parts.append(tmp)
    if q_mean:
        tmp = df.groupby("_minute")[q_mean].mean()
        tmp.columns = [f"q__{c}" for c in tmp.columns]
        q_parts.append(tmp)
    qdf = pd.concat(q_parts, axis=1) if q_parts else pd.DataFrame()

    # ── Microstructure (prefixed m__) ────────────────────────────────────────
    m_sum  = [c for c in ["volume", "count"] if c in df.columns]
    m_mean = [c for c in MICRO_COLS if c not in m_sum and c in df.columns]
    m_parts = []
    if m_sum:
        tmp = df.groupby("_minute")[m_sum].sum()
        tmp.columns = [f"m__{c}" for c in tmp.columns]
        m_parts.append(tmp)
    if m_mean:
        tmp = df.groupby("_minute")[m_mean].mean()
        tmp.columns = [f"m__{c}" for c in tmp.columns]
        m_parts.append(tmp)
    mdf = pd.concat(m_parts, axis=1) if m_parts else pd.DataFrame()

    # ── Underlying price & contract count ────────────────────────────────────
    up_col = "underlying_price" if "underlying_price" in df.columns else None
    meta_agg = {"_ts": "count"}
    if up_col:
        meta_agg[up_col] = "last"
    meta = df.groupby("_minute").agg(meta_agg).rename(columns={"_ts": "_contract_count"})

    # ── Merge on minute index (no collision possible with prefixes) ──────────
    result = meta.copy()
    for part in [gdf, qdf, mdf]:
        if not part.empty:
            result = result.join(part, how="left")

    result.index.name = "minute"
    return result.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN 2D BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_chain_snapshot(df_minute: pd.DataFrame) -> np.ndarray:
    """Build (5, CHAIN_N_STRIKES) Greek array for one minute."""
    out = np.zeros((5, CHAIN_N_STRIKES), dtype=np.float32)
    needed = CHAIN_GREEKS + ["dist_atm_pct"]
    if not all(c in df_minute.columns for c in needed):
        return out

    sub = df_minute[needed].dropna()
    if len(sub) < 3:
        return out

    sub = sub.sort_values("dist_atm_pct").reset_index(drop=True)
    atm_idx = int((sub["dist_atm_pct"].abs()).idxmin())
    half = CHAIN_N_STRIKES // 2
    lo = max(0, atm_idx - half)
    hi = lo + CHAIN_N_STRIKES
    if hi > len(sub):
        hi = len(sub)
        lo = max(0, hi - CHAIN_N_STRIKES)
    window = sub.iloc[lo:hi]
    n = len(window)
    for i, g in enumerate(CHAIN_GREEKS):
        out[i, :n] = window[g].to_numpy(dtype=np.float32)
    return out


def _build_chain_series(df: pd.DataFrame) -> dict:
    """
    Build per-minute chain snapshots.
    BUG-4 fix: keys stored as pd.Timestamp so lookup in _make_sequences matches.
    """
    if "dist_atm_pct" not in df.columns:
        return {}
    df = df.copy()
    df["_minute"] = df["_ts"].dt.floor("min")
    needed = CHAIN_GREEKS + ["dist_atm_pct", "_minute"]
    sub = df[[c for c in needed if c in df.columns]].dropna(subset=["dist_atm_pct"])
    chains: dict = {}
    for ts, grp in sub.groupby("_minute"):
        # ts is already pd.Timestamp when groupby is on a datetime column
        chains[pd.Timestamp(ts)] = _build_chain_snapshot(grp)
    return chains


# ─────────────────────────────────────────────────────────────────────────────
# VIX FEATURE EXTRACTION (VIXW only)
# BUG-6 fix: vix_hilo_range lambda used raw=True then x.iloc[0] → crash.
#            Both percentile and hilo_range now use raw=False.
# BUG-7 fix: features computed per-session to avoid overnight rolling leakage.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_vix_features_session(close: pd.Series, m: int = 1) -> pd.DataFrame:
    """
    10-dim VIX features for a single trading session (no overnight contamination).
    """
    feat = pd.DataFrame(index=close.index)
    feat["vix_level"]    = close
    feat["vix_pct_5m"]   = close.pct_change(1 * m).fillna(0.0)
    feat["vix_pct_15m"]  = close.pct_change(3 * m).fillna(0.0)
    feat["vix_pct_1h"]   = close.pct_change(12 * m).fillna(0.0)

    r100 = 20 * m
    roll_mean = close.rolling(r100, min_periods=5).mean()
    roll_std  = close.rolling(r100, min_periods=5).std()
    feat["vix_zscore_15m"] = np.where(roll_std > 1e-6, (close - roll_mean) / roll_std, 0.0)

    r60 = 12 * m
    # BUG-6 fix: raw=False so x is a pd.Series and .iloc[-1] works correctly
    feat["vix_percentile_1h"] = close.rolling(r60, min_periods=3).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / max(len(x) - 1, 1),
        raw=False
    )

    feat["vix_term_slope"] = close.pct_change(6 * m).fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    feat["vvix_level"] = ret.rolling(r100, min_periods=5).std().fillna(0.0)

    sess_open = close.iloc[0] if len(close) > 0 else 1.0
    feat["vix_vix1d_spread"] = np.where(
        sess_open > 0, (close - sess_open) / sess_open, 0.0
    )

    # BUG-6 fix: raw=False; x is pd.Series so x.iloc[0] is valid
    feat["vix_hilo_range"] = close.rolling(5, min_periods=1).apply(
        lambda x: (x.max() - x.min()) / max(abs(x.iloc[0]), 1e-6),
        raw=False
    ).fillna(0.0)

    return feat.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def _extract_vix_features(price_series: pd.Series, resolution: str = "1min") -> pd.DataFrame:
    """
    Compute VIX features per trading session, then concatenate.
    BUG-7 fix: prevents overnight rolling contamination.
    """
    m = 1 if resolution == "1min" else 5
    session_feats = []
    for _date, session in price_series.groupby(price_series.index.date):
        if len(session) < 2:
            continue
        session_feats.append(_extract_vix_features_session(session, m))
    if not session_feats:
        return pd.DataFrame()
    return pd.concat(session_feats).sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# BUG-5 fix: _zscore_normalize works correctly for 3-D (N,T,F) tensors.
#            Chain 4-D normalization uses a dedicated function below.
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_normalize(
    train_data: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score using train stats. Works for (N, T, F) tensors."""
    flat  = train_data.reshape(-1, train_data.shape[-1])
    mean  = flat.mean(axis=0).astype(np.float32)
    std   = flat.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return ((data - mean) / std).astype(np.float32), mean, std


def _zscore_normalize_chain(
    train_data: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BUG-5 fix: Chain tensor shape is (N, 5, CHAIN_N_STRIKES, SEQ_LEN).
    Normalize per Greek channel (axis-1), so stats have shape (5,).
    """
    # Flatten all dims except the Greek-channel axis (axis 1)
    # train_data: (N_train, 5, 20, SEQ_LEN)
    N_train = train_data.shape[0]
    n_greek = train_data.shape[1]
    flat = train_data.reshape(N_train, n_greek, -1)   # (N_train, 5, 20*SEQ_LEN)
    mean = flat.mean(axis=(0, 2)).astype(np.float32)  # (5,)
    std  = flat.std(axis=(0, 2)).astype(np.float32)   # (5,)
    std[std < 1e-8] = 1.0

    # Broadcast mean/std over (N, 5, 20, SEQ_LEN)
    mean_bc = mean[None, :, None, None]
    std_bc  = std[None,  :, None, None]
    normed  = ((data - mean_bc) / std_bc).astype(np.float32)
    return normed, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _make_sequences(
    minute_df: pd.DataFrame,
    chain_map: dict,
    horizon: int,
    symbol: str,
) -> dict | None:
    """
    Slide SEQ_LEN window over minute_df, compute labels, filter flat samples.
    Returns dict with per-domain raw (un-normalized) arrays + metadata.
    """
    n = len(minute_df)
    if n < SEQ_LEN + horizon + 1:
        return None

    # BUG-4 fix: keep timestamps as pd.Timestamp for chain_map lookup
    timestamps_pd  = minute_df.index.to_list()             # list of pd.Timestamp
    timestamps_ns  = np.array([t.value for t in timestamps_pd], dtype=np.int64)

    up_col = "underlying_price"
    if up_col not in minute_df.columns:
        logger.warning(f"[{symbol}] No underlying_price column — skipping")
        return None
    prices = minute_df[up_col].to_numpy(dtype=np.float64)

    # BUG-1 fix: _feat_matrix always returns (mat, avail) tuple
    def _feat_matrix(prefixed_cols: list[str], orig_cols: list[str]):
        avail_prefixed = [c for c in prefixed_cols if c in minute_df.columns]
        avail_orig     = [c.split("__", 1)[1] for c in avail_prefixed]  # strip prefix for metadata
        if not avail_prefixed:
            return np.zeros((n, 1), dtype=np.float32), []               # ← always tuple
        mat = minute_df[avail_prefixed].to_numpy(dtype=np.float32)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        return mat, avail_orig

    greek_mat, greek_avail = _feat_matrix(_GREEK_UNIQUE, GREEK_COLS)
    quote_mat, quote_avail = _feat_matrix(_QUOTE_UNIQUE, QUOTE_COLS)
    micro_mat, micro_avail = _feat_matrix(_MICRO_UNIQUE, MICRO_COLS)

    # ── Chain 2D matrix ─────────────────────────────────────────────────────
    chain_arr = np.zeros((n, 5, CHAIN_N_STRIKES), dtype=np.float32)
    for i, ts in enumerate(timestamps_pd):
        snap = chain_map.get(ts)     # BUG-4 fix: lookup with pd.Timestamp key
        if snap is not None:
            chain_arr[i] = snap

    # ── Slide window ────────────────────────────────────────────────────────
    seqs_greek, seqs_quote, seqs_micro, seqs_chain = [], [], [], []
    labels_list, returns_list, ts_list = [], [], []

    max_i = n - SEQ_LEN - horizon
    for i in range(max_i):
        cur_price = prices[i + SEQ_LEN - 1]
        fut_price = prices[i + SEQ_LEN - 1 + horizon]
        if cur_price <= 0 or fut_price <= 0:
            continue
        ret = (fut_price - cur_price) / cur_price
        if RETURN_THRESHOLD > 0 and abs(ret) < RETURN_THRESHOLD:
            continue

        seqs_greek.append(greek_mat[i:i + SEQ_LEN])
        seqs_quote.append(quote_mat[i:i + SEQ_LEN])
        seqs_micro.append(micro_mat[i:i + SEQ_LEN])

        # Chain: (SEQ_LEN, 5, 20) → transpose to (5, 20, SEQ_LEN) for Conv2D
        chain_seq = chain_arr[i:i + SEQ_LEN]             # (SEQ_LEN, 5, 20)
        seqs_chain.append(chain_seq.transpose(1, 2, 0))  # (5, 20, SEQ_LEN)

        labels_list.append(1 if ret > 0 else 0)
        returns_list.append(float(ret))
        ts_list.append(timestamps_ns[i + SEQ_LEN - 1])

    if not seqs_greek:
        return None

    return {
        "greek":      np.array(seqs_greek, dtype=np.float32),
        "quote":      np.array(seqs_quote, dtype=np.float32),
        "micro":      np.array(seqs_micro, dtype=np.float32),
        "chain":      np.array(seqs_chain, dtype=np.float32),
        "labels":     np.array(labels_list, dtype=np.int64),
        "returns":    np.array(returns_list, dtype=np.float32),
        "timestamps": np.array(ts_list, dtype=np.int64),
        "greek_cols": greek_avail,
        "quote_cols": quote_avail,
        "micro_cols": micro_avail,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _save_domain(
    out_dir: Path, name: str,
    train: np.ndarray, val: np.ndarray, test: np.ndarray,
):
    """Normalize using train stats and save {name}_sequences.npy + norm files."""
    all_data = np.concatenate([train, val, test], axis=0)
    normed, mean, std = _zscore_normalize(train, all_data)
    np.save(out_dir / f"{name}_sequences.npy", normed)
    np.save(out_dir / f"norm_{name}_mean.npy", mean)
    np.save(out_dir / f"norm_{name}_std.npy",  std)
    logger.info(
        f"  Saved {name}_sequences.npy shape={normed.shape} "
        f"({len(train)}/{len(val)}/{len(test)} train/val/test)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SYMBOL PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_symbol(
    symbol: str, horizons: list[int], smoke: bool,
    csv_root: Path, output_root: Path,
):
    t0 = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"[{symbol}] Starting direct CSV → domain dataset build")

    files = _all_csv_files(symbol, csv_root, smoke)
    if not files:
        logger.error(f"[{symbol}] No CSV files found under {csv_root / symbol}")
        return

    all_needed = list(set(
        GREEK_COLS + QUOTE_COLS + MICRO_COLS +
        CHAIN_GREEKS + ["underlying_price", "dist_atm_pct"]
    ))

    minute_frames: list[tuple] = []
    total_raw_rows = 0

    for fpath in files:
        logger.info(f"  Reading {fpath.name} ...")
        df_raw = _load_csv(fpath, all_needed)
        if df_raw.empty:
            continue
        total_raw_rows += len(df_raw)

        df_raw["_date"] = df_raw["_ts"].dt.date
        for _day, day_df in df_raw.groupby("_date"):
            day_df  = day_df.sort_values("_ts")
            min_df  = _aggregate_minute(day_df)
            chain_map = _build_chain_series(day_df)
            if len(min_df) >= SEQ_LEN + 2:
                minute_frames.append((min_df, chain_map))

    logger.info(f"[{symbol}] Loaded {total_raw_rows:,} raw rows → {len(minute_frames)} trading days")

    if not minute_frames:
        logger.error(f"[{symbol}] No valid minute data — aborting")
        return

    for horizon in horizons:
        logger.info(f"\n[{symbol}] Building horizon={horizon}min sequences ...")
        day_results = []
        for min_df, chain_map in minute_frames:
            res = _make_sequences(min_df, chain_map, horizon, symbol)
            if res:
                day_results.append(res)

        if not day_results:
            logger.warning(f"[{symbol}] h{horizon}: no sequences generated")
            continue

        greek_all  = np.concatenate([r["greek"]      for r in day_results], axis=0)
        quote_all  = np.concatenate([r["quote"]      for r in day_results], axis=0)
        micro_all  = np.concatenate([r["micro"]      for r in day_results], axis=0)
        chain_all  = np.concatenate([r["chain"]      for r in day_results], axis=0)
        labels_all = np.concatenate([r["labels"]     for r in day_results], axis=0)
        returns_all= np.concatenate([r["returns"]    for r in day_results], axis=0)
        ts_all     = np.concatenate([r["timestamps"] for r in day_results], axis=0)

        N  = len(labels_all)
        i1 = int(N * 0.60)
        i2 = int(N * 0.80)
        logger.info(
            f"[{symbol}] h{horizon}: {N:,} total sequences "
            f"(train={i1:,} val={i2-i1:,} test={N-i2:,}) "
            f"UP%={100*labels_all.mean():.1f}%"
        )

        out_dir = output_root / symbol / f"horizon_{horizon}min"
        out_dir.mkdir(parents=True, exist_ok=True)

        _save_domain(out_dir, "greek",          greek_all[:i1], greek_all[i1:i2], greek_all[i2:])
        _save_domain(out_dir, "quote",          quote_all[:i1], quote_all[i1:i2], quote_all[i2:])
        _save_domain(out_dir, "microstructure", micro_all[:i1], micro_all[i1:i2], micro_all[i2:])

        # BUG-5 fix: chain-specific normalization per Greek channel
        chain_normed, chain_mean, chain_std = _zscore_normalize_chain(chain_all[:i1], chain_all)
        np.save(out_dir / "chain_2d.npy",         chain_normed)
        np.save(out_dir / "norm_chain_mean.npy",   chain_mean)
        np.save(out_dir / "norm_chain_std.npy",    chain_std)
        logger.info(f"  Saved chain_2d.npy shape={chain_normed.shape}")

        np.save(out_dir / "labels.npy",     labels_all)
        np.save(out_dir / "returns.npy",    returns_all)
        np.save(out_dir / "timestamps.npy", ts_all)

        split_info = {
            "n_total":       int(N),
            "n_train":       int(i1),
            "n_val":         int(i2 - i1),
            "n_test":        int(N - i2),
            "train_end_idx": int(i1),
            "val_end_idx":   int(i2),
            "up_pct_train":  round(float(labels_all[:i1].mean())      * 100, 2),
            "up_pct_val":    round(float(labels_all[i1:i2].mean())    * 100, 2),
            "up_pct_test":   round(float(labels_all[i2:].mean())      * 100, 2),
        }
        (out_dir / "split_indices.json").write_text(json.dumps(split_info, indent=2))

        greek_cols = day_results[0].get("greek_cols", GREEK_COLS)
        quote_cols = day_results[0].get("quote_cols", QUOTE_COLS)
        micro_cols = day_results[0].get("micro_cols", MICRO_COLS)

        metadata = {
            "symbol": symbol, "horizon_min": horizon, "seq_len": SEQ_LEN,
            "build_time": datetime.now().isoformat(),
            "source": str(csv_root / symbol),
            "n_csv_files": len(files),
            "n_trading_days": len(minute_frames),
            "total_raw_rows": total_raw_rows,
            "return_threshold": RETURN_THRESHOLD,
            "filter_policy": "bid>0 AND ask>0 AND ask>=bid (no volume>0 requirement)",
            "greek_feat_dim": greek_all.shape[2],  "greek_cols": greek_cols,
            "quote_feat_dim": quote_all.shape[2],  "quote_cols": quote_cols,
            "micro_feat_dim": micro_all.shape[2],  "micro_cols": micro_cols,
            "chain_shape": list(chain_all.shape[1:]),
            "smoke_test": len(files) <= 2,
            **split_info,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info(f"[{symbol}] h{horizon}: ✅ Written to {out_dir}")

        del greek_all, quote_all, micro_all, chain_all, chain_normed
        del labels_all, returns_all, ts_all, day_results
        gc.collect()

    elapsed = time.time() - t0
    logger.info(f"[{symbol}] Done in {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# VIX PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_vix(horizons: list[int], smoke: bool, csv_root: Path, output_root: Path):
    """
    Build vix_sequences.npy from VIXW historical CSVs.
    VIX regime labels: 0=CALM(<15), 1=NORMAL(<20), 2=ELEVATED(<25), 3=HIGH(<35), 4=EXTREME
    BUG-6 + BUG-7: VIX feature extraction now per-session, fixes raw=True crash.
    """
    symbol = VIX_SYMBOL
    t0 = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"[{symbol}] Building VIX regime sequences")

    files = _all_csv_files(symbol, csv_root, smoke)
    if not files:
        logger.error(f"[{symbol}] No VIXW CSV files found")
        return

    thresholds = [15.0, 20.0, 25.0, 35.0]

    price_series_parts = []
    for fpath in files:
        df = _load_csv(fpath, ["underlying_price", "implied_vol"])
        if df.empty:
            continue
        df["_minute"] = df["_ts"].dt.floor("min")
        up_col = "underlying_price" if "underlying_price" in df.columns else None
        if up_col is None:
            continue
        pmin = df.groupby("_minute")[up_col].last()
        price_series_parts.append(pmin)

    if not price_series_parts:
        logger.error(f"[{symbol}] No price data found")
        return

    price_series = pd.concat(price_series_parts).sort_index()
    # Remove duplicates that arise from overlapping files
    price_series = price_series[~price_series.index.duplicated(keep="last")]
    price_series = price_series[price_series > 0].dropna()
    logger.info(f"[{symbol}] {len(price_series):,} 1-min VIX bars")

    # VIX regime labels
    levels = price_series.to_numpy(dtype=np.float32)
    regime_labels = np.zeros(len(levels), dtype=np.int64)
    for i, thr in enumerate(thresholds):
        regime_labels[levels >= thr] = i + 1

    # BUG-7 fix: per-session feature extraction
    feat_df = _extract_vix_features(price_series, resolution="1min")
    if feat_df.empty:
        logger.error(f"[{symbol}] VIX feature extraction produced empty result")
        return

    # Align regime_labels to feat_df index (sessions may drop short days)
    regime_series = pd.Series(regime_labels, index=price_series.index)
    regime_series = regime_series.reindex(feat_df.index)
    feat_arr = feat_df.to_numpy(dtype=np.float32)
    regime_arr = regime_series.to_numpy(dtype=np.int64)
    N = len(feat_arr)

    if N < VIX_LOOKBACK + 2:
        logger.error(f"[{symbol}] Not enough bars for VIX lookback")
        return

    seqs_x, seqs_y = [], []
    for i in range(N - VIX_LOOKBACK + 1):
        seqs_x.append(feat_arr[i:i + VIX_LOOKBACK])
        seqs_y.append(regime_arr[i + VIX_LOOKBACK - 1])

    X = np.array(seqs_x, dtype=np.float32)
    y = np.array(seqs_y, dtype=np.int64)

    n  = len(X)
    i1 = int(n * 0.60)
    i2 = int(n * 0.80)

    flat_train = X[:i1].reshape(-1, X.shape[-1])
    mean_v = flat_train.mean(axis=0).astype(np.float32)
    std_v  = flat_train.std(axis=0).astype(np.float32)
    std_v[std_v < 1e-8] = 1.0
    X_norm = ((X - mean_v) / std_v).astype(np.float32)

    regime_names = ["CALM", "NORMAL", "ELEVATED", "HIGH", "EXTREME"]
    regime_dist  = {regime_names[i]: int((y == i).sum()) for i in range(5)}

    for horizon in horizons:
        out_dir = output_root / symbol / f"horizon_{horizon}min"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "vix_sequences.npy", X_norm)
        np.save(out_dir / "vix_labels.npy",    y)
        np.save(out_dir / "norm_vix_mean.npy", mean_v)
        np.save(out_dir / "norm_vix_std.npy",  std_v)

        metadata = {
            "symbol": symbol, "horizon_min": horizon,
            "vix_lookback": VIX_LOOKBACK, "vix_feat_dim": VIX_FEAT_DIM,
            "n_sequences": int(n), "n_train": int(i1),
            "n_val": int(i2 - i1), "n_test": int(n - i2),
            "regime_distribution": regime_dist,
            "regime_thresholds": thresholds,
            "build_time": datetime.now().isoformat(),
            "smoke_test": len(files) <= 2,
        }
        (out_dir / "vix_metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info(f"[{symbol}] h{horizon}: ✅ vix_sequences.npy shape={X_norm.shape}")

    logger.info(f"[{symbol}] Done in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_report(output_root: Path, symbols: list[str], horizons: list[int]):
    """Print a quick sanity summary of what was written."""
    print(f"\n{'='*60}")
    print("SMOKE TEST REPORT")
    print(f"{'='*60}")
    domain_files = [
        "greek_sequences.npy", "quote_sequences.npy",
        "microstructure_sequences.npy", "chain_2d.npy",
    ]
    for sym in symbols:
        for h in horizons:
            d = output_root / sym / f"horizon_{h}min"
            if not d.exists():
                print(f"  {sym} h{h}: ❌ directory missing")
                continue
            print(f"\n  {sym} h{h}min → {d}")
            for fname in domain_files:
                fp = d / fname
                if fp.exists():
                    arr  = np.load(fp, mmap_mode="r")
                    flat = arr.reshape(arr.shape[0], -1)
                    live = int((np.std(flat, axis=0) > 1e-8).sum())
                    total_dims = flat.shape[1]
                    print(f"    ✅ {fname:35s} shape={arr.shape} live_dims={live}/{total_dims}")
                else:
                    print(f"    ❌ {fname:35s} MISSING")
            lf = d / "labels.npy"
            if lf.exists():
                lbl = np.load(lf)
                print(f"    ✅ labels.npy n={len(lbl)} UP%={100*lbl.mean():.1f}%")
            meta_f = d / "metadata.json"
            if meta_f.exists():
                m = json.loads(meta_f.read_text())
                print(f"    ℹ  filter={m.get('filter_policy','?')}")
    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global RETURN_THRESHOLD
    parser = argparse.ArgumentParser(
        description="Direct CSV → Per-Agent-Domain .npy Builder (No DuckDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol",      type=str,  default=None,
                        help="Single symbol to process (e.g. SPXW)")
    parser.add_argument("--all-symbols", action="store_true",
                        help="Process all main symbols (SPXW SPY QQQ IWM TLT)")
    parser.add_argument("--vixw",        action="store_true",
                        help="Also process VIXW → vix_sequences.npy")
    parser.add_argument("--horizons",    type=int, nargs="+", default=HORIZONS,
                        help="Forward-return horizons in minutes (default: 5 15 30)")
    parser.add_argument("--smoke",       action="store_true",
                        help="Smoke test: use only first 2 CSV files per symbol")
    parser.add_argument("--csv-root",    type=str, default=str(CSV_ROOT),
                        help=f"CSV source root (default: {CSV_ROOT})")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT),
                        help=f"Output root (default: {OUTPUT_ROOT})")
    parser.add_argument("--return-threshold", type=float, default=RETURN_THRESHOLD,
                        help=f"Flat-return filter threshold (default: {RETURN_THRESHOLD}). Set 0 to disable.")
    args = parser.parse_args()

    RETURN_THRESHOLD = args.return_threshold

    csv_root    = Path(args.csv_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not csv_root.exists():
        logger.error(f"CSV root not found: {csv_root}")
        sys.exit(1)

    if args.all_symbols:
        symbols = ALL_SYMBOLS
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        parser.error("Specify --symbol SYMBOL or --all-symbols")

    print(f"\n{'='*60}")
    print("BUILD DIRECT DOMAIN DATASETS  (v2 — bug-fixed)")
    print(f"  CSV source : {csv_root}")
    print(f"  Output     : {output_root}")
    print(f"  Symbols    : {symbols}")
    print(f"  Horizons   : {args.horizons}")
    print(f"  Seq len    : {SEQ_LEN}")
    print(f"  Return filter: {RETURN_THRESHOLD} ({'disabled' if RETURN_THRESHOLD == 0 else 'enabled'})")
    print(f"  Smoke test : {args.smoke}")
    print(f"  Filter policy: bid>0 AND ask>0 AND ask>=bid (NO volume>0)")
    print(f"{'='*60}\n")

    for sym in symbols:
        # BUG-9 companion: skip running process_vix via process_symbol path for VIXW
        if sym.upper() == VIX_SYMBOL.upper():
            continue
        process_symbol(sym, args.horizons, args.smoke, csv_root, output_root)

    # VIX: only via dedicated path
    if args.vixw or VIX_SYMBOL.upper() in [s.upper() for s in symbols]:
        process_vix(args.horizons, args.smoke, csv_root, output_root)

    if args.smoke:
        _smoke_report(output_root, [s for s in symbols if s.upper() != VIX_SYMBOL.upper()], args.horizons)

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
