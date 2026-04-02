#!/usr/bin/env python3
"""
Build tier2 minute-bar features from tier1_hybrid55 per-trade_date parquets.

⚠️  DEPRECATION WARNING ⚠️
This script still uses the OLD master_extractor_v2.MasterFeatureExtractor which has been
moved to old_mixed_processing/ backup. This processes all agent features together in a
monolithic fashion, which caused training failures in Hybrid55.

TODO: Refactor this script to use per-agent extractors from hybrid55_preprocessing/agents/
For each agent that needs tier2 data, call the dedicated agent extractor instead of
MasterFeatureExtractor.

Tier1 layout per date:
  {YYYY-MM-DD}_greek.parquet
  {YYYY-MM-DD}_trade.parquet
  {YYYY-MM-DD}_quote.parquet
  {YYYY-MM-DD}_ohlc.parquet
  Optional legacy: {YYYY-MM-DD}_tq.parquet (combined)

When split trade+quote files exist, they are preferred over legacy *_tq.parquet so Tier2
stays aligned with the split extract path (extract_tier1_joined.py).

Reads per-tradedate Greek + TQ parquets that already have '_minute' column,
eliminating the week_key matching and ±1min tolerance lookup.

~3-5× faster than the original build_tier2.py / tier2_reprocess.py.

Usage:
  python3.13 build_tier2_fast.py --symbol SPXW --workers 10
  python3.13 build_tier2_fast.py --all-symbols --workers 10
"""

import argparse
import json
import logging
import gc
import time
import sys
import os
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import duckdb

# ── Path setup ────────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from hybrid55_preprocessing.chain_2d import Chain2DProcessor
from hybrid55_preprocessing.feature_config_v2 import (
    TOTAL_FEATURES,
    FEATURE_SCHEMA_VERSION,
)
from hybrid55_preprocessing.data_validation import (
    get_excluded_columns,
    get_trade_quote_excluded_columns,
)
from hybrid55_preprocessing.quality_checks import DataQualityChecker

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TIER1_ROOT = Path("/workspace/data/tier1_hybrid55")
OUTPUT_ROOT = Path("/workspace/data/tier2_minutes_hybrid55")
FEAT_DIM = TOTAL_FEATURES  # e.g. 311 in Hybrid55 (feature_config_v2.TOTAL_FEATURES)
ALL_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]

# Global counter for extraction errors (multiprocessing-safe via return value inspection)
_EXTRACTION_ERROR_SAMPLE_LIMIT = 5  # log first N unique error messages per worker
# Use centralized column exclusion from hybrid55_preprocessing.data_validation
HARD_EXCLUDED_GREEK_COLS = set(get_excluded_columns())
HARD_EXCLUDED_TQ_COLS = set(get_trade_quote_excluded_columns())
RECOVERED_GREEK_COLS = ["rho", "epsilon", "vomma", "veta", "color", "dual_delta", "d1", "d2", "ultima"]


def _build_sparse_tq_lookup(
    tq_df: pd.DataFrame,
    greek_minutes: list[pd.Timestamp],
    max_fill_minutes: int = 5,
):
    """Build minute-aligned TQ groups with bounded quote ffill and VWAP fallback."""
    if tq_df is None or tq_df.empty:
        return {}, {
            "quote_fill_applied_minutes": 0,
            "trade_vwap_applied_minutes": 0,
            "fallback_zero_minutes": len(greek_minutes),
        }

    tq_by_minute_raw = {m: g for m, g in tq_df.groupby("_minute")}
    result = {}
    stats = {
        "quote_fill_applied_minutes": 0,
        "trade_vwap_applied_minutes": 0,
        "fallback_zero_minutes": 0,
    }

    quote_state = None
    quote_state_minute = None
    trade_window = []
    sorted_minutes = sorted(greek_minutes)

    for minute_ts in sorted_minutes:
        current_group = tq_by_minute_raw.get(minute_ts)
        if current_group is not None and len(current_group) > 0:
            result[minute_ts] = current_group

            if "bid" in current_group.columns and "ask" in current_group.columns:
                quote_valid = current_group[current_group["bid"].notna() & current_group["ask"].notna()]
                if len(quote_valid) > 0:
                    quote_state = quote_valid.iloc[-1].to_dict()
                    quote_state_minute = minute_ts

            if "price" in current_group.columns and "size" in current_group.columns:
                trade_valid = current_group[
                    current_group["price"].notna() & current_group["size"].notna() &
                    (current_group["price"] > 0) & (current_group["size"] > 0)
                ]
                for _, row in trade_valid.iterrows():
                    trade_window.append((minute_ts, float(row["price"]), float(row["size"])))

            trade_window = [
                t for t in trade_window
                if (minute_ts - t[0]).total_seconds() / 60.0 <= max_fill_minutes
            ]
            continue

        rows = []
        if (
            quote_state is not None and quote_state_minute is not None and
            (minute_ts - quote_state_minute).total_seconds() / 60.0 <= max_fill_minutes
        ):
            q_row = {col: quote_state.get(col, np.nan) for col in tq_df.columns}
            q_row["_minute"] = minute_ts
            if "trade_timestamp" in q_row:
                q_row["trade_timestamp"] = minute_ts
            if "quote_timestamp" in q_row:
                q_row["quote_timestamp"] = minute_ts
            rows.append(q_row)
            stats["quote_fill_applied_minutes"] += 1

        trade_window = [
            t for t in trade_window
            if (minute_ts - t[0]).total_seconds() / 60.0 <= max_fill_minutes
        ]
        if trade_window:
            prices = np.array([p for _, p, _ in trade_window], dtype=np.float64)
            sizes = np.array([s for _, _, s in trade_window], dtype=np.float64)
            size_sum = float(np.maximum(sizes.sum(), 1e-8))
            vwap = float(np.dot(prices, sizes) / size_sum)
            avg_size = float(sizes.mean())

            t_row = {col: np.nan for col in tq_df.columns}
            t_row["_minute"] = minute_ts
            if "trade_timestamp" in t_row:
                t_row["trade_timestamp"] = minute_ts
            if "quote_timestamp" in t_row:
                t_row["quote_timestamp"] = minute_ts
            if "price" in t_row:
                t_row["price"] = vwap
            if "size" in t_row:
                t_row["size"] = avg_size
            if quote_state is not None:
                for col in ("bid", "ask", "bid_size", "ask_size"):
                    if col in t_row:
                        t_row[col] = quote_state.get(col, t_row[col])
            rows.append(t_row)
            stats["trade_vwap_applied_minutes"] += 1

        if rows:
            result[minute_ts] = pd.DataFrame(rows)
        else:
            stats["fallback_zero_minutes"] += 1

    return result, stats


def _build_sparse_ohlc_lookup(
    ohlc_df: pd.DataFrame,
    greek_minutes: list[pd.Timestamp],
    max_fill_minutes: int = 3,
):
    """Build minute-aligned OHLC lookup with bounded carry-forward fallback."""
    if ohlc_df is None or ohlc_df.empty:
        return {}, {
            "ohlc_fill_applied_minutes": 0,
            "ohlc_fallback_zero_minutes": len(greek_minutes),
        }

    grouped = {m: g for m, g in ohlc_df.groupby("_minute")}
    if not grouped:
        return {}, {
            "ohlc_fill_applied_minutes": 0,
            "ohlc_fallback_zero_minutes": len(greek_minutes),
        }

    available = sorted(grouped.keys())
    avail_ns = np.array([pd.Timestamp(m).value for m in available], dtype=np.int64)
    out = {}
    stats = {
        "ohlc_fill_applied_minutes": 0,
        "ohlc_fallback_zero_minutes": 0,
    }

    for minute in sorted(greek_minutes):
        m = pd.Timestamp(minute)
        g = grouped.get(m)
        if g is not None:
            out[m] = g
            continue
        pos = np.searchsorted(avail_ns, m.value) - 1
        if pos >= 0:
            prev_m = available[pos]
            gap_min = int((m - prev_m) / pd.Timedelta(minutes=1))
            if gap_min <= max_fill_minutes:
                out[m] = grouped[prev_m]
                stats["ohlc_fill_applied_minutes"] += 1
                continue
        stats["ohlc_fallback_zero_minutes"] += 1

    return out, stats


def _coverage_ratio(df: pd.DataFrame, cols: list[str], nonzero: bool = True) -> dict[str, float]:
    out = {}
    if df is None or len(df) == 0:
        return {c: 0.0 for c in cols}
    n = float(len(df))
    for c in cols:
        if c not in df.columns:
            out[c] = 0.0
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if nonzero:
            out[c] = float(((s.abs() > 1e-12) & s.notna()).sum() / n)
        else:
            out[c] = float(s.notna().sum() / n)
    return out


def _filter_sparse_ohlc_rows(ohlc_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Drop structurally inactive OHLC rows before minute lookup.

    Rule:
    - drop when count <= 0, OR
    - drop when all OHLCV fields are non-positive/zero.
    """
    if ohlc_df is None or len(ohlc_df) == 0:
        return ohlc_df, {"ohlc_rows_input": 0, "ohlc_rows_kept": 0, "ohlc_rows_filtered": 0}

    df = ohlc_df.copy()
    n_input = int(len(df))

    if "count" in df.columns:
        count_pos = pd.to_numeric(df["count"], errors="coerce").fillna(0.0) > 0
    else:
        count_pos = pd.Series(True, index=df.index)

    ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if ohlcv_cols:
        any_ohlcv_pos = pd.Series(False, index=df.index)
        for c in ohlcv_cols:
            any_ohlcv_pos = any_ohlcv_pos | (pd.to_numeric(df[c], errors="coerce").fillna(0.0) > 0)
    else:
        any_ohlcv_pos = pd.Series(True, index=df.index)

    keep_mask = count_pos & any_ohlcv_pos
    filtered = df.loc[keep_mask].copy()
    n_kept = int(len(filtered))
    return filtered, {
        "ohlc_rows_input": n_input,
        "ohlc_rows_kept": n_kept,
        "ohlc_rows_filtered": n_input - n_kept,
    }


def create_feature_extractor():
    # TEMPORARY: Using old master_extractor from backup until refactored to per-agent extractors
    from hybrid55_preprocessing.old_mixed_processing.master_extractor_v2 import MasterFeatureExtractor
    return MasterFeatureExtractor(
        include_chain_2d=False, include_phase1=False, normalize=False
    )


def _audit_greek_schema(greek_df: pd.DataFrame, greek_path: Path) -> None:
    """Log column names and null percentages of the first date's Greek parquet."""
    logger.info(
        f"[SCHEMA AUDIT] {greek_path.name}: {len(greek_df.columns)} cols: "
        f"{list(greek_df.columns)[:30]}{'...' if len(greek_df.columns) > 30 else ''}"
    )
    null_pct = greek_df.isnull().mean()
    dead_cols = null_pct[null_pct > 0.5]
    if len(dead_cols) > 0:
        logger.warning(
            f"[SCHEMA AUDIT] Columns >50% null: "
            f"{dict(dead_cols.round(3))}"
        )
    live_cols = null_pct[null_pct < 0.05]
    logger.info(f"[SCHEMA AUDIT] {len(live_cols)} live cols (<5% null): {list(live_cols.index)}")


def process_one_date(args):
    """Process one tradedate: load pre-joined Greek+TQ, extract per-minute feature vectors."""
    greek_path, tq_path, trade_path, quote_path, ohlc_path, symbol, chain_only = args
    greek_path, tq_path, trade_path, quote_path, ohlc_path = (
        Path(greek_path), Path(tq_path), Path(trade_path), Path(quote_path), Path(ohlc_path)
    )

    extractor = None if chain_only else create_feature_extractor()
    chain_processor = Chain2DProcessor(n_greeks=5, n_strikes=30, n_timesteps=1)

    # Per-worker extraction error tracker (reset each date)
    extraction_errors_seen = set()

    sparse_stats = {
        "excluded_columns_hit_count": 0,
        "quote_fill_applied_minutes": 0,
        "trade_vwap_applied_minutes": 0,
        "fallback_zero_minutes": 0,
        "ohlc_fill_applied_minutes": 0,
        "ohlc_fallback_zero_minutes": 0,
        "ohlc_rows_input": 0,
        "ohlc_rows_kept": 0,
        "ohlc_rows_filtered": 0,
        "feature_zero_fallback_minutes": 0,
        "phase1_trade_coverage": 0.0,
        "ohlc_row_coverage": 0.0,
        "recovered_greek_coverage": 0.0,
    }

    try:
        # ── Load Greek ────────────────────────────────────────────────────────────────
        con = duckdb.connect()
        greek_df = con.execute(
            f"SELECT * FROM read_parquet('{str(greek_path).replace(chr(39), chr(39)*2)}')"
        ).fetchdf()
        con.close()

        if greek_df.empty:
            return {"minutes": [], "stats": sparse_stats}

        greek_excluded = [c for c in HARD_EXCLUDED_GREEK_COLS if c in greek_df.columns]
        sparse_stats["excluded_columns_hit_count"] += len(greek_excluded)
        if greek_excluded:
            greek_df = greek_df.drop(columns=greek_excluded, errors="ignore")
        rec_cov = _coverage_ratio(greek_df, RECOVERED_GREEK_COLS, nonzero=True)
        sparse_stats["recovered_greek_coverage"] = float(np.mean(list(rec_cov.values()))) if rec_cov else 0.0

        # One-time schema audit: only fires for the very first date processed by this worker
        if not hasattr(process_one_date, '_schema_audited'):
            _audit_greek_schema(greek_df, greek_path)
            process_one_date._schema_audited = True

        # Parse timestamps
        ts_col = None
        for col in ['timestamp', 'underlying_timestamp', 'trade_date']:
            if col in greek_df.columns:
                ts_col = col
                break
        if ts_col is None:
            logger.warning(f"{symbol}/{greek_path.name}: no timestamp column found (cols={list(greek_df.columns)[:10]})")
            return []

        greek_df[ts_col] = pd.to_datetime(greek_df[ts_col])

        # Use pre-computed _minute if available, else compute
        if '_minute' in greek_df.columns:
            greek_df['_minute'] = pd.to_datetime(greek_df['_minute'])
        else:
            greek_df['_minute'] = greek_df[ts_col].dt.floor('min')

        # ── Load TQ: prefer split *_trade + *_quote; else legacy combined *_tq ───────────
        tq_by_minute = {}
        ohlc_by_minute = {}
        if tq_path.exists() or trade_path.exists() or quote_path.exists():
            try:
                con = duckdb.connect()
                tq_df = pd.DataFrame()
                if trade_path.exists() or quote_path.exists():
                    if tq_path.exists():
                        logger.debug(
                            "%s: using split trade/quote parquet (ignoring legacy %s)",
                            greek_path.stem,
                            tq_path.name,
                        )
                    frames = []
                    if trade_path.exists():
                        tr_df = con.execute(
                            f"SELECT * FROM read_parquet('{str(trade_path).replace(chr(39), chr(39)*2)}')"
                        ).fetchdf()
                        if len(tr_df) > 0:
                            if "trade_timestamp" not in tr_df.columns and "timestamp" in tr_df.columns:
                                tr_df["trade_timestamp"] = tr_df["timestamp"]
                            tr_df["row_type"] = "trade"
                            frames.append(tr_df)
                    if quote_path.exists():
                        q_df = con.execute(
                            f"SELECT * FROM read_parquet('{str(quote_path).replace(chr(39), chr(39)*2)}')"
                        ).fetchdf()
                        if len(q_df) > 0:
                            if "quote_timestamp" not in q_df.columns and "timestamp" in q_df.columns:
                                q_df["quote_timestamp"] = q_df["timestamp"]
                            q_df["row_type"] = "quote"
                            frames.append(q_df)
                    if frames:
                        tq_df = pd.concat(frames, ignore_index=True, sort=False)
                elif tq_path.exists():
                    tq_df = con.execute(
                        f"SELECT * FROM read_parquet('{str(tq_path).replace(chr(39), chr(39)*2)}')"
                    ).fetchdf()
                con.close()

                if len(tq_df) > 0:
                    for ts_c in ['trade_timestamp', 'quote_timestamp']:
                        if ts_c in tq_df.columns:
                            tq_df[ts_c] = pd.to_datetime(tq_df[ts_c], errors='coerce')

                    if '_minute' in tq_df.columns:
                        tq_df['_minute'] = pd.to_datetime(tq_df['_minute'])
                    else:
                        if 'trade_timestamp' in tq_df.columns and tq_df['trade_timestamp'].notna().any():
                            tc = 'trade_timestamp'
                        elif 'quote_timestamp' in tq_df.columns and tq_df['quote_timestamp'].notna().any():
                            tc = 'quote_timestamp'
                        elif 'timestamp' in tq_df.columns:
                            tc = 'timestamp'
                        else:
                            tc = None
                        if tc is not None:
                            tq_df['_minute'] = pd.to_datetime(tq_df[tc], errors='coerce').dt.floor('min')
                        else:
                            tq_df['_minute'] = pd.NaT
                    tq_excluded = [c for c in HARD_EXCLUDED_TQ_COLS if c in tq_df.columns]
                    sparse_stats["excluded_columns_hit_count"] += len(tq_excluded)
                    if tq_excluded:
                        tq_df = tq_df.drop(columns=tq_excluded, errors="ignore")

                    greek_minutes = list(greek_df["_minute"].dropna().unique())
                    tq_by_minute, fill_stats = _build_sparse_tq_lookup(tq_df, greek_minutes, max_fill_minutes=5)
                    sparse_stats.update(fill_stats)
                    tq_cov = _coverage_ratio(tq_df, ["price", "size", "bid", "ask", "bid_size", "ask_size"], nonzero=True)
                    sparse_stats["phase1_trade_coverage"] = float(np.mean(list(tq_cov.values()))) if tq_cov else 0.0
            except Exception as tq_err:
                logger.warning(f"{symbol}/{greek_path.name}: TQ load failed ({tq_err}) — continuing without TQ data")

        if ohlc_path.exists():
            try:
                con = duckdb.connect()
                ohlc_df = con.execute(
                    f"SELECT * FROM read_parquet('{str(ohlc_path).replace(chr(39), chr(39)*2)}')"
                ).fetchdf()
                con.close()
                if len(ohlc_df) > 0:
                    ts_col = "timestamp" if "timestamp" in ohlc_df.columns else None
                    if ts_col:
                        ohlc_df[ts_col] = pd.to_datetime(ohlc_df[ts_col], errors="coerce")
                        ohlc_df["_minute"] = ohlc_df[ts_col].dt.floor("min")
                        ohlc_df, ohlc_row_stats = _filter_sparse_ohlc_rows(ohlc_df)
                        sparse_stats.update(ohlc_row_stats)
                        greek_minutes = list(greek_df["_minute"].dropna().unique())
                        ohlc_by_minute, ohlc_fill_stats = _build_sparse_ohlc_lookup(
                            ohlc_df, greek_minutes, max_fill_minutes=3
                        )
                        sparse_stats.update(ohlc_fill_stats)
                        ohlc_cov = _coverage_ratio(ohlc_df, ["open", "high", "low", "close", "volume", "count"], nonzero=True)
                        sparse_stats["ohlc_row_coverage"] = float(np.mean(list(ohlc_cov.values()))) if ohlc_cov else 0.0
            except Exception as ohlc_err:
                logger.warning(f"{symbol}/{greek_path.name}: OHLC load failed ({ohlc_err}) — continuing without OHLC data")

        # ── Extract features per minute ─────────────────────────────────────────────────
        minutes = []
        n_extracted_ok = 0
        n_extracted_zero = 0

        for minute_ts, greek_group in greek_df.groupby('_minute'):
            if len(greek_group) < 3:
                continue

            # Direct minute lookup (no ±1 min needed — pre-aligned)
            tq_group = tq_by_minute.get(minute_ts)
            ohlc_group = ohlc_by_minute.get(minute_ts)

            if extractor is None:
                features = np.zeros(FEAT_DIM, dtype=np.float32)
                n_extracted_zero += 1
            else:
                try:
                    result = extractor.extract(greek_df=greek_group, trade_df=tq_group, ohlc_df=ohlc_group)
                    features = result.features
                    n_extracted_ok += 1
                except Exception as ex:
                    err_key = type(ex).__name__ + str(ex)[:80]
                    if err_key not in extraction_errors_seen:
                        extraction_errors_seen.add(err_key)
                        if len(extraction_errors_seen) <= _EXTRACTION_ERROR_SAMPLE_LIMIT:
                            logger.warning(
                                f"{symbol}/{greek_path.name} @ {minute_ts}: "
                                f"MasterFeatureExtractor failed — {type(ex).__name__}: {ex}. "
                                f"Falling back to zero vector. Check column compatibility."
                            )
                    features = np.zeros(FEAT_DIM, dtype=np.float32)
                    n_extracted_zero += 1

            try:
                chain_slice = chain_processor.snapshot_to_slice(greek_group)
                chain_flat = chain_slice.flatten()
            except Exception:
                chain_flat = np.zeros(150, dtype=np.float32)

            underlying_price = (
                greek_group['underlying_price'].mean()
                if 'underlying_price' in greek_group.columns else 0.0
            )

            minutes.append({
                'timestamp': minute_ts,
                'features': features.tolist(),
                'chain_2d': chain_flat.tolist(),
                'underlying_price': float(underlying_price),
                'contract_count': int(len(greek_group)),
                'trade_count': int(len(tq_group)) if tq_group is not None else 0,
            })

        total_minutes = n_extracted_ok + n_extracted_zero
        if not chain_only and total_minutes > 0:
            zero_pct = 100.0 * n_extracted_zero / total_minutes
            if zero_pct > 20.0:
                logger.error(
                    f"{symbol}/{greek_path.name}: HIGH FAILURE RATE — "
                    f"{n_extracted_zero}/{total_minutes} ({zero_pct:.1f}%) "
                    f"minutes fell back to zero-vector features"
                )
            elif zero_pct > 5.0:
                logger.warning(
                    f"{symbol}/{greek_path.name}: {n_extracted_zero}/{total_minutes} "
                    f"({zero_pct:.1f}%) minutes fell back to zero-vector features"
                )

        sparse_stats["feature_zero_fallback_minutes"] = n_extracted_zero
        return {"minutes": minutes, "stats": sparse_stats}

    except Exception as e:
        logger.error(f"{symbol}/{greek_path.name}: {e}")
        return {"minutes": [], "stats": sparse_stats}


def process_symbol(symbol, tier1_root, output_root, n_workers, chain_only=False):
    t0 = time.time()
    sym_dir = tier1_root / symbol
    if not sym_dir.exists():
        logger.error(f"{symbol}: tier1_v4 dir not found: {sym_dir}")
        return None

    output_file = output_root / f"{symbol}_minutes.parquet"
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Resume support ────────────────────────────────────────────────────────────────
    progress_file = output_root / f"{symbol}_progress.json"
    completed_dates = set()
    cached_minutes = []
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        completed_dates = set(progress.get('completed_dates', []))
        partial_file = output_root / f"{symbol}_partial.parquet"
        if partial_file.exists():
            try:
                con = duckdb.connect()
                cached_minutes = con.execute(
                    f"SELECT * FROM read_parquet('{str(partial_file).replace(chr(39), chr(39)*2)}')"
                ).fetchdf().to_dict('records')
                con.close()
                logger.info(f"{symbol}: Resuming from {len(completed_dates)} dates, "
                           f"{len(cached_minutes)} cached minutes")
            except Exception:
                cached_minutes = []
                completed_dates = set()

    # ── Discover tradedate files ────────────────────────────────────────────────────────
    greek_files = sorted(sym_dir.glob("*_greek.parquet"))
    all_dates = [f.name.replace("_greek.parquet", "") for f in greek_files]

    work_items = []
    for d, gf in zip(all_dates, greek_files):
        if d in completed_dates:
            continue
        tq_path = sym_dir / f"{d}_tq.parquet"
        trade_path = sym_dir / f"{d}_trade.parquet"
        quote_path = sym_dir / f"{d}_quote.parquet"
        ohlc_path = sym_dir / f"{d}_ohlc.parquet"
        work_items.append((
            str(gf), str(tq_path), str(trade_path), str(quote_path), str(ohlc_path), symbol, chain_only
        ))

    logger.info(f"{symbol}: {len(all_dates)} total dates, "
               f"{len(completed_dates)} done, {len(work_items)} to process, "
               f"{n_workers} workers")

    if not work_items and not cached_minutes:
        logger.warning(f"{symbol}: Nothing to process")
        return None

    all_minutes = cached_minutes
    done = 0

    aggregate_sparse_stats = {
        "excluded_columns_hit_count": 0,
        "quote_fill_applied_minutes": 0,
        "trade_vwap_applied_minutes": 0,
        "fallback_zero_minutes": 0,
        "ohlc_fill_applied_minutes": 0,
        "ohlc_fallback_zero_minutes": 0,
        "ohlc_rows_input": 0,
        "ohlc_rows_kept": 0,
        "ohlc_rows_filtered": 0,
        "feature_zero_fallback_minutes": 0,
        "phase1_trade_coverage": 0.0,
        "ohlc_row_coverage": 0.0,
        "recovered_greek_coverage": 0.0,
    }
    _coverage_samples = 0

    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_one_date, work_items):
            result_minutes = result.get("minutes", [])
            result_stats = result.get("stats", {})
            all_minutes.extend(result_minutes)
            for k in aggregate_sparse_stats:
                v = result_stats.get(k, 0)
                if k in ("phase1_trade_coverage", "ohlc_row_coverage", "recovered_greek_coverage"):
                    aggregate_sparse_stats[k] += float(v)
                else:
                    aggregate_sparse_stats[k] += int(v)
            _coverage_samples += 1
            done += 1

            if done % 20 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(work_items) - done) / rate if rate > 0 else 0
                logger.info(f"{symbol}: {done}/{len(work_items)} dates, "
                           f"{len(all_minutes)} minutes, "
                           f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

                # Checkpoint
                try:
                    partial_df = pd.DataFrame(all_minutes)
                    partial_file = output_root / f"{symbol}_partial.parquet"
                    con = duckdb.connect()
                    con.register('partial_df', partial_df)
                    con.execute(
                        f"COPY partial_df TO '{str(partial_file).replace(chr(39), chr(39)*2)}' "
                        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
                    )
                    con.close()
                except Exception as e:
                    logger.warning(f"{symbol}: Checkpoint failed: {e}")

                progress_file.write_text(json.dumps({
                    'completed_dates': list(completed_dates) + all_dates[:done],
                    'total_dates': len(all_dates),
                    'total_minutes': len(all_minutes),
                    'last_update': datetime.now().isoformat(),
                }))

    logger.info(f"{symbol}: All {done} dates processed")
    if _coverage_samples > 0:
        aggregate_sparse_stats["phase1_trade_coverage"] /= _coverage_samples
        aggregate_sparse_stats["ohlc_row_coverage"] /= _coverage_samples
        aggregate_sparse_stats["recovered_greek_coverage"] /= _coverage_samples
    logger.info(
        "%s: sparse-data counters — excluded_cols=%d quote_ffill=%d trade_vwap=%d fallback_zero=%d "
        "ohlc_fill=%d ohlc_fallback_zero=%d ohlc_rows_in=%d ohlc_rows_kept=%d ohlc_rows_filtered=%d "
        "feature_zero_fallback=%d tq_cov=%.3f ohlc_cov=%.3f recovered_greek_cov=%.3f",
        symbol,
        aggregate_sparse_stats["excluded_columns_hit_count"],
        aggregate_sparse_stats["quote_fill_applied_minutes"],
        aggregate_sparse_stats["trade_vwap_applied_minutes"],
        aggregate_sparse_stats["fallback_zero_minutes"],
        aggregate_sparse_stats["ohlc_fill_applied_minutes"],
        aggregate_sparse_stats["ohlc_fallback_zero_minutes"],
        aggregate_sparse_stats["ohlc_rows_input"],
        aggregate_sparse_stats["ohlc_rows_kept"],
        aggregate_sparse_stats["ohlc_rows_filtered"],
        aggregate_sparse_stats["feature_zero_fallback_minutes"],
        aggregate_sparse_stats["phase1_trade_coverage"],
        aggregate_sparse_stats["ohlc_row_coverage"],
        aggregate_sparse_stats["recovered_greek_coverage"],
    )
    gc.collect()

    if not all_minutes:
        logger.warning(f"{symbol}: No minute bars")
        return None

    # ── Write final output ────────────────────────────────────────────────────────────────
    logger.info(f"{symbol}: Writing {len(all_minutes)} minute bars...")
    minutes_df = pd.DataFrame(all_minutes).sort_values('timestamp')
    con = duckdb.connect()
    con.register('minutes_df', minutes_df)
    con.execute(
        f"COPY minutes_df TO '{str(output_file).replace(chr(39), chr(39)*2)}' "
        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()

    # Cleanup
    for f in [progress_file, output_root / f"{symbol}_partial.parquet"]:
        if f.exists():
            f.unlink()

    elapsed = time.time() - t0
    size_mb = output_file.stat().st_size / 1e6

    sample = np.array([np.array(f) for f in minutes_df['features'].iloc[:1000]])
    useful = int((np.std(sample, axis=0) >= 1e-8).sum())
    has_tq = (minutes_df['trade_count'] > 0).sum()

    # Enhanced validation using DataQualityChecker from hybrid55_preprocessing
    try:
        all_features = np.array([np.array(f) for f in minutes_df['features']])
        quality_checker = DataQualityChecker(
            missing_threshold=0.05,
            zero_threshold=0.95,
            constant_threshold=1e-10
        )
        quality_report = quality_checker.check_features(all_features)

        logger.info(f"{symbol}: Quality - overall={quality_report.overall_quality:.2%}, "
                   f"missing={quality_report.missing_pct:.2%}, zero={quality_report.zero_pct:.2%}, "
                   f"inf={quality_report.inf_pct:.2%}")

        if quality_report.errors:
            logger.error(f"{symbol}: Quality errors: {'; '.join(quality_report.errors[:3])}")
        if quality_report.warnings and len(quality_report.warnings) > 5:
            logger.warning(f"{symbol}: {len(quality_report.warnings)} quality warnings detected")

        # Save quality report
        quality_report_path = output_root / f"{symbol}_quality_report.json"
        from hybrid55_preprocessing.quality_checks import save_quality_report
        save_quality_report(quality_report, str(quality_report_path))
        logger.info(f"{symbol}: Quality report saved to {quality_report_path}")
    except Exception as e:
        logger.warning(f"{symbol}: Quality check failed: {e}")

    if useful == 0:
        logger.error(
            f"{symbol}: ALL {FEAT_DIM} features are zero-variance! "
            f"MasterFeatureExtractor is almost certainly incompatible with the tier1_hybrid55 schema. "
            f"Re-run with --workers 1 and check WARNING lines above for the exact error."
        )
    elif useful < FEAT_DIM // 2:
        logger.warning(
            f"{symbol}: Only {useful}/{FEAT_DIM} features have variance — "
            f"feature extraction is partially failing. Check schema compatibility."
        )

    result = {
        'symbol': symbol, 'total_minutes': len(minutes_df),
        'useful_features': useful, 'feat_dim': FEAT_DIM,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'tq_coverage': f"{has_tq}/{len(minutes_df)} ({100*has_tq/len(minutes_df):.1f}%)",
        'size_mb': round(size_mb, 1), 'elapsed_s': round(elapsed, 1),
        'output': str(output_file),
    }

    logger.info(f"{symbol}: DONE — {len(minutes_df)} minutes, "
               f"{useful}/{FEAT_DIM} features, {size_mb:.1f} MB, {elapsed:.0f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build tier2 from tier1_hybrid55 per-date parquets")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--all-symbols", action="store_true")
    parser.add_argument("--tier1-root", type=str, default=str(TIER1_ROOT))
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chain-only", action="store_true", help="Skip flat feature extraction and emit chain_2d + price-alignment data only")
    args = parser.parse_args()

    if args.all_symbols:
        symbols = ALL_SYMBOLS
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        parser.error("Specify --symbol or --all-symbols")

    tier1_root = Path(args.tier1_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"TIER2 FAST BUILD (pre-joined tier1_v4)")
    print(f"  Tier1:   {tier1_root}")
    print(f"  Output:  {output_root}")
    print(f"  Symbols: {symbols}")
    print(f"  Workers: {args.workers}")
    print(f"  Chain-only: {args.chain_only}")
    print(f"  Feature dim: {FEAT_DIM}")
    print(f"  Excluded Greek cols: {sorted(HARD_EXCLUDED_GREEK_COLS)}")
    print(f"  Excluded TQ cols: {sorted(HARD_EXCLUDED_TQ_COLS)}")
    print(f"{'=' * 70}")

    results = []
    for symbol in symbols:
        r = process_symbol(symbol, tier1_root, output_root, args.workers, chain_only=args.chain_only)
        if r:
            results.append(r)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    total = 0
    for r in results:
        print(f"  {r['symbol']:6s}: {r['total_minutes']:>7,} min, "
              f"{r['useful_features']}/{r['feat_dim']} feat, "
              f"{r['size_mb']:.1f} MB, {r['elapsed_s']:.0f}s")
        total += r['total_minutes']
    print(f"  {'TOTAL':6s}: {total:>7,} minutes across {len(results)} symbols")
    print(f"{'=' * 70}")

    (output_root / "build_summary.json").write_text(json.dumps({
        'results': results, 'total_minutes': total,
        'feat_dim': FEAT_DIM,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'timestamp': datetime.now().isoformat(),
    }, indent=2))


if __name__ == "__main__":
    main()
