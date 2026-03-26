#!/usr/bin/env python3
"""
Build tier2 minute-bar features (286-dim historical mode) from PRE-JOINED tier1_v4 data.

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
ROOT = Path(__file__).resolve().parent.parent.parent  # → /workspace/ Hybrid52_New training
sys.path.insert(0, str(ROOT))

from hybrid52_preprocessing.chain_2d import Chain2DProcessor
from hybrid52_preprocessing.feature_config_v2 import TOTAL_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TIER1_ROOT = Path("/workspace/data/tier1_hybrid52")
OUTPUT_ROOT = Path("/workspace/data/tier2_minutes_hybrid52")
FEAT_DIM = TOTAL_FEATURES  # 286 (historical mode with expanded CSV-derived aux)
ALL_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]

# Global counter for extraction errors (multiprocessing-safe via return value inspection)
_EXTRACTION_ERROR_SAMPLE_LIMIT = 5  # log first N unique error messages per worker


def create_feature_extractor():
    from hybrid52_preprocessing.master_extractor_v2 import MasterFeatureExtractor
    return MasterFeatureExtractor(
        include_chain_2d=False, include_phase1=False, normalize=False
    )


def _audit_greek_schema(greek_df: pd.DataFrame, greek_path: Path) -> None:
    """Log column names of the first date's Greek parquet to aid schema debugging."""
    logger.info(
        f"[SCHEMA AUDIT] {greek_path.name}: {len(greek_df.columns)} cols: "
        f"{list(greek_df.columns)[:30]}{'...' if len(greek_df.columns) > 30 else ''}"
    )


def process_one_date(args):
    """Process one tradedate: load pre-joined Greek+TQ, extract per-minute feature vectors."""
    greek_path, tq_path, symbol, chain_only = args
    greek_path, tq_path = Path(greek_path), Path(tq_path)

    extractor = None if chain_only else create_feature_extractor()
    chain_processor = Chain2DProcessor(n_greeks=5, n_strikes=30, n_timesteps=1)

    # Per-worker extraction error tracker (reset each date)
    extraction_errors_seen = set()

    try:
        # ── Load Greek ────────────────────────────────────────────────────────────────
con = duckdb.connect()
        greek_df = con.execute(
            f"SELECT * FROM read_parquet('{str(greek_path).replace(chr(39), chr(39)*2)}')"
        ).fetchdf()
        con.close()

        if greek_df.empty:
            return []

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

        # ── Load TQ (pre-grouped by minute) ───────────────────────────────────────────────
con = duckdb.connect()
        tq_by_minute = {}
        if tq_path.exists():
            try:
                con = duckdb.connect()
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
                        tc = 'trade_timestamp' if 'trade_timestamp' in tq_df.columns else 'quote_timestamp'
                        tq_df['_minute'] = tq_df[tc].dt.floor('min')

                    tq_by_minute = {m: g for m, g in tq_df.groupby('_minute')}
            except Exception as tq_err:
                logger.warning(f"{symbol}/{greek_path.name}: TQ load failed ({tq_err}) — continuing without TQ data")

        # ── Extract features per minute ─────────────────────────────────────────────────
minutes = []
        n_extracted_ok = 0
        n_extracted_zero = 0

        for minute_ts, greek_group in greek_df.groupby('_minute'):
            if len(greek_group) < 3:
                continue

            # Direct minute lookup (no ±1 min needed — pre-aligned)
            tq_group = tq_by_minute.get(minute_ts)

            if extractor is None:
                features = np.zeros(FEAT_DIM, dtype=np.float32)
                n_extracted_zero += 1
            else:
                try:
                    result = extractor.extract(greek_df=greek_group, trade_df=tq_group)
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

        if n_extracted_zero > 0 and not chain_only:
            zero_pct = 100.0 * n_extracted_zero / max(n_extracted_ok + n_extracted_zero, 1)
            if zero_pct > 10.0:  # only warn if >10% are zeros
                logger.warning(
                    f"{symbol}/{greek_path.name}: {n_extracted_zero}/{n_extracted_ok + n_extracted_zero} "
                    f"({zero_pct:.1f}%) minutes fell back to zero-vector features"
                )

        return minutes

    except Exception as e:
        logger.error(f"{symbol}/{greek_path.name}: {e}")
        return []


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
        work_items.append((str(gf), str(tq_path), symbol, chain_only))

    logger.info(f"{symbol}: {len(all_dates)} total dates, "
               f"{len(completed_dates)} done, {len(work_items)} to process, "
               f"{n_workers} workers")

    if not work_items and not cached_minutes:
        logger.warning(f"{symbol}: Nothing to process")
        return None

    all_minutes = cached_minutes
    done = 0

    with mp.Pool(processes=n_workers) as pool:
        for result_minutes in pool.imap_unordered(process_one_date, work_items):
            all_minutes.extend(result_minutes)
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

    if useful == 0:
        logger.error(
            f"{symbol}: ALL {FEAT_DIM} features are zero-variance! "
            f"MasterFeatureExtractor is almost certainly incompatible with the tier1_hybrid52 schema. "
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
        'tq_coverage': f"{has_tq}/{len(minutes_df)} ({100*has_tq/len(minutes_df):.1f}%)",
        'size_mb': round(size_mb, 1), 'elapsed_s': round(elapsed, 1),
        'output': str(output_file),
    }

    logger.info(f"{symbol}: DONE — {len(minutes_df)} minutes, "
               f"{useful}/{FEAT_DIM} features, {size_mb:.1f} MB, {elapsed:.0f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build tier2 from pre-joined tier1_v4")
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
        'feat_dim': FEAT_DIM, 'timestamp': datetime.now().isoformat(),
    }, indent=2))


if __name__ == "__main__":
    main()
