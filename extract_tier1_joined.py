#!/usr/bin/env python3
"""
Phase 0 Step 2b: Extract PRE-JOINED Greek + TQ from DuckDB to per-trade_date parquets.

Instead of separate Greek/TQ directories with week_key matching, this outputs:
  /workspace/data/tier1_v4/{SYMBOL}/{YYYY-MM-DD}_greek.parquet
  /workspace/data/tier1_v4/{SYMBOL}/{YYYY-MM-DD}_tq.parquet

Both have a pre-computed '_minute' column (timestamp floored to minute),
eliminating the expensive pandas merge + ±1min tolerance lookup in tier2.

Tier1 rebuild takes ~10 min total. Tier2 with pre-joined data: ~2-3 hrs (vs 8-12 hrs).

Usage:
  python3.13 extract_tier1_joined.py --db-path /workspace/data/data_in_2026/db/spxw.duckdb --symbols SPXW
  python3.13 extract_tier1_joined.py --all-symbols
"""

import argparse
import time
import datetime as dt
from pathlib import Path
import duckdb

# ── Defaults ──────────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path("/workspace/data/tier1_v4_recreat")

# Filters (same as extract_tier1.py)
DELTA_MIN = 0.2
DELTA_MAX = 0.9
DTE_MIN = 0
DTE_MAX = 3
MIN_VEGA = 0.01

ALL_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]

DB_MAP = {
    "SPXW": "/workspace/data/data_in_2026/db_spxw.duckdb",
    "SPY":  "/workspace/data/data_in_2026/db_spy.duckdb",
    "QQQ":  "/workspace/data/data_in_2026/db_qqq.duckdb",
    "IWM":  "/workspace/data/data_in_2026/db_small.duckdb",
    "TLT":  "/workspace/data/data_in_2026/db_small.duckdb",
}


def list_trade_dates(con, table, symbol):
    rows = con.execute(
        f"SELECT DISTINCT trade_date FROM {table} "
        f"WHERE symbol = ? AND trade_date IS NOT NULL ORDER BY trade_date",
        [symbol],
    ).fetchall()
    return [r[0] for r in rows]


def extract_symbol(con, symbol, out_root, overwrite=False):
    sym_dir = out_root / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    trade_dates = list_trade_dates(con, "options_greek", symbol)
    print(f"  {symbol}: {len(trade_dates)} trade dates from Greek table")
    if not trade_dates:
        return

    sym_sql = symbol.replace("'", "''")
    t0 = time.time()
    skipped = 0

    for i, d in enumerate(trade_dates):
        if isinstance(d, str):
            d = dt.date.fromisoformat(d)
        d_iso = d.isoformat()

        greek_path = sym_dir / f"{d_iso}_greek.parquet"
        tq_path    = sym_dir / f"{d_iso}_tq.parquet"

        # Skip if both exist and not overwriting
        if not overwrite and greek_path.exists() and tq_path.exists():
            skipped += 1
            continue

        gp = str(greek_path).replace("'", "''")
        tp = str(tq_path).replace("'", "''")

        # ── Greek: filtered + _minute column ──────────────────────────────
        con.execute(f"""
            COPY (
                SELECT *,
                       time_bucket(INTERVAL '1 minute', timestamp) AS _minute
                FROM options_greek
                WHERE symbol = '{sym_sql}'
                  AND trade_date = DATE '{d_iso}'
                  AND ABS(delta) BETWEEN {DELTA_MIN} AND {DELTA_MAX}
                  AND CAST(expiration AS DATE) - trade_date BETWEEN {DTE_MIN} AND {DTE_MAX}
                  AND ABS(vega) > {MIN_VEGA}
                  AND bid != 0 AND ask != 0
                  AND bid IS NOT NULL AND ask IS NOT NULL
                ORDER BY timestamp, strike
            ) TO '{gp}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true)
        """)

        # ── TQ: filtered + _minute column ─────────────────────────────────
        con.execute(f"""
            COPY (
                SELECT *,
                       time_bucket(INTERVAL '1 minute', trade_timestamp) AS _minute
                FROM options_trade_quote
                WHERE symbol = '{sym_sql}'
                  AND trade_date = DATE '{d_iso}'
                  AND CAST(expiration AS DATE) - trade_date BETWEEN {DTE_MIN} AND {DTE_MAX}
                  AND price > 0 AND size > 0
                  AND bid > 0 AND ask > 0
                ORDER BY trade_timestamp, strike
            ) TO '{tp}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true)
        """)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {symbol}: {i+1}/{len(trade_dates)} dates, "
                  f"{skipped} skipped, {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    print(f"  {symbol}: DONE {len(trade_dates)} dates "
          f"({skipped} skipped) in {elapsed:.1f}s → {sym_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract pre-joined tier1 Greek+TQ per trade_date"
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--all-symbols", action="store_true")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Single DB path (use with --symbols)")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--memory-limit", type=str, default="10GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.all_symbols:
        symbols = ALL_SYMBOLS
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        parser.error("Specify --symbols or --all-symbols")

    print("=" * 70)
    print("EXTRACT TIER1 JOINED (per-trade_date Greek + TQ)")
    print(f"  Symbols:  {symbols}")
    print(f"  Output:   {out_root}")
    print(f"  Filters:  delta={DELTA_MIN}-{DELTA_MAX}, DTE={DTE_MIN}-{DTE_MAX}, vega>{MIN_VEGA}")
    print(f"  Overwrite: {args.overwrite}")
    print("=" * 70)

    t_total = time.time()

    for symbol in symbols:
        db_path = args.db_path or DB_MAP.get(symbol)
        if db_path is None:
            print(f"  {symbol}: SKIP — no DB path configured")
            continue
        if not Path(db_path).exists():
            print(f"  {symbol}: SKIP — DB not found at {db_path}")
            continue

        print(f"\n  {symbol}: connecting to {db_path}")
        con = duckdb.connect(db_path, read_only=True)
        con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
        con.execute(f"PRAGMA threads={args.threads}")

        extract_symbol(con, symbol, out_root, overwrite=args.overwrite)
        con.close()

    print(f"\n{'=' * 70}")
    print(f"ALL DONE in {time.time() - t_total:.1f}s")
    print(f"Output: {out_root}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
