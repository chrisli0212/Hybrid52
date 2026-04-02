#!/usr/bin/env python3
"""Phase 0 Step 2: Extract filtered Greek + TQ from DuckDB to tier1 parquet files.

Outputs week_key-named parquet files compatible with the existing Tier2 reprocess script
(`tier2_reprocess.py` expects filenames containing a substring like `YYYY-Www_partNNN`).

Typical usage (per-symbol DB):
    python scripts/phase0/extract_tier1.py --db-path /workspace/data/data_in_2026/db_spxw.duckdb --symbols SPXW
"""

import argparse
from pathlib import Path
import time
import datetime as _dt

import duckdb

# Legacy consolidated DBs — used only when --db-path is not supplied.
# For Hybrid52, always pass --db-path (per-symbol DB from data_in_2026/).
DB_PART1 = '/workspace/data/data_in_2026/consolidated_options.duckdb'
DB_PART2 = '/workspace/data/data_in_2026/consolidated_options_part2.duckdb'

TIER1_ROOT = Path('/workspace/data/tier1_hybrid55')

# Active contract filters
DELTA_MIN = 0.2
DELTA_MAX = 0.9
DTE_MIN = 0
DTE_MAX = 3
MIN_VEGA = 0.01

ALL_SYMBOLS = ['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT']


def _list_week_keys(con: duckdb.DuckDBPyConnection, table: str, symbol: str):
    rows = con.execute(
        f"""
        SELECT DISTINCT week_key
        FROM {table}
        WHERE symbol = ? AND week_key IS NOT NULL
        ORDER BY week_key
        """,
        [symbol],
    ).fetchall()
    return [r[0] for r in rows]


def _table_has_column(con: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    try:
        cols = con.execute(f"DESCRIBE {table}").fetchall()
    except Exception:
        return False
    return any(str(r[0]).lower() == column.lower() for r in cols)


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    rows = con.execute("SHOW TABLES").fetchall()
    names = {str(r[0]).lower() for r in rows}
    return table.lower() in names


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    try:
        cols = con.execute(f"DESCRIBE {table}").fetchall()
    except Exception:
        return []
    return [str(r[0]) for r in cols]


def _list_trade_dates(con: duckdb.DuckDBPyConnection, table: str, symbol: str):
    rows = con.execute(
        f"""
        SELECT DISTINCT trade_date
        FROM {table}
        WHERE symbol = ? AND trade_date IS NOT NULL
        ORDER BY trade_date
        """,
        [symbol],
    ).fetchall()
    return [r[0] for r in rows]


def _derived_week_part_key(trade_date: _dt.date, part: int) -> str:
    iso = trade_date.isocalendar()
    iso_year = int(iso.year)
    iso_week = int(iso.week)
    return f"{iso_year}-W{iso_week:02d}_part{part:03d}"


def extract_symbol_by_trade_date(con: duckdb.DuckDBPyConnection, symbol: str):
    """Extract Greek + TQ for one symbol, one file per trade_date with derived week_key.

    This keeps files small enough for Tier2 to load in pandas and avoids relying on
    inconsistent/missing week_key columns in per-symbol DuckDBs.
    """

    greek_out_dir = TIER1_ROOT / 'greek' / f'symbol={symbol}'
    tq_out_dir = TIER1_ROOT / 'tradequote' / f'symbol={symbol}'
    ohlc_out_dir = TIER1_ROOT / 'ohlc' / f'symbol={symbol}'
    greek_out_dir.mkdir(parents=True, exist_ok=True)
    tq_out_dir.mkdir(parents=True, exist_ok=True)
    ohlc_out_dir.mkdir(parents=True, exist_ok=True)

    symbol_sql = symbol.replace("'", "''")

    trade_dates = _list_trade_dates(con, 'options_greek', symbol)
    ohlc_table = None
    for cand in ("options_ohlc", "ohlcv"):
        if _table_exists(con, cand):
            ohlc_table = cand
            break
    ohlc_cols = _table_columns(con, ohlc_table) if ohlc_table else []

    print(f"  Trade dates (from Greek): {len(trade_dates)}")
    if not trade_dates:
        return

    # Assign a stable part number per ISO week to keep filenames compatible with tier2_reprocess
    # and avoid giant single-week parquet files.
    part_by_week = {}

    t0 = time.time()
    for i, d in enumerate(trade_dates):
        if isinstance(d, str):
            d = _dt.date.fromisoformat(d)
        week_id = f"{d.isocalendar().year}-W{int(d.isocalendar().week):02d}"
        part_by_week[week_id] = part_by_week.get(week_id, 0) + 1
        part = part_by_week[week_id]
        key = _derived_week_part_key(d, part)

        greek_path = greek_out_dir / f"option_greek_all_{key}_filtered.parquet"
        tq_path = tq_out_dir / f"option_quote_{key}.parquet"
        ohlc_path = ohlc_out_dir / f"option_ohlc_{key}.parquet"

        greek_sql = str(greek_path).replace("'", "''")
        tq_sql = str(tq_path).replace("'", "''")
        ohlc_sql = str(ohlc_path).replace("'", "''")

        # Greek
        con.execute(
            f"""
            COPY (
                SELECT
                    symbol, expiration, strike, "right", timestamp,
                    bid, ask, delta, theta, vega, rho, epsilon, "lambda", gamma,
                    vanna, charm, vomma, veta, vera, speed, zomma, color, ultima,
                    d1, d2, dual_delta, dual_gamma, implied_vol, iv_error,
                    underlying_timestamp, underlying_price, open_interest,
                    moneyness, dist_atm_pct, mid, spread, spread_pct, lambda_ratio,
                    dte_int, cp_sign, trade_date,
                    '{key}' AS week_key
                FROM options_greek
                WHERE symbol = '{symbol_sql}'
                    AND trade_date = DATE '{d.isoformat()}'
                    AND ABS(delta) BETWEEN {DELTA_MIN} AND {DELTA_MAX}
                    AND (CAST(expiration AS DATE) - trade_date) BETWEEN {DTE_MIN} AND {DTE_MAX}
                    AND ABS(vega) > {MIN_VEGA}
                    AND bid != 0 AND ask != 0
                    AND bid IS NOT NULL AND ask IS NOT NULL
            ) TO '{greek_sql}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE true)
            """
        )

        # Trade/Quote
        con.execute(
            f"""
            COPY (
                SELECT
                    symbol, expiration, strike, "right",
                    trade_timestamp, quote_timestamp, sequence,
                    ext_condition1, ext_condition2, ext_condition3, ext_condition4,
                    "condition", size, exchange, price,
                    bid_size, bid_exchange, bid, bid_condition,
                    ask_size, ask_exchange, ask, ask_condition,
                    trade_date,
                    '{key}' AS week_key
                FROM options_trade_quote
                WHERE symbol = '{symbol_sql}'
                    AND trade_date = DATE '{d.isoformat()}'
                    AND (CAST(expiration AS DATE) - trade_date) BETWEEN {DTE_MIN} AND {DTE_MAX}
                    AND (
                        (bid > 0 AND ask > 0) OR
                        (price > 0 AND size > 0)
                    )
            ) TO '{tq_sql}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE true)
            """
        )
        # NOTE — historical snapshots can have sparse trade prints while quotes remain live.
        # Keep rows with valid quotes OR valid trades to maximize TQ coverage.

        if ohlc_table:
            base_cols = ["symbol", "expiration", "strike", '"right"', "timestamp", "open", "high", "low", "close", "volume", "count", "trade_date"]
            opt_cols = ["underlying_price", "moneyness", "dist_atm_pct", "dte_int", "cp_sign"]
            selected = [c for c in base_cols if c.strip('"') in ohlc_cols]
            selected.extend([c for c in opt_cols if c in ohlc_cols])
            if "trade_date" not in [c.strip('"') for c in selected]:
                selected.append("trade_date")
            if selected:
                sel_sql = ", ".join(selected + [f"'{key}' AS week_key"])
                con.execute(
                    f"""
                    COPY (
                        SELECT {sel_sql}
                        FROM {ohlc_table}
                        WHERE symbol = '{symbol_sql}'
                            AND trade_date = DATE '{d.isoformat()}'
                    ) TO '{ohlc_sql}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE true)
                    """
                )

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{symbol}] {i+1}/{len(trade_dates)} trade_date files ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  ✓ {symbol} extracted by trade_date in {elapsed:.1f}s → {TIER1_ROOT}")


def extract_greek_for_symbol(con: duckdb.DuckDBPyConnection, symbol: str):
    """Extract Greek data for a symbol, one parquet per week_key."""
    out_dir = TIER1_ROOT / 'greek' / f'symbol={symbol}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check data availability
    stats = con.execute(
        """
        SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date), COUNT(*)
        FROM options_greek WHERE symbol=?
        """,
        [symbol],
    ).fetchone()

    if stats[3] == 0:
        print(f"  [Greek:{symbol}] No data found, skipping")
        return

    print(f"  Date range: {stats[0]} to {stats[1]} ({stats[2]} dates, {stats[3]:,} rows)")

    week_keys = _list_week_keys(con, 'options_greek', symbol)
    print(f"  Week keys: {len(week_keys)}")

    symbol_sql = symbol.replace("'", "''")

    t0 = time.time()
    for i, wk in enumerate(week_keys):
        out_path = out_dir / f"option_greek_all_{wk}_filtered.parquet"
        wk_sql = str(wk).replace("'", "''")
        out_sql = str(out_path).replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT
                    symbol, expiration, strike, "right", timestamp,
                    bid, ask, delta, theta, vega, rho, epsilon, "lambda", gamma,
                    vanna, charm, vomma, veta, vera, speed, zomma, color, ultima,
                    d1, d2, dual_delta, dual_gamma, implied_vol, iv_error,
                    underlying_timestamp, underlying_price, open_interest,
                    moneyness, dist_atm_pct, mid, spread, spread_pct, lambda_ratio,
                    dte_int, cp_sign, trade_date, week_key
                FROM options_greek
                WHERE symbol = '{symbol_sql}'
                    AND week_key = '{wk_sql}'
                    AND ABS(delta) BETWEEN {DELTA_MIN} AND {DELTA_MAX}
                    AND (CAST(expiration AS DATE) - trade_date) BETWEEN {DTE_MIN} AND {DTE_MAX}
                    AND ABS(vega) > {MIN_VEGA}
                    AND bid != 0 AND ask != 0
                    AND bid IS NOT NULL AND ask IS NOT NULL
            ) TO '{out_sql}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE true)
            """
        )

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [Greek:{symbol}] {i+1}/{len(week_keys)} week files ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  ✓ Greek:{symbol} extracted in {elapsed:.1f}s → {out_dir}")


def extract_tq_for_symbol(con: duckdb.DuckDBPyConnection, symbol: str):
    """Extract TQ data for a symbol, one parquet per week_key."""
    out_dir = TIER1_ROOT / 'tradequote' / f'symbol={symbol}'
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = con.execute(
        """
        SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date), COUNT(*)
        FROM options_trade_quote WHERE symbol=?
        """,
        [symbol],
    ).fetchone()

    if stats[3] == 0:
        print(f"  [TQ:{symbol}] No data found, skipping")
        return

    print(f"  Date range: {stats[0]} to {stats[1]} ({stats[2]} dates, {stats[3]:,} rows)")

    # Some per-symbol DBs do not have week_key on options_trade_quote.
    # In that case, derive week partitions from options_greek.week_key and use the
    # Greek week_key date windows to extract matching TQ rows.
    if _table_has_column(con, 'options_trade_quote', 'week_key'):
        week_keys = _list_week_keys(con, 'options_trade_quote', symbol)
    else:
        week_keys = _list_week_keys(con, 'options_greek', symbol)
    print(f"  Week keys: {len(week_keys)}")

    symbol_sql = symbol.replace("'", "''")

    t0 = time.time()
    for i, wk in enumerate(week_keys):
        out_path = out_dir / f"option_quote_{wk}.parquet"
        wk_sql = str(wk).replace("'", "''")
        out_sql = str(out_path).replace("'", "''")

        if _table_has_column(con, 'options_trade_quote', 'week_key'):
            where_week = f"week_key = '{wk_sql}'"
        else:
            # Use Greek date window for this week_key
            d0, d1 = con.execute(
                f"""
                SELECT MIN(trade_date), MAX(trade_date)
                FROM options_greek
                WHERE symbol = '{symbol_sql}' AND week_key = '{wk_sql}'
                """
            ).fetchone()
            if d0 is None or d1 is None:
                continue
            where_week = f"trade_date BETWEEN DATE '{d0}' AND DATE '{d1}'"

        con.execute(
            f"""
            COPY (
                SELECT
                    symbol, expiration, strike, "right",
                    trade_timestamp, quote_timestamp, sequence,
                    ext_condition1, ext_condition2, ext_condition3, ext_condition4,
                    "condition", size, exchange, price,
                    bid_size, bid_exchange, bid, bid_condition,
                    ask_size, ask_exchange, ask, ask_condition,
                    trade_date,
                    '{wk_sql}' AS week_key
                FROM options_trade_quote
                WHERE symbol = '{symbol_sql}'
                    AND {where_week}
                    AND (
                        (bid > 0 AND ask > 0) OR
                        (price > 0 AND size > 0)
                    )
            ) TO '{out_sql}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE true)
            """
        )

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [TQ:{symbol}] {i+1}/{len(week_keys)} week files ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  ✓ TQ:{symbol} extracted in {elapsed:.1f}s → {out_dir}")


def main():
    global TIER1_ROOT
    parser = argparse.ArgumentParser(description='Extract tier1 from DuckDB')
    parser.add_argument('--symbols', nargs='+', default=ALL_SYMBOLS)
    parser.add_argument('--db-path', type=str, default=None,
                        help='Optional path to a per-symbol DuckDB (contains options_greek + options_trade_quote)')
    parser.add_argument('--out-root', type=str, default=str(TIER1_ROOT),
                        help='Output root directory for tier1 parquet artifacts')
    args = parser.parse_args()
    TIER1_ROOT = Path(args.out_root)

    print("=" * 70)
    print("PHASE 0 STEP 2: Extract Tier1 from DuckDB")
    print(f"Symbols: {args.symbols}")
    print(f"DB: {args.db_path if args.db_path else f'{DB_PART1} (+ {DB_PART2} for TQ)'}")
    print(f"Output: {TIER1_ROOT}")
    print(f"Filters: delta=[{DELTA_MIN},{DELTA_MAX}], DTE=[{DTE_MIN},{DTE_MAX}], vega>{MIN_VEGA}")
    print("=" * 70)

    t_total = time.time()

    if args.db_path:
        con = duckdb.connect(args.db_path, read_only=True)
        con.execute("PRAGMA memory_limit='6GB'")
        con.execute("PRAGMA threads=4")
        for symbol in args.symbols:
            symbol = symbol.upper()
            print(f"\n[{symbol}] Extracting Greek + TQ from {args.db_path}...")
            extract_symbol_by_trade_date(con, symbol)
        con.close()
    else:
        # Legacy mode: Greek from Part1, TQ from Part1+Part2 (not used for 2026 per-symbol DBs)
        con_g = duckdb.connect(DB_PART1, read_only=True)
        con_g.execute("PRAGMA memory_limit='6GB'")
        con_g.execute("PRAGMA threads=4")
        con_tq1 = duckdb.connect(DB_PART1, read_only=True)
        con_tq1.execute("PRAGMA memory_limit='6GB'")
        con_tq1.execute("PRAGMA threads=4")
        con_tq2 = duckdb.connect(DB_PART2, read_only=True)
        con_tq2.execute("PRAGMA memory_limit='6GB'")
        con_tq2.execute("PRAGMA threads=4")

        for symbol in args.symbols:
            symbol = symbol.upper()
            print(f"\n[Greek:{symbol}] Extracting from Part1...")
            extract_greek_for_symbol(con_g, symbol)
            print(f"\n[TQ:{symbol}] Extracting from Part1...")
            extract_tq_for_symbol(con_tq1, symbol)
            print(f"\n[TQ:{symbol}] Extracting from Part2...")
            extract_tq_for_symbol(con_tq2, symbol)

        con_g.close()
        con_tq1.close()
        con_tq2.close()

    print(f"\n{'='*70}")
    print(f"PHASE 0 STEP 2 COMPLETE in {time.time()-t_total:.1f}s")
    print(f"Output: {TIER1_ROOT}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
