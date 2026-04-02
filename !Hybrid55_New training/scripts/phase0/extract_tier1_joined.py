#!/usr/bin/env python3
"""
Phase 0 Step 2b: Extract per-trade_date Tier1 artifacts from theta_data_3year.

Outputs:
  {output_root}/{SYMBOL}/{YYYY-MM-DD}_greek.parquet
  {output_root}/{SYMBOL}/{YYYY-MM-DD}_trade.parquet
  {output_root}/{SYMBOL}/{YYYY-MM-DD}_quote.parquet
  {output_root}/{SYMBOL}/{YYYY-MM-DD}_ohlc.parquet

Optional backward-compatible artifact:
  {output_root}/{SYMBOL}/{YYYY-MM-DD}_tq.parquet
"""

import argparse
import time
import datetime as dt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import duckdb

# ── Defaults ──────────────────────────────────────────────────────────────────
SOURCE_ROOT = Path("/workspace/data/theta_data_3year")
OUTPUT_ROOT = Path("/workspace/data/tier1_hybrid55")

# Filters
DELTA_MIN = 0.2
DELTA_MAX = 0.9
DTE_MIN = 0
DTE_MAX_DEFAULT = 5
DTE_MAX_VIXW = 30
MIN_VEGA = 0.01

ALL_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT", "VIXW"]


def list_trade_dates(con, table, symbol):
    rows = con.execute(
        f"SELECT DISTINCT CAST(timestamp AS DATE) AS d FROM {table} "
        f"WHERE symbol = ? AND timestamp IS NOT NULL ORDER BY d",
        [symbol],
    ).fetchall()
    return [r[0] for r in rows]


def dte_bounds_for_symbol(symbol: str) -> tuple[int, int]:
    if symbol.upper() == "VIXW":
        return DTE_MIN, DTE_MAX_VIXW
    return DTE_MIN, DTE_MAX_DEFAULT


def extract_symbol(
    symbol: str,
    source_root: Path,
    out_root: Path,
    memory_limit: str = "10GB",
    threads: int = 8,
    overwrite: bool = False,
    write_combined_tq: bool = False,
):
    sym_src = source_root / symbol
    gdb = sym_src / "greeks_all.duckdb"
    tdb = sym_src / "trade_quote.duckdb"
    odb = sym_src / "ohlcv.duckdb"
    if not (gdb.exists() and tdb.exists() and odb.exists()):
        print(f"  {symbol}: SKIP — missing one of greeks_all/trade_quote/ohlcv duckdb")
        return

    gcon = duckdb.connect(str(gdb), read_only=True)
    tcon = duckdb.connect(str(tdb), read_only=True)
    ocon = duckdb.connect(str(odb), read_only=True)
    for con in (gcon, tcon, ocon):
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        con.execute(f"PRAGMA threads={threads}")

    sym_dir = out_root / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    trade_dates = list_trade_dates(gcon, "greeks_all", symbol)
    print(f"  {symbol}: {len(trade_dates)} trade dates from Greek table")
    if not trade_dates:
        gcon.close(); tcon.close(); ocon.close()
        return

    sym_sql = symbol.replace("'", "''")
    dte_min, dte_max = dte_bounds_for_symbol(symbol)
    t0 = time.time()
    skipped = 0
    print(f"  {symbol}: using DTE range {dte_min}-{dte_max}")

    for i, d in enumerate(trade_dates):
        if isinstance(d, str):
            d = dt.date.fromisoformat(d)
        d_iso = d.isoformat()

        greek_path = sym_dir / f"{d_iso}_greek.parquet"
        trade_path = sym_dir / f"{d_iso}_trade.parquet"
        quote_path = sym_dir / f"{d_iso}_quote.parquet"
        ohlc_path = sym_dir / f"{d_iso}_ohlc.parquet"
        tq_path = sym_dir / f"{d_iso}_tq.parquet"

        # Skip if all required files already exist.
        needed = [greek_path, trade_path, quote_path, ohlc_path]
        if write_combined_tq:
            needed.append(tq_path)
        if not overwrite and all(p.exists() for p in needed):
            skipped += 1
            continue

        gp = str(greek_path).replace("'", "''")
        trp = str(trade_path).replace("'", "''")
        qp = str(quote_path).replace("'", "''")
        op = str(ohlc_path).replace("'", "''")
        tp = str(tq_path).replace("'", "''")

        # ── Greek: filtered + _minute column ──────────────────────────────
        gcon.execute(f"""
            COPY (
                SELECT *,
                       CAST(timestamp AS DATE) AS trade_date,
                       time_bucket(INTERVAL '1 minute', CAST(timestamp AS TIMESTAMP)) AS _minute
                FROM greeks_all
                WHERE symbol = '{sym_sql}'
                  AND CAST(timestamp AS DATE) = DATE '{d_iso}'
                  AND ABS(delta) BETWEEN {DELTA_MIN} AND {DELTA_MAX}
                  AND CAST(expiration AS DATE) - DATE '{d_iso}' BETWEEN {dte_min} AND {dte_max}
                  AND ABS(vega) > {MIN_VEGA}
                  AND bid != 0 AND ask != 0
                  AND bid IS NOT NULL AND ask IS NOT NULL
                ORDER BY CAST(timestamp AS TIMESTAMP), strike
            ) TO '{gp}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true)
        """)

        # ── Trades: filtered + _minute column ─────────────────────────────
        tcon.execute(f"""
            COPY (
                SELECT *,
                       CAST(timestamp AS DATE) AS trade_date,
                       time_bucket(INTERVAL '1 minute', CAST(timestamp AS TIMESTAMP)) AS _minute
                FROM trades
                WHERE symbol = '{sym_sql}'
                  AND CAST(timestamp AS DATE) = DATE '{d_iso}'
                  AND CAST(expiration AS DATE) - DATE '{d_iso}' BETWEEN {dte_min} AND {dte_max}
                  AND price > 0
                  AND size > 0
                ORDER BY CAST(timestamp AS TIMESTAMP), strike
            ) TO '{trp}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true)
        """)

        # ── Quotes: filtered + _minute column ─────────────────────────────
        tcon.execute(f"""
            COPY (
                SELECT *,
                       CAST(timestamp AS DATE) AS trade_date,
                       time_bucket(INTERVAL '1 minute', CAST(timestamp AS TIMESTAMP)) AS _minute
                FROM quotes
                WHERE symbol = '{sym_sql}'
                  AND CAST(timestamp AS DATE) = DATE '{d_iso}'
                  AND CAST(expiration AS DATE) - DATE '{d_iso}' BETWEEN {dte_min} AND {dte_max}
                  AND bid > 0
                  AND ask > 0
                ORDER BY CAST(timestamp AS TIMESTAMP), strike
            ) TO '{qp}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true)
        """)

        # ── OHLC: filtered + _minute column ───────────────────────────────
        ocon.execute(f"""
            COPY (
                SELECT *,
                       CAST(timestamp AS DATE) AS trade_date,
                       time_bucket(INTERVAL '1 minute', CAST(timestamp AS TIMESTAMP)) AS _minute
                FROM ohlcv
                WHERE symbol = '{sym_sql}'
                  AND CAST(timestamp AS DATE) = DATE '{d_iso}'
                  AND CAST(expiration AS DATE) - DATE '{d_iso}' BETWEEN {dte_min} AND {dte_max}
                ORDER BY CAST(timestamp AS TIMESTAMP), strike
            ) TO '{op}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true)
        """)

        # Optional backward-compatible combined TQ parquet.
        if write_combined_tq:
            tcon.execute(f"""
                COPY (
                    WITH q AS (
                        SELECT symbol, expiration, strike, "right",
                               NULL::TIMESTAMP AS trade_timestamp,
                               CAST(timestamp AS TIMESTAMP) AS quote_timestamp,
                               NULL::BIGINT AS "sequence",
                               NULL::VARCHAR AS ext_condition1,
                               NULL::VARCHAR AS ext_condition2,
                               NULL::VARCHAR AS ext_condition3,
                               NULL::VARCHAR AS ext_condition4,
                               NULL::VARCHAR AS "condition",
                               NULL::DOUBLE AS size,
                               NULL::VARCHAR AS exchange,
                               NULL::DOUBLE AS price,
                               bid_size, bid_exchange, bid, bid_condition,
                               ask_size, ask_exchange, ask, ask_condition,
                               CAST(timestamp AS DATE) AS trade_date,
                               time_bucket(INTERVAL '1 minute', CAST(timestamp AS TIMESTAMP)) AS _minute
                        FROM quotes
                        WHERE symbol = '{sym_sql}'
                          AND CAST(timestamp AS DATE) = DATE '{d_iso}'
                          AND CAST(expiration AS DATE) - DATE '{d_iso}' BETWEEN {dte_min} AND {dte_max}
                          AND bid > 0 AND ask > 0
                    ), tr AS (
                        SELECT symbol, expiration, strike, "right",
                               CAST(timestamp AS TIMESTAMP) AS trade_timestamp,
                               NULL::TIMESTAMP AS quote_timestamp,
                               "sequence",
                               ext_condition1, ext_condition2, ext_condition3, ext_condition4,
                               "condition", size, exchange, price,
                               NULL::DOUBLE AS bid_size,
                               NULL::VARCHAR AS bid_exchange,
                               NULL::DOUBLE AS bid,
                               NULL::VARCHAR AS bid_condition,
                               NULL::DOUBLE AS ask_size,
                               NULL::VARCHAR AS ask_exchange,
                               NULL::DOUBLE AS ask,
                               NULL::VARCHAR AS ask_condition,
                               CAST(timestamp AS DATE) AS trade_date,
                               time_bucket(INTERVAL '1 minute', CAST(timestamp AS TIMESTAMP)) AS _minute
                        FROM trades
                        WHERE symbol = '{sym_sql}'
                          AND CAST(timestamp AS DATE) = DATE '{d_iso}'
                          AND CAST(expiration AS DATE) - DATE '{d_iso}' BETWEEN {dte_min} AND {dte_max}
                          AND price > 0 AND size > 0
                    )
                    SELECT * FROM q UNION ALL SELECT * FROM tr
                ) TO '{tp}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true)
            """)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {symbol}: {i+1}/{len(trade_dates)} dates, "
                  f"{skipped} skipped, {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    print(f"  {symbol}: DONE {len(trade_dates)} dates "
          f"({skipped} skipped) in {elapsed:.1f}s → {sym_dir}")
    gcon.close(); tcon.close(); ocon.close()


def _extract_symbol_worker(payload):
    symbol, source_root_s, out_root_s, memory_limit, threads, overwrite, write_combined_tq = payload
    source_root = Path(source_root_s)
    out_root = Path(out_root_s)
    t0 = time.time()
    try:
        extract_symbol(
            symbol=symbol,
            source_root=source_root,
            out_root=out_root,
            memory_limit=memory_limit,
            threads=threads,
            overwrite=overwrite,
            write_combined_tq=write_combined_tq,
        )
        return symbol, True, time.time() - t0, ""
    except Exception as e:
        return symbol, False, time.time() - t0, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Extract theta_data_3year tier1 per-trade_date artifacts"
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--all-symbols", action="store_true")
    parser.add_argument("--source-root", type=str, default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--memory-limit", type=str, default="10GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of symbols to process concurrently")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-combined-tq", action="store_true",
                        help="Also write legacy *_tq.parquet by unioning quote+trade rows")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.all_symbols:
        symbols = sorted([p.name.upper() for p in source_root.iterdir() if p.is_dir()]) if source_root.exists() else ALL_SYMBOLS
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        parser.error("Specify --symbols or --all-symbols")

    print("=" * 70)
    print("EXTRACT TIER1 (theta_data_3year per-trade_date)")
    print(f"  Symbols:  {symbols}")
    print(f"  Source:   {source_root}")
    print(f"  Output:   {out_root}")
    print(f"  Filters:  delta={DELTA_MIN}-{DELTA_MAX}, DTE default={DTE_MIN}-{DTE_MAX_DEFAULT}, VIXW DTE={DTE_MIN}-{DTE_MAX_VIXW}, vega>{MIN_VEGA}")
    print(f"  Jobs: {args.jobs}, threads/job: {args.threads}")
    print(f"  Overwrite: {args.overwrite}")
    print(f"  Write legacy combined tq: {args.write_combined_tq}")
    print("=" * 70)

    t_total = time.time()

    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")

    runnable = []
    for symbol in symbols:
        if not (source_root / symbol).exists():
            print(f"  {symbol}: SKIP — symbol folder not found in {source_root}")
            continue
        runnable.append(symbol)

    if args.jobs <= 1 or len(runnable) <= 1:
        for symbol in runnable:
            print(f"\n  {symbol}: extracting from {source_root / symbol}")
            extract_symbol(
                symbol=symbol,
                source_root=source_root,
                out_root=out_root,
                memory_limit=args.memory_limit,
                threads=args.threads,
                overwrite=args.overwrite,
                write_combined_tq=args.write_combined_tq,
            )
    else:
        print(f"\nRunning {len(runnable)} symbols with jobs={args.jobs}")
        payloads = [
            (
                s,
                str(source_root),
                str(out_root),
                args.memory_limit,
                args.threads,
                args.overwrite,
                args.write_combined_tq,
            )
            for s in runnable
        ]
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            fut_map = {ex.submit(_extract_symbol_worker, p): p[0] for p in payloads}
            for fut in as_completed(fut_map):
                symbol = fut_map[fut]
                sym, ok, elapsed, err = fut.result()
                if ok:
                    print(f"  [{sym}] completed in {elapsed:.1f}s")
                else:
                    print(f"  [{symbol}] FAILED in {elapsed:.1f}s: {err}")

    print(f"\n{'=' * 70}")
    print(f"ALL DONE in {time.time() - t_total:.1f}s")
    print(f"Output: {out_root}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
