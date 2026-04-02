#!/usr/bin/env python3
"""
Build DuckDB databases from historical CSV archive (no API calls).

Input archive (default):
  /workspace/historical_data_1yr/{SYMBOL}/{SYMBOL}_historical_*.csv
  /workspace/historical_data_1yr/{SYMBOL}/OI/{SYMBOL}_oi_*.csv  (optional)

Output DBs (default):
  /workspace/data/data_in_2026/db_spxw.duckdb   (SPXW)
  /workspace/data/data_in_2026/db_spy.duckdb    (SPY)
  /workspace/data/data_in_2026/db_qqq.duckdb    (QQQ)
  /workspace/data/data_in_2026/db_small.duckdb  (IWM,TLT,VIXW by default)

Each DB contains:
  - options_greek
  - options_trade_quote

These tables match the interface expected by extract_tier1.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import duckdb


DEFAULT_CSV_ROOT = Path("/workspace/historical_data_1yr")
DEFAULT_OUT_DIR = Path("/workspace/data/data_in_2026")

DB_MAP = {
    "SPXW": "db_spxw.duckdb",
    "SPY": "db_spy.duckdb",
    "QQQ": "db_qqq.duckdb",
}


def _glob_to_sql(path_glob: str) -> str:
    return path_glob.replace("'", "''")


def _symbol_hist_glob(csv_root: Path, symbol: str) -> str:
    nested = csv_root / symbol / f"{symbol}_historical_*.csv"
    if list((csv_root / symbol).glob(f"{symbol}_historical_*.csv")):
        return str(nested)
    return str(csv_root / f"{symbol}_historical_*.csv")


def _iter_hist_files(csv_root: Path, symbol: str) -> list[Path]:
    nested = sorted((csv_root / symbol).glob(f"{symbol}_historical_*.csv"))
    if nested:
        return nested
    return sorted(csv_root.glob(f"{symbol}_historical_*.csv"))


def _symbol_oi_globs(csv_root: Path, symbol: str) -> list[str]:
    globs = []
    nested_oi = csv_root / symbol / "OI" / f"{symbol}_oi_*.csv"
    if list((csv_root / symbol / "OI").glob(f"{symbol}_oi_*.csv")):
        globs.append(str(nested_oi))
    nested_flat = csv_root / symbol / f"{symbol}_oi_*.csv"
    if list((csv_root / symbol).glob(f"{symbol}_oi_*.csv")):
        globs.append(str(nested_flat))
    root_flat = csv_root / f"{symbol}_oi_*.csv"
    if list(csv_root.glob(f"{symbol}_oi_*.csv")):
        globs.append(str(root_flat))
    return sorted(set(globs))


def _create_base_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS options_greek (
            symbol VARCHAR,
            expiration DATE,
            strike DOUBLE,
            "right" VARCHAR,
            timestamp TIMESTAMP,
            bid DOUBLE,
            ask DOUBLE,
            delta DOUBLE,
            theta DOUBLE,
            vega DOUBLE,
            rho DOUBLE,
            epsilon DOUBLE,
            "lambda" DOUBLE,
            gamma DOUBLE,
            vanna DOUBLE,
            charm DOUBLE,
            vomma DOUBLE,
            veta DOUBLE,
            vera DOUBLE,
            speed DOUBLE,
            zomma DOUBLE,
            color DOUBLE,
            ultima DOUBLE,
            d1 DOUBLE,
            d2 DOUBLE,
            dual_delta DOUBLE,
            dual_gamma DOUBLE,
            implied_vol DOUBLE,
            iv_error DOUBLE,
            underlying_timestamp TIMESTAMP,
            underlying_price DOUBLE,
            open_interest DOUBLE,
            trade_date DATE,
            week_key VARCHAR,
            moneyness DOUBLE,
            dist_atm_pct DOUBLE,
            mid DOUBLE,
            spread DOUBLE,
            spread_pct DOUBLE,
            lambda_ratio DOUBLE,
            dte_int INTEGER,
            cp_sign INTEGER,
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS options_trade_quote (
            symbol VARCHAR,
            expiration DATE,
            strike DOUBLE,
            "right" VARCHAR,
            trade_timestamp TIMESTAMP,
            quote_timestamp TIMESTAMP,
            sequence BIGINT,
            ext_condition1 VARCHAR,
            ext_condition2 VARCHAR,
            ext_condition3 VARCHAR,
            ext_condition4 VARCHAR,
            "condition" VARCHAR,
            size DOUBLE,
            exchange VARCHAR,
            price DOUBLE,
            bid_size DOUBLE,
            bid_exchange VARCHAR,
            bid DOUBLE,
            bid_condition VARCHAR,
            ask_size DOUBLE,
            ask_exchange VARCHAR,
            ask DOUBLE,
            ask_condition VARCHAR,
            trade_date DATE,
            week_key VARCHAR
        )
        """
    )


def _build_oi_temp(con: duckdb.DuckDBPyConnection, oi_globs: list[str]) -> bool:
    if not oi_globs:
        return False

    def _col_expr(cols: set[str], names: Sequence[str], target_type: str) -> str:
        parts = [
            f'TRY_CAST(raw."{name}" AS {target_type})'
            for name in names
            if name.lower() in cols
        ]
        if not parts:
            return f"NULL::{target_type}"
        if len(parts) == 1:
            return parts[0]
        return f"COALESCE({', '.join(parts)})"

    union_sql = " UNION ALL ".join(
        [
            f"""
            SELECT
                CAST({_col_expr(_read_csv_columns(con, g), ['expiration'], 'DATE')} AS DATE) AS expiration,
                {_col_expr(_read_csv_columns(con, g), ['strike'], 'DOUBLE')} AS strike,
                UPPER(TRIM(CAST({_col_expr(_read_csv_columns(con, g), ['right', 'option_type'], 'VARCHAR')} AS VARCHAR))) AS "right",
                CAST({_col_expr(_read_csv_columns(con, g), ['query_date', 'timestamp', 'datetime', 'date'], 'TIMESTAMP')} AS DATE) AS oi_date,
                {_col_expr(_read_csv_columns(con, g), ['open_interest', 'oi'], 'DOUBLE')} AS open_interest
            FROM read_csv_auto('{_glob_to_sql(g)}', all_varchar=true, union_by_name=true, ignore_errors=true) AS raw
            """
            for g in oi_globs
        ]
    )
    con.execute("DROP TABLE IF EXISTS oi_raw")
    con.execute(f"CREATE TEMP TABLE oi_raw AS {union_sql}")
    con.execute("DROP TABLE IF EXISTS oi_dedup")
    con.execute(
        """
        CREATE TEMP TABLE oi_dedup AS
        SELECT expiration, strike, "right", oi_date, open_interest
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY expiration, strike, "right", oi_date
                       ORDER BY oi_date DESC
                   ) AS rn
            FROM oi_raw
            WHERE expiration IS NOT NULL
              AND strike IS NOT NULL
              AND "right" IS NOT NULL
              AND oi_date IS NOT NULL
        )
        WHERE rn = 1
        """
    )
    return True


def _insert_symbol(
    con: duckdb.DuckDBPyConnection,
    csv_root: Path,
    symbol: str,
    include_oi: bool = True,
    verbose: bool = True,
) -> tuple[int, int]:
    hist_files = _iter_hist_files(csv_root, symbol)
    if not hist_files:
        if verbose:
            print(f"[{symbol}] no historical CSV files, skip")
        return 0, 0

    oi_globs = _symbol_oi_globs(csv_root, symbol)
    has_oi = _build_oi_temp(con, oi_globs) if include_oi and oi_globs else False

    for i, hist_file in enumerate(hist_files, start=1):
        raw_cols = _read_csv_columns(con, str(hist_file))

        def _raw_col(names: Sequence[str], target_type: str) -> str:
            parts = [
                f'TRY_CAST(raw."{name}" AS {target_type})'
                for name in names
                if name.lower() in raw_cols
            ]
            if not parts:
                return f"NULL::{target_type}"
            if len(parts) == 1:
                return parts[0]
            return f"COALESCE({', '.join(parts)})"

        strike_expr = _raw_col(["strike"], "DOUBLE")
        right_expr = 'UPPER(TRIM(CAST(' + _raw_col(["right", "option_type"], "VARCHAR") + " AS VARCHAR)))"
        ts_expr = _raw_col(["timestamp", "datetime"], "TIMESTAMP")
        qdate_expr = _raw_col(["query_date", "date"], "TIMESTAMP")
        trade_date_expr = f"CAST(COALESCE({qdate_expr}, {ts_expr}) AS DATE)"
        bid_expr = _raw_col(["bid"], "DOUBLE")
        ask_expr = _raw_col(["ask"], "DOUBLE")
        underlying_price_expr = _raw_col(["underlying_price", "close", "vwap", "mid"], "DOUBLE")

        join_oi = f"""
            LEFT JOIN oi_dedup oi
              ON oi.expiration = CAST(TRY_CAST(raw."expiration" AS DATE) AS DATE)
             AND oi.strike = {strike_expr}
             AND oi."right" = {right_expr}
             AND oi.oi_date = {trade_date_expr}
        """ if has_oi else ""
        oi_col = "oi.open_interest" if has_oi else "NULL::DOUBLE"

        con.execute(
            f"""
            INSERT INTO options_greek
            SELECT
                UPPER('{symbol}') AS symbol,
                CAST(TRY_CAST(raw."expiration" AS DATE) AS DATE) AS expiration,
                {strike_expr} AS strike,
                {right_expr} AS "right",
                {ts_expr} AS timestamp,
                {bid_expr} AS bid,
                {ask_expr} AS ask,
                {_raw_col(["delta"], "DOUBLE")} AS delta,
                {_raw_col(["theta"], "DOUBLE")} AS theta,
                {_raw_col(["vega"], "DOUBLE")} AS vega,
                {_raw_col(["rho"], "DOUBLE")} AS rho,
                {_raw_col(["epsilon"], "DOUBLE")} AS epsilon,
                {_raw_col(["lambda"], "DOUBLE")} AS "lambda",
                {_raw_col(["gamma"], "DOUBLE")} AS gamma,
                {_raw_col(["vanna"], "DOUBLE")} AS vanna,
                {_raw_col(["charm"], "DOUBLE")} AS charm,
                {_raw_col(["vomma"], "DOUBLE")} AS vomma,
                {_raw_col(["veta"], "DOUBLE")} AS veta,
                {_raw_col(["vera"], "DOUBLE")} AS vera,
                {_raw_col(["speed"], "DOUBLE")} AS speed,
                {_raw_col(["zomma"], "DOUBLE")} AS zomma,
                {_raw_col(["color"], "DOUBLE")} AS color,
                {_raw_col(["ultima"], "DOUBLE")} AS ultima,
                {_raw_col(["d1"], "DOUBLE")} AS d1,
                {_raw_col(["d2"], "DOUBLE")} AS d2,
                {_raw_col(["dual_delta"], "DOUBLE")} AS dual_delta,
                {_raw_col(["dual_gamma"], "DOUBLE")} AS dual_gamma,
                {_raw_col(["implied_vol"], "DOUBLE")} AS implied_vol,
                {_raw_col(["iv_error"], "DOUBLE")} AS iv_error,
                {ts_expr} AS underlying_timestamp,
                {underlying_price_expr} AS underlying_price,
                {oi_col} AS open_interest,
                {trade_date_expr} AS trade_date,
                regexp_extract(raw.filename, '(\\d{{4}}-W\\d{{2}}_part\\d{{3}})', 1) AS week_key,
                {_raw_col(["moneyness"],    "DOUBLE")} AS moneyness,
                {_raw_col(["dist_atm_pct"], "DOUBLE")} AS dist_atm_pct,
                COALESCE({_raw_col(["mid"], "DOUBLE")}, ({bid_expr} + {ask_expr}) / 2.0) AS mid,
                COALESCE({_raw_col(["spread"], "DOUBLE")}, {ask_expr} - {bid_expr}) AS spread,
                COALESCE({_raw_col(["spread_pct"], "DOUBLE")},
                    CASE WHEN ({bid_expr} + {ask_expr}) > 0
                         THEN ({ask_expr} - {bid_expr}) / (({bid_expr} + {ask_expr}) / 2.0)
                         ELSE NULL END) AS spread_pct,
                {_raw_col(["lambda"], "DOUBLE")} AS lambda_ratio,
                COALESCE(TRY_CAST({_raw_col(["dte"], "INTEGER")} AS INTEGER),
                    CAST(CAST(TRY_CAST(raw."expiration" AS DATE) AS DATE) - {trade_date_expr} AS INTEGER)) AS dte_int,
                COALESCE(TRY_CAST({_raw_col(["cp_sign"], "INTEGER")} AS INTEGER),
                    CASE WHEN {right_expr} = 'CALL' THEN 1
                         WHEN {right_expr} = 'PUT'  THEN -1
                         ELSE 0 END) AS cp_sign
            FROM read_csv_auto(
                '{_glob_to_sql(str(hist_file))}',
                all_varchar=true,
                union_by_name=true,
                ignore_errors=true,
                filename=true
            ) AS raw
            {join_oi}
            WHERE {strike_expr} IS NOT NULL
              AND {ts_expr} IS NOT NULL
              AND {bid_expr} IS NOT NULL
              AND {ask_expr} IS NOT NULL
            """
        )

        con.execute(
            f"""
            INSERT INTO options_trade_quote
            SELECT
                UPPER('{symbol}') AS symbol,
                CAST(TRY_CAST(raw."expiration" AS DATE) AS DATE) AS expiration,
                {strike_expr} AS strike,
                {right_expr} AS "right",
                {ts_expr} AS trade_timestamp,
                {ts_expr} AS quote_timestamp,
                NULL::BIGINT AS sequence,
                NULL::VARCHAR AS ext_condition1,
                NULL::VARCHAR AS ext_condition2,
                NULL::VARCHAR AS ext_condition3,
                NULL::VARCHAR AS ext_condition4,
                NULL::VARCHAR AS "condition",
                COALESCE({_raw_col(["volume"], "DOUBLE")}, {_raw_col(["count"], "DOUBLE")}) AS size,
                NULL::VARCHAR AS exchange,
                COALESCE({_raw_col(["vwap"], "DOUBLE")}, {_raw_col(["close"], "DOUBLE")}, {_raw_col(["mid"], "DOUBLE")}) AS price,
                {_raw_col(["bid_size"], "DOUBLE")} AS bid_size,
                COALESCE({_raw_col(["bid_exchange"], "VARCHAR")}, NULL::VARCHAR) AS bid_exchange,
                {bid_expr} AS bid,
                NULL::VARCHAR AS bid_condition,
                {_raw_col(["ask_size"], "DOUBLE")} AS ask_size,
                COALESCE({_raw_col(["ask_exchange"], "VARCHAR")}, NULL::VARCHAR) AS ask_exchange,
                {ask_expr} AS ask,
                NULL::VARCHAR AS ask_condition,
                {trade_date_expr} AS trade_date,
                regexp_extract(raw.filename, '(\\d{{4}}-W\\d{{2}}_part\\d{{3}})', 1) AS week_key,
            FROM read_csv_auto(
                '{_glob_to_sql(str(hist_file))}',
                all_varchar=true,
                union_by_name=true,
                ignore_errors=true,
                filename=true
            ) AS raw
            WHERE {strike_expr} IS NOT NULL
              AND {ts_expr} IS NOT NULL
            """
        )
        if verbose and (i == 1 or i % 10 == 0 or i == len(hist_files)):
            print(f"[{symbol}] processed {i}/{len(hist_files)} files")

    greek_n = con.execute("SELECT COUNT(*) FROM options_greek WHERE symbol = ?", [symbol]).fetchone()[0]
    tq_n = con.execute("SELECT COUNT(*) FROM options_trade_quote WHERE symbol = ?", [symbol]).fetchone()[0]
    if verbose:
        print(f"[{symbol}] inserted greek={greek_n:,} tq={tq_n:,}")
    return int(greek_n), int(tq_n)


def _db_groups(symbols: Iterable[str], include_vixw_in_small: bool) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "db_spxw.duckdb": [],
        "db_spy.duckdb": [],
        "db_qqq.duckdb": [],
        "db_small.duckdb": [],
    }
    for s in symbols:
        su = s.upper()
        if su in DB_MAP:
            groups[DB_MAP[su]].append(su)
        else:
            if su == "VIXW" and not include_vixw_in_small:
                continue
            groups["db_small.duckdb"].append(su)
    return {k: v for k, v in groups.items() if v}


def _read_csv_columns(con: duckdb.DuckDBPyConnection, file_glob: str) -> set[str]:
    rows = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_csv_auto(
            '{_glob_to_sql(file_glob)}',
            all_varchar=true,
            union_by_name=true,
            ignore_errors=true,
            filename=true
        )
        """
    ).fetchall()
    return {str(r[0]).lower() for r in rows}


def build_databases(
    csv_root: Path,
    out_dir: Path,
    symbols: list[str],
    include_vixw_in_small: bool = True,
    include_oi: bool = True,
    memory_limit_gb: int = 2,
    threads: int = 1,
    temp_dir: Path | None = Path("/tmp/duckdb_tmp"),
    overwrite: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = _db_groups(symbols, include_vixw_in_small=include_vixw_in_small)
    summary: dict[str, dict] = {}

    for db_name, db_symbols in groups.items():
        db_path = out_dir / db_name
        if overwrite and db_path.exists():
            db_path.unlink()

        con = duckdb.connect(str(db_path))
        con.execute(f"PRAGMA memory_limit='{max(1, memory_limit_gb)}GB'")
        con.execute(f"PRAGMA threads={max(1, threads)}")
        con.execute("PRAGMA preserve_insertion_order=false")
        if temp_dir is not None:
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_dir_sql = str(temp_dir).replace("'", "''")
            con.execute(f"PRAGMA temp_directory='{temp_dir_sql}'")
        _create_base_tables(con)

        print(f"\nDB: {db_path}  symbols={db_symbols}")
        db_stats = {"symbols": db_symbols, "greek_rows": 0, "tq_rows": 0}
        for sym in db_symbols:
            g, t = _insert_symbol(con, csv_root=csv_root, symbol=sym, include_oi=include_oi, verbose=True)
            db_stats["greek_rows"] += g
            db_stats["tq_rows"] += t
            con.execute("CHECKPOINT")

        con.execute("ANALYZE")
        con.close()
        summary[str(db_path)] = db_stats

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-symbol DuckDB files from historical CSV archive")
    parser.add_argument("--csv-root", default=str(DEFAULT_CSV_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--symbols", nargs="+", default=["SPXW", "SPY", "QQQ", "IWM", "TLT", "VIXW"])
    parser.add_argument("--exclude-vixw-small", action="store_true",
                        help="Exclude VIXW from db_small (default includes VIXW in db_small).")
    parser.add_argument("--skip-oi", action="store_true",
                        help="Do not join OI files (lower memory usage, open_interest stays NULL).")
    parser.add_argument("--memory-limit-gb", type=int, default=2,
                        help="DuckDB memory limit in GB (default: 2).")
    parser.add_argument("--threads", type=int, default=1,
                        help="DuckDB worker threads (default: 1 for low-memory pods).")
    parser.add_argument("--temp-dir", default="/tmp/duckdb_tmp",
                        help="DuckDB spill temp directory (default: /tmp/duckdb_tmp).")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Do not delete existing DB files before rebuilding.")
    args = parser.parse_args()

    csv_root = Path(args.csv_root)
    out_dir = Path(args.out_dir)
    if not csv_root.exists():
        raise FileNotFoundError(f"CSV root not found: {csv_root}")

    summary = build_databases(
        csv_root=csv_root,
        out_dir=out_dir,
        symbols=[s.upper() for s in args.symbols],
        include_vixw_in_small=not args.exclude_vixw_small,
        include_oi=not args.skip_oi,
        memory_limit_gb=args.memory_limit_gb,
        threads=args.threads,
        temp_dir=Path(args.temp_dir) if args.temp_dir else None,
        overwrite=not args.no_overwrite,
    )
    print("\nSummary:")
    for db, st in summary.items():
        print(f"  {db}: greek={st['greek_rows']:,} tq={st['tq_rows']:,} symbols={st['symbols']}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
