#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import duckdb


DEFAULT_DB1 = "/workspace/data/data_in_2026/consolidated_options.duckdb"
DEFAULT_DB2 = "/workspace/data/data_in_2026/consolidated_options_part2.duckdb"


def _connect_with_optional_attach(db1: str, db2: str | None):
    con = duckdb.connect(db1, read_only=True)
    if db2 and Path(db2).exists():
        con.execute(f"ATTACH '{db2}' AS part2 (READ_ONLY)")
    return con


def _make_view(con: duckdb.DuckDBPyConnection, table: str, view: str, use_part2: bool):
    if use_part2:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {view} AS
            SELECT * FROM {table}
            UNION ALL
            SELECT * FROM part2.{table}
            """
        )
    else:
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM {table}")


def _describe_cols(con: duckdb.DuckDBPyConnection, table_or_view: str):
    return [r[0] for r in con.execute(f"DESCRIBE {table_or_view}").fetchall()]


def _col_stats_sql(table_or_view: str, symbol: str | None, col: str) -> str:
    where = []
    if symbol:
        where.append(f"symbol='{symbol}'")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    return f"""
        SELECT
            COUNT(*) AS n,
            COUNT({col}) AS n_nonnull,
            COUNT(DISTINCT {col}) AS n_distinct,
            MIN({col}) AS v_min,
            MAX({col}) AS v_max
        FROM {table_or_view}
        {where_sql}
    """


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db1", default=DEFAULT_DB1)
    ap.add_argument("--db2", default=DEFAULT_DB2)
    ap.add_argument("--table", choices=["options_trade_quote", "options_greek"], default="options_trade_quote")
    ap.add_argument("--symbol", default="SPXW")
    ap.add_argument("--all-symbols", action="store_true")
    ap.add_argument("--use-part2", action="store_true", help="ATTACH db2 and UNION ALL into a view")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument("--limit-cols", type=int, default=0, help="Only audit first N cols (0=all)")
    args = ap.parse_args()

    con = _connect_with_optional_attach(args.db1, args.db2 if args.use_part2 else None)
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='2GB'")

    view = f"v_{args.table}"
    _make_view(con, args.table, view, use_part2=args.use_part2)

    cols = _describe_cols(con, view)
    if args.limit_cols and args.limit_cols > 0:
        cols = cols[: args.limit_cols]

    if args.all_symbols:
        symbols = [r[0] for r in con.execute(f"SELECT DISTINCT symbol FROM {view} ORDER BY symbol").fetchall()]
    else:
        symbols = [args.symbol]

    report: dict = {
        "db1": args.db1,
        "db2": args.db2 if args.use_part2 else None,
        "table": args.table,
        "view": view,
        "use_part2": bool(args.use_part2),
        "columns": cols,
        "symbols": symbols,
        "stats": {},
    }

    for sym in symbols:
        per_sym = {}
        for col in cols:
            try:
                row = con.execute(_col_stats_sql(view, sym, col)).fetchone()
            except Exception as e:
                per_sym[col] = {"error": f"{type(e).__name__}: {e}"}
                continue

            n, n_nonnull, n_distinct, v_min, v_max = row
            is_constant = (n_distinct == 1) and (n_nonnull == n)
            is_all_null = n_nonnull == 0
            per_sym[col] = {
                "n": int(n),
                "n_nonnull": int(n_nonnull),
                "n_distinct": int(n_distinct),
                "min": None if v_min is None else str(v_min),
                "max": None if v_max is None else str(v_max),
                "is_constant": bool(is_constant),
                "is_all_null": bool(is_all_null),
            }
        report["stats"][sym] = per_sym

    con.close()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(args.out)
    else:
        print(json.dumps(report, indent=2)[:20000])


if __name__ == "__main__":
    main()
