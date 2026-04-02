#!/usr/bin/env python3
"""
Preflight coverage audit for candidate Hybrid55 feature sources.

Reports per-symbol coverage for:
- recovered Greek columns
- trade/quote columns used by Phase1 modules
- OHLC columns used by AgentH/OHLC block
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

RECOVERED_GREEKS = ["rho", "epsilon", "vomma", "veta", "color", "dual_delta", "d1", "d2", "ultima", "iv_error"]
PHASE1_TQ = ["price", "size", "bid", "ask", "bid_size", "ask_size", "condition", "exchange", "sequence"]
OHLC_COLS = ["open", "high", "low", "close", "volume", "count"]


def _coverage(con: duckdb.DuckDBPyConnection, table: str, cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    schema = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
    for c in cols:
        if c not in schema:
            out[c] = 0.0
            continue
        q = f"SELECT AVG(CASE WHEN {c} IS NOT NULL AND ABS(COALESCE(CAST({c} AS DOUBLE),0.0)) > 1e-12 THEN 1.0 ELSE 0.0 END) FROM {table}"
        out[c] = float(con.execute(q).fetchone()[0] or 0.0)
    return out


def _mean(d: dict[str, float]) -> float:
    return float(sum(d.values()) / max(1, len(d)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/workspace/data/theta_data_3year")
    p.add_argument("--symbols", nargs="+", default=["SPXW", "SPY", "QQQ", "IWM", "TLT", "VIXW"])
    p.add_argument("--out", default="")
    args = p.parse_args()

    root = Path(args.data_root)
    report = {"data_root": str(root), "symbols": {}}

    for sym in [s.upper() for s in args.symbols]:
        sym_dir = root / sym
        greeks_db = sym_dir / "greeks_all.duckdb"
        tq_db = sym_dir / "trade_quote.duckdb"
        ohlc_db = sym_dir / "ohlcv.duckdb"

        sym_out = {}
        if greeks_db.exists():
            con = duckdb.connect(str(greeks_db), read_only=True)
            table = con.execute("SHOW TABLES").fetchone()[0]
            cov = _coverage(con, table, RECOVERED_GREEKS)
            sym_out["recovered_greeks"] = {"per_col": cov, "mean": _mean(cov)}
            con.close()
        if tq_db.exists():
            con = duckdb.connect(str(tq_db), read_only=True)
            table = con.execute("SHOW TABLES").fetchone()[0]
            cov = _coverage(con, table, PHASE1_TQ)
            sym_out["phase1_tq"] = {"per_col": cov, "mean": _mean(cov)}
            con.close()
        if ohlc_db.exists():
            con = duckdb.connect(str(ohlc_db), read_only=True)
            table = con.execute("SHOW TABLES").fetchone()[0]
            cov = _coverage(con, table, OHLC_COLS)
            sym_out["ohlc"] = {"per_col": cov, "mean": _mean(cov)}
            con.close()

        report["symbols"][sym] = sym_out

    text = json.dumps(report, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"saved: {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

