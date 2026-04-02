#!/usr/bin/env python3
"""
Preflight checks for /workspace/historical_data_1yr before CSV-first reprocessing.

This script is read-only and does not modify source data. It validates:
  - expected symbol folders / file patterns
  - required columns in historical weekly CSV files
  - timestamp parseability in sampled files
  - basic OI presence and parseability
  - week coverage gaps between first and last observed week

Usage:
  python scripts/phase0/preflight_historical_data_1yr.py
  python scripts/phase0/preflight_historical_data_1yr.py --strict
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


DEFAULT_CSV_ROOT = Path("/workspace/historical_data_1yr")
DEFAULT_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT", "VIXW"]
REQUIRED_HIST_COLS = ["timestamp", "underlying_price", "bid", "ask", "strike", "right", "expiration"]


@dataclass(frozen=True)
class SymbolLayout:
    symbol: str
    symbol_dir: Path
    hist_files: list[Path]
    oi_files: list[Path]


def _iso_week_key(p: Path) -> tuple[int, int] | None:
    m = re.search(r"(\d{4})-W(\d{2})", p.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _resolve_symbol_layout(csv_root: Path, symbol: str) -> SymbolLayout | None:
    nested = csv_root / symbol
    if nested.exists():
        hist = sorted(nested.glob(f"{symbol}_historical_*.csv"))
        oi = sorted((nested / "OI").glob(f"{symbol}_oi_*.csv")) if (nested / "OI").exists() else []
        oi.extend(sorted(nested.glob(f"{symbol}_oi_*.csv")))
        return SymbolLayout(symbol=symbol, symbol_dir=nested, hist_files=hist, oi_files=sorted(set(oi)))
    # Flat fallback
    hist = sorted(csv_root.glob(f"{symbol}_historical_*.csv"))
    oi = sorted(csv_root.glob(f"{symbol}_oi_*.csv"))
    if not hist and not oi:
        return None
    return SymbolLayout(symbol=symbol, symbol_dir=csv_root, hist_files=hist, oi_files=oi)


def _sample_files(files: list[Path]) -> list[Path]:
    if not files:
        return []
    if len(files) <= 3:
        return files
    mid = len(files) // 2
    return [files[0], files[mid], files[-1]]


def _safe_read_csv(path: Path, nrows: int = 2000) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _analyze_hist_samples(paths: Iterable[Path]) -> dict:
    out = {"files_checked": 0, "missing_columns": {}, "timestamp_nat_ratio": {}, "parse_errors": {}}
    for p in paths:
        try:
            df = _safe_read_csv(p)
            out["files_checked"] += 1
            cols = [c.lower().strip() for c in df.columns]
            missing = [c for c in REQUIRED_HIST_COLS if c not in cols]
            if missing:
                out["missing_columns"][str(p)] = missing
            if "timestamp" in cols:
                ts = pd.to_datetime(df[df.columns[cols.index("timestamp")]], errors="coerce")
                out["timestamp_nat_ratio"][str(p)] = float(ts.isna().mean())
        except Exception as e:  # pragma: no cover - defensive
            out["parse_errors"][str(p)] = str(e)
    return out


def _analyze_oi_samples(paths: Iterable[Path]) -> dict:
    out = {"files_checked": 0, "timestamp_nat_ratio": {}, "parse_errors": {}}
    for p in paths:
        try:
            df = _safe_read_csv(p)
            out["files_checked"] += 1
            cols = [c.lower().strip() for c in df.columns]
            if "timestamp" in cols:
                ts = pd.to_datetime(df[df.columns[cols.index("timestamp")]], errors="coerce")
                out["timestamp_nat_ratio"][str(p)] = float(ts.isna().mean())
        except Exception as e:  # pragma: no cover - defensive
            out["parse_errors"][str(p)] = str(e)
    return out


def _missing_weeks(week_keys: list[tuple[int, int]]) -> list[str]:
    if not week_keys:
        return []
    # Build an ordered integer axis: year * 100 + week.
    vals = sorted({y * 100 + w for y, w in week_keys})
    missing = []
    for i in range(len(vals) - 1):
        a, b = vals[i], vals[i + 1]
        if b - a <= 1:
            continue
        for k in range(a + 1, b):
            y = k // 100
            w = k % 100
            if 1 <= w <= 53:
                missing.append(f"{y}-W{w:02d}")
    return missing


def run_preflight(csv_root: Path, symbols: list[str]) -> dict:
    report = {
        "csv_root": str(csv_root),
        "symbols": {},
        "summary": {"symbols_ok": 0, "symbols_failed": 0},
    }
    for sym in symbols:
        layout = _resolve_symbol_layout(csv_root, sym)
        if layout is None:
            report["symbols"][sym] = {"ok": False, "error": "symbol_dir_not_found"}
            report["summary"]["symbols_failed"] += 1
            continue

        week_keys = [wk for wk in (_iso_week_key(p) for p in layout.hist_files) if wk is not None]
        missing_weeks = _missing_weeks(week_keys)
        hist_analysis = _analyze_hist_samples(_sample_files(layout.hist_files))
        oi_analysis = _analyze_oi_samples(_sample_files(layout.oi_files))

        hard_fail = (
            len(layout.hist_files) == 0
            or len(hist_analysis["missing_columns"]) > 0
            or len(hist_analysis["parse_errors"]) > 0
        )
        symbol_report = {
            "ok": not hard_fail,
            "symbol_dir": str(layout.symbol_dir),
            "historical_files": len(layout.hist_files),
            "oi_files": len(layout.oi_files),
            "missing_weeks_between_first_last": missing_weeks[:200],
            "hist_sample_analysis": hist_analysis,
            "oi_sample_analysis": oi_analysis,
        }
        report["symbols"][sym] = symbol_report
        if symbol_report["ok"]:
            report["summary"]["symbols_ok"] += 1
        else:
            report["summary"]["symbols_failed"] += 1
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight /workspace/historical_data_1yr")
    parser.add_argument("--csv-root", default=str(DEFAULT_CSV_ROOT))
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--report-out", default="/tmp/historical_data_1yr_preflight.json")
    parser.add_argument("--strict", action="store_true",
                        help="Return non-zero if any symbol has hard preflight failure.")
    args = parser.parse_args()

    csv_root = Path(args.csv_root)
    report = run_preflight(csv_root, [s.upper() for s in args.symbols])
    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report["summary"], indent=2))
    print(f"Report saved: {out_path}")
    if args.strict and report["summary"]["symbols_failed"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
