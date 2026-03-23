#!/usr/bin/env python3
"""
csv_to_tier1.py
Convert historical_fetcher CSVs + OI CSVs → tier1 parquets.

NEW format (fetcher v1.4):
  - One combined CSV per week per symbol: {SYM}_historical_{YYYY}-W{WW}_part{NNN}.csv
    Contains: greek + ohlc columns merged (no separate TQ file)
  - Separate OI CSV per day: {SYM}_oi_{YYYY-MM-DD}.csv
    OI is EOD only → joined by (expiration,strike,right,trade_date), NOT row-expanded

Outputs (compatible with build_tier2.py / tier2_reprocess.py):
  tier1_root/greek/symbol={SYM}/option_greek_all_{YYYY}-W{WW}_part{NNN}_filtered.parquet
  tier1_root/tradequote/symbol={SYM}/option_quote_{YYYY}-W{WW}_part{NNN}.parquet

Usage:
  python csv_to_tier1.py \
    --csv-dir /workspace/historical_data_90d \
    --out-root /workspace/data/tier1_2026_v1 \
    --symbols SPXW SPY QQQ IWM TLT
"""

import argparse
import datetime as _dt
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Filters (matching extract_tier1.py) ───────────────────────────────────────
DELTA_MIN = 0.20
DELTA_MAX = 0.90
MIN_VEGA  = 0.01
DTE_MAX_PER_SYMBOL = {
    # Keep aligned with historical_fetcher_v15 / theta_fetching_v5.
    "SPXW": 5, "SPY": 5, "QQQ": 5, "IWM": 5, "TLT": 5, "VIXW": 30,
}
DEFAULT_DTE_MAX = 5
DEFAULT_DTE_MIN = 0

ALL_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]

# ── Greek columns tier2_reprocess expects ─────────────────────────────────────
GREEK_OUT_COLS = [
    "symbol", "expiration", "strike", "right", "timestamp",
    "bid", "ask", "delta", "theta", "vega", "rho", "epsilon", "lambda", "gamma",
    "vanna", "charm", "vomma", "veta", "vera", "speed", "zomma", "color", "ultima",
    "d1", "d2", "dual_delta", "dual_gamma", "implied_vol", "iv_error",
    "underlying_timestamp", "underlying_price", "open_interest",
    "trade_date", "week_key",
]

# ── TQ columns tier2_reprocess expects ────────────────────────────────────────
TQ_OUT_COLS = [
    "symbol", "expiration", "strike", "right",
    "trade_timestamp", "quote_timestamp", "sequence",
    "ext_condition1", "ext_condition2", "ext_condition3", "ext_condition4",
    "condition", "size", "exchange", "price",
    "bid_size", "bid_exchange", "bid", "bid_condition",
    "ask_size", "ask_exchange", "ask", "ask_condition",
    "open", "high", "low", "close", "volume", "count", "vwap",
    "trade_date", "week_key",
]


def _parse_date(val):
    if isinstance(val, _dt.date) and not isinstance(val, _dt.datetime):
        return val
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None


def _week_key(trade_date: _dt.date, part: int) -> str:
    iso = trade_date.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}_part{part:03d}"


def _load_oi_for_dir(symbol_dir: Path, symbol: str) -> pd.DataFrame:
    """Load all OI CSVs for a symbol and return a single df keyed by
    (expiration_str, strike, right, query_date_str)."""
    oi_files = sorted(symbol_dir.glob(f"{symbol}_oi_*.csv"))
    oi_subdir = symbol_dir / "OI"
    if oi_subdir.exists():
        oi_files.extend(sorted(oi_subdir.glob(f"{symbol}_oi_*.csv")))
    oi_files = sorted(set(oi_files))
    if not oi_files:
        return pd.DataFrame()
    frames = []
    for f in oi_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            frames.append(df)
        except Exception as e:
            print(f"    WARN OI: {f.name}: {e}")
    if not frames:
        return pd.DataFrame()
    oi = pd.concat(frames, ignore_index=True)
    oi.columns = [c.lower().strip() for c in oi.columns]
    # Normalise key columns
    oi["expiration"] = pd.to_datetime(oi["expiration"], errors="coerce").dt.strftime("%Y-%m-%d")
    oi["strike"]     = pd.to_numeric(oi["strike"], errors="coerce")
    oi["right"]      = oi["right"].astype(str).str.upper().str.strip()
    # Use query_date as the join date (OI is EOD for that date)
    date_col = "query_date" if "query_date" in oi.columns else "timestamp"
    oi["_oi_date"]   = pd.to_datetime(oi[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    oi = oi[["expiration", "strike", "right", "_oi_date", "open_interest"]].drop_duplicates(
        subset=["expiration", "strike", "right", "_oi_date"]
    )
    return oi


def _resolve_symbol_dir(csv_root: Path, symbol: str) -> Optional[Path]:
    """
    Resolve source dir for a symbol from either layout:
      A) flat:   <root>/{SYM}_historical_*.csv
      B) nested: <root>/{SYM}/{SYM}_historical_*.csv (with optional OI/ subdir)
    """
    flat_has = any(csv_root.glob(f"{symbol}_historical_*.csv"))
    if flat_has:
        return csv_root
    nested = csv_root / symbol
    nested_has = nested.exists() and any(nested.glob(f"{symbol}_historical_*.csv"))
    if nested_has:
        return nested
    return None


def process_symbol(symbol: str, csv_root: Path, out_root: Path, dte_max: int, overwrite: bool = False) -> None:
    greek_out = out_root / "greek"      / f"symbol={symbol}"
    tq_out    = out_root / "tradequote" / f"symbol={symbol}"
    greek_out.mkdir(parents=True, exist_ok=True)
    tq_out.mkdir(parents=True, exist_ok=True)

    symbol_dir = _resolve_symbol_dir(csv_root, symbol)
    if symbol_dir is None:
        print(f"  [{symbol}] No historical CSVs found under {csv_root} — skipping")
        return

    # Find all historical weekly CSVs
    csv_files = sorted(symbol_dir.glob(f"{symbol}_historical_*.csv"))
    if not csv_files:
        print(f"  [{symbol}] No historical CSVs found — skipping")
        return
    print(f"  [{symbol}] {len(csv_files)} weekly CSV file(s) found in {symbol_dir}")

    # Load all OI once up front (small files)
    oi_df = _load_oi_for_dir(symbol_dir, symbol)
    has_oi = not oi_df.empty
    if has_oi:
        print(f"  [{symbol}] OI loaded: {len(oi_df):,} rows across {oi_df['_oi_date'].nunique()} dates")
    else:
        print(f"  [{symbol}] No OI files found — open_interest will be NaN")

    n_done = 0
    n_skipped_existing = 0
    for csv_path in csv_files:
        # Extract week key from filename e.g. SPXW_historical_2025-W51_part001.csv
        m = re.search(r'(\d{4}-W\d{2}_part\d{3})', csv_path.name)
        if not m:
            print(f"    SKIP: cannot parse week_key from {csv_path.name}")
            continue
        wk = m.group(1)
        greek_path = greek_out / f"option_greek_all_{wk}_filtered.parquet"
        tq_path = tq_out / f"option_quote_{wk}.parquet"
        if not overwrite and greek_path.exists() and tq_path.exists():
            n_skipped_existing += 1
            continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"    ERROR reading {csv_path.name}: {e}")
            continue

        df.columns = [c.lower().strip() for c in df.columns]
        n_raw = len(df)

        # ── Normalise key columns ─────────────────────────────────────────────
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["strike"]     = pd.to_numeric(df["strike"], errors="coerce")
        df["right"]      = df["right"].astype(str).str.upper().str.strip()

        # trade_date: use query_date if present, else derive from timestamp
        if "query_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["query_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            df["trade_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")

        for col in ["delta", "vega", "bid", "ask"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Filters (same as extract_tier1.py) ───────────────────────────────
        mask = df["bid"].notna() & (df["bid"] != 0) & df["ask"].notna() & (df["ask"] != 0)
        if "delta" in df.columns:
            mask &= df["delta"].abs().between(DELTA_MIN, DELTA_MAX)
        if "vega" in df.columns:
            mask &= df["vega"].abs() > MIN_VEGA
        if "dte" in df.columns:
            df["dte"] = pd.to_numeric(df["dte"], errors="coerce")
            mask &= df["dte"].between(DEFAULT_DTE_MIN, dte_max)

        df = df[mask].copy()
        print(f"    {csv_path.name}: {n_raw:,} → {len(df):,} rows after filter  (week_key={wk})")

        if df.empty:
            continue

        df["week_key"] = wk
        df["symbol"]   = symbol

        # ── Join OI (LEFT JOIN — no row explosion) ────────────────────────────
        if has_oi:
            df = df.merge(
                oi_df,
                left_on=["expiration", "strike", "right", "trade_date"],
                right_on=["expiration", "strike", "right", "_oi_date"],
                how="left",
            ).drop(columns=["_oi_date"], errors="ignore")
        else:
            df["open_interest"] = np.nan

        # ── Build GREEK parquet ───────────────────────────────────────────────
        # Rename fetcher cols → tier1 expected names
        df_g = df.rename(columns={
            "timestamp":        "timestamp",
            "underlying_price": "underlying_price",
            "implied_vol":      "implied_vol",
        }).copy()
        # Add missing greek cols as NaN (rho, epsilon, vomma, etc.)
        for col in GREEK_OUT_COLS:
            if col not in df_g.columns:
                df_g[col] = np.nan
        # underlying_timestamp = same as timestamp (no separate field in fetcher)
        if "underlying_timestamp" not in df.columns:
            df_g["underlying_timestamp"] = df_g["timestamp"]
        greek_cols_present = [c for c in GREEK_OUT_COLS if c in df_g.columns]
        df_g[greek_cols_present].to_parquet(greek_path, index=False)

        # ── Build TQ parquet ──────────────────────────────────────────────────
        # Map OHLCV columns as trade/quote proxies
        df_t = df.rename(columns={
            "timestamp": "quote_timestamp",
            "volume":    "volume",
            "count":     "count",
        }).copy()
        df_t["trade_timestamp"] = df_t["quote_timestamp"]
        df_t["price"]           = df_t.get("vwap", df_t.get("close", np.nan))
        df_t["size"]            = pd.to_numeric(df_t.get("volume", pd.Series(np.nan, index=df_t.index)), errors="coerce")
        df_t["exchange"]        = np.nan
        df_t["sequence"]        = np.nan
        df_t["condition"]       = np.nan
        df_t["bid_condition"]   = np.nan
        for ec in ["ext_condition1", "ext_condition2", "ext_condition3", "ext_condition4"]:
            df_t[ec] = np.nan
        # Keep only rows where price > 0 and size > 0 (same as extract_tier1 TQ filter)
        price_col = pd.to_numeric(df_t["price"], errors="coerce")
        size_col  = pd.to_numeric(df_t["size"],  errors="coerce")
        tq_mask   = price_col.notna() & (price_col > 0) & size_col.notna() & (size_col > 0)
        df_t      = df_t[tq_mask]
        tq_cols_present = [c for c in TQ_OUT_COLS if c in df_t.columns]
        df_t[tq_cols_present].to_parquet(tq_path, index=False)
        n_done += 1

    print(
        f"  [{symbol}] ✅ Done → {out_root} | "
        f"written={n_done}, skipped_existing={n_skipped_existing}, total={len(csv_files)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Historical fetcher CSV → Tier1 parquets (no DuckDB)")
    parser.add_argument("--csv-dir",  required=True,
                        help=("CSV root; supports either flat files at root or nested "
                              "{root}/{SYM}/ with optional OI/ subdir"))
    parser.add_argument("--out-root", default="/workspace/data/tier1_2026_v1",
                        help="Tier1 output root (default: /workspace/data/tier1_2026_v1)")
    parser.add_argument("--symbols",  nargs="+", default=ALL_SYMBOLS)
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild weeks even if output parquet files already exist")
    args = parser.parse_args()

    csv_dir  = Path(args.csv_dir)
    out_root = Path(args.out_root)

    print("=" * 70)
    print("CSV → Tier1 (fetcher v1.4 format, no DuckDB)")
    print(f"  CSV dir  : {csv_dir}")
    print(f"  Out root : {out_root}")
    print(f"  Symbols  : {args.symbols}")
    print("=" * 70)

    for sym in [s.upper() for s in args.symbols]:
        dte_max = DTE_MAX_PER_SYMBOL.get(sym, DEFAULT_DTE_MAX)
        print(f"\n[{sym}] DTE_MAX={dte_max}")
        process_symbol(sym, csv_dir, out_root, dte_max, overwrite=args.overwrite)

    print("\nDONE — tier1 ready for build_tier2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
