#!/usr/bin/env python3
"""
Historical Options Data Collector v1.0
======================================
Fetches 1-month of 1-minute historical options data from Theta Data v3 API.

Based on:
  - theta_fetching_v5.py  (robust patterns: retry, filtering, enrichment)
  - This-week-1.ipynb     (historical endpoint: /v3/option/history/greeks/all)

Theta Data v3 Historical Endpoints:
  /v3/option/list/expirations        -> list all expirations
  /v3/option/history/greeks/all      -> historical greeks  (all strikes, 1 exp, 1 date, 1min)
  /v3/option/history/quote           -> historical quotes  (all strikes, 1 exp, 1 date, 1min)
  /v3/option/history/ohlc            -> historical OHLC    (all strikes, 1 exp, 1 date, 1min)
  /v3/option/history/trade           -> historical trades   (all strikes, 1 exp, 1 date, 1min)

Output: Weekly partitioned CSV files
  Pattern: {SYMBOL}_historical_{YYYY}-W{WW}_part{NNN}.csv

Usage:
  python historical_fetcher.py
  python historical_fetcher.py --symbols SPXW,SPY --days 30 --dte 5
  python historical_fetcher.py --greeks-only --no-resume
"""

import time
import os
import sys
import signal
import json
import argparse
import httpx
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from collections import defaultdict

# ═══════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════

BASE_URL = "http://144.202.59.33:25503"
DEFAULT_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "VIXW", "TLT"]
FORMAT = "csv"
INTERVAL = "1m"
TIMEZONE = ZoneInfo("America/New_York")

DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MAX_DTE = 5
DEFAULT_MAX_ROWS_PER_FILE = 500_000

# API tuning
TIMEOUT = 180
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0       # exponential backoff base
RATE_LIMIT_DELAY = 0.03      # 30 ms between API calls

# ── Native API filters (sent as request params to Theta Data v3) ──
# These are the ONLY filters the API supports natively:
#   symbol, expiration, date, interval, strike ("*" = all), right ("both")
# The API returns ALL strikes/deltas/zeros — no server-side bid/ask/vega filtering.

# ── Write-to-CSV filter (applied locally before writing) ──
# Removes rows where ANY of bid, ask, or vega == 0
# Toggle with --no-filter to keep raw API output for diagnosis
WRITE_FILTER_ZERO_COLS = ["bid", "ask", "vega"]

# Columns to drop (noisy or redundant)
COLUMNS_TO_DROP = [
    "bid_condition", "ask_condition",
    "vera", "speed", "zomma", "dual_gamma", "dual_delta",
    "d1", "d2", "ultima", "color", "veta", "vomma",
    "epsilon", "rho", "iv_error",
]


# ═══════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Historical Options Data Collector v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                    help="Comma-separated symbols (default: SPXW,SPY,QQQ,IWM,VIXW,TLT)")
    p.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                    help="Lookback days from yesterday (default: 30)")
    p.add_argument("--dte", type=int, default=DEFAULT_MAX_DTE,
                    help="Max DTE filter for expirations (default: 5)")
    p.add_argument("--outdir", type=str, default="historical_data",
                    help="Output directory (default: historical_data)")
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS_PER_FILE,
                    help="Max rows per CSV part file (default: 500000)")
    p.add_argument("--no-resume", action="store_true",
                    help="Disable resume (re-fetch everything)")
    p.add_argument("--greeks-only", action="store_true",
                    help="Only fetch greeks (skip quotes & OHLC)")
    p.add_argument("--no-filter", action="store_true",
                    help="Disable ALL filtering (keep zeros, all deltas) for raw data diagnosis")
    p.add_argument("--start-date", type=str, default=None,
                    help="Override start date (YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, default=None,
                    help="Override end date (YYYY-MM-DD)")
    return p.parse_args()


# ═══════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════

def log(msg, level="INFO"):
    ts = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    icons = {
        "ERROR": "❌", "WARN": "⚠️", "SUCCESS": "✅",
        "DEBUG": "🔍", "INFO": "📊", "SKIP": "⏭️",
    }
    icon = icons.get(level, "")
    out = sys.stderr if level == "ERROR" else sys.stdout
    print(f"[{ts}] {icon} {msg}", file=out, flush=True)


# ═══════════════════════════════════════════
# DATE UTILITIES
# ═══════════════════════════════════════════

def parse_date(date_str):
    if not date_str:
        return None
    date_str = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def get_trading_days(start_date, end_date):
    """Return weekdays between start and end (inclusive)."""
    days = []
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def get_iso_week(dt):
    iso = dt.isocalendar()
    return (iso.year, iso.week)


# ═══════════════════════════════════════════
# HTTP CLIENT WITH RETRY (from v5)
# ═══════════════════════════════════════════

def fetch_with_retry(client, url, params, label=""):
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = client.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200 and r.text.strip():
                return r.text
            if r.status_code == 472:
                return None  # no data for this combo
            if r.status_code >= 500 and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                log(f"[{label}] HTTP {r.status_code}, retry {attempt+1}/{MAX_RETRIES} in {delay:.0f}s", "WARN")
                time.sleep(delay)
                continue
            return None
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                log(f"[{label}] {type(e).__name__}, retry {attempt+1}/{MAX_RETRIES} in {delay:.0f}s", "WARN")
                time.sleep(delay)
                continue
            log(f"[{label}] Failed after {MAX_RETRIES+1} attempts: {e}", "ERROR")
            return None
        except Exception as e:
            log(f"[{label}] Unexpected: {e}", "ERROR")
            return None
    return None


def _parse_csv(text):
    if text is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(text))
        df = df.replace([np.inf, -np.inf], np.nan)
        if "strike" in df.columns and "right" in df.columns:
            df = df.dropna(subset=["strike", "right"])
        return df
    except Exception as e:
        log(f"CSV parse error: {e}", "ERROR")
        return pd.DataFrame()


# ═══════════════════════════════════════════
# THETA DATA v3 HISTORICAL FETCHERS
# ═══════════════════════════════════════════

def fetch_expirations(client, symbol):
    """List all available expirations for a symbol."""
    url = f"{BASE_URL}/v3/option/list/expirations"
    text = fetch_with_retry(
        client, url,
        {"symbol": symbol, "format": FORMAT},
        f"{symbol} Exp",
    )
    if text is None:
        return []
    df = pd.read_csv(StringIO(text))
    if "expiration" in df.columns:
        return list(df["expiration"].dropna().astype(str))
    return []


def fetch_historical_greeks(client, symbol, expiration, query_date):
    """All greeks for (symbol, expiration, date) at 1-min intervals."""
    url = f"{BASE_URL}/v3/option/history/greeks/all"
    params = {
        "symbol": symbol, "expiration": expiration,
        "date": query_date.strftime("%Y-%m-%d"),
        "interval": INTERVAL, "format": FORMAT,
    }
    return _parse_csv(fetch_with_retry(client, url, params, f"{symbol} G {expiration} {query_date}"))


def fetch_historical_quotes(client, symbol, expiration, query_date):
    """All quotes for (symbol, expiration, date) at 1-min intervals."""
    url = f"{BASE_URL}/v3/option/history/quote"
    params = {
        "symbol": symbol, "expiration": expiration,
        "date": query_date.strftime("%Y-%m-%d"),
        "interval": INTERVAL, "format": FORMAT,
    }
    return _parse_csv(fetch_with_retry(client, url, params, f"{symbol} Q {expiration} {query_date}"))


def fetch_historical_ohlc(client, symbol, expiration, query_date):
    """All OHLC bars for (symbol, expiration, date) at 1-min intervals."""
    url = f"{BASE_URL}/v3/option/history/ohlc"
    params = {
        "symbol": symbol, "expiration": expiration,
        "date": query_date.strftime("%Y-%m-%d"),
        "interval": INTERVAL, "format": FORMAT,
    }
    return _parse_csv(fetch_with_retry(client, url, params, f"{symbol} O {expiration} {query_date}"))


def fetch_historical_trades(client, symbol, expiration, query_date):
    """All trade summaries for (symbol, expiration, date) at 1-min intervals."""
    url = f"{BASE_URL}/v3/option/history/trade"
    params = {
        "symbol": symbol, "expiration": expiration,
        "date": query_date.strftime("%Y-%m-%d"),
        "interval": INTERVAL, "format": FORMAT,
    }
    return _parse_csv(fetch_with_retry(client, url, params, f"{symbol} T {expiration} {query_date}"))


# ═══════════════════════════════════════════
# FILTERING (adapted from v5)
# ═══════════════════════════════════════════

def diagnose_raw_data(df):
    """Print per-column zero/nan/unique counts on raw API response (no rows removed)."""
    if df.empty:
        return
    diag_cols = ["bid", "ask", "vega", "delta", "gamma", "theta",
                 "implied_volatility", "open_interest"]
    parts = []
    for col in diag_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        n_zero = int((vals == 0).sum())
        n_nan  = int(vals.isna().sum())
        n_uniq = int(vals.nunique())
        if n_zero > 0 or n_nan > 0 or n_uniq <= 1:
            parts.append(f"{col}(zero={n_zero} nan={n_nan} uniq={n_uniq})")
    if parts:
        log(f"  API RAW → {', '.join(parts)}", "DEBUG")


def write_filter(df, skip=False):
    """
    Write-to-CSV filter: remove rows where ANY of bid/ask/vega == 0.
    This is a LOCAL filter — NOT a native Theta Data API filter.
    The API has no server-side filtering for zeros.

    Returns (filtered_df, num_removed).
    If skip=True, returns all rows unfiltered (for raw data diagnosis).
    """
    if df.empty:
        return df, 0

    # Always run diagnostics so you can compare API raw vs filtered
    diagnose_raw_data(df)

    if skip:
        return df, 0

    initial = len(df)
    for col in WRITE_FILTER_ZERO_COLS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            df = df[vals != 0]
    removed = initial - len(df)

    if removed > 0:
        log(f"  CSV filter: {initial:,} → {len(df):,} (removed {removed:,} where bid/ask/vega=0)", "DEBUG")

    return df, removed


# ═══════════════════════════════════════════
# ENRICHMENT (adapted from v5)
# ═══════════════════════════════════════════

def enrich_data(df, query_date, expiration_date):
    """Add DTE, moneyness, mid, spread, cp_sign, spot, dist_atm_pct."""
    if df.empty:
        return df
    df = df.copy()

    # DTE & query_date
    df["dte"] = (expiration_date - query_date).days
    df["query_date"] = query_date.strftime("%Y-%m-%d")

    # Numeric strike
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    # Spot / underlying price
    spot_col = next((c for c in df.columns if "underlying_price" in c.lower()), None)
    if spot_col:
        df["spot"] = pd.to_numeric(df[spot_col], errors="coerce")
        if "strike" in df.columns:
            df["moneyness"] = df["strike"] / df["spot"]
            df["dist_atm_pct"] = ((df["strike"] - df["spot"]) / df["spot"]) * 100

    # Mid / Spread
    bid_col = next((c for c in df.columns if c.lower() == "bid"), None)
    ask_col = next((c for c in df.columns if c.lower() == "ask"), None)
    if bid_col and ask_col:
        bv = pd.to_numeric(df[bid_col], errors="coerce")
        av = pd.to_numeric(df[ask_col], errors="coerce")
        df["mid"] = (bv + av) / 2
        df["spread"] = av - bv
        df["spread_pct"] = (df["spread"] / df["mid"].replace(0, np.nan)) * 100

    # Call/Put sign
    if "right" in df.columns:
        df["cp_sign"] = df["right"].apply(lambda x: 1 if str(x).upper().startswith("C") else -1)

    # Drop noisy columns
    drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop:
        df = df.drop(columns=drop, errors="ignore")

    # Reorder: lead columns first
    lead = [c for c in ["symbol", "expiration", "strike", "right", "dte",
                         "cp_sign", "query_date"] if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest]

    return df


# ═══════════════════════════════════════════
# MERGE HELPER
# ═══════════════════════════════════════════

def _detect_merge_keys(df):
    """Find the best merge keys from a dataframe's columns."""
    candidates = ["strike", "right", "ms_of_day", "timestamp", "expiration"]
    return [c for c in candidates if c in df.columns]


def merge_dataframes(primary, secondary, label=""):
    """Left-merge secondary into primary on shared keys."""
    if secondary.empty:
        return primary
    keys = _detect_merge_keys(primary)
    valid = [k for k in keys if k in secondary.columns]
    if not valid:
        return primary
    try:
        return pd.merge(primary, secondary, on=valid, how="left", suffixes=("", f"_{label}"))
    except Exception as e:
        log(f"Merge error ({label}): {e}", "WARN")
        return primary


# ═══════════════════════════════════════════
# WEEKLY FILE WRITER (improved from notebook)
# ═══════════════════════════════════════════

class WeeklyFileWriter:
    def __init__(self, symbol, outdir, max_rows=500_000):
        self.symbol = symbol
        self.outdir = os.path.join(outdir, symbol)
        os.makedirs(self.outdir, exist_ok=True)
        self.max_rows = max_rows
        self.buffers = defaultdict(list)       # (year, week) -> [df, ...]
        self.row_counts = defaultdict(int)
        self.part_nums = defaultdict(int)
        self.total_written = 0
        self.total_buffered = 0
        self.files_created = []

    def add(self, df, query_date):
        if df.empty:
            return
        key = get_iso_week(query_date)
        self.buffers[key].append(df)
        self.row_counts[key] += len(df)
        self.total_buffered += len(df)
        if self.row_counts[key] >= self.max_rows:
            self.flush_week(*key)

    def flush_week(self, year, week):
        key = (year, week)
        if not self.buffers[key]:
            return
        try:
            combined = pd.concat(self.buffers[key], ignore_index=True)
        except Exception as e:
            log(f"Concat error {year}-W{week:02d}: {e}", "ERROR")
            return

        if combined.empty:
            self.buffers[key], self.row_counts[key] = [], 0
            return

        # Sort
        sort_cols = [c for c in ["query_date", "ms_of_day", "timestamp",
                                  "symbol", "expiration", "strike", "right"]
                     if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols)

        # Write in chunks
        idx = 0
        while idx < len(combined):
            self.part_nums[key] += 1
            part = self.part_nums[key]
            chunk = combined.iloc[idx : idx + self.max_rows]

            fname = f"{self.symbol}_historical_{year}-W{week:02d}_part{part:03d}.csv"
            fpath = os.path.join(self.outdir, fname)
            try:
                chunk.to_csv(fpath, index=False)
                mb = os.path.getsize(fpath) / (1024 * 1024)
                self.total_written += len(chunk)
                self.files_created.append(fname)
                log(f"Wrote {len(chunk):,} rows → {fname} ({mb:.1f} MB)", "SUCCESS")
            except Exception as e:
                log(f"Write error {fname}: {e}", "ERROR")
                break
            idx += self.max_rows

        self.buffers[key], self.row_counts[key] = [], 0

    def flush_all(self):
        for key in sorted(self.buffers.keys()):
            if self.buffers[key]:
                self.flush_week(*key)


# ═══════════════════════════════════════════
# PROGRESS / RESUME TRACKER
# ═══════════════════════════════════════════

class ProgressTracker:
    def __init__(self, filepath, enabled=True):
        self.filepath = filepath
        self.enabled = enabled
        self.done = set()
        self._dirty = 0
        if enabled and os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    self.done = set(json.load(f).get("completed", []))
                log(f"Resumed: {len(self.done)} combos already fetched", "INFO")
            except Exception:
                pass

    def is_done(self, symbol, qdate, exp):
        return self.enabled and f"{symbol}|{qdate}|{exp}" in self.done

    def mark_done(self, symbol, qdate, exp):
        if not self.enabled:
            return
        self.done.add(f"{symbol}|{qdate}|{exp}")
        self._dirty += 1
        if self._dirty >= 50:
            self.save()

    def save(self):
        if not self.enabled:
            return
        try:
            with open(self.filepath, "w") as f:
                json.dump({"completed": sorted(self.done)}, f)
            self._dirty = 0
        except Exception:
            pass


# ═══════════════════════════════════════════
# SIGNAL HANDLING
# ═══════════════════════════════════════════

running = True

def _handle_signal(signum, frame):
    global running
    running = False
    log("Interrupt received — finishing current task & flushing data …", "WARN")

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ═══════════════════════════════════════════
# PER-SYMBOL COLLECTION
# ═══════════════════════════════════════════

def collect_symbol(client, symbol, trading_days, max_dte, outdir,
                   max_rows, progress, greeks_only=False, no_filter=False):
    # 1) Fetch expirations
    raw_exps = fetch_expirations(client, symbol)
    if not raw_exps:
        log(f"[{symbol}] No expirations — check API connection", "ERROR")
        return 0

    exp_dates = sorted({d for s in raw_exps if (d := parse_date(s))})
    log(f"[{symbol}] {len(exp_dates)} unique expirations", "INFO")

    writer = WeeklyFileWriter(symbol, outdir, max_rows)
    total_raw = 0
    total_removed = 0
    days_data = 0
    cur_week = None

    for di, qdate in enumerate(trading_days, 1):
        if not running:
            break

        # Valid expirations within DTE
        valid_exps = [e for e in exp_dates if 0 <= (e - qdate).days <= max_dte]
        if not valid_exps:
            continue

        # Week header
        yr, wk = get_iso_week(qdate)
        if wk != cur_week:
            if cur_week is not None:
                print()
            print(f"\n=== {symbol}  Week {yr}-W{wk:02d} ===")
            cur_week = wk

        day_rows = 0
        day_removed = 0
        day_exps = 0

        for edate in valid_exps:
            if not running:
                break
            estr = edate.strftime("%Y-%m-%d")
            if progress.is_done(symbol, str(qdate), estr):
                continue

            # ---- Fetch greeks (primary, confirmed endpoint) ----
            gdf = fetch_historical_greeks(client, symbol, estr, qdate)
            time.sleep(RATE_LIMIT_DELAY)

            if gdf.empty:
                progress.mark_done(symbol, str(qdate), estr)
                continue

            merged = gdf

            # ---- Optionally fetch quotes + OHLC ----
            if not greeks_only:
                qdf = fetch_historical_quotes(client, symbol, estr, qdate)
                time.sleep(RATE_LIMIT_DELAY)
                merged = merge_dataframes(merged, qdf, "q")

                odf = fetch_historical_ohlc(client, symbol, estr, qdate)
                time.sleep(RATE_LIMIT_DELAY)
                merged = merge_dataframes(merged, odf, "ohlc")

            merged["symbol"] = symbol

            # ---- Enrich & filter ----
            merged = enrich_data(merged, qdate, edate)
            filtered, removed = write_filter(merged, skip=no_filter)

            total_raw += len(merged)
            total_removed += removed

            if not filtered.empty:
                writer.add(filtered, qdate)
                day_rows += len(filtered)
                day_removed += removed
                day_exps += 1

            progress.mark_done(symbol, str(qdate), estr)

        # Day summary line
        if day_rows > 0:
            days_data += 1
            buf_total = writer.total_buffered
            filt_info = "" if no_filter else f" (-{day_removed:,} filtered)"
            print(f"  [{di:3d}/{len(trading_days)}] {qdate} | "
                  f"{day_exps} exps | {day_rows:>8,} rows"
                  f"{filt_info} | buf {buf_total:,}",
                  flush=True)
        else:
            print(f"  [{di:3d}/{len(trading_days)}] {qdate} | no data", flush=True)

    # Flush
    log(f"[{symbol}] Flushing remaining buffers …", "INFO")
    writer.flush_all()
    progress.save()

    # Symbol summary
    print()
    log(f"[{symbol}] ── DONE ──", "SUCCESS")
    log(f"  Trading days with data : {days_data}/{len(trading_days)}", "INFO")
    log(f"  Total raw rows         : {total_raw:,}", "INFO")
    log(f"  Filtered out           : {total_removed:,}", "INFO")
    log(f"  Rows written to disk   : {writer.total_written:,}", "INFO")
    log(f"  CSV files created      : {len(writer.files_created)}", "INFO")
    for fn in writer.files_created:
        log(f"    {fn}", "INFO")
    return writer.total_written


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Date range
    today = datetime.now(TIMEZONE).date()
    end_date = parse_date(args.end_date) if args.end_date else today - timedelta(days=1)
    start_date = parse_date(args.start_date) if args.start_date else end_date - timedelta(days=args.days - 1)
    trading_days = get_trading_days(start_date, end_date)

    progress = ProgressTracker(
        os.path.join(outdir, ".progress.json"),
        enabled=not args.no_resume,
    )

    # ── Banner ──
    dtype_str = "greeks only" if args.greeks_only else "greeks + quotes + OHLC"
    print("\n" + "=" * 70)
    print("  Historical Options Data Collector v1.0")
    print("=" * 70)
    print(f"  Symbols      : {', '.join(symbols)}")
    print(f"  Date range   : {start_date} → {end_date}  ({len(trading_days)} trading days)")
    print(f"  Max DTE      : {args.dte}")
    print(f"  Interval     : {INTERVAL}")
    print(f"  Data types   : {dtype_str}")
    print(f"  Output       : {os.path.abspath(outdir)}/")
    print(f"  Max rows/file: {args.max_rows:,}")
    print(f"  Resume       : {'ON' if not args.no_resume else 'OFF'}")
    print(f"  Native API   : symbol + expiration + date + interval (no zero filtering)")
    csv_filt = "OFF (raw API data, diagnostics only)" if args.no_filter else f"remove rows where {' or '.join(WRITE_FILTER_ZERO_COLS)} = 0"
    print(f"  CSV filter   : {csv_filt}")
    print("=" * 70 + "\n")

    grand_total = 0

    with httpx.Client() as client:
        for sym in symbols:
            if not running:
                break
            log(f"[{sym}] ── Starting collection ──", "INFO")
            rows = collect_symbol(
                client, sym, trading_days, args.dte,
                outdir, args.max_rows, progress,
                greeks_only=args.greeks_only,
                no_filter=args.no_filter,
            )
            grand_total += rows
            print()

    # ── Final report ──
    print("\n" + "=" * 70)
    print("  FILES CREATED")
    print("=" * 70)
    try:
        grand_files, grand_mb = 0, 0.0
        for sym in symbols:
            sym_dir = os.path.join(outdir, sym)
            if not os.path.isdir(sym_dir):
                continue
            csvs = sorted(f for f in os.listdir(sym_dir) if f.endswith(".csv"))
            if not csvs:
                continue
            sym_mb = sum(os.path.getsize(os.path.join(sym_dir, f)) for f in csvs) / (1024*1024)
            grand_files += len(csvs)
            grand_mb += sym_mb
            print(f"  [{sym}]  {len(csvs)} file(s), {sym_mb:.1f} MB  →  {sym_dir}/")
            week_files = defaultdict(list)
            for fn in csvs:
                parts = fn.split("_historical_")
                wk_key = parts[1].split("_part")[0] if len(parts) > 1 else "other"
                week_files[wk_key].append(fn)
            for wk in sorted(week_files):
                print(f"      {wk}: {len(week_files[wk])} file(s)")
        print(f"\n  Total : {grand_files} files, {grand_mb:.1f} MB")
    except Exception as e:
        log(f"Listing error: {e}", "ERROR")

    print(f"  Dir   : {os.path.abspath(outdir)}/")
    print(f"  Rows  : {grand_total:,}")
    print("\n" + "=" * 70)
    print("  COLLECTION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        log(f"Fatal: {e}", "ERROR")
        import traceback
        traceback.print_exc()
