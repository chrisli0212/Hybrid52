#!/usr/bin/env python3
"""
Historical Options Data Collector v1.3
=======================================
Fetches 1-minute historical options data from Theta Data v3 API.
Designed for ML training datasets (full distribution, no junk rows).

v1.3 additions (vs v1.2):
  - Added /v3/option/history/implied_volatility endpoint
  - Added /v3/option/history/trade_iv endpoint
  - Added --smoke-redundancy-test: probes whether greeks/all, implied_volatility,
    and first/second/third-order endpoints return overlapping or unique columns,
    so you can decide whether to add more endpoints in future runs
  - iv_col (implied_vol) disambiguated: if IV from implied_volatility endpoint
    differs from greeks/all, both are preserved as implied_vol_iv and implied_vol

v1.2 fixes (vs v1.1):
  - REMOVED rth=true from all endpoint params (caused 100% empty responses)
  - Reverted to exact working params: symbol, expiration, date, interval, format

v1.1 changes (vs v1.0):
  - Safety lag: default end_date = today - 2
  - US market holiday exclusion in get_trading_days()
  - Schema cleanup: dropped sparse/redundant trade tick columns
  - Optional stricter filters: --min-bid, --max-spread-pct
  - Lightweight cached plan probe (~2s) replaces full smoke test
  - Stronger chunk-skip and progress validation
  - _staggered() at module level (no nested closures)
  - enrich_data no longer creates redundant 'spot' column

Theta Data v3 Historical Endpoints used:
  /v3/option/list/expirations             -> list all expirations
  /v3/option/history/greeks/all           -> all greeks (1st+2nd+3rd order) + IV + underlying
  /v3/option/history/quote                -> bid/ask/size/exchange
  /v3/option/history/ohlc                 -> open/high/low/close/volume/count/vwap
  /v3/option/history/trade                -> 1-min aggregated trade bars
  /v3/option/history/trade_greeks         -> greeks at trade execution (Pro only)
  /v3/option/history/implied_volatility   -> IV surface timeseries (may add columns vs greeks/all)
  /v3/option/history/trade_iv             -> IV computed at actual trade execution price
  /v3/option/history/open_interest        -> EOD open interest (written to separate OI folder)

Output:
  1-min CSVs : {outdir}/{symbol}/{symbol}_historical_{YYYY}-W{WW}_part{NNN}.csv
  OI (daily) : {outdir}/{symbol}/OI/{symbol}_oi_{YYYY-MM-DD}.csv

Usage:
  python historical_fetcher_final.py
  python historical_fetcher_final.py --symbols SPXW,SPY --days 365 --dte 5
  python historical_fetcher_final.py --start-date 2025-03-19 --end-date 2026-03-17
  python historical_fetcher_final.py --smoke-redundancy-test   # probe endpoint overlap
  python historical_fetcher_final.py --min-bid 0.10 --max-spread-pct 80
"""

import os
import sys
import signal
import json
import asyncio
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

DEFAULT_LOOKBACK_DAYS = 700
DEFAULT_MAX_DTE = 5
DEFAULT_MAX_ROWS_PER_FILE = 500_000
DEFAULT_LAG_DAYS = 2

TIMEOUT = 180
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

# FREE/VALUE=1, STANDARD=2, PRO=4
HTTP_CONCURRENCY = 4

CHUNK_WEEKS = 5
CHUNK_PAUSE_SECS = 30
ENDPOINT_STAGGER_SECS = 0.15

WRITE_FILTER_ZERO_COLS = ["bid", "ask", "vega"]

COLUMNS_TO_DROP = [
    "bid_condition", "ask_condition",
    "vera", "speed", "zomma", "dual_gamma", "dual_delta",
    "d1", "d2", "ultima", "color", "veta", "vomma",
    "epsilon", "rho", "iv_error",
]

POST_MERGE_DROP = [
    "bid_q", "ask_q",
    "symbol_q", "symbol_ohlc", "symbol_trade", "symbol_iv", "symbol_tiv",
    "underlying_timestamp",
    "sequence", "ext_condition1", "ext_condition2",
    "ext_condition3", "ext_condition4",
    "condition", "condition_trade",
    "size", "size_trade",
    "exchange", "exchange_trade",
    "price", "price_trade",
]

PROBE_CACHE_FILE = ".probe_cache.json"


# ═══════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Historical Options Data Collector v1.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    p.add_argument("--dte", type=int, default=DEFAULT_MAX_DTE)
    p.add_argument("--outdir", type=str, default="historical_data")
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS_PER_FILE)
    p.add_argument("--lag-days", type=int, default=DEFAULT_LAG_DAYS,
                   help="Safety lag: end_date = today - N (default: 2)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--greeks-only", action="store_true")
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--skip-trade-greeks", action="store_true")
    p.add_argument("--skip-iv", action="store_true",
                   help="Skip implied_volatility endpoint (if redundant with greeks/all)")
    p.add_argument("--skip-trade-iv", action="store_true",
                   help="Skip trade_iv endpoint")
    p.add_argument("--min-bid", type=float, default=None)
    p.add_argument("--max-spread-pct", type=float, default=None)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--smoke-redundancy-test", action="store_true",
                   help="Probe endpoint column overlap then exit (no data collection)")
    return p.parse_args()


# ═══════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════

def log(msg, level="INFO"):
    ts = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    icons = {"ERROR": "❌", "WARN": "⚠️", "SUCCESS": "✅",
             "DEBUG": "🔍", "INFO": "📊", "SKIP": "⏭️"}
    out = sys.stderr if level == "ERROR" else sys.stdout
    print(f"[{ts}] {icons.get(level,'')} {msg}", file=out, flush=True)


# ═══════════════════════════════════════════
# DATE UTILITIES (US market holiday aware)
# ═══════════════════════════════════════════

def _easter(year):
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)

def _us_market_holidays(year):
    holidays = set()
    def obs(d):
        if d.weekday() == 5: return d - timedelta(days=1)
        if d.weekday() == 6: return d + timedelta(days=1)
        return d
    for mo, dy in [(1,1),(6,19),(7,4),(12,25)]:
        holidays.add(obs(date(year, mo, dy)))
    jan1 = date(year, 1, 1)
    mlk = jan1 + timedelta(days=(7 - jan1.weekday()) % 7) + timedelta(weeks=2)
    holidays.add(mlk)
    feb1 = date(year, 2, 1)
    pres = feb1 + timedelta(days=(7 - feb1.weekday()) % 7) + timedelta(weeks=2)
    holidays.add(pres)
    holidays.add(_easter(year) - timedelta(days=2))
    may31 = date(year, 5, 31)
    holidays.add(may31 - timedelta(days=(may31.weekday() - 0) % 7))
    sep1 = date(year, 9, 1)
    holidays.add(sep1 + timedelta(days=(7 - sep1.weekday()) % 7))
    nov1 = date(year, 11, 1)
    first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
    holidays.add(first_thu + timedelta(weeks=3))
    return holidays

def get_trading_days(start_date, end_date):
    all_holidays = set()
    for y in range(start_date.year, end_date.year + 1):
        all_holidays |= _us_market_holidays(y)
    days, cur = [], start_date
    while cur <= end_date:
        if cur.weekday() < 5 and cur not in all_holidays:
            days.append(cur)
        cur += timedelta(days=1)
    return days

def parse_date(date_str):
    if not date_str: return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try: return datetime.strptime(str(date_str).strip(), fmt).date()
        except (ValueError, AttributeError): pass
    return None

def get_iso_week(dt):
    iso = dt.isocalendar()
    return (iso.year, iso.week)

def get_date_chunks(trading_days, chunk_weeks=CHUNK_WEEKS):
    if not trading_days: return []
    chunk_size = chunk_weeks * 5
    return [trading_days[i:i+chunk_size] for i in range(0, len(trading_days), chunk_size)]


# ═══════════════════════════════════════════
# ASYNC HTTP WITH RETRY
# ═══════════════════════════════════════════

def _parse_csv(text):
    if text is None: return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(text))
        df = df.replace([np.inf, -np.inf], np.nan)
        if "strike" in df.columns and "right" in df.columns:
            df = df.dropna(subset=["strike", "right"])
        return df
    except Exception as e:
        log(f"CSV parse error: {e}", "ERROR")
        return pd.DataFrame()

async def fetch_with_retry(client, sem, url, params, label=""):
    async with sem:
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = await client.get(url, params=params, timeout=TIMEOUT)
                if r.status_code == 200 and r.text.strip():
                    return r.text
                if r.status_code == 472:
                    return None
                if r.status_code == 429:
                    delay = RETRY_BASE_DELAY * (4 ** attempt)
                    log(f"[{label}] 429 rate limited, backing off {delay:.0f}s", "WARN")
                    await asyncio.sleep(delay); continue
                if r.status_code >= 500 and attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    log(f"[{label}] HTTP {r.status_code}, retry {attempt+1}/{MAX_RETRIES}", "WARN")
                    await asyncio.sleep(delay); continue
                return None
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt)); continue
                log(f"[{label}] Failed: {e}", "ERROR"); return None
            except Exception as e:
                log(f"[{label}] Unexpected: {e}", "ERROR"); return None
        return None


# ═══════════════════════════════════════════
# STAGGER HELPER (module-level, no closures)
# ═══════════════════════════════════════════

async def _staggered(coro, delay):
    await asyncio.sleep(delay)
    return await coro


# ═══════════════════════════════════════════
# THETA DATA v3 FETCHERS
# NOTE: No rth=true — not supported by historical endpoints, causes empty responses
# Working params: symbol, expiration, date, interval, format
# ═══════════════════════════════════════════

def _hist_params(symbol, expiration, query_date, interval=True):
    p = {"symbol": symbol, "expiration": expiration,
         "date": query_date.strftime("%Y-%m-%d"), "format": FORMAT}
    if interval:
        p["interval"] = INTERVAL
    return p

async def fetch_expirations(client, sem, symbol):
    url = f"{BASE_URL}/v3/option/list/expirations"
    text = await fetch_with_retry(client, sem, url,
                                   {"symbol": symbol, "format": FORMAT}, f"{symbol} Exp")
    if text is None: return []
    df = pd.read_csv(StringIO(text))
    return list(df["expiration"].dropna().astype(str)) if "expiration" in df.columns else []

async def fetch_historical_greeks(client, sem, symbol, expiration, query_date):
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/greeks/all",
        _hist_params(symbol, expiration, query_date),
        f"{symbol} G {expiration} {query_date}"))

async def fetch_historical_quotes(client, sem, symbol, expiration, query_date):
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/quote",
        _hist_params(symbol, expiration, query_date),
        f"{symbol} Q {expiration} {query_date}"))

async def fetch_historical_ohlc(client, sem, symbol, expiration, query_date):
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/ohlc",
        _hist_params(symbol, expiration, query_date),
        f"{symbol} O {expiration} {query_date}"))

async def fetch_historical_trades(client, sem, symbol, expiration, query_date):
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/trade",
        _hist_params(symbol, expiration, query_date),
        f"{symbol} T {expiration} {query_date}"))

async def fetch_historical_trade_greeks(client, sem, symbol, expiration, query_date):
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/trade_greeks",
        _hist_params(symbol, expiration, query_date),
        f"{symbol} TG {expiration} {query_date}"))

async def fetch_historical_implied_volatility(client, sem, symbol, expiration, query_date):
    """IV surface timeseries — may contain iv_bid/iv_ask/iv_mid not in greeks/all."""
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/implied_volatility",
        _hist_params(symbol, expiration, query_date),
        f"{symbol} IV {expiration} {query_date}"))

async def fetch_historical_trade_iv(client, sem, symbol, expiration, query_date):
    """IV computed at actual trade execution price — unique data not in greeks/all."""
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/trade_iv",
        _hist_params(symbol, expiration, query_date),
        f"{symbol} TIV {expiration} {query_date}"))

async def fetch_historical_open_interest(client, sem, symbol, expiration, query_date):
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/open_interest",
        _hist_params(symbol, expiration, query_date, interval=False),
        f"{symbol} OI {expiration} {query_date}"))


# ═══════════════════════════════════════════
# SMOKE REDUNDANCY TEST
# Probes which columns each endpoint returns vs greeks/all.
# Answers: "Is implied_volatility / trade_iv adding NEW columns?"
# Run once with: --smoke-redundancy-test
# ═══════════════════════════════════════════

async def run_smoke_redundancy_test(client, sem, symbol, dte):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  ENDPOINT REDUNDANCY TEST  symbol={symbol}")
    print(sep)

    # Get a recent valid probe date
    today = datetime.now(TIMEZONE).date()
    probe_date = today - timedelta(days=DEFAULT_LAG_DAYS)
    while probe_date.weekday() >= 5:
        probe_date -= timedelta(days=1)

    raw_exps = await fetch_expirations(client, sem, symbol)
    if not raw_exps:
        log("No expirations — check API connection", "ERROR"); return
    exp_dates = sorted({d for s in raw_exps if (d := parse_date(s))})
    valid = [e for e in exp_dates if probe_date <= e <= probe_date + timedelta(days=dte)]
    if not valid:
        valid = [e for e in exp_dates if e >= probe_date]
    if not valid:
        log("No valid expirations for probe date", "ERROR"); return

    estr = valid[0].strftime("%Y-%m-%d")
    print(f"\nProbe: date={probe_date}  expiration={estr}  DTE={(valid[0]-probe_date).days}\n")

    # Fetch all endpoints in parallel
    endpoints = {
        "greeks/all":           fetch_historical_greeks(client, sem, symbol, estr, probe_date),
        "quote":                fetch_historical_quotes(client, sem, symbol, estr, probe_date),
        "ohlc":                 fetch_historical_ohlc(client, sem, symbol, estr, probe_date),
        "trade":                fetch_historical_trades(client, sem, symbol, estr, probe_date),
        "trade_greeks":         fetch_historical_trade_greeks(client, sem, symbol, estr, probe_date),
        "implied_volatility":   fetch_historical_implied_volatility(client, sem, symbol, estr, probe_date),
        "trade_iv":             fetch_historical_trade_iv(client, sem, symbol, estr, probe_date),
    }
    names = list(endpoints.keys())
    results = await asyncio.gather(*endpoints.values(), return_exceptions=True)
    dfs = {name: (r if isinstance(r, pd.DataFrame) else pd.DataFrame())
           for name, r in zip(names, results)}

    # Reference columns from greeks/all
    greeks_cols = set(dfs["greeks/all"].columns) if not dfs["greeks/all"].empty else set()

    print(f"{'Endpoint':<26} {'Rows':>8}  {'Cols':>5}  Status    Unique cols vs greeks/all")
    print("-" * 90)
    for name, df in dfs.items():
        if df.empty:
            print(f"  {name:<24} {'—':>8}  {'—':>5}  ❌ EMPTY")
            continue
        cols = set(df.columns)
        unique = cols - greeks_cols - {"ms_of_day", "date", "strike", "right", "expiration"}
        status = "✅ DATA"
        if name == "greeks/all":
            unique_str = "(baseline)"
        else:
            unique_str = ", ".join(sorted(unique)) if unique else "⚠️  NO new columns (fully redundant)"
        print(f"  {name:<24} {len(df):>8,}  {len(cols):>5}  {status}    {unique_str}")

    print(f"\n{'─'*70}")
    print("VERDICT:")

    iv_df = dfs["implied_volatility"]
    if not iv_df.empty:
        iv_unique = set(iv_df.columns) - greeks_cols - {"ms_of_day","date","strike","right","expiration"}
        if iv_unique:
            print(f"  ✅ implied_volatility adds NEW columns: {sorted(iv_unique)}")
            print(f"     → KEEP fetching implied_volatility (adds value)")
        else:
            print(f"  ⚠️  implied_volatility has NO new columns vs greeks/all")
            iv_col = "implied_vol" if "implied_vol" in iv_df.columns else (list(iv_df.columns)[0] if not iv_df.empty else "?")
            # Check if IV values differ
            if "implied_vol" in dfs["greeks/all"].columns and "implied_vol" in iv_df.columns:
                g_iv = dfs["greeks/all"]["implied_vol"].dropna()
                iv_iv = iv_df["implied_vol"].dropna()
                if len(g_iv) > 0 and len(iv_iv) > 0:
                    corr = g_iv.corr(iv_iv[:len(g_iv)]) if len(g_iv)==len(iv_iv) else None
                    if corr and corr > 0.999:
                        print(f"     → SKIP with --skip-iv (values identical, corr={corr:.4f})")
                    else:
                        print(f"     → Values DIFFER (corr={corr}) — may be computed differently, keep fetching")
            else:
                print(f"     → Use --skip-iv to omit this endpoint")
    else:
        print(f"  ❌ implied_volatility returned empty — endpoint may not be available on your plan")

    tiv_df = dfs["trade_iv"]
    if not tiv_df.empty:
        tiv_unique = set(tiv_df.columns) - greeks_cols - {"ms_of_day","date","strike","right","expiration"}
        if tiv_unique:
            print(f"  ✅ trade_iv adds NEW columns: {sorted(tiv_unique)}")
        else:
            print(f"  ✅ trade_iv data unique (trade execution IV — different rows than greeks/all)")
        print(f"     → KEEP fetching trade_iv ({len(tiv_df):,} rows)")
    else:
        print(f"  ❌ trade_iv returned empty — may need Pro plan or use --skip-trade-iv")

    print(f"\n  Recommended flags based on above:")
    flags = []
    if iv_df.empty: flags.append("--skip-iv")
    if tiv_df.empty: flags.append("--skip-trade-iv")
    if dfs["trade_greeks"].empty: flags.append("--skip-trade-greeks")
    print(f"    {' '.join(flags) if flags else 'No flags needed — all endpoints returning data'}")
    print(f"\n{sep}\n")


# ═══════════════════════════════════════════
# FILTERING
# ═══════════════════════════════════════════

def diagnose_raw_data(df):
    if df.empty: return
    diag_cols = ["bid","ask","vega","delta","gamma","theta","implied_vol"]
    parts = []
    for col in diag_cols:
        if col not in df.columns: continue
        vals = pd.to_numeric(df[col], errors="coerce")
        n0, nn, nu = int((vals==0).sum()), int(vals.isna().sum()), int(vals.nunique())
        if n0 > 0 or nn > 0 or nu <= 1:
            parts.append(f"{col}(zero={n0} nan={nn} uniq={nu})")
    if parts: log(f"  API RAW → {', '.join(parts)}", "DEBUG")

def write_filter(df, skip=False, min_bid=None, max_spread_pct=None):
    if df.empty: return df, 0
    diagnose_raw_data(df)
    if skip: return df, 0
    initial = len(df)
    for col in WRITE_FILTER_ZERO_COLS:
        if col in df.columns:
            df = df[pd.to_numeric(df[col], errors="coerce") != 0]
    if min_bid is not None and "bid" in df.columns:
        df = df[pd.to_numeric(df["bid"], errors="coerce") >= min_bid]
    if max_spread_pct is not None and "spread_pct" in df.columns:
        sp = pd.to_numeric(df["spread_pct"], errors="coerce")
        df = df[(sp <= max_spread_pct) | sp.isna()]
    removed = initial - len(df)
    if removed > 0:
        log(f"  CSV filter: {initial:,} → {len(df):,} (removed {removed:,})", "DEBUG")
    return df, removed


# ═══════════════════════════════════════════
# ENRICHMENT
# ═══════════════════════════════════════════

def enrich_data(df, query_date, expiration_date):
    if df.empty: return df
    df = df.copy()
    df["dte"] = (expiration_date - query_date).days
    df["query_date"] = query_date.strftime("%Y-%m-%d")
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    up = pd.to_numeric(df["underlying_price"], errors="coerce") if "underlying_price" in df.columns else None
    if up is not None and "strike" in df.columns:
        df["moneyness"] = df["strike"] / up
        df["dist_atm_pct"] = ((df["strike"] - up) / up) * 100
    if "bid" in df.columns and "ask" in df.columns:
        bv = pd.to_numeric(df["bid"], errors="coerce")
        av = pd.to_numeric(df["ask"], errors="coerce")
        df["mid"] = (bv + av) / 2
        df["spread"] = av - bv
        df["spread_pct"] = (df["spread"] / df["mid"].replace(0, np.nan)) * 100
    if "right" in df.columns:
        df["cp_sign"] = df["right"].apply(lambda x: 1 if str(x).upper().startswith("C") else -1)
    drop_all = [c for c in COLUMNS_TO_DROP + POST_MERGE_DROP if c in df.columns]
    if drop_all: df = df.drop(columns=drop_all, errors="ignore")
    lead = [c for c in ["symbol","expiration","strike","right","dte","cp_sign","query_date"]
            if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    return df


# ═══════════════════════════════════════════
# MERGE HELPER
# ═══════════════════════════════════════════

def merge_dataframes(primary, secondary, label=""):
    if secondary.empty: return primary
    candidates = ["strike", "right", "ms_of_day", "timestamp", "expiration"]
    keys = [c for c in candidates if c in primary.columns and c in secondary.columns]
    if not keys: return primary
    try:
        return pd.merge(primary, secondary, on=keys, how="left", suffixes=("", f"_{label}"))
    except Exception as e:
        log(f"Merge error ({label}): {e}", "WARN")
        return primary


# ═══════════════════════════════════════════
# WEEKLY FILE WRITER
# ═══════════════════════════════════════════

class WeeklyFileWriter:
    def __init__(self, symbol, outdir, max_rows=500_000):
        self.symbol = symbol
        self.outdir = os.path.join(outdir, symbol)
        os.makedirs(self.outdir, exist_ok=True)
        self.max_rows = max_rows
        self.buffers = defaultdict(list)
        self.row_counts = defaultdict(int)
        self.part_nums = defaultdict(int)
        self.total_written = 0
        self.total_buffered = 0
        self.files_created = []

    def add(self, df, query_date):
        if df.empty: return
        key = get_iso_week(query_date)
        self.buffers[key].append(df)
        self.row_counts[key] += len(df)
        self.total_buffered += len(df)
        if self.row_counts[key] >= self.max_rows:
            self.flush_week(*key)

    def flush_week(self, year, week):
        key = (year, week)
        if not self.buffers[key]: return
        try:
            combined = pd.concat(self.buffers[key], ignore_index=True)
        except Exception as e:
            log(f"Concat error {year}-W{week:02d}: {e}", "ERROR"); return
        if combined.empty:
            self.buffers[key], self.row_counts[key] = [], 0; return
        sort_cols = [c for c in ["query_date","ms_of_day","timestamp",
                                  "symbol","expiration","strike","right"]
                     if c in combined.columns]
        if sort_cols: combined = combined.sort_values(sort_cols)
        idx = 0
        while idx < len(combined):
            self.part_nums[key] += 1
            chunk = combined.iloc[idx : idx + self.max_rows]
            fname = f"{self.symbol}_historical_{year}-W{week:02d}_part{self.part_nums[key]:03d}.csv"
            fpath = os.path.join(self.outdir, fname)
            try:
                chunk.to_csv(fpath, index=False)
                mb = os.path.getsize(fpath) / (1024*1024)
                self.total_written += len(chunk)
                self.files_created.append(fname)
                log(f"Wrote {len(chunk):,} rows → {fname} ({mb:.1f} MB)", "SUCCESS")
            except Exception as e:
                log(f"Write error {fname}: {e}", "ERROR"); break
            idx += self.max_rows
        self.buffers[key], self.row_counts[key] = [], 0

    def flush_all(self):
        for key in sorted(self.buffers.keys()):
            if self.buffers[key]: self.flush_week(*key)


# ═══════════════════════════════════════════
# CHUNK SKIP HELPER (disk + progress validated)
# ═══════════════════════════════════════════

def get_covered_weeks(symbol, outdir):
    sym_dir = os.path.join(outdir, symbol)
    if not os.path.isdir(sym_dir): return set()
    covered = set()
    for fname in os.listdir(sym_dir):
        if not fname.endswith(".csv"): continue
        try:
            week_part = fname.split("_historical_")[1].split("_part")[0]
            year, wk = week_part.split("-W")
            covered.add((int(year), int(wk)))
        except (IndexError, ValueError): continue
    return covered

def chunk_already_done(chunk_days, symbol, outdir, progress):
    covered = get_covered_weeks(symbol, outdir)
    if not {get_iso_week(d) for d in chunk_days}.issubset(covered):
        return False
    prefix = f"{symbol}|"
    day_strs = {str(d) for d in chunk_days}
    done_days = {e.split("|")[1] for e in progress.done
                 if e.startswith(prefix) and len(e.split("|")) >= 2}
    return day_strs.issubset(done_days)


# ═══════════════════════════════════════════
# DAILY OI WRITER
# ═══════════════════════════════════════════

class DailyOIWriter:
    def __init__(self, symbol, outdir):
        self.symbol = symbol
        self.oi_dir = os.path.join(outdir, symbol, "OI")
        os.makedirs(self.oi_dir, exist_ok=True)
        self.daily_frames = defaultdict(list)

    def add(self, df, query_date):
        if df.empty: return
        df = df.copy()
        df["query_date"] = query_date.strftime("%Y-%m-%d")
        df["symbol"] = self.symbol
        self.daily_frames[str(query_date)].append(df)

    def flush_day(self, query_date):
        key = str(query_date)
        if not self.daily_frames[key]: return
        try:
            combined = pd.concat(self.daily_frames[key], ignore_index=True)
            sort_cols = [c for c in ["expiration","strike","right"] if c in combined.columns]
            if sort_cols: combined = combined.sort_values(sort_cols)
            fname = f"{self.symbol}_oi_{key}.csv"
            combined.to_csv(os.path.join(self.oi_dir, fname), index=False)
            mb = os.path.getsize(os.path.join(self.oi_dir, fname)) / (1024*1024)
            log(f"[OI] {fname} — {len(combined):,} rows ({mb:.2f} MB)", "SUCCESS")
        except Exception as e:
            log(f"[OI] Write error {query_date}: {e}", "ERROR")
        finally:
            self.daily_frames[key] = []

    def flush_all(self):
        for key in sorted(self.daily_frames.keys()):
            if self.daily_frames[key]: self.flush_day(key)


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
            except Exception: pass

    def is_done(self, symbol, qdate, exp):
        return self.enabled and f"{symbol}|{qdate}|{exp}" in self.done

    def validate_against_disk(self, outdir, symbols):
        if not self.enabled: return
        removed = 0
        for sym in symbols:
            covered = get_covered_weeks(sym, outdir)
            stale = set()
            for entry in self.done:
                if not entry.startswith(f"{sym}|"): continue
                parts = entry.split("|")
                if len(parts) >= 2:
                    d = parse_date(parts[1])
                    if d and get_iso_week(d) not in covered:
                        stale.add(entry)
            self.done -= stale
            removed += len(stale)
        if removed:
            log(f"Progress validation: removed {removed} stale entries", "WARN")
            self._dirty += removed

    def mark_done(self, symbol, qdate, exp):
        if not self.enabled: return
        self.done.add(f"{symbol}|{qdate}|{exp}")
        self._dirty += 1
        if self._dirty >= 50: self.save()

    def save(self):
        if not self.enabled: return
        try:
            with open(self.filepath, "w") as f:
                json.dump({"completed": sorted(self.done)}, f)
            self._dirty = 0
        except Exception: pass


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
# LIGHTWEIGHT PLAN PROBE (cached 7 days)
# ═══════════════════════════════════════════

async def run_plan_probe(client, sem, symbol, outdir):
    cache_path = os.path.join(outdir, PROBE_CACHE_FILE)
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            age_days = (datetime.now().timestamp() - cache.get("ts", 0)) / 86400
            if age_days < 7:
                log(f"Plan probe from cache (age={age_days:.1f}d): "
                    f"skip_tg={cache.get('skip_trade_greeks')} "
                    f"skip_iv={cache.get('skip_iv')} "
                    f"skip_tiv={cache.get('skip_trade_iv')}", "INFO")
                return cache
        except Exception: pass

    log(f"Plan probe: checking endpoint availability on {symbol} ...", "INFO")
    raw_exps = await fetch_expirations(client, sem, symbol)
    if not raw_exps:
        return {"skip_trade_greeks": True, "skip_iv": False, "skip_trade_iv": False}

    today = datetime.now(TIMEZONE).date()
    probe_date = today - timedelta(days=DEFAULT_LAG_DAYS)
    while probe_date.weekday() >= 5:
        probe_date -= timedelta(days=1)

    exp_dates = sorted({d for s in raw_exps if (d := parse_date(s))})
    valid = [e for e in exp_dates if probe_date <= e <= probe_date + timedelta(days=DEFAULT_MAX_DTE)]
    if not valid:
        valid = [e for e in exp_dates if e >= probe_date]
    if not valid:
        return {"skip_trade_greeks": True, "skip_iv": False, "skip_trade_iv": False}

    estr = valid[0].strftime("%Y-%m-%d")

    tg_df, iv_df, tiv_df = await asyncio.gather(
        fetch_historical_trade_greeks(client, sem, symbol, estr, probe_date),
        fetch_historical_implied_volatility(client, sem, symbol, estr, probe_date),
        fetch_historical_trade_iv(client, sem, symbol, estr, probe_date),
    )

    result = {
        "skip_trade_greeks": tg_df.empty,
        "skip_iv": False,       # default: fetch it (let user decide via --smoke-redundancy-test)
        "skip_trade_iv": tiv_df.empty,
        "ts": datetime.now().timestamp(),
    }

    parts = []
    parts.append(f"trade_greeks={'❌ unavailable' if tg_df.empty else '✅ available'}")
    parts.append(f"implied_volatility={'❌ unavailable' if iv_df.empty else '✅ available'}")
    parts.append(f"trade_iv={'❌ unavailable' if tiv_df.empty else '✅ available'}")
    log(f"Plan probe complete: {' | '.join(parts)}", "INFO")

    try:
        with open(cache_path, "w") as f: json.dump(result, f)
    except Exception: pass
    return result


# ═══════════════════════════════════════════
# PER-SYMBOL COLLECTION
# ═══════════════════════════════════════════

async def collect_symbol(client, sem, symbol, trading_days, max_dte, outdir,
                         max_rows, progress, greeks_only=False, no_filter=False,
                         skip_trade_greeks=False, skip_iv=False, skip_trade_iv=False,
                         min_bid=None, max_spread_pct=None):

    raw_exps = await fetch_expirations(client, sem, symbol)
    if not raw_exps:
        log(f"[{symbol}] No expirations — check API connection", "ERROR")
        return 0

    exp_dates = sorted({d for s in raw_exps if (d := parse_date(s))})
    log(f"[{symbol}] {len(exp_dates)} unique expirations", "INFO")

    writer = WeeklyFileWriter(symbol, outdir, max_rows)
    oi_writer = DailyOIWriter(symbol, outdir)
    total_raw = total_removed = days_data = 0
    cur_week = None

    for di, qdate in enumerate(trading_days, 1):
        if not running: break
        valid_exps = [e for e in exp_dates if 0 <= (e - qdate).days <= max_dte]
        if not valid_exps: continue

        yr, wk = get_iso_week(qdate)
        if wk != cur_week:
            if cur_week is not None: print()
            print(f"\n=== {symbol} Week {yr}-W{wk:02d} ===")
            cur_week = wk

        day_rows = day_removed = day_exps = day_skipped = 0

        for edate in valid_exps:
            if not running: break
            estr = edate.strftime("%Y-%m-%d")
            if progress.is_done(symbol, str(qdate), estr):
                day_skipped += 1; continue

            # Primary: greeks/all
            gdf = await fetch_historical_greeks(client, sem, symbol, estr, qdate)
            if gdf.empty:
                progress.mark_done(symbol, str(qdate), estr); continue

            merged = gdf

            if not greeks_only:
                # Build endpoint coroutine list with stagger
                ep_coros = [
                    fetch_historical_quotes(client, sem, symbol, estr, qdate),
                    fetch_historical_ohlc(client, sem, symbol, estr, qdate),
                    fetch_historical_trades(client, sem, symbol, estr, qdate),
                ]
                ep_labels = ["q", "ohlc", "trade"]

                if not skip_trade_greeks:
                    ep_coros.append(fetch_historical_trade_greeks(client, sem, symbol, estr, qdate))
                    ep_labels.append("tg")
                if not skip_iv:
                    ep_coros.append(fetch_historical_implied_volatility(client, sem, symbol, estr, qdate))
                    ep_labels.append("iv")
                if not skip_trade_iv:
                    ep_coros.append(fetch_historical_trade_iv(client, sem, symbol, estr, qdate))
                    ep_labels.append("tiv")

                # Always fetch OI last
                ep_coros.append(fetch_historical_open_interest(client, sem, symbol, estr, qdate))
                ep_labels.append("oi")

                results = await asyncio.gather(
                    *[_staggered(c, i * ENDPOINT_STAGGER_SECS) for i, c in enumerate(ep_coros)],
                    return_exceptions=True
                )

                result_map = {}
                for label, r in zip(ep_labels, results):
                    result_map[label] = r if isinstance(r, pd.DataFrame) else pd.DataFrame()

                merged = merge_dataframes(merged, result_map.get("q", pd.DataFrame()), "q")
                merged = merge_dataframes(merged, result_map.get("ohlc", pd.DataFrame()), "ohlc")
                merged = merge_dataframes(merged, result_map.get("trade", pd.DataFrame()), "trade")
                if "tg" in result_map and not result_map["tg"].empty:
                    merged = merge_dataframes(merged, result_map["tg"], "tg")
                if "iv" in result_map and not result_map["iv"].empty:
                    # Rename IV columns to avoid clash with greeks/all implied_vol
                    iv_df = result_map["iv"].copy()
                    if "implied_vol" in iv_df.columns and "implied_vol" in merged.columns:
                        iv_df = iv_df.rename(columns={"implied_vol": "implied_vol_iv"})
                    merged = merge_dataframes(merged, iv_df, "iv")
                if "tiv" in result_map and not result_map["tiv"].empty:
                    merged = merge_dataframes(merged, result_map["tiv"], "tiv")
                if "oi" in result_map and not result_map["oi"].empty:
                    oi_writer.add(result_map["oi"], qdate)

            merged["symbol"] = symbol
            merged = enrich_data(merged, qdate, edate)
            filtered, removed = write_filter(
                merged, skip=no_filter,
                min_bid=min_bid, max_spread_pct=max_spread_pct)

            total_raw += len(merged)
            total_removed += removed
            if not filtered.empty:
                writer.add(filtered, qdate)
                day_rows += len(filtered)
                day_removed += removed
                day_exps += 1
            progress.mark_done(symbol, str(qdate), estr)

        if not greeks_only:
            oi_writer.flush_day(qdate)

        if day_skipped > 0 and day_rows == 0:
            print(f"  [{di:3d}/{len(trading_days)}] {qdate} | skipped ({day_skipped} exps)", flush=True)
        elif day_rows > 0:
            days_data += 1
            filt = "" if no_filter else f" (-{day_removed:,} filtered)"
            print(f"  [{di:3d}/{len(trading_days)}] {qdate} | {day_exps} exps | "
                  f"{day_rows:>8,} rows{filt} | buf {writer.total_buffered:,}", flush=True)
        else:
            print(f"  [{di:3d}/{len(trading_days)}] {qdate} | no data", flush=True)

    log(f"[{symbol}] Flushing …", "INFO")
    writer.flush_all()
    if not greeks_only: oi_writer.flush_all()
    progress.save()

    print()
    log(f"[{symbol}] ── DONE ──", "SUCCESS")
    log(f"  Days with data : {days_data}/{len(trading_days)}", "INFO")
    log(f"  Raw rows       : {total_raw:,}", "INFO")
    log(f"  Filtered out   : {total_removed:,}", "INFO")
    log(f"  Rows to disk   : {writer.total_written:,}", "INFO")
    log(f"  Files created  : {len(writer.files_created)}", "INFO")
    for fn in writer.files_created: log(f"    {fn}", "INFO")
    return writer.total_written


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

async def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    today = datetime.now(TIMEZONE).date()
    end_date   = parse_date(args.end_date)   or today - timedelta(days=args.lag_days)
    start_date = parse_date(args.start_date) or end_date - timedelta(days=args.days - 1)
    trading_days = get_trading_days(start_date, end_date)

    sem = asyncio.Semaphore(HTTP_CONCURRENCY)

    async with httpx.AsyncClient() as client:

        # ── Smoke redundancy test (exit after) ──
        if args.smoke_redundancy_test:
            await run_smoke_redundancy_test(client, sem, symbols[0], args.dte)
            return

        # ── Progress tracker ──
        progress = ProgressTracker(os.path.join(outdir, ".progress.json"),
                                    enabled=not args.no_resume)
        progress.validate_against_disk(outdir, symbols)

        # ── Lightweight plan probe (cached 7 days) ──
        findings = await run_plan_probe(client, sem, symbols[0], outdir)
        skip_tg  = args.skip_trade_greeks or findings.get("skip_trade_greeks", False)
        skip_iv  = args.skip_iv           or findings.get("skip_iv", False)
        skip_tiv = args.skip_trade_iv     or findings.get("skip_trade_iv", False)

        if findings.get("skip_trade_greeks") and not args.skip_trade_greeks:
            log("Auto-detected: trade_greeks unavailable — skipping.", "WARN")
        if findings.get("skip_trade_iv") and not args.skip_trade_iv:
            log("Auto-detected: trade_iv unavailable — skipping.", "WARN")

        # ── Build endpoint description for banner ──
        active_eps = ["greeks/all", "quote", "ohlc", "trade"]
        if not skip_tg:  active_eps.append("trade_greeks (Pro)")
        if not skip_iv:  active_eps.append("implied_volatility")
        if not skip_tiv: active_eps.append("trade_iv")
        active_eps.append("open_interest (OI folder)")

        filt_parts = [f"zero({'/'.join(WRITE_FILTER_ZERO_COLS)})"]
        if args.min_bid:        filt_parts.append(f"min_bid={args.min_bid}")
        if args.max_spread_pct: filt_parts.append(f"max_spread_pct={args.max_spread_pct}%")
        csv_filt = "OFF (raw)" if args.no_filter else " + ".join(filt_parts)

        print("\n" + "=" * 70)
        print("  Historical Options Data Collector v1.3")
        print("=" * 70)
        print(f"  Symbols      : {', '.join(symbols)}")
        print(f"  Date range   : {start_date} → {end_date} ({len(trading_days)} trading days)")
        print(f"  Safety lag   : today - {args.lag_days} days")
        print(f"  Max DTE      : {args.dte}")
        print(f"  Interval     : {INTERVAL}")
        print(f"  Endpoints    : {', '.join(active_eps)}")
        print(f"  Output       : {os.path.abspath(outdir)}/")
        print(f"  Max rows/file: {args.max_rows:,}")
        print(f"  Resume       : {'ON' if not args.no_resume else 'OFF'}")
        print(f"  CSV filter   : {csv_filt}")
        print(f"  Holidays     : US market holidays excluded")
        print("=" * 70 + "\n")

        date_chunks  = get_date_chunks(trading_days, CHUNK_WEEKS)
        total_chunks = len(date_chunks)
        log(f"Date range split into {total_chunks} chunk(s) of ~{CHUNK_WEEKS} weeks each.", "INFO")

        grand_total = 0
        for sym in symbols:
            if not running: break
            log(f"[{sym}] Starting collection — {total_chunks} chunk(s)", "INFO")

            for chunk_idx, chunk_days in enumerate(date_chunks, 1):
                if not running: break
                if chunk_already_done(chunk_days, sym, outdir, progress):
                    log(f"[{sym}] Chunk {chunk_idx}/{total_chunks}: "
                        f"{chunk_days[0]} → {chunk_days[-1]} — already done, skipping.", "SKIP")
                    continue
                log(f"[{sym}] Chunk {chunk_idx}/{total_chunks}: "
                    f"{chunk_days[0]} → {chunk_days[-1]} ({len(chunk_days)} trading days)", "INFO")

                rows = await collect_symbol(
                    client, sem, sym, chunk_days, args.dte,
                    outdir, args.max_rows, progress,
                    greeks_only=args.greeks_only,
                    no_filter=args.no_filter,
                    skip_trade_greeks=skip_tg,
                    skip_iv=skip_iv,
                    skip_trade_iv=skip_tiv,
                    min_bid=args.min_bid if not args.no_filter else None,
                    max_spread_pct=args.max_spread_pct if not args.no_filter else None,
                )
                grand_total += rows

                if chunk_idx < total_chunks and running:
                    log(f"[{sym}] Chunk {chunk_idx} done. Pausing {CHUNK_PAUSE_SECS}s …", "INFO")
                    await asyncio.sleep(CHUNK_PAUSE_SECS)
            print()

    print("\n" + "=" * 70)
    print("  FILES CREATED")
    print("=" * 70)
    try:
        grand_files = grand_mb = 0
        for sym in symbols:
            sym_dir = os.path.join(outdir, sym)
            if not os.path.isdir(sym_dir): continue
            csvs = sorted(f for f in os.listdir(sym_dir) if f.endswith(".csv"))
            sym_mb = sum(os.path.getsize(os.path.join(sym_dir, f)) for f in csvs) / (1024*1024)
            oi_dir = os.path.join(sym_dir, "OI")
            oi_csvs = sorted(f for f in os.listdir(oi_dir) if f.endswith(".csv")) if os.path.isdir(oi_dir) else []
            oi_mb = sum(os.path.getsize(os.path.join(oi_dir, f)) for f in oi_csvs) / (1024*1024) if oi_csvs else 0
            grand_files += len(csvs) + len(oi_csvs)
            grand_mb += sym_mb + oi_mb
            wk_map = defaultdict(list)
            for fn in csvs:
                parts = fn.split("_historical_")
                wk_map[parts[1].split("_part")[0] if len(parts)>1 else "other"].append(fn)
            print(f"  [{sym}]  1-min: {len(csvs)} files, {sym_mb:.1f} MB  |  OI: {len(oi_csvs)} files")
            for wk in sorted(wk_map): print(f"    {wk}: {len(wk_map[wk])} part(s)")
        print(f"\n  Total: {grand_files} files, {grand_mb:.1f} MB | Rows: {grand_total:,}")
    except Exception as e:
        log(f"Listing error: {e}", "ERROR")
    print("=" * 70)
    print("  COLLECTION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        log(f"Fatal: {e}", "ERROR")
        import traceback; traceback.print_exc()
