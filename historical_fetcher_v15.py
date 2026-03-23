#!/usr/bin/env python3
"""
Historical Options Data Collector v1.5
=======================================
Optimized for 1-YEAR fetch runs.

Changes vs v1.4:
- CHUNK_WEEKS: 2 → 4  (fewer pauses, better for long runs)
- CHUNK_PAUSE_SECS: 10 → 15  (slightly more breathing room)
- DEFAULT_LOOKBACK_DAYS: 365 (unchanged, now accurate comment)
- Header updated to reflect 1-year use case
- Chunk skip: if every ISO week in a chunk already has *_historical_* CSVs on
  disk (year-Www in filename), the whole chunk is skipped (no .progress.json
  required for that).
- Seed .progress.json from existing historical CSVs under --outdir (auto when
  progress is empty; use --seed-from-disk to merge from disk anytime)

All other logic unchanged from v1.4.
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

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_MAX_DTE = 5  # fallback
MAX_DTE_PER_SYMBOL = {
    "SPXW": 5,
    "SPY":  5,
    "QQQ":  5,
    "IWM":  5,
    "TLT":  5,
    "VIXW": 30,  # VIX weeklies expire monthly; need wider window or no data
}

DEFAULT_MAX_ROWS_PER_FILE = 500_000
DEFAULT_LAG_DAYS = 2

TIMEOUT = 180
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

# Optimised for 1-year run
HTTP_CONCURRENCY    = 8    # reduced to avoid startup and sustained 429 pressure
CHUNK_WEEKS         = 4    # v1.4=2 → v1.5=4 (fewer pauses for long run)
CHUNK_PAUSE_SECS    = 15   # v1.4=10 → v1.5=15 (more breathing room)
ENDPOINT_STAGGER_SECS = 0.05  # small anti-burst guard kept

WRITE_FILTER_ZERO_COLS = ["bid", "ask", "vega"]

COLUMNS_TO_DROP = [
    "bid_condition", "ask_condition",
    "vera", "speed", "zomma", "dual_gamma", "dual_delta",
    "d1", "d2", "ultima", "color", "veta", "vomma",
    "epsilon", "rho", "iv_error",
]

POST_MERGE_DROP = [
    "bid_q", "ask_q",
    "symbol_q", "symbol_ohlc", "symbol_trade",
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
        description="Historical Options Data Collector v1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    p.add_argument("--dte", type=int, default=DEFAULT_MAX_DTE,
                   help="Default max DTE for all symbols (VIXW always uses 30 per MAX_DTE_PER_SYMBOL)")
    p.add_argument("--outdir", type=str, default="historical_data_1yr",
                   help="Output root (default: historical_data_1yr)")
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS_PER_FILE)
    p.add_argument("--lag-days", type=int, default=DEFAULT_LAG_DAYS,
                   help="Safety lag: end_date = today - N (default: 2)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--seed-from-disk", action="store_true",
                   help="Scan existing *historical*.csv files and merge "
                        "(symbol|query_date|expiration) into .progress.json")
    p.add_argument("--no-auto-seed", action="store_true",
                   help="Disable auto seed when .progress.json has no entries")
    p.add_argument("--greeks-only", action="store_true")
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--min-bid", type=float, default=None)
    p.add_argument("--max-spread-pct", type=float, default=None)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=HTTP_CONCURRENCY,
                   help=f"HTTP concurrency semaphore (default: {HTTP_CONCURRENCY})")
    p.add_argument("--no-live-log", action="store_true",
                   help="Disable per-expiry log lines (default: log each expiry as it completes)")
    return p.parse_args()

# ═══════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════

def log(msg, level="INFO"):
    ts = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    icons = {"ERROR": "❌", "WARN": "⚠️", "SUCCESS": "✅",
             "DEBUG": "🔍", "INFO": "📊", "SKIP": "⏭️"}
    out = sys.stderr if level == "ERROR" else sys.stdout
    print(f"[{ts}] {icons.get(level, '')} {msg}", file=out, flush=True)

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
    for mo, dy in [(1, 1), (6, 19), (7, 4), (12, 25)]:
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
    return [trading_days[i:i + chunk_size] for i in range(0, len(trading_days), chunk_size)]

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
                    await asyncio.sleep(delay)
                    continue
                if r.status_code >= 500 and attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    log(f"[{label}] HTTP {r.status_code}, retry {attempt + 1}/{MAX_RETRIES}", "WARN")
                    await asyncio.sleep(delay)
                    continue
                return None
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                log(f"[{label}] Failed: {e}", "ERROR")
                return None
            except Exception as e:
                log(f"[{label}] Unexpected: {e}", "ERROR")
                return None
        return None

# ═══════════════════════════════════════════
# STAGGER HELPER
# ═══════════════════════════════════════════

async def _staggered(coro, delay):
    if delay > 0:
        await asyncio.sleep(delay)
    return await coro

# ═══════════════════════════════════════════
# THETA DATA v3 FETCHERS
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

async def fetch_historical_open_interest(client, sem, symbol, expiration, query_date):
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/open_interest",
        _hist_params(symbol, expiration, query_date, interval=False),
        f"{symbol} OI {expiration} {query_date}"))

# ═══════════════════════════════════════════
# FILTERING
# ═══════════════════════════════════════════

def diagnose_raw_data(df):
    if df.empty: return
    diag_cols = ["bid", "ask", "vega", "delta", "gamma", "theta", "implied_vol"]
    parts = []
    for col in diag_cols:
        if col not in df.columns: continue
        vals = pd.to_numeric(df[col], errors="coerce")
        n0, nn, nu = int((vals == 0).sum()), int(vals.isna().sum()), int(vals.nunique())
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
    lead = [c for c in ["symbol", "expiration", "strike", "right", "dte", "cp_sign", "query_date"]
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
            log(f"Concat error {year}-W{week:02d}: {e}", "ERROR")
            return
        if combined.empty:
            self.buffers[key], self.row_counts[key] = [], 0
            return
        sort_cols = [c for c in ["query_date", "ms_of_day", "timestamp",
                                  "symbol", "expiration", "strike", "right"]
                     if c in combined.columns]
        if sort_cols: combined = combined.sort_values(sort_cols)
        idx = 0
        while idx < len(combined):
            self.part_nums[key] += 1
            chunk = combined.iloc[idx: idx + self.max_rows]
            fname = f"{self.symbol}_historical_{year}-W{week:02d}_part{self.part_nums[key]:03d}.csv"
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
            if self.buffers[key]: self.flush_week(*key)

# ═══════════════════════════════════════════
# CHUNK SKIP HELPER
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
        except (IndexError, ValueError):
            continue
    return covered

def chunk_already_done(chunk_days, symbol, outdir):
    """
    True if every ISO week touched by this chunk already has at least one
    {SYMBOL}_historical_{year}-W{ww}_part*.csv under outdir/{SYMBOL}/.
    Uses on-disk week filenames only (not .progress.json).
    """
    covered = get_covered_weeks(symbol, outdir)
    weeks_needed = {get_iso_week(d) for d in chunk_days}
    return weeks_needed.issubset(covered)


def _norm_date_cell(val):
    """Normalize CSV date cell to YYYY-MM-DD for progress keys."""
    try:
        if val is None or pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    d = parse_date(s)
    return d.strftime("%Y-%m-%d") if d else None


def seed_progress_from_disk(outdir, symbols, progress, force=False):
    """
    Rebuild progress keys from rows already written under outdir/<symbol>/.
    Each CSV row yields symbol|query_date|expiration matching mark_done().
    """
    if not progress.enabled:
        return 0
    if not force and len(progress.done) > 0:
        return 0

    before = len(progress.done)
    chunk_rows = 250_000
    for sym in symbols:
        sym_dir = os.path.join(outdir, sym)
        if not os.path.isdir(sym_dir):
            continue
        for fname in sorted(os.listdir(sym_dir)):
            if not fname.endswith(".csv") or "_historical_" not in fname:
                continue
            path = os.path.join(sym_dir, fname)
            try:
                for chunk in pd.read_csv(
                    path,
                    usecols=["query_date", "expiration"],
                    chunksize=chunk_rows,
                    dtype=str,
                    low_memory=False,
                ):
                    chunk = chunk.dropna(how="all")
                    for q, e in zip(chunk["query_date"], chunk["expiration"]):
                        qn, en = _norm_date_cell(q), _norm_date_cell(e)
                        if qn and en:
                            progress.done.add(f"{sym}|{qn}|{en}")
            except (ValueError, KeyError) as ex:
                log(f"Seed skipped {sym}/{fname} (need query_date, expiration): {ex}", "WARN")
            except Exception as ex:
                log(f"Seed error {sym}/{fname}: {ex}", "WARN")

    added = len(progress.done) - before
    if added:
        log(f"Seeded progress from disk: +{added:,} keys "
            f"(total {len(progress.done):,})", "INFO")
        progress._dirty += 1
    elif force or before == 0:
        log("Seed from disk: no new keys (missing columns or no CSV rows)", "INFO")
    return added

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
            sort_cols = [c for c in ["expiration", "strike", "right"] if c in combined.columns]
            if sort_cols: combined = combined.sort_values(sort_cols)
            fname = f"{self.symbol}_oi_{key}.csv"
            combined.to_csv(os.path.join(self.oi_dir, fname), index=False)
            mb = os.path.getsize(os.path.join(self.oi_dir, fname)) / (1024 * 1024)
            log(f"[OI] {fname} — {len(combined):,} rows ({mb:.2f} MB)", "SUCCESS")
        except Exception as e:
            log(f"[OI] Write error {query_date}: {e}", "ERROR")
        finally:
            self.daily_frames[key] = []

    def flush_all(self):
        for key in sorted(self.daily_frames.keys()):
            if self.daily_frames[key]: self.flush_day(key)

# ═══════════════════════════════════════════
# PROGRESS / RESUME TRACKER (async-safe with Lock)
# ═══════════════════════════════════════════

class ProgressTracker:
    def __init__(self, filepath, enabled=True):
        self.filepath = filepath
        self.enabled = enabled
        self.done = set()
        self._dirty = 0
        self._lock = asyncio.Lock()
        if enabled and os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    self.done = set(json.load(f).get("completed", []))
                log(f"Resumed: {len(self.done)} combos already fetched", "INFO")
            except Exception:
                pass

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

    async def mark_done(self, symbol, qdate, exp):
        if not self.enabled: return
        async with self._lock:
            self.done.add(f"{symbol}|{qdate}|{exp}")
            self._dirty += 1
            if self._dirty >= 50:
                await self._flush()

    async def _flush(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump({"completed": sorted(self.done)}, f)
            self._dirty = 0
        except Exception:
            pass

    async def save(self):
        if not self.enabled: return
        async with self._lock:
            await self._flush()

# ═══════════════════════════════════════════
# SIGNAL HANDLING
# ═══════════════════════════════════════════

running = True

def _handle_signal(signum, frame):
    global running
    running = False
    log("Interrupt received — finishing current tasks & flushing data …", "WARN")

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ═══════════════════════════════════════════
# PER-EXPIRY FETCH & MERGE
# ═══════════════════════════════════════════

async def fetch_and_merge_expiry(
    client, sem, symbol, estr, qdate, edate,
    greeks_only=False, no_filter=False,
    min_bid=None, max_spread_pct=None
):
    if greeks_only:
        gdf = await fetch_historical_greeks(client, sem, symbol, estr, qdate)
        if gdf.empty:
            return None, None, 0, 0
        oi_df = await fetch_historical_open_interest(client, sem, symbol, estr, qdate)
        merged = gdf
    else:
        gdf, qdf, odf, tdf, oi_df = await asyncio.gather(
            _staggered(fetch_historical_greeks(client, sem, symbol, estr, qdate), 0.0),
            _staggered(fetch_historical_quotes(client, sem, symbol, estr, qdate), ENDPOINT_STAGGER_SECS * 1),
            _staggered(fetch_historical_ohlc(client, sem, symbol, estr, qdate), ENDPOINT_STAGGER_SECS * 2),
            _staggered(fetch_historical_trades(client, sem, symbol, estr, qdate), ENDPOINT_STAGGER_SECS * 3),
            _staggered(fetch_historical_open_interest(client, sem, symbol, estr, qdate), ENDPOINT_STAGGER_SECS * 4),
        )
        if gdf.empty:
            return None, None, 0, 0
        merged = gdf
        merged = merge_dataframes(merged, qdf, "q")
        merged = merge_dataframes(merged, odf, "ohlc")
        merged = merge_dataframes(merged, tdf, "trade")

    merged["symbol"] = symbol
    merged = enrich_data(merged, qdate, edate)
    filtered, removed = write_filter(
        merged, skip=no_filter, min_bid=min_bid, max_spread_pct=max_spread_pct
    )
    return filtered, oi_df, len(merged), removed

# ═══════════════════════════════════════════
# PER-SYMBOL CHUNK PROCESSOR
# ═══════════════════════════════════════════

async def collect_symbol_chunk(
    client, sem, symbol, chunk_days, exp_dates, max_dte,
    outdir, max_rows, progress, writer, oi_writer,
    greeks_only=False, no_filter=False,
    min_bid=None, max_spread_pct=None,
    live_log=True,
):
    total_raw = total_removed = days_data = 0
    cur_week = None

    for di, qdate in enumerate(chunk_days, 1):
        if not running: break
        valid_exps = [e for e in exp_dates if 0 <= (e - qdate).days <= max_dte]
        if not valid_exps: continue

        yr, wk = get_iso_week(qdate)
        if wk != cur_week:
            if cur_week is not None: print()
            print(f"\n=== {symbol} Week {yr}-W{wk:02d} ===")
            cur_week = wk

        pending = [e for e in valid_exps
                   if not progress.is_done(symbol, str(qdate), e.strftime("%Y-%m-%d"))]
        day_skipped = len(valid_exps) - len(pending)
        day_rows = day_removed = day_exps = 0

        if pending:
            n_pending = len(pending)

            async def _run_expiry(edate):
                try:
                    r = await fetch_and_merge_expiry(
                        client, sem, symbol,
                        edate.strftime("%Y-%m-%d"), qdate, edate,
                        greeks_only=greeks_only, no_filter=no_filter,
                        min_bid=min_bid, max_spread_pct=max_spread_pct
                    )
                    return edate, r, None
                except Exception as exc:
                    return edate, None, exc

            if live_log:
                log(f"[{symbol}] {qdate} — {n_pending} expir(y/ies) in flight …", "INFO")

            if live_log:
                done_n = 0
                for coro in asyncio.as_completed([_run_expiry(e) for e in pending]):
                    edate, result, exc = await coro
                    done_n += 1
                    estr = edate.strftime("%Y-%m-%d")
                    if exc is not None:
                        log(f"[{symbol}] {qdate} [{done_n}/{n_pending}] {estr} | error: {exc}", "ERROR")
                        continue
                    filtered, oi_df, n_raw, n_removed = result
                    if filtered is None:
                        await progress.mark_done(symbol, str(qdate), estr)
                        log(f"[{symbol}] {qdate} [{done_n}/{n_pending}] {estr} | no greeks", "INFO")
                        continue
                    total_raw += n_raw
                    total_removed += n_removed
                    if not filtered.empty:
                        writer.add(filtered, qdate)
                        day_rows += len(filtered)
                        day_removed += n_removed
                        day_exps += 1
                        filt = "" if no_filter else f", -{n_removed:,} filtered"
                        log(f"[{symbol}] {qdate} [{done_n}/{n_pending}] {estr} | "
                            f"{len(filtered):,} rows{filt}", "INFO")
                    else:
                        log(f"[{symbol}] {qdate} [{done_n}/{n_pending}] {estr} | 0 rows", "INFO")
                    if oi_df is not None and not oi_df.empty:
                        oi_writer.add(oi_df, qdate)
                    await progress.mark_done(symbol, str(qdate), estr)
            else:
                tasks = [
                    fetch_and_merge_expiry(
                        client, sem, symbol,
                        edate.strftime("%Y-%m-%d"), qdate, edate,
                        greeks_only=greeks_only, no_filter=no_filter,
                        min_bid=min_bid, max_spread_pct=max_spread_pct
                    )
                    for edate in pending
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for edate, result in zip(pending, results):
                    estr = edate.strftime("%Y-%m-%d")
                    if isinstance(result, Exception):
                        log(f"[{symbol}] Error {estr}: {result}", "ERROR")
                        continue
                    filtered, oi_df, n_raw, n_removed = result
                    if filtered is None:
                        await progress.mark_done(symbol, str(qdate), estr)
                        continue
                    total_raw += n_raw
                    total_removed += n_removed
                    if not filtered.empty:
                        writer.add(filtered, qdate)
                        day_rows += len(filtered)
                        day_removed += n_removed
                        day_exps += 1
                    if oi_df is not None and not oi_df.empty:
                        oi_writer.add(oi_df, qdate)
                    await progress.mark_done(symbol, str(qdate), estr)

        if not greeks_only:
            oi_writer.flush_day(qdate)

        if day_skipped > 0 and day_rows == 0:
            print(f"  [{di:3d}/{len(chunk_days)}] {qdate} | skipped ({day_skipped} exps)", flush=True)
        elif day_rows > 0:
            days_data += 1
            filt = "" if no_filter else f" (-{day_removed:,} filtered)"
            print(f"  [{di:3d}/{len(chunk_days)}] {qdate} | {day_exps} exps | "
                  f"{day_rows:>8,} rows{filt} | buf {writer.total_buffered:,}", flush=True)
        else:
            print(f"  [{di:3d}/{len(chunk_days)}] {qdate} | no data", flush=True)

    return total_raw, total_removed, days_data

# ═══════════════════════════════════════════
# PER-SYMBOL ORCHESTRATOR
# ═══════════════════════════════════════════

async def collect_symbol_all(
    client, sem, symbol, date_chunks, max_dte,
    outdir, max_rows, progress,
    greeks_only=False, no_filter=False,
    min_bid=None, max_spread_pct=None,
    live_log=True,
    startup_delay=0,
):
    if startup_delay > 0:
        await asyncio.sleep(startup_delay)
    raw_exps = await fetch_expirations(client, sem, symbol)
    if not raw_exps:
        log(f"[{symbol}] No expirations — check API connection", "ERROR")
        return 0
    exp_dates = sorted({d for s in raw_exps if (d := parse_date(s))})
    max_dte = MAX_DTE_PER_SYMBOL.get(symbol, max_dte)
    log(f"[{symbol}] {len(exp_dates)} unique expirations (max_dte={max_dte})", "INFO")

    writer = WeeklyFileWriter(symbol, outdir, max_rows)
    oi_writer = DailyOIWriter(symbol, outdir)
    total_raw = total_removed = total_days_data = 0
    total_chunks = len(date_chunks)

    for chunk_idx, chunk_days in enumerate(date_chunks, 1):
        if not running: break
        if chunk_already_done(chunk_days, symbol, outdir):
            wk_list = sorted({f'{y}-W{w:02d}' for y, w in {get_iso_week(d) for d in chunk_days}})
            log(f"[{symbol}] Chunk {chunk_idx}/{total_chunks}: "
                f"{chunk_days[0]} → {chunk_days[-1]} — skip (week file(s) on disk: {', '.join(wk_list)})", "SKIP")
            continue
        log(f"[{symbol}] Chunk {chunk_idx}/{total_chunks}: "
            f"{chunk_days[0]} → {chunk_days[-1]} ({len(chunk_days)} trading days)", "INFO")

        r, rm, dd = await collect_symbol_chunk(
            client, sem, symbol, chunk_days, exp_dates, max_dte,
            outdir, max_rows, progress, writer, oi_writer,
            greeks_only=greeks_only, no_filter=no_filter,
            min_bid=min_bid, max_spread_pct=max_spread_pct,
            live_log=live_log,
        )
        total_raw += r
        total_removed += rm
        total_days_data += dd

        if chunk_idx < total_chunks and running:
            log(f"[{symbol}] Chunk {chunk_idx} done. Pausing {CHUNK_PAUSE_SECS}s …", "INFO")
            await asyncio.sleep(CHUNK_PAUSE_SECS)

    log(f"[{symbol}] Flushing …", "INFO")
    writer.flush_all()
    if not greeks_only:
        oi_writer.flush_all()
    await progress.save()

    print()
    log(f"[{symbol}] ── DONE ──", "SUCCESS")
    log(f"  Raw rows      : {total_raw:,}", "INFO")
    log(f"  Filtered out  : {total_removed:,}", "INFO")
    log(f"  Rows to disk  : {writer.total_written:,}", "INFO")
    log(f"  Files created : {len(writer.files_created)}", "INFO")
    for fn in writer.files_created:
        log(f"    {fn}", "INFO")
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

    concurrency = args.concurrency
    sem = asyncio.Semaphore(concurrency)

    print("\n" + "=" * 70)
    print(" Historical Options Data Collector v1.5")
    print("=" * 70)
    print(f"  Symbols      : {', '.join(symbols)}")
    print(f"  Date range   : {start_date} → {end_date} ({len(trading_days)} trading days)")
    print(f"  Safety lag   : today - {args.lag_days} days")
    print(f"  Max DTE      : {args.dte}")
    print(f"  Interval     : {INTERVAL}")
    print(f"  Endpoints    : greeks/all, quote, ohlc, trade, open_interest")
    print(f"  Skipped      : trade_greeks, implied_volatility, trade_iv")
    print(f"  Concurrency  : {concurrency} simultaneous HTTP requests")
    print(f"  Stagger      : {ENDPOINT_STAGGER_SECS}s per endpoint (anti-burst)")
    print(f"  Chunk size   : {CHUNK_WEEKS} weeks ({CHUNK_WEEKS*5} trading days)")
    print(f"  Chunk pause  : {CHUNK_PAUSE_SECS}s between chunks")
    print(f"  Symbol mode  : ALL {len(symbols)} symbols run concurrently")
    print(f"  Live expiry log: {'OFF (--no-live-log)' if args.no_live_log else 'ON (each expiry as it completes)'}")
    print(f"  Output       : {os.path.abspath(outdir)}/")
    print(f"  Max rows/file: {args.max_rows:,}")
    print(f"  Resume       : {'ON' if not args.no_resume else 'OFF'}")
    if args.seed_from_disk:
        seed_line = "merge (--seed-from-disk)"
    elif args.no_auto_seed:
        seed_line = "off (--no-auto-seed)"
    else:
        seed_line = "auto if .progress.json empty"
    print(f"  Seed disk    : {seed_line}")
    print(f"  Holidays     : US market holidays excluded")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    date_chunks = get_date_chunks(trading_days, CHUNK_WEEKS)
    log(f"Date range split into {len(date_chunks)} chunk(s) of ~{CHUNK_WEEKS} weeks each.", "INFO")

    async with httpx.AsyncClient() as client:
        progress = ProgressTracker(
            os.path.join(outdir, ".progress.json"),
            enabled=not args.no_resume
        )
        progress.validate_against_disk(outdir, symbols)
        do_seed = args.seed_from_disk or (
            not args.no_auto_seed and len(progress.done) == 0
        )
        if do_seed:
            n_seed = seed_progress_from_disk(
                outdir, symbols, progress, force=args.seed_from_disk
            )
            if n_seed:
                await progress.save()

        sym_tasks = [
            collect_symbol_all(
                client, sem, sym, date_chunks, args.dte,
                outdir, args.max_rows, progress,
                greeks_only=args.greeks_only,
                no_filter=args.no_filter,
                min_bid=args.min_bid if not args.no_filter else None,
                max_spread_pct=args.max_spread_pct if not args.no_filter else None,
                live_log=not args.no_live_log,
                startup_delay=i * 2.0,
            )
            for i, sym in enumerate(symbols)
        ]
        results = await asyncio.gather(*sym_tasks, return_exceptions=True)

    grand_total = 0
    for sym, r in zip(symbols, results):
        if isinstance(r, Exception):
            log(f"[{sym}] Failed: {r}", "ERROR")
        else:
            grand_total += r

    print("\n" + "=" * 70)
    print(" FILES CREATED")
    print("=" * 70)
    try:
        grand_files = grand_mb = 0
        for sym in symbols:
            sym_dir = os.path.join(outdir, sym)
            if not os.path.isdir(sym_dir): continue
            csvs = sorted(f for f in os.listdir(sym_dir) if f.endswith(".csv"))
            sym_mb = sum(os.path.getsize(os.path.join(sym_dir, f)) for f in csvs) / (1024 * 1024)
            oi_dir = os.path.join(sym_dir, "OI")
            oi_csvs = sorted(f for f in os.listdir(oi_dir) if f.endswith(".csv")) if os.path.isdir(oi_dir) else []
            oi_mb = sum(os.path.getsize(os.path.join(oi_dir, f)) for f in oi_csvs) / (1024 * 1024) if oi_csvs else 0
            grand_files += len(csvs) + len(oi_csvs)
            grand_mb += sym_mb + oi_mb
            wk_map = defaultdict(list)
            for fn in csvs:
                parts = fn.split("_historical_")
                wk_map[parts[1].split("_part")[0] if len(parts) > 1 else "other"].append(fn)
            print(f"  [{sym}] 1-min: {len(csvs)} files, {sym_mb:.1f} MB | OI: {len(oi_csvs)} files")
            for wk in sorted(wk_map):
                print(f"    {wk}: {len(wk_map[wk])} part(s)")
        print(f"\n  Total: {grand_files} files, {grand_mb:.1f} MB | Rows: {grand_total:,}")
    except Exception as e:
        log(f"Listing error: {e}", "ERROR")
    print("=" * 70)
    print(" COLLECTION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except Exception as e:
        log(f"Fatal: {e}", "ERROR")
        import traceback
        traceback.print_exc()
