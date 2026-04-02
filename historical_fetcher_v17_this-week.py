#!/usr/bin/env python3
"""
Historical Options Data Collector v1.7 — CSV Edition
=====================================================
Three separate endpoints → three separate CSV file groups per date.
No cross-endpoint merging. All raw timestamps preserved.

Output layout (/workspace/data/this week/):
  greeks/   → greeks_YYYY-MM-DD[_partNNN].csv
  trade_quote/ → trade_quote_YYYY-MM-DD[_partNNN].csv   (quotes + trades combined)
  ohlcv/    → ohlcv_YYYY-MM-DD[_partNNN].csv

Filters:
- greeks/all : vega != 0, bid != 0, ask != 0 (no delta filter)
- quotes     : bid_size > 100 AND ask_size > 100
- trades     : no filter
- ohlcv      : no filter

Key changes vs v1.6:
- Output to CSV instead of DuckDB (no duckdb dependency)
- All symbols written into same date-partitioned CSV files
- Separate subdirs: greeks/, trade_quote/, ohlcv/
- quotes and trades merged into trade_quote (with 'endpoint' column to distinguish)
- CSV split when file size approaches MAX_CSV_BYTES (10 MB)
- Output root defaults to /workspace/data/this week
- No DuckDB imports or writers
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

DEFAULT_LOOKBACK_DAYS = 7   # last-week / this-week default
DEFAULT_MAX_DTE = 5
MAX_DTE_PER_SYMBOL = {
    "SPXW": 5,
    "SPY":  5,
    "QQQ":  5,
    "IWM":  5,
    "TLT":  5,
    "VIXW": 30,
}

DEFAULT_LAG_DAYS = 0        # 0 → include today / 27 Mar 2026

TIMEOUT = 180
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

HTTP_CONCURRENCY = 12
EXPIRY_CONCURRENCY = 2
CHUNK_WEEKS = 2             # smaller chunks for short date ranges
CHUNK_PAUSE_SECS = 4
ENDPOINT_STAGGER_SECS = 0.05

PROBE_CACHE_FILE = ".probe_cache.json"

# CSV split threshold: split before file exceeds this size
MAX_CSV_BYTES = 9 * 1024 * 1024   # 9 MB hard limit (leaves headroom below 10 MB)

# Output subdirectory names
DIR_GREEKS     = "greeks"
DIR_TRADE_QUOTE = "trade_quote"
DIR_OHLCV      = "ohlcv"

# ═══════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Historical Options Data Collector v1.7 (CSV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols",  type=str, default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--days",     type=int, default=DEFAULT_LOOKBACK_DAYS)
    p.add_argument("--dte",      type=int, default=DEFAULT_MAX_DTE)
    p.add_argument("--outdir",   type=str, default="/workspace/data/this week",
                   help="Output root (default: /workspace/data/this week)")
    p.add_argument("--lag-days", type=int, default=DEFAULT_LAG_DAYS)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date",   type=str, default=None)
    p.add_argument("--concurrency", type=int, default=HTTP_CONCURRENCY)
    p.add_argument("--expiry-concurrency", type=int, default=EXPIRY_CONCURRENCY,
                   help="Max expirations processed at once per symbol/day (lower = less RAM)")
    p.add_argument("--no-live-log", action="store_true")
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

async def fetch_historical_ohlcv(client, sem, symbol, expiration, query_date):
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/ohlc",
        _hist_params(symbol, expiration, query_date),
        f"{symbol} O {expiration} {query_date}"))

async def fetch_historical_trades(client, sem, symbol, expiration, query_date):
    return _parse_csv(await fetch_with_retry(
        client, sem, f"{BASE_URL}/v3/option/history/trade",
        _hist_params(symbol, expiration, query_date),
        f"{symbol} T {expiration} {query_date}"))

# ═══════════════════════════════════════════
# FILTERS (no clamping; only row exclusion)
# ═══════════════════════════════════════════

def filter_greeks(df):
    """Keep rows where vega != 0, bid != 0, ask != 0. No delta filter."""
    if df.empty:
        return df, 0
    initial = len(df)
    for col in ("vega", "bid", "ask"):
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df = df[numeric != 0]
    return df, initial - len(df)

def filter_quotes(df):
    """Keep rows where bid_size > 100 AND ask_size > 100."""
    if df.empty:
        return df, 0
    initial = len(df)
    for col in ("bid_size", "ask_size"):
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df = df[numeric > 100]
    return df, initial - len(df)

def filter_trades(df):
    """No filter for trades — return as-is."""
    return df, 0

def filter_ohlcv(df):
    """No filter — return as-is."""
    return df, 0

# ═══════════════════════════════════════════
# ENRICHMENT (adds dte / query_date / symbol)
# ═══════════════════════════════════════════

def enrich(df, symbol, query_date, expiration_date):
    """Add dte, query_date, symbol. Preserve all timestamps raw."""
    if df.empty:
        return df
    df = df.copy()
    df["symbol"]     = symbol
    df["dte"]        = (expiration_date - query_date).days
    df["query_date"] = query_date.strftime("%Y-%m-%d")
    return df

# ═══════════════════════════════════════════
# CSV WRITER (date-partitioned, size-limited)
# ═══════════════════════════════════════════

class CsvWriter:
    """
    Append-only, date-partitioned CSV writer with automatic file splitting.

    File naming:
        <subdir>/<prefix>_<YYYY-MM-DD>.csv
        <subdir>/<prefix>_<YYYY-MM-DD>_part002.csv  (when size > MAX_CSV_BYTES)

    The first part has NO part suffix to stay consistent with simple date-only
    lookups; additional parts are numbered _part002, _part003, …
    """

    def __init__(self, subdir: str, prefix: str):
        self.subdir  = subdir
        self.prefix  = prefix
        os.makedirs(subdir, exist_ok=True)
        # {date_str -> {"path": str, "part": int, "header_written": bool}}
        self._state: dict[str, dict] = {}
        self.total_written = 0

    def _current_path(self, date_str: str) -> str:
        s = self._state[date_str]
        part = s["part"]
        if part == 1:
            return os.path.join(self.subdir, f"{self.prefix}_{date_str}.csv")
        return os.path.join(self.subdir, f"{self.prefix}_{date_str}_part{part:03d}.csv")

    def _file_size(self, path: str) -> int:
        try:
            return os.path.getsize(path)
        except FileNotFoundError:
            return 0

    def write(self, df: pd.DataFrame):
        if df is None or df.empty:
            return

        # Group by query_date so we route to the right date file
        if "query_date" not in df.columns:
            log("CsvWriter: missing query_date column — skipping batch", "WARN")
            return

        for date_str, group in df.groupby("query_date", sort=False):
            date_str = str(date_str)

            # Initialise state for this date if needed
            if date_str not in self._state:
                self._state[date_str] = {"part": 1, "header_written": False}

            while not group.empty:
                path = self._current_path(date_str)
                current_size = self._file_size(path)
                state = self._state[date_str]

                # If this file already exceeds limit, roll to next part
                if state["header_written"] and current_size >= MAX_CSV_BYTES:
                    state["part"] += 1
                    state["header_written"] = False
                    path = self._current_path(date_str)
                    current_size = 0

                # Estimate how many rows fit in the remaining space
                # Use a sample to get bytes-per-row estimate
                sample_csv = group.head(min(200, len(group))).to_csv(index=False)
                bytes_per_row = len(sample_csv.encode()) / min(200, len(group))
                remaining_bytes = MAX_CSV_BYTES - current_size
                max_rows = max(1, int(remaining_bytes / bytes_per_row)) if bytes_per_row > 0 else len(group)

                chunk = group.head(max_rows)
                group = group.iloc[max_rows:]

                write_header = not state["header_written"]
                chunk.to_csv(path, mode="a", index=False, header=write_header)
                state["header_written"] = True
                self.total_written += len(chunk)

                # After writing, check if we're over limit → will roll on next call
                if self._file_size(path) >= MAX_CSV_BYTES and not group.empty:
                    state["part"] += 1
                    state["header_written"] = False

    def summary(self) -> list[str]:
        """Return list of all CSV files written."""
        paths = []
        for date_str, s in self._state.items():
            for part in range(1, s["part"] + 1):
                if part == 1:
                    p = os.path.join(self.subdir, f"{self.prefix}_{date_str}.csv")
                else:
                    p = os.path.join(self.subdir, f"{self.prefix}_{date_str}_part{part:03d}.csv")
                if os.path.exists(p):
                    paths.append(p)
        return sorted(paths)

# ═══════════════════════════════════════════
# PROGRESS / RESUME TRACKER
# ═══════════════════════════════════════════

class ProgressTracker:
    def __init__(self, filepath, enabled=True):
        self.filepath = filepath
        self.enabled  = enabled
        self.done     = set()
        self._dirty   = 0
        self._lock    = asyncio.Lock()
        if enabled and os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    self.done = set(json.load(f).get("completed", []))
                log(f"Resumed: {len(self.done)} combos already fetched", "INFO")
            except Exception:
                pass

    def is_done(self, symbol, qdate, exp):
        return self.enabled and f"{symbol}|{qdate}|{exp}" in self.done

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
    log("Interrupt received — finishing current tasks & flushing …", "WARN")

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

# ═══════════════════════════════════════════
# PER-EXPIRY FETCH (no merge — separate paths)
# ═══════════════════════════════════════════

async def fetch_expiry_data(client, sem, symbol, estr, qdate, edate):
    """
    Fetch all four endpoints independently.
    Returns (greeks_df, quotes_df, trades_df, ohlcv_df, g_rm, q_rm, t_rm, o_rm).
    Quotes and trades are kept separate (endpoint column added before merging).
    """
    gdf, qdf, odf, tdf = await asyncio.gather(
        _staggered(fetch_historical_greeks(client, sem, symbol, estr, qdate), 0.0),
        _staggered(fetch_historical_quotes(client, sem, symbol, estr, qdate), ENDPOINT_STAGGER_SECS * 1),
        _staggered(fetch_historical_ohlcv(client, sem,  symbol, estr, qdate), ENDPOINT_STAGGER_SECS * 2),
        _staggered(fetch_historical_trades(client, sem, symbol, estr, qdate), ENDPOINT_STAGGER_SECS * 3),
    )

    # greeks/all — enrich then filter
    gdf = enrich(gdf, symbol, qdate, edate)
    gdf, g_removed = filter_greeks(gdf)

    # quotes — enrich then filter
    qdf = enrich(qdf, symbol, qdate, edate)
    qdf, q_removed = filter_quotes(qdf)
    if not qdf.empty:
        qdf.insert(0, "endpoint", "quote")

    # trades — enrich, no filter
    tdf = enrich(tdf, symbol, qdate, edate)
    tdf, t_removed = filter_trades(tdf)
    if not tdf.empty:
        tdf.insert(0, "endpoint", "trade")

    # ohlcv — enrich, no filter
    odf = enrich(odf, symbol, qdate, edate)
    odf, o_removed = filter_ohlcv(odf)

    return gdf, qdf, tdf, odf, g_removed, q_removed, t_removed, o_removed

# ═══════════════════════════════════════════
# PER-SYMBOL CHUNK PROCESSOR
# ═══════════════════════════════════════════

async def collect_symbol_chunk(
    client, sem, symbol, chunk_days, exp_dates, max_dte,
    greek_writer, tq_writer, ohlcv_writer,
    progress, live_log=True, expiry_concurrency=EXPIRY_CONCURRENCY,
):
    total_g = total_q = total_t = total_ohlcv = 0
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
        day_g = day_q = day_t = day_ohlcv = 0

        if pending:
            n_pending  = len(pending)
            batch_size = max(1, int(expiry_concurrency))
            if live_log:
                log(f"[{symbol}] {qdate} — {n_pending} expir(y/ies), batch={batch_size} …", "INFO")

            async def _run_expiry(edate):
                try:
                    r = await fetch_expiry_data(client, sem, symbol,
                                                edate.strftime("%Y-%m-%d"), qdate, edate)
                    return edate, r, None
                except Exception as exc:
                    return edate, None, exc

            done_n = 0
            for start in range(0, n_pending, batch_size):
                batch = pending[start:start + batch_size]
                for coro in asyncio.as_completed([_run_expiry(e) for e in batch]):
                    edate, result, exc = await coro
                    done_n += 1
                    estr = edate.strftime("%Y-%m-%d")
                    if exc is not None:
                        log(f"[{symbol}] {qdate} [{done_n}/{n_pending}] {estr} | error: {exc}", "ERROR")
                        continue
                    if result is None:
                        await progress.mark_done(symbol, str(qdate), estr)
                        continue

                    gdf, qdf, tdf, odf, g_rm, q_rm, t_rm, o_rm = result

                    # Write to date-partitioned CSVs immediately
                    greek_writer.write(gdf)
                    # Merge quotes + trades into trade_quote file
                    parts = [df for df in (qdf, tdf) if not df.empty]
                    if parts:
                        tq_df = pd.concat(parts, ignore_index=True)
                        tq_writer.write(tq_df)
                    ohlcv_writer.write(odf)

                    ng = len(gdf) if not gdf.empty else 0
                    nq = len(qdf) if not qdf.empty else 0
                    nt = len(tdf) if not tdf.empty else 0
                    no = len(odf) if not odf.empty else 0
                    day_g += ng; day_q += nq; day_t += nt; day_ohlcv += no
                    if live_log:
                        log(f"[{symbol}] {qdate} [{done_n}/{n_pending}] {estr} | "
                            f"greeks={ng:,}(-{g_rm}) quotes={nq:,}(-{q_rm}) "
                            f"trades={nt:,} ohlcv={no:,}", "INFO")
                    await progress.mark_done(symbol, str(qdate), estr)

        total_g += day_g; total_q += day_q; total_t += day_t; total_ohlcv += day_ohlcv
        print(
            f"  [{di:3d}/{len(chunk_days)}] {qdate} | "
            f"greeks={day_g:>7,} quotes={day_q:>7,} trades={day_t:>7,} ohlcv={day_ohlcv:>7,}",
            flush=True,
        )

    return total_g, total_q, total_t, total_ohlcv

# ═══════════════════════════════════════════
# PER-SYMBOL ORCHESTRATOR
# ═══════════════════════════════════════════

async def collect_symbol_all(
    client, sem, symbol, date_chunks, max_dte,
    greek_writer, tq_writer, ohlcv_writer,
    progress, live_log=True, startup_delay=0,
    expiry_concurrency=EXPIRY_CONCURRENCY,
):
    if startup_delay > 0:
        await asyncio.sleep(startup_delay)

    raw_exps = await fetch_expirations(client, sem, symbol)
    if not raw_exps:
        log(f"[{symbol}] No expirations — check API connection", "ERROR")
        return 0, 0, 0, 0

    exp_dates = sorted({d for s in raw_exps if (d := parse_date(s))})
    max_dte   = MAX_DTE_PER_SYMBOL.get(symbol, max_dte)
    log(f"[{symbol}] {len(exp_dates)} unique expirations (max_dte={max_dte})", "INFO")

    total_g = total_q = total_t = total_ohlcv = 0
    total_chunks = len(date_chunks)

    for chunk_idx, chunk_days in enumerate(date_chunks, 1):
        if not running: break
        log(f"[{symbol}] Chunk {chunk_idx}/{total_chunks}: "
            f"{chunk_days[0]} → {chunk_days[-1]} ({len(chunk_days)} trading days)", "INFO")

        cg, cq, ct, co = await collect_symbol_chunk(
            client, sem, symbol, chunk_days, exp_dates, max_dte,
            greek_writer, tq_writer, ohlcv_writer,
            progress, live_log=live_log,
            expiry_concurrency=expiry_concurrency,
        )
        total_g += cg; total_q += cq; total_t += ct; total_ohlcv += co

        if chunk_idx < total_chunks and running:
            log(f"[{symbol}] Chunk {chunk_idx} done. Pausing {CHUNK_PAUSE_SECS}s …", "INFO")
            await asyncio.sleep(CHUNK_PAUSE_SECS)

    log(f"[{symbol}] ── DONE ──", "SUCCESS")
    log(f"  greeks rows     : {total_g:,}", "INFO")
    log(f"  quotes rows     : {total_q:,}", "INFO")
    log(f"  trades rows     : {total_t:,}", "INFO")
    log(f"  ohlcv rows      : {total_ohlcv:,}", "INFO")

    return total_g, total_q, total_t, total_ohlcv

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

async def main():
    args    = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    outdir  = args.outdir
    os.makedirs(outdir, exist_ok=True)

    today    = datetime.now(TIMEZONE).date()
    end_date = parse_date(args.end_date)   or today - timedelta(days=args.lag_days)
    start_date = parse_date(args.start_date) or end_date - timedelta(days=args.days - 1)
    trading_days = get_trading_days(start_date, end_date)

    sem = asyncio.Semaphore(args.concurrency)

    # Shared writers — all symbols write into the same date-partitioned files
    greek_dir = os.path.join(outdir, DIR_GREEKS)
    tq_dir    = os.path.join(outdir, DIR_TRADE_QUOTE)
    ohlcv_dir = os.path.join(outdir, DIR_OHLCV)

    greek_writer = CsvWriter(greek_dir,  "greeks")
    tq_writer    = CsvWriter(tq_dir,     "trade_quote")
    ohlcv_writer = CsvWriter(ohlcv_dir,  "ohlcv")

    print("\n" + "=" * 70)
    print("  Historical Options Data Collector v1.7 — CSV Edition")
    print("=" * 70)
    print(f"  Symbols     : {', '.join(symbols)}")
    print(f"  Date range  : {start_date} → {end_date} ({len(trading_days)} trading days)")
    print(f"  Safety lag  : today - {args.lag_days} days")
    print(f"  Max DTE     : {args.dte}")
    print(f"  Interval    : {INTERVAL}")
    print(f"  Endpoints   : greeks/all | quote | trade | ohlc")
    print(f"  Filters     : greeks(vega/bid/ask≠0) | quotes(bid_size>100 & ask_size>100) | trades(none) | ohlcv(none)")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Expiry conc.: {max(1, int(args.expiry_concurrency))}")
    print(f"  Chunk size  : {CHUNK_WEEKS} weeks")
    print(f"  Chunk pause : {CHUNK_PAUSE_SECS}s")
    print(f"  Max CSV size: {MAX_CSV_BYTES // (1024*1024)} MB")
    print(f"  Output      : {os.path.abspath(outdir)}/")
    print(f"    greeks/       → greeks_YYYY-MM-DD[_partNNN].csv")
    print(f"    trade_quote/  → trade_quote_YYYY-MM-DD[_partNNN].csv")
    print(f"    ohlcv/        → ohlcv_YYYY-MM-DD[_partNNN].csv")
    print(f"  Resume      : {'ON' if not args.no_resume else 'OFF'}")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    date_chunks = get_date_chunks(trading_days, CHUNK_WEEKS)
    log(f"Date range split into {len(date_chunks)} chunk(s) of ~{CHUNK_WEEKS} weeks each.", "INFO")

    async with httpx.AsyncClient() as client:
        progress = ProgressTracker(
            os.path.join(outdir, ".progress.json"),
            enabled=not args.no_resume,
        )

        # Run all symbols concurrently; they share the same CSV writers (thread-safe
        # via asyncio single-thread model — no extra locking needed)
        sym_tasks = [
            collect_symbol_all(
                client, sem, sym, date_chunks, args.dte,
                greek_writer, tq_writer, ohlcv_writer,
                progress,
                live_log=not args.no_live_log,
                startup_delay=i * 2.0,
                expiry_concurrency=max(1, int(args.expiry_concurrency)),
            )
            for i, sym in enumerate(symbols)
        ]

        results = await asyncio.gather(*sym_tasks, return_exceptions=True)

    print("\n" + "=" * 70)
    print("  COLLECTION COMPLETE")
    print("=" * 70)
    grand_g = grand_q = grand_t = grand_o = 0
    for sym, r in zip(symbols, results):
        if isinstance(r, Exception):
            log(f"[{sym}] Failed: {r}", "ERROR")
        else:
            cg, cq, ct, co = r
            grand_g += cg; grand_q += cq; grand_t += ct; grand_o += co
            print(f"  [{sym}] greeks={cg:,} quotes={cq:,} trades={ct:,} ohlcv={co:,}")
    print(f"\n  TOTAL: greeks={grand_g:,} quotes={grand_q:,} trades={grand_t:,} ohlcv={grand_o:,}")
    print("=" * 70)

    print("\n  Output files written:")
    for path in greek_writer.summary() + tq_writer.summary() + ohlcv_writer.summary():
        size_kb = os.path.getsize(path) / 1024
        print(f"    {path}  ({size_kb:,.1f} KB)")
    print()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except Exception as e:
        log(f"Fatal: {e}", "ERROR")
        import traceback
        traceback.print_exc()
