#!/usr/bin/env python3
"""
Historical Options Data Collector v1.6 — DuckDB Edition
========================================================
Three separate endpoints → three separate DuckDB databases per symbol.
No cross-endpoint merging. All raw timestamps preserved.

Endpoints & output:
  1) greeks/all  → {SYMBOL}/greeks_all.duckdb   table: greeks_all
  2) quote       → {SYMBOL}/trade_quote.duckdb   table: quotes
     trade       → {SYMBOL}/trade_quote.duckdb   table: trades
  3) ohlc        → {SYMBOL}/ohlcv.duckdb         table: ohlcv

Filters:
  - greeks/all  : vega != 0, bid != 0, ask != 0 (no delta filter)
  - quotes      : bid_size > 100 AND ask_size > 100
  - trades      : no filter
  - ohlcv       : no filter

Key changes vs v1.5:
  - 3-year default lookback (1095 days)
  - Output to DuckDB (one DB per endpoint group per symbol)
  - NO cross-endpoint merging (preserve raw timestamps from each API)
  - No OI fetch, no CSV writers, no weekly parts
  - Do not clamp inf/nan to 0 or NaN (preserve raw API values)
  - All data written via DuckDB append
  - Quotes and trades stored as separate tables inside trade_quote.duckdb
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
import duckdb
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

DEFAULT_LOOKBACK_DAYS = 1095  # 3 years
DEFAULT_MAX_DTE = 5
MAX_DTE_PER_SYMBOL = {
    "SPXW": 5,
    "SPY": 5,
    "QQQ": 5,
    "IWM": 5,
    "TLT": 5,
    "VIXW": 30,
}

DEFAULT_LAG_DAYS = 2

TIMEOUT = 180
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

HTTP_CONCURRENCY = 12
EXPIRY_CONCURRENCY = 2
CHUNK_WEEKS = 8
CHUNK_PAUSE_SECS = 8
ENDPOINT_STAGGER_SECS = 0.05

PROBE_CACHE_FILE = ".probe_cache.json"

# DuckDB table names
TABLE_GREEKS = "greeks_all"
TABLE_QUOTES = "quotes"
TABLE_TRADES = "trades"
TABLE_OHLCV = "ohlcv"

# ═══════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Historical Options Data Collector v1.6 (DuckDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    p.add_argument("--dte", type=int, default=DEFAULT_MAX_DTE)
    p.add_argument("--outdir", type=str, default="data/theta_data_3year",
                   help="Output root (default: data/theta_data_3year)")
    p.add_argument("--lag-days", type=int, default=DEFAULT_LAG_DAYS)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--seed-from-disk", action="store_true")
    p.add_argument("--no-auto-seed", action="store_true")
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
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
        # Do NOT clamp inf/nan — preserve raw API values
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
    df["symbol"] = symbol
    df["dte"] = (expiration_date - query_date).days
    df["query_date"] = query_date.strftime("%Y-%m-%d")
    return df

# ═══════════════════════════════════════════
# DUCKDB WRITER (one connection per DB file)
# ═══════════════════════════════════════════

class DuckDBWriter:
    """
    Append-only DuckDB writer.
    Opens a persistent connection; creates table on first write,
    then INSERT INTO ... SELECT * for subsequent batches.
    """
    def __init__(self, db_path: str, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._con = duckdb.connect(db_path)
        self._table_exists = False
        self.total_written = 0

    def _ensure_table(self, df: pd.DataFrame):
        if not self._table_exists:
            existing = self._con.execute(
                f"SELECT table_name FROM information_schema.tables "
                f"WHERE table_name='{self.table_name}'"
            ).fetchone()
            if existing:
                self._table_exists = True
                return
            self._con.register("_init_df", df.head(0))
            self._con.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table_name} AS "
                f"SELECT * FROM _init_df"
            )
            self._con.unregister("_init_df")
            self._table_exists = True

    def write(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        self._ensure_table(df)
        self._con.register("_batch", df)
        self._con.execute(f"INSERT INTO {self.table_name} SELECT * FROM _batch")
        self._con.unregister("_batch")
        self.total_written += len(df)

    def close(self):
        try:
            self._con.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

# ═══════════════════════════════════════════
# PROGRESS / RESUME TRACKER
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
signal.signal(signal.SIGINT, _handle_signal)

# ═══════════════════════════════════════════
# PER-EXPIRY FETCH (no merge — separate paths)
# ═══════════════════════════════════════════

async def fetch_expiry_data(client, sem, symbol, estr, qdate, edate):
    """
    Fetch all four endpoints independently.
    Returns (greeks_df, quotes_df, trades_df, ohlcv_df, g_rm, q_rm, t_rm, o_rm).
    Quotes and trades are kept separate (no merge).
    """
    gdf, qdf, odf, tdf = await asyncio.gather(
        _staggered(fetch_historical_greeks(client, sem, symbol, estr, qdate), 0.0),
        _staggered(fetch_historical_quotes(client, sem, symbol, estr, qdate), ENDPOINT_STAGGER_SECS * 1),
        _staggered(fetch_historical_ohlcv(client, sem, symbol, estr, qdate), ENDPOINT_STAGGER_SECS * 2),
        _staggered(fetch_historical_trades(client, sem, symbol, estr, qdate), ENDPOINT_STAGGER_SECS * 3),
    )

    # greeks/all — enrich then filter
    gdf = enrich(gdf, symbol, qdate, edate)
    gdf, g_removed = filter_greeks(gdf)

    # quotes — enrich then filter (bid_size > 100 AND ask_size > 100)
    qdf = enrich(qdf, symbol, qdate, edate)
    qdf, q_removed = filter_quotes(qdf)

    # trades — enrich, no filter
    tdf = enrich(tdf, symbol, qdate, edate)
    tdf, t_removed = filter_trades(tdf)

    # ohlcv — enrich, no filter
    odf = enrich(odf, symbol, qdate, edate)
    odf, o_removed = filter_ohlcv(odf)

    return gdf, qdf, tdf, odf, g_removed, q_removed, t_removed, o_removed

# ═══════════════════════════════════════════
# PER-SYMBOL CHUNK PROCESSOR
# ═══════════════════════════════════════════

async def collect_symbol_chunk(
    client, sem, symbol, chunk_days, exp_dates, max_dte,
    writers, progress, live_log=True, expiry_concurrency=EXPIRY_CONCURRENCY,
):
    greek_writer, quote_writer, trade_writer, ohlcv_writer = writers
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
            n_pending = len(pending)
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
                    # Write to separate DuckDB tables immediately to keep memory low.
                    greek_writer.write(gdf)
                    quote_writer.write(qdf)
                    trade_writer.write(tdf)
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
    outdir, progress, live_log=True, startup_delay=0,
    expiry_concurrency=EXPIRY_CONCURRENCY,
):
    if startup_delay > 0:
        await asyncio.sleep(startup_delay)

    raw_exps = await fetch_expirations(client, sem, symbol)
    if not raw_exps:
        log(f"[{symbol}] No expirations — check API connection", "ERROR")
        return 0, 0, 0, 0

    exp_dates = sorted({d for s in raw_exps if (d := parse_date(s))})
    max_dte = MAX_DTE_PER_SYMBOL.get(symbol, max_dte)
    log(f"[{symbol}] {len(exp_dates)} unique expirations (max_dte={max_dte})", "INFO")

    sym_dir = os.path.join(outdir, symbol)
    os.makedirs(sym_dir, exist_ok=True)

    # Three DuckDB files; quotes & trades are separate tables in trade_quote.duckdb
    greek_writer = DuckDBWriter(os.path.join(sym_dir, "greeks_all.duckdb"), TABLE_GREEKS)
    quote_writer = DuckDBWriter(os.path.join(sym_dir, "trade_quote.duckdb"), TABLE_QUOTES)
    trade_writer = DuckDBWriter(os.path.join(sym_dir, "trade_quote.duckdb"), TABLE_TRADES)
    ohlcv_writer = DuckDBWriter(os.path.join(sym_dir, "ohlcv.duckdb"), TABLE_OHLCV)
    writers = (greek_writer, quote_writer, trade_writer, ohlcv_writer)

    total_g = total_q = total_t = total_ohlcv = 0
    total_chunks = len(date_chunks)

    for chunk_idx, chunk_days in enumerate(date_chunks, 1):
        if not running: break
        log(f"[{symbol}] Chunk {chunk_idx}/{total_chunks}: "
            f"{chunk_days[0]} → {chunk_days[-1]} ({len(chunk_days)} trading days)", "INFO")

        cg, cq, ct, co = await collect_symbol_chunk(
            client, sem, symbol, chunk_days, exp_dates, max_dte,
            writers, progress, live_log=live_log,
            expiry_concurrency=expiry_concurrency,
        )
        total_g += cg; total_q += cq; total_t += ct; total_ohlcv += co

        if chunk_idx < total_chunks and running:
            log(f"[{symbol}] Chunk {chunk_idx} done. Pausing {CHUNK_PAUSE_SECS}s …", "INFO")
            await asyncio.sleep(CHUNK_PAUSE_SECS)

    await progress.save()

    log(f"[{symbol}] ── DONE ──", "SUCCESS")
    log(f"  greeks_all rows : {total_g:,}", "INFO")
    log(f"  quotes rows     : {total_q:,}", "INFO")
    log(f"  trades rows     : {total_t:,}", "INFO")
    log(f"  ohlcv rows      : {total_ohlcv:,}", "INFO")
    log(f"  DB: {sym_dir}/greeks_all.duckdb", "INFO")
    log(f"  DB: {sym_dir}/trade_quote.duckdb (tables: quotes, trades)", "INFO")
    log(f"  DB: {sym_dir}/ohlcv.duckdb", "INFO")

    for w in writers:
        w.close()

    return total_g, total_q, total_t, total_ohlcv

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

async def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    today = datetime.now(TIMEZONE).date()
    end_date = parse_date(args.end_date) or today - timedelta(days=args.lag_days)
    start_date = parse_date(args.start_date) or end_date - timedelta(days=args.days - 1)
    trading_days = get_trading_days(start_date, end_date)

    sem = asyncio.Semaphore(args.concurrency)

    print("\n" + "=" * 70)
    print(" Historical Options Data Collector v1.6 — 3-Year / DuckDB")
    print("=" * 70)
    print(f"  Symbols    : {', '.join(symbols)}")
    print(f"  Date range : {start_date} → {end_date} ({len(trading_days)} trading days)")
    print(f"  Safety lag : today - {args.lag_days} days")
    print(f"  Max DTE    : {args.dte}")
    print(f"  Interval   : {INTERVAL}")
    print(f"  Endpoints  : greeks/all | quote | trade | ohlc")
    print(f"  Filters    : greeks(vega/bid/ask≠0) | quotes(bid_size>100 & ask_size>100) | trades(none) | ohlcv(none)")
    print(f"  OI fetch   : DISABLED")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Expiry conc.: {max(1, int(args.expiry_concurrency))}")
    print(f"  Chunk size : {CHUNK_WEEKS} weeks")
    print(f"  Chunk pause: {CHUNK_PAUSE_SECS}s")
    print(f"  Output     : {os.path.abspath(outdir)}/{{SYMBOL}}/{{greeks_all|trade_quote|ohlcv}}.duckdb")
    print(f"  Resume     : {'ON' if not args.no_resume else 'OFF'}")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    date_chunks = get_date_chunks(trading_days, CHUNK_WEEKS)
    log(f"Date range split into {len(date_chunks)} chunk(s) of ~{CHUNK_WEEKS} weeks each.", "INFO")

    async with httpx.AsyncClient() as client:
        progress = ProgressTracker(
            os.path.join(outdir, ".progress.json"),
            enabled=not args.no_resume,
        )

        sym_tasks = [
            collect_symbol_all(
                client, sem, sym, date_chunks, args.dte,
                outdir, progress,
                live_log=not args.no_live_log,
                startup_delay=i * 2.0,
                expiry_concurrency=max(1, int(args.expiry_concurrency)),
            )
            for i, sym in enumerate(symbols)
        ]

        results = await asyncio.gather(*sym_tasks, return_exceptions=True)

    print("\n" + "=" * 70)
    print(" COLLECTION COMPLETE")
    print("=" * 70)
    grand_g = grand_q = grand_t = grand_o = 0
    for sym, r in zip(symbols, results):
        if isinstance(r, Exception):
            log(f"[{sym}] Failed: {r}", "ERROR")
        else:
            cg, cq, ct, co = r
            grand_g += cg; grand_q += cq; grand_t += ct; grand_o += co
            print(f"  [{sym}] greeks={cg:,}  quotes={cq:,}  trades={ct:,}  ohlcv={co:,}")
    print(f"\n  TOTAL: greeks={grand_g:,}  quotes={grand_q:,}  trades={grand_t:,}  ohlcv={grand_o:,}")
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
