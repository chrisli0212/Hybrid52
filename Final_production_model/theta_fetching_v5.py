#!/usr/bin/env python3
# ========================================
# THETA FETCHING v5.2 - DASHBOARD-READY (DTE-AWARE)
# Location: /workspace/theta_fetching_v5.pyw
# ========================================
#
# Data architecture:
#   daily_data/theta_agg.csv        ← append-only, all batches (dashboard time-series)
#   daily_data/theta_snapshot.csv   ← overwritten each batch (dashboard strike analysis)
#   daily_data/theta_archive/       ← full historical, rotated (for AI/backtesting)
#
# Run standalone: python theta_fetching_v5.py
# Or controlled by dashboard via subprocess

import time
import os
import sys
import signal
import httpx
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

# ==============================
# CONFIG
# ==============================

BASE_URL = "http://144.202.59.33:25503"
SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "VIXW", "TLT"]
FORMAT = "csv"
MAX_DTE = {
    "SPXW": 5,
    "SPY": 5,
    "QQQ": 5,
    "IWM": 5,
    "TLT": 5,
    "VIXW": 30,  # VIX weeklies expire weekly — need wider DTE window
}
TIMEOUT = 30
MAX_RETRIES = 2
SLEEP_SECONDS = 10

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_data")
ARCHIVE_DIR = os.path.join(OUTDIR, "theta_archive")
TIMEZONE = ZoneInfo("America/New_York")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Dashboard files (legacy compatibility; endpoint archives are primary)
AGG_FILE = os.path.join(OUTDIR, "theta_agg.csv")
AGG_FILE_0DTE = os.path.join(OUTDIR, "theta_agg_0dte.csv")
AGG_FILE_01DTE = os.path.join(OUTDIR, "theta_agg_0_1dte.csv")
AGG_FILE_02DTE = os.path.join(OUTDIR, "theta_agg_0_2dte.csv")
SNAPSHOT_FILE = os.path.join(OUTDIR, "theta_snapshot.csv")
SNAPSHOT_FILE_AI = os.path.join(OUTDIR, "theta_snapshot_0_1dte.csv")  # 0-1 DTE for AI/models
SNAPSHOT_DIR = os.path.join(OUTDIR, "snapshots")
SNAPSHOT_HISTORY_FILE = os.path.join(OUTDIR, "theta_snapshot_history.csv")
STATUS_FILE = os.path.join(OUTDIR, ".fetcher_status")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
# Single-batch snapshots for prediction (overwritten each successful fetch; not archive/test-only).
MODEL_GREEKS_FILE = os.path.join(OUTDIR, "theta_model_greeks.csv")
MODEL_TRADE_QUOTE_FILE = os.path.join(OUTDIR, "theta_model_trade_quote.csv")
KEEP_SNAPSHOT_FILES = 800  # ~6.6 hours at 30s interval
MAX_HISTORY_ROWS = 50000  # Cap snapshot history file

# Archive files (timestamped, rotated)
SESSION_TS = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
ARCHIVE_FILE = os.path.join(ARCHIVE_DIR, f"theta_options_{SESSION_TS}.csv")
ARCHIVE_AGG_FILE = os.path.join(ARCHIVE_DIR, f"theta_agg_{SESSION_TS}.csv")
MAX_ARCHIVE_BYTES = 10 * 1024 * 1024
ARCHIVE_PART = 1
ENDPOINT_ARCHIVE_PARTS = {
    "quotes": 1,
    "greeks": 1,
    "trades": 1,
    "ohlc": 1,
    "oi": 1,
}

STRIKE = "*"
RIGHT = "both"
# Deprecated merged-delta filter params retained for backward compatibility.
DELTA_MIN = 0.05
DELTA_MAX = 0.95

COLUMNS_TO_DROP = [
    'bid_condition', 'ask_condition',
    'vera', 'speed', 'zomma', 'dual_gamma', 'dual_delta',
]

FINAL_COLUMNS_TO_DROP = [
    'symbol_quote', 'symbol_greeks', 'symbol_trade',
    'bid_greeks', 'ask_greeks', 'timestamp_greeks',
    'sequence', 'condition', 'size', 'iv_error',
    'd1', 'd2', 'ultima', 'color', 'veta', 'vomma',
    'epsilon', 'rho',
]

# ==============================
# STATE
# ==============================

prev_batch = pd.DataFrame()
oi_cache = {}
oi_fetched = False
running = True

def handle_signal(signum, frame):
    global running
    running = False
    write_status("stopped")
    print("\n✋ Graceful shutdown...")

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

# ==============================
# STATUS FILE (for dashboard)
# ==============================

def write_status(status, batch_id=0, extra=""):
    try:
        with open(STATUS_FILE, "w") as f:
            ts = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
            f.write(f"{status}|{batch_id}|{ts}|{os.getpid()}|{extra}")
    except Exception:
        pass

# ==============================
# ARCHIVE ROTATION
# ==============================

def current_archive_path():
    global ARCHIVE_PART
    if ARCHIVE_PART == 1:
        return ARCHIVE_FILE
    base, ext = os.path.splitext(ARCHIVE_FILE)
    return f"{base}_part{ARCHIVE_PART}{ext}"

def rotate_archive_if_needed():
    global ARCHIVE_PART
    path = current_archive_path()
    try:
        if os.path.exists(path) and os.path.getsize(path) >= MAX_ARCHIVE_BYTES:
            ARCHIVE_PART += 1
    except OSError:
        pass
    return current_archive_path()

def _endpoint_archive_base(endpoint):
    return os.path.join(ARCHIVE_DIR, f"theta_{endpoint}_{SESSION_TS}.csv")

def current_endpoint_archive_path(endpoint):
    part = ENDPOINT_ARCHIVE_PARTS.get(endpoint, 1)
    base = _endpoint_archive_base(endpoint)
    if part == 1:
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}_part{part}{ext}"

def rotate_endpoint_archive_if_needed(endpoint):
    part = ENDPOINT_ARCHIVE_PARTS.get(endpoint, 1)
    path = current_endpoint_archive_path(endpoint)
    try:
        if os.path.exists(path) and os.path.getsize(path) >= MAX_ARCHIVE_BYTES:
            part += 1
            ENDPOINT_ARCHIVE_PARTS[endpoint] = part
    except OSError:
        pass
    return current_endpoint_archive_path(endpoint)

def write_endpoint_archive(endpoint, df):
    if df is None or df.empty:
        return 0
    out_path = rotate_endpoint_archive_if_needed(endpoint)
    header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    df.to_csv(out_path, mode="a", header=header, index=False)
    return len(df)

# ==============================
# UTILITIES
# ==============================

def parse_expiration(exp_str):
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(exp_str.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None

def filter_expirations_by_dte(expirations, max_dte):
    today = datetime.now(TIMEZONE).date()
    filtered = []
    for exp in expirations:
        exp_date = parse_expiration(exp)
        if exp_date is None:
            continue
        dte = (exp_date - today).days
        if 0 <= dte <= max_dte:
            filtered.append((exp, dte))
    filtered.sort(key=lambda x: x[1])
    return [exp for exp, _ in filtered]

def normalize_strikes(*dfs):
    for df in dfs:
        if df is not None and not df.empty and "strike" in df.columns:
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

# ==============================
# HTTP HELPER
# ==============================

def fetch_with_retry(client, url, params, label=""):
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = client.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200 and r.text.strip():
                return r.text
            if r.status_code >= 500 and attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return None
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            print(f"  [{label}] failed: {e}")
            return None
        except Exception as e:
            print(f"  [{label}] error: {e}")
            return None
    return None

def _parse_response(text):
    if text is None:
        return pd.DataFrame()
    df = pd.read_csv(StringIO(text))
    df = df.replace([np.inf, -np.inf], np.nan)
    if "strike" in df.columns and "right" in df.columns:
        df = df.dropna(subset=["strike", "right"])
    return df

# ==============================
# THETA DATA FETCHERS
# ==============================

def fetch_expirations(client, symbol):
    url = f"{BASE_URL}/v3/option/list/expirations"
    text = fetch_with_retry(client, url, {"symbol": symbol, "format": FORMAT}, f"{symbol} Exp")
    if text is None:
        return []
    df = pd.read_csv(StringIO(text))
    return list(df["expiration"].dropna().astype(str))

def fetch_quotes(client, symbol, expiration):
    url = f"{BASE_URL}/v3/option/snapshot/quote"
    return _parse_response(fetch_with_retry(client, url,
        {"symbol": symbol, "expiration": expiration, "strike": STRIKE, "right": RIGHT, "format": FORMAT},
        f"{symbol} Q {expiration}"))

def fetch_greeks(client, symbol, expiration):
    url = f"{BASE_URL}/v3/option/snapshot/greeks/all"
    return _parse_response(fetch_with_retry(client, url,
        {"symbol": symbol, "expiration": expiration, "strike": STRIKE, "right": RIGHT, "format": FORMAT},
        f"{symbol} G {expiration}"))

def fetch_trades(client, symbol, expiration):
    url = f"{BASE_URL}/v3/option/snapshot/trade"
    return _parse_response(fetch_with_retry(client, url,
        {"symbol": symbol, "expiration": expiration, "strike": STRIKE, "right": RIGHT, "format": FORMAT},
        f"{symbol} T {expiration}"))

def fetch_ohlc(client, symbol, expiration):
    url = f"{BASE_URL}/v3/option/snapshot/ohlc"
    return _parse_response(fetch_with_retry(client, url,
        {"symbol": symbol, "expiration": expiration, "strike": STRIKE, "right": RIGHT, "format": FORMAT},
        f"{symbol} O {expiration}"))

def fetch_open_interest(client, symbol, expiration):
    url = f"{BASE_URL}/v3/option/snapshot/open_interest"
    return _parse_response(fetch_with_retry(client, url,
        {"symbol": symbol, "expiration": expiration, "strike": STRIKE, "right": RIGHT, "format": FORMAT},
        f"{symbol} OI {expiration}"))

# ==============================
# FIND ATM STRIKE
# ==============================

def find_atm_strike(greeks_df):
    if greeks_df.empty or "strike" not in greeks_df.columns:
        return None
    spot_col = next((c for c in greeks_df.columns if 'underlying_price' in c.lower()), None)
    if spot_col:
        spot = pd.to_numeric(greeks_df[spot_col], errors="coerce").dropna()
        if not spot.empty:
            strikes = pd.to_numeric(greeks_df["strike"], errors="coerce").dropna()
            if not strikes.empty:
                return float(strikes.iloc[(strikes - spot.iloc[0]).abs().argsort().iloc[0]])
    return None

# ==============================
# LOCAL FILTERS
# ==============================

def filter_for_csv(merged_df):
    if merged_df.empty:
        return merged_df
    df = merged_df.copy()

    delta_col = next((c for c in df.columns if c.lower() == 'delta'), None)
    if delta_col:
        df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
        df = df[df[delta_col].abs().between(DELTA_MIN, DELTA_MAX)]

    bid_col = next((c for c in df.columns if 'bid' in c.lower()
                    and 'size' not in c.lower() and 'exchange' not in c.lower()
                    and 'condition' not in c.lower()), None)
    ask_col = next((c for c in df.columns if 'ask' in c.lower()
                    and 'size' not in c.lower() and 'exchange' not in c.lower()
                    and 'condition' not in c.lower()), None)
    if bid_col and ask_col:
        df[bid_col] = pd.to_numeric(df[bid_col], errors="coerce")
        df[ask_col] = pd.to_numeric(df[ask_col], errors="coerce")
        df = df[(df[bid_col] != 0) & df[bid_col].notna()
                & (df[ask_col] != 0) & df[ask_col].notna()]

    vega_col = next((c for c in df.columns if c.lower() == 'vega'), None)
    if vega_col:
        df[vega_col] = pd.to_numeric(df[vega_col], errors="coerce")
        df = df[(df[vega_col] != 0) & df[vega_col].notna()]

    return df

def _find_first_col(df, candidates):
    if df is None or df.empty:
        return None
    by_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        c = by_lower.get(str(cand).lower())
        if c is not None:
            return c
    return None

def _coerce_num(df, col):
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

def filter_greeks_endpoint(gdf):
    if gdf is None or gdf.empty:
        return pd.DataFrame()
    df = gdf.copy()
    vega_col = _find_first_col(df, ["vega"])
    bid_col = _find_first_col(df, ["bid", "bid_greeks", "bid_quote"])
    ask_col = _find_first_col(df, ["ask", "ask_greeks", "ask_quote"])
    vega = _coerce_num(df, vega_col)
    bid = _coerce_num(df, bid_col)
    ask = _coerce_num(df, ask_col)
    mask = vega.notna() & bid.notna() & ask.notna() & (vega != 0) & (bid != 0) & (ask != 0)
    return df[mask].copy()

def filter_trade_quote_endpoint(df_in):
    if df_in is None or df_in.empty:
        return pd.DataFrame()
    # Quote and trade streams: no row filter (preserve API rows as-is).
    return df_in.copy()

def _tag_endpoint(df, endpoint, symbol, expiration, batch_id, ts):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["endpoint"] = endpoint
    out["symbol"] = out["symbol"] if "symbol" in out.columns else symbol
    out["expiration"] = out["expiration"] if "expiration" in out.columns else expiration
    out["batch_id"] = int(batch_id)
    out["ts"] = ts
    return out


def _latest_by_contract(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    keys = [k for k in ["symbol", "expiration", "strike", "right"] if k in df.columns]
    if not keys:
        return pd.DataFrame()
    keep = keys + [c for c in cols if c in df.columns]
    out = df[keep].copy()
    return out.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def _compute_atm_strikes(gdf: pd.DataFrame) -> dict:
    out = {}
    if gdf is None or gdf.empty:
        return out
    for (sym, exp), sdf in gdf.groupby(["symbol", "expiration"]):
        spot_col = _find_first_col(sdf, ["underlying_price", "spot"])
        if spot_col is None:
            continue
        spot_vals = pd.to_numeric(sdf[spot_col], errors="coerce").dropna()
        strike_vals = pd.to_numeric(sdf.get("strike"), errors="coerce").dropna()
        if spot_vals.empty or strike_vals.empty:
            continue
        spot = float(spot_vals.iloc[-1])
        nearest = float(strike_vals.iloc[(strike_vals - spot).abs().argsort().iloc[0]])
        out[(str(sym), str(exp))] = nearest
    return out


def _build_dashboard_snapshot(greeks_df, quotes_df, trades_df, batch_id, now_ts):
    if greeks_df is None or greeks_df.empty:
        return pd.DataFrame()
    keys = [k for k in ["symbol", "expiration", "strike", "right"] if k in greeks_df.columns]
    if len(keys) < 4:
        return pd.DataFrame()

    g = greeks_df.copy()
    q = _latest_by_contract(quotes_df, ["bid", "ask", "bid_size", "ask_size", "volume"])
    t = _latest_by_contract(trades_df, ["price", "size", "volume", "trade_timestamp"])

    if not q.empty:
        q = q.rename(columns={c: f"{c}_q" for c in q.columns if c not in keys})
    if not t.empty:
        t = t.rename(columns={c: f"{c}_t" for c in t.columns if c not in keys})

    merged = g
    if not q.empty:
        merged = merged.merge(q, on=keys, how="left")
    if not t.empty:
        merged = merged.merge(t, on=keys, how="left")

    # Prefer trade volume where present, else quote volume, else size fallback.
    if "volume_t" in merged.columns:
        merged["volume"] = pd.to_numeric(merged["volume_t"], errors="coerce")
        if "volume_q" in merged.columns:
            vq = pd.to_numeric(merged["volume_q"], errors="coerce")
            merged["volume"] = merged["volume"].where(merged["volume"].notna(), vq)
    elif "volume_q" in merged.columns:
        merged["volume"] = pd.to_numeric(merged["volume_q"], errors="coerce")

    if "volume" not in merged.columns or merged["volume"].isna().all():
        if "size_t" in merged.columns:
            merged["volume"] = pd.to_numeric(merged["size_t"], errors="coerce")
        elif "size_q" in merged.columns:
            merged["volume"] = pd.to_numeric(merged["size_q"], errors="coerce")

    atm_strikes = _compute_atm_strikes(g)
    snap = enrich_for_ai(merged, batch_id, now_ts, atm_strikes)
    return snap

# ==============================
# ENRICHMENT
# ==============================

def enrich_for_ai(merged_df, batch_id, now_timestamp, atm_strikes):
    global prev_batch
    if merged_df.empty:
        return merged_df
    df = merged_df.copy()

    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    rename_map = {}
    for col in df.columns:
        if col.endswith('_q'):
            rename_map[col] = col[:-2] + '_quote'
        elif col.endswith('_g'):
            rename_map[col] = col[:-2] + '_greeks'
        elif col.endswith('_t'):
            rename_map[col] = col[:-2] + '_trade'
    if rename_map:
        df = df.rename(columns=rename_map)

    df["batch_id"] = batch_id
    df["ts"] = now_timestamp

    underlying_col = next((c for c in df.columns if 'underlying_price' in c.lower()), None)
    if underlying_col:
        df["spot"] = pd.to_numeric(df[underlying_col], errors="coerce")
        strike_vals = pd.to_numeric(df["strike"], errors="coerce")
        df["moneyness"] = strike_vals / df["spot"]
        df["dist_atm_pct"] = ((strike_vals - df["spot"]) / df["spot"]) * 100

    df["atm_strike"] = df.apply(
        lambda r: atm_strikes.get((r.get("symbol", ""), str(r.get("expiration", ""))), np.nan), axis=1)

    if "expiration" in df.columns:
        today = datetime.now(TIMEZONE).date()
        df["dte"] = df["expiration"].apply(
            lambda x: (parse_expiration(str(x)) - today).days if parse_expiration(str(x)) else np.nan)

    bid_col = next((c for c in df.columns if 'bid' in c.lower() and 'quote' in c.lower()), None)
    if bid_col is None:
        bid_col = next((c for c in df.columns if c.lower() == 'bid'), None)
    ask_col = next((c for c in df.columns if 'ask' in c.lower() and 'quote' in c.lower()), None)
    if ask_col is None:
        ask_col = next((c for c in df.columns if c.lower() == 'ask'), None)
    if bid_col and ask_col:
        bid_vals = pd.to_numeric(df[bid_col], errors="coerce")
        ask_vals = pd.to_numeric(df[ask_col], errors="coerce")
        df["mid"] = (bid_vals + ask_vals) / 2
        df["spread"] = ask_vals - bid_vals
        df["spread_pct"] = (df["spread"] / df["mid"]) * 100

    if oi_cache:
        def _oi_lookup(row):
            strike_val = pd.to_numeric(row.get("strike", np.nan), errors="coerce")
            exp_parsed = parse_expiration(str(row.get("expiration", "")))
            exp_key = exp_parsed.strftime("%Y-%m-%d") if exp_parsed else str(row.get("expiration", ""))
            key = (str(row.get("symbol", "")),
                   exp_key,
                   str(float(strike_val)) if pd.notna(strike_val) else "",
                   str(row.get("right", "")))
            return oi_cache.get(key, np.nan)
        df["oi"] = df.apply(_oi_lookup, axis=1)

    if "right" in df.columns:
        df["cp_sign"] = df["right"].apply(lambda x: 1 if str(x).upper().startswith("C") else -1)

    if "oi" in df.columns:
        oi_vals = pd.to_numeric(df["oi"], errors="coerce")
        cp_sign_vals = df["cp_sign"] if "cp_sign" in df.columns else 1
        spot_vals = pd.to_numeric(df["spot"], errors="coerce") if "spot" in df.columns else 1
        for greek in ["gamma", "vega", "theta", "delta"]:
            gcol = next((c for c in df.columns if greek in c.lower()
                         and 'exp' not in c.lower()), None)
            if gcol:
                greek_vals = pd.to_numeric(df[gcol], errors="coerce")
                if greek == "delta":
                    df[f"{greek}_exp"] = greek_vals * oi_vals * 100
                elif greek == "theta":
                    # Negate: API theta is negative (buyer's cost); dealer perspective is positive (premium collected)
                    df[f"{greek}_exp"] = -greek_vals * oi_vals * 100
                elif greek == "gamma":
                    # Dealer GEX: gamma × OI × spot × 100 × cp_sign
                    # cp_sign (+1 call, -1 put) gives dealer perspective:
                    #   Calls: +gamma×OI×spot×100 (dealers short calls → long gamma hedge)
                    #   Puts:  -gamma×OI×spot×100 (dealers short puts → short gamma hedge)
                    df[f"{greek}_exp"] = greek_vals * oi_vals * spot_vals * 100 * cp_sign_vals
                else:
                    df[f"{greek}_exp"] = greek_vals * oi_vals * cp_sign_vals * 100

    iv_col = next((c for c in df.columns if 'implied_vol' in c.lower()), None)
    track_cols = []
    if iv_col:
        track_cols.append(("iv", iv_col))
    if "mid" in df.columns:
        track_cols.append(("mid", "mid"))
    if "spread" in df.columns:
        track_cols.append(("spread", "spread"))

    merge_keys = ["symbol", "expiration", "strike", "right"]

    if prev_batch is not None and not prev_batch.empty and track_cols:
        chg_df = df[merge_keys].copy()
        for short_name, col_name in track_cols:
            chg_df[f"_curr_{short_name}"] = pd.to_numeric(df[col_name], errors="coerce")
        merged_chg = chg_df.merge(prev_batch, on=merge_keys, how="left")
        for short_name, _ in track_cols:
            curr = merged_chg[f"_curr_{short_name}"]
            prev_val = merged_chg.get(f"_prev_{short_name}", pd.Series(np.nan, index=merged_chg.index))
            df[f"{short_name}_chg"] = (curr - prev_val).round(4)
    elif track_cols:
        for short_name, _ in track_cols:
            df[f"{short_name}_chg"] = np.nan

    if track_cols:
        new_prev = df[merge_keys].copy()
        for short_name, col_name in track_cols:
            new_prev[f"_prev_{short_name}"] = pd.to_numeric(df[col_name], errors="coerce")
        prev_batch = new_prev
    else:
        prev_batch = pd.DataFrame()

    drop_cols = [c for c in df.columns if c != 'spot' and (
        'underlying_price' in c.lower() or 'bid_exchange' in c.lower() or 'ask_exchange' in c.lower())]
    final_drop = [c for c in FINAL_COLUMNS_TO_DROP if c in df.columns]
    drop_cols = list(set(drop_cols + final_drop))
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    lead_cols = [c for c in ["symbol", "expiration", "strike", "right", "dte", "cp_sign"] if c in df.columns]
    rest_cols = [c for c in df.columns if c not in lead_cols]
    df = df[lead_cols + rest_cols]

    sort_cols = [c for c in ["symbol", "expiration", "strike", "right"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df

# ==============================
# AGGREGATES
# ==============================

def compute_batch_aggregates(df, batch_id, now_timestamp, dte_group="all"):
    if df.empty:
        return pd.DataFrame()
    agg_rows = []
    for symbol in df["symbol"].unique():
        sdf = df[df["symbol"] == symbol]
        spot = sdf["spot"].dropna().iloc[0] if "spot" in sdf.columns and not sdf["spot"].dropna().empty else np.nan
        calls = sdf[sdf["cp_sign"] == 1] if "cp_sign" in sdf.columns else pd.DataFrame()
        puts = sdf[sdf["cp_sign"] == -1] if "cp_sign" in sdf.columns else pd.DataFrame()

        vol_col = next((c for c in sdf.columns if 'volume' in c.lower()), None)
        call_vol = pd.to_numeric(calls[vol_col], errors="coerce").sum() if vol_col and not calls.empty else 0
        put_vol = pd.to_numeric(puts[vol_col], errors="coerce").sum() if vol_col and not puts.empty else 0
        pc_ratio = round(put_vol / call_vol, 3) if call_vol > 0 else np.nan

        # Net premium: mid * volume * 100 (options multiplier)
        mid_col = "mid" if "mid" in sdf.columns else None
        if mid_col and vol_col:
            c_prem = (pd.to_numeric(calls[mid_col], errors="coerce") * pd.to_numeric(calls[vol_col], errors="coerce") * 100).sum() if not calls.empty else 0
            p_prem = (pd.to_numeric(puts[mid_col], errors="coerce") * pd.to_numeric(puts[vol_col], errors="coerce") * 100).sum() if not puts.empty else 0
            call_premium = round(c_prem, 0)
            put_premium = round(p_prem, 0)
            net_premium = round(c_prem - p_prem, 0)
        else:
            call_premium = put_premium = net_premium = np.nan

        gex_col = next((c for c in sdf.columns if 'gamma_exp' in c.lower()), None)
        if gex_col:
            call_gex = pd.to_numeric(calls[gex_col], errors="coerce").sum() if not calls.empty else 0
            put_gex = pd.to_numeric(puts[gex_col], errors="coerce").sum() if not puts.empty else 0
            net_gex = round(call_gex + put_gex, 2)
        else:
            net_gex = np.nan

        iv_col = next((c for c in sdf.columns if 'implied_vol' in c.lower()), None)
        call_iv = pd.to_numeric(calls[iv_col], errors="coerce").mean() if iv_col and not calls.empty else np.nan
        put_iv = pd.to_numeric(puts[iv_col], errors="coerce").mean() if iv_col and not puts.empty else np.nan

        atm_straddle = np.nan
        if "mid" in sdf.columns and "atm_strike" in sdf.columns:
            atm_val = sdf["atm_strike"].dropna().iloc[0] if not sdf["atm_strike"].dropna().empty else None
            if atm_val:
                atm_c = sdf[(sdf["strike"] == atm_val) & (sdf["cp_sign"] == 1)]
                atm_p = sdf[(sdf["strike"] == atm_val) & (sdf["cp_sign"] == -1)]
                c_mid = pd.to_numeric(atm_c["mid"], errors="coerce").iloc[0] if not atm_c.empty else 0
                p_mid = pd.to_numeric(atm_p["mid"], errors="coerce").iloc[0] if not atm_p.empty else 0
                atm_straddle = round(c_mid + p_mid, 2)

        # Quote/trade microstructure metrics
        traded = sdf[sdf.get("volume", pd.Series(dtype=float)) > 0] if "volume" in sdf.columns else pd.DataFrame()
        near_atm = sdf[sdf["dist_atm_pct"].abs() < 5] if "dist_atm_pct" in sdf.columns else sdf

        # Spread: volume-weighted avg spread_pct near ATM
        if "spread_pct" in near_atm.columns and not near_atm.empty:
            w = pd.to_numeric(near_atm.get("volume", pd.Series([1]*len(near_atm))), errors="coerce").fillna(1)
            sp = pd.to_numeric(near_atm["spread_pct"], errors="coerce")
            avg_spread_pct = round(float(np.average(sp.fillna(0), weights=w)), 3) if w.sum() > 0 else np.nan
        else:
            avg_spread_pct = np.nan

        # Bid/ask size imbalance: volume-weighted
        bs_col = _find_first_col(traded, ["bid_size", "bid_size_quote"])
        ak_col = _find_first_col(traded, ["ask_size", "ask_size_quote"])
        if not traded.empty and bs_col and ak_col:
            bs = pd.to_numeric(traded[bs_col], errors="coerce").fillna(0)
            ak = pd.to_numeric(traded[ak_col], errors="coerce").fillna(0)
            total = bs + ak
            imb = ((bs - ak) / total.replace(0, np.nan)).fillna(0)
            vol_w = pd.to_numeric(traded["volume"], errors="coerce").fillna(1)
            bid_ask_imbalance = round(float(np.average(imb, weights=vol_w)), 4) if vol_w.sum() > 0 else np.nan
        else:
            bid_ask_imbalance = np.nan

        # Trade aggression: (price - mid) / spread, volume-weighted
        price_col = _find_first_col(traded, ["price", "price_trade"])
        if not traded.empty and price_col and "mid" in traded.columns and "spread" in traded.columns:
            price_v = pd.to_numeric(traded[price_col], errors="coerce")
            mid_v = pd.to_numeric(traded["mid"], errors="coerce")
            spread_v = pd.to_numeric(traded["spread"], errors="coerce").replace(0, np.nan)
            agg_v = ((price_v - mid_v) / spread_v).fillna(0)
            vol_w2 = pd.to_numeric(traded["volume"], errors="coerce").fillna(1)
            trade_aggression = round(float(np.average(agg_v.clip(-2, 2), weights=vol_w2)), 4) if vol_w2.sum() > 0 else np.nan
        else:
            trade_aggression = np.nan

        # Average trade size
        count_col = _find_first_col(traded, ["count", "size_trade", "size"])
        if not traded.empty and count_col:
            total_vol = pd.to_numeric(traded["volume"], errors="coerce").sum()
            total_count = pd.to_numeric(traded[count_col], errors="coerce").sum()
            avg_trade_size = round(total_vol / total_count, 2) if total_count > 0 else np.nan
        else:
            avg_trade_size = np.nan

        agg_rows.append({
            "batch_id": batch_id, "ts": now_timestamp, "dte_group": dte_group, "symbol": symbol,
            "spot": spot, "n_contracts": len(sdf),
            "call_vol": call_vol, "put_vol": put_vol, "pc_ratio": pc_ratio,
            "net_gex": net_gex,
            "call_premium": call_premium, "put_premium": put_premium, "net_premium": net_premium,
            "call_iv": round(call_iv, 4) if pd.notna(call_iv) else np.nan,
            "put_iv": round(put_iv, 4) if pd.notna(put_iv) else np.nan,
            "iv_skew": round(put_iv - call_iv, 4) if pd.notna(put_iv) and pd.notna(call_iv) else np.nan,
            "atm_straddle": atm_straddle,
            "avg_spread_pct": avg_spread_pct,
            "bid_ask_imbalance": bid_ask_imbalance,
            "trade_aggression": trade_aggression,
            "avg_trade_size": avg_trade_size,
        })
    return pd.DataFrame(agg_rows)

# ==============================
# SNAPSHOT RUNNER
# ==============================

def run_options_snapshot(client, batch_count):
    global oi_cache, oi_fetched

    now_ny = datetime.now(TIMEZONE)
    now_ny_str = now_ny.strftime("%Y-%m-%d %H:%M:%S %Z")
    greek_frames = []
    quote_frames = []
    trade_frames = []
    ohlc_frames = []
    oi_frames = []
    agg_seed_rows = []

    for symbol in SYMBOLS:
        expirations = fetch_expirations(client, symbol)
        if not expirations:
            continue
        symbol_max_dte = MAX_DTE.get(symbol, 5) if isinstance(MAX_DTE, dict) else MAX_DTE
        expirations = filter_expirations_by_dte(expirations, symbol_max_dte)
        if not expirations:
            continue

        for exp in expirations:
            qdf = fetch_quotes(client, symbol, exp)
            gdf = fetch_greeks(client, symbol, exp)
            ohlc_df = fetch_ohlc(client, symbol, exp)
            tdf = fetch_trades(client, symbol, exp)
            normalize_strikes(qdf, gdf, ohlc_df, tdf)

            # Endpoint-specific filtering rules (no cross-endpoint merging).
            qdf_f = filter_trade_quote_endpoint(qdf)
            gdf_f = filter_greeks_endpoint(gdf)
            tdf_f = filter_trade_quote_endpoint(tdf)
            ohlc_f = ohlc_df.copy() if ohlc_df is not None else pd.DataFrame()

            if not oi_fetched:
                oi_df = fetch_open_interest(client, symbol, exp)
                if not oi_df.empty:
                    normalize_strikes(oi_df)
                    oi_col_name = next((c for c in oi_df.columns if 'open_interest' in c.lower()), None)
                    if oi_col_name:
                        for _, row in oi_df.iterrows():
                            exp_normalized = parse_expiration(str(row.get("expiration", exp)))
                            exp_key = exp_normalized.strftime("%Y-%m-%d") if exp_normalized else str(row.get("expiration", exp))
                            key = (symbol, exp_key,
                                   str(float(row["strike"])) if pd.notna(row.get("strike")) else "",
                                   str(row.get("right", "")))
                            oi_cache[key] = row[oi_col_name]
                    oi_frames.append(_tag_endpoint(oi_df, "oi", symbol, exp, batch_count, now_ny_str))

            if not gdf_f.empty:
                greek_frames.append(_tag_endpoint(gdf_f, "greeks", symbol, exp, batch_count, now_ny_str))
                # Minimal aggregate seed from greeks only (preserves per-symbol spot + IV stats).
                agg_seed_rows.append(_tag_endpoint(gdf_f, "greeks", symbol, exp, batch_count, now_ny_str))
            if not qdf_f.empty:
                quote_frames.append(_tag_endpoint(qdf_f, "quotes", symbol, exp, batch_count, now_ny_str))
            if not tdf_f.empty:
                trade_frames.append(_tag_endpoint(tdf_f, "trades", symbol, exp, batch_count, now_ny_str))
            if not ohlc_f.empty:
                ohlc_frames.append(_tag_endpoint(ohlc_f, "ohlc", symbol, exp, batch_count, now_ny_str))

    if not oi_fetched:
        oi_fetched = True

    # Write endpoint-specific human archives only (no merged snapshot archives).
    n_g = write_endpoint_archive("greeks", pd.concat(greek_frames, ignore_index=True) if greek_frames else pd.DataFrame())
    n_q = write_endpoint_archive("quotes", pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame())
    n_t = write_endpoint_archive("trades", pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame())
    n_o = write_endpoint_archive("ohlc", pd.concat(ohlc_frames, ignore_index=True) if ohlc_frames else pd.DataFrame())
    n_oi = write_endpoint_archive("oi", pd.concat(oi_frames, ignore_index=True) if oi_frames else pd.DataFrame())

    if (n_g + n_q + n_t + n_o + n_oi) == 0:
        write_status("running", batch_count, "no_data")
        return

    # Rolling model inputs: prediction_service loads these before theta_archive parts.
    df_g_model = pd.concat(greek_frames, ignore_index=True) if greek_frames else pd.DataFrame()
    tq_model_parts = quote_frames + trade_frames
    df_tq_model = pd.concat(tq_model_parts, ignore_index=True) if tq_model_parts else pd.DataFrame()
    if not df_g_model.empty:
        df_g_model.to_csv(MODEL_GREEKS_FILE, index=False)
        if not df_tq_model.empty:
            df_tq_model.to_csv(MODEL_TRADE_QUOTE_FILE, index=False)
        elif os.path.exists(MODEL_TRADE_QUOTE_FILE):
            try:
                os.remove(MODEL_TRADE_QUOTE_FILE)
            except OSError:
                pass

    # Dashboard/human snapshot: merged strike-level view (independent from model files).
    df_dash_snapshot = _build_dashboard_snapshot(
        df_g_model, pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame(),
        pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(),
        batch_count, now_ny_str,
    )
    if not df_dash_snapshot.empty:
        df_dash_snapshot.to_csv(SNAPSHOT_FILE, index=False)
        # Keep 0-1 DTE helper file expected by dashboard filters.
        if "dte" in df_dash_snapshot.columns:
            dte_vals = pd.to_numeric(df_dash_snapshot["dte"], errors="coerce")
            df_dash_snapshot.loc[dte_vals.between(0, 1, inclusive="both")].to_csv(
                SNAPSHOT_FILE_AI, index=False
            )
        else:
            df_dash_snapshot.to_csv(SNAPSHOT_FILE_AI, index=False)

        snap_batch_file = os.path.join(SNAPSHOT_DIR, f"snapshot_{int(batch_count):06d}.csv")
        df_dash_snapshot.to_csv(snap_batch_file, index=False)
        if KEEP_SNAPSHOT_FILES > 0:
            existing = sorted(
                [p for p in os.listdir(SNAPSHOT_DIR) if p.startswith("snapshot_") and p.endswith(".csv")]
            )
            excess = len(existing) - KEEP_SNAPSHOT_FILES
            for old_name in existing[:max(0, excess)]:
                try:
                    os.remove(os.path.join(SNAPSHOT_DIR, old_name))
                except OSError:
                    pass

        # Optional append-only history for quick troubleshooting.
        hist_header = not os.path.exists(SNAPSHOT_HISTORY_FILE) or os.path.getsize(SNAPSHOT_HISTORY_FILE) == 0
        df_dash_snapshot.to_csv(SNAPSHOT_HISTORY_FILE, mode="a", header=hist_header, index=False)

    # Build light aggregate output from greek seed rows for backward compatibility.
    if agg_seed_rows:
        agg_src = pd.concat(agg_seed_rows, ignore_index=True)
    else:
        agg_src = pd.DataFrame()

    def _build_agg_from_greeks(df):
        if df is None or df.empty:
            return pd.DataFrame()
        out_rows = []
        now_d = datetime.now(TIMEZONE).date()
        for sym, sdf in df.groupby("symbol"):
            spot_col = _find_first_col(sdf, ["underlying_price", "spot"])
            spot = float(pd.to_numeric(sdf[spot_col], errors="coerce").dropna().iloc[-1]) if spot_col and not pd.to_numeric(sdf[spot_col], errors="coerce").dropna().empty else 0.0
            right_col = _find_first_col(sdf, ["right"])
            iv_col = _find_first_col(sdf, ["implied_vol"])
            call_iv = np.nan
            put_iv = np.nan
            if right_col and iv_col:
                rr = sdf[right_col].astype(str).str.upper()
                call_iv = pd.to_numeric(sdf.loc[rr.str.startswith("C"), iv_col], errors="coerce").mean()
                put_iv = pd.to_numeric(sdf.loc[rr.str.startswith("P"), iv_col], errors="coerce").mean()
            exp_col = _find_first_col(sdf, ["expiration"])
            dte_val = np.nan
            if exp_col and not sdf[exp_col].dropna().empty:
                ex = parse_expiration(str(sdf[exp_col].dropna().iloc[-1]))
                if ex:
                    dte_val = (ex - now_d).days
            out_rows.append({
                "batch_id": int(batch_count),
                "ts": now_ny_str,
                "dte_group": "all",
                "symbol": sym,
                "spot": spot,
                "n_contracts": int(len(sdf)),
                "call_vol": np.nan,
                "put_vol": np.nan,
                "pc_ratio": np.nan,
                "net_gex": np.nan,
                "call_premium": np.nan,
                "put_premium": np.nan,
                "net_premium": np.nan,
                "call_iv": round(call_iv, 4) if pd.notna(call_iv) else np.nan,
                "put_iv": round(put_iv, 4) if pd.notna(put_iv) else np.nan,
                "iv_skew": round(put_iv - call_iv, 4) if pd.notna(put_iv) and pd.notna(call_iv) else np.nan,
                "atm_straddle": np.nan,
                "avg_spread_pct": np.nan,
                "bid_ask_imbalance": np.nan,
                "trade_aggression": np.nan,
                "avg_trade_size": np.nan,
                "dte": dte_val,
            })
        return pd.DataFrame(out_rows)

    if not df_dash_snapshot.empty:
        agg_df = compute_batch_aggregates(df_dash_snapshot, int(batch_count), now_ny_str, dte_group="all")
    else:
        agg_df = _build_agg_from_greeks(agg_src)
    if not agg_df.empty:
        header = not os.path.exists(AGG_FILE) or os.path.getsize(AGG_FILE) == 0
        agg_df.to_csv(AGG_FILE, mode="a", header=header, index=False)

    # === WRITE: Archive agg (append, per-session) ===
    if not agg_df.empty:
        agg_header = not os.path.exists(ARCHIVE_AGG_FILE) or os.path.getsize(ARCHIVE_AGG_FILE) == 0
        agg_df.to_csv(ARCHIVE_AGG_FILE, mode="a", header=agg_header, index=False)

    total_rows = n_g + n_q + n_t + n_o + n_oi
    write_status("running", batch_count, f"{total_rows} endpoint rows")
    print(
        f"[Batch {batch_count}] endpoint rows g={n_g} q={n_q} t={n_t} ohlc={n_o} oi={n_oi} "
        f"| model→{MODEL_GREEKS_FILE} + {MODEL_TRADE_QUOTE_FILE} "
        f"| dashboard→{SNAPSHOT_FILE} | agg→{AGG_FILE} | archive→{ARCHIVE_DIR}"
    )

# ==============================
# MAIN LOOP
# ==============================

def main_loop():
    batch_count = 1
    global prev_batch
    prev_batch = pd.DataFrame()

    print(f"=== Theta Fetching v5.2 (DTE-aware) ===")
    print(f"PID: {os.getpid()}")
    print(f"Agg (compat): {AGG_FILE}")
    print(f"Model (pred): {MODEL_GREEKS_FILE} + {MODEL_TRADE_QUOTE_FILE}")
    print(f"Archive Dir:  {ARCHIVE_DIR}")
    print(f"Arch Agg:     {ARCHIVE_AGG_FILE}")
    print(f"Endpoint cap: {MAX_ARCHIVE_BYTES // (1024*1024)}MB per file")
    print(f"Symbols:  {', '.join(SYMBOLS)}")
    print(f"Interval: {SLEEP_SECONDS}s\n")

    write_status("running", 0, "starting")

    with httpx.Client() as client:
        while running:
            try:
                run_options_snapshot(client, batch_count)
            except Exception as exc:
                print(f"[Batch {batch_count}] Error: {exc}")
                write_status("error", batch_count, str(exc)[:100])

            batch_count += 1
            for _ in range(SLEEP_SECONDS):
                if not running:
                    break
                time.sleep(1)

    write_status("stopped")
    print("Fetcher stopped.")

if __name__ == "__main__":
    if any(a in ("--once", "-1", "once") for a in sys.argv[1:]):
        with httpx.Client(timeout=TIMEOUT) as client:
            run_options_snapshot(client, 1)
        print("Single batch complete.")
        sys.exit(0)
    main_loop()
