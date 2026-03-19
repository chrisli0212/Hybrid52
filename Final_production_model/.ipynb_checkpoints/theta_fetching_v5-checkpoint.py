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
MAX_DTE = 5
TIMEOUT = 30
MAX_RETRIES = 2
SLEEP_SECONDS = 10

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_data")
ARCHIVE_DIR = os.path.join(OUTDIR, "theta_archive")
TIMEZONE = ZoneInfo("America/New_York")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Dashboard files (fixed names - easy to find)
AGG_FILE = os.path.join(OUTDIR, "theta_agg.csv")
AGG_FILE_0DTE = os.path.join(OUTDIR, "theta_agg_0dte.csv")
AGG_FILE_01DTE = os.path.join(OUTDIR, "theta_agg_0_1dte.csv")
AGG_FILE_02DTE = os.path.join(OUTDIR, "theta_agg_0_2dte.csv")
SNAPSHOT_FILE = os.path.join(OUTDIR, "theta_snapshot.csv")
SNAPSHOT_FILE_AI = os.path.join(OUTDIR, "theta_snapshot_0_1dte.csv")  # 0-1 DTE for AI/models
SNAPSHOT_DIR = os.path.join(OUTDIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
SNAPSHOT_HISTORY_FILE = os.path.join(OUTDIR, "theta_snapshot_history.csv")
STATUS_FILE = os.path.join(OUTDIR, ".fetcher_status")
KEEP_SNAPSHOT_FILES = 800  # ~6.6 hours at 30s interval
MAX_HISTORY_ROWS = 50000  # Cap snapshot history file

# Archive files (timestamped, rotated)
SESSION_TS = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
ARCHIVE_FILE = os.path.join(ARCHIVE_DIR, f"theta_options_{SESSION_TS}.csv")
ARCHIVE_AGG_FILE = os.path.join(ARCHIVE_DIR, f"theta_agg_{SESSION_TS}.csv")
MAX_ARCHIVE_BYTES = 10 * 1024 * 1024
ARCHIVE_PART = 1

STRIKE = "*"
RIGHT = "both"
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
            key = (str(row.get("symbol", "")),
                   str(row.get("expiration", "")),
                   str(float(strike_val)) if pd.notna(strike_val) else "",
                   str(row.get("right", "")))
            return oi_cache.get(key, np.nan)
        df["oi"] = df.apply(_oi_lookup, axis=1)

    if "oi" in df.columns:
        oi_vals = pd.to_numeric(df["oi"], errors="coerce")
        for greek in ["gamma", "vega", "theta", "delta"]:
            gcol = next((c for c in df.columns if greek in c.lower()
                         and 'exp' not in c.lower()), None)
            if gcol:
                df[f"{greek}_exp"] = pd.to_numeric(df[gcol], errors="coerce") * oi_vals

    if "right" in df.columns:
        df["cp_sign"] = df["right"].apply(lambda x: 1 if str(x).upper().startswith("C") else -1)

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
            net_gex = round(call_gex - abs(put_gex), 2)
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
        if not traded.empty and "bid_size" in traded.columns and "ask_size" in traded.columns:
            bs = pd.to_numeric(traded["bid_size"], errors="coerce").fillna(0)
            ak = pd.to_numeric(traded["ask_size"], errors="coerce").fillna(0)
            total = bs + ak
            imb = ((bs - ak) / total.replace(0, np.nan)).fillna(0)
            vol_w = pd.to_numeric(traded["volume"], errors="coerce").fillna(1)
            bid_ask_imbalance = round(float(np.average(imb, weights=vol_w)), 4) if vol_w.sum() > 0 else np.nan
        else:
            bid_ask_imbalance = np.nan

        # Trade aggression: (price - mid) / spread, volume-weighted
        if not traded.empty and "price" in traded.columns and "mid" in traded.columns and "spread" in traded.columns:
            price_v = pd.to_numeric(traded["price"], errors="coerce")
            mid_v = pd.to_numeric(traded["mid"], errors="coerce")
            spread_v = pd.to_numeric(traded["spread"], errors="coerce").replace(0, np.nan)
            agg_v = ((price_v - mid_v) / spread_v).fillna(0)
            vol_w2 = pd.to_numeric(traded["volume"], errors="coerce").fillna(1)
            trade_aggression = round(float(np.average(agg_v.clip(-2, 2), weights=vol_w2)), 4) if vol_w2.sum() > 0 else np.nan
        else:
            trade_aggression = np.nan

        # Average trade size
        if not traded.empty and "count" in traded.columns:
            total_vol = pd.to_numeric(traded["volume"], errors="coerce").sum()
            total_count = pd.to_numeric(traded["count"], errors="coerce").sum()
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

    merged_frames = []
    atm_strikes = {}

    for symbol in SYMBOLS:
        expirations = fetch_expirations(client, symbol)
        if not expirations:
            continue
        expirations = filter_expirations_by_dte(expirations, MAX_DTE)
        if not expirations:
            continue

        for exp in expirations:
            qdf = fetch_quotes(client, symbol, exp)
            gdf = fetch_greeks(client, symbol, exp)
            if qdf.empty or gdf.empty:
                continue

            ohlc_df = fetch_ohlc(client, symbol, exp)
            tdf = fetch_trades(client, symbol, exp)
            normalize_strikes(qdf, gdf, ohlc_df, tdf)

            if not oi_fetched:
                oi_df = fetch_open_interest(client, symbol, exp)
                if not oi_df.empty:
                    normalize_strikes(oi_df)
                    oi_col_name = next((c for c in oi_df.columns if 'open_interest' in c.lower()), None)
                    if oi_col_name:
                        for _, row in oi_df.iterrows():
                            key = (symbol, str(row.get("expiration", exp)),
                                   str(float(row["strike"])) if pd.notna(row.get("strike")) else "",
                                   str(row.get("right", "")))
                            oi_cache[key] = row[oi_col_name]

            atm = find_atm_strike(gdf)
            if atm is not None:
                atm_strikes[(symbol, str(exp))] = atm

            keys = ["strike", "right", "expiration"]
            merged = pd.merge(qdf, gdf, on=keys, how="inner", suffixes=("_q", "_g"))

            if not ohlc_df.empty:
                try:
                    merged = pd.merge(merged, ohlc_df, on=keys, how="left", suffixes=("", "_ohlc"))
                except Exception:
                    pass

            if tdf is not None and not tdf.empty:
                try:
                    merged = pd.merge(merged, tdf, on=keys, how="left", suffixes=("", "_t"))
                except Exception:
                    pass

            if merged.empty:
                continue

            merged["symbol"] = symbol
            merged = filter_for_csv(merged)
            if not merged.empty:
                merged_frames.append(merged)

    if not oi_fetched:
        oi_fetched = True

    if not merged_frames:
        write_status("running", batch_count, "no_data")
        return

    final_df = pd.concat(merged_frames, ignore_index=True)
    final_df = enrich_for_ai(final_df, batch_count, now_ny_str, atm_strikes)

    # === WRITE 1: Aggregate (append) - ALL DTE ===
    agg_df = compute_batch_aggregates(final_df, batch_count, now_ny_str, dte_group="all")
    if not agg_df.empty:
        header = not os.path.exists(AGG_FILE) or os.path.getsize(AGG_FILE) == 0
        agg_df.to_csv(AGG_FILE, mode="a", header=header, index=False)

    # === WRITE 1b: DTE-split aggregates (v5.2) ===
    if "dte" in final_df.columns:
        _dte_vals = pd.to_numeric(final_df["dte"], errors="coerce")

        # 0DTE only
        _df_0dte = final_df[_dte_vals == 0]
        if not _df_0dte.empty:
            _agg_0 = compute_batch_aggregates(_df_0dte, batch_count, now_ny_str, dte_group="0dte")
            if not _agg_0.empty:
                _hdr = not os.path.exists(AGG_FILE_0DTE) or os.path.getsize(AGG_FILE_0DTE) == 0
                _agg_0.to_csv(AGG_FILE_0DTE, mode="a", header=_hdr, index=False)

        # 0-1 DTE (for credit spread decisions + AI)
        _df_01dte = final_df[_dte_vals <= 1]
        if not _df_01dte.empty:
            _agg_01 = compute_batch_aggregates(_df_01dte, batch_count, now_ny_str, dte_group="0_1dte")
            if not _agg_01.empty:
                _hdr = not os.path.exists(AGG_FILE_01DTE) or os.path.getsize(AGG_FILE_01DTE) == 0
                _agg_01.to_csv(AGG_FILE_01DTE, mode="a", header=_hdr, index=False)

        # 0-2 DTE
        _df_02dte = final_df[_dte_vals <= 2]
        if not _df_02dte.empty:
            _agg_02 = compute_batch_aggregates(_df_02dte, batch_count, now_ny_str, dte_group="0_2dte")
            if not _agg_02.empty:
                _hdr = not os.path.exists(AGG_FILE_02DTE) or os.path.getsize(AGG_FILE_02DTE) == 0
                _agg_02.to_csv(AGG_FILE_02DTE, mode="a", header=_hdr, index=False)

    # === WRITE 2: Snapshot (overwrite - dashboard reads this, ALL DTE) ===
    final_df.to_csv(SNAPSHOT_FILE, index=False)

    # === WRITE 2-AI: 0-1 DTE snapshot for AI/model upload (v5.2) ===
    if "dte" in final_df.columns:
        _dte_snap = pd.to_numeric(final_df["dte"], errors="coerce")
        _df_ai = final_df[_dte_snap <= 1]
        if not _df_ai.empty:
            _df_ai.to_csv(SNAPSHOT_FILE_AI, index=False)

    # === WRITE 2b: Snapshot per-batch file (for time comparisons) ===
    snap_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{batch_count:06d}.csv")
    final_df.to_csv(snap_path, index=False)

    # Cleanup old snapshot files
    try:
        if KEEP_SNAPSHOT_FILES is not None:
            old_batch = batch_count - KEEP_SNAPSHOT_FILES
            if old_batch > 0:
                old_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{old_batch:06d}.csv")
                if os.path.exists(old_path):
                    os.remove(old_path)
    except Exception:
        pass

    # === WRITE 2c: Snapshot history (append - for dashboard comparisons) ===
    hist_header = not os.path.exists(SNAPSHOT_HISTORY_FILE) or os.path.getsize(SNAPSHOT_HISTORY_FILE) == 0
    final_df.to_csv(SNAPSHOT_HISTORY_FILE, mode="a", header=hist_header, index=False)

    # Trim history file if too large (every 100 batches)
    if batch_count % 100 == 0:
        try:
            if os.path.exists(SNAPSHOT_HISTORY_FILE):
                hist = pd.read_csv(SNAPSHOT_HISTORY_FILE)
                if len(hist) > MAX_HISTORY_ROWS:
                    hist = hist.tail(MAX_HISTORY_ROWS)
                    hist.to_csv(SNAPSHOT_HISTORY_FILE, index=False)
        except Exception:
            pass

    # === WRITE 3: Archive options (append, rotated) ===
    archive_path = rotate_archive_if_needed()
    header = not os.path.exists(archive_path) or os.path.getsize(archive_path) == 0
    final_df.to_csv(archive_path, mode="a", header=header, index=False)

    # === WRITE 4: Archive agg (append, per-session) ===
    if not agg_df.empty:
        agg_header = not os.path.exists(ARCHIVE_AGG_FILE) or os.path.getsize(ARCHIVE_AGG_FILE) == 0
        agg_df.to_csv(ARCHIVE_AGG_FILE, mode="a", header=agg_header, index=False)

    _n0 = len(final_df[pd.to_numeric(final_df.get("dte", pd.Series()), errors="coerce") == 0]) if "dte" in final_df.columns else 0
    _n01 = len(final_df[pd.to_numeric(final_df.get("dte", pd.Series()), errors="coerce") <= 1]) if "dte" in final_df.columns else 0
    write_status("running", batch_count, f"{len(final_df)} rows")
    print(f"[Batch {batch_count}] {len(final_df)} rows (0DTE:{_n0}, 0-1DTE:{_n01}, all:{len(final_df)}) | agg→{AGG_FILE} | snap→{SNAPSHOT_FILE}")

# ==============================
# MAIN LOOP
# ==============================

def main_loop():
    batch_count = 1
    global prev_batch
    prev_batch = pd.DataFrame()

    print(f"=== Theta Fetching v5.2 (DTE-aware) ===")
    print(f"PID: {os.getpid()}")
    print(f"Agg:         {AGG_FILE}")
    print(f"Snapshot:    {SNAPSHOT_FILE}")
    print(f"Snap History:{SNAPSHOT_HISTORY_FILE}")
    print(f"Snap Dir:    {SNAPSHOT_DIR}")
    print(f"Archive:     {ARCHIVE_DIR}")
    print(f"Arch Agg:    {ARCHIVE_AGG_FILE}")
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
    main_loop()
