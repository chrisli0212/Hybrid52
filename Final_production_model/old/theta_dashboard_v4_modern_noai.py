"""
Theta Options Intelligence Dashboard — Modern Pro Terminal  (STANDALONE)

Reads three CSV files:
  - daily_data/theta_agg.csv       (market data)
  - daily_data/theta_snapshot.csv  (strike-level data)
  - daily_data/prediction.csv      (model predictions from prediction_service.py)

NO PyTorch dependency. NO model loading. Pure CSV reading + Dash rendering.
Modern Bloomberg/SpotGamma-inspired design with glassmorphism cards.

STANDALONE: All chart/data/insight functions from theta_dashboard_v3_10.py
are inlined below. No external module dependency.
"""

import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import signal
import os
import time
from datetime import datetime
import sys
import traceback
from collections import deque
import warnings
warnings.filterwarnings('ignore')
import json
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Script-relative paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "daily_data"

AGG_FILE       = DATA_DIR / "theta_agg.csv"
SNAPSHOT_FILE  = DATA_DIR / "theta_snapshot.csv"
SNAPSHOT_DIR   = DATA_DIR / "snapshots"
STATUS_FILE    = DATA_DIR / ".fetcher_status"
FETCHER_SCRIPT = SCRIPT_DIR / "theta_fetching_v5.py"
DEBUG_LOG_PATH = Path("/workspace/.cursor/debug-e5f32e.log")

MAX_HISTORY    = 200


def _debug_log(hypothesis_id, location, message, data=None, run_id="run1"):
    """Append NDJSON debug log line for runtime hypothesis testing."""
    # #region agent log
    try:
        payload = {
            "sessionId": "e5f32e",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as e:
        # Keep this visible while in debug mode so we can see logger failures.
        try:
            print(f"[DEBUG_LOG_WRITE_ERROR] {e}")
        except Exception:
            pass
    # #endregion

# Per-agent decision thresholds from SPXW h30 training (F1-optimized on validation set).
# These are the correct "neutral" points: prob >= threshold means the agent signals BULL.
# All agents share positive_class_prior=0.5321 (53.21% bull in training data).
# Source: results/stage1/SPXW_h30_results.json + checkpoint optimal_threshold fields.
AGENT_TRAIN_MEDIAN = {
    "A": 0.35, "B": 0.37, "C": 0.43,
    "K": 0.38, "T": 0.43, "Q": 0.38, "2D": 0.38,
}
# Fraction of bull labels in training — true baseline output expected at market neutral.
AGENT_BULL_PRIOR = 0.5321
# Stage 3 LogReg coefficients — reflect true model importance per agent
AGENT_S3_COEF = {
    "A": 0.764, "B": 0.748, "C": 0.297,
    "K": 1.207, "T": 0.065, "Q": 0.354, "2D": 0.161,
}
# Cross-feature coefficients: [mean, std, spread, agree_up, max, min]
S3_CROSS_COEF = [0.514, 0.015, 0.029, 0.815, 0.503, 0.475]
S3_INTERCEPT = -3.2518

# Stage 3 model baseline — prob output when all agents are exactly at 0.5 (true neutral input).
# Derived from model intercept=-3.2518 + sum(coef*0.5). ~0.43 because the large negative
# intercept suppresses outputs; real market is ~53% bull, so prob > 0.43 = net bullish signal.
S3_NEUTRAL = 0.43

# Colors aligned with theta_dashboard_v3_10.py (production reference)
MC = {
    "bg_dark":       "#0f172a",
    "bg_card":       "#1e293b",
    "bg_card_hover": "#283548",
    "bg_input":      "#334155",
    "border":        "rgba(59,130,246,0.15)",
    "border_active": "rgba(59,130,246,0.4)",
    "text":          "#f1f5f9",
    "text_sec":      "#cbd5e1",
    "text_muted":    "#94a3b8",
    "accent":        "#3b82f6",
    "accent_glow":   "rgba(59,130,246,0.2)",
    "call":          "#10b981",
    "put":           "#ef4444",
    "warning":       "#f59e0b",
    "info":          "#3b82f6",
    "grid":          "#334155",
    "neutral":       "#3b82f6",
}

# ══════════════════════════════════════════════════════════════════════════════
# INLINED FROM theta_dashboard_v3_10.py — ALL chart/data/insight functions
# ══════════════════════════════════════════════════════════════════════════════

# Market hours window (New York / Eastern Time)
# 30 min before regular open (8:30 AM) to 30 min after close (5:00 PM)
ET = ZoneInfo("America/New_York")


def _now_et_naive():
    """Return current time in Eastern Time as a naive datetime (no tzinfo).
    All CSV timestamps are naive-ET, so this ensures consistent comparisons."""
    return datetime.now(ET).replace(tzinfo=None)

def _hex_to_rgba(hex_color, alpha=0.1):
    """Convert a hex color string to rgba() with the given alpha.
    Plotly does not accept 8-char hex (#RRGGBBAA), only 6-char or rgba()."""
    hc = hex_color.lstrip('#')
    if len(hc) == 8:
        hc = hc[:6]
    r, g, b = int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


MARKET_OPEN_ET = (8, 30)    # 8:30 AM ET (30 min before open)
MARKET_CLOSE_ET = (17, 0)   # 5:00 PM ET (30 min after close)


def filter_market_hours(df, ts_col='_ts_parsed'):
    """Filter DataFrame to only include rows within market hours window (8:30-17:00 ET).
    Includes 30 min before/after market session. Rows outside this window are dropped.
    Works on any df with a datetime column."""
    if df is None or df.empty:
        return df
    if ts_col not in df.columns or df[ts_col].isna().all():
        return df  # can't filter without timestamps
    # Convert to ET for comparison
    ts = df[ts_col].copy()
    if ts.dt.tz is None:
        # Assume naive timestamps are already ET (fetcher runs in NY market hours)
        ts_et = ts
    else:
        ts_et = ts.dt.tz_convert(ET)
    # Build boolean mask: hour:minute between MARKET_OPEN_ET and MARKET_CLOSE_ET
    time_minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
    open_minutes = MARKET_OPEN_ET[0] * 60 + MARKET_OPEN_ET[1]   # 510
    close_minutes = MARKET_CLOSE_ET[0] * 60 + MARKET_CLOSE_ET[1]  # 1020
    mask = (time_minutes >= open_minutes) & (time_minutes <= close_minutes)
    filtered = df[mask].copy()
    return filtered if not filtered.empty else df  # fallback to unfiltered if nothing passes

# ========================================
# CONFIG  (uses script-relative paths from top of file)
# ========================================
# FETCHER_SCRIPT, DATA_DIR, AGG_FILE, SNAPSHOT_FILE, SNAPSHOT_DIR,
# STATUS_FILE  are already defined at lines 42-49 relative to __file__.
# Do NOT redefine them here.

REFRESH_INTERVAL = 10
# MAX_HISTORY already defined above

# ============================================================================
# DEBUG MODE - Set to True to see filtering diagnostics in console
# ============================================================================
DEBUG_FILTER = False  # Set to True to debug time filtering issues


C = MC  # Alias so v3 functions can reference C[]


LAYOUT_DEFAULTS = dict(
    paper_bgcolor=C['bg_card'],
    plot_bgcolor=C['bg_card'],
    font=dict(color=C['text_sec'], family='system-ui', size=11),
    hovermode='x unified',
    showlegend=True,
    legend=dict(x=0.02, y=0.98),
    margin=dict(l=60, r=40, t=30, b=40),
)


def base_layout(**kw):
    return {**LAYOUT_DEFAULTS, **kw}

def style_axes(fig):
    fig.update_xaxes(gridcolor=C['grid'], showgrid=True)
    fig.update_yaxes(gridcolor=C['grid'], showgrid=True)
    return fig


def parse_agg_timestamps(df):
    """Parse the 'ts' column into proper datetime objects for x-axis.
    Returns array of datetime values positioned correctly on a time axis.
    Falls back to batch_id if parsing fails."""
    if 'ts' not in df.columns or df.empty:
        return df['batch_id'].values if 'batch_id' in df.columns else []
    try:
        ts_clean = df['ts'].str.replace(r'\s+[A-Z]{2,5}$', '', regex=True)
        ts_parsed = pd.to_datetime(ts_clean, errors='coerce')
        if ts_parsed.isna().all():
            return df['batch_id'].values
        return ts_parsed.values
    except Exception:
        return df['batch_id'].values


def add_now_annotation(fig, x_vals, row=None, col=None):
    """Draw a 'NOW' vertical line at the latest timestamp."""
    if x_vals is None or len(x_vals) == 0:
        return
    try:
        last_x = x_vals[-1]
        vline_kwargs = dict(
            x=last_x, line_dash="solid", line_color=C['accent'],
            line_width=2, opacity=0.7,
            annotation_text="NOW", annotation_position="top",
            annotation_font_size=10, annotation_font_color=C['accent'],
        )
        if row and col:
            fig.add_vline(**vline_kwargs, row=row, col=col)
        else:
            fig.add_vline(**vline_kwargs)
    except Exception:
        pass


# ========================================
# FETCHER PROCESS CONTROL
# ========================================

fetcher_process = None

def get_fetcher_status():
    try:
        if STATUS_FILE.exists():
            parts = STATUS_FILE.read_text().strip().split("|")
            return {
                "status": parts[0] if len(parts) > 0 else "unknown",
                "batch_id": parts[1] if len(parts) > 1 else "0",
                "timestamp": parts[2] if len(parts) > 2 else "",
                "pid": int(parts[3]) if len(parts) > 3 else 0,
                "extra": parts[4] if len(parts) > 4 else ""
            }
    except Exception:
        pass
    return {"status": "stopped", "batch_id": "0", "timestamp": "", "pid": 0, "extra": ""}

def is_fetcher_running():
    global fetcher_process
    if fetcher_process is not None:
        if fetcher_process.poll() is None:
            return True
        else:
            fetcher_process = None
    status = get_fetcher_status()
    if status["pid"] > 0:
        try:
            os.kill(status["pid"], 0)
            return True
        except (ProcessLookupError, PermissionError):
            pass
    return False

def start_fetcher(btn=None):
    global fetcher_process
    if is_fetcher_running():
        print("Fetcher already running")
        return
    if not FETCHER_SCRIPT.exists():
        print(f"Fetcher script not found: {FETCHER_SCRIPT}")
        return
    import sys
    fetcher_process = subprocess.Popen(
        [sys.executable, str(FETCHER_SCRIPT)],
        stdout=open(DATA_DIR / "fetcher.log", "a"), stderr=subprocess.STDOUT,
        cwd=str(FETCHER_SCRIPT.parent)
    )
    print(f"Fetcher started (PID: {fetcher_process.pid})")

def stop_fetcher(btn=None):
    global fetcher_process
    if fetcher_process is not None and fetcher_process.poll() is None:
        fetcher_process.terminate()
        try:
            fetcher_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            fetcher_process.kill()
        fetcher_process = None
        print("Fetcher stopped")
        return
    status = get_fetcher_status()
    if status["pid"] > 0:
        try:
            os.kill(status["pid"], signal.SIGTERM)
            print(f"Sent SIGTERM to PID {status['pid']}")
        except (ProcessLookupError, PermissionError):
            print("Fetcher not running")
    else:
        print("Fetcher not running")


def delete_all_data(btn=None):
    """Delete all CSV files under DATA_DIR (but not folders) and key status/log files."""
    try:
        removed = 0
        for p in DATA_DIR.rglob("*.csv"):
            try:
                p.unlink()
                removed += 1
            except Exception:
                # Skip files we can't delete
                continue

        # Delete prediction.csv if present
        prediction_file = DATA_DIR / "prediction.csv"
        if prediction_file.exists():
            try:
                prediction_file.unlink()
            except Exception:
                pass

        # Delete status file (.fetcher_status) if present
        status_path = STATUS_FILE
        if status_path.exists():
            try:
                status_path.unlink()
            except Exception:
                pass

        # Delete fetcher.log if present
        log_file = DATA_DIR / "fetcher.log"
        if log_file.exists():
            try:
                log_file.unlink()
            except Exception:
                pass

        print(f"[Delete All] Removed {removed} CSV files, prediction.csv, .fetcher_status, and fetcher.log under {DATA_DIR}")
    except Exception as e:
        print(f"[Delete All] Error while deleting data: {e}")


# ========================================
# DATA LOADERS
# ========================================

# ---------------------------------------------------------------------------
# Column name adapter (fetcher CSV vs training CSV)
# ---------------------------------------------------------------------------
_COLUMN_RENAME_MAP = {
    # Snapshot columns that differ from training names
    "bid_quote": "bid",
    "ask_quote": "ask",
    # Agg columns (in case fetcher uses different naming)
    "callvol": "call_vol",
    "putvol": "put_vol",
    "pcratio": "pc_ratio",
    "ivskew": "iv_skew",
    "atmstraddle": "atm_straddle",
    "calliv": "call_iv",
    "putiv": "put_iv",
    "netgex": "net_gex",
}


def _adapt_csv_columns(df):
    """Rename fetcher CSV columns to match the names expected by the dashboard
    and the prediction service.  Only renames if the target name is absent."""
    if df is None or df.empty:
        return df
    for old, new in _COLUMN_RENAME_MAP.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    # Normalize 'right' column (CALL/PUT -> C/P) if present
    if "right" in df.columns:
        df["right"] = df["right"].astype(str).str.upper().replace({"CALL": "C", "PUT": "P"})
    return df


def load_agg_data():
    if not AGG_FILE.exists():
        return None
    try:
        df = pd.read_csv(AGG_FILE, on_bad_lines='skip')
        if df.empty:
            return None
        df = _adapt_csv_columns(df)
        df['batch_id'] = df['batch_id'].astype(int)
        ts_clean = df['ts'].str.replace(r'\s+[A-Z]{2,5}$', '', regex=True)
        df['_ts_parsed'] = pd.to_datetime(ts_clean, errors='coerce')
        df = df.sort_values(['_ts_parsed', 'symbol'])
        # Filter to market hours only (8:30 AM - 5:00 PM ET)
        df = filter_market_hours(df, '_ts_parsed')
        return df
    except Exception:
        return None

def load_snapshot_data():
    """Load current snapshot. Falls back to latest snapshot file on disk."""
    # Try the live snapshot file first
    if SNAPSHOT_FILE.exists():
        try:
            df = pd.read_csv(SNAPSHOT_FILE)
            if df is not None and not df.empty:
                return _adapt_csv_columns(df)
        except Exception:
            pass
    # Fallback: load the latest snapshot from the snapshots directory
    snaps = list_available_snapshots()
    if snaps:
        try:
            df = pd.read_csv(snaps[-1][2])  # latest = last in sorted list
            if df is not None and not df.empty:
                return _adapt_csv_columns(df)
        except Exception:
            pass
    return None

def list_available_snapshots():
    """Scan snapshot directory and return sorted list of (batch_num, ts_str, filepath) tuples.
    Only includes snapshots within market hours (8:30 AM - 5:00 PM ET)."""
    if not SNAPSHOT_DIR.exists():
        return []
    files = sorted(SNAPSHOT_DIR.glob("snapshot_*.csv"))
    if not files:
        return []
    open_min = MARKET_OPEN_ET[0] * 60 + MARKET_OPEN_ET[1]   # 510
    close_min = MARKET_CLOSE_ET[0] * 60 + MARKET_CLOSE_ET[1]  # 1020
    result = []
    for f in files:
        # Try to extract batch number from filename: snapshot_000042.csv
        stem = f.stem  # e.g. "snapshot_000042"
        try:
            batch_num = int(stem.split("_")[1])
        except (IndexError, ValueError):
            batch_num = 0
        # Get file timestamp and check market hours
        try:
            peek = pd.read_csv(f, nrows=1)
            ts_col = peek.get('ts', peek.get('_ts_parsed', pd.Series()))
            if not ts_col.empty:
                ts_str = str(ts_col.iloc[0])[:19]
                ts_dt = pd.to_datetime(ts_str, errors='coerce')
                if pd.notna(ts_dt):
                    # Check if within market hours (assume timestamps are ET)
                    t_min = ts_dt.hour * 60 + ts_dt.minute
                    if t_min < open_min or t_min > close_min:
                        continue  # skip out-of-hours snapshot
            else:
                # Fallback to file mtime
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=ET).replace(tzinfo=None)
                ts_str = mtime.strftime("%H:%M:%S")
                t_min = mtime.hour * 60 + mtime.minute
                # Can't reliably determine ET from mtime, include it
        except Exception:
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=ET).replace(tzinfo=None)
            ts_str = mtime.strftime("%H:%M:%S")
        result.append((batch_num, ts_str, f))
    return result  # sorted by filename = sorted by batch


def load_snapshot_by_batch(batch_id):
    """Load snapshot by batch_id number — tries exact match first, then nearest."""
    fname = SNAPSHOT_DIR / f"snapshot_{int(batch_id):06d}.csv"
    if fname.exists():
        try:
            df = pd.read_csv(fname)
            return df if not df.empty else None
        except Exception:
            return None
    # Fallback: find nearest available snapshot
    snaps = list_available_snapshots()
    if not snaps:
        return None
    # Find closest batch_id
    target = int(batch_id)
    closest = min(snaps, key=lambda s: abs(s[0] - target))
    try:
        df = pd.read_csv(closest[2])
        return df if not df.empty else None
    except Exception:
        return None


def load_snapshot_by_path(filepath):
    """Load a snapshot CSV by its full path."""
    try:
        df = pd.read_csv(filepath)
        return df if not df.empty else None
    except Exception:
        return None




def find_snapshot_by_time_offset(minutes_ago):
    """Find the snapshot closest to `minutes_ago` minutes before the latest.
    Uses `ts` column for real timestamps, falls back to file mtime."""
    if minutes_ago <= 0:
        return None
    snaps = list_available_snapshots()
    if len(snaps) < 2:
        return None
    latest_path = snaps[-1][2]
    try:
        latest_peek = pd.read_csv(latest_path, nrows=1)
        ts_col = latest_peek.get('ts', pd.Series())
        if not ts_col.empty:
            latest_ts = pd.to_datetime(str(ts_col.iloc[0])[:19], errors='coerce')
        else:
            latest_ts = pd.Timestamp(datetime.fromtimestamp(latest_path.stat().st_mtime, tz=ET).replace(tzinfo=None))
    except Exception:
        latest_ts = pd.Timestamp(datetime.fromtimestamp(latest_path.stat().st_mtime, tz=ET).replace(tzinfo=None))
    if pd.isna(latest_ts):
        return None
    target_ts = latest_ts - pd.Timedelta(minutes=minutes_ago)
    best_path, best_diff = None, float('inf')
    for batchnum, ts_str, fpath in snaps[:-1]:
        try:
            peek = pd.read_csv(fpath, nrows=1)
            ts_val = peek.get('ts', pd.Series())
            if not ts_val.empty:
                snap_ts = pd.to_datetime(str(ts_val.iloc[0])[:19], errors='coerce')
            else:
                snap_ts = pd.Timestamp(datetime.fromtimestamp(fpath.stat().st_mtime, tz=ET).replace(tzinfo=None))
        except Exception:
            snap_ts = pd.Timestamp(datetime.fromtimestamp(fpath.stat().st_mtime, tz=ET).replace(tzinfo=None))
        if pd.isna(snap_ts):
            continue
        diff = abs((snap_ts - target_ts).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_path = fpath
    tolerance_seconds = minutes_ago * 60 * 0.5
    if best_path is not None and best_diff <= tolerance_seconds:
        return load_snapshot_by_path(best_path)
    # Fallback: estimate from batch rate
    if len(snaps) >= 2:
        try:
            first_peek = pd.read_csv(snaps[0][2], nrows=1)
            first_ts_val = first_peek.get('ts', pd.Series())
            if not first_ts_val.empty:
                first_ts = pd.to_datetime(str(first_ts_val.iloc[0])[:19], errors='coerce')
            else:
                first_ts = pd.Timestamp(datetime.fromtimestamp(snaps[0][2].stat().st_mtime, tz=ET).replace(tzinfo=None))
            total_minutes = (latest_ts - first_ts).total_seconds() / 60
            if total_minutes > 0:
                batches_per_min = len(snaps) / total_minutes
                estimated_back = int(minutes_ago * batches_per_min)
                idx = max(0, len(snaps) - 1 - estimated_back)
                return load_snapshot_by_path(snaps[idx][2])
        except Exception:
            pass
    return None

# ========================================
# METRIC CALCULATIONS
# ========================================

def calculate_gex_by_strike(options_df, symbol):
    if options_df is None or options_df.empty:
        return None
    sym = options_df[options_df['symbol'] == symbol].copy()
    if sym.empty or 'gamma_exp' not in sym.columns:
        return None
    gex = sym.groupby('strike').agg({'gamma_exp': 'sum'}).reset_index()
    return gex.sort_values('strike')

def calculate_volume_profile(options_df, symbol):
    if options_df is None or options_df.empty:
        return None
    sym = options_df[options_df['symbol'] == symbol]
    if sym.empty or 'volume' not in sym.columns:
        return None
    return sym.groupby(['strike', 'cp_sign']).agg({'volume': 'sum'}).reset_index()

def calculate_iv_term_structure(options_df, symbol):
    if options_df is None or options_df.empty:
        return None
    sym = options_df[options_df['symbol'] == symbol]
    if sym.empty or 'implied_vol' not in sym.columns or 'dte' not in sym.columns:
        return None
    return sym.groupby(['dte', 'cp_sign']).agg({'implied_vol': 'mean'}).reset_index()

def calculate_vanna_by_strike(options_df, symbol):
    if options_df is None or options_df.empty:
        return None
    sym = options_df[options_df['symbol'] == symbol]
    if sym.empty:
        return None
    if 'vanna' in sym.columns:
        return sym.groupby('strike').agg({'vanna': 'sum'}).reset_index().sort_values('strike')
    if all(c in sym.columns for c in ['gamma', 'implied_vol', 'oi', 'cp_sign']):
        sym = sym.copy()
        sym['vanna_est'] = sym['gamma'] * sym['implied_vol'] * sym['oi']
        return sym.groupby('strike').agg({'vanna_est': 'sum'}).reset_index().rename(
            columns={'vanna_est': 'vanna'}).sort_values('strike')
    return None

def calculate_dealer_greeks(options_df, symbol):
    if options_df is None or options_df.empty:
        return None
    sym = options_df[options_df['symbol'] == symbol]
    if sym.empty:
        return None
    greeks = {}
    for col in ['delta', 'gamma', 'vega', 'theta']:
        if col not in sym.columns or 'oi' not in sym.columns:
            continue
        if col == 'delta':
            greeks[col.title()] = (sym[col] * sym['oi']).sum()
        elif col == 'theta':
            # Negate: API theta is negative for buyers; dealer collected perspective is positive
            greeks[col.title()] = -(sym[col] * sym['oi']).sum()
        else:
            if 'cp_sign' not in sym.columns:
                continue
            greeks[col.title()] = (sym[col] * sym['oi'] * sym['cp_sign']).sum()
    return greeks if greeks else None

def calculate_metrics(snapshot_df, agg_df, symbol):
    result = {
        'spot': 'N/A', 'net_gex': 'N/A', 'pc_ratio': 'N/A',
        'iv_rank': 'N/A', 'call_vol_contracts': 'N/A', 'put_vol_contracts': 'N/A',
        'n_contracts': 'N/A', 'atm_straddle': 'N/A', 'iv_skew': 'N/A',
        'pc_ratio_raw': 1, 'net_gex_raw': 0, 'spot_raw': 0, 'iv_skew_raw': 0
    }
    if agg_df is not None and not agg_df.empty:
        sym_agg = agg_df[agg_df['symbol'] == symbol]
        if not sym_agg.empty:
            row = sym_agg.iloc[-1]
            result['spot'] = f"${row['spot']:.2f}" if 'spot' in row.index else 'N/A'
            result['spot_raw'] = row.get('spot', 0)
            result['net_gex'] = f"{row['net_gex']:,.0f}" if 'net_gex' in row.index else 'N/A'
            result['net_gex_raw'] = row.get('net_gex', 0)
            result['pc_ratio'] = f"{row['pc_ratio']:.3f}" if 'pc_ratio' in row.index else 'N/A'
            result['pc_ratio_raw'] = row.get('pc_ratio', 1)
            result['iv_skew'] = f"{row['iv_skew']:.4f}" if 'iv_skew' in row.index else 'N/A'
            result['iv_skew_raw'] = row.get('iv_skew', 0)
            result['n_contracts'] = f"{int(row['n_contracts']):,}" if 'n_contracts' in row.index else 'N/A'
            result['atm_straddle'] = f"${row['atm_straddle']:.2f}" if 'atm_straddle' in row.index else 'N/A'
    if snapshot_df is not None and not snapshot_df.empty:
        sym = snapshot_df[snapshot_df['symbol'] == symbol]
        if not sym.empty and 'volume' in sym.columns and 'cp_sign' in sym.columns:
            result['call_vol_contracts'] = f"{sym[sym['cp_sign']==1]['volume'].sum():,.0f}"
            result['put_vol_contracts']  = f"{sym[sym['cp_sign']==-1]['volume'].sum():,.0f}"
    # Gamma flip info
    flip_info = calculate_gamma_flip(snapshot_df, symbol)
    if flip_info and flip_info.get('flip_strike'):
        result['gamma_flip'] = f"{flip_info['flip_strike']:,.0f}"
        result['gamma_flip_raw'] = flip_info['flip_strike']
        if flip_info['spot'] > 0:
            dist = (flip_info['spot'] - flip_info['flip_strike']) / flip_info['spot'] * 100
            result['gamma_flip_dist'] = f"{dist:+.1f}%"
        else:
            result['gamma_flip_dist'] = ''
    else:
        result['gamma_flip'] = 'N/A'
        result['gamma_flip_dist'] = ''
    return result


# ========================================
# EMPTY CHART
# ========================================

def empty_chart(msg, height=320):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, color=C['text_muted']))
    fig.update_layout(**base_layout(height=height, showlegend=False))
    return fig


# ========================================
# IMPLICATION + ANOMALY ENGINE
# ========================================

def implication_box_html(text, anomaly=None):
    """Render an insight box below a chart. anomaly = None | 'warn' | 'alert'"""
    if anomaly == 'alert':
        border_c = C['put']
        bg = 'rgba(239,68,68,0.08)'
        icon = '🚨'
    elif anomaly == 'warn':
        border_c = C['warning']
        bg = 'rgba(245,158,11,0.08)'
        icon = '⚠️'
    else:
        border_c = C['accent']
        bg = 'rgba(59,130,246,0.06)'
        icon = '💡'
    return f"""
    <div style="background:{bg};padding:10px 14px;border-radius:6px;margin:-4px 0 14px 0;
                border-left:3px solid {border_c};font-family:system-ui,sans-serif;font-size:0.95em;
                line-height:1.6;color:{C['text_sec']};">
        {icon} {text}
    </div>"""


def pc_ratio_insight(val):
    if val is None or np.isnan(val):
        return None, None
    if val > 1.5:
        return (f"<b>P/C Ratio = {val:.2f} — Extreme put buying.</b> "
                "This often signals heavy hedging or fear. Historically, extreme readings "
                "(&gt;1.5) can act as a contrarian indicator — when everyone hedges, "
                "selloffs may be nearing exhaustion."), 'alert'
    elif val > 1.2:
        return (f"<b>P/C Ratio = {val:.2f} — Elevated put activity.</b> "
                "More puts trading than calls. Suggests rising risk aversion and "
                "portfolio hedging demand. Watch whether this is persistent or a one-off spike."), 'warn'
    elif val < 0.5:
        return (f"<b>P/C Ratio = {val:.2f} — Extreme call dominance.</b> "
                "Very few puts relative to calls. Can signal complacency — "
                "historically extreme lows precede corrections as hedging is absent."), 'warn'
    elif val < 0.7:
        return (f"<b>P/C Ratio = {val:.2f} — Bullish sentiment.</b> "
                "Call volume outpaces puts. Participants expect upside. "
                "Moderate readings are normal in trending markets."), None
    else:
        return (f"<b>P/C Ratio = {val:.2f} — Balanced.</b> "
                "Near-neutral put/call mix. No extreme positioning detected."), None


def gex_insight(val):
    if val is None or np.isnan(val):
        return None, None
    if val > 0:
        return (f"<b>Positive gamma ({val:,.0f}).</b> "
                "Dealers are long gamma — they sell into rallies and buy dips to hedge. "
                "This <em>dampens</em> volatility and creates a stabilizing, range-bound environment. "
                "Expect mean-reversion and lower realized vol."), None
    else:
        severity = 'alert' if val < -1e9 else 'warn'
        return (f"<b>Negative gamma ({val:,.0f}).</b> "
                "Dealers are short gamma — they must chase price (buy into rallies, sell into drops). "
                "This <em>amplifies</em> moves and creates a positive feedback loop. "
                "Expect higher realized vol and larger intraday swings."), severity


def iv_skew_insight(val):
    if val is None or np.isnan(val):
        return None, None
    if val > 0.05:
        return (f"<b>IV Skew = {val:.4f} — Put IV well above Call IV.</b> "
                "Demand for downside protection exceeds upside speculation. "
                "Wide skew often precedes volatile moves or reflects hedging around events (FOMC, earnings)."), 'warn'
    elif val > 0.02:
        return (f"<b>IV Skew = {val:.4f} — Moderate put premium.</b> "
                "Normal state — puts trade at a slight premium to calls due to structural "
                "demand for portfolio hedging."), None
    elif val < -0.02:
        return (f"<b>IV Skew = {val:.4f} — Call IV exceeds Put IV.</b> "
                "Unusual — upside speculation dominates. This can occur during "
                "melt-up rallies or short-squeeze dynamics."), 'warn'
    else:
        return (f"<b>IV Skew = {val:.4f} — Flat.</b> "
                "Balanced IV across calls and puts. No directional fear premium."), None


def straddle_insight(val, spot):
    if val is None or np.isnan(val) or spot is None or spot == 0:
        return None, None
    implied_move_pct = (val / spot) * 100
    if implied_move_pct > 3:
        return (f"<b>ATM Straddle = ${val:.2f} ({implied_move_pct:.1f}% implied move).</b> "
                "Market pricing a large move. High straddle cost means option sellers demand "
                "large premium — typically before events or in high-vol regimes."), 'warn'
    elif implied_move_pct > 1.5:
        return (f"<b>ATM Straddle = ${val:.2f} ({implied_move_pct:.1f}% implied move).</b> "
                "Moderate expected move. Compare to recent realized moves — "
                "if implied exceeds realized, vol sellers have an edge (and vice versa)."), None
    else:
        return (f"<b>ATM Straddle = ${val:.2f} ({implied_move_pct:.1f}% implied move).</b> "
                "Low implied move. Market expects a quiet session. "
                "Historically, compressed vol often precedes expansion."), None


# Chart-level insight generators

def gamma_chart_insight(options_df, symbol, spot):
    gex = calculate_gex_by_strike(options_df, symbol)
    if gex is None or gex.empty:
        return None, None
    total_gex = gex['gamma_exp'].sum()
    max_strike = gex.loc[gex['gamma_exp'].abs().idxmax(), 'strike']
    pos_gex = gex[gex['gamma_exp'] > 0]['gamma_exp'].sum()
    neg_gex = gex[gex['gamma_exp'] < 0]['gamma_exp'].sum()

    flip_candidates = gex.sort_values('strike')
    flip_strike = None
    flips = []
    for i in range(len(flip_candidates) - 1):
        if flip_candidates.iloc[i]['gamma_exp'] * flip_candidates.iloc[i+1]['gamma_exp'] < 0:
            flips.append((flip_candidates.iloc[i]['strike'] + flip_candidates.iloc[i+1]['strike']) / 2)
    if flips:
        flip_strike = min(flips, key=lambda x: abs(x - spot)) if spot else flips[0]

    text = (f"<b>Gamma Exposure Profile:</b> Dealer gamma positioning by strike. "
            "Positive bars = dealers long gamma (will sell rallies, buy dips → stabilizing). "
            "Negative bars = dealers short gamma (will chase price → amplifying). "
            f"Highest concentration at <b>${max_strike:,.0f}</b> strike. ")

    if flip_strike and spot:
        dist = ((spot - flip_strike) / spot) * 100
        text += (f"Gamma flip level near <b>${flip_strike:,.0f}</b> "
                 f"(spot is {abs(dist):.1f}% {'above' if dist > 0 else 'below'}). ")

    anomaly = None
    if total_gex < 0:
        text += "<br><b>Overall negative gamma</b> — amplified moves likely."
        anomaly = 'warn'
    if flip_strike and spot and abs(spot - flip_strike) / spot < 0.005:
        text += "<br><b>Spot is at the gamma flip zone</b> — regime change possible, expect volatility shift."
        anomaly = 'alert'

    return text, anomaly


def strike_chart_insight(options_df, symbol, spot):
    vol = calculate_volume_profile(options_df, symbol)
    if vol is None or vol.empty:
        return None, None
    calls = vol[vol['cp_sign'] == 1]
    puts = vol[vol['cp_sign'] == -1]
    top_call = calls.loc[calls['volume'].idxmax(), 'strike'] if not calls.empty else None
    top_put = puts.loc[puts['volume'].idxmax(), 'strike'] if not puts.empty else None

    text = ("<b>Key Strike Levels:</b> "
            "High volume/OI strikes act as support and resistance. "
            "Dealers must hedge large OI positions, creating 'gravity' that pins price toward these strikes "
            "(especially near expiration). ")
    if top_call:
        text += f"Heaviest call strike: <b>${top_call:,.0f}</b> (acts as resistance/magnet). "
    if top_put:
        text += f"Heaviest put strike: <b>${top_put:,.0f}</b> (acts as support/floor). "

    anomaly = None
    if top_call and top_put and spot:
        range_pct = abs(top_call - top_put) / spot * 100
        if range_pct < 1:
            text += "<br><b>Call and put walls very close</b> — expect tight pinning action."
            anomaly = 'warn'
    return text, anomaly


def iv_chart_insight(options_df, symbol):
    iv = calculate_iv_term_structure(options_df, symbol)
    if iv is None or iv.empty:
        return None, None
    calls = iv[iv['cp_sign'] == 1].sort_values('dte')
    puts = iv[iv['cp_sign'] == -1].sort_values('dte')

    text = ("<b>IV Term Structure:</b> "
            "Shows implied volatility across expirations. "
            "<em>Contango</em> (upward slope) is normal — longer-dated options carry more uncertainty. ")

    anomaly = None
    if not calls.empty and len(calls) >= 2:
        short_iv = calls.iloc[0]['implied_vol']
        long_iv = calls.iloc[-1]['implied_vol']
        if short_iv > long_iv * 1.05:
            text += ("<br><b>BACKWARDATION detected</b> — short-term IV exceeds long-term. "
                     "Market expects imminent volatility (event risk, fear). "
                     "Historically rare and signals elevated near-term risk.")
            anomaly = 'alert'
        else:
            text += "Currently in contango (normal). No imminent event premium detected."
    return text, anomaly


def flow_chart_insight(agg_df, symbol):
    if agg_df is None or agg_df.empty:
        return None, MC["border"], None
    sym = agg_df[agg_df['symbol'] == symbol].tail(MAX_HISTORY)
    if sym.empty:
        return None, None

    text = ("<b>Options Flow:</b> "
            "Tracks call vs put volume/premium over time. "
            "Sustained call dominance suggests bullish positioning; sustained put dominance suggests hedging or bearish bets. "
            "Watch for sudden spikes — they often coincide with institutional block trades or sweeps. ")

    anomaly = None
    cv_col = next((c for c in ['call_vol','callvol'] if c in sym.columns), None)
    pv_col = next((c for c in ['put_vol','putvol'] if c in sym.columns), None)
    if cv_col and pv_col and len(sym) >= 5:
        recent_cv = sym[cv_col].tail(3).mean()
        recent_pv = sym[pv_col].tail(3).mean()
        avg_cv = sym[cv_col].mean()
        avg_pv = sym[pv_col].mean()
        if recent_pv > avg_pv * 2:
            text += "<br><b>Put volume surging</b> — recent put flow is 2x+ above average. Heavy hedging underway."
            anomaly = 'alert'
        elif recent_cv > avg_cv * 2:
            text += "<br><b>Call volume surging</b> — recent call flow is 2x+ above average. Aggressive upside positioning."
            anomaly = 'warn'
    return text, anomaly


def mm_flow_insight(agg_df, symbols):
    if agg_df is None or agg_df.empty:
        return None, MC["border"], None
    text = ("<b>Market Maker Flow Changes:</b> "
            "Shows the batch-over-batch change in net GEX (delta of gamma). "
            "Rising values = dealers accumulating gamma (stabilizing). "
            "Falling values = dealers losing gamma (destabilizing). "
            "Divergence between symbols (e.g., SPX up, IWM down) can signal rotation or relative-value positioning.")
    return text, None


def vanna_chart_insight(options_df, symbol):
    vanna = calculate_vanna_by_strike(options_df, symbol)
    if vanna is None or vanna.empty:
        return None, None

    net_vanna = vanna['vanna'].sum()
    text = ("<b>Vanna Exposure:</b> "
            "Measures how dealer delta changes when implied volatility moves. ")

    if net_vanna > 0:
        text += ("Net positive vanna — if IV drops, dealers must <em>buy</em> the underlying to re-hedge, "
                 "providing a tailwind. If IV rises, they sell, adding pressure. "
                 "Positive vanna + falling vol = supportive for price.")
    else:
        text += ("Net negative vanna — if IV drops, dealers must <em>sell</em> the underlying, "
                 "removing support. If IV rises, they buy. "
                 "Negative vanna + rising vol = supportive for price (but usually chaotic).")

    anomaly = None
    if abs(net_vanna) > 1e10:
        text += "<br><b>Extreme vanna magnitude</b> — vol moves will trigger outsized hedging flows."
        anomaly = 'warn'
    return text, anomaly


def dealer_chart_insight(options_df, symbol):
    greeks = calculate_dealer_greeks(options_df, symbol)
    if greeks is None:
        return None, None

    text = ("<b>Dealer Positioning (estimated net greeks):</b> "
            "<em>Delta</em>: directional exposure — positive = dealers long (will sell to hedge) → bearish pressure. "
            "<em>Gamma</em>: convexity — positive = stabilizing, negative = amplifying. "
            "<em>Vega</em>: vol sensitivity — positive = dealers benefit from IV rise. "
            "<em>Theta</em>: time decay — positive = dealers net time-decay collection. Negative = dealers paying decay (unusual for short-vol book).")

    anomaly = None
    gamma_val = greeks.get('Gamma', 0)
    if gamma_val < 0:
        text += "<br>Dealers have <b>negative net gamma</b> — they are chasing price, amplifying moves."
        anomaly = 'warn'
    return text, anomaly


def multi_gamma_insight(agg_df):
    if agg_df is None or agg_df.empty:
        return None, MC["border"], None
    latest = agg_df.groupby('symbol').tail(1)
    if 'net_gex' not in latest.columns:
        return None, None

    neg_count = (latest['net_gex'] < 0).sum()
    total = len(latest)
    text = ("<b>Cross-Symbol Gamma Comparison:</b> "
            "Compares net dealer gamma across all tracked symbols. "
            "When most symbols are positive gamma, broad market tends to be stable. "
            "When most are negative, expect correlated vol expansion.")

    anomaly = None
    if neg_count >= total * 0.7 and total >= 2:
        text += (f"<br><b>{neg_count}/{total} symbols in negative gamma</b> — "
                 "broad negative gamma regime. Correlated selloffs more likely.")
        anomaly = 'alert'
    return text, anomaly


def multi_sentiment_insight(agg_df):
    if agg_df is None or agg_df.empty:
        return None, MC["border"], None
    text = ("<b>Cross-Symbol Sentiment:</b> "
            "Compares call vs put volume (or P/C ratio) across symbols. "
            "Uniform bullish/bearish readings suggest broad consensus. "
            "Divergence (e.g., calls on SPX, puts on IWM) suggests rotation or hedged positioning.")
    return text, None


def vix_put_flow_insight(agg_df):
    if agg_df is None or agg_df.empty:
        return None, MC["border"], None
    text = ("<b>VIX Put Flow:</b> "
            "Tracks put volume on VIX options over time. "
            "Heavy VIX put buying = institutions selling volatility (expecting calm). "
            "Heavy VIX call buying = institutions buying crash protection (expecting turmoil). "
            "Sudden spikes in VIX put volume often precede complacency traps.")

    anomaly = None
    vix = agg_df[agg_df['symbol'].isin(['VIXW', 'VIX'])]
    if 'VIXW' in vix['symbol'].values:
        vix = vix[vix['symbol'] == 'VIXW'].tail(MAX_HISTORY)
    pv_col = next((c for c in ['put_vol','putvol'] if c in vix.columns), None)
    if pv_col and len(vix) >= 5:
        recent = vix[pv_col].tail(3).mean()
        avg = vix[pv_col].mean()
        if recent > avg * 2.5:
            text += "<br><b>VIX put volume surging</b> — heavy institutional hedging activity detected."
            anomaly = 'alert'
    return text, anomaly


def vix_hedging_insight(options_df):
    if options_df is None or options_df.empty:
        return None, None
    vix = options_df[options_df['symbol'].isin(['VIXW', 'VIX'])]
    # Prefer VIXW
    if 'VIXW' in vix['symbol'].values:
        vix = vix[vix['symbol'] == 'VIXW']
    if vix.empty:
        return None, None
    puts = vix[vix['cp_sign'] == -1]
    text = ("<b>VIX Hedging Intensity:</b> "
            "Shows where institutional VIX put volume concentrates by strike. "
            "High vol/OI ratio (&gt;20%) at a strike = fresh aggressive positioning (not legacy OI). "
            "Clustered buying at low strikes = institutions expect VIX to stay low (selling vol). "
            "Clustered buying at high strikes = institutions hedging tail risk.")
    return text, None


# ========================================
# CHART CREATORS
# ========================================

def create_gamma_chart(options_df, symbol, spot, lookback_df=None, atm_straddle=None):
    """
    Unusual Whales-style horizontal GEX profile chart.
    - Horizontal bars per strike (Y=strike, X=net dealer GEX)
    - Green = dealers net long gamma (mean-reversion); Red = net short (momentum amplifier)
    - Orange = GEX flipped sign vs previous slice
    - Purple = GEX changed magnitude by >20% vs previous slice (without flipping)
    - Dots = previous-slice GEX values for each strike
    - Spot line (red dashed) + straddle breakeven lines (white dashed)
    """
    gex = calculate_gex_by_strike(options_df, symbol)
    if gex is None or gex.empty:
        return empty_chart("No GEX data - waiting for snapshot", 500)
    if spot and spot > 0:
        lo, hi = spot * 0.93, spot * 1.07
        gex = gex[(gex['strike'] >= lo) & (gex['strike'] <= hi)]
        if gex.empty:
            return empty_chart("No GEX data in ±7% range", 500)
    gex = gex.sort_values('strike', ascending=True).copy()  # ascending data; autorange='reversed' flips display

    # Drop thin/near-zero gamma bars so the chart focuses on actionable levels.
    # Dynamic floor = max(30th percentile, 3% of max abs gamma).
    gex['abs_gamma'] = pd.to_numeric(gex['gamma_exp'], errors='coerce').abs()
    abs_max = float(gex['abs_gamma'].max() or 0.0)
    abs_q30 = float(gex['abs_gamma'].quantile(0.30) or 0.0)
    min_abs_gamma = max(abs_q30, abs_max * 0.03)
    if min_abs_gamma > 0:
        gex = gex[gex['abs_gamma'] >= min_abs_gamma].copy()
    if gex.empty:
        return empty_chart("No significant GEX levels after thin-bar filter", 500)
    gex = gex.drop(columns=['abs_gamma'], errors='ignore')

    # --- Previous slice ---
    gex_prev = None
    if lookback_df is not None:
        gex_prev_raw = calculate_gex_by_strike(lookback_df, symbol)
        if gex_prev_raw is not None and not gex_prev_raw.empty:
            gex_prev = gex_prev_raw.set_index('strike')['gamma_exp'].to_dict()

    # --- Assign colors per bar ---
    COL_POS    = C['call']            # green  — positive GEX
    COL_NEG    = C['put']             # red    — negative GEX
    COL_FLIP   = C['warning']         # orange — sign flipped vs prev
    COL_SURGE  = '#8b5cf6'            # purple — large magnitude change
    SURGE_THRESH = 0.20               # 20% change triggers purple

    bar_colors = []
    flipped_bars = []   # (strike, cur, prev)
    surged_bars  = []   # (strike, cur, prev)
    for _, row in gex.iterrows():
        cur = row['gamma_exp']
        prev = gex_prev.get(row['strike']) if gex_prev else None
        if prev is not None and prev != 0:
            if (cur > 0) != (prev > 0):
                bar_colors.append(COL_FLIP)
                flipped_bars.append((row['strike'], cur, prev))
            elif abs(cur - prev) / abs(prev) >= SURGE_THRESH:
                bar_colors.append(COL_SURGE)
                surged_bars.append((row['strike'], cur, prev))
            else:
                bar_colors.append(COL_POS if cur >= 0 else COL_NEG)
        else:
            bar_colors.append(COL_POS if cur >= 0 else COL_NEG)

    fig = go.Figure()

    # --- Current GEX bars (horizontal) ---
    fig.add_trace(go.Bar(
        y=gex['strike'],
        x=gex['gamma_exp'],
        orientation='h',
        marker=dict(color=bar_colors, opacity=0.90),
        name='Net Dealer GEX',
        hovertemplate='Strike: %{y}<br>Net GEX: %{x:,.0f}<extra></extra>',
        width=[abs(gex['strike'].diff().median()) * 0.7 if len(gex) > 1 else 5.0] * len(gex),
    ))

    # --- Previous-slice dots ---
    if gex_prev:
        dot_strikes = [s for s in gex['strike'] if s in gex_prev]
        dot_vals    = [gex_prev[s] for s in dot_strikes]
        if dot_strikes:
            fig.add_trace(go.Scatter(
                y=dot_strikes,
                x=dot_vals,
                mode='markers',
                marker=dict(color='#ffffff', size=9, symbol='circle',
                            line=dict(color='#94a3b8', width=1.5)),
                name='Prev slice',
                hovertemplate='Strike: %{y}<br>Prev GEX: %{x:,.0f}<extra></extra>',
            ))

    # --- Zero line ---
    fig.add_vline(x=0, line_color=C['text_muted'], line_width=1)

    # --- Spot price line (teal dotted, like Unusual Whales) ---
    if spot and spot > 0:
        fig.add_hline(
            y=spot,
            line_dash='dot', line_color='#06b6d4', line_width=2,
            annotation_text=f'{spot:,.2f}',
            annotation_font=dict(color='#06b6d4', size=11),
            annotation_position='right',
        )

    # --- Straddle breakeven lines ---
    if atm_straddle and np.isfinite(atm_straddle) and atm_straddle > 0 and spot and spot > 0:
        upper_be = spot + atm_straddle
        lower_be = spot - atm_straddle
        for be_level, label in [(upper_be, f'+BE {upper_be:,.0f}'), (lower_be, f'-BE {lower_be:,.0f}')]:
            fig.add_hline(
                y=be_level,
                line_dash='dot', line_color='#ffffff', line_width=1.2,
                annotation_text=f'  {label}',
                annotation_font=dict(color='#94a3b8', size=10),
                annotation_position='right',
            )

    # --- Legend annotation (color key, bottom-left, compact) ---
    legend_lines = [
        '<span style="color:#10b981">■</span> Green: Dealers LONG γ — suppress moves',
        '<span style="color:#ef4444">■</span> Red: Dealers SHORT γ — amplify moves',
        '<span style="color:#f59e0b">■</span> Orange: GEX sign FLIPPED vs prev snapshot',
        '<span style="color:#8b5cf6">■</span> Purple: GEX magnitude surged ≥20% vs prev',
        '<span style="color:#ffffff">●</span> Dot: Previous snapshot level',
    ]
    fig.add_annotation(
        xref='paper', yref='paper', x=0.01, y=0.01,
        text='<br>'.join(legend_lines),
        showarrow=False,
        align='left',
        font=dict(size=10, color=MC['text']),
        bgcolor='rgba(15,23,42,0.88)',
        borderpad=5,
        bordercolor=MC['border'],
        borderwidth=1,
        xanchor='left', yanchor='bottom',
    )

    # --- Flipped & surged bar annotations — anchored left of plot to avoid blocking bars ---
    for strike, cur, prev in sorted(flipped_bars, key=lambda x: abs(x[1]), reverse=True)[:3]:
        direction = "SHORT→LONG" if cur > 0 else "LONG→SHORT"
        label = f"<b>Flip @ {strike:,.0f}</b>  {direction}"
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.01, y=strike,
            text=label,
            showarrow=True, arrowhead=2, arrowwidth=1.5,
            ax=-30, ay=0,
            arrowcolor=COL_FLIP, font=dict(size=9, color=COL_FLIP),
            bgcolor="rgba(15,23,42,0.88)", bordercolor=COL_FLIP,
            borderwidth=1, borderpad=3, align="left",
            xanchor="left",
        )

    for strike, cur, prev in sorted(surged_bars, key=lambda x: abs(x[1]), reverse=True)[:3]:
        pct = abs((cur - prev) / abs(prev) * 100) if prev else 0
        label = (f"<b>Surge @ {strike:,.0f}</b><br>"
                 f"+{pct:.0f}% vs prev snapshot<br>"
                 f"Large new {'long' if cur > 0 else 'short'} gamma positioning")
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.01, y=strike,
            text=label,
            showarrow=True, arrowhead=2, arrowwidth=1.5,
            ax=-30, ay=0,
            arrowcolor=COL_SURGE, font=dict(size=9, color=COL_SURGE),
            bgcolor="rgba(15,23,42,0.88)", bordercolor=COL_SURGE,
            borderwidth=1, borderpad=3, align="left",
            xanchor="left",
        )

    # --- "You are here" regime box — right side, arrow points at spot ---
    if spot and spot > 0:
        net_gex_total = gex['gamma_exp'].sum()
        if net_gex_total >= 0:
            regime_text  = "<b>+GEX</b> suppress moves"
            regime_color = C['call']
        else:
            regime_text  = "<b>-GEX</b> amplify moves"
            regime_color = C['put']
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.01, y=spot,
            ax=-30, ay=0,
            text=regime_text,
            showarrow=True, arrowhead=2, arrowwidth=2,
            arrowcolor=regime_color,
            font=dict(size=10, color=regime_color),
            bgcolor="rgba(15,23,42,0.88)", bordercolor=regime_color,
            borderwidth=1, borderpad=4, align="left",
            xanchor="left",
        )

    # --- Gamma Flip label (right margin) ---
    try:
        flip_info = calculate_gamma_flip(options_df, symbol)
    except Exception:
        flip_info = None
    if flip_info and flip_info.get('flip_strike'):
        fs = flip_info['flip_strike']
        fig.add_annotation(
            xref="paper", yref="y", x=1.01, y=fs,
            ax=-30, ay=0,
            text=f"<b>Flip @ {fs:,.0f}</b>",
            showarrow=True, arrowhead=2, arrowwidth=1.5,
            arrowcolor=C['warning'],
            font=dict(size=9, color=C['warning']),
            bgcolor="rgba(15,23,42,0.82)", bordercolor=C['warning'],
            borderwidth=1, borderpad=3, xanchor="left",
        )
        fig.add_hline(y=fs, line_dash="dot", line_color=C['warning'], line_width=1, opacity=0.6)

    n_strikes = len(gex)
    chart_h = max(500, min(n_strikes * 22 + 80, 900))

    fig.update_layout(**base_layout(
        height=chart_h,
        xaxis_title='Net Dealer Gamma Exposure',
        yaxis_title='Strike',
        barmode='overlay',
        hovermode='y unified',
    ))
    fig.update_layout(
        legend=dict(
            orientation='h',
            x=0.01, y=0.99,
            xanchor='left', yanchor='top',
            font=dict(size=10, color=MC['text']),
            bgcolor='rgba(15,23,42,0.35)',
            borderwidth=0,
        ),
        margin=dict(l=60, r=200, t=56, b=40),
    )
    fig.update_yaxes(
        tickformat=',d', automargin=True,
        autorange='reversed',
    )
    fig.update_xaxes(tickformat='~s', zeroline=True, zerolinecolor=C['text_muted'], zerolinewidth=2)
    style_axes(fig)
    return fig

def create_strike_chart(options_df, symbol, spot, lookback_df=None):
    """Key Strike Levels chart with optional ghost overlay from a past snapshot (v3.8)."""
    vol = calculate_volume_profile(options_df, symbol)
    if vol is None or vol.empty:
        return empty_chart("No strike data")
    if spot and spot > 0:
        lo, hi = spot * 0.95, spot * 1.05
        vol = vol[(vol['strike'] >= lo) & (vol['strike'] <= hi)]
        if vol.empty:
            return empty_chart("No strike data in ±5% range")
    vol_threshold = vol['volume'].quantile(0.15)
    vol = vol[vol['volume'] >= max(vol_threshold, 1)]
    calls = vol[vol['cp_sign'] == 1].sort_values('strike')
    puts = vol[vol['cp_sign'] == -1].sort_values('strike')
    fig = go.Figure()
    # Ghost overlay — lookback snapshot (lighter, draw first so current is on top)
    if lookback_df is not None:
        vol_lb = calculate_volume_profile(lookback_df, symbol)
        if vol_lb is not None and not vol_lb.empty:
            if spot and spot > 0:
                vol_lb = vol_lb[(vol_lb['strike'] >= lo) & (vol_lb['strike'] <= hi)]
            calls_lb = vol_lb[vol_lb['cp_sign'] == 1].sort_values('strike')
            puts_lb = vol_lb[vol_lb['cp_sign'] == -1].sort_values('strike')
            if not calls_lb.empty:
                fig.add_trace(go.Bar(y=calls_lb['strike'], x=calls_lb['volume'], orientation='h',
                    marker=dict(color='rgba(16,185,129,0.20)', line=dict(color=C['call'], width=1)),
                    name='Call Vol (prev)',
                    hovertemplate='Strike: %{y}<br>Prev Call Vol: %{x:,.0f}<extra></extra>'))
            if not puts_lb.empty:
                fig.add_trace(go.Bar(y=puts_lb['strike'], x=-puts_lb['volume'], orientation='h',
                    marker=dict(color='rgba(239,68,68,0.20)', line=dict(color=C['put'], width=1)),
                    name='Put Vol (prev)',
                    hovertemplate='Strike: %{y}<br>Prev Put Vol: %{customdata:,.0f}<extra></extra>',
                    customdata=puts_lb['volume']))
    # Current bars (solid, on top)
    if not calls.empty:
        fig.add_trace(go.Bar(y=calls['strike'], x=calls['volume'], orientation='h',
            marker=dict(color=C['call']), name='Call Vol',
            hovertemplate='Strike: %{y}<br>Call Vol: %{x:,.0f}<extra></extra>'))
    if not puts.empty:
        fig.add_trace(go.Bar(y=puts['strike'], x=-puts['volume'], orientation='h',
            marker=dict(color=C['put']), name='Put Vol',
            hovertemplate='Strike: %{y}<br>Put Vol: %{customdata:,.0f}<extra></extra>',
            customdata=puts['volume']))
    if spot and spot > 0:
        fig.add_hline(y=spot, line_dash="dash", line_color=C['warning'], line_width=2,
            annotation_text=f"Spot {spot:.0f}")

    # Call wall: heaviest call strike
    if not calls.empty:
        call_wall_row = calls.loc[calls['volume'].idxmax()]
        x_max = calls['volume'].max() or 1.0
        fig.add_annotation(
            x=x_max * 0.92, y=call_wall_row['strike'],
            ax=0, ay=-36,
            text=f"<b>Call Wall @ {call_wall_row['strike']:,.0f}</b><br>Resistance — dealers short calls here",
            showarrow=True, arrowhead=2, arrowwidth=1.5,
            arrowcolor=C['call'], font=dict(size=10, color=C['call']),
            bgcolor="rgba(15,23,42,0.82)", bordercolor=C['call'],
            borderwidth=1, borderpad=4, align="left", xanchor="right",
        )

    # Put wall: heaviest put strike (x is negative in chart)
    if not puts.empty:
        put_wall_row = puts.loc[puts['volume'].idxmax()]
        x_min = -puts['volume'].max() or -1.0
        fig.add_annotation(
            x=x_min * 0.92, y=put_wall_row['strike'],
            ax=0, ay=36,
            text=f"<b>Put Wall @ {put_wall_row['strike']:,.0f}</b><br>Support floor — dealers long puts here",
            showarrow=True, arrowhead=2, arrowwidth=1.5,
            arrowcolor=C['put'], font=dict(size=10, color=C['put']),
            bgcolor="rgba(15,23,42,0.82)", bordercolor=C['put'],
            borderwidth=1, borderpad=4, align="left", xanchor="left",
        )

    # Adaptive height: more strike rows need more vertical room for bars + labels.
    n_levels = int(pd.to_numeric(vol.get('strike', pd.Series(dtype=float)), errors='coerce').dropna().nunique())
    chart_h = max(420, min(640, 300 + n_levels * 18))
    fig.update_layout(**base_layout(height=chart_h, barmode='overlay',
        xaxis_title="Volume", yaxis_title="Strike", hovermode='y unified'))
    fig.update_layout(margin=dict(l=70, r=70, t=44, b=44))
    fig.update_yaxes(automargin=True)
    style_axes(fig)
    return fig

def create_iv_chart(options_df, symbol):
    iv = calculate_iv_term_structure(options_df, symbol)
    if iv is None or iv.empty:
        return empty_chart("No IV data")
    calls = iv[iv['cp_sign'] == 1].sort_values('dte')
    puts = iv[iv['cp_sign'] == -1].sort_values('dte')
    fig = go.Figure()
    if not calls.empty:
        fig.add_trace(go.Scatter(x=calls['dte'], y=calls['implied_vol']*100,
            mode='lines+markers', line=dict(color=C['call'], width=3),
            marker=dict(size=8), name='Call IV'))
    if not puts.empty:
        fig.add_trace(go.Scatter(x=puts['dte'], y=puts['implied_vol']*100,
            mode='lines+markers', line=dict(color=C['put'], width=3),
            marker=dict(size=8), name='Put IV'))
    unique_dte = int(pd.to_numeric(iv.get('dte', pd.Series(dtype=float)), errors='coerce').dropna().nunique())
    if unique_dte <= 1:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.98,
            text="Only one DTE bucket in current filter; switch DTE to All or 0-2 DTE for full curve.",
            showarrow=False, align="left",
            font=dict(size=11, color=C['warning']),
            bgcolor="rgba(15,23,42,0.65)", bordercolor=C['border'], borderwidth=1,
        )
    if not calls.empty and len(calls) >= 2:
        short_iv = calls.iloc[0]['implied_vol']
        long_iv  = calls.iloc[-1]['implied_vol']
        if short_iv > long_iv * 1.05:
            struct_text  = "<b>BACKWARDATION</b><br>Near-term fear premium — event risk"
            struct_color = C['put']
        else:
            struct_text  = "<b>Contango (normal)</b><br>No imminent event premium"
            struct_color = C['call']
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99,
            text=struct_text, showarrow=False, align="left",
            font=dict(size=10, color=struct_color),
            bgcolor="rgba(15,23,42,0.82)", bordercolor=struct_color,
            borderwidth=1, borderpad=4, xanchor="left", yanchor="top",
        )
    fig.update_layout(**base_layout(height=320,
        xaxis_title="Days to Expiration", yaxis_title="Implied Volatility (%)"))
    style_axes(fig)
    return fig

def create_flow_chart(agg_df, symbol):
    if agg_df is None or agg_df.empty:
        return empty_chart("No flow data")
    sym = agg_df[agg_df['symbol'] == symbol].sort_values('_ts_parsed').tail(MAX_HISTORY)
    if sym.empty:
        return empty_chart("No flow data for " + symbol)
    has_premium = 'net_premium' in sym.columns and sym['net_premium'].notna().any()
    cv_col = next((c for c in ['call_vol','callvol'] if c in sym.columns), None)
    pv_col = next((c for c in ['put_vol','putvol'] if c in sym.columns), None)
    n_rows = 2 if has_premium else 1
    row_heights = [0.6, 0.4] if has_premium else [1.0]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.08, row_heights=row_heights)
    x_time = parse_agg_timestamps(sym)
    if cv_col:
        fig.add_trace(go.Scatter(x=x_time, y=sym[cv_col], mode='lines', fill='tozeroy',
            line=dict(color=C['call'], width=2), fillcolor='rgba(16,185,129,0.15)',
            name='Call Vol'), row=1, col=1)
    if pv_col:
        fig.add_trace(go.Scatter(x=x_time, y=sym[pv_col], mode='lines', fill='tozeroy',
            line=dict(color=C['put'], width=2), fillcolor='rgba(239,68,68,0.15)',
            name='Put Vol'), row=1, col=1)
    if has_premium:
        net_p = sym['net_premium'] / 1e6
        np_colors = [C['call'] if v >= 0 else C['put'] for v in net_p]
        fig.add_trace(go.Bar(x=x_time, y=net_p,
            marker=dict(color=np_colors, opacity=0.8),
            name='Net Premium ($M)'), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color=C['text_muted'], opacity=0.4, row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=1, gridcolor=C['grid'], showgrid=True)
        fig.update_yaxes(title_text="Net Premium ($M)", row=2, col=1, gridcolor=C['grid'], showgrid=True)
    if not cv_col and not pv_col and not has_premium and 'pc_ratio' in sym.columns:
        fig.add_trace(go.Scatter(x=x_time, y=sym['pc_ratio'], mode='lines+markers',
            line=dict(color=C['accent'], width=2), name='P/C Ratio'), row=1, col=1)

    # Spike annotations: flag when put or call vol exceeds 2× rolling mean
    spike_anns = []
    for col, label, color in [(cv_col, "Call surge", C['call']), (pv_col, "Put surge", C['put'])]:
        if col and col in sym.columns:
            series = sym[col].fillna(0)
            avg = series.rolling(min(20, max(3, len(series)//4)), min_periods=1).mean()
            for i, (ts_val, val, av) in enumerate(zip(x_time, series, avg)):
                if av > 0 and val > av * 2.0:
                    spike_anns.append(dict(
                        x=ts_val, y=val,
                        ax=0, ay=-40,
                        text=f"<b>{label}</b><br>heavy hedging",
                        showarrow=True, arrowhead=2, arrowwidth=1.5,
                        arrowcolor=color, font=dict(size=9, color=color),
                        bgcolor="rgba(15,23,42,0.82)", bordercolor=color,
                        borderwidth=1, borderpad=3, align="center",
                        xref="x", yref="y",
                    ))
                    break  # one annotation per series max
    if spike_anns:
        for ann in spike_anns:
            fig.add_annotation(**ann)

    fig.update_xaxes(gridcolor=C['grid'], showgrid=True, type='date', tickformat='%H:%M')
    fig.update_layout(**base_layout(height=320, xaxis_title="Time"))
    style_axes(fig)
    return fig


def create_mm_flow_chart(agg_df, symbols=None):
    if agg_df is None or agg_df.empty:
        return empty_chart("No MM flow data")
    if symbols is None:
        symbols = agg_df['symbol'].unique()[:3]
    fig = go.Figure()
    colors = [C['neutral'], C['call'], C['put'], C['warning']]
    for i, sym in enumerate(symbols):
        sym_df = agg_df[agg_df['symbol'] == sym].sort_values('_ts_parsed').tail(MAX_HISTORY)
        if sym_df.empty or 'net_gex' not in sym_df.columns:
            continue
        gex_diff = sym_df['net_gex'].diff().fillna(0)
        mm_x = parse_agg_timestamps(sym_df)
        fig.add_trace(go.Scatter(
            x=mm_x, y=gex_diff.values,
            mode='lines+markers', line=dict(color=colors[i%len(colors)], width=3),
            marker=dict(size=8), name=f'{sym} Net Flow'))
    fig.add_hline(y=0, line_color=C['text_muted'], line_width=1, opacity=0.5)
    fig.update_layout(**base_layout(height=320, xaxis_title="Time", xaxis_type="date", xaxis_tickformat="%H:%M",
                                     yaxis_title="Net GEX Change"))
    style_axes(fig)
    return fig

def create_multi_gamma_chart(agg_df):
    if agg_df is None or agg_df.empty:
        return empty_chart("No multi-symbol data")
    latest = agg_df.groupby('symbol').tail(1)
    if 'net_gex' not in latest.columns:
        return empty_chart("No GEX in aggregate")
    symbols = latest['symbol'].values
    net_gex = latest['net_gex'].values
    colors = [C['call'] if g > 0 else C['put'] for g in net_gex]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=symbols, y=net_gex, marker=dict(color=colors),
        text=[f"{'+' if v>0 else ''}{v:,.0f}" for v in net_gex], textposition='outside'))
    fig.add_hline(y=0, line_color=C['text_muted'], line_width=1)
    fig.update_layout(**base_layout(height=320, showlegend=False,
        xaxis_title="Symbol", yaxis_title="Net Gamma Exposure"))
    style_axes(fig)
    return fig

def create_multi_sentiment_chart(agg_df):
    if agg_df is None or agg_df.empty:
        return empty_chart("No sentiment data")
    latest = agg_df.groupby('symbol').tail(1)
    symbols = latest['symbol'].values
    fig = go.Figure()
    cv_col = next((c for c in ['call_vol','callvol'] if c in latest.columns), None)
    pv_col = next((c for c in ['put_vol','putvol'] if c in latest.columns), None)
    if cv_col and pv_col:
        fig.add_trace(go.Bar(x=symbols, y=latest[cv_col].values,
            marker=dict(color=C['call']), name='Call Volume'))
        fig.add_trace(go.Bar(x=symbols, y=latest[pv_col].values,
            marker=dict(color=C['put']), name='Put Volume'))
    elif 'pc_ratio' in latest.columns:
        pc = latest['pc_ratio'].values
        colors = [C['call'] if p < 1 else C['put'] for p in pc]
        fig.add_trace(go.Bar(x=symbols, y=pc, marker=dict(color=colors), name='P/C Ratio'))
        fig.add_hline(y=1.0, line_dash="dash", line_color=C['warning'])
    fig.update_layout(**base_layout(height=320, barmode='group',
        xaxis_title="Symbol", yaxis_title="Volume / Ratio"))
    style_axes(fig)
    return fig

def create_vix_put_flow_chart(agg_df):
    if agg_df is None or agg_df.empty:
        return empty_chart("No VIX flow data")
    vix = agg_df[agg_df['symbol'].isin(['VIXW', 'VIX'])].sort_values('_ts_parsed').copy()
    if vix.empty:
        return empty_chart("No VIX data in agg")
    # Prefer VIXW over VIX
    if 'VIXW' in vix['symbol'].values:
        vix = vix[vix['symbol'] == 'VIXW']
    vix = vix.tail(MAX_HISTORY)
    fig = go.Figure()
    pv_col = next((c for c in ['put_vol','putvol'] if c in vix.columns), None)
    cv_col = next((c for c in ['call_vol','callvol'] if c in vix.columns), None)
    vix_x = parse_agg_timestamps(vix)
    if pv_col:
        # Compute incremental volume (diff between batches) for meaningful changes
        raw_vol = pd.to_numeric(vix[pv_col], errors='coerce').fillna(0)
        incr_vol = raw_vol.diff().clip(lower=0)
        incr_vol.iloc[0] = 0  # first batch has no prior
        has_incremental = incr_vol.sum() > 0
        if has_incremental:
            fig.add_trace(go.Bar(x=vix_x, y=incr_vol.values,
                marker=dict(color=C['warning'], opacity=0.7), name='Put Vol Δ'))
            fig.add_trace(go.Scatter(x=vix_x, y=raw_vol.values,
                mode='lines', line=dict(color=C['put'], width=2, dash='dot'),
                name='Cumul Put Vol', yaxis='y2'))
        else:
            fig.add_trace(go.Scatter(x=vix_x, y=raw_vol.values,
                mode='lines', fill='tozeroy', line=dict(color=C['warning'], width=2),
                fillcolor='rgba(245,158,11,0.3)', name='VIX Put Volume'))
    if cv_col and pv_col:
        cv_raw = pd.to_numeric(vix[cv_col], errors='coerce').fillna(0)
        pv_raw = pd.to_numeric(vix[pv_col], errors='coerce').fillna(0)
        pc = (pv_raw / cv_raw.replace(0, np.nan)).fillna(0)
        fig.add_trace(go.Scatter(x=vix_x, y=pc.values,
            mode='lines+markers', line=dict(color=C['accent'], width=2),
            marker=dict(size=4), name='P/C Ratio', yaxis='y2'))
    layout_kw = base_layout(height=320, xaxis_title="Time", xaxis_type="date",
                            xaxis_tickformat="%H:%M", yaxis_title="VIX Put Flow")
    layout_kw['yaxis2'] = dict(overlaying='y', side='right', showgrid=False,
                               title='P/C Ratio / Cumul Vol', color=C['text_muted'])
    fig.update_layout(**layout_kw)
    style_axes(fig)
    return fig

def create_vix_hedging_chart(options_df):
    if options_df is None or options_df.empty:
        return empty_chart("No VIX hedging data")
    vix = options_df[options_df['symbol'].isin(['VIXW', 'VIX'])]
    # Prefer VIXW
    if 'VIXW' in vix['symbol'].values:
        vix = vix[vix['symbol'] == 'VIXW']
    if vix.empty:
        return empty_chart("No VIX options in snapshot")
    puts = vix[vix['cp_sign'] == -1]
    if puts.empty:
        return empty_chart("No VIX puts")
    agg_cols = {'volume': ('volume', 'sum')}
    if 'oi' in puts.columns:
        agg_cols['oi'] = ('oi', 'sum')
    by_strike = puts.groupby('strike').agg(**agg_cols).reset_index().sort_values('strike')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=by_strike['strike'], y=by_strike['volume'],
        marker=dict(color=C['warning']), name='Put Volume'), secondary_y=False)
    if 'oi' in by_strike.columns and by_strike['oi'].sum() > 0:
        ratio = (by_strike['volume'] / by_strike['oi'].replace(0, np.nan) * 100).fillna(0)
        fig.add_trace(go.Scatter(x=by_strike['strike'], y=ratio,
            mode='lines+markers', line=dict(color=C['put'], width=3),
            marker=dict(size=8), name='Vol/OI %'), secondary_y=True)
    fig.update_layout(**base_layout(height=320))
    fig.update_xaxes(title_text="VIX Strike", gridcolor=C['grid'])
    fig.update_yaxes(title_text="Put Volume", gridcolor=C['grid'], secondary_y=False)
    fig.update_yaxes(title_text="Vol/OI %", showgrid=False, secondary_y=True)
    return fig

def create_vanna_chart(options_df, symbol, spot=0):
    vanna = calculate_vanna_by_strike(options_df, symbol)
    if vanna is None or vanna.empty:
        return empty_chart("No vanna data")
    if spot and spot > 0:
        lo, hi = spot * 0.90, spot * 1.10
        vanna = vanna[(vanna['strike'] >= lo) & (vanna['strike'] <= hi)]
        if vanna.empty:
            return empty_chart("No vanna data in ±10% range")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vanna['strike'], y=vanna['vanna'],
        mode='lines', fill='tozeroy', line=dict(color=C['neutral'], width=3),
        fillcolor='rgba(59,130,246,0.3)', name='Vanna Exposure'))

    annotations = []
    y_range = vanna['vanna'].abs().max() or 1.0

    # Peak positive vanna
    pos_mask = vanna['vanna'] > 0
    if pos_mask.any():
        pos_sub = vanna[pos_mask]
        peak_row = pos_sub.loc[pos_sub['vanna'].idxmax()]
        annotations.append(dict(
            x=peak_row['strike'], y=peak_row['vanna'],
            ax=60, ay=-50,
            text="<b>Peak +Vanna</b><br>IV drop → dealers BUY<br>to re-hedge → price support",
            showarrow=True, arrowhead=2, arrowwidth=1.5,
            arrowcolor=C['call'], font=dict(size=10, color=C['call']),
            bgcolor=MC['bg_card'], bordercolor=C['call'], borderwidth=1,
            borderpad=4, align="left",
        ))

    # Trough negative vanna
    neg_mask = vanna['vanna'] < 0
    if neg_mask.any():
        neg_sub = vanna[neg_mask]
        trough_row = neg_sub.loc[neg_sub['vanna'].idxmin()]
        annotations.append(dict(
            x=trough_row['strike'], y=trough_row['vanna'],
            ax=60, ay=50,
            text="<b>Trough −Vanna</b><br>IV drop → dealers SELL<br>to re-hedge → price pressure",
            showarrow=True, arrowhead=2, arrowwidth=1.5,
            arrowcolor=C['put'], font=dict(size=10, color=C['put']),
            bgcolor=MC['bg_card'], bordercolor=C['put'], borderwidth=1,
            borderpad=4, align="left",
        ))

    # Spot price line + label
    if spot and spot > 0:
        fig.add_vline(x=spot, line_dash="dash", line_color=MC['text_muted'], line_width=1)
        annotations.append(dict(
            x=spot, y=y_range * 0.85,
            ax=0, ay=0,
            text=f"<b>Spot {spot:,.0f}</b>",
            showarrow=False,
            font=dict(size=10, color=MC['text_muted']),
            bgcolor=MC['bg_card'], bordercolor=MC['text_muted'], borderwidth=1,
            borderpad=3,
        ))

    # Net bias summary box (top-left)
    net_vanna = vanna['vanna'].sum()
    if net_vanna > 0:
        bias_text = "<b>Net +Vanna</b><br>IV drop = dealer BUY tailwind"
        bias_color = C['call']
    else:
        bias_text = "<b>Net \u2212Vanna</b><br>IV drop = dealer SELL pressure"
        bias_color = C['put']
    annotations.append(dict(
        xref="paper", yref="paper", x=0.01, y=0.99,
        text=bias_text, showarrow=False, align="left",
        font=dict(size=10, color=bias_color),
        bgcolor="rgba(15,23,42,0.80)", bordercolor=bias_color,
        borderwidth=1, borderpad=4, xanchor="left", yanchor="top",
    ))

    # Zero-cross label
    zero_crosses = []
    for i in range(len(vanna) - 1):
        v0, v1 = vanna['vanna'].iloc[i], vanna['vanna'].iloc[i + 1]
        if v0 * v1 < 0:
            s0, s1 = vanna['strike'].iloc[i], vanna['strike'].iloc[i + 1]
            zero_crosses.append((s0 + s1) / 2.0)
    for zx in zero_crosses:
        annotations.append(dict(
            x=zx, y=0,
            ax=0, ay=-40,
            text="<b>Zero cross</b><br>sell\u2192buy pressure transition",
            showarrow=True, arrowhead=2, arrowwidth=1.5,
            arrowcolor=MC['text_muted'], font=dict(size=9, color=MC['text_muted']),
            bgcolor="rgba(15,23,42,0.80)", bordercolor=MC['text_muted'],
            borderwidth=1, borderpad=3, align="center",
        ))

    fig.update_layout(**base_layout(height=400,
        xaxis_title="Strike Price", yaxis_title="Vanna"))
    if annotations:
        fig.update_layout(annotations=annotations)
    style_axes(fig)
    return fig

def create_dealer_chart(options_df, symbol):
    greeks = calculate_dealer_greeks(options_df, symbol)
    if greeks is None or len(greeks) == 0:
        return empty_chart("No dealer greek data")
    # Split into two panels: Delta/Gamma (directional) vs Vega/Theta (vol/time)
    # These have very different magnitudes so a single bar chart is misleading
    dg_names, dg_vals, vt_names, vt_vals = [], [], [], []
    for k, v in greeks.items():
        if k in ('Delta', 'Gamma'):
            dg_names.append(k)
            dg_vals.append(v)
        else:
            vt_names.append(k)
            vt_vals.append(v)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            'Directional<br>(Delta/Gamma)',
            'Vol & Time<br>(Vega/Theta)',
        ],
        horizontal_spacing=0.24
    )
    if dg_vals:
        dg_colors = [C['call'] if v > 0 else C['put'] for v in dg_vals]
        fig.add_trace(go.Bar(x=dg_names, y=dg_vals, marker=dict(color=dg_colors),
            text=[f"{'+' if v>0 else ''}{v:,.0f}" for v in dg_vals],
            textposition='outside', showlegend=False), row=1, col=1)
    if vt_vals:
        vt_colors = [C['call'] if v > 0 else C['put'] for v in vt_vals]
        fig.add_trace(go.Bar(x=vt_names, y=vt_vals, marker=dict(color=vt_colors),
            text=[f"{'+' if v>0 else ''}{v:,.0f}" for v in vt_vals],
            textposition='outside', showlegend=False), row=1, col=2)
    fig.add_hline(y=0, line_color=C['text_muted'], line_width=1, opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_color=C['text_muted'], line_width=1, opacity=0.5, row=1, col=2)

    # Regime label above the Gamma bar
    gamma_val = next((v for n, v in zip(dg_names, dg_vals) if n == 'Gamma'), None)
    if gamma_val is not None:
        if gamma_val >= 0:
            regime_text  = "LONG GAMMA — mean-reversion"
            regime_color = C['call']
        else:
            regime_text  = "SHORT GAMMA — momentum amplifier"
            regime_color = C['put']
        fig.add_annotation(
            xref="x domain", yref="paper", x=0.5, y=1.04,
            text=f"<b>{regime_text}</b>",
            showarrow=False, align="center",
            font=dict(size=10, color=regime_color),
            bgcolor="rgba(15,23,42,0.82)", bordercolor=regime_color,
            borderwidth=1, borderpad=3,
            xanchor="center", yanchor="bottom",
            row=1, col=1,
        )

    fig.update_layout(**base_layout(height=340, showlegend=False))
    fig.update_layout(margin=dict(l=55, r=30, t=72, b=42))
    fig.update_yaxes(title_text="Net Position", row=1, col=1, gridcolor=C['grid'])
    fig.update_yaxes(title_text="Net Position", row=1, col=2, gridcolor=C['grid'])
    fig.update_xaxes(automargin=True, tickfont=dict(size=11), row=1, col=1)
    fig.update_xaxes(automargin=True, tickfont=dict(size=11), row=1, col=2)
    # Style subplot title text
    for ann in fig.layout.annotations:
        ann.font.color = C['text']
        ann.font.size = 11
    style_axes(fig)
    return fig


# ========================================
# NEW FEATURE: OI Concentration & Walls
# ========================================
def calculate_oi_walls(options_df, symbol):
    """Compute put/call wall strikes and pinning probability from OI concentration."""
    if options_df is None or options_df.empty:
        return None
    sym = options_df[options_df['symbol'] == symbol]
    if sym.empty or 'oi' not in sym.columns:
        return None
    calls = sym[sym['cp_sign'] == 1].copy()
    puts = sym[sym['cp_sign'] == -1].copy()
    if calls.empty or puts.empty:
        return None
    # Top call and put walls by OI
    call_wall = calls.loc[calls['oi'].idxmax()]
    put_wall = puts.loc[puts['oi'].idxmax()]
    spot = sym['spot'].iloc[0] if 'spot' in sym.columns else 0
    result = {
        'call_wall_strike': call_wall['strike'],
        'call_wall_oi': call_wall['oi'],
        'put_wall_strike': put_wall['strike'],
        'put_wall_oi': put_wall['oi'],
        'spot': spot,
    }
    if spot > 0:
        result['dist_to_call_wall'] = (call_wall['strike'] - spot) / spot * 100
        result['dist_to_put_wall'] = (spot - put_wall['strike']) / spot * 100
        result['wall_range_pct'] = (call_wall['strike'] - put_wall['strike']) / spot * 100
        # Pinning probability: how close is spot to max OI strike?
        all_oi = sym.groupby('strike')['oi'].sum().reset_index()
        max_oi_strike = all_oi.loc[all_oi['oi'].idxmax(), 'strike']
        result['max_oi_strike'] = max_oi_strike
        result['dist_to_max_oi'] = abs(spot - max_oi_strike) / spot * 100
        result['pinning_score'] = max(0, 1 - result['dist_to_max_oi'] / 1.0)  # 1.0 = within 1% is high
    return result

def create_oi_walls_chart(options_df, symbol, spot=0):
    """Horizontal bar chart: top N OI strikes with call/put split + wall markers."""
    if options_df is None or options_df.empty:
        return empty_chart("No OI data")
    sym = options_df[options_df['symbol'] == symbol]
    if sym.empty or 'oi' not in sym.columns:
        return empty_chart("No OI column in data")
    if spot and spot > 0:
        lo, hi = spot * 0.95, spot * 1.05
        sym = sym[(sym['strike'] >= lo) & (sym['strike'] <= hi)]
        if sym.empty:
            return empty_chart("No OI data in ±5% range")
    # Aggregate OI by strike and side
    oi_data = sym.groupby(['strike', 'cp_sign'])['oi'].sum().reset_index()
    calls = oi_data[oi_data['cp_sign'] == 1].sort_values('strike')
    puts = oi_data[oi_data['cp_sign'] == -1].sort_values('strike')
    # Filter to meaningful OI
    all_oi = sym.groupby('strike')['oi'].sum()
    oi_thresh = all_oi.quantile(0.25)
    significant = all_oi[all_oi >= max(oi_thresh, 1)].index
    calls = calls[calls['strike'].isin(significant)]
    puts = puts[puts['strike'].isin(significant)]
    fig = go.Figure()
    if not calls.empty:
        fig.add_trace(go.Bar(y=calls['strike'], x=calls['oi'], orientation='h',
            marker=dict(color=C['call']), name='Call OI', opacity=0.8,
            hovertemplate='Strike: %{y}<br>Call OI: %{x:,.0f}<extra></extra>'))
    if not puts.empty:
        fig.add_trace(go.Bar(y=puts['strike'], x=-puts['oi'], orientation='h',
            marker=dict(color=C['put']), name='Put OI', opacity=0.8,
            hovertemplate='Strike: %{y}<br>Put OI: %{customdata:,.0f}<extra></extra>',
            customdata=puts['oi']))
    if spot and spot > 0:
        fig.add_hline(y=spot, line_dash="dash", line_color=C['warning'], line_width=2,
            annotation_text=f"Spot {spot:.0f}")
    # Mark the max OI strike
    total_oi = sym.groupby('strike')['oi'].sum()
    if not total_oi.empty:
        max_strike = total_oi.idxmax()
        fig.add_hline(y=max_strike, line_dash="dot", line_color=C['accent'], line_width=2,
            annotation_text=f"Max OI {max_strike:.0f}")
    fig.update_layout(**base_layout(height=400, barmode='overlay',
        xaxis_title="Open Interest", yaxis_title="Strike", hovermode='y unified'))
    style_axes(fig)
    return fig

def oi_walls_insight(options_df, symbol):
    """Generate insight text for OI walls chart."""
    walls = calculate_oi_walls(options_df, symbol)
    if walls is None:
        return None, None
    text = (f"<b>OI Walls & Pinning</b> "
            f"Call wall at <b>{walls['call_wall_strike']:,.0f}</b> "
            f"({walls['call_wall_oi']:,.0f} OI) — acts as resistance/magnet. "
            f"Put wall at <b>{walls['put_wall_strike']:,.0f}</b> "
            f"({walls['put_wall_oi']:,.0f} OI) — acts as support/floor.")
    anomaly = None
    if 'dist_to_max_oi' in walls:
        if walls['dist_to_max_oi'] < 0.3:
            text += (f"<br><b>Pinning alert:</b> Spot is within 0.3% of max OI strike "
                     f"({walls['max_oi_strike']:,.0f}). Strong gravitational pull — expect pin action.")
            anomaly = 'alert'
        elif walls.get('wall_range_pct', 99) < 2:
            text += "<br>Call and put walls are very tight — compressed range, breakout likely."
            anomaly = 'warn'
    return text, anomaly

# ========================================
# NEW FEATURE: DTE Concentration Heatmap
# ========================================
def create_dte_concentration_chart(options_df, symbol):
    """Bar chart showing OI and volume distribution across individual DTE days (0-5)."""
    if options_df is None or options_df.empty:
        return empty_chart("No DTE data")
    sym = options_df[options_df['symbol'] == symbol]
    if sym.empty or 'dte' not in sym.columns:
        return empty_chart("No DTE column")
    sym = sym.copy()
    sym['dte'] = pd.to_numeric(sym['dte'], errors='coerce')
    sym = sym[sym['dte'].between(0, 5)]
    if sym.empty:
        return empty_chart("No valid DTE data")
    sym['dte_label'] = sym['dte'].astype(int).map(lambda d: f"{d}DTE")
    dte_order = ['0DTE', '1DTE', '2DTE', '3DTE', '4DTE', '5DTE']
    has_oi = 'oi' in sym.columns and sym['oi'].sum() > 0
    has_vol = 'volume' in sym.columns and sym['volume'].sum() > 0
    if not has_oi and not has_vol:
        return empty_chart("No OI or volume data")
    fig = go.Figure()
    if has_oi:
        oi_by_dte = sym.groupby('dte_label')['oi'].sum().reindex(dte_order, fill_value=0)
        fig.add_trace(go.Bar(x=oi_by_dte.index, y=oi_by_dte.values,
            marker=dict(color=C['accent']), name='Open Interest', opacity=0.8,
            text=[f"{v:,.0f}" for v in oi_by_dte.values], textposition='outside'))
    if has_vol:
        vol_by_dte = sym.groupby('dte_label')['volume'].sum().reindex(dte_order, fill_value=0)
        fig.add_trace(go.Bar(x=vol_by_dte.index, y=vol_by_dte.values,
            marker=dict(color=C['warning']), name='Volume', opacity=0.6))
    fig.update_layout(**base_layout(height=320, barmode='group',
        xaxis_title="Days to Expiration", yaxis_title="Contracts"))
    style_axes(fig)
    return fig

def dte_concentration_insight(options_df, symbol):
    """Generate insight for DTE distribution (0-5 DTE range)."""
    if options_df is None or options_df.empty:
        return None, None
    sym = options_df[options_df['symbol'] == symbol]
    if sym.empty or 'dte' not in sym.columns:
        return None, None
    sym = sym.copy()
    sym['dte'] = pd.to_numeric(sym['dte'], errors='coerce')
    has_oi = 'oi' in sym.columns and sym['oi'].sum() > 0
    col = 'oi' if has_oi else 'volume'
    total = sym[col].sum()
    if total == 0:
        return None, None
    zero_dte = sym[sym['dte'] == 0][col].sum()
    one_dte = sym[sym['dte'] == 1][col].sum()
    rest_dte = sym[sym['dte'] >= 2][col].sum()
    zero_pct = zero_dte / total * 100
    one_pct = one_dte / total * 100
    rest_pct = rest_dte / total * 100
    text = (f"<b>Expiration Concentration</b> "
            f"0DTE: <b>{zero_pct:.1f}%</b> of {col.upper()}. "
            f"1DTE: <b>{one_pct:.1f}%</b>. "
            f"2-5DTE: <b>{rest_pct:.1f}%</b>. "
            "Heavy 0DTE = extreme gamma sensitivity, pin risk, and intraday mean-reversion. "
            "More 2-5DTE weight = multi-day positioning and lower decay urgency.")
    anomaly = None
    if zero_pct > 50:
        text += "<br><b>⚠ Extreme 0DTE concentration</b> — gamma-driven intraday dynamics dominate."
        anomaly = 'warn'
    return text, anomaly

# ========================================
# NEW FEATURE: Gamma Flip Zone Indicator
# ========================================
def calculate_gamma_flip(options_df, symbol):
    """Find the gamma flip level where dealer GEX changes sign."""
    gex = calculate_gex_by_strike(options_df, symbol)
    if gex is None or gex.empty:
        return None
    gex = gex.sort_values('strike')
    spot = 0
    sym = options_df[options_df['symbol'] == symbol]
    if not sym.empty and 'spot' in sym.columns:
        spot = sym['spot'].iloc[0]
    flip_strike = None
    flips = []
    for i in range(len(gex) - 1):
        if gex.iloc[i]['gamma_exp'] * gex.iloc[i + 1]['gamma_exp'] < 0:
            mid = (gex.iloc[i]['strike'] + gex.iloc[i + 1]['strike']) / 2.0
            flips.append(mid)
    if flips:
        flip_strike = min(flips, key=lambda x: abs(x - spot))
    net_gex = gex['gamma_exp'].sum()
    pos_gex = gex[gex['gamma_exp'] > 0]['gamma_exp'].sum()
    neg_gex = gex[gex['gamma_exp'] < 0]['gamma_exp'].sum()
    return {
        'flip_strike': flip_strike,
        'spot': spot,
        'net_gex': net_gex,
        'pos_gex': pos_gex,
        'neg_gex': neg_gex,
        'gamma_ratio': pos_gex / abs(neg_gex) if neg_gex != 0 else float('inf'),
    }

def _filter_by_window(agg_df, symbol, window_minutes=30):
    """Filter agg data by symbol and time window.

    If window_minutes == 'session', shows full trading session (8:30 AM - 5:00 PM ET).
    Otherwise, shows last N minutes rolling window (legacy behavior).
    """
    sym_df = agg_df[agg_df['symbol'] == symbol].sort_values('_ts_parsed').copy()
    if sym_df.empty:
        return sym_df

    if '_ts_parsed' in sym_df.columns and sym_df['_ts_parsed'].notna().any():
        if window_minutes == 'session':
            # Show full session: filter by session date, keeping 8:30 AM - 5:00 PM ET
            latest = sym_df['_ts_parsed'].max()

            # DEBUG: Print filtering info (comment out in production)
            if DEBUG_FILTER: print(f"[DEBUG] {symbol}: Latest timestamp = {latest}, window_minutes = {window_minutes}")

            # Handle timezone-aware timestamps properly
            if sym_df['_ts_parsed'].dt.tz is None:
                # Naive timestamps - assume already in ET
                ts_et = sym_df['_ts_parsed']
                session_date = latest.date()
            else:
                # Timezone-aware - convert to ET
                ts_et = sym_df['_ts_parsed'].dt.tz_convert(ET)
                session_date = latest.tz_convert(ET).date()

            # Filter by date and time window
            date_mask = ts_et.dt.date == session_date
            time_minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
            open_minutes = MARKET_OPEN_ET[0] * 60 + MARKET_OPEN_ET[1]  # 8:30 = 510
            close_minutes = MARKET_CLOSE_ET[0] * 60 + MARKET_CLOSE_ET[1]  # 17:00 = 1020
            time_mask = (time_minutes >= open_minutes) & (time_minutes <= close_minutes)

            filtered = sym_df[date_mask & time_mask].copy()

            # If there is only pre-open/post-close data for the latest date,
            # keep available rows instead of returning an empty chart.
            if filtered.empty:
                filtered = sym_df[date_mask].copy()
            if filtered.empty:
                filtered = sym_df.tail(MAX_HISTORY).copy()

            # DEBUG: Print results (comment out in production)
            if DEBUG_FILTER: print(f"[DEBUG] {symbol}: Filtered {len(sym_df)} -> {len(filtered)} rows (session mode)")

            return filtered
        else:
            # Legacy rolling window behavior (last N minutes)
            latest = sym_df['_ts_parsed'].max()
            cutoff = latest - pd.Timedelta(minutes=window_minutes)
            filtered = sym_df[sym_df['_ts_parsed'] >= cutoff].copy()

            # DEBUG: Print results (comment out in production)
            if DEBUG_FILTER: print(f"[DEBUG] {symbol}: Filtered {len(sym_df)} -> {len(filtered)} rows ({window_minutes}min window)")

            return filtered
    else:
        # No valid timestamps - fallback to tail
        return sym_df.tail(MAX_HISTORY)


def _single_ts_chart(x_time, y_data, title, chart_type='line', color=None,
                     fill=False, hline=None, height=320):
    """Create a single full-width time-series chart."""
    fig = go.Figure()
    if chart_type == 'bar':
        if isinstance(color, list):
            bar_colors = color
        else:
            bar_colors = color or C['accent']
        fig.add_trace(go.Bar(x=x_time, y=y_data,
            marker=dict(color=bar_colors, opacity=0.85), name=title))
    else:
        line_color = color if isinstance(color, str) else C['accent']
        kw = dict(x=x_time, y=y_data, mode='lines+markers',
            line=dict(color=line_color, width=2), marker=dict(size=4), name=title)
        if fill:
            kw['fill'] = 'tozeroy'
            kw['fillcolor'] = line_color.replace(')', ',0.15)').replace('rgb', 'rgba') if 'rgb' in line_color else f'rgba(100,100,200,0.15)'
        fig.add_trace(go.Scatter(**kw))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color=C['warning'], opacity=0.5)
    fig.update_xaxes(gridcolor=C['grid'], showgrid=True, type='date', tickformat='%H:%M')
    fig.update_yaxes(gridcolor=C['grid'], showgrid=True)
    fig.update_layout(height=height, showlegend=False, title=dict(text=title, font=dict(size=13)),
        paper_bgcolor=C['bg_dark'], plot_bgcolor=C['bg_card'],
        font=dict(color=C['text'], size=11),
        margin=dict(l=60, r=20, t=40, b=30), hovermode='x unified')
    return fig


def create_timeseries_individual(agg_df, symbol, window_minutes=30):
    """Returns list of (fig, insight_html) tuples for each time-series metric."""
    sym_df = _filter_by_window(agg_df, symbol, window_minutes)
    if sym_df.empty:
        return [(empty_chart("No time-series data", 280), "")]
    x_time = parse_agg_timestamps(sym_df)
    charts = []
    if 'pc_ratio' in sym_df.columns:
        fig = _single_ts_chart(x_time, sym_df['pc_ratio'], 'P/C Ratio',
            color=C['accent'], hline=1.0)
        val = sym_df['pc_ratio'].iloc[-1]
        text, anomaly = pc_ratio_insight(val)
        charts.append((fig, implication_box_html(text, anomaly) if text else ""))
    if 'net_gex' in sym_df.columns:
        gex_colors = [C['call'] if g > 0 else C['put'] for g in sym_df['net_gex']]
        fig = _single_ts_chart(x_time, sym_df['net_gex'], 'Net GEX',
            chart_type='bar', color=gex_colors)
        val = sym_df['net_gex'].iloc[-1]
        text, anomaly = gex_insight(val)
        charts.append((fig, implication_box_html(text, anomaly) if text else ""))
    if 'iv_skew' in sym_df.columns:
        fig = _single_ts_chart(x_time, sym_df['iv_skew'], 'IV Skew',
            color=C['put'], fill=True)
        val = sym_df['iv_skew'].iloc[-1]
        text, anomaly = iv_skew_insight(val)
        charts.append((fig, implication_box_html(text, anomaly) if text else ""))
    if 'atm_straddle' in sym_df.columns:
        fig = _single_ts_chart(x_time, sym_df['atm_straddle'], 'ATM Straddle',
            color=C['warning'])
        spot_raw = sym_df['spot'].iloc[-1] if 'spot' in sym_df.columns else 0
        val = sym_df['atm_straddle'].iloc[-1]
        text, anomaly = straddle_insight(val, spot_raw)
        charts.append((fig, implication_box_html(text, anomaly) if text else ""))
    has_premium = 'net_premium' in sym_df.columns and sym_df['net_premium'].notna().any()
    if has_premium:
        net_p = sym_df['net_premium'] / 1e6
        np_colors = [C['call'] if v >= 0 else C['put'] for v in net_p]
        fig = _single_ts_chart(x_time, net_p, 'Net Premium ($M)',
            chart_type='bar', color=np_colors)
        insight = ("<b>Net Premium ($M):</b> Call premium minus put premium. "
                   "Green bars = more dollar flow into calls (bullish conviction). "
                   "Red bars = put premium dominance (hedging/bearish bets).")
        charts.append((fig, implication_box_html(insight)))
        if 'call_premium' in sym_df.columns and 'put_premium' in sym_df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x_time, y=sym_df['call_premium'] / 1e6,
                mode='lines', line=dict(color=C['call'], width=2), fill='tozeroy',
                fillcolor='rgba(16,185,129,0.15)', name='Call $'))
            fig2.add_trace(go.Scatter(x=x_time, y=sym_df['put_premium'] / 1e6,
                mode='lines', line=dict(color=C['put'], width=2), fill='tozeroy',
                fillcolor='rgba(239,68,68,0.15)', name='Put $'))
            fig2.update_xaxes(gridcolor=C['grid'], showgrid=True, type='date', tickformat='%H:%M')
            fig2.update_yaxes(gridcolor=C['grid'], showgrid=True)
            fig2.update_layout(height=320, showlegend=True,
                title=dict(text='Call vs Put Premium ($M)', font=dict(size=13)),
                paper_bgcolor=C['bg_dark'], plot_bgcolor=C['bg_card'],
                font=dict(color=C['text'], size=11),
                margin=dict(l=60, r=20, t=40, b=30), hovermode='x unified',
                legend=dict(x=0.02, y=0.98))
            insight2 = ("<b>Call vs Put Premium:</b> Compares raw dollar flow. "
                        "When call premium surges above put = aggressive bullish bets. "
                        "Crossovers signal sentiment shifts.")
            charts.append((fig2, implication_box_html(insight2)))
    return charts


def create_microstructure_individual(agg_df, symbol, window_minutes=30):
    """Returns list of (fig, insight_html) tuples for each microstructure metric."""
    sym_df = _filter_by_window(agg_df, symbol, window_minutes)
    if sym_df.empty:
        return [(empty_chart("No microstructure data", 280), "")]
    has_data = any(c in sym_df.columns for c in
                   ['avg_spread_pct', 'bid_ask_imbalance', 'trade_aggression', 'avg_trade_size'])
    if not has_data:
        return [(empty_chart("Microstructure metrics not yet available - restart fetcher", 280), "")]
    x_time = parse_agg_timestamps(sym_df)
    charts = []
    if 'avg_spread_pct' in sym_df.columns:
        fig = _single_ts_chart(x_time, sym_df['avg_spread_pct'], 'Avg Spread %',
            color=C['warning'])
        insight = ("<b>Avg Spread %:</b> Widening = MMs pulling liquidity, expect volatility. "
                   "Narrowing = confident market-making, stable conditions. "
                   "Sudden spikes often precede large directional moves.")
        charts.append((fig, implication_box_html(insight)))
    if 'bid_ask_imbalance' in sym_df.columns:
        vals = sym_df['bid_ask_imbalance']
        colors = [C['call'] if v > 0 else C['put'] for v in vals]
        fig = _single_ts_chart(x_time, vals, 'Bid/Ask Size Imbalance',
            chart_type='bar', color=colors)
        recent = vals.iloc[-3:].mean() if len(vals) >= 3 else vals.iloc[-1]
        severity = 'warn' if abs(recent) > 0.15 else None
        insight = (f"<b>Bid/Ask Imbalance:</b> Current avg: {recent:.3f}. "
                   "Green = bid-heavy (buying pressure, demand absorbing supply). "
                   "Red = ask-heavy (selling pressure, supply overwhelming demand). "
                   "Persistent imbalance often foreshadows price direction.")
        charts.append((fig, implication_box_html(insight, severity)))
    if 'trade_aggression' in sym_df.columns:
        vals = sym_df['trade_aggression']
        colors = [C['call'] if v > 0 else C['put'] for v in vals]
        # v3.8: 2-panel — raw bars + normalized Z-score with smoothed trend
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.08, row_heights=[0.5, 0.5],
                            subplot_titles=['Trade Aggression (Raw)',
                                            'Trade Aggression (Normalized)'])
        # Row 1: raw bars
        fig.add_trace(go.Bar(x=x_time, y=vals, marker=dict(color=colors),
                             opacity=0.85, name='Raw', showlegend=False), row=1, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color=C['text_muted'],
                      opacity=0.4, row=1, col=1)
        # Row 2: robust Z-score + smoothed EMA trend line
        raw = pd.Series(vals.values, index=range(len(vals)))
        win = min(20, max(5, len(raw) // 4))
        roll_med = raw.rolling(win, min_periods=1).median()
        roll_mad = raw.rolling(win, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        roll_mad = roll_mad.replace(0, np.nan).fillna(0.001)
        z_score = ((raw - roll_med) / (roll_mad * 1.4826)).clip(-3, 3)
        z_smooth = z_score.ewm(span=max(3, win // 2), min_periods=1).mean()
        z_colors = [C['call'] if v > 0 else C['put'] for v in z_score]
        fig.add_trace(go.Bar(x=x_time, y=z_score.values, marker=dict(color=z_colors),
                             opacity=0.35, name='Z-Score', showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_time, y=z_smooth.values, mode='lines',
                                 line=dict(color=C['warning'], width=3),
                                 name='Smoothed Trend', showlegend=True), row=2, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color=C['text_muted'],
                      opacity=0.4, row=2, col=1)
        fig.add_hline(y=2, line_dash='dot', line_color=C['call'],
                      opacity=0.3, row=2, col=1)
        fig.add_hline(y=-2, line_dash='dot', line_color=C['put'],
                      opacity=0.3, row=2, col=1)
        fig.update_xaxes(gridcolor=C['grid'], showgrid=True, type='date', tickformat='%H:%M')
        fig.update_yaxes(gridcolor=C['grid'], showgrid=True)
        fig.update_layout(height=320, paper_bgcolor=C['bg_dark'], plot_bgcolor=C['bg_card'],
                          font=dict(color=C['text'], size=11),
                          margin=dict(l=60, r=20, t=50, b=30), hovermode='x unified',
                          legend=dict(x=0.02, y=0.48, font=dict(size=10)))
        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=12, color=C['text'])
        recent = vals.iloc[-3:].mean() if len(vals) >= 3 else vals.iloc[-1]
        recent_z = z_smooth.iloc[-1] if len(z_smooth) > 0 else 0
        severity = 'alert' if abs(recent_z) > 2.5 else ('warn' if abs(recent_z) > 1.5 else None)
        insight = (f"<b>Trade Aggression:</b> Raw avg: {recent:.3f}, "
                   f"Normalized Z: {recent_z:.2f}. "
                   "<b>Top panel:</b> raw tick-by-tick aggression (noisy by design). "
                   "<b>Bottom panel:</b> robust Z-score (orange line = smoothed trend). "
                   "Z > +2 = sustained aggressive buying; "
                   "Z < −2 = sustained aggressive selling. "
                   "Dotted lines mark ±2σ thresholds.")
        charts.append((fig, implication_box_html(insight, severity)))
    if 'avg_trade_size' in sym_df.columns:
        fig = _single_ts_chart(x_time, sym_df['avg_trade_size'], 'Avg Trade Size',
            color=C['neutral'], fill=True)
        val = sym_df['avg_trade_size'].iloc[-1]
        avg = sym_df['avg_trade_size'].mean()
        severity = 'warn' if val > avg * 1.5 else None
        insight = (f"<b>Avg Trade Size:</b> Current: {val:.1f}, session avg: {avg:.1f}. "
                   "Rising = institutional block trades entering. "
                   "Falling = retail fragmentation. "
                   "Spikes above 2x average often signal large fund positioning.")
        charts.append((fig, implication_box_html(insight, severity)))
    return charts


# ========================================
# HTML HELPERS
# ========================================

def section_header_html(title, subtitle=""):
    return f"""
    <div style="background:{C['bg_card']};padding:12px 18px;border-radius:8px;
                margin:18px 0 6px 0;border-left:4px solid {C['accent']};">
        <div style="color:{C['text']};font-size:1.1em;font-weight:600;">{title}</div>
        <div style="color:{C['text_muted']};font-size:0.78em;margin-top:3px;">{subtitle}</div>
    </div>"""

def metric_card_html(label, value, color=None):
    c = color or C['text']
    return f"""
    <div style="min-width:130px;">
        <div style="color:{C['text_muted']};font-size:0.72em;text-transform:uppercase;
                    letter-spacing:0.05em;">{label}</div>
        <div style="color:{c};font-size:1.5em;font-weight:600;">{value}</div>
    </div>"""

def alert_html(icon, title, message, color):
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    return f"""
    <div style="background:rgba({r},{g},{b},0.1);
                padding:12px 16px;border-radius:6px;margin-bottom:10px;
                border-left:4px solid {color};font-family:sans-serif;">
        <div style="font-weight:600;font-size:0.875rem;color:{C['text']};">{icon} {title}</div>
        <div style="font-size:0.85rem;color:{C['text_sec']};margin-top:4px;">{message}</div>
    </div>"""


# ========================================
# MAIN RENDER
# ========================================


# ========================================
# VOL/OI RATIO CHART (v3.8)
# ========================================

def create_vol_oi_chart(options_df, symbol, spot=0):
    """Vol/OI ratio by strike — updates every snapshot as volume changes."""
    if options_df is None or options_df.empty:
        return empty_chart('No Vol/OI data')
    sym = options_df[options_df['symbol'] == symbol]
    if sym.empty or 'volume' not in sym.columns or 'oi' not in sym.columns:
        return empty_chart('No volume/OI columns')
    sym = sym.copy()
    sym['oi_safe'] = sym['oi'].replace(0, np.nan)
    sym['vol_oi'] = (sym['volume'] / sym['oi_safe']).fillna(0)
    if spot and spot > 0:
        lo, hi = spot * 0.93, spot * 1.07
        sym = sym[(sym['strike'] >= lo) & (sym['strike'] <= hi)]
    if sym.empty:
        return empty_chart('No Vol/OI data in range')
    calls = sym[sym['cp_sign'] == 1].groupby('strike').agg(vol_oi=('vol_oi', 'mean')).reset_index()
    puts = sym[sym['cp_sign'] == -1].groupby('strike').agg(vol_oi=('vol_oi', 'mean')).reset_index()
    fig = go.Figure()
    if not calls.empty:
        fig.add_trace(go.Bar(x=calls['strike'], y=calls['vol_oi'],
                             marker=dict(color=C['call'], opacity=0.7), name='Call Vol/OI'))
    if not puts.empty:
        fig.add_trace(go.Bar(x=puts['strike'], y=puts['vol_oi'],
                             marker=dict(color=C['put'], opacity=0.7), name='Put Vol/OI'))
    fig.add_hline(y=1.0, line_dash='dash', line_color=C['warning'], line_width=2,
                  annotation_text='Vol/OI = 1 (new money)')
    if spot and spot > 0:
        fig.add_vline(x=spot, line_dash='dash', line_color=C['warning'], line_width=2,
                      annotation_text=f'Spot {spot:.0f}')
    fig.update_layout(**base_layout(height=400, barmode='group',
        xaxis_title='Strike Price', yaxis_title='Volume / Open Interest'))
    style_axes(fig)
    return fig


def vol_oi_insight(options_df, symbol):
    """Insight for Vol/OI chart."""
    if options_df is None or options_df.empty:
        return None, None
    sym = options_df[options_df['symbol'] == symbol]
    if sym.empty or 'volume' not in sym.columns or 'oi' not in sym.columns:
        return None, None
    sym = sym.copy()
    sym['oi_safe'] = sym['oi'].replace(0, np.nan)
    sym['vol_oi'] = (sym['volume'] / sym['oi_safe']).fillna(0)
    hot_strikes = sym[sym['vol_oi'] > 1.0]
    n_hot = len(hot_strikes['strike'].unique()) if not hot_strikes.empty else 0
    max_voi = sym['vol_oi'].max()
    text = (f"<b>Volume/OI Ratio:</b> {n_hot} strikes with Vol/OI > 1.0 "
            f"(new positions). Peak ratio: {max_voi:.1f}x. "
            "Vol/OI > 1 = more contracts traded today than total open interest "
            "— signals fresh aggressive positioning. "
            "Vol/OI > 3 = highly unusual, likely institutional block activity. "
            "<i>Updates every snapshot as volume accumulates.</i>")
    anomaly = 'alert' if max_voi > 5 else ('warn' if max_voi > 2 else None)
    return text, anomaly


# ========================================
# CUMULATIVE VOLUME DELTA (v3.8)
# ========================================

def create_cum_vol_delta_chart(agg_df, symbol, window_minutes=30):
    """Cumulative call vol minus put vol — directional flow pressure."""
    if agg_df is None or agg_df.empty:
        return empty_chart('No cumulative volume delta data')
    sym_df = _filter_by_window(agg_df, symbol, window_minutes)
    if sym_df.empty:
        return empty_chart('No data for symbol')
    cv_col = next((c for c in ['call_vol', 'callvol'] if c in sym_df.columns), None)
    pv_col = next((c for c in ['put_vol', 'putvol'] if c in sym_df.columns), None)
    if not cv_col or not pv_col:
        return empty_chart('No call/put volume columns')
    x_time = parse_agg_timestamps(sym_df)
    call_v = pd.to_numeric(sym_df[cv_col], errors='coerce').fillna(0)
    put_v = pd.to_numeric(sym_df[pv_col], errors='coerce').fillna(0)
    delta = call_v.values - put_v.values
    cum_delta = np.cumsum(delta)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_time, y=cum_delta, mode='lines', fill='tozeroy',
        line=dict(color=C['accent'], width=2),
        fillcolor='rgba(59,130,246,0.15)', name='Cum Vol Delta'))
    bar_colors = [C['call'] if v > 0 else C['put'] for v in delta]
    fig.add_trace(go.Bar(
        x=x_time, y=delta, marker=dict(color=bar_colors),
        opacity=0.3, name='Per-Batch Δ', yaxis='y2'))
    fig.add_hline(y=0, line_dash='dash', line_color=C['text_muted'], line_width=1)
    layout_kw = base_layout(height=320)
    layout_kw['xaxis'] = dict(title='Time', type='date', tickformat='%H:%M',
                              gridcolor=C['grid'], showgrid=True)
    layout_kw['yaxis'] = dict(title='Cumulative Volume Delta',
                              gridcolor=C['grid'], showgrid=True)
    layout_kw['yaxis2'] = dict(overlaying='y', side='right', showgrid=False,
                               title='Per-Batch Δ', color=C['text_muted'])
    layout_kw['hovermode'] = 'x unified'
    layout_kw['legend'] = dict(x=0.02, y=0.98, font=dict(size=10))
    fig.update_layout(**layout_kw)
    return fig


def cum_vol_delta_insight(agg_df, symbol, window_minutes=30):
    """Insight for cumulative volume delta."""
    if agg_df is None or agg_df.empty:
        return None, MC["border"], None
    sym_df = _filter_by_window(agg_df, symbol, window_minutes)
    if sym_df.empty:
        return None, MC["border"], None
    cv_col = next((c for c in ['call_vol', 'callvol'] if c in sym_df.columns), None)
    pv_col = next((c for c in ['put_vol', 'putvol'] if c in sym_df.columns), None)
    if not cv_col or not pv_col:
        return None, None
    call_v = pd.to_numeric(sym_df[cv_col], errors='coerce').fillna(0)
    put_v = pd.to_numeric(sym_df[pv_col], errors='coerce').fillna(0)
    delta = call_v.values - put_v.values
    cum = np.cumsum(delta)
    final = cum[-1] if len(cum) > 0 else 0
    trend = 'bullish (call-dominated)' if final > 0 else 'bearish (put-dominated)'
    text = (f"<b>Cumulative Volume Delta:</b> Net {final:,.0f} contracts "
            f"({trend}). Rising line = sustained call volume dominance. "
            "Falling = put dominance. Reversals in the cum-delta often "
            "lead price by 5-15 min. <i>Updates every snapshot.</i>")
    anomaly = 'warn' if abs(final) > 10000 else None
    return text, anomaly



def _apply_dte_filter(df, dte_filter='0_1dte'):
    """Filter a DataFrame by DTE bucket (v3.10).
    'all' = no filter, '0dte' = DTE==0 only,
    '0_1dte' = DTE <= 1 (default, best for credit spreads),
    '0_2dte' = DTE <= 2."""
    if df is None or df.empty or dte_filter == 'all':
        return df
    if 'dte' not in df.columns:
        return df
    dte_vals = pd.to_numeric(df['dte'], errors='coerce')
    if dte_filter == '0dte':
        mask = dte_vals == 0
    elif dte_filter == '0_1dte':
        mask = dte_vals <= 1
    elif dte_filter == '0_2dte':
        mask = dte_vals <= 2
    else:
        return df
    filtered = df[mask].copy()
    return filtered if not filtered.empty else df


def _apply_dte_filter_agg(df, dte_filter='0_1dte'):
    """Filter agg DataFrame by dte_group column if present (v3.10).
    Falls back to 'all' rows if dte_group column doesn't exist
    (backwards-compatible with old agg files from fetcher v5.0/v5.1)."""
    if df is None or df.empty or dte_filter == 'all':
        return df
    if 'dte_group' not in df.columns:
        return df  # old fetcher without DTE split - show everything
    if dte_filter == '0dte':
        filtered = df[df['dte_group'] == '0dte'].copy()
    elif dte_filter == '0_1dte':
        filtered = df[df['dte_group'] == '0_1dte'].copy()
    elif dte_filter == '0_2dte':
        filtered = df[df['dte_group'] == '0_2dte'].copy()
    else:
        return df
    # Fall back to 'all' rows if selected dte_group has no data yet
    return filtered if not filtered.empty else df[df['dte_group'] == 'all'].copy()



# ---------------------------------------------------------------------------
# Cache variables
# ---------------------------------------------------------------------------
_cached_agg_df = None
_cached_snap_df = None
_cached_agg_mtime = None
_cached_agg_size = None
_cached_snap_mtime = None
_cached_snap_size = None
_cached_filtered_data = {}
_cached_pred_df = None
_cached_pred_mtime = None
_cached_pred_size = None
_cache_max_age = 30  # force re-read after this many seconds
_cached_agg_ts = 0.0
_cached_snap_ts = 0.0
_cached_pred_ts = 0.0


def _file_changed(path, cached_mtime, cached_size, cached_ts):
    """Return (changed, mtime, size) using mtime+size+max-age as signals."""
    if not path.exists():
        return (cached_mtime is not None), None, None  # file removed
    try:
        st = path.stat()
        mt, sz = st.st_mtime, st.st_size
    except OSError:
        return False, cached_mtime, cached_size
    age = time.time() - cached_ts if cached_ts else 999
    changed = (mt != cached_mtime) or (sz != cached_size) or (age > _cache_max_age)
    return changed, mt, sz

# Alert system
_alerts_log = deque(maxlen=50)
_last_alert_state = {}

# Track last non-suppressed prediction time for countdown
_last_live_non_suppressed_ts = None


# ---------------------------------------------------------------------------
# Prediction CSV reading
# ---------------------------------------------------------------------------

def _load_prediction_csv():
    """Load prediction.csv with mtime+size+max-age caching. Returns DataFrame or empty DataFrame."""
    global _cached_pred_df, _cached_pred_mtime, _cached_pred_size, _cached_pred_ts
    pred_path = DATA_DIR / "prediction.csv"
    changed, new_mt, new_sz = _file_changed(
        pred_path, _cached_pred_mtime, _cached_pred_size, _cached_pred_ts
    )
    if not changed and _cached_pred_df is not None:
        return _cached_pred_df
    if not pred_path.exists():
        _cached_pred_df = pd.DataFrame()
        _cached_pred_mtime = None
        _cached_pred_size = None
        # #region agent log
        _debug_log(
            "H2_missing_or_stale_pred",
            "theta_dashboard_v4_modern.py:_load_prediction_csv",
            "prediction.csv missing",
            {"path": str(pred_path)},
        )
        # #endregion
        return _cached_pred_df
    try:
        df = pd.read_csv(pred_path, on_bad_lines='skip')
        df = _normalize_prediction_df(df)
        # #region agent log
        _ts_min = None
        _ts_max = None
        if not df.empty and "ts" in df.columns:
            _tmp_ts = pd.to_datetime(df["ts"].astype(str).str.replace(r"\s+[A-Z]{2,5}$", "", regex=True), errors="coerce").dropna()
            if not _tmp_ts.empty:
                _ts_min = str(_tmp_ts.min())
                _ts_max = str(_tmp_ts.max())
        _debug_log(
            "H2_missing_or_stale_pred",
            "theta_dashboard_v4_modern.py:_load_prediction_csv",
            "prediction.csv loaded",
            {
                "rows": int(len(df)),
                "has_batch_id": bool("batch_id" in df.columns),
                "ts_min": _ts_min,
                "ts_max": _ts_max,
                "cached_mtime": float(new_mt) if new_mt is not None else None,
                "cached_size": int(new_sz) if new_sz is not None else None,
            },
        )
        # #endregion
        _cached_pred_df = df
        _cached_pred_mtime = new_mt
        _cached_pred_size = new_sz
        _cached_pred_ts = time.time()
        return df
    except Exception:
        _cached_pred_df = pd.DataFrame()
        _cached_pred_mtime = None
        _cached_pred_size = None
        return _cached_pred_df


def _get_latest_prediction(pred_df):
    """Get latest prediction row as dict, or None."""
    if pred_df is None or pred_df.empty:
        return None
    return pred_df.iloc[-1].to_dict()


def _get_prediction_history(pred_df, n=40):
    """Get last N predictions for sparklines, accuracy, persistence checks."""
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()
    return pred_df.tail(n)


def _normalize_prediction_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dashboard prediction history for stable downstream rendering.

    - Keeps the latest row per batch_id when duplicates exist.
    - Preserves raw row order semantics when no batch_id is available.
    """
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    df = pred_df.copy()
    df["_row_ord"] = np.arange(len(df))
    if "ts" in df.columns:
        ts_clean = df["ts"].astype(str).str.replace(r"\s+[A-Z]{2,5}$", "", regex=True)
        df["_ts_sort"] = pd.to_datetime(ts_clean, errors="coerce")
    else:
        df["_ts_sort"] = pd.NaT

    if "batch_id" in df.columns:
        df["_batch_id_num"] = pd.to_numeric(df["batch_id"], errors="coerce")
        has_batch = df["_batch_id_num"].notna()
        if has_batch.any():
            dedup = df[has_batch].sort_values(["_batch_id_num", "_ts_sort", "_row_ord"])
            dedup = dedup.drop_duplicates(subset=["_batch_id_num"], keep="last")
            no_batch = df[~has_batch]
            df = pd.concat([dedup, no_batch], ignore_index=True)
            df = df.sort_values(["_ts_sort", "_row_ord"])

    return df.drop(columns=["_row_ord", "_ts_sort", "_batch_id_num"], errors="ignore").reset_index(drop=True)


def _prediction_row_to_model_out(row_dict):
    """
    Adapt a flat CSV prediction row dict into the model_out dict format
    that the dashboard components expect.

    CSV columns: batch_id, ts, prob, pred, threshold, confidence, signal_strength,
    direction, agent_A_prob..agent_2D_prob, gate_A..gate_2D,
    quality_score, feature_completeness, warmup_fraction, latency_ms,
    stage1_missing_count, suppressed, reason, vix_level, spot_price
    """
    if row_dict is None:
        return None

    def _safe_float(v, default):
        try:
            if v is None or v == "":
                return default
            fv = float(v)
            return fv if np.isfinite(fv) else default
        except Exception:
            return default

    suppressed = str(row_dict.get("suppressed", "False")).strip().lower() in ("true", "1", "yes")
    prob = _safe_float(row_dict.get("prob", 0.5), 0.5)
    confidence = _safe_float(row_dict.get("confidence", 0.0), 0.0)
    signal_strength = _safe_float(row_dict.get("signal_strength", 0.0), 0.0)
    # Evidence-based confidence decomposition (new columns)
    agent_std = _safe_float(row_dict.get("agent_std", 0.0), 0.0)
    consensus_ratio = _safe_float(row_dict.get("consensus_ratio", 0.0), 0.0)
    conf_agreement = _safe_float(row_dict.get("conf_agreement", 0.0), 0.0)
    conf_consensus = _safe_float(row_dict.get("conf_consensus", 0.0), 0.0)
    conf_gate_conviction = _safe_float(row_dict.get("conf_gate_conviction", 0.0), 0.0)
    conf_data_quality = _safe_float(row_dict.get("conf_data_quality", 0.0), 0.0)
    pred = int(_safe_float(row_dict.get("pred", 0), 0.0))
    threshold = _safe_float(row_dict.get("threshold", 0.36), 0.36)
    reason = str(row_dict.get("reason", "") or "")
    direction = str(row_dict.get("direction", "") or "")
    if not direction:
        direction = "BULL" if pred == 1 else "BEAR"

    # Build stage2_probs dict from flat agent columns
    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]
    stage2_probs = {}
    for k in agent_keys:
        col = f"agent_{k}_prob"
        _fallback = AGENT_TRAIN_MEDIAN.get(k, 0.5)
        stage2_probs[k] = _safe_float(row_dict.get(col, _fallback), _fallback)

    # Build gates dict
    gates = {}
    for k in agent_keys:
        col = f"gate_{k}"
        _gv = row_dict.get(col, 1.0)
        gates[k] = _safe_float(_gv, 1.0)

    # Diagnostics sub-dict
    quality_score = _safe_float(row_dict.get("quality_score", 0.0), 0.0)
    feature_completeness = _safe_float(row_dict.get("feature_completeness", 0.0), 0.0)
    warmup_fraction = _safe_float(row_dict.get("warmup_fraction", 0.0), 0.0)
    latency_ms = row_dict.get("latency_ms", None)
    if latency_ms is not None:
        try:
            latency_ms = float(latency_ms)
        except (ValueError, TypeError):
            latency_ms = None
    stage1_missing_count = int(_safe_float(row_dict.get("stage1_missing_count", 0), 0.0))
    vix_level = _safe_float(row_dict.get("vix_level", 0.0), 0.0)
    spot_price = _safe_float(row_dict.get("spot_price", 0.0), 0.0)

    # vix_valid: treat 0.0 as "data unavailable" (not stressed).
    # Only mark invalid when VIX is genuinely out of range (> 80).
    # Missing VIXW data should not penalise sizing — use vix_known to
    # distinguish "data absent" from "high vol" in the sizing logic.
    vix_known = vix_level > 0.0
    vix_valid = (not vix_known) or (5.0 < vix_level < 80.0)

    diagnostics = {
        "quality_score": quality_score,
        "feature_completeness": feature_completeness,
        "warmup_fraction": warmup_fraction,
        "latency_ms": latency_ms,
        "stage1_missing_count": stage1_missing_count,
        "vix_valid": vix_valid,
        "vix_level": vix_level,
    }

    ok = not suppressed

    if suppressed:
        source_state = "SUPPRESSED"
    else:
        source_state = "CSV_PREDICTION"

    return {
        "ok": ok,
        "suppressed": suppressed,
        "reason": reason,
        "prob": prob,
        "pred": pred,
        "threshold": threshold,
        "confidence": confidence,
        "signal_strength": signal_strength,
        "direction": direction,
        "stage2_probs": stage2_probs,
        "gates": gates,
        "diagnostics": diagnostics,
        "source_state": source_state,
        "cache_hit": False,
        "spot_price": spot_price,
        "vix_level": vix_level,
        "batch_id": row_dict.get("batch_id"),
        "ts": row_dict.get("ts"),
        # Confidence decomposition
        "agent_std": agent_std,
        "consensus_ratio": consensus_ratio,
        "conf_agreement": conf_agreement,
        "conf_consensus": conf_consensus,
        "conf_gate_conviction": conf_gate_conviction,
        "conf_data_quality": conf_data_quality,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_batch_id(df):
    try:
        if df is not None and not df.empty and "batch_id" in df.columns:
            s = pd.to_numeric(df["batch_id"], errors="coerce").dropna()
            if not s.empty:
                return int(s.max())
    except Exception:
        pass
    return None


def _latest_time_marker(df):
    if df is None or df.empty:
        return None
    if "_ts_parsed" in df.columns:
        try:
            ts = pd.to_datetime(df["_ts_parsed"], errors="coerce").dropna()
            if not ts.empty:
                return ts.max().isoformat()
        except Exception:
            pass
    if "ts" in df.columns:
        try:
            ts = pd.to_datetime(df["ts"], errors="coerce").dropna()
            if not ts.empty:
                return ts.max().isoformat()
        except Exception:
            pass
    return None


def _prediction_history_as_roll(pred_df):
    """
    Convert prediction history DataFrame to a list of dicts matching
    the old _model_roll_history format, for components that rely on it.
    """
    if pred_df is None or pred_df.empty:
        return []
    records = []
    for _, row in pred_df.iterrows():
        d = row.to_dict()
        suppressed = str(d.get("suppressed", "False")).strip().lower() in ("true", "1", "yes")
        prob = float(d.get("prob", 0.5) or 0.5)
        confidence = float(d.get("confidence", 0.0) or 0.0)
        strength = float(d.get("signal_strength", 0.0) or 0.0)
        # Confidence decomposition
        agent_std = float(d.get("agent_std", 0.0) or 0.0)
        consensus_ratio = float(d.get("consensus_ratio", 0.0) or 0.0)
        conf_agreement = float(d.get("conf_agreement", 0.0) or 0.0)
        conf_consensus = float(d.get("conf_consensus", 0.0) or 0.0)
        conf_gate_conviction = float(d.get("conf_gate_conviction", 0.0) or 0.0)
        conf_data_quality = float(d.get("conf_data_quality", 0.0) or 0.0)

        agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]
        stage2_probs = {}
        for k in agent_keys:
            stage2_probs[k] = float(d.get(f"agent_{k}_prob", 0.5) or 0.5)

        ts_raw = d.get("ts", "")
        ts_dt = pd.to_datetime(ts_raw, errors="coerce")
        if pd.isna(ts_dt):
            # Do not inject current wall-clock time — use None so charts skip the point
            ts_dt = None

        records.append({
            "batch_id": int(float(d.get("batch_id", -1) or -1)),
            "ts": ts_dt,
            "suppressed": suppressed,
            "prob": prob,
            "confidence": confidence,
            "strength": strength,
            "stage2_probs": stage2_probs,
            "source_state": "SUPPRESSED" if suppressed else "CSV_PREDICTION",
            "cache_hit": False,
            "agent_std": agent_std,
            "consensus_ratio": consensus_ratio,
            "conf_agreement": conf_agreement,
            "conf_consensus": conf_consensus,
            "conf_gate_conviction": conf_gate_conviction,
            "conf_data_quality": conf_data_quality,
        })
    return records


# ---------------------------------------------------------------------------
# Prediction unavailable card
# ---------------------------------------------------------------------------

def _prediction_unavailable_card():
    """Friendly card when prediction.csv is missing or empty."""
    return html.Div(
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderRadius": "8px",
            "padding": "24px",
            "textAlign": "center",
            "marginBottom": "16px",
        },
        children=[
            html.Div("MODEL PREDICTION", style={
                "fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px",
                "color": MC["accent"], "marginBottom": "12px",
            }),
            html.Div("Waiting for prediction data...", style={
                "fontSize": "16px", "color": MC["text_muted"], "marginBottom": "8px",
            }),
            html.Div("Start prediction_service.py to enable model predictions.", style={
                "fontSize": "13px", "color": MC["text_muted"], "marginBottom": "12px",
            }),
            html.Code("python prediction_service.py", style={
                "backgroundColor": MC["bg_dark"],
                "padding": "8px 16px",
                "borderRadius": "4px",
                "fontSize": "13px",
                "color": MC["accent"],
            }),
        ]
    )


# ---------------------------------------------------------------------------
# Alert system
# ---------------------------------------------------------------------------

def _generate_alerts(model_out):
    """Check model_out against previous state and generate alerts."""
    global _last_alert_state
    if not model_out:
        return

    now_str = _now_et_naive().strftime("%H:%M:%S")
    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    prob = float(model_out.get("prob", 0.5) or 0.5)
    confidence = float(model_out.get("confidence", 0.0) or 0.0)
    diagnostics = model_out.get("diagnostics", {}) or {}
    quality = float(diagnostics.get("quality_score", 0.0) or 0.0)
    stage2_probs = dict(model_out.get("stage2_probs", {}) or {})
    _thr = float(model_out.get("threshold", 0.36) or 0.36)
    direction = "UP" if prob >= _thr else "DOWN"

    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]
    up_count = sum(1 for k in agent_keys if float(stage2_probs.get(k, AGENT_TRAIN_MEDIAN.get(k, 0.5))) >= AGENT_TRAIN_MEDIAN.get(k, 0.5))

    prev = _last_alert_state

    # Signal flip
    prev_dir = prev.get("direction")
    if prev_dir is not None and prev_dir != direction and not suppressed and ok:
        _alerts_log.append({
            "ts": now_str, "severity": "high",
            "msg": f"Signal FLIPPED: {prev_dir} -> {direction} (P(up)={prob:.1%})"
        })
        _last_alert_state["flip_ts"] = _now_et_naive()
        _last_alert_state["flip_to"] = direction

    # Confidence collapse
    prev_conf = prev.get("confidence")
    if prev_conf is not None and prev_conf >= 0.3 and confidence < 0.3 and not suppressed:
        _alerts_log.append({
            "ts": now_str, "severity": "high",
            "msg": f"Confidence COLLAPSED to {confidence:.1%} (was {prev_conf:.1%})"
        })

    # High conviction
    if prev_conf is not None and prev_conf < 0.7 and confidence >= 0.7 and not suppressed and ok:
        _alerts_log.append({
            "ts": now_str, "severity": "info",
            "msg": f"High conviction alert: confidence {confidence:.1%} {direction}"
        })

    # Consensus shift
    prev_consensus = prev.get("consensus")
    if prev_consensus is not None and abs(up_count - prev_consensus) >= 3:
        _alerts_log.append({
            "ts": now_str, "severity": "medium",
            "msg": f"Consensus shifted: {prev_consensus}/7 -> {up_count}/7 agents UP"
        })

    # Quality drop
    prev_quality = prev.get("quality")
    if prev_quality is not None and prev_quality >= 0.5 and quality < 0.5:
        _alerts_log.append({
            "ts": now_str, "severity": "medium",
            "msg": f"Quality dropped below threshold: {quality:.2f} (was {prev_quality:.2f})"
        })

    # Suppression state change
    prev_suppressed = prev.get("suppressed")
    if prev_suppressed is not None:
        if prev_suppressed and not suppressed and ok:
            _alerts_log.append({
                "ts": now_str, "severity": "info",
                "msg": "Model went LIVE (was suppressed)"
            })
        elif not prev_suppressed and suppressed:
            _alerts_log.append({
                "ts": now_str, "severity": "high",
                "msg": f"Model SUPPRESSED: {model_out.get('reason', 'unknown')}"
            })

    _last_alert_state = {
        "direction": direction if (not suppressed and ok) else prev.get("direction"),
        "confidence": confidence,
        "consensus": up_count,
        "quality": quality,
        "suppressed": suppressed,
        "flip_ts": prev.get("flip_ts") if prev.get("flip_to") == direction else None,
        "flip_to": prev.get("flip_to") if prev.get("flip_to") == direction else None,
    }


def _create_alert_panel():
    """Render the last 10 alerts in a scrollable list."""
    severity_colors = {
        "high": MC["put"],
        "medium": MC["warning"],
        "info": MC["call"],
    }

    alerts = list(_alerts_log)[-10:]
    if not alerts:
        rows = [html.Div(
            "No alerts yet -- monitoring for signal changes...",
            style={"color": MC["text_muted"], "fontSize": "12px", "padding": "8px"}
        )]
    else:
        rows = []
        for a in reversed(alerts):
            sev_color = severity_colors.get(a.get("severity", "info"), MC["text_muted"])
            rows.append(html.Div(
                style={
                    "display": "flex", "gap": "8px", "alignItems": "flex-start",
                    "padding": "5px 0",
                    "borderBottom": f"1px solid {MC['border']}",
                },
                children=[
                    html.Span(a["ts"], style={"fontSize": "11px", "color": MC["text_muted"], "minWidth": "65px"}),
                    html.Span("*", style={"color": sev_color, "fontSize": "10px", "marginTop": "2px"}),
                    html.Span(a["msg"], style={"fontSize": "12px", "color": MC["text"]}),
                ]
            ))

    return html.Div(
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderRadius": "8px",
            "padding": "14px",
            "maxHeight": "280px",
            "overflowY": "auto",
        },
        children=[
            html.Div("ALERTS", style={
                "fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px",
                "color": MC["warning"], "marginBottom": "8px",
            }),
            *rows,
        ]
    )


# ---------------------------------------------------------------------------
# Decision Engine Panel
# ---------------------------------------------------------------------------

def _create_decision_engine_panel(model_out, pred_history_roll):
    """Structured trade decision flowchart card."""
    if not model_out:
        return html.Div(
            style={
                "backgroundColor": MC["bg_card"],
                "border": f"1px solid {MC['border']}",
                "borderRadius": "8px", "padding": "16px", "flex": "1",
            },
            children=[
                html.Div("DECISION ENGINE", style={"fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px", "color": MC["accent"]}),
                html.Div("Awaiting model data...", style={"color": MC["text_muted"], "fontSize": "13px", "marginTop": "10px"}),
            ]
        )

    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    prob = float(model_out.get("prob", 0.5) or 0.5)
    confidence = float(model_out.get("confidence", 0.0) or 0.0)
    diagnostics = model_out.get("diagnostics", {}) or {}
    stage2_probs = dict(model_out.get("stage2_probs", {}) or {})
    quality = float(diagnostics.get("quality_score", 0.0) or 0.0)
    vix_valid = bool(diagnostics.get("vix_valid", False))

    _thr = float(model_out.get("threshold", 0.36) or 0.36)
    direction = "UP" if prob >= _thr else "DOWN"
    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]

    # Compute checks
    regime_pass = vix_valid and quality > 0.5

    if confidence >= 0.7:
        conf_label = f"HIGH ({confidence:.0%})"
    elif confidence >= 0.5:
        conf_label = f"MEDIUM ({confidence:.0%})"
    else:
        conf_label = f"LOW ({confidence:.0%})"
    conf_pass = confidence >= 0.5

    # Signal persistence from prediction history
    recent = [h for h in pred_history_roll if not h["suppressed"]][-3:]
    if len(recent) >= 3:
        if direction == "UP":
            persistent = all(h["prob"] >= _thr for h in recent)
        else:
            persistent = all(h["prob"] < _thr for h in recent)
        persist_label = f"{sum(1 for h in recent if (h['prob'] >= _thr) == (direction == 'UP'))}/3 bars {direction}"
    else:
        persistent = False
        persist_label = f"{len(recent)}/3 bars (need 3)"

    # Agent consensus — each agent judged vs its own training median
    if direction == "UP":
        consensus_count = sum(1 for k in agent_keys if float(stage2_probs.get(k, AGENT_TRAIN_MEDIAN.get(k, 0.5))) >= AGENT_TRAIN_MEDIAN.get(k, 0.5))
    else:
        consensus_count = sum(1 for k in agent_keys if float(stage2_probs.get(k, AGENT_TRAIN_MEDIAN.get(k, 0.5))) < AGENT_TRAIN_MEDIAN.get(k, 0.5))
    consensus_pass = consensus_count >= 4

    # Final action
    if suppressed or not ok:
        action_text = "PAUSED NO SIGNAL"
        action_color = MC["text_muted"]
        action_detail = "Model suppressed or unavailable"
    elif confidence < 0.5:
        action_text = "PAUSED WAIT -- Low Confidence"
        action_color = MC["text_muted"]
        action_detail = f"Confidence {confidence:.0%} below 50% threshold"
    elif consensus_count < 4:
        action_text = "PAUSED WAIT -- Low Consensus"
        action_color = MC["text_muted"]
        action_detail = f"Only {consensus_count}/7 agents agree"
    elif not persistent:
        action_text = "PAUSED WAIT -- Signal Not Stable"
        action_color = MC["warning"]
        action_detail = "Last 3 predictions don't agree on direction"
    elif confidence >= 0.7 and consensus_count >= 5 and persistent:
        action_text = f">> ENTER {'LONG' if direction == 'UP' else 'SHORT'}"
        action_color = MC["call"]
        action_detail = "High conviction setup"
    elif confidence >= 0.5 and consensus_count >= 4 and persistent:
        action_text = f">> WATCH {'LONG' if direction == 'UP' else 'SHORT'}"
        action_color = MC["warning"]
        action_detail = "Marginal -- monitor for confirmation"
    else:
        action_text = ">> SKIP"
        action_color = MC["put"]
        action_detail = "Conditions not met"

    # Sizing guidance (inline)
    if suppressed or not ok or confidence < 0.55:
        size_mult = "0x"
        size_note = "Skip"
    elif confidence < 0.65:
        size_mult = "0.5x" if vix_valid else "0x"
        size_note = "Reduced" if vix_valid else "Elevated vol -> skip"
    elif confidence < 0.75:
        size_mult = "1.0x" if vix_valid else "0.5x"
        size_note = "Standard" if vix_valid else "Elevated vol -> scale down"
    else:
        size_mult = "1.25x" if vix_valid else "0.75x"
        size_note = "Full" if vix_valid else "Elevated vol -> scale down"

    def _check_row(label, passed, detail_text):
        icon = "PASS" if passed else "FAIL"
        color = MC["call"] if passed else MC["put"]
        return html.Div(
            style={"display": "flex", "gap": "8px", "alignItems": "center", "padding": "4px 0"},
            children=[
                html.Span(icon, style={"fontSize": "11px", "fontWeight": 700, "color": color, "minWidth": "32px"}),
                html.Span(f"{label}:", style={"fontSize": "12px", "color": MC["text_muted"], "minWidth": "90px"}),
                html.Span(detail_text, style={"fontSize": "12px", "color": color, "fontWeight": 600}),
            ]
        )

    return html.Div(
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderRadius": "8px", "padding": "16px", "flex": "1",
        },
        children=[
            html.Div("DECISION ENGINE", style={
                "fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px",
                "color": MC["accent"], "marginBottom": "10px",
                "borderBottom": f"1px solid {MC['border']}", "paddingBottom": "6px",
            }),
            _check_row("Confidence", conf_pass, conf_label),
            _check_row("Regime", regime_pass, "PASS" if regime_pass else "FAIL"),
            _check_row("Consensus", consensus_pass, f"{consensus_count}/7 agents {direction}"),
            _check_row("Persistence", persistent, persist_label),
            _check_row("Quality", quality >= 0.5, f"{quality:.2f}"),
            html.Div(style={
                "borderTop": f"1px solid {MC['border']}",
                "marginTop": "10px", "paddingTop": "8px",
            }, children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.Span(f">> SUGGESTED SIZE: {size_mult}", style={
                            "fontSize": "13px", "fontWeight": 700, "color": MC["text"],
                        }),
                    ]
                ),
                html.Div(size_note, style={"fontSize": "11px", "color": MC["text_muted"], "marginTop": "2px"}),
            ]),
            html.Div(style={
                "borderTop": f"1px solid {MC['border']}",
                "marginTop": "10px", "paddingTop": "10px", "textAlign": "center",
            }, children=[
                html.Div(action_text, style={
                    "fontSize": "20px", "fontWeight": 800, "color": action_color,
                }),
                html.Div(action_detail, style={
                    "fontSize": "12px", "color": MC["text_muted"], "marginTop": "3px",
                }),
            ]),
        ]
    )


# ---------------------------------------------------------------------------
# Confidence Decomposition Row (evidence-based)
# ---------------------------------------------------------------------------

def _build_confidence_decomposition_row(model_out, colors):
    """Compact row showing the 4 confidence components as mini-bars."""
    if not model_out or model_out.get("suppressed", False) or not model_out.get("ok", False):
        return html.Div()  # Empty when suppressed or no data

    conf_agreement = model_out.get("conf_agreement", 0.0)
    conf_consensus = model_out.get("conf_consensus", 0.0)
    conf_gate = model_out.get("conf_gate_conviction", 0.0)
    conf_dq = model_out.get("conf_data_quality", 0.0)
    agent_std = model_out.get("agent_std", 0.0)
    consensus_ratio = model_out.get("consensus_ratio", 0.0)

    def _mini_bar(label, value, weight, tooltip):
        bar_color = "#4caf50" if value >= 0.7 else "#ff9800" if value >= 0.4 else "#ef5350"
        return html.Div(
            title=tooltip,
            style={"display": "flex", "alignItems": "center", "gap": "4px"},
            children=[
                html.Span(f"{label}", style={
                    "color": colors.get("text_muted", "#888"), "fontSize": "10px", "minWidth": "65px",
                }),
                html.Div(style={
                    "width": "60px", "height": "6px", "backgroundColor": colors.get("border", "#333"),
                    "borderRadius": "3px", "overflow": "hidden",
                }, children=[
                    html.Div(style={
                        "width": f"{value * 100:.0f}%", "height": "100%",
                        "backgroundColor": bar_color, "borderRadius": "3px",
                    })
                ]),
                html.Span(f"{value:.0%}", style={
                    "color": colors.get("text_sec", "#aaa"), "fontSize": "10px", "minWidth": "28px",
                }),
                html.Span(f"({weight})", style={
                    "color": colors.get("text_muted", "#888"), "fontSize": "9px",
                }),
            ]
        )

    return html.Div(
        style={"display": "flex", "gap": "16px", "alignItems": "center", "marginTop": "5px", "flexWrap": "wrap"},
        children=[
            html.Span("conf breakdown:", style={"color": colors.get("text_muted", "#888"), "fontSize": "10px"}),
            _mini_bar("agreement", conf_agreement, "40%", f"Agent std: {agent_std:.4f} \u2014 low std = high agreement"),
            _mini_bar("consensus", conf_consensus, "20%", f"Consensus: {consensus_ratio:.0%} of agents agree"),
            _mini_bar("conviction", conf_gate, "20%", "Gate-weighted agent conviction strength"),
            _mini_bar("data qual", conf_dq, "20%", "Feature completeness + warmup coverage"),
        ]
    )


# ---------------------------------------------------------------------------
# Agent Consensus HUD Strip
# ---------------------------------------------------------------------------

def _create_agent_hud_strip(model_out, symbol, agg_df):
    """Compact horizontal bar at top of model section."""
    if not model_out:
        return None

    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    prob = float(model_out.get("prob", 0.5) or 0.5)
    confidence = float(model_out.get("confidence", 0.0) or 0.0)
    diagnostics = model_out.get("diagnostics", {}) or {}
    stage2_probs = dict(model_out.get("stage2_probs", {}) or {})
    gates = dict(model_out.get("gates", {}) or {})
    vix_valid = bool(diagnostics.get("vix_valid", False))
    quality = float(diagnostics.get("quality_score", 0.0) or 0.0)

    # Spot price
    spot_text = "--"
    spot_val = model_out.get("spot_price", 0.0) or 0.0
    if spot_val and float(spot_val) > 0:
        spot_text = f"{float(spot_val):.2f}"
    elif agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns and symbol != "ALL":
        sym_agg = agg_df[agg_df["symbol"] == symbol]
        if not sym_agg.empty and "spot" in sym_agg.columns:
            sv = float(sym_agg.iloc[-1].get("spot", 0.0) or 0.0)
            if sv > 0:
                spot_text = f"{sv:.2f}"

    _thr = float(model_out.get("threshold", 0.36) or 0.36)
    direction = "UP" if prob >= _thr else "DOWN"
    dir_color = MC["call"] if prob >= _thr else MC["put"]

    if suppressed or not ok:
        dir_badge = html.Span("PAUSED SUPPRESSED", style={"color": MC["text_muted"], "fontWeight": 700, "fontSize": "18px"})
    else:
        dir_icon = "[+]" if prob >= _thr else "[-]"
        dir_badge = html.Span(f"{dir_icon} {direction} {prob:.0%}", style={"color": dir_color, "fontWeight": 700, "fontSize": "18px"})

    # Confidence badge
    if confidence >= 0.7:
        conf_badge_color = MC["call"]
        conf_label = "HIGH"
    elif confidence >= 0.5:
        conf_badge_color = MC["warning"]
        conf_label = "MED"
    else:
        conf_badge_color = MC["put"]
        conf_label = "LOW"

    # Regime
    regime_text = "CALM" if (vix_valid and quality > 0.5) else "STRESSED"
    regime_color = MC["call"] if regime_text == "CALM" else MC["warning"]

    # Countdown (30-min horizon from last prediction)
    countdown_text = "--"
    if _last_live_non_suppressed_ts is not None:
        elapsed = (_now_et_naive() - _last_live_non_suppressed_ts).total_seconds()
        remaining = max(0, 30 * 60 - elapsed)
        countdown_text = f"{int(remaining // 60)}m left"

    # Agent votes + gate weights
    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]
    agent_votes = []
    up_count = 0
    for k in agent_keys:
        val = float(stage2_probs.get(k, AGENT_TRAIN_MEDIAN.get(k, 0.5)))
        gate_val = float(gates.get(k, 1.0))
        neutral = AGENT_TRAIN_MEDIAN.get(k, 0.5)
        is_up = val >= neutral
        if is_up:
            up_count += 1
        arrow = "^" if is_up else "v"
        color = MC["call"] if is_up else MC["put"]
        if suppressed or not ok:
            color = MC["text_muted"]
        gate_opacity = f"{max(0.3, gate_val):.2f}"
        agent_votes.append(html.Span(
            children=[
                html.Span(f"{k}{arrow}", style={
                    "color": color, "fontWeight": 600, "fontSize": "14px",
                }),
                html.Span(f"g{gate_val:.2f}", style={
                    "color": MC["text_muted"], "fontSize": "11px",
                    "marginLeft": "1px", "opacity": gate_opacity,
                }),
            ],
            style={"marginRight": "8px", "display": "inline-flex", "alignItems": "baseline", "gap": "1px"},
        ))

    # Tracking status
    tracking_text = "--"
    tracking_color = MC["text_muted"]
    if (not suppressed and ok and agg_df is not None and not agg_df.empty
            and "symbol" in agg_df.columns and symbol != "ALL"):
        sym_agg = agg_df[agg_df["symbol"] == symbol]
        if len(sym_agg) >= 2 and "spot" in sym_agg.columns:
            recent_spot = pd.to_numeric(sym_agg["spot"], errors="coerce").dropna()
            if len(recent_spot) >= 2:
                spot_move = float(recent_spot.iloc[-1]) - float(recent_spot.iloc[-2])
                if (spot_move >= 0 and prob >= _thr) or (spot_move < 0 and prob < _thr):
                    tracking_text = "OK"
                    tracking_color = MC["call"]
                else:
                    tracking_text = "X"
                    tracking_color = MC["put"]

    return html.Div(
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderRadius": "8px", "padding": "12px 18px",
            "marginBottom": "12px",
        },
        children=[
            # Row 1
            html.Div(
                style={"display": "flex", "gap": "20px", "alignItems": "center", "flexWrap": "wrap"},
                children=[
                    html.Span(f"{symbol} {spot_text}", style={
                        "fontSize": "18px", "fontWeight": 700, "color": MC["text"],
                    }),
                    html.Span("|", style={"color": MC["border"]}),
                    dir_badge,
                    html.Span("|", style={"color": MC["border"]}),
                    html.Span("conf: ", style={"color": MC["text_muted"], "fontSize": "14px"}),
                    html.Span(f"{confidence:.0%} {conf_label}", style={
                        "color": conf_badge_color, "fontSize": "14px", "fontWeight": 700,
                        "backgroundColor": _hex_to_rgba(conf_badge_color, 0.13),
                        "padding": "2px 8px", "borderRadius": "4px",
                    }),
                    html.Span("|", style={"color": MC["border"]}),
                    html.Span("regime: ", style={"color": MC["text_muted"], "fontSize": "14px"}),
                    html.Span(regime_text, style={"color": regime_color, "fontSize": "14px", "fontWeight": 700}),
                    html.Span("|", style={"color": MC["border"]}),
                    html.Span(countdown_text, style={"color": MC["text_muted"], "fontSize": "14px"}),
                ]
            ),
            # Row 2
            html.Div(
                style={"display": "flex", "gap": "10px", "alignItems": "center", "marginTop": "8px", "flexWrap": "wrap"},
                children=[
                    *agent_votes,
                    html.Span("|", style={"color": MC["border"]}),
                    html.Span(f"consensus: {up_count}/7", style={
                        "color": MC["text_sec"], "fontSize": "14px",
                    }),
                    html.Span("|", style={"color": MC["border"]}),
                    html.Span("tracking: ", style={"color": MC["text_muted"], "fontSize": "14px"}),
                    html.Span(tracking_text, style={"color": tracking_color, "fontSize": "14px", "fontWeight": 700}),
                ]
            ),
            # Row 3: Confidence decomposition (evidence-based)
            _build_confidence_decomposition_row(model_out, MC),
        ]
    )


# ---------------------------------------------------------------------------
# Position Sizing Guidance
# ---------------------------------------------------------------------------

def _create_sizing_guidance(model_out, pred_history_roll, stats=None):
    """4-rank market risk panel (replaces position sizing)."""
    if not model_out:
        return html.Div(
            style={
                "backgroundColor": MC["bg_card"],
                "border": f"1px solid {MC['border']}",
                "borderRadius": "8px", "padding": "16px", "flex": "1",
            },
            children=[
                html.Div("RISK LEVEL", style={"fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px", "color": MC["accent"]}),
                html.Div("Awaiting model data...", style={"color": MC["text_muted"], "fontSize": "13px", "marginTop": "10px"}),
            ]
        )

    diagnostics = model_out.get("diagnostics", {}) or {}
    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    quality = float(diagnostics.get("quality_score", 0.0) or 0.0)
    completeness = float(diagnostics.get("feature_completeness", 0.0) or 0.0)

    vix_level = float(diagnostics.get("vix_level", 0.0) or 0.0)
    spot = float((stats or {}).get("price", model_out.get("spot_price", 0.0)) or 0.0)
    atm_straddle = float((stats or {}).get("atm_straddle", 0.0) or 0.0)
    net_gamma = float((stats or {}).get("net_gamma", 0.0) or 0.0)
    pc_ratio = float((stats or {}).get("pc_ratio", 1.0) or 1.0)
    iv_skew = float((stats or {}).get("iv_skew", 0.0) or 0.0)

    # Online convention anchors:
    # - VIX: <15 calm, 15-20 normal, 20-30 elevated, >=30 stressed.
    # - Implied move: ~0.85 * ATM straddle, normalized by spot.
    em_pct = (0.85 * atm_straddle / spot * 100.0) if (atm_straddle > 0 and spot > 0) else np.nan

    # 0..3 contribution from each primary factor.
    if vix_level <= 0:
        vix_score = 1
    elif vix_level < 15:
        vix_score = 0
    elif vix_level < 20:
        vix_score = 1
    elif vix_level < 30:
        vix_score = 2
    else:
        vix_score = 3

    if np.isnan(em_pct):
        em_score = 1
    elif em_pct < 0.8:
        em_score = 0
    elif em_pct < 1.4:
        em_score = 1
    elif em_pct < 2.2:
        em_score = 2
    else:
        em_score = 3

    gex_score = 2 if net_gamma < 0 else 0
    flow_score = int(pc_ratio >= 1.2) + int(abs(iv_skew) >= 0.05)
    health_score = 2 if (suppressed or not ok) else (1 if (quality < 0.60 or completeness < 0.75) else 0)

    risk_points = int(vix_score + em_score + gex_score + flow_score + health_score)

    if risk_points <= 2:
        risk_rank = "LOW"
        risk_color = MC["call"]
        rank_idx = 0
    elif risk_points <= 5:
        risk_rank = "MODERATE"
        risk_color = MC["accent"]
        rank_idx = 1
    elif risk_points <= 8:
        risk_rank = "HIGH"
        risk_color = MC["warning"]
        rank_idx = 2
    else:
        risk_rank = "EXTREME"
        risk_color = MC["put"]
        rank_idx = 3

    non_supp_count = sum(1 for h in pred_history_roll if not h.get("suppressed", True))
    em_txt = f"{em_pct:.2f}%" if np.isfinite(em_pct) else "N/A"
    vix_txt = f"{vix_level:.1f}" if vix_level > 0 else "N/A"

    tiers = [
        ("LOW", "0-2", "Calm / mean-reverting conditions"),
        ("MODERATE", "3-5", "Tradable but watch regime shifts"),
        ("HIGH", "6-8", "Vol elevated; reduce directional exposure"),
        ("EXTREME", "9+", "Stress regime; tail-risk control"),
    ]
    table_rows = []
    for i, (label, pts, note) in enumerate(tiers):
        active = (i == rank_idx)
        bg = _hex_to_rgba(risk_color, 0.11) if active else "transparent"
        table_rows.append(html.Tr(style={"backgroundColor": bg}, children=[
            html.Td(label, style={"padding": "4px 8px", "fontSize": "12px", "color": MC["text"], "fontWeight": 700 if active else 500}),
            html.Td(pts, style={"padding": "4px 8px", "fontSize": "12px", "color": MC["text_sec"], "textAlign": "center"}),
            html.Td(note, style={"padding": "4px 8px", "fontSize": "12px", "color": MC["text_muted"]}),
        ]))

    return html.Div(
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderRadius": "8px", "padding": "16px", "flex": "1",
        },
        children=[
            html.Div("RISK LEVEL", style={
                "fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px",
                "color": MC["accent"], "marginBottom": "10px",
                "borderBottom": f"1px solid {MC['border']}", "paddingBottom": "6px",
            }),
            html.Div(style={"textAlign": "center", "marginBottom": "12px"}, children=[
                html.Div(risk_rank, style={"fontSize": "32px", "fontWeight": 800, "color": risk_color}),
                html.Div(f"Risk score: {risk_points} pts", style={"fontSize": "11px", "color": MC["text_muted"]}),
            ]),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "columnGap": "12px", "rowGap": "4px", "marginBottom": "8px"}, children=[
                html.Div(f"VIX: {vix_txt}", style={"fontSize": "12px", "color": MC["text_sec"]}),
                html.Div(f"Implied Move: {em_txt}", style={"fontSize": "12px", "color": MC["text_sec"], "textAlign": "right"}),
                html.Div(f"Gamma Regime: {'NEGATIVE' if net_gamma < 0 else 'POSITIVE'}", style={"fontSize": "12px", "color": MC["warning"] if net_gamma < 0 else MC["call"]}),
                html.Div(f"P/C: {pc_ratio:.2f}  |  IV Skew: {iv_skew:+.3f}", style={"fontSize": "12px", "color": MC["text_sec"], "textAlign": "right"}),
            ]),
            html.Table(
                style={"width": "100%", "borderCollapse": "collapse"},
                children=[
                    html.Thead(html.Tr(children=[
                        html.Th("Rank", style={"padding": "4px 8px", "fontSize": "11px", "color": MC["text_muted"], "textAlign": "left", "borderBottom": f"1px solid {MC['border']}"}),
                        html.Th("Points", style={"padding": "4px 8px", "fontSize": "11px", "color": MC["text_muted"], "textAlign": "center", "borderBottom": f"1px solid {MC['border']}"}),
                        html.Th("Interpretation", style={"padding": "4px 8px", "fontSize": "11px", "color": MC["text_muted"], "textAlign": "left", "borderBottom": f"1px solid {MC['border']}"}),
                    ])),
                    html.Tbody(table_rows),
                ]
            ),
            html.Div(style={
                "borderTop": f"1px solid {MC['border']}",
                "marginTop": "10px", "paddingTop": "8px",
            }, children=[
                html.Div("RISK DRIVERS", style={"fontSize": "10px", "fontWeight": 700, "color": MC["text_muted"], "letterSpacing": "0.5px"}),
                html.Div(f"Signals today: {non_supp_count}", style={"fontSize": "12px", "color": MC["text_sec"], "marginTop": "3px"}),
                html.Div(
                    "Uses VIX regime + implied move (0.85 x ATM straddle / spot) + gamma/flow/data stress.",
                    style={"fontSize": "12px", "color": MC["text_muted"], "marginTop": "2px"},
                ),
            ]),
        ]
    )


# ---------------------------------------------------------------------------
# Model Health Panel
# ---------------------------------------------------------------------------

def _create_model_health_panel(model_out, pred_history_roll):
    """2x3 grid of mini-cards showing model health metrics."""
    if not model_out:
        return None

    diagnostics = model_out.get("diagnostics", {}) or {}
    quality = float(diagnostics.get("quality_score", 0.0) or 0.0)
    latency = diagnostics.get("latency_ms", None)
    completeness = float(diagnostics.get("feature_completeness", 0.0) or 0.0)
    missing_stage1 = int(diagnostics.get("stage1_missing_count", 0) or 0)

    hist = pred_history_roll

    # Signal Persistence: fraction of consecutive non-suppressed predictions
    # that stayed on the same side as the current prediction.
    # Note: this is NOT real accuracy (no price data available in dashboard).
    # It measures directional consistency of the model output over recent history.
    non_supp = [h for h in hist if not h["suppressed"]]
    model_thr = float(model_out.get("threshold", 0.36) or 0.36)
    current_dir_up = non_supp[-1]["prob"] >= model_thr if non_supp else True
    consistent = sum(1 for h in non_supp if (h["prob"] >= model_thr) == current_dir_up)
    total = len(non_supp)
    persistence = consistent / total if total > 0 else 0.0
    accuracy_text = f"{persistence:.0%}" if total > 0 else "N/A"
    accuracy_sub = f"({consistent}/{total} bars same side)"

    # Sparkline data from history
    quality_hist = [h.get("confidence", 0.0) for h in hist[-30:] if not h["suppressed"]]

    def _mini_sparkline(values, color, height=35):
        """Create a tiny Plotly sparkline figure."""
        if not values or len(values) < 2:
            return html.Div("--", style={"color": MC["text_muted"], "fontSize": "11px"})
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=values, mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.09),
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": f"{height}px", "width": "100%"})

    def _health_card(title, value, subtitle, sparkline_values=None, spark_color=None, value_color=None):
        children = [
            html.Div(title, style={"fontSize": "10px", "fontWeight": 700, "color": MC["text_muted"], "letterSpacing": "0.5px", "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": 800, "color": value_color or MC["text"], "marginTop": "3px"}),
            html.Div(subtitle, style={"fontSize": "11px", "color": MC["text_muted"], "marginTop": "2px"}),
        ]
        if sparkline_values and spark_color:
            children.append(html.Div(style={"marginTop": "4px"}, children=[
                _mini_sparkline(sparkline_values, spark_color)
            ]))
        return html.Div(
            style={
                "backgroundColor": MC["bg_dark"],
                "border": f"1px solid {MC['border']}",
                "borderRadius": "6px", "padding": "10px",
            },
            children=children,
        )

    acc_color = MC["call"] if persistence >= 0.55 else (MC["warning"] if persistence >= 0.45 else MC["put"])
    qual_color = MC["call"] if quality >= 0.6 else (MC["warning"] if quality >= 0.4 else MC["put"])
    comp_color = MC["call"] if completeness >= 0.8 else (MC["warning"] if completeness >= 0.5 else MC["put"])
    lat_color = MC["call"] if (latency is not None and latency < 500) else MC["warning"]

    cards = [
        _health_card("Signal Persistence", accuracy_text, accuracy_sub,
                      sparkline_values=quality_hist, spark_color=acc_color, value_color=acc_color),
        _health_card("Quality Score", f"{quality:.2f}", "Current inference",
                      sparkline_values=quality_hist, spark_color=qual_color, value_color=qual_color),
        _health_card("Completeness", f"{completeness:.0%}", "Feature coverage",
                      value_color=comp_color),
        _health_card("Latency", f"{latency:.0f}ms" if latency is not None else "N/A",
                      "Inference time", value_color=lat_color),
        _health_card("Stage1 Missing", str(missing_stage1), "Missing features",
                      value_color=MC["call"] if missing_stage1 == 0 else MC["warning"]),
        _health_card("History Depth", str(len(hist)), f"{len(non_supp)} non-suppressed",
                      value_color=MC["text"]),
    ]

    return html.Div(
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderRadius": "8px", "padding": "14px",
        },
        children=[
            html.Div("MODEL HEALTH", style={
                "fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px",
                "color": MC["accent"], "marginBottom": "10px",
            }),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "10px",
                },
                children=cards,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Enhanced Signal Meters
# ---------------------------------------------------------------------------

def _create_signal_meters(model_out):
    if not model_out:
        return None
    stage2_probs = dict(model_out.get("stage2_probs", {}) or {})
    stage3_prob = float(model_out.get("prob", 0.5) or 0.5)
    stage3_thr = float(model_out.get("threshold", 0.36) or 0.36)
    confidence = float(model_out.get("confidence", 0.0) or 0.0)  # No fake fallback
    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    source_state = str(model_out.get("source_state", "UNKNOWN") or "UNKNOWN")
    neutral_mode = suppressed or (not ok)
    bar_color = MC["text_muted"] if neutral_mode else (MC["call"] if stage3_prob >= stage3_thr else MC["put"])
    agents = [("S2-A", "A"), ("S2-B", "B"), ("S2-C", "C"), ("S2-K", "K"), ("S2-T", "T"), ("S2-Q", "Q"), ("S2-2D", "2D")]

    fig = make_subplots(
        rows=3,
        cols=4,
        specs=[
            [{"type": "indicator", "colspan": 4}, None, None, None],
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        ],
        subplot_titles=["", *[a[0] for a in agents], ""],
        horizontal_spacing=0.06,
        vertical_spacing=0.2,
    )

    s3_arrow = "^" if stage3_prob >= stage3_thr else "v"
    s3_delta_text = f"{stage3_prob - stage3_thr:+.2f}"

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=stage3_prob,
            number={"valueformat": ".2f", "font": {"size": 54}},
            delta={"reference": stage3_thr, "valueformat": ".2f", "increasing": {"color": MC["call"]}, "decreasing": {"color": MC["put"]}},
            title={"text": f"Stage 3 {s3_arrow} ({source_state})  |  Confidence {confidence*100:.0f}%  |  dThr: {s3_delta_text}", "font": {"size": 16, "color": MC["text"]}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": MC["text_muted"]},
                "bar": {"color": bar_color, "thickness": 0.42},
                "bgcolor": MC["bg_card"],
                "borderwidth": 1,
                "bordercolor": MC["border"],
                "steps": [
                    {"range": [0, 0.4], "color": ("rgba(148,163,184,0.12)" if neutral_mode else "rgba(239,68,68,0.18)")},
                    {"range": [0.4, 0.6], "color": "rgba(148,163,184,0.16)"},
                    {"range": [0.6, 1.0], "color": ("rgba(148,163,184,0.12)" if neutral_mode else "rgba(16,185,129,0.18)")},
                ],
                "threshold": {"line": {"color": MC["accent"], "width": 3}, "thickness": 0.8, "value": stage3_thr},
            },
        ),
        row=1,
        col=1,
    )

    for idx, (label, key) in enumerate(agents):
        val = float(stage2_probs.get(key, AGENT_TRAIN_MEDIAN.get(key, 0.5)))
        row = 2 if idx < 4 else 3
        col = (idx % 4) + 1
        neutral = AGENT_TRAIN_MEDIAN.get(key, 0.5)
        is_up = val >= neutral
        arrow = "^" if is_up else "v"
        delta_from_median = val - neutral
        delta_text = f"{delta_from_median:+.2f}"

        if neutral_mode:
            s2_bar = MC["text_muted"]
            title_color = MC["text_muted"]
        else:
            s2_bar = MC["call"] if is_up else MC["put"]
            title_color = MC["call"] if is_up else MC["put"]

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                number={"valueformat": ".2f", "font": {"size": 24}},
                title={"text": f"{label} {arrow}<br><span style='font-size:10px;color:{title_color}'>{delta_text}</span>", "font": {"size": 12}},
                gauge={
                    "shape": "angular",
                    "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": MC["text_muted"]},
                    "bar": {"color": s2_bar, "thickness": 0.3},
                    "bgcolor": MC["bg_card"],
                    "borderwidth": 1,
                    "bordercolor": MC["border"],
                    "steps": [
                        {"range": [0, neutral - 0.05], "color": ("rgba(148,163,184,0.10)" if neutral_mode else "rgba(239,68,68,0.16)")},
                        {"range": [neutral - 0.05, neutral + 0.05], "color": "rgba(148,163,184,0.12)"},
                        {"range": [neutral + 0.05, 1.0], "color": ("rgba(148,163,184,0.10)" if neutral_mode else "rgba(16,185,129,0.16)")},
                    ],
                    "threshold": {"line": {"color": MC["accent"], "width": 2}, "thickness": 0.7, "value": neutral},
                },
            ),
            row=row,
            col=col,
        )

    layout_cfg = base_layout(title="Model Signal Meters", height=320)
    layout_cfg["margin"] = dict(l=30, r=30, t=70, b=30)
    fig.update_layout(**layout_cfg)

    return fig


# ---------------------------------------------------------------------------
# Sparkline helper for Stage 3 prob trend
# ---------------------------------------------------------------------------
def _build_sparkline_fig(history_roll, threshold):
    """Return a minimal Plotly figure showing Stage 3 prob trend (last 40 bars)."""
    rows = history_roll[-40:] if len(history_roll) > 40 else history_roll
    xs = list(range(len(rows)))
    ys = [float(r.get("prob", 0.5) or 0.5) for r in rows]

    colors = [MC["call"] if y >= threshold else MC["put"] for y in ys]

    fig = go.Figure()
    for i in range(1, len(xs)):
        fig.add_trace(go.Scatter(
            x=[xs[i-1], xs[i]], y=[ys[i-1], ys[i]],
            mode="lines",
            line={"color": colors[i], "width": 2},
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.add_hline(
        y=threshold,
        line={"color": MC["text_muted"], "width": 1, "dash": "dot"},
    )

    latest = ys[-1] if ys else threshold
    trend = ys[-1] - ys[0] if len(ys) > 1 else 0
    trend_sym = "^" if trend >= 0 else "v"
    trend_color = MC["call"] if trend >= 0 else MC["put"]
    fig.add_annotation(
        x=1.0, y=latest,
        xref="paper", yref="y",
        text=f"<b>{trend_sym} {latest:.3f}</b>",
        showarrow=False,
        font={"size": 10, "color": trend_color},
        xanchor="right",
        bgcolor=MC["bg_card"],
    )

    fig.update_layout(
        margin={"l": 0, "r": 60, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "visible": False,
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "visible": False,
            "showgrid": False,
            "zeroline": False,
            "range": [max(0, min(ys) - 0.05), min(1, max(ys) + 0.05)] if ys else [0, 1],
        },
        height=70,
    )
    return fig


# ---------------------------------------------------------------------------
# Stage 3 contribution helper
# ---------------------------------------------------------------------------
def _s3_top_contributors(stage2_probs, n=3):
    """Return top-n (label, logit_contribution) sorted by abs contribution.

    Each agent/cross-feature contribution = coef * (value - 0.5), so that
    being exactly at neutral (0.5) contributes 0, above 0.5 is positive, below is negative.
    """
    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]
    agent_coefs = [AGENT_S3_COEF[k] for k in agent_keys]
    probs = [float(stage2_probs.get(k, 0.5)) for k in agent_keys]

    mean_p   = sum(probs) / len(probs)
    std_p    = (sum((p - mean_p) ** 2 for p in probs) / len(probs)) ** 0.5
    spread   = max(probs) - min(probs)
    agree_up = sum(1 for p in probs if p > 0.5) / len(probs)
    max_p    = max(probs)
    min_p    = min(probs)

    cross_vals   = [mean_p, std_p, spread, agree_up, max_p, min_p]
    cross_labels = ["mean", "std", "spread", "agree_up", "max_p", "min_p"]

    contribs = []
    for k, coef, val in zip(agent_keys, agent_coefs, probs):
        contribs.append((k, coef * (val - 0.5)))
    for lbl, coef, val in zip(cross_labels, S3_CROSS_COEF, cross_vals):
        contribs.append((lbl, coef * (val - 0.5)))

    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return contribs[:n]


# ---------------------------------------------------------------------------
# Model Production Panel (card + agent bars)
# ---------------------------------------------------------------------------
def _create_model_production_panel(model_out, symbol, agg_df, pred_history_roll=None):
    """Card-style production block replacing gauge meters."""
    if not model_out:
        return None

    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    prob = float(model_out.get("prob", 0.5) or 0.5)
    _thr = float(model_out.get("threshold", 0.36) or 0.36)
    confidence = float(model_out.get("confidence", 0.0) or 0.0)
    strength = float(model_out.get("signal_strength", 0.0) or 0.0)
    direction = "BULL" if prob >= _thr else "BEAR"
    dir_color = MC["call"] if prob >= _thr else MC["put"]
    stage2_probs = dict(model_out.get("stage2_probs", {}) or {})
    diagnostics = model_out.get("diagnostics", {}) or {}
    quality = float(diagnostics.get("quality_score", 0.0) or 0.0)

    # Top contributors for "Why this signal?" row
    _top_contribs = _s3_top_contributors(stage2_probs) if (not suppressed and ok) else []

    # Detect signal flip vs previous state
    prev_direction = _last_alert_state.get("direction")
    flipped = (
        prev_direction is not None
        and prev_direction != ("BULL" if prob >= _thr else "BEAR")
        and not suppressed and ok
    )
    flip_class = ("signal-flip-bull" if direction == "BULL" else "signal-flip-bear") if flipped else ""

    # Compute flip persistence (how long since the flip occurred)
    flip_ts = _last_alert_state.get("flip_ts")
    flip_to = _last_alert_state.get("flip_to")
    flip_elapsed_str = None
    if flip_ts is not None and flip_to == direction and not suppressed and ok:
        elapsed_sec = int((_now_et_naive() - flip_ts).total_seconds())
        if elapsed_sec >= 0:
            mm, ss = divmod(elapsed_sec, 60)
            flip_elapsed_str = f"{mm}m {ss:02d}s" if mm > 0 else f"{ss}s"

    # Spot price from model output, fallback to latest agg.
    spot_val = float(model_out.get("spot_price", 0.0) or 0.0)
    if spot_val <= 0 and agg_df is not None and not agg_df.empty and symbol != "ALL" and "symbol" in agg_df.columns:
        sym_agg = agg_df[agg_df["symbol"] == symbol]
        if not sym_agg.empty and "spot" in sym_agg.columns:
            spot_val = float(sym_agg.iloc[-1].get("spot", 0.0) or 0.0)
    spot_text = f"${spot_val:,.0f}" if spot_val > 0 else "--"

    if suppressed or not ok:
        direction_text = "SUPPRESSED"
        direction_color = MC["warning"]
        reason = str(model_out.get("reason", "") or "waiting for live prediction inputs")
        arrow = "\u25c6"
    else:
        arrow = "\u25b2" if prob >= _thr else "\u25bc"
        direction_text = f"{arrow} {'BULL' if prob >= _thr else 'BEAR'}"
        direction_color = dir_color
        reason = "live signal active"

    # Confidence label
    if confidence >= 0.80:
        conf_label, conf_color = "High conf", MC["call"]
    elif confidence >= 0.60:
        conf_label, conf_color = "Med conf", MC["warning"]
    else:
        conf_label, conf_color = "Low conf", MC["text_muted"]

    # Flip badge
    show_flip_badge = flipped or (flip_elapsed_str is not None)
    if show_flip_badge:
        badge_label = f"FLIP \u2192 {direction}"
        if flip_elapsed_str:
            badge_label += f"  {flip_elapsed_str}"
        flip_badge = html.Div(
            style={
                "backgroundColor": f"{dir_color}22",
                "border": f"1px solid {dir_color}66",
                "borderRadius": "4px", "padding": "2px 8px",
            },
            children=html.Span(badge_label, style={
                "fontSize": "11px", "fontWeight": 800, "color": dir_color,
                "letterSpacing": "1px", "textTransform": "uppercase",
            }),
        )
    else:
        flip_badge = None

    # ── Nested _signal_bar helper (centered bar, same as _mc_rule_signal) ──
    def _signal_bar(score):
        """Mini horizontal bar centered at 0; right half = bull (green), left half = bear (red)."""
        bull_w = f"{max(0.0, score) * 50:.1f}%"
        bear_w = f"{max(0.0, -score) * 50:.1f}%"
        return html.Div(style={
            "display": "flex", "alignItems": "center", "height": "8px",
            "width": "100%", "gap": "0",
        }, children=[
            # left half (bear side)
            html.Div(style={"flex": "1", "height": "8px", "display": "flex",
                            "justifyContent": "flex-end",
                            "backgroundColor": "rgba(239,68,68,0.12)",
                            "borderRadius": "3px 0 0 3px", "overflow": "hidden"}, children=[
                html.Div(style={"width": bear_w, "height": "100%",
                                "backgroundColor": MC["put"],
                                "borderRadius": "3px 0 0 3px"}),
            ]),
            # center tick
            html.Div(style={"width": "1px", "height": "12px",
                            "backgroundColor": "rgba(148,163,184,0.4)"}),
            # right half (bull side)
            html.Div(style={"flex": "1", "height": "8px", "display": "flex",
                            "justifyContent": "flex-start",
                            "backgroundColor": "rgba(16,185,129,0.12)",
                            "borderRadius": "0 3px 3px 0", "overflow": "hidden"}, children=[
                html.Div(style={"width": bull_w, "height": "100%",
                                "backgroundColor": MC["call"],
                                "borderRadius": "0 3px 3px 0"}),
            ]),
        ])

    # ── Stage 3 centered bar — centered at decision threshold (_thr), not S3_NEUTRAL ──
    # _thr (0.36) is the F1-optimised decision boundary: prob >= _thr → BULL pred=1.
    # Centering here means the bar is green whenever the model actually predicts BULL.
    if prob >= _thr:
        s3_score = (prob - _thr) / (1.0 - _thr)
    else:
        s3_score = -(_thr - prob) / _thr
    s3_score = max(-1.0, min(1.0, s3_score))

    s3_bar_section = [
        html.Div(style={"marginBottom": "2px"}, children=[
            html.Span(f"Stage 3: {prob:.3f}", style={
                "fontSize": "12px", "fontWeight": 700,
                "color": MC["call"] if s3_score > 0 else (MC["put"] if s3_score < 0 else MC["text_muted"]),
            }),
            html.Span(f"  (threshold {_thr})", style={
                "fontSize": "10px", "color": MC["text_muted"],
            }),
        ]),
        _signal_bar(s3_score),
    ]

    # ── Agent breakdown rows ──
    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]
    agent_rows = []
    for k in agent_keys:
        v = float(stage2_probs.get(k, AGENT_TRAIN_MEDIAN.get(k, 0.5)))
        neutral = AGENT_TRAIN_MEDIAN.get(k, 0.5)
        if v >= neutral:
            bull_frac = min(1.0, (v - neutral) / (1.0 - neutral)) if neutral < 1.0 else 0.0
            bear_frac = 0.0
        else:
            bull_frac = 0.0
            bear_frac = min(1.0, (neutral - v) / neutral) if neutral > 0.0 else 0.0
        agent_score = bull_frac - bear_frac
        val_color = MC["text_muted"] if (suppressed or not ok) else (
            MC["call"] if v >= neutral else MC["put"]
        )
        agent_rows.append(
            html.Div(style={
                "display": "grid", "gridTemplateColumns": "140px 1fr 44px",
                "alignItems": "center", "gap": "8px", "marginBottom": "5px",
            }, children=[
                html.Span(f"Agent {k}", style={
                    "fontSize": "11px", "color": MC["text_muted"],
                    "fontWeight": 600, "letterSpacing": "0.3px",
                }),
                _signal_bar(agent_score if not (suppressed or not ok) else 0.0),
                html.Span(f"{v:.3f}", style={
                    "fontSize": "11px", "fontWeight": 700, "textAlign": "right",
                    "color": val_color,
                }),
            ])
        )

    # ── Header right side children ──
    header_right_children = [
        html.Span(direction_text, style={
            "fontSize": "20px", "fontWeight": 800, "color": direction_color,
            "letterSpacing": "1px",
        }),
        html.Div(style={
            "backgroundColor": f"{conf_color}22",
            "border": f"1px solid {conf_color}66",
            "borderRadius": "4px", "padding": "2px 8px",
        }, children=html.Span(f"{confidence:.0%} {conf_label}", style={
            "fontSize": "11px", "fontWeight": 700, "color": conf_color,
        })),
    ]
    if flip_badge:
        header_right_children.append(flip_badge)

    return html.Div(
        className=flip_class,
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderLeft": f"4px solid {dir_color}",
            "borderRadius": "10px", "padding": "14px 18px", "marginBottom": "14px",
        },
        children=[
            # Header row
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "alignItems": "center", "marginBottom": "10px"}, children=[
                html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "8px"}, children=[
                    html.Span("MODEL PREDICTION", style={
                        "fontSize": "12px", "fontWeight": 700, "letterSpacing": "1.2px",
                        "color": MC["accent"], "textTransform": "uppercase",
                    }),
                    html.Span("Hybrid51 Ensemble", style={
                        "fontSize": "10px", "color": MC["text_muted"],
                    }),
                ]),
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px"},
                         children=header_right_children),
            ]),
            # Stage 3 bar section
            *s3_bar_section,
            # Divider
            html.Hr(style={"borderColor": MC["border"], "margin": "10px 0"}),
            # Agent breakdown label
            html.Div("AGENT BREAKDOWN", style={
                "fontSize": "10px", "fontWeight": 700, "letterSpacing": "1px",
                "color": MC["text_muted"], "marginBottom": "7px",
            }),
            # Agent rows
            html.Div(agent_rows),
            # Second divider
            html.Hr(style={"borderColor": MC["border"], "margin": "10px 0"}),
            # WHY row (kept exactly as-is)
            *([
                html.Div(
                    style={
                        "marginBottom": "6px",
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "6px",
                        "flexWrap": "wrap",
                    },
                    children=[
                        html.Span("WHY:", style={
                            "fontSize": "10px", "fontWeight": 700, "letterSpacing": "0.8px",
                            "color": MC["text_muted"],
                        }),
                        *[
                            html.Span(
                                f"{lbl} {contrib:+.2f}",
                                style={
                                    "fontSize": "11px", "fontWeight": 600,
                                    "color": MC["call"] if contrib > 0 else MC["put"],
                                    "backgroundColor": _hex_to_rgba(MC["call"] if contrib > 0 else MC["put"], 0.12),
                                    "padding": "1px 7px", "borderRadius": "4px",
                                },
                            )
                            for lbl, contrib in _top_contribs
                        ],
                        html.Span(
                            f"(K weight={AGENT_S3_COEF['K']:.2f} dominates, agree_up coef={S3_CROSS_COEF[3]:.2f})",
                            style={"fontSize": "10px", "color": MC["text_muted"], "fontStyle": "italic"},
                        ),
                    ],
                ),
            ] if _top_contribs else []),
            # Additional metrics row
            html.Div(
                f"Conf: {confidence:.2f}  |  Strength: {strength:+.2f}  |  Quality: {quality:.2f}"
                f"  |  Spot: {spot_text}  |  Status: {reason}",
                style={"fontSize": "11px", "color": MC["text_muted"], "marginBottom": "6px"},
            ),
            # Sparkline (kept exactly as-is)
            *([
                html.Div("STAGE 3 PROB TREND (last 40 bars)", style={
                    "fontSize": "10px", "fontWeight": 700, "letterSpacing": "0.8px",
                    "color": MC["text_muted"], "marginTop": "4px", "marginBottom": "2px",
                }),
                dcc.Graph(
                    figure=_build_sparkline_fig(pred_history_roll, _thr),
                    config={"displayModeBar": False, "staticPlot": True},
                    style={"height": "70px", "width": "100%"},
                ),
            ] if pred_history_roll is not None and len(pred_history_roll) > 1 else []),
        ],
    )


# ---------------------------------------------------------------------------
# Agent Agreement Bar
# ---------------------------------------------------------------------------

def _create_agent_agreement_bar(model_out):
    """Horizontal stacked bar showing UP vs DOWN agents."""
    if not model_out:
        return None

    stage2_probs = dict(model_out.get("stage2_probs", {}) or {})
    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    neutral_mode = suppressed or (not ok)

    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]
    up_agents = []
    down_agents = []
    for k in agent_keys:
        val = float(stage2_probs.get(k, AGENT_TRAIN_MEDIAN.get(k, 0.5)))
        neutral = AGENT_TRAIN_MEDIAN.get(k, 0.5)
        confidence_from_median = abs(val - neutral)
        if val >= neutral:
            up_agents.append((k, confidence_from_median))
        else:
            down_agents.append((k, confidence_from_median))

    up_total = sum(c for _, c in up_agents) if up_agents else 0
    down_total = sum(c for _, c in down_agents) if down_agents else 0
    total = up_total + down_total
    if total == 0:
        total = 1

    up_pct = (up_total / total) * 100
    down_pct = (down_total / total) * 100

    up_color = MC["text_muted"] if neutral_mode else MC["call"]
    down_color = MC["text_muted"] if neutral_mode else MC["put"]

    up_names = ", ".join(f"{k}({c:.2f})" for k, c in up_agents) if up_agents else "none"
    down_names = ", ".join(f"{k}({c:.2f})" for k, c in down_agents) if down_agents else "none"

    return html.Div(
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderRadius": "6px", "padding": "10px 14px", "marginTop": "8px",
        },
        children=[
            html.Div("AGENT AGREEMENT", style={
                "fontSize": "10px", "fontWeight": 700, "letterSpacing": "0.5px",
                "color": MC["text_muted"], "marginBottom": "6px",
            }),
            html.Div(style={"display": "flex", "height": "18px", "borderRadius": "4px", "overflow": "hidden"}, children=[
                html.Div(style={
                    "width": f"{up_pct}%", "backgroundColor": up_color,
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontSize": "10px", "fontWeight": 700, "color": "white",
                }, children=[f"^ {len(up_agents)}" if up_pct > 15 else ""]),
                html.Div(style={
                    "width": f"{down_pct}%", "backgroundColor": down_color,
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontSize": "10px", "fontWeight": 700, "color": "white",
                }, children=[f"v {len(down_agents)}" if down_pct > 15 else ""]),
            ]),
            html.Div(style={"display": "flex", "justifyContent": "space-between", "marginTop": "4px"}, children=[
                html.Span(f"UP: {up_names}", style={"fontSize": "10px", "color": up_color}),
                html.Span(f"DOWN: {down_names}", style={"fontSize": "10px", "color": down_color}),
            ]),
        ]
    )


# ---------------------------------------------------------------------------
# Price Expected Move Chart — Candlestick + Forward Range
# ---------------------------------------------------------------------------
#
# Left side: OHLC candlesticks derived from spot prices (5-min buckets)
# Right side: Expected move cone (ATM straddle + IV-scaled range)
# Model prediction: directional bias arrow with intensity = confidence
# ---------------------------------------------------------------------------

def _derive_ohlc_candles(sdf, freq_minutes=5):
    """Derive OHLC candles from spot price time series.

    Groups spot prices into `freq_minutes` buckets and computes OHLC.
    Returns DataFrame with columns: ts_bucket, open, high, low, close.
    """
    if sdf.empty or "spot" not in sdf.columns or "_ts_parsed" not in sdf.columns:
        return pd.DataFrame()
    df = sdf[["_ts_parsed", "spot"]].copy()
    df["spot"] = pd.to_numeric(df["spot"], errors="coerce")
    df = df.dropna()
    if df.empty:
        return pd.DataFrame()
    df = df.set_index("_ts_parsed").sort_index()
    ohlc = df["spot"].resample(f"{freq_minutes}min").ohlc()
    ohlc = ohlc.dropna(how="all")
    if ohlc.empty:
        return pd.DataFrame()
    ohlc = ohlc.reset_index()
    ohlc.columns = ["ts_bucket", "open", "high", "low", "close"]
    return ohlc


def _create_expected_move_chart(df_agg, symbol, model_out, pred_history_roll):
    """Price + Expected Move chart: candlesticks (left) + widening cone (right)."""
    if df_agg is None or df_agg.empty or symbol == "ALL" or "symbol" not in df_agg.columns:
        return None
    sdf = df_agg[df_agg["symbol"] == symbol].copy()
    if sdf.empty or "spot" not in sdf.columns or "_ts_parsed" not in sdf.columns:
        return None
    sdf = sdf[sdf["_ts_parsed"].notna()].copy()
    sdf["spot"] = pd.to_numeric(sdf["spot"], errors="coerce")
    sdf = sdf[sdf["spot"].notna()].copy()
    if len(sdf) < 3:
        return None
    sdf = sdf.sort_values("_ts_parsed")

    ohlc = _derive_ohlc_candles(sdf, freq_minutes=5)
    if ohlc.empty:
        return None

    last_row     = sdf.iloc[-1]
    spot         = float(last_row["spot"])
    now_ts       = sdf["_ts_parsed"].iloc[-1]
    atm_straddle = float(pd.to_numeric(last_row.get("atm_straddle", np.nan), errors="coerce"))
    call_iv      = float(pd.to_numeric(last_row.get("call_iv",      np.nan), errors="coerce"))
    put_iv       = float(pd.to_numeric(last_row.get("put_iv",       np.nan), errors="coerce"))
    pc_ratio     = float(pd.to_numeric(last_row.get("pc_ratio",     np.nan), errors="coerce"))

    have_straddle = np.isfinite(atm_straddle) and atm_straddle > 0
    have_iv       = np.isfinite(call_iv) and np.isfinite(put_iv) and call_iv > 0

    if have_straddle:
        base_em = atm_straddle
    elif have_iv:
        avg_iv  = (call_iv + put_iv) / 2.0
        base_em = spot * avg_iv * np.sqrt(1.0 / 252.0)
    else:
        tail    = sdf.tail(min(60, len(sdf)))
        base_em = max(float(tail["spot"].max() - tail["spot"].min()) * 2.0, spot * 0.003)

    horizon_min = 90
    em_horizon  = base_em * np.sqrt(horizon_min / 390.0)

    suppressed     = bool((model_out or {}).get("suppressed", False))
    ok             = bool((model_out or {}).get("ok", True))
    has_prediction = model_out is not None and not suppressed and ok
    pup            = float((model_out or {}).get("prob",       0.5) or 0.5) if has_prediction else 0.5
    confidence     = float((model_out or {}).get("confidence", 0.0) or 0.0) if has_prediction else 0.0
    _thr           = float((model_out or {}).get("threshold",  0.36) or 0.36) if has_prediction else 0.36
    stronger_up    = pup >= _thr

    bias_factor = (0.5 + 0.22 * confidence * (1.0 if stronger_up else -1.0)) if has_prediction else 0.5
    up_share    = bias_factor
    dn_share    = 1.0 - bias_factor

    future_x = [now_ts + pd.Timedelta(minutes=i) for i in range(horizon_min + 1)]
    t_frac   = np.linspace(0, 1, horizon_min + 1)
    em_env   = em_horizon * np.sqrt(t_frac)
    up_path  = spot + em_env * up_share * 2
    dn_path  = spot - em_env * dn_share * 2
    mid_path = spot + em_env * (up_share - dn_share)

    # Colors
    bull_c = "rgba(16,185,129,1.0)"
    bear_c = "rgba(239,68,68,1.0)"
    neut_c = "rgba(148,163,184,1.0)"
    if has_prediction and stronger_up:
        cone_color = bull_c
        cone_fill  = f"rgba(16,185,129,{0.07 + confidence * 0.10:.2f})"
    elif has_prediction:
        cone_color = bear_c
        cone_fill  = f"rgba(239,68,68,{0.07 + confidence * 0.10:.2f})"
    else:
        cone_color = neut_c
        cone_fill  = "rgba(148,163,184,0.08)"

    mid_alpha = 0.55 + confidence * 0.35
    mid_color = (f"rgba(16,185,129,{mid_alpha:.2f})" if stronger_up
                 else f"rgba(239,68,68,{mid_alpha:.2f})")

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]])

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=ohlc["ts_bucket"],
        open=ohlc["open"], high=ohlc["high"], low=ohlc["low"], close=ohlc["close"],
        name="Price",
        increasing_line_color="#10b981", decreasing_line_color="#ef4444",
        increasing_fillcolor="#10b981",  decreasing_fillcolor="#ef4444",
        whiskerwidth=0.6, opacity=1.0,
        showlegend=False,
    ))

    # Cone fill (toself polygon)
    fig.add_trace(go.Scatter(
        x=list(future_x) + list(future_x)[::-1],
        y=list(up_path)  + list(dn_path)[::-1],
        fill="toself", mode="none", name="EM Range",
        fillcolor=cone_fill, hoverinfo="skip",
        showlegend=False,
    ))

    # Upper boundary
    fig.add_trace(go.Scatter(
        x=future_x, y=list(up_path), mode="lines",
        name=f"Upper +{em_horizon:.1f}",
        line=dict(color="rgba(16,185,129,0.75)", width=1.5, dash="dot"),
        hovertemplate="Upper: %{y:.2f}<extra></extra>",
    ))
    # Lower boundary
    fig.add_trace(go.Scatter(
        x=future_x, y=list(dn_path), mode="lines",
        name=f"Lower −{em_horizon:.1f}",
        line=dict(color="rgba(239,68,68,0.75)", width=1.5, dash="dot"),
        hovertemplate="Lower: %{y:.2f}<extra></extra>",
    ))

    # Model midline (bias direction)
    if has_prediction and confidence > 0.2:
        fig.add_trace(go.Scatter(
            x=future_x, y=list(mid_path), mode="lines",
            name=f"{'BULL' if stronger_up else 'BEAR'} {confidence:.0%} bias",
            line=dict(color=mid_color, width=2.5),
            hovertemplate="Midline: %{y:.2f}<extra></extra>",
        ))

    y_low  = float(min(ohlc["low"].min(),  np.nanmin(dn_path)))
    y_high = float(max(ohlc["high"].max(), np.nanmax(up_path)))
    y_pad  = (y_high - y_low) * 0.05
    y_low -= y_pad;  y_high += y_pad * 1.5

    # NOW divider — solid vertical line
    fig.add_shape(type="line", x0=now_ts, x1=now_ts, y0=y_low, y1=y_high,
                  line=dict(color="rgba(255,255,255,0.45)", width=1.5))
    fig.add_annotation(
        x=now_ts, y=y_high, text="NOW", showarrow=False,
        xanchor="center", yanchor="top",
        font=dict(color="rgba(255,255,255,0.70)", size=11, family="monospace"),
        bgcolor="rgba(15,23,42,0.0)", borderpad=2,
    )

    # Spot reference line
    spot_label = f"  {spot:,.0f}"
    fig.add_hline(y=spot, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1.0,
                  annotation_text=spot_label, annotation_position="right",
                  annotation_font=dict(color=MC["text"], size=12, family="monospace"))

    # Bias annotation at cone tip
    if has_prediction and confidence > 0.2:
        bias_label  = f"{'▲' if stronger_up else '▼'} {'BULL' if stronger_up else 'BEAR'}  {confidence:.0%}"
        bias_color  = "#10b981" if stronger_up else "#ef4444"
        bias_y      = float(up_path[-1]) if stronger_up else float(dn_path[-1])
        fig.add_annotation(
            x=future_x[-1], y=bias_y, text=bias_label,
            showarrow=False, xanchor="right", yanchor="middle",
            font=dict(color=bias_color, size=13, family="monospace", weight=700),
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor=bias_color, borderwidth=1, borderpad=5,
        )

    # Tracking status above history candles
    tracking_status = None;  tracking_color = MC["text_muted"]
    if has_prediction and len(sdf) >= 5:
        recent_spots = sdf["spot"].tail(10)
        if len(recent_spots) >= 2:
            spot_delta = float(recent_spots.iloc[-1] - recent_spots.iloc[0])
            if (pup >= _thr) == (spot_delta > 0):
                tracking_status = "ON PATH";     tracking_color = "#10b981"
            elif abs(spot_delta) < em_horizon * 0.15:
                tracking_status = "NEUTRAL";     tracking_color = MC["warning"]
            elif abs(spot_delta) > em_horizon * 0.7:
                tracking_status = "INVALIDATED"; tracking_color = "#ef4444"
            else:
                tracking_status = "DIVERGING";   tracking_color = MC["warning"]
    if tracking_status:
        hist_mid_ts = ohlc["ts_bucket"].iloc[len(ohlc) // 2] if len(ohlc) > 0 else now_ts
        fig.add_annotation(
            x=hist_mid_ts, y=y_high - y_pad * 0.3,
            text=tracking_status, showarrow=False,
            xanchor="center", yanchor="top",
            font=dict(color=tracking_color, size=11, family="monospace"),
            bgcolor="rgba(15,23,42,0.82)",
            bordercolor=tracking_color, borderwidth=1, borderpad=3,
        )

    # Info strip (EM value, straddle, P/C)
    em_pct = em_horizon / spot * 100.0 if spot > 0 else 0.0
    info_parts = [f"90m EM ±{em_horizon:.1f} ({em_pct:.1f}%)"]
    if have_straddle:
        info_parts.append(f"Straddle ${atm_straddle:.2f}")
    if np.isfinite(pc_ratio) and pc_ratio > 0:
        info_parts.append(f"P/C {pc_ratio:.2f}")
    fig.add_annotation(
        x=0.01, y=0.98, xref="paper", yref="paper",
        text="  ".join(info_parts), showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(color=MC["text_muted"], size=10, family="monospace"),
        bgcolor="rgba(15,23,42,0.0)",
    )

    layout_cfg = base_layout(title=f"{symbol}  ·  Price & Expected Move", height=400)
    layout_cfg.update({
        "yaxis": dict(
            range=[y_low, y_high],
            showgrid=True, gridcolor="rgba(255,255,255,0.05)", gridwidth=1,
            zeroline=False, tickformat=",.0f",
        ),
        "xaxis": dict(
            rangeslider={"visible": False},
            showgrid=False,
        ),
        "hovermode": "x unified",
        "legend": dict(
            orientation="h", x=0.01, y=-0.06,
            font=dict(size=11, color=MC["text_muted"]),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        "margin": dict(l=60, r=90, t=44, b=36),
    })
    fig.update_layout(layout_cfg)
    return style_axes(fig)


def _create_accumulated_prediction_chart(pred_history_roll):
    """Running cumulative sum of (prob - training_median) for Stage 3 and all 7 agents.

    Each agent uses its own training-data median as neutral (not 0.5), so the
    cumulative reflects genuine deviation from that agent's learned baseline.
    Line width is scaled by Stage 3 LogReg coefficient importance.
    """
    if len(pred_history_roll) < 2:
        return None

    hist = pred_history_roll
    x    = [h["ts"] for h in hist]

    # Market-hours gate: rows outside 8:30–17:00 ET contribute 0 delta
    _open_min  = MARKET_OPEN_ET[0]  * 60 + MARKET_OPEN_ET[1]
    _close_min = MARKET_CLOSE_ET[0] * 60 + MARKET_CLOSE_ET[1]
    def _in_market(h):
        try:
            ts = pd.to_datetime(str(h["ts"])[:19], errors="coerce")
            if pd.isna(ts):
                return True
            ts_et = ts.tz_localize("UTC").tz_convert(ET) if ts.tzinfo is None else ts.tz_convert(ET)
            m = ts_et.hour * 60 + ts_et.minute
            return _open_min <= m <= _close_min
        except Exception:
            return True

    # Per-agent training medians and S3 coefficients defined at module level
    # (AGENT_TRAIN_MEDIAN, AGENT_S3_COEF)
    _coef_max = max(AGENT_S3_COEF.values())
    # Width: most important agent=3.0, least=0.8
    def _agent_width(label):
        return 0.8 + 2.2 * (AGENT_S3_COEF[label] / _coef_max)

    # Stage 3 cumulative — deviation from model baseline (~0.43).
    # 0.43 = prob when all agents neutral (model intercept=-3.25 suppresses outputs).
    # Above 0.43 = net bullish signal; below = bearish. Threshold 0.36 is F1-optimised decision boundary.
    s3_deltas = [
        (h["prob"] - S3_NEUTRAL)
        if (not h.get("suppressed", True) and _in_market(h)) else 0.0
        for h in hist
    ]
    s3_cum    = list(np.cumsum(s3_deltas))

    final_val  = s3_cum[-1] if s3_cum else 0.0
    is_bull    = final_val >= 0
    s3_color   = MC["call"] if is_bull else MC["put"]
    fill_color = "rgba(16,185,129,0.13)" if is_bull else "rgba(239,68,68,0.13)"

    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]

    AGENT_COLORS = {
        "A":  "rgba(139,92,246,0.85)",   # violet
        "B":  "rgba(251,146,60,0.85)",   # orange
        "C":  "rgba(34,211,238,0.85)",   # cyan
        "K":  "rgba(250,204,21,0.85)",   # yellow  — highest S3 weight
        "T":  "rgba(244,114,182,0.85)",  # pink    — lowest S3 weight
        "Q":  "rgba(99,179,237,0.85)",   # sky blue
        "2D": "rgba(167,243,208,0.85)",  # mint
    }

    fig = go.Figure()

    # Zero reference line
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(148,163,184,0.35)", line_width=1)

    # ── 1. Agent lines — width scaled by S3 coefficient, neutral = training median ──
    agent_endpoints = []
    for label in agent_keys:
        neutral = AGENT_TRAIN_MEDIAN.get(label, 0.5)
        a_deltas = [
            (h.get("stage2_probs", {}).get(label, neutral) - neutral)
            if (not h.get("suppressed", True) and _in_market(h)) else 0.0
            for h in hist
        ]
        a_cum = list(np.cumsum(a_deltas))
        end_val = a_cum[-1] if a_cum else 0.0
        agent_endpoints.append((label, end_val))
        fig.add_trace(go.Scatter(
            x=x, y=a_cum,
            mode="lines",
            name=f"Agent {label}",
            showlegend=False,
            line=dict(color=AGENT_COLORS[label], width=_agent_width(label)),
            hovertemplate=f"Agent {label}: %{{y:.3f}}<extra></extra>",
        ))

    # ── 2. Stage 3 area fill base ──
    fig.add_trace(go.Scatter(
        x=x, y=s3_cum,
        mode="none",
        fill="tozeroy",
        fillcolor=fill_color,
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── 3. Stage 3 thick line (on top) ──
    n_live = sum(1 for h in hist if not h.get("suppressed", True))
    fig.add_trace(go.Scatter(
        x=x, y=s3_cum,
        mode="lines",
        name="Stage 3 Ensemble",
        showlegend=False,
        line=dict(color=s3_color, width=4.5),
        hovertemplate="<b>Stage 3: %{y:.3f}</b><extra></extra>",
    ))

    # ── 4. Endpoint marker on Stage 3 ──
    if x and s3_cum:
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[s3_cum[-1]],
            mode="markers",
            marker=dict(color=s3_color, size=10, symbol="circle",
                        line=dict(color="#0f172a", width=2)),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Inline labels at the right end of each line (instead of a separate legend).
        for label, end_val in agent_endpoints:
            a_probs = [h.get("stage2_probs", {}).get(label, AGENT_TRAIN_MEDIAN[label])
                       for h in hist if not h.get("suppressed", True)]
            latest_p  = a_probs[-1] if a_probs else AGENT_TRAIN_MEDIAN[label]
            a_std     = float(np.std(a_probs)) if len(a_probs) > 2 else 0.0
            t_median  = AGENT_TRAIN_MEDIAN[label]
            drift     = latest_p - t_median
            stuck     = a_std < 0.02
            drift_str = f" ({drift:+.2f}vs trn)" if abs(drift) > 0.05 else ""
            warn      = " ⚠" if stuck else ""
            fig.add_annotation(
                x=x[-1], y=end_val,
                xref="x", yref="y",
                text=f"{label} {latest_p:.2f}{drift_str}{warn}",
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(size=10, color=AGENT_COLORS[label]),
                align="left",
                bgcolor="rgba(15,23,42,0.55)",
                bordercolor=C['warning'] if stuck else "rgba(148,163,184,0.18)",
                borderwidth=1,
                borderpad=2,
            )
        fig.add_annotation(
            x=x[-1], y=s3_cum[-1],
            xref="x", yref="y",
            text="Stage 3",
            showarrow=False,
            xanchor="left",
            xshift=10,
            font=dict(size=11, color=s3_color),
            align="left",
            bgcolor="rgba(15,23,42,0.75)",
            bordercolor="rgba(148,163,184,0.28)",
            borderwidth=1,
            borderpad=3,
        )

        # ── Agent divergence diagnostic box (top-left) ──
        diag_lines = []
        for label, end_val in agent_endpoints:
            a_probs_live = [h.get("stage2_probs", {}).get(label, AGENT_TRAIN_MEDIAN[label])
                            for h in hist if not h.get("suppressed", True)]
            if len(a_probs_live) > 2:
                a_std    = float(np.std(a_probs_live))
                a_mean   = float(np.mean(a_probs_live))
                t_median = AGENT_TRAIN_MEDIAN[label]
                drift    = a_mean - t_median
                flags = []
                if a_std < 0.02:
                    flags.append("LOW VAR")
                if abs(drift) > 0.08:
                    flags.append(f"DRIFT {drift:+.2f}")
                if flags:
                    flag_str = "  ⚠ " + " | ".join(flags)
                    diag_lines.append(
                        f"<span style='color:{AGENT_COLORS[label]}'><b>{label}</b></span>{flag_str}"
                    )
        if diag_lines:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.01, y=0.99,
                text="<b>Alerts</b><br>" + "<br>".join(diag_lines),
                showarrow=False, align="left",
                font=dict(size=9, color=MC["text"]),
                bgcolor="rgba(15,23,42,0.88)",
                bordercolor=MC["border"], borderwidth=1, borderpad=4,
                xanchor="left", yanchor="top",
                width=140,
            )

    # ── Reading guide annotation (right margin, vertical) ──
    fig.add_annotation(
        xref="paper", yref="paper", x=1.02, y=0.5,
        text=(
            "<b>How to read</b><br>"
            "─────────────<br>"
            "Zero = model baseline<br>"
            "(all agents neutral)<br>"
            "<br>"
            "Rising = above baseline<br>"
            "= net bullish signal<br>"
            "<br>"
            "Falling = below baseline<br>"
            "= net bearish signal<br>"
            "<br>"
            "<i>BULL/BEAR label uses<br>"
            "threshold=0.36<br>"
            "baseline≈0.43</i>"
        ),
        showarrow=False, align="left",
        font=dict(size=9, color=MC["text_muted"]),
        bgcolor="rgba(15,23,42,0.0)",
        bordercolor="rgba(0,0,0,0)",
        borderwidth=0, borderpad=4,
        xanchor="left", yanchor="middle",
    )

    # ── Layout ──
    layout_cfg = base_layout(
        title=f"Accumulated Directional Signal  ({n_live} live bars)  |  S3 zero = model baseline (≈0.43); agents zero = own baseline",
        height=340,
    )
    layout_cfg.update({
        "margin": dict(l=55, r=160, t=50, b=40),
        "yaxis": dict(
            title="Cumulative (prob − baseline)",
            zeroline=True,
            zerolinecolor="rgba(59,130,246,0.30)",
            zerolinewidth=1,
            gridcolor="rgba(51,65,85,0.5)",
        ),
        "xaxis": dict(
            showgrid=False,
        ),
        "showlegend": False,
        "hovermode": "x unified",
    })
    fig.update_layout(layout_cfg)
    return style_axes(fig)

def _create_model_rollover_chart(pred_history_roll):
    """Stage3 probability/confidence/strength over time from prediction.csv."""
    if len(pred_history_roll) < 2:
        return None
    hist = pred_history_roll
    x = [h["ts"] for h in hist]
    prob = [None if h["suppressed"] else h["prob"] for h in hist]
    conf = [None if h["suppressed"] else h["confidence"] for h in hist]
    strength = [None if h["suppressed"] else h["strength"] for h in hist]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=prob, mode="lines+markers", name="Stage3 P(up)", line=dict(color=MC["accent"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=conf, mode="lines", name="Confidence", line=dict(color=MC["warning"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=strength, mode="lines", name="Signal Strength", line=dict(color=MC["call"])), secondary_y=True)
    _thr_roll = next(
        (float(h.get("threshold", 0.36) or 0.36) for h in reversed(hist) if not h.get("suppressed", True)),
        0.36,
    )
    fig.add_hline(y=_thr_roll, line_dash="dot", line_color=MC["text_muted"], secondary_y=False,
                  annotation_text=f"thr={_thr_roll:.2f}", annotation_position="right",
                  annotation_font=dict(size=9, color=MC["text_muted"]))
    fig.add_hline(y=0.0, line_dash="dot", line_color=MC["text_muted"], secondary_y=True)
    fig.update_yaxes(range=[0, 1], title_text="Probability / Confidence", secondary_y=False)
    fig.update_yaxes(range=[-1, 1], title_text="Strength", secondary_y=True)
    fig.update_layout(**base_layout(title="Stage 3 Rollover Prediction", height=320))
    return style_axes(fig)


# ---------------------------------------------------------------------------
# Model Signal Card (kept for backward compat, simplified)
# ---------------------------------------------------------------------------

def _model_signal_card(model_out):
    if not model_out:
        return None

    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    prob = float(model_out.get("prob", 0.5) or 0.5)
    pred = int(model_out.get("pred", 0) or 0)
    reason = str(model_out.get("reason", "") or "")
    diagnostics = model_out.get("diagnostics", {}) or {}
    source_state = str(model_out.get("source_state", "UNKNOWN") or "UNKNOWN")
    quality = float(diagnostics.get("quality_score", 0.0) or 0.0)
    latency = diagnostics.get("latency_ms", None)
    warmup_fraction = float(diagnostics.get("warmup_fraction", 0.0) or 0.0)
    completeness = float(diagnostics.get("feature_completeness", 0.0) or 0.0)
    vix_valid = bool(diagnostics.get("vix_valid", False))
    missing_stage1 = int(diagnostics.get("stage1_missing_count", 0) or 0)

    last_live_text = "N/A"
    if _last_live_non_suppressed_ts is not None:
        elapsed = _now_et_naive() - _last_live_non_suppressed_ts
        secs = int(max(0, elapsed.total_seconds()))
        if secs < 60:
            last_live_text = f"{secs}s ago"
        elif secs < 3600:
            last_live_text = f"{secs // 60}m ago"
        else:
            last_live_text = f"{secs // 3600}h ago"

    text_parts = [f"Source: {source_state}"]
    if suppressed:
        label = f"SUPPRESSED: {reason}" if reason else "SUPPRESSED"
        color = MC["warning"]
    elif ok:
        direction = "BULL" if pred == 1 else "BEAR"
        color = MC["call"] if pred == 1 else MC["put"]
        label = f"{direction} {prob:.0%}"
    else:
        label = "Model unavailable"
        color = MC["text_muted"]

    text_parts.append(f"Last live: {last_live_text}")
    if latency is not None:
        text_parts.append(f"Latency: {latency:.0f}ms")
    text_parts.append(f"Quality: {quality:.2f} | Complete: {completeness:.0%} | Warmup: {warmup_fraction:.0%}")
    text_parts.append(f"VIX valid: {vix_valid} | Missing S1: {missing_stage1}")

    return html.Div(
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderRadius": "8px",
            "padding": "14px 18px",
            "marginBottom": "12px",
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                children=[
                    html.Span("MODEL", style={"fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px", "color": MC["text_muted"]}),
                    html.Span(label, style={"fontSize": "16px", "fontWeight": 700, "color": color}),
                ]
            ),
            html.Div(
                " | ".join(text_parts),
                style={"fontSize": "11px", "color": MC["text_muted"], "marginTop": "6px"},
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Data loading (unchanged from original, but paths use DATA_DIR)
# ---------------------------------------------------------------------------

def load_data(dte_filter="0_1dte"):
    """Thin wrapper around the original loaders + DTE filters with caching.
    Uses mtime + size + max-age to detect file changes reliably."""
    global _cached_agg_df, _cached_snap_df, _cached_filtered_data
    global _cached_agg_mtime, _cached_agg_size, _cached_agg_ts
    global _cached_snap_mtime, _cached_snap_size, _cached_snap_ts

    agg_changed, agg_mt, agg_sz = _file_changed(
        AGG_FILE, _cached_agg_mtime, _cached_agg_size, _cached_agg_ts
    )
    if agg_changed or _cached_agg_df is None:
        agg_df = load_agg_data()
        if agg_df is None:
            agg_df = pd.DataFrame()
        _cached_agg_df = agg_df
        _cached_agg_mtime = agg_mt
        _cached_agg_size = agg_sz
        _cached_agg_ts = time.time()
        _cached_filtered_data.clear()
    else:
        agg_df = _cached_agg_df

    snap_changed, snap_mt, snap_sz = _file_changed(
        SNAPSHOT_FILE, _cached_snap_mtime, _cached_snap_size, _cached_snap_ts
    )
    if snap_changed or _cached_snap_df is None:
        snap_df = load_snapshot_data()
        if snap_df is None:
            snap_df = pd.DataFrame()
        _cached_snap_df = snap_df
        _cached_snap_mtime = snap_mt
        _cached_snap_size = snap_sz
        _cached_snap_ts = time.time()
        _cached_filtered_data.clear()
    else:
        snap_df = _cached_snap_df

    cache_key = (dte_filter, _cached_agg_mtime, _cached_snap_mtime)
    if cache_key in _cached_filtered_data:
        return _cached_filtered_data[cache_key]['agg'], _cached_filtered_data[cache_key]['snap']

    filtered_agg = agg_df.copy()
    filtered_snap = snap_df.copy()

    try:
        filtered_agg = _apply_dte_filter_agg(filtered_agg, dte_filter)
    except Exception:
        pass
    try:
        filtered_snap = _apply_dte_filter(filtered_snap, dte_filter)
    except Exception:
        pass

    _cached_filtered_data[cache_key] = {
        'agg': filtered_agg,
        'snap': filtered_snap
    }

    return filtered_agg, filtered_snap


def get_latest_stats(agg_df, snap_df):
    """Compute lightweight per-symbol stats for the header tiles."""
    stats = {}
    if agg_df is None or agg_df.empty or "symbol" not in agg_df.columns:
        return stats
    for sym in agg_df["symbol"].unique():
        sym_df = agg_df[agg_df["symbol"] == sym].sort_values("_ts_parsed")
        if sym_df.empty:
            continue
        last = sym_df.iloc[-1]
        first = sym_df.iloc[0]
        price = float(last.get("spot", float("nan"))) if "spot" in last.index else float("nan")
        base = float(first.get("spot", float("nan"))) if "spot" in first.index else price
        if pd.notna(price) and pd.notna(base) and base != 0:
            price_change = (price / base - 1.0) * 100.0
        else:
            price_change = 0.0
        cv_col = next((c for c in ("call_vol", "callvol") if c in sym_df.columns), None)
        pv_col = next((c for c in ("put_vol", "putvol") if c in sym_df.columns), None)
        if cv_col and pv_col:
            cv = float(last.get(cv_col, 0.0) or 0.0)
            pv = float(last.get(pv_col, 0.0) or 0.0)
            vol_ratio = cv / pv if pv > 0 else (cv if cv > 0 else 1.0)
        else:
            cv = 0.0
            pv = 0.0
            vol_ratio = 1.0
        net_gamma = float(last.get("net_gex", 0.0) or 0.0)
        pc_ratio = float(last.get("pc_ratio", 0.0) or 0.0)
        iv_skew = float(last.get("iv_skew", 0.0) or 0.0)
        net_premium = float(last.get("net_premium", 0.0) or 0.0)
        call_premium = float(last.get("call_premium", 0.0) or 0.0)
        put_premium = float(last.get("put_premium", 0.0) or 0.0)
        atm_straddle = float(last.get("atm_straddle", 0.0) or 0.0)
        call_iv = float(last.get("call_iv", 0.0) or 0.0)
        put_iv = float(last.get("put_iv", 0.0) or 0.0)
        trade_aggression = float(last.get("trade_aggression", 0.0) or 0.0)
        bid_ask_imbalance = float(last.get("bid_ask_imbalance", 0.0) or 0.0)
        n_contracts = float(last.get("n_contracts", 0.0) or 0.0)
        avg_spread_pct = float(last.get("avg_spread_pct", 0.0) or 0.0)
        avg_trade_size = float(last.get("avg_trade_size", 0.0) or 0.0)
        stats[sym] = {
            "price": price,
            "price_change": price_change,
            "vol_ratio": vol_ratio,
            "net_gamma": net_gamma,
            "pc_ratio": pc_ratio,
            "iv_skew": iv_skew,
            "net_premium": net_premium,
            "call_premium": call_premium,
            "put_premium": put_premium,
            "atm_straddle": atm_straddle,
            "call_iv": call_iv,
            "put_iv": put_iv,
            "trade_aggression": trade_aggression,
            "bid_ask_imbalance": bid_ask_imbalance,
            "n_contracts": n_contracts,
            "call_vol": cv,
            "put_vol": pv,
            "avg_spread_pct": avg_spread_pct,
            "avg_trade_size": avg_trade_size,
        }
    return stats


# ---------------------------------------------------------------------------
# Modern Pro Terminal Color Palette
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers for modern layout
# ---------------------------------------------------------------------------

def _fmt_premium(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    av = abs(val)
    if av >= 1e9:
        return f"${val/1e9:+.2f}B" if val != 0 else "$0"
    if av >= 1e6:
        return f"${val/1e6:+.1f}M" if val != 0 else "$0"
    if av >= 1e3:
        return f"${val/1e3:+.0f}K" if val != 0 else "$0"
    return f"${val:+.0f}" if val != 0 else "$0"


def _fmt_premium_abs(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    av = abs(val)
    if av >= 1e9:
        return f"${av/1e9:.2f}B"
    if av >= 1e6:
        return f"${av/1e6:.1f}M"
    if av >= 1e3:
        return f"${av/1e3:.0f}K"
    return f"${av:.0f}"


def _mc_metric_card(label, value, color, sub=None):
    """Glassmorphism metric card for the modern layout."""
    children = [
        html.Div(label, style={
            'fontSize': '12px', 'fontWeight': 600, 'letterSpacing': '1.2px',
            'textTransform': 'uppercase', 'color': MC['text_muted'], 'marginBottom': '4px',
        }),
        html.Div(value, style={
            'fontSize': '32px', 'fontWeight': 700, 'color': color, 'lineHeight': '1.1',
        }),
    ]
    if sub:
        children.append(html.Div(sub, style={'fontSize': '13px', 'color': MC['text_muted'], 'marginTop': '2px'}))
    return html.Div(
        style={
            'background': _hex_to_rgba(MC['bg_card'], 0.78), 'backdropFilter': 'blur(12px)',
            'WebkitBackdropFilter': 'blur(12px)',
            'border': f'1px solid {MC["border"]}', 'borderRadius': '8px',
            'padding': '10px 12px', 'flex': '1', 'minWidth': '95px',
            'transition': 'all 0.2s ease',
        },
        children=children,
    )


def _mc_ticker_ribbon(all_stats):
    if not all_stats:
        return None
    items = []
    for sym in ['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT', 'VIXW']:
        if sym not in all_stats:
            continue
        st = all_stats[sym]
        chg = st.get('price_change', 0.0)
        chg_color = MC['call'] if chg >= 0 else MC['put']
        items.append(html.Div(
            style={
                'display': 'flex', 'alignItems': 'center', 'gap': '6px',
                'padding': '4px 12px', 'flex': '1', 'justifyContent': 'center',
                'borderRight': f'1px solid {MC["border"]}',
            },
            children=[
                html.Span(sym, style={'fontWeight': 700, 'fontSize': '14px', 'color': MC['accent']}),
                html.Span(f"${st['price']:.2f}" if pd.notna(st.get('price')) else "--", style={'fontSize': '16px', 'color': MC['text'], 'fontWeight': 600}),
                html.Span(f"{chg:+.2f}%" if pd.notna(chg) else "--", style={'fontSize': '13px', 'color': chg_color, 'fontWeight': 700}),
            ]
        ))
    if not items:
        return None
    return html.Div(
        style={
            'display': 'flex', 'background': MC['bg_card'],
            'border': f'1px solid {MC["border"]}', 'borderRadius': '8px',
            'marginBottom': '8px', 'overflow': 'hidden',
        },
        children=items,
    )


def _mc_premium_flow(st):
    call_p = st.get('call_premium', 0.0)
    put_p = st.get('put_premium', 0.0)
    total = call_p + put_p
    if total <= 0:
        return None
    call_pct = call_p / total * 100
    put_pct = put_p / total * 100
    return html.Div(
        style={
            'background': MC['bg_card'], 'border': f'1px solid {MC["border"]}',
            'borderRadius': '10px', 'padding': '14px 18px', 'marginBottom': '14px',
        },
        children=[
            html.Div('PREMIUM FLOW', style={
                'fontSize': '13px', 'fontWeight': 700, 'letterSpacing': '1.5px',
                'color': MC['accent'], 'marginBottom': '10px',
            }),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                html.Span(f"CALL {_fmt_premium_abs(call_p)} ({call_pct:.0f}%)", style={
                    'fontSize': '15px', 'fontWeight': 700, 'color': MC['call'], 'minWidth': '170px',
                }),
                html.Div(style={
                    'flex': '1', 'height': '24px', 'display': 'flex', 'borderRadius': '6px',
                    'overflow': 'hidden', 'background': 'rgba(255,255,255,0.03)',
                }, children=[
                    html.Div(style={
                        'width': f'{call_pct:.1f}%',
                        'background': f'linear-gradient(90deg, {MC["call"]}80, {MC["call"]}40)',
                        'borderRadius': '6px 0 0 6px',
                    }),
                    html.Div(style={
                        'width': f'{put_pct:.1f}%',
                        'background': f'linear-gradient(90deg, {MC["put"]}40, {MC["put"]}80)',
                        'borderRadius': '0 6px 6px 0',
                    }),
                ]),
                html.Span(f"PUT {_fmt_premium_abs(put_p)} ({put_pct:.0f}%)", style={
                    'fontSize': '15px', 'fontWeight': 700, 'color': MC['put'],
                    'minWidth': '170px', 'textAlign': 'right',
                }),
            ]),
            html.Div(f"Net: {_fmt_premium(st.get('net_premium', 0.0))} | Total: {_fmt_premium_abs(total)}", style={
                'fontSize': '13px', 'color': MC['text_muted'], 'marginTop': '6px', 'textAlign': 'center',
            }),
        ]
    )


def _mc_regime_badge(net_gex, vix_level=None):
    if net_gex >= 0:
        txt = "POSITIVE GAMMA"
        sub = "Mean-Reverting \u2014 Dealer hedging dampens moves"
        col = MC['call']
    else:
        txt = "NEGATIVE GAMMA"
        sub = "Trend-Following \u2014 Dealer hedging amplifies moves"
        col = MC['put']
    children = [
        html.Span(txt, style={'fontWeight': 700, 'fontSize': '19px', 'color': col}),
        html.Span(f" \u2014 {sub}", style={'fontSize': '16px', 'color': MC['text_sec']}),
        html.Span(
            f"  (net GEX: {net_gex:,.0f} \u2014 dealer position, independent of today\u2019s flow)",
            style={'fontSize': '11px', 'color': MC['text_muted'], 'marginLeft': '6px'},
        ),
    ]
    if vix_level and vix_level > 0:
        vc = MC['put'] if vix_level > 25 else (MC['warning'] if vix_level > 18 else MC['call'])
        children.append(html.Span(f"  |  VIX: {vix_level:.1f}", style={'fontSize': '16px', 'color': vc, 'fontWeight': 700}))
    return html.Div(
        style={
            'background': MC['bg_card'],
            'borderLeft': f'4px solid {col}',
            'border': f'1px solid {MC["border"]}',
            'borderRadius': '8px', 'padding': '10px 16px', 'marginBottom': '12px',
        },
        children=children,
    )


def _mc_rule_signal(st):
    """
    Rule-based directional indicator using 7 options-microstructure signals.

    Formula (industry-standard components):
      1. Premium Flow      (25%) — (call_premium - put_premium) / total  [-1, +1]
      2. P/C Ratio         (20%) — inverted: high put buying = bearish
      3. Vol Ratio C/P     (15%) — (call_vol - put_vol) / total_vol  [-1, +1]
      4. Trade Aggression  (15%) — buyer- vs seller-initiated trades (price vs mid)
      5. Bid/Ask Imbalance (10%) — quote-side size: ask lift > bid hit = bullish
      6. IV Skew           (10%) — put_iv - call_iv; elevated put skew = bearish
      7. GEX Regime         (5%) — neg GEX = trend continuation, pos = mean reversion

    Composite score in [-1, +1]:
      > +0.20  → BULL  | 0.20-0.40 = Low, 0.40-0.65 = Medium, >0.65 = High
      < -0.20  → BEAR  | same magnitude thresholds
      ±0.20    → NEUTRAL (conflicting signals)

    Confidence is additionally penalised when avg_spread_pct is wide
    (wide spreads = market-maker uncertainty = less reliable quote-based signals).
    """
    import math

    def _safe(key, default=0.0):
        v = st.get(key, default)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    # ── 1. Premium Flow score (25%) ──
    call_p  = _safe('call_premium')
    put_p   = _safe('put_premium')
    total_p = call_p + put_p
    s_flow  = (call_p - put_p) / total_p if total_p > 0 else 0.0
    s_flow  = max(-1.0, min(1.0, s_flow))

    # ── 2. P/C Ratio score (20%) ──
    pcr   = _safe('pc_ratio', 1.0)
    s_pcr = max(-1.0, min(1.0, (1.0 - pcr)))

    # ── 3. Volume Ratio score (15%) ──
    cv      = _safe('call_vol')
    pv      = _safe('put_vol')
    total_v = cv + pv
    s_vol   = (cv - pv) / total_v if total_v > 0 else 0.0
    s_vol   = max(-1.0, min(1.0, s_vol))

    # ── 4. Trade Aggression score (15%) ──
    # trade_aggression: fraction of trades near ask minus near bid; range ~[-0.5, +0.5] → scale x2
    s_agg = max(-1.0, min(1.0, _safe('trade_aggression') * 2.0))

    # ── 5. Bid/Ask Imbalance score (10%) ──
    # bid_ask_imbalance: (ask_size - bid_size) / (ask_size + bid_size)
    # Positive = more size on ask = buyers lifting offers = bullish
    s_bai = max(-1.0, min(1.0, _safe('bid_ask_imbalance') * 2.0))

    # ── 6. IV Skew score (10%) ──
    # iv_skew = put_iv - call_iv; elevated put skew = bearish
    skew   = _safe('iv_skew')
    s_skew = max(-1.0, min(1.0, -skew * 10.0))

    # ── 7. GEX Regime score (5%) ──
    net_gex  = _safe('net_gamma')
    base_dir_val = (0.25 * s_flow + 0.20 * s_pcr + 0.15 * s_vol
                    + 0.15 * s_agg + 0.10 * s_bai + 0.10 * s_skew)
    base_dir = 1.0 if base_dir_val > 0 else (-1.0 if base_dir_val < 0 else 0.0)
    if net_gex == 0 or base_dir == 0:
        s_gex = 0.0
    elif net_gex < 0:
        s_gex = base_dir        # neg GEX amplifies prevailing trend
    else:
        s_gex = -base_dir       # pos GEX dampens / mean-reverts

    # ── Weighted composite ──
    weights   = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
    scores    = [s_flow, s_pcr, s_vol, s_agg, s_bai, s_skew, s_gex]
    composite = sum(w * s for w, s in zip(weights, scores))

    # ── Confidence: weighted agreement + magnitude + spread penalty ──
    if composite == 0:
        w_agree = 0.0
    else:
        w_agree = sum(w for w, s in zip(weights, scores)
                      if s != 0 and (s > 0) == (composite > 0))
    magnitude = min(abs(composite) / 0.6, 1.0)
    raw_conf  = w_agree * (0.50 + 0.50 * magnitude)
    confidence = 1.0 / (1.0 + math.exp(-5.5 * (raw_conf - 0.45)))

    # Spread penalty: wide spreads reduce reliability of quote-based signals
    avg_spread = _safe('avg_spread_pct')
    if avg_spread > 5.0:        # >5% spread = very illiquid
        confidence *= 0.70
    elif avg_spread > 2.5:      # 2.5–5% = somewhat illiquid
        confidence *= 0.85

    # ── Direction & label ──
    if composite > 0.20:
        direction = "BULL"
        dir_color = MC['call']
        arrow = "▲"
    elif composite < -0.20:
        direction = "BEAR"
        dir_color = MC['put']
        arrow = "▼"
    else:
        direction = "NEUTRAL"
        dir_color = MC['warning']
        arrow = "◆"

    if confidence >= 0.80:
        conf_label, conf_color = "High conf", MC['call']
    elif confidence >= 0.60:
        conf_label, conf_color = "Med conf", MC['warning']
    else:
        conf_label, conf_color = "Low conf", MC['text_muted']

    # ── Sub-signal breakdown rows ──
    signal_rows = [
        ("Premium Flow",       s_flow,  0.25),
        ("P/C Ratio",          s_pcr,   0.20),
        ("Vol Ratio C/P",      s_vol,   0.15),
        ("Trade Aggression",   s_agg,   0.15),
        ("Bid/Ask Imbalance",  s_bai,   0.10),
        ("IV Skew",            s_skew,  0.10),
        ("GEX Regime",         s_gex,   0.05),
    ]

    def _signal_bar(score):
        """Mini horizontal bar centered at 0; green left=bear, green right=bull."""
        bull_w = f"{max(0.0, score) * 50:.1f}%"
        bear_w = f"{max(0.0, -score) * 50:.1f}%"
        bar_col = MC['call'] if score > 0 else (MC['put'] if score < 0 else MC['text_muted'])
        return html.Div(style={
            'display': 'flex', 'alignItems': 'center', 'height': '8px',
            'width': '100%', 'gap': '0',
        }, children=[
            # left half (bear side)
            html.Div(style={'flex': '1', 'height': '8px', 'display': 'flex',
                            'justifyContent': 'flex-end', 'backgroundColor': 'rgba(239,68,68,0.12)',
                            'borderRadius': '3px 0 0 3px', 'overflow': 'hidden'}, children=[
                html.Div(style={'width': bear_w, 'height': '100%',
                                'backgroundColor': MC['put'], 'borderRadius': '3px 0 0 3px'}),
            ]),
            # center tick
            html.Div(style={'width': '1px', 'height': '12px', 'backgroundColor': 'rgba(148,163,184,0.4)'}),
            # right half (bull side)
            html.Div(style={'flex': '1', 'height': '8px', 'display': 'flex',
                            'justifyContent': 'flex-start', 'backgroundColor': 'rgba(16,185,129,0.12)',
                            'borderRadius': '0 3px 3px 0', 'overflow': 'hidden'}, children=[
                html.Div(style={'width': bull_w, 'height': '100%',
                                'backgroundColor': MC['call'], 'borderRadius': '0 3px 3px 0'}),
            ]),
        ])

    breakdown = [
        html.Div(style={
            'display': 'grid', 'gridTemplateColumns': '140px 1fr 44px',
            'alignItems': 'center', 'gap': '8px', 'marginBottom': '5px',
        }, children=[
            html.Span(label, style={'fontSize': '11px', 'color': MC['text_muted'],
                                    'fontWeight': 600, 'letterSpacing': '0.3px'}),
            _signal_bar(score),
            html.Span(f"{score:+.2f}", style={
                'fontSize': '11px', 'fontWeight': 700, 'textAlign': 'right',
                'color': MC['call'] if score > 0 else (MC['put'] if score < 0 else MC['text_muted']),
            }),
        ])
        for label, score, _ in signal_rows
    ]

    # ── Composite bar (wider) ──
    comp_pct = abs(composite) * 100
    comp_bar = html.Div(style={
        'height': '10px', 'borderRadius': '5px',
        'backgroundColor': 'rgba(255,255,255,0.06)', 'overflow': 'hidden',
        'marginTop': '4px', 'marginBottom': '2px',
    }, children=[
        html.Div(style={
            'width': f'{min(comp_pct, 100):.1f}%', 'height': '100%',
            'backgroundColor': dir_color, 'borderRadius': '5px',
            'transition': 'width 0.4s ease',
        })
    ])

    return html.Div(
        style={
            'backgroundColor': MC['bg_card'],
            'border': f'1px solid {MC["border"]}',
            'borderLeft': f'4px solid {dir_color}',
            'borderRadius': '10px', 'padding': '14px 18px', 'marginBottom': '14px',
        },
        children=[
            # Header row
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                            'alignItems': 'center', 'marginBottom': '10px'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'baseline', 'gap': '8px'}, children=[
                    html.Span("RULE-BASED SIGNAL", style={
                        'fontSize': '12px', 'fontWeight': 700, 'letterSpacing': '1.2px',
                        'color': MC['accent'], 'textTransform': 'uppercase',
                    }),
                    html.Span("vs ML Model", style={
                        'fontSize': '10px', 'color': MC['text_muted'],
                    }),
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Span(f"{arrow} {direction}", style={
                        'fontSize': '20px', 'fontWeight': 800, 'color': dir_color,
                        'letterSpacing': '1px',
                    }),
                    html.Div(style={
                        'backgroundColor': f'{conf_color}22', 'border': f'1px solid {conf_color}66',
                        'borderRadius': '4px', 'padding': '2px 8px',
                    }, children=html.Span(f"{conf_label} conf · {confidence:.0%}", style={
                        'fontSize': '11px', 'fontWeight': 700, 'color': conf_color,
                    })),
                ]),
            ]),
            # Composite score bar
            html.Div(style={'marginBottom': '2px'}, children=[
                html.Span(f"Score: {composite:+.3f}", style={
                    'fontSize': '12px', 'fontWeight': 700, 'color': dir_color,
                }),
                html.Span("  (−1 = full bear, +1 = full bull)", style={
                    'fontSize': '10px', 'color': MC['text_muted'],
                }),
            ]),
            comp_bar,
            # Divider
            html.Hr(style={'borderColor': MC['border'], 'margin': '10px 0'}),
            # Sub-signal breakdown
            html.Div("SIGNAL BREAKDOWN", style={
                'fontSize': '10px', 'fontWeight': 700, 'letterSpacing': '1px',
                'color': MC['text_muted'], 'marginBottom': '7px',
            }),
            html.Div(breakdown),
            html.Div(
                f"Spread: {_safe('avg_spread_pct'):.2f}%"
                + (" (wide — conf penalised)" if _safe('avg_spread_pct') > 2.5 else "")
                + f"  |  Avg trade size: {_safe('avg_trade_size'):.1f}",
                style={'fontSize': '10px', 'color': MC['text_muted'], 'marginTop': '6px'},
            ),
        ],
    )


def _create_signal_divergence_chart(agg_df, symbol, pred_source):
    """
    Dual-axis signal divergence chart.

    Left  axis — RULE-BASED SIGNAL  [-1, +1]   zero line at y=0
    Right axis — HYBRID51 ENSEMBLE  [0.20,0.80] neutral line at prob=0.50

    Natural ranges from 5-year training data:
      Prob p5-p95  : 0.35 – 0.72   (normal trade range)
      Prob p10-p90 : 0.45 – 0.70   (tight range)

    Each line is BLUE when bullish (rule>0 / prob>0.5), RED when bearish.
    Shadow fill between the line and its own neutral axis:
      Both bull  → blue,  Both bear → red,  Opposite → purple (diverging).
    """
    import bisect

    if agg_df is None or agg_df.empty:
        return None, MC["border"]

    # ── 1. Rule-based composite ────────────────────────────────────────────
    sym_df = agg_df[agg_df["symbol"] == symbol].copy()
    if "_ts_parsed" not in sym_df.columns:
        sym_df["_ts_parsed"] = pd.to_datetime(sym_df["ts"], errors="coerce")
    sym_df = sym_df.sort_values("_ts_parsed")

    if sym_df.empty:
        return None, MC["border"]

    def _rule_composite(row):
        def _v(k, d=0.0):
            v = row.get(k, d)
            try:
                f = float(v)
                return d if (f != f) else f
            except Exception:
                return d
        cp = _v("call_premium"); pp = _v("put_premium"); tp = cp + pp
        s_flow = max(-1.0, min(1.0, (cp - pp) / tp)) if tp > 0 else 0.0
        s_pcr  = max(-1.0, min(1.0, 1.0 - _v("pc_ratio", 1.0)))
        cv = _v("call_vol"); pv = _v("put_vol"); tv = cv + pv
        s_vol  = max(-1.0, min(1.0, (cv - pv) / tv)) if tv > 0 else 0.0
        s_agg  = max(-1.0, min(1.0, _v("trade_aggression") * 2.0))
        s_bai  = max(-1.0, min(1.0, _v("bid_ask_imbalance") * 2.0))
        s_skew = max(-1.0, min(1.0, -_v("iv_skew") * 10.0))
        gex    = _v("net_gex")
        base_dir_val = (0.25 * s_flow + 0.20 * s_pcr + 0.15 * s_vol
                        + 0.15 * s_agg + 0.10 * s_bai + 0.10 * s_skew)
        base_dir = 1.0 if base_dir_val > 0 else (-1.0 if base_dir_val < 0 else 0.0)
        if gex == 0 or base_dir == 0:
            s_gex = 0.0
        elif gex < 0:
            s_gex = base_dir
        else:
            s_gex = -base_dir
        return (0.25*s_flow + 0.20*s_pcr + 0.15*s_vol
                + 0.15*s_agg + 0.10*s_bai + 0.10*s_skew + 0.05*s_gex)

    # Build batch->timestamp map from agg so both lines share the same time basis.
    _batch_ts_map = {}
    if "batch_id" in sym_df.columns and "_ts_parsed" in sym_df.columns:
        try:
            _bt = sym_df[["batch_id", "_ts_parsed"]].copy()
            _bt["batch_id"] = pd.to_numeric(_bt["batch_id"], errors="coerce")
            _bt = _bt.dropna(subset=["batch_id", "_ts_parsed"]).sort_values(["batch_id", "_ts_parsed"])
            _bt = _bt.drop_duplicates(subset=["batch_id"], keep="last")
            _batch_ts_map = {int(r["batch_id"]): r["_ts_parsed"] for _, r in _bt.iterrows()}
        except Exception:
            _batch_ts_map = {}

    # ── 2. Hybrid51 raw prob (keep on [0,1] scale) — anchor to agg batch times ──
    hybrid_times  = []
    hybrid_probs  = []
    hybrid_thrs   = []
    if isinstance(pred_source, pd.DataFrame) and not pred_source.empty:
        _ts_clean  = pred_source["ts"].astype(str).str.replace(r'\s+[A-Z]{2,5}$', '', regex=True)
        _ts_parsed = pd.to_datetime(_ts_clean, errors="coerce")
        for idx, row in pred_source.iterrows():
            ts_val = _ts_parsed.iloc[idx] if isinstance(idx, int) else _ts_parsed[idx]
            # Prefer agg timestamp for the same batch_id to align with rule-based line.
            try:
                b = row.get("batch_id", None)
                b_num = int(float(b)) if b is not None and str(b) != "" else None
                if b_num is not None and b_num in _batch_ts_map:
                    ts_val = _batch_ts_map[b_num]
            except Exception:
                pass
            if pd.isna(ts_val):
                continue
            suppressed = str(row.get("suppressed", "False")).strip().lower() in ("true", "1", "yes")
            prob = float(row.get("prob", 0.5) or 0.5)
            thr = float(row.get("threshold", 0.36) or 0.36)
            hybrid_times.append(ts_val)
            hybrid_probs.append(0.5 if suppressed else prob)
            hybrid_thrs.append(thr)
    else:
        for h in (pred_source if pred_source is not None else []):
            if h.get("ts") is None:
                continue
            prob = float(h.get("prob", 0.5) or 0.5)
            thr = float(h.get("threshold", 0.36) or 0.36)
            hybrid_times.append(h["ts"])
            hybrid_probs.append(0.5 if h.get("suppressed", False) else prob)
            hybrid_thrs.append(thr)

    hybrid_thr = float(hybrid_thrs[-1]) if hybrid_thrs else 0.36

    # ── 1b. Compute rule composite ──────────────────────────────────────────
    rule_times  = list(sym_df["_ts_parsed"])
    rule_scores = [_rule_composite(r) for _, r in sym_df.iterrows()]

    if not rule_times:
        return None, MC["border"]

    # #region agent log
    try:
        _rule_min = str(min(rule_times)) if rule_times else None
        _rule_max = str(max(rule_times)) if rule_times else None
        _hyb_min = str(min(hybrid_times)) if hybrid_times else None
        _hyb_max = str(max(hybrid_times)) if hybrid_times else None
        _lag_secs = None
        _last_common_batch = None
        if isinstance(pred_source, pd.DataFrame) and not pred_source.empty and "batch_id" in pred_source.columns and "batch_id" in sym_df.columns:
            _pred_tmp = pred_source.copy()
            _pred_tmp["_batch_num"] = pd.to_numeric(_pred_tmp["batch_id"], errors="coerce")
            _pred_tmp["_pred_ts"] = pd.to_datetime(
                _pred_tmp["ts"].astype(str).str.replace(r"\s+[A-Z]{2,5}$", "", regex=True),
                errors="coerce",
            )
            _rule_tmp = sym_df.copy()
            _rule_tmp["_batch_num"] = pd.to_numeric(_rule_tmp["batch_id"], errors="coerce")
            _rule_tmp = _rule_tmp.sort_values(["_batch_num", "_ts_parsed"]).drop_duplicates(subset=["_batch_num"], keep="last")
            _pred_tmp = _pred_tmp.sort_values(["_batch_num", "_pred_ts"]).drop_duplicates(subset=["_batch_num"], keep="last")
            _merged = _rule_tmp[["_batch_num", "_ts_parsed"]].merge(
                _pred_tmp[["_batch_num", "_pred_ts"]],
                on="_batch_num",
                how="inner",
            ).dropna()
            if not _merged.empty:
                _merged["_delta_s"] = (_merged["_pred_ts"] - _merged["_ts_parsed"]).dt.total_seconds()
                _lag_secs = float(_merged["_delta_s"].iloc[-1])
                _last_common_batch = int(_merged["_batch_num"].iloc[-1])
        _debug_log(
            "H1_timestamp_source_mismatch",
            "theta_dashboard_v4_modern.py:_create_signal_divergence_chart",
            "timeline comparison",
            {
                "symbol": str(symbol),
                "rule_points": int(len(rule_times)),
                "hybrid_points": int(len(hybrid_times)),
                "rule_ts_min": _rule_min,
                "rule_ts_max": _rule_max,
                "hybrid_ts_min": _hyb_min,
                "hybrid_ts_max": _hyb_max,
                "last_common_batch": _last_common_batch,
                "pred_minus_rule_last_common_sec": _lag_secs,
            },
        )
    except Exception as _elog:
        _debug_log(
            "H4_log_failure_guard",
            "theta_dashboard_v4_modern.py:_create_signal_divergence_chart",
            "timeline logging failed",
            {"error": str(_elog)[:180]},
        )
    # #endregion

    # ── 3. Interpolate hybrid onto rule time-grid ──────────────────────────
    def _interp(t, xs, ys):
        if not xs:
            return 0.5
        if t <= xs[0]:  return ys[0]
        if t >= xs[-1]: return ys[-1]
        i = max(0, min(bisect.bisect_right(xs, t) - 1, len(xs) - 2))
        try:
            frac = (t - xs[i]).total_seconds() / (xs[i+1] - xs[i]).total_seconds()
        except Exception:
            frac = 0.0
        return ys[i] + frac * (ys[i+1] - ys[i])

    h_on_rule = [_interp(t, hybrid_times, hybrid_probs) for t in rule_times]

    # ── 4. Colors ──────────────────────────────────────────────────────────
    BLUE      = "#3B82F6"
    RED       = "#EF4444"
    FILL_BLUE = "rgba(59,130,246,0.30)"
    FILL_RED  = "rgba(239,68,68,0.30)"
    FILL_PURP = "rgba(139,92,246,0.55)"
    NO_LINE   = dict(color="rgba(0,0,0,0)", width=0)

    # ── 5. Shadow fill — each segment colored by both signals' direction ───
    # Rule neutral = 0, Hybrid neutral = S3_NEUTRAL (~0.43, model baseline when all agents neutral)
    # NOT the threshold (0.36) — threshold is the decision boundary, not the directional neutral.
    # Fill is the combined area of both lines toward their respective neutrals,
    # drawn on the LEFT axis scale (rule). Hybrid is mapped linearly so that
    # prob=S3_NEUTRAL → 0, prob=0.72 → +0.6, prob=0.35 → -0.6 (p5-p95 maps to ±0.6)
    P_CENTER = S3_NEUTRAL
    # Scale: map prob at (p95=0.715) relative to threshold to +0.6 on left axis
    P_SCALE  = 0.60 / max(0.715 - P_CENTER, 0.01)

    def _prob_to_rule_scale(p):
        return max(-1.0, min(1.0, (p - P_CENTER) * P_SCALE))

    h_mapped = [_prob_to_rule_scale(p) for p in h_on_rule]

    def _fill_segs(xs, rule_ys, hybrid_mapped):
        if len(xs) < 2:
            return
        # Color the fill between the two lines using ORIGINAL prob value (not mapped):
        #   Both lines bullish  → blue fill
        #   Both lines bearish  → red fill
        #   Lines in opposite directions → purple fill (diverging)
        # rule_ys  is on [-1,1] scale; neutral=0
        # hybrid_mapped is already mapped from prob via _prob_to_rule_scale; neutral=0
        # They share the same zero-crossing neutral so comparison is consistent.
        def _col(r, h):
            r_bull = r >= 0;  h_bull = h >= 0
            if r_bull and h_bull:   return FILL_BLUE
            if (not r_bull) and (not h_bull): return FILL_RED
            return FILL_PURP
        cur_col = _col(rule_ys[0], hybrid_mapped[0])
        cur_x   = [xs[0]];  cur_r = [rule_ys[0]];  cur_h = [hybrid_mapped[0]]
        segs = []
        for xi, ri, hi in zip(xs[1:], rule_ys[1:], hybrid_mapped[1:]):
            nc = _col(ri, hi)
            if nc != cur_col:
                segs.append((list(cur_x), list(cur_r), list(cur_h), cur_col))
                cur_x = [cur_x[-1], xi]; cur_r = [cur_r[-1], ri]; cur_h = [cur_h[-1], hi]
                cur_col = nc
            else:
                cur_x.append(xi); cur_r.append(ri); cur_h.append(hi)
        segs.append((cur_x, cur_r, cur_h, cur_col))
        for sx, sr, sh, col in segs:
            # Fill is drawn between the two lines (rule on forward pass, hybrid on reverse)
            poly_x = sx + sx[::-1]
            poly_y = sr + sh[::-1]
            yield poly_x, poly_y, col

    # ── 6. Colored line segments (blue above neutral, red below) ──────────
    def _line_segs_rule(xs, ys, width, name):
        """Rule: neutral=0."""
        if not xs: return []
        result = []; seg_x, seg_y = [xs[0]], [ys[0]]; cur_pos = ys[0] >= 0
        for xi, yi in zip(xs[1:], ys[1:]):
            np_ = yi >= 0
            if np_ != cur_pos:
                x0, y0 = seg_x[-1], seg_y[-1]
                if (yi - y0) != 0:
                    frac = -y0 / (yi - y0)
                    try:    mid_x = x0 + frac * (xi - x0)
                    except: mid_x = xi
                    seg_x.append(mid_x); seg_y.append(0.0)
                result.append((list(seg_x), list(seg_y), cur_pos))
                seg_x = [seg_x[-1], xi]; seg_y = [seg_y[-1], yi]; cur_pos = np_
            else:
                seg_x.append(xi); seg_y.append(yi)
        result.append((seg_x, seg_y, cur_pos))
        traces = []
        for i, (sx, sy, pos) in enumerate(result):
            traces.append(go.Scatter(
                x=sx, y=sy, mode="lines",
                name=name, legendgroup=name, showlegend=(i == 0),
                connectgaps=True,
                line=dict(color=BLUE if pos else RED, width=width),
                hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>",
            ))
        return traces

    def _line_segs_hybrid(xs, ys, width, name):
        """Hybrid: on yaxis2, neutral=S3_NEUTRAL (~0.43). Color by prob vs model baseline."""
        if not xs: return []
        result = []; seg_x, seg_y = [xs[0]], [ys[0]]; cur_pos = ys[0] >= S3_NEUTRAL
        for xi, yi in zip(xs[1:], ys[1:]):
            np_ = yi >= S3_NEUTRAL
            if np_ != cur_pos:
                x0, y0 = seg_x[-1], seg_y[-1]
                if (yi - y0) != 0:
                    frac = (S3_NEUTRAL - y0) / (yi - y0)
                    try:    mid_x = x0 + frac * (xi - x0)
                    except: mid_x = xi
                    seg_x.append(mid_x); seg_y.append(S3_NEUTRAL)
                result.append((list(seg_x), list(seg_y), cur_pos))
                seg_x = [seg_x[-1], xi]; seg_y = [seg_y[-1], yi]; cur_pos = np_
            else:
                seg_x.append(xi); seg_y.append(yi)
        result.append((seg_x, seg_y, cur_pos))
        traces = []
        for i, (sx, sy, pos) in enumerate(result):
            traces.append(go.Scatter(
                x=sx, y=sy, mode="lines",
                name=name, legendgroup=name, showlegend=(i == 0),
                yaxis="y2",
                connectgaps=True,
                line=dict(color=BLUE if pos else RED, width=width),
                hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
            ))
        return traces

    # ── 7. Assemble figure ─────────────────────────────────────────────────
    fig = go.Figure()

    # Zero / neutral reference lines
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(148,163,184,0.25)", line_width=1)

    # Shadow fills: each line fills toward zero with its own direction color.
    # Rule fills to y=0; hybrid_mapped fills to y=0 (both on left-axis scale).
    # When they point in opposite directions, the fills are on opposite sides of zero,
    # making divergence clearly visible without needing a combined purple polygon.

    def _fill_to_zero(xs, ys, pos_color, neg_color):
        """Yield (x_list, y_list, fill_color) segments, each filling to y=0."""
        if len(xs) < 2:
            return
        cur_pos = ys[0] >= 0
        cur_x = [xs[0]]; cur_y = [ys[0]]
        segs = []
        for xi, yi in zip(xs[1:], ys[1:]):
            np_ = yi >= 0
            if np_ != cur_pos:
                x0, y0 = cur_x[-1], cur_y[-1]
                if (yi - y0) != 0:
                    frac = -y0 / (yi - y0)
                    try:    mid_x = x0 + frac * (xi - x0)
                    except: mid_x = xi
                    cur_x.append(mid_x); cur_y.append(0.0)
                segs.append((list(cur_x), list(cur_y), cur_pos))
                cur_x = [cur_x[-1], xi]; cur_y = [cur_y[-1], yi]; cur_pos = np_
            else:
                cur_x.append(xi); cur_y.append(yi)
        segs.append((cur_x, cur_y, cur_pos))
        for sx, sy, pos in segs:
            poly_x = sx + sx[::-1]
            poly_y = sy + [0.0]*len(sy)
            yield poly_x, poly_y, (pos_color if pos else neg_color)

    FILL_RULE_BULL  = "rgba(59,130,246,0.25)"
    FILL_RULE_BEAR  = "rgba(239,68,68,0.25)"
    FILL_MODEL_BULL = "rgba(59,130,246,0.18)"
    FILL_MODEL_BEAR = "rgba(239,68,68,0.18)"

    for px, py, col in _fill_to_zero(rule_times, rule_scores, FILL_RULE_BULL, FILL_RULE_BEAR):
        fig.add_trace(go.Scatter(
            x=px, y=py, fill="toself", fillcolor=col,
            line=NO_LINE, showlegend=False, hoverinfo="skip",
        ))
    for px, py, col in _fill_to_zero(rule_times, h_mapped, FILL_MODEL_BULL, FILL_MODEL_BEAR):
        fig.add_trace(go.Scatter(
            x=px, y=py, fill="toself", fillcolor=col,
            line=NO_LINE, showlegend=False, hoverinfo="skip",
        ))

    # Rule line (left axis, thick color-segmented overlay)
    for tr in _line_segs_rule(rule_times, rule_scores, width=3.0, name="Rule Score"):
        fig.add_trace(tr)

    # Hybrid line (right axis, thinner dotted)
    for tr in _line_segs_hybrid(hybrid_times, hybrid_probs, width=2.0, name="Hybrid51 Probability"):
        fig.add_trace(tr)

    # Model neutral reference on right axis (model baseline, not decision threshold)
    fig.add_hline(y=S3_NEUTRAL, line_dash="dot",
                  line_color="rgba(148,163,184,0.20)", line_width=1,
                  annotation_text=f"Model neutral: {S3_NEUTRAL:.2f}", annotation_position="right",
                  annotation_font=dict(size=9, color="rgba(148,163,184,0.6)"))
    # Decision threshold reference (separate, dimmer)
    fig.add_hline(y=hybrid_thr, line_dash="dash",
                  line_color="rgba(148,163,184,0.12)", line_width=1,
                  annotation_text=f"Decision thr: {hybrid_thr:.2f}", annotation_position="right",
                  annotation_font=dict(size=8, color="rgba(148,163,184,0.4)"))

    # ── 8. Layout ──────────────────────────────────────────────────────────
    last_rule   = rule_scores[-1]  if rule_scores  else 0.0
    last_prob   = hybrid_probs[-1] if hybrid_probs else S3_NEUTRAL
    diverging   = (last_rule >= 0) != (last_prob >= S3_NEUTRAL)
    status_text = "DIVERGING" if diverging else "ALIGNED"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=MC["bg_card"],
        plot_bgcolor=MC["bg_card"],
        title=dict(
            text=(
                "Signal Alignment: Rule Score vs Hybrid51 Probability"
                + ("  <span style='color:#8B5CF6;font-size:11px;'>● DIVERGING</span>"
                   if diverging else "")
            ),
            font=dict(size=13, color=MC["text"]),
            x=0.0, xanchor="left",
        ),
        legend=dict(
            orientation="h", x=0.0, y=1.08,
            font=dict(size=11, color=MC["text_sec"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            color=MC["text_muted"], tickfont=dict(size=10),
            domain=[0.0, 1.0],
        ),
        yaxis=dict(
            title=dict(text="Rule Score (-1 to +1)", font=dict(size=9, color=MC["text_muted"])),
            range=[-1.05, 1.05],
            showgrid=True, gridcolor=MC["grid"], zeroline=False,
            color=MC["text_muted"], tickfont=dict(size=9),
            tickvals=[-1.0, -0.6, 0.0, 0.6, 1.0],
            ticktext=["−1", "−0.6", "0", "+0.6", "+1"],
            side="left",
        ),
        yaxis2=dict(
            title=dict(text="Hybrid51 Probability", font=dict(size=9, color=MC["text_muted"])),
            range=[0.20, 0.80],
            showgrid=False, zeroline=False,
            color=MC["text_muted"], tickfont=dict(size=9),
            tickvals=[0.30, 0.36, S3_NEUTRAL, 0.60, 0.70],
            ticktext=["0.30", "0.36(thr)", f"{S3_NEUTRAL}↑(neutral)", "0.60", "0.70"],
            side="right",
            overlaying="y",
        ),
        margin=dict(l=50, r=130, t=42, b=10),
        height=270,
        hovermode="x unified",
        annotations=[
            dict(
                x=1.0, y=1.06, xref="paper", yref="paper",
                text=(
                    f"<b style='color:{BLUE if last_rule>=0 else RED}'>Rule score: {last_rule:+.2f}</b>"
                    f"  <b style='color:{BLUE if last_prob>=S3_NEUTRAL else RED}'>Model probability: {last_prob:.3f}</b>"
                    + f"  <b style='color:{'#8B5CF6' if diverging else MC['text_sec']}'>Status: {status_text}</b>"
                ),
                showarrow=False, font=dict(size=11), xanchor="right",
            ),
            # Box+arrow label: Rule Score — arrow from right-margin box to last point on left axis
            dict(
                x=rule_times[-1] if rule_times else 0,
                y=last_rule,
                xref="x", yref="y",
                text=f"<b>Rule Score</b>",
                showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.2,
                arrowcolor=BLUE if last_rule >= 0 else RED,
                ax=52, ay=0,
                font=dict(size=10, color=BLUE if last_rule >= 0 else RED),
                bgcolor="rgba(15,23,42,0.85)",
                bordercolor=BLUE if last_rule >= 0 else RED,
                borderwidth=1, borderpad=3,
                xanchor="left", yanchor="middle",
            ),
            # Box+arrow label: Hybrid51 Probability — arrow from right-margin box to last point on right axis
            dict(
                x=hybrid_times[-1] if hybrid_times else 0,
                y=last_prob,
                xref="x", yref="y2",
                text=f"<b>Model Prob</b>",
                showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.2,
                arrowcolor=BLUE if last_prob >= S3_NEUTRAL else RED,
                ax=52, ay=-28,
                font=dict(size=10, color=BLUE if last_prob >= S3_NEUTRAL else RED),
                bgcolor="rgba(15,23,42,0.85)",
                bordercolor=BLUE if last_prob >= S3_NEUTRAL else RED,
                borderwidth=1, borderpad=3,
                xanchor="left", yanchor="middle",
            ),
        ],
    )

    if diverging:
        border_col = "#8B5CF6"
    elif last_rule >= 0 and last_prob >= S3_NEUTRAL:
        border_col = "#3B82F6"
    elif last_rule < 0 and last_prob < S3_NEUTRAL:
        border_col = "#EF4444"
    else:
        border_col = MC["border"]
    return fig, border_col


def _create_pipeline_monitor():
    """
    System health monitor panel — one cell per pipeline step.
    Green = OK, Yellow = warn/stale, Red = error/missing.
    Positioned just above Volume & Flow.
    """
    import os
    import time as _time

    now = _time.time()
    STALE_SEC  = 90   # file not updated for this long → warning
    DEAD_SEC   = 180  # file not updated for this long → error

    # ── helpers ──────────────────────────────────────────────────────────────
    def _file_age(path):
        """Seconds since file was last modified. None if file missing."""
        try:
            return now - os.path.getmtime(str(path))
        except OSError:
            return None

    def _file_rows(path):
        """Row count of a CSV (excluding header). None if file missing."""
        try:
            count = 0
            with open(str(path), "rb") as f:
                for _ in f:
                    count += 1
            return max(0, count - 1)
        except OSError:
            return None

    def _last_csv_field(path, col):
        """Return the value of `col` in the last row of a CSV. None on any error."""
        try:
            df = pd.read_csv(str(path), usecols=[col])
            if df.empty:
                return None
            return df[col].iloc[-1]
        except Exception:
            return None

    # ── colour / status tokens ───────────────────────────────────────────────
    OK   = MC['call']
    WARN = MC['warning']
    ERR  = MC['put']

    _DOT_GLOW = {
        MC['call']:    "rgba(16,185,129,0.12)",
        MC['put']:     "rgba(239,68,68,0.12)",
        MC['warning']: "rgba(245,158,11,0.12)",
    }

    def _cell(label, color, detail):
        """One pill-shaped status cell."""
        bg_col = _DOT_GLOW.get(color, "rgba(59,130,246,0.08)")
        return html.Div(
            [
                html.Div(
                    style={
                        'width': '8px', 'height': '8px', 'borderRadius': '50%',
                        'backgroundColor': color, 'flexShrink': '0',
                        'boxShadow': f'0 0 6px {color}', 'marginTop': '3px',
                    }
                ),
                html.Div([
                    html.Div(label, style={
                        'fontSize': '11px', 'fontWeight': 700,
                        'letterSpacing': '0.5px', 'color': MC['text'],
                        'textTransform': 'uppercase',
                    }),
                    html.Div(detail, style={
                        'fontSize': '10px', 'color': MC['text_muted'],
                        'marginTop': '2px', 'lineHeight': '1.3',
                    }),
                ], style={'minWidth': '0', 'overflow': 'hidden'}),
            ],
            style={
                'display': 'flex', 'alignItems': 'flex-start', 'gap': '8px',
                'padding': '10px 12px',
                'borderRadius': '8px',
                'border': f'1px solid {color}',
                'backgroundColor': bg_col,
                'overflow': 'hidden',
            },
        )

    # ── 1. Fetcher process ────────────────────────────────────────────────────
    try:
        raw = open(str(STATUS_FILE)).read().strip()
        parts = raw.split("|")
        f_status = parts[0] if parts else "unknown"
        f_detail_raw = parts[4] if len(parts) > 4 else ""
    except Exception:
        f_status = "missing"
        f_detail_raw = ""

    age_status = _file_age(STATUS_FILE)

    if f_status == "running" and age_status is not None and age_status < DEAD_SEC:
        _c1 = OK
        _d1 = f"running · {int(age_status)}s ago"
    elif f_status == "error":
        _c1 = ERR
        _d1 = f"error: {f_detail_raw[:30]}" if f_detail_raw else "fetcher error"
    elif f_status == "stopped":
        _c1 = ERR
        _d1 = "stopped"
    elif age_status is None or age_status > DEAD_SEC:
        _c1 = ERR
        _d1 = "status file missing / stale"
    else:
        _c1 = WARN
        _d1 = f"status={f_status} · {int(age_status)}s ago"

    # ── 2. theta_agg.csv written ──────────────────────────────────────────────
    age_agg = _file_age(AGG_FILE)
    rows_agg = _file_rows(AGG_FILE)
    if age_agg is None:
        _c2, _d2 = ERR, "theta_agg.csv missing"
    elif age_agg < STALE_SEC:
        _c2, _d2 = OK, f"{int(age_agg)}s ago · {rows_agg} rows"
    elif age_agg < DEAD_SEC:
        _c2, _d2 = WARN, f"stale {int(age_agg)}s · {rows_agg} rows"
    else:
        _c2, _d2 = ERR, f"dead {int(age_agg)}s · {rows_agg} rows"

    # ── 3. theta_snapshot.csv written ────────────────────────────────────────
    age_snap = _file_age(SNAPSHOT_FILE)
    rows_snap = _file_rows(SNAPSHOT_FILE)
    if age_snap is None:
        _c3, _d3 = ERR, "theta_snapshot.csv missing"
    elif age_snap < STALE_SEC:
        _c3, _d3 = OK, f"{int(age_snap)}s ago · {rows_snap} rows"
    elif age_snap < DEAD_SEC:
        _c3, _d3 = WARN, f"stale {int(age_snap)}s · {rows_snap} rows"
    else:
        _c3, _d3 = ERR, f"dead {int(age_snap)}s · {rows_snap} rows"

    # ── 4. Model weights on disk ──────────────────────────────────────────────
    _model_dir = SCRIPT_DIR / "models"
    _s1_files  = list((_model_dir / "stage1").glob("*.pt")) if (_model_dir / "stage1").exists() else []
    _s2_files  = list((_model_dir / "stage2").glob("*.pt")) if (_model_dir / "stage2").exists() else []
    _s3_file   = _model_dir / "stage3" / "stage3_logreg.joblib"
    s1_ok = len(_s1_files) == 35
    s2_ok = len(_s2_files) == 7
    s3_ok = _s3_file.exists()
    if s1_ok and s2_ok and s3_ok:
        _c4, _d4 = OK, f"S1:{len(_s1_files)} S2:{len(_s2_files)} S3:logreg"
    else:
        missing = []
        if not s1_ok: missing.append(f"S1:{len(_s1_files)}/35")
        if not s2_ok: missing.append(f"S2:{len(_s2_files)}/7")
        if not s3_ok: missing.append("S3:missing")
        _c4, _d4 = ERR, " ".join(missing)

    # ── 5. Prediction service alive (prediction.csv freshness) ───────────────
    # Service only writes when fetcher produces a new batch_id — gaps of 60-120s
    # are normal. Use a generous threshold: warn at 3 min, error at 6 min.
    _PRED_STALE = 180
    _PRED_DEAD  = 360
    _pred_path = DATA_DIR / "prediction.csv"
    age_pred = _file_age(_pred_path)
    if age_pred is None:
        _c5, _d5 = ERR, "prediction.csv missing"
    elif age_pred < _PRED_STALE:
        _c5, _d5 = OK, f"writing · {int(age_pred)}s ago"
    elif age_pred < _PRED_DEAD:
        _c5, _d5 = WARN, f"no new batch {int(age_pred)}s"
    else:
        _c5, _d5 = ERR, f"dead {int(age_pred)}s — service down?"

    # ── 6. Inference quality (suppressed? stage failures?) ───────────────────
    try:
        _pred_df_mon = pd.read_csv(str(_pred_path), on_bad_lines='skip')
        if not _pred_df_mon.empty:
            _last = _pred_df_mon.iloc[-1]
            _suppressed = str(_last.get("suppressed", "True")).strip().lower() in ("true", "1")
            _reason     = str(_last.get("reason", "")).strip()
            _s1_miss    = int(_last.get("stage1_missing_count", 0))
            _s2_fail    = str(_last.get("stage2_failed_agents", "")).strip()
            _warmup     = float(_last.get("warmup_fraction", 0))
            _latency    = float(_last.get("latency_ms", 0))
            _s2_has_fail = bool(_s2_fail) and _s2_fail.lower() not in ("false", "none", "0", "nan")
            if _suppressed and _reason.startswith("warmup"):
                # Use total daily batch count instead of per-session warmup fraction
                import re as _re
                _wm = _re.search(r'of_(\d+)', _reason)
                _warmup_need = int(_wm.group(1)) if _wm else 20
                # Persist warmup progress across dashboard/service restarts:
                # count prediction rows for current ET date (not max batch_id,
                # which can reset when fetcher restarts).
                _batch_today = 0
                try:
                    if "ts" in _pred_df_mon.columns:
                        _ts_today = pd.to_datetime(
                            _pred_df_mon["ts"].astype(str).str.replace(r'\s+[A-Z]{2,5}$', '', regex=True),
                            errors="coerce",
                        )
                        _today_et = _now_et_naive().date()
                        _batch_today = int((_ts_today.dt.date == _today_et).sum())
                except Exception:
                    _batch_today = 0
                if _batch_today <= 0:
                    _batch_today = int(pd.to_numeric(_pred_df_mon["batch_id"], errors="coerce").max() or 0)
                _c6 = OK
                _d6 = f"warming up · {_batch_today}/{_warmup_need} batches today"
            elif _suppressed:
                _c6 = ERR
                _d6 = f"suppressed: {_reason[:28]}"
            elif _s1_miss > 0 or _s2_has_fail:
                _c6 = WARN
                _d6 = f"S1 miss:{_s1_miss}" if not _s2_has_fail else f"S2 fail:{_s2_fail[:20]}"
            else:
                _c6 = OK
                _d6 = f"live · {_latency:.0f}ms latency"
        else:
            _c6, _d6 = WARN, "no predictions yet"
    except Exception:
        _c6, _d6 = ERR, "cannot read prediction.csv"

    # ── 7. Prediction CSV batch lag vs agg ────────────────────────────────────
    try:
        _last_pred_batch = _last_csv_field(_pred_path, "batch_id")
        _last_agg_batch  = _last_csv_field(AGG_FILE, "batch_id")
        if _last_pred_batch is None or _last_agg_batch is None:
            _c7, _d7 = WARN, "batch_id unreadable"
        else:
            _lag = int(_last_agg_batch) - int(_last_pred_batch)
            rows_pred = _file_rows(_pred_path)
            if _lag <= 2:
                _c7 = OK
                _d7 = f"batch {int(_last_pred_batch)} · {rows_pred} rows"
            elif _lag <= 5:
                _c7 = WARN
                _d7 = f"lag {_lag} batches behind"
            else:
                _c7 = ERR
                _d7 = f"lag {_lag} batches — stalled"
    except Exception:
        _c7, _d7 = ERR, "cannot check batch lag"

    # ── 8. Dashboard prediction cache ─────────────────────────────────────────
    import time as _t2
    _cache_age = _t2.time() - _cached_pred_ts if _cached_pred_ts else 9999
    if _cached_pred_df is not None and not _cached_pred_df.empty:
        _n_cached = len(_cached_pred_df)
        if _cache_age < DEAD_SEC:
            _c8 = OK
            _d8 = f"{_n_cached} rows · refreshed {int(_cache_age)}s ago"
        else:
            _c8 = WARN
            _d8 = f"cache stale {int(_cache_age)}s"
    else:
        _c8 = WARN
        _d8 = "cache empty — loading..."

    # ── assemble ──────────────────────────────────────────────────────────────
    cells = [
        _cell("1 · Fetcher",         _c1, _d1),
        _cell("2 · Agg CSV",          _c2, _d2),
        _cell("3 · Snapshot CSV",     _c3, _d3),
        _cell("4 · Model Weights",    _c4, _d4),
        _cell("5 · Pred Service",     _c5, _d5),
        _cell("6 · Inference",        _c6, _d6),
        _cell("7 · Pred CSV",         _c7, _d7),
        _cell("8 · Dashboard Cache",  _c8, _d8),
    ]

    _all_colors = [_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8]
    if ERR in _all_colors:
        _banner_color = ERR
        _banner_label = "SYSTEM ERROR"
    elif WARN in _all_colors:
        _banner_color = WARN
        _banner_label = "WARNING"
    else:
        _banner_color = OK
        _banner_label = "ALL SYSTEMS OK"

    _banner_bg = _DOT_GLOW.get(_banner_color, "rgba(59,130,246,0.1)")

    return html.Div(
        [
            # Header row
            html.Div(
                [
                    html.Div("PIPELINE MONITOR", style={
                        'fontSize': '11px', 'fontWeight': 700, 'letterSpacing': '1.5px',
                        'color': MC['text_muted'], 'textTransform': 'uppercase',
                    }),
                    html.Div(_banner_label, style={
                        'fontSize': '11px', 'fontWeight': 700, 'letterSpacing': '1px',
                        'color': _banner_color,
                        'padding': '2px 10px', 'borderRadius': '10px',
                        'border': f'1px solid {_banner_color}',
                        'backgroundColor': _banner_bg,
                    }),
                ],
                style={
                    'display': 'flex', 'alignItems': 'center',
                    'justifyContent': 'space-between',
                    'marginBottom': '12px',
                },
            ),
            # 4-column × 2-row grid
            html.Div(
                cells,
                style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(4, 1fr)',
                    'gap': '8px',
                },
            ),
        ],
        style={
            'backgroundColor': MC['bg_card'],
            'border': f'1px solid {_banner_color}',
            'borderRadius': '10px',
            'padding': '16px',
            'marginBottom': '18px',
        },
    )


def _mc_section_header(title):
    return html.Div(title, style={
        'fontSize': '14px', 'fontWeight': 700, 'letterSpacing': '1.5px',
        'textTransform': 'uppercase', 'color': MC['accent'],
        'borderLeft': f'3px solid {MC["accent"]}',
        'paddingLeft': '12px', 'margin': '28px 0 14px 0',
    })


def _mc_group_header(title, subtitle=None):
    """Larger section divider for major dashboard groups (e.g. 'VOLUME & FLOW')."""
    children = [
        html.Div(title, style={
            'fontSize': '20px', 'fontWeight': 700, 'letterSpacing': '2px',
            'textTransform': 'uppercase', 'color': MC['text'],
        }),
    ]
    if subtitle:
        children.append(html.Div(subtitle, style={
            'fontSize': '14px', 'color': MC['text_muted'], 'marginTop': '4px',
        }))
    return html.Div(
        style={
            'borderBottom': f'2px solid {MC["accent"]}',
            'paddingBottom': '10px', 'margin': '36px 0 18px 0',
        },
        children=children,
    )


def _chart_insight_box(what, detail, suggestion, anomaly=None):
    """Compact insight box below each chart.
    - what: one-line description of what the chart shows
    - detail: current data-driven observation
    - suggestion: specific actionable guidance
    - anomaly: None | 'warn' | 'alert'
    """
    if anomaly == 'alert':
        border_c = MC['put']
        bg = 'rgba(239,68,68,0.06)'
        icon = '\U0001F6A8'  # siren
    elif anomaly == 'warn':
        border_c = MC['warning']
        bg = 'rgba(245,158,11,0.06)'
        icon = '\u26A0\uFE0F'  # warning
    else:
        border_c = MC['accent']
        bg = f'rgba(59,130,246,0.04)'
        icon = '\U0001F4A1'  # lightbulb
    return html.Div(
        style={
            'background': bg, 'borderLeft': f'3px solid {border_c}',
            'borderRadius': '4px', 'padding': '8px 12px',
            'margin': '-2px 0 14px 0', 'fontSize': '13px',
            'lineHeight': '1.5',
        },
        children=[
            html.Span(what, style={'color': MC['text_sec'], 'display': 'block'}),
            dcc.Markdown(
                detail,
                dangerously_allow_html=True,
                style={'color': MC['text'], 'display': 'block', 'marginTop': '2px', 'lineHeight': '1.5'}
            ) if detail else None,
            html.Span(
                f"{icon} {suggestion}",
                style={'color': border_c, 'display': 'block', 'marginTop': '4px', 'fontWeight': 600},
            ) if suggestion else None,
        ],
    )


def _mc_cross_symbol_table(all_stats):
    if not all_stats:
        return None
    hdr_s = {
        'fontSize': '9px', 'fontWeight': 700, 'letterSpacing': '0.8px',
        'color': MC['text_muted'], 'textTransform': 'uppercase',
        'padding': '10px 12px', 'textAlign': 'right',
        'borderBottom': f'1px solid {MC["border"]}',
    }
    cell_s = {
        'fontSize': '13px', 'padding': '10px 12px', 'textAlign': 'right',
        'borderBottom': f'1px solid {MC["border"]}',
    }
    cols = ['Symbol', 'Price', 'Chg%', 'P/C', 'Net GEX', 'IV Skew', 'Net Prem', 'Aggression']
    thead = html.Tr([html.Th(c, style={**hdr_s, 'textAlign': 'left' if c == 'Symbol' else 'right'}) for c in cols])
    rows = []
    for sym in ['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT', 'VIXW']:
        if sym not in all_stats:
            continue
        st = all_stats[sym]
        chg = st.get('price_change', 0.0)
        pc = st.get('pc_ratio', 0.0)
        gex = st.get('net_gamma', 0.0)
        agg = st.get('trade_aggression', 0.0)
        rows.append(html.Tr([
            html.Td(sym, style={**cell_s, 'textAlign': 'left', 'fontWeight': 700, 'color': MC['accent']}),
            html.Td(f"${st['price']:.2f}", style={**cell_s, 'color': MC['text']}),
            html.Td(f"{chg:+.2f}%", style={**cell_s, 'color': MC['call'] if chg >= 0 else MC['put'], 'fontWeight': 700}),
            html.Td(f"{pc:.3f}", style={**cell_s, 'color': MC['put'] if pc > 1.0 else MC['call'], 'fontWeight': 700}),
            html.Td(f"{gex/1e3:.1f}K" if abs(gex) < 1e6 else f"{gex/1e6:.1f}M", style={**cell_s, 'color': MC['call'] if gex >= 0 else MC['put']}),
            html.Td(f"{st.get('iv_skew', 0.0):.4f}", style={**cell_s, 'color': MC['text_sec']}),
            html.Td(_fmt_premium(st.get('net_premium', 0.0)), style={**cell_s, 'color': MC['call'] if st.get('net_premium', 0.0) >= 0 else MC['put']}),
            html.Td(f"{agg:+.3f}", style={**cell_s, 'color': MC['call'] if agg > 0 else MC['put'], 'fontWeight': 700}),
        ]))
    return html.Div(
        style={
            'background': MC['bg_card'], 'border': f'1px solid {MC["border"]}',
            'borderRadius': '10px', 'padding': '18px', 'marginBottom': '20px', 'overflowX': 'auto',
        },
        children=[
            html.Div('CROSS-SYMBOL OVERVIEW', style={
                'fontSize': '10px', 'fontWeight': 700, 'letterSpacing': '1.5px',
                'color': MC['accent'], 'marginBottom': '12px',
            }),
            html.Table(
                style={'width': '100%', 'borderCollapse': 'collapse'},
                children=[html.Thead(thead), html.Tbody(rows)],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Dash App Setup
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="Theta Options Pro Terminal")


def _insight(box_html):
    if not box_html:
        return None
    return dcc.Markdown(box_html, dangerously_allow_html=True)


def _mc_chart_card(graph_element, insight_element=None):
    """Wrap a dcc.Graph and optional insight in a styled card."""
    children = [graph_element]
    if insight_element is not None:
        children.append(insight_element)
    return html.Div(
        style={
            'background': MC['bg_card'], 'border': f'1px solid {MC["border"]}',
            'borderRadius': '10px', 'padding': '8px', 'marginBottom': '14px',
            'overflow': 'hidden',
        },
        children=children,
    )


app.layout = html.Div(
    style={
        "backgroundColor": MC["bg_dark"],
        "minHeight": "100vh",
        "color": MC["text"],
        "padding": "0",
        "fontFamily": "'Inter', 'SF Pro Display', 'Segoe UI', system-ui, -apple-system, sans-serif",
    },
    children=[
        # Global CSS — Modern Pro Terminal
        dcc.Markdown(
            f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

            :root {{
              --bg-dark: {MC['bg_dark']};
              --bg-card: {MC['bg_card']};
              --bg-input: {MC['bg_input']};
              --text: {MC['text']};
              --text-sec: {MC['text_sec']};
              --text-muted: {MC['text_muted']};
              --border: {MC['border']};
              --accent: {MC['accent']};
              --call: {MC['call']};
              --put: {MC['put']};
              --warning: {MC['warning']};
              --info: {MC['info']};
            }}

            html, body {{
              margin: 0 !important;
              padding: 0 !important;
              width: 100% !important;
              min-height: 100% !important;
              background: {MC['bg_dark']} !important;
              overflow-x: hidden !important;
              border: 0 !important;
              outline: 0 !important;
              box-shadow: none !important;
            }}
            body {{ background: {MC['bg_dark']} !important; }}

            /* Remove outer app frame/boundary so page blends into dashboard bg */
            #react-entry-point,
            #_dash-app-content,
            #_dash-app-content > div,
            #react-entry-point > div,
            .dash-spreadsheet-container,
            .dash-graph {{
              margin: 0 !important;
              padding: 0 !important;
              width: 100% !important;
              min-height: 100vh !important;
              background: {MC['bg_dark']} !important;
              border: none !important;
              outline: none !important;
              box-shadow: none !important;
            }}

            /* Safety reset for any default white frame wrappers */
            * {{
              box-sizing: border-box;
            }}

            /* Scrollbar */
            ::-webkit-scrollbar {{ width: 6px; }}
            ::-webkit-scrollbar-track {{ background: {MC['bg_dark']}; }}
            ::-webkit-scrollbar-thumb {{ background: {MC['border']}; border-radius: 3px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: {MC['accent']}; }}

            /* Dropdown overrides (force dark menu + readable text) */
            .mc-dropdown .Select-control,
            .mc-dropdown .Select-menu-outer,
            .mc-dropdown .Select-menu,
            .mc-dropdown .Select-option,
            .mc-dropdown .Select-value,
            .mc-dropdown .Select-value-label,
            .mc-dropdown .Select-placeholder,
            .mc-dropdown .VirtualizedSelectOption,
            .mc-dropdown .VirtualizedSelectFocusedOption,
            .mc-dropdown .VirtualizedSelectSelectedOption {{
              background: {MC['bg_input']} !important;
              color: {MC['text']} !important;
              border-color: {MC['border']} !important;
              font-size: 14px !important;
              font-weight: 700 !important;
            }}
            .mc-dropdown .Select-control:hover {{
              border-color: {MC['border_active']} !important;
            }}
            .mc-dropdown .Select.is-open > .Select-control,
            .mc-dropdown .is-open > .Select-control {{
              background: {MC['bg_input']} !important;
              border-color: {MC['border_active']} !important;
            }}
            .mc-dropdown .Select--single > .Select-control .Select-value,
            .mc-dropdown .Select--single > .Select-control .Select-placeholder {{
              color: {MC['text']} !important;
              opacity: 1 !important;
            }}
            .mc-dropdown .has-value.Select--single > .Select-control .Select-value .Select-value-label,
            .mc-dropdown .Select-value-label {{
              color: {MC['text']} !important;
              opacity: 1 !important;
              font-weight: 600 !important;
            }}
            .mc-dropdown .Select-placeholder {{
              color: {MC['text_muted']} !important;
            }}
            .mc-dropdown .Select-value {{
              color: {MC['text']} !important;
              opacity: 1 !important;
            }}
            .mc-dropdown .Select.is-disabled > .Select-control {{
              opacity: 1 !important;
              background: {MC['bg_input']} !important;
            }}
            .mc-dropdown .Select.is-disabled .Select-value-label {{
              color: {MC['text']} !important;
              opacity: 1 !important;
            }}
            .mc-dropdown .Select-arrow-zone .Select-arrow {{
              border-top-color: {MC['text_muted']} !important;
            }}
            .mc-dropdown .Select-option.is-focused {{
              background: rgba(59,130,246,0.18) !important;
              color: {MC['text']} !important;
            }}
            .mc-dropdown .Select-option.is-selected,
            .mc-dropdown .VirtualizedSelectFocusedOption {{
              background: rgba(59,130,246,0.28) !important;
              color: {MC['text']} !important;
            }}
            .mc-dropdown .Select-input input {{
              color: {MC['text']} !important;
            }}
            .mc-dropdown .Select-menu-outer {{
              border: 1px solid {MC['border']} !important;
              background: {MC['bg_input']} !important;
              z-index: 1002 !important;
            }}
            .mc-dropdown .Select-menu-outer .VirtualizedSelectOption {{
              background: {MC['bg_input']} !important;
              color: {MC['text']} !important;
              border-bottom: 1px solid rgba(148,163,184,0.16) !important;
            }}
            .mc-dropdown .Select-menu-outer .VirtualizedSelectFocusedOption {{
              background: rgba(59,130,246,0.22) !important;
              color: {MC['text']} !important;
            }}
            .mc-dropdown .Select-menu-outer .Select-noresults {{
              background: {MC['bg_input']} !important;
              color: {MC['text_muted']} !important;
            }}
            /* final fallback: force dark selects regardless wrapper class */
            .Select-control,
            .Select-menu-outer,
            .Select-menu,
            .Select-option,
            .Select-value,
            .Select-value-label,
            .Select-placeholder,
            .VirtualizedSelectOption,
            .VirtualizedSelectFocusedOption {{
              background: {MC['bg_input']} !important;
              color: {MC['text']} !important;
              border-color: {MC['border']} !important;
            }}
            .Select--single > .Select-control .Select-value,
            .Select--single > .Select-control .Select-value .Select-value-label,
            .Select--single > .Select-control .Select-placeholder {{
              color: {MC['text']} !important;
              opacity: 1 !important;
            }}
            .Select.is-disabled > .Select-control,
            .is-disabled.Select--single > .Select-control {{
              opacity: 1 !important;
            }}
            .Select.is-disabled .Select-value-label {{
              color: {MC['text']} !important;
              opacity: 1 !important;
            }}
            /* ID-level hard override for control boxes */
            #symbol-dropdown .Select-control,
            #dte-dropdown .Select-control,
            #compare-dropdown .Select-control,
            #window-dropdown .Select-control {{
              background: {MC['bg_input']} !important;
              border-color: {MC['border']} !important;
              min-height: 38px !important;
              box-shadow: none !important;
            }}
            #symbol-dropdown .Select-value-label,
            #dte-dropdown .Select-value-label,
            #compare-dropdown .Select-value-label,
            #window-dropdown .Select-value-label,
            #symbol-dropdown .Select-placeholder,
            #dte-dropdown .Select-placeholder,
            #compare-dropdown .Select-placeholder,
            #window-dropdown .Select-placeholder {{
              color: {MC['text']} !important;
              opacity: 1 !important;
              font-weight: 600 !important;
            }}
            #symbol-dropdown > div,
            #dte-dropdown > div,
            #compare-dropdown > div,
            #window-dropdown > div {{
              background: {MC['bg_input']} !important;
              border-radius: 6px !important;
            }}
            #symbol-dropdown .Select-input > input,
            #dte-dropdown .Select-input > input,
            #compare-dropdown .Select-input > input,
            #window-dropdown .Select-input > input {{
              color: {MC['text']} !important;
            }}
            #symbol-dropdown .Select-arrow,
            #dte-dropdown .Select-arrow,
            #compare-dropdown .Select-arrow,
            #window-dropdown .Select-arrow {{
              border-top-color: {MC['text']} !important;
            }}
            #symbol-dropdown .Select-menu-outer,
            #dte-dropdown .Select-menu-outer,
            #compare-dropdown .Select-menu-outer,
            #window-dropdown .Select-menu-outer {{
              background: {MC['bg_input']} !important;
            }}
            /* ultimate override: force every nested node in control wrappers */
            #symbol-dropdown *, #dte-dropdown *, #compare-dropdown *, #window-dropdown * {{
              color: {MC['text']} !important;
            }}
            #symbol-dropdown .Select-control, #dte-dropdown .Select-control,
            #compare-dropdown .Select-control, #window-dropdown .Select-control,
            #symbol-dropdown .Select__control, #dte-dropdown .Select__control,
            #compare-dropdown .Select__control, #window-dropdown .Select__control {{
              background-color: {MC['bg_input']} !important;
              border: 1px solid {MC['border']} !important;
            }}
            #symbol-dropdown .Select-value, #dte-dropdown .Select-value,
            #compare-dropdown .Select-value, #window-dropdown .Select-value,
            #symbol-dropdown .Select-input, #dte-dropdown .Select-input,
            #compare-dropdown .Select-input, #window-dropdown .Select-input,
            #symbol-dropdown .Select-multi-value-wrapper, #dte-dropdown .Select-multi-value-wrapper,
            #compare-dropdown .Select-multi-value-wrapper, #window-dropdown .Select-multi-value-wrapper {{
              background: transparent !important;
            }}
            #symbol-dropdown .Select__single-value, #dte-dropdown .Select__single-value,
            #compare-dropdown .Select__single-value, #window-dropdown .Select__single-value {{
              color: {MC['text']} !important;
              opacity: 1 !important;
              font-weight: 600 !important;
            }}
            #symbol-dropdown .Select__menu, #dte-dropdown .Select__menu,
            #compare-dropdown .Select__menu, #window-dropdown .Select__menu,
            #symbol-dropdown .Select__menu-list, #dte-dropdown .Select__menu-list,
            #compare-dropdown .Select__menu-list, #window-dropdown .Select__menu-list {{
              background: {MC['bg_input']} !important;
            }}

            /* Plotly overrides */
            .js-plotly-plot, .plotly {{
              background: {MC['bg_card']} !important;
              border-radius: 10px;
              border: none !important;
              outline: none !important;
            }}

            /* Lock all Plotly SVG text to fixed px — prevents font chaos on zoom */
            .js-plotly-plot .plotly .gtitle,
            .js-plotly-plot .plotly .xtitle,
            .js-plotly-plot .plotly .ytitle,
            .js-plotly-plot .plotly .xtick text,
            .js-plotly-plot .plotly .ytick text,
            .js-plotly-plot .plotly .legendtext,
            .js-plotly-plot .plotly .annotation-text,
            .js-plotly-plot text {{
              font-family: 'Inter', 'SF Pro Display', 'Segoe UI', system-ui, sans-serif !important;
              font-size: 11px !important;
            }}
            .js-plotly-plot .plotly .gtitle {{
              font-size: 13px !important;
              font-weight: 600 !important;
            }}

            /* Hide modebar (zoom toolbar) icons that show garbled text */
            .modebar-container {{
              display: none !important;
            }}

            /* Button hover effects */
            button:hover {{ opacity: 0.88; }}
            button:active {{ transform: scale(0.97); }}

            /* Signal flip blink animation */
            @keyframes flip-pulse {{
              0%   {{ box-shadow: 0 0 0 0 rgba(245,158,11,0.9); background-color: rgba(245,158,11,0.18); }}
              40%  {{ box-shadow: 0 0 0 14px rgba(245,158,11,0); background-color: rgba(245,158,11,0.08); }}
              100% {{ box-shadow: 0 0 0 0 rgba(245,158,11,0); background-color: transparent; }}
            }}
            @keyframes flip-pulse-bull {{
              0%   {{ box-shadow: 0 0 0 0 rgba(16,185,129,0.9); background-color: rgba(16,185,129,0.18); }}
              40%  {{ box-shadow: 0 0 0 14px rgba(16,185,129,0); background-color: rgba(16,185,129,0.08); }}
              100% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0); background-color: transparent; }}
            }}
            @keyframes flip-pulse-bear {{
              0%   {{ box-shadow: 0 0 0 0 rgba(239,68,68,0.9); background-color: rgba(239,68,68,0.18); }}
              40%  {{ box-shadow: 0 0 0 14px rgba(239,68,68,0); background-color: rgba(239,68,68,0.08); }}
              100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0); background-color: transparent; }}
            }}
            .signal-flip-bull {{
              animation: flip-pulse-bull 0.9s ease-out 3;
              border-color: {MC['call']} !important;
            }}
            .signal-flip-bear {{
              animation: flip-pulse-bear 0.9s ease-out 3;
              border-color: {MC['put']} !important;
            }}
            </style>
            <script>
            /* Force dropdown styling via JavaScript after React-Select renders */
            function forceDropdownStyles() {{
                /* Target all Select components in the page */
                const selectors = [
                    '.Select-control', '.Select-menu-outer', '.Select-menu',
                    '.Select-option', '.Select-value', '.Select-value-label',
                    '.Select-placeholder', '.VirtualizedSelectOption',
                    '.Select--single > .Select-control .Select-value'
                ];
                
                selectors.forEach(sel => {{
                    const elements = document.querySelectorAll(sel);
                    elements.forEach(el => {{
                        el.style.setProperty('background-color', '#334155', 'important');
                        el.style.setProperty('color', '#f1f5f9', 'important');
                        el.style.setProperty('border-color', 'rgba(59,130,246,0.15)', 'important');
                        el.style.setProperty('opacity', '1', 'important');
                        el.style.setProperty('font-weight', '600', 'important');
                    }});
                }});
                
                /* Force dropdown arrows */
                document.querySelectorAll('.Select-arrow').forEach(el => {{
                    el.style.setProperty('border-top-color', '#f1f5f9', 'important');
                }});
                
                /* Force menu options */
                document.querySelectorAll('.Select-option, .VirtualizedSelectOption').forEach(el => {{
                    el.style.setProperty('background-color', '#334155', 'important');
                    el.style.setProperty('color', '#f1f5f9', 'important');
                }});
            }}
            
            /* Run on load and on mutations (when dropdowns open) */
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', forceDropdownStyles);
            }} else {{
                forceDropdownStyles();
            }}
            
            /* Watch for dropdown menu rendering (portaled to body) */
            const observer = new MutationObserver(forceDropdownStyles);
            observer.observe(document.body, {{ childList: true, subtree: true }});
            
            /* Also run periodically for first 10 seconds */
            let count = 0;
            const interval = setInterval(() => {{
                forceDropdownStyles();
                count++;
                if (count > 20) clearInterval(interval);
            }}, 500);
            </script>
            """,
            dangerously_allow_html=True
        ),

        dcc.Store(id="refresh-paused", data=False),

        # ── Top Bar: Title + Status ──
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {MC['bg_dark']} 0%, {MC['bg_card']} 100%)",
                "padding": "16px 28px",
                "borderBottom": f"1px solid {MC['border']}",
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            },
            children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px"}, children=[
                    html.Div(style={
                        "width": "8px", "height": "32px", "borderRadius": "4px",
                        "background": f"linear-gradient(180deg, {MC['accent']}, {MC['info']})",
                    }),
                    html.Div(children=[
                        html.H1("THETA OPTIONS PRO", style={
                            "color": MC["text"], "margin": 0, "fontSize": "26px",
                            "fontWeight": 800, "letterSpacing": "2px",
                            "fontFamily": "'Inter', 'SF Pro Display', sans-serif",
                        }),
                        html.Div("Intelligence Terminal", style={
                            "color": MC["text_muted"], "fontSize": "14px",
                            "fontWeight": 500, "letterSpacing": "1px", "marginTop": "1px",
                        }),
                    ]),
                ]),
                html.Div(id="live-status", style={"fontSize": "15px"}),
            ],
        ),

        # ── Subheader info bar ──
        html.Div(
            id="subheader",
            style={
                "color": MC["text_muted"], "fontSize": "15px", "padding": "8px 28px",
                "background": MC["bg_dark"], "borderBottom": f"1px solid {MC['border']}",
                "fontFamily": "'JetBrains Mono', monospace", "letterSpacing": "0.3px",
            },
        ),

        # ── Controls Bar ──
        html.Div(
            style={
                "display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap",
                "backgroundColor": MC["bg_card"], "padding": "8px 28px",
                "borderBottom": f"1px solid {MC['border']}", "fontSize": "15px",
            },
            children=[
                html.Div([
                    html.Label("Symbol", style={'marginRight': '5px', 'color': MC['text_sec'], 'fontSize': '13px', 'fontWeight': 700, 'letterSpacing': '0.5px', 'textTransform': 'uppercase'}),
                    dcc.Dropdown(id='symbol-dropdown',
                        options=[{'label': s, 'value': s} for s in ['SPXW', 'SPY', 'QQQ', 'IWM', 'VIX', 'VIXW', 'TLT', 'ALL']],
                        value='SPXW', clearable=False, searchable=False,
                        className='mc-dropdown',
                        style={'width': '140px', 'backgroundColor': MC['bg_input'], 'color': MC['text'], 'fontSize': '14px', 'fontWeight': '700'})
                ]),
                html.Div([
                    html.Label("DTE", style={'marginRight': '5px', 'color': MC['text_sec'], 'fontSize': '13px', 'fontWeight': 700, 'letterSpacing': '0.5px', 'textTransform': 'uppercase'}),
                    dcc.Dropdown(id='dte-dropdown',
                        options=[{'label': '0-1 DTE', 'value': '0_1dte'}, {'label': '0DTE Only', 'value': '0dte'},
                                 {'label': '0-2 DTE', 'value': '0_2dte'}, {'label': 'All DTE', 'value': 'all'}],
                        value='0_1dte', clearable=False, searchable=False,
                        className='mc-dropdown',
                        style={'width': '140px', 'backgroundColor': MC['bg_input'], 'color': MC['text'], 'fontSize': '14px', 'fontWeight': '700'})
                ]),
                html.Div([
                    html.Label("Compare", style={'marginRight': '5px', 'color': MC['text_sec'], 'fontSize': '13px', 'fontWeight': 700, 'letterSpacing': '0.5px', 'textTransform': 'uppercase'}),
                    dcc.Dropdown(id='compare-dropdown',
                        options=[{'label': 'No Compare', 'value': 0}, {'label': 'vs 5m', 'value': 5},
                                 {'label': 'vs 15m', 'value': 15}, {'label': 'vs 30m', 'value': 30},
                                 {'label': 'vs 1h', 'value': 60}, {'label': 'vs 2h', 'value': 120}],
                        value=0, clearable=False, searchable=False,
                        className='mc-dropdown',
                        style={'width': '150px', 'backgroundColor': MC['bg_input'], 'color': MC['text'], 'fontSize': '14px', 'fontWeight': '700'})
                ]),
                html.Div([
                    html.Label("Window", style={'marginRight': '5px', 'color': MC['text_sec'], 'fontSize': '13px', 'fontWeight': 700, 'letterSpacing': '0.5px', 'textTransform': 'uppercase'}),
                    dcc.Dropdown(id='window-dropdown',
                        options=[{'label': 'Full Session', 'value': 'session'}, {'label': '15m', 'value': 15},
                                 {'label': '30m', 'value': 30}, {'label': '45m', 'value': 45}, {'label': '60m', 'value': 60}],
                        value='session', clearable=False, searchable=False,
                        className='mc-dropdown',
                        style={'width': '140px', 'backgroundColor': MC['bg_input'], 'color': MC['text'], 'fontSize': '14px', 'fontWeight': '700'})
                ]),
                html.Div(style={'marginLeft': 'auto', 'display': 'flex', 'gap': '6px'}, children=[
                    html.Button('START', id='btn-start', style={
                        'backgroundColor': MC['call'], 'color': '#fff', 'border': 'none',
                        'padding': '6px 14px', 'borderRadius': '6px', 'cursor': 'pointer',
                        'fontWeight': 700, 'fontSize': '13px', 'letterSpacing': '0.5px',
                    }),
                    html.Button('STOP', id='btn-stop', style={
                        'backgroundColor': MC['put'], 'color': '#fff', 'border': 'none',
                        'padding': '6px 14px', 'borderRadius': '6px', 'cursor': 'pointer',
                        'fontWeight': 700, 'fontSize': '13px', 'letterSpacing': '0.5px',
                    }),
                    html.Button('DELETE ALL', id='btn-delete-all', style={
                        'backgroundColor': '#4b5563', 'color': '#fff', 'border': 'none',
                        'padding': '6px 14px', 'borderRadius': '6px', 'cursor': 'pointer',
                        'fontWeight': 700, 'fontSize': '13px', 'letterSpacing': '0.5px',
                    }),
                    html.Button('PAUSE', id='btn-pause', n_clicks=0, style={
                        'backgroundColor': MC['warning'], 'color': '#fff', 'border': 'none',
                        'padding': '6px 14px', 'borderRadius': '6px', 'cursor': 'pointer',
                        'fontWeight': 700, 'fontSize': '13px', 'letterSpacing': '0.5px',
                    }),
                    html.Button('REFRESH', id='btn-refresh', n_clicks=0, style={
                        'backgroundColor': MC['accent'], 'color': '#fff', 'border': 'none',
                        'padding': '6px 14px', 'borderRadius': '6px', 'cursor': 'pointer',
                        'fontWeight': 700, 'fontSize': '13px', 'letterSpacing': '0.5px',
                    }),
                ]),
                html.Div(id='fetcher-status', style={'fontSize': '13px', 'color': MC['text_muted']}),
            ],
        ),

        # Auto-refresh interval (10 seconds)
        dcc.Interval(id='interval-update', interval=10*1000, n_intervals=0),
        html.Div(id='action-trigger', style={'display': 'none'}),

        # ── Dashboard Content ──
        html.Div(id='dashboard-content', style={'padding': '18px 28px'}),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    [Output("refresh-paused", "data"), Output("btn-pause", "children")],
    Input("btn-pause", "n_clicks"),
    State("refresh-paused", "data"),
    prevent_initial_call=True,
)
def toggle_pause(n_clicks, paused):
    paused = not bool(paused)
    return paused, ("RESUME REFRESH" if paused else "PAUSE REFRESH")


@app.callback(
    Output('action-trigger', 'children'),
    [Input('btn-start', 'n_clicks'),
     Input('btn-stop', 'n_clicks'),
     Input('btn-delete-all', 'n_clicks')],
    prevent_initial_call=True
)
def manage_fetcher(start_clicks, stop_clicks, delete_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-start':
        start_fetcher()
    elif button_id == 'btn-stop':
        stop_fetcher()
    elif button_id == 'btn-delete-all':
        # Ensure fetcher is stopped before deleting data
        stop_fetcher()
        delete_all_data()

    return str(time.time())


@app.callback(
    [Output('dashboard-content', 'children'),
     Output('fetcher-status', 'children'),
     Output('live-status', 'children'),
     Output('subheader', 'children')],
    [Input('interval-update', 'n_intervals'),
     Input('action-trigger', 'children'),
     Input('symbol-dropdown', 'value'),
     Input('dte-dropdown', 'value'),
     Input('compare-dropdown', 'value'),
     Input('window-dropdown', 'value'),
     Input('btn-refresh', 'n_clicks')],
    State("refresh-paused", "data"),
)
def update_dashboard(n, trigger, symbol, dte, compare, window, manual_refresh, paused):
    global _last_live_non_suppressed_ts

    ctx = dash.callback_context
    triggered_by = (ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "")
    if paused and triggered_by == "interval-update":
        return no_update, no_update, no_update, no_update

    # Load data
    df_agg, df_snap = load_data(dte_filter=dte)

    # Load prediction CSV (decoupled — no PyTorch needed)
    pred_df = _load_prediction_csv()
    latest_pred_row = _get_latest_prediction(pred_df)
    model_out = _prediction_row_to_model_out(latest_pred_row)

    # Build roll history from prediction CSV for components that need it
    pred_hist_df = _get_prediction_history(pred_df, n=40)
    pred_history_roll = _prediction_history_as_roll(pred_hist_df)

    # Update last live timestamp
    if model_out and not model_out.get("suppressed", False) and model_out.get("ok", False):
        _last_live_non_suppressed_ts = _now_et_naive()

    # Generate alerts from latest prediction
    _generate_alerts(model_out)

    # Use max batch_id from prediction.csv for a persistent daily count
    _csv_max_batch = "?"
    if pred_df is not None and not pred_df.empty and "batch_id" in pred_df.columns:
        try:
            _csv_max_batch = int(pd.to_numeric(pred_df["batch_id"], errors="coerce").max())
        except Exception:
            pass

    # Fetcher status
    status_html = []
    fs = {}
    if is_fetcher_running():
        fs = get_fetcher_status()
        status_html = html.Span([
            html.Span("\u25CF ", style={'color': MC['call'], 'fontSize': '14px'}),
            html.Span("Running ", style={'color': MC['call'], 'fontWeight': 600}),
            html.Span(f"B#{_csv_max_batch} | PID {fs.get('pid', '?')}", style={'color': MC['text_muted']})
        ])
    else:
        status_html = html.Span([
            html.Span("\u25CF ", style={'color': MC['put'], 'fontSize': '14px'}),
            html.Span("Stopped", style={'color': MC['put'], 'fontWeight': 600}),
        ])

    # Header status + subheader
    all_symbols = df_agg["symbol"].unique().tolist() if (df_agg is not None and not df_agg.empty and "symbol" in df_agg.columns) else []
    snap_count = 0
    try:
        snap_count = len(list_available_snapshots())
    except Exception:
        pass
    total_batches = int(df_agg["batch_id"].nunique()) if (df_agg is not None and not df_agg.empty and "batch_id" in df_agg.columns) else 0
    live_badge = (
        html.Span(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
            html.Span(style={
                'width': '8px', 'height': '8px', 'borderRadius': '50%', 'backgroundColor': MC['call'],
                'boxShadow': f'0 0 8px {MC["call"]}', 'display': 'inline-block',
            }),
            html.Span("LIVE", style={'color': MC['call'], 'fontWeight': 700, 'fontSize': '11px', 'letterSpacing': '1px'}),
            html.Span(f"Batch #{_csv_max_batch}", style={'color': MC['text_muted'], 'fontSize': '11px'}),
        ])
        if is_fetcher_running()
        else (html.Span(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
            html.Span(style={
                'width': '8px', 'height': '8px', 'borderRadius': '50%', 'backgroundColor': MC['warning'],
                'display': 'inline-block',
            }),
            html.Span("REVIEW", style={'color': MC['warning'], 'fontWeight': 700, 'fontSize': '11px', 'letterSpacing': '1px'}),
            html.Span(f"{snap_count} snapshots", style={'color': MC['text_muted'], 'fontSize': '11px'}),
        ]) if snap_count > 0 else html.Span(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
            html.Span(style={
                'width': '8px', 'height': '8px', 'borderRadius': '50%', 'backgroundColor': MC['put'],
                'display': 'inline-block',
            }),
            html.Span("STOPPED", style={'color': MC['put'], 'fontWeight': 700, 'fontSize': '11px', 'letterSpacing': '1px'}),
        ]))
    )

    last_update = "No data yet"
    if df_agg is not None and not df_agg.empty and "_ts_parsed" in df_agg.columns and df_agg["_ts_parsed"].notna().any():
        try:
            last_ts = df_agg["_ts_parsed"].max()
            if pd.notna(last_ts):
                age = _now_et_naive() - last_ts.to_pydatetime()
                age_secs = int(age.total_seconds())
                if age_secs < 60:
                    age_str = f"{age_secs}s ago"
                elif age_secs < 3600:
                    age_str = f"{age_secs // 60}m ago"
                else:
                    age_str = f"{age_secs // 3600}h {(age_secs % 3600) // 60}m ago"
                last_update = last_ts.strftime("%Y-%m-%d %H:%M:%S") + f" ({age_str})"
        except Exception:
            pass

    # Prediction status for subheader
    pred_status = "No predictions"
    if model_out:
        if model_out.get("suppressed"):
            pred_status = f"Suppressed: {model_out.get('reason', '')}"
        else:
            pred_status = f"P(up)={model_out['prob']:.1%} | {model_out.get('direction', '?')}"

    subheader = (
        f"Last data: {last_update} | {total_batches} batches | {snap_count} snapshots | "
        f"Window: {MARKET_OPEN_ET[0]}:{MARKET_OPEN_ET[1]:02d}-"
        f"{MARKET_CLOSE_ET[0]}:{MARKET_CLOSE_ET[1]:02d} ET | "
        f"{'Auto-refresh: 10s' if is_fetcher_running() else 'Market closed / Fetcher stopped'} | "
        f"{('Compare: vs ' + str(compare) + ' min ago | ') if int(compare) > 0 else ''}"
        f"DTE: {str(dte).replace('_','-').upper() if dte != 'all' else 'ALL'} | "
        f"Model: {pred_status}"
    )

    if df_agg.empty and df_snap.empty:
        return html.Div([
            html.H3("Waiting for data...", style={'color': MC['warning']}),
            html.P("Make sure the fetcher is running and data is being collected in daily_data/",
                   style={'color': MC['text_muted']})
        ]), status_html, live_badge, subheader

    content = []
    latest_stats = get_latest_stats(df_agg, df_snap)
    st = latest_stats.get(symbol, {}) if isinstance(latest_stats, dict) else {}

    # ── Ticker Ribbon (all symbols at a glance) ──
    ticker_ribbon = _mc_ticker_ribbon(latest_stats)
    if ticker_ribbon is not None:
        content.append(ticker_ribbon)

    if symbol != 'ALL' and symbol in latest_stats:
        st = latest_stats[symbol]
        price_color = MC['call'] if st['price_change'] >= 0 else MC['put']

        # ── Glassmorphism Stat Cards ──
        # Helper: format stat or show '--' only when truly unavailable (None/NaN).
        # Zeros are VALID data — the model was trained on a dataset where
        # ~50% of values are zero, and still achieves 72% accuracy.
        def _s(key, fmt="{:.2f}", prefix="", suffix=""):
            v = st.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "--"
            return f"{prefix}{fmt.format(v)}{suffix}"

        content.append(html.Div(
            style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'marginBottom': '14px'},
            children=[
                # Zero values are normal / valid — only show '--' for NaN/None
                _mc_metric_card('Price',
                f"${st['price']:,.0f}" if pd.notna(st.get('price')) else "--",
                                MC['text'],
                                sub=f"{st['price_change']:+.2f}%" if pd.notna(st.get('price_change')) else "--"),
                _mc_metric_card('P/C Ratio',
                                _s('pc_ratio', '{:.3f}'),
                                MC['put'] if st.get('pc_ratio', 0) > 1.0 else MC['call'],
                                sub='Bearish' if st.get('pc_ratio', 0) > 1.0 else ('Bullish' if st.get('pc_ratio', 0) > 0 else None)),
                _mc_metric_card('IV Skew',
                                _s('iv_skew', '{:.4f}'),
                                MC['text_sec']),
                _mc_metric_card('Net GEX',
                                (f"{st['net_gamma']/1e6:.1f}M" if abs(st.get('net_gamma', 0)) >= 1e6
                                 else f"{st.get('net_gamma', 0)/1e3:.1f}K")
                                if not (st.get('net_gamma') is None or (isinstance(st.get('net_gamma'), float) and np.isnan(st.get('net_gamma', 0)))) else "--",
                                MC['call'] if st.get('net_gamma', 0) >= 0 else MC['put'],
                                sub='Pos Gamma' if st.get('net_gamma', 0) > 0 else ('Neg Gamma' if st.get('net_gamma', 0) < 0 else None)),
                _mc_metric_card('ATM Straddle',
                                _s('atm_straddle', '${:.2f}'),
                                MC['warning']),
                _mc_metric_card('Call IV',
                                (f"{st['call_iv']*100:.1f}%" if st.get('call_iv') is not None and np.isfinite(st['call_iv']) else "--"),
                                MC['call']),
                _mc_metric_card('Put IV',
                                (f"{st['put_iv']*100:.1f}%" if st.get('put_iv') is not None and np.isfinite(st['put_iv']) else "--"),
                                MC['put']),
                _mc_metric_card('Aggression',
                                _s('trade_aggression', '{:+.3f}'),
                                MC['call'] if st.get('trade_aggression', 0) > 0 else MC['put'],
                                sub='Buyers' if st.get('trade_aggression', 0) > 0 else ('Sellers' if st.get('trade_aggression', 0) < 0 else None)),
            ]
        ))

        # ── Premium Flow Bar ──
        pf = _mc_premium_flow(st)
        if pf is not None:
            content.append(pf)

        # ── Market Regime Badge ──
        vix_lvl = model_out.get('vix_level', 0.0) if model_out else None
        content.append(_mc_regime_badge(st['net_gamma'], vix_level=vix_lvl))

        # ── Rule-Based Signal Indicator ──
        content.append(_mc_rule_signal(st))

    elif symbol == 'ALL':
        # ── Cross-symbol table ──
        tbl = _mc_cross_symbol_table(latest_stats)
        if tbl is not None:
            content.append(tbl)

    if symbol != 'ALL':
        # Spot price for charts
        spot_raw = 0.0
        if not df_agg.empty and "symbol" in df_agg.columns:
            sym_agg = df_agg[df_agg["symbol"] == symbol]
            if not sym_agg.empty and "spot" in sym_agg.columns:
                spot_raw = float(sym_agg.iloc[-1].get("spot", 0.0) or 0.0)

        # Uniform chart height for all charts
        UNIFORM_CHART_HEIGHT = "320px"

        # Helper: append chart + insight box in one shot
        def _append_chart(fig, height, what, detail, suggestion, anomaly=None, compact=True):
            if fig is None:
                return
            if height is None:
                fig_h = (fig.layout.height or 500)
                chart_style = {'height': f'{fig_h}px'}
            else:
                chart_style = {'height': height if height else UNIFORM_CHART_HEIGHT}
            content.append(dcc.Graph(figure=fig, style=chart_style))
            content.append(_chart_insight_box(what, detail, suggestion, anomaly))

        def _safe_insight(fn, *args):
            """Call insight function safely, return (detail, suggestion, anomaly)."""
            try:
                text, anomaly = fn(*args)
                return text or "", anomaly
            except Exception:
                return "", None

        # =================================================================
        # SIGNAL PANEL — pinned at top, before all groups
        # =================================================================
        if symbol != 'ALL' and model_out is not None:
            _prod_top = _create_model_production_panel(model_out, symbol, df_agg, pred_history_roll=pred_history_roll)
            if _prod_top is not None:
                content.append(_prod_top)
            _hud_top = _create_agent_hud_strip(model_out, symbol, df_agg)
            if _hud_top is not None:
                content.append(_hud_top)

        # =================================================================
        # SIGNAL DIVERGENCE CHART (Rule-Based vs Hybrid51 Ensemble)
        # =================================================================
        try:
            fig_div, _div_border = _create_signal_divergence_chart(df_agg, symbol, pred_df)
        except Exception as _ediv:
            import traceback as _tb
            print(f"[WARN] Signal divergence chart error: {_ediv}\n{_tb.format_exc()}")
            fig_div, _div_border = None, MC["border"]
        if fig_div is not None:
            content.append(html.Div(
                dcc.Graph(
                    figure=fig_div,
                    style={"height": "270px"},
                    config={"displayModeBar": False},
                ),
                style={
                    "border": f"1px solid {_div_border}",
                    "borderRadius": "10px",
                    "overflow": "hidden",
                    "backgroundColor": MC["bg_card"],
                    "marginBottom": "16px",
                },
            ))

        # =================================================================
        # PIPELINE MONITOR
        # =================================================================
        content.append(_create_pipeline_monitor())

        # =================================================================
        # GROUP 1: VOLUME & FLOW
        # Related: Call/Put volume, premium, cumulative delta, flow, MM flow
        # =================================================================
        content.append(_mc_group_header(
            "Volume & Flow",
            "Options volume, premium flow, and directional aggression"
        ))

        # Time-series metrics (P/C ratio, GEX, IV skew, volume, premium)
        try:
            ts_charts = create_timeseries_individual(df_agg, symbol, window_minutes=window)
        except Exception:
            ts_charts = []
        if ts_charts:
            for fig, box in ts_charts:
                content.append(dcc.Graph(figure=fig, style={'height': UNIFORM_CHART_HEIGHT}))
                ins = _insight(box)
                if ins is not None:
                    content.append(ins)

        # Cumulative volume delta
        try:
            fig_vol = create_cum_vol_delta_chart(df_agg, symbol, window_minutes=window)
        except Exception:
            fig_vol = None
        if fig_vol is not None:
            content.append(_mc_section_header("Cumulative Volume Delta"))
            txt, anom = _safe_insight(cum_vol_delta_insight, df_agg, symbol, window)
            _append_chart(fig_vol, '400px',
                "Cumulative Volume Delta: running sum of (call vol - put vol) over time.",
                txt,
                "Rising CVD = call-side aggression dominating. Falling = put-heavy flow. Use as confirmation for directional bias.",
                anom)

        # Options flow history
        try:
            fig_flow = create_flow_chart(df_agg, symbol)
        except Exception:
            fig_flow = None
        if fig_flow is not None:
            content.append(_mc_section_header("Options Flow History"))
            txt, anom = _safe_insight(flow_chart_insight, df_agg, symbol)
            _append_chart(fig_flow, '400px',
                "Options Flow: historical call vs put volume and premium by batch.",
                txt,
                "Look for sustained volume imbalance. Sudden spikes often precede large directional moves within 5-15 minutes.",
                anom)

        # Market maker flow changes
        try:
            fig_mm = create_mm_flow_chart(df_agg, [symbol])
        except Exception:
            fig_mm = None
        if fig_mm is not None:
            content.append(_mc_section_header("Market Maker Flow Changes"))
            txt, anom = _safe_insight(mm_flow_insight, df_agg, [symbol])
            _append_chart(fig_mm, '350px',
                "MM Flow: batch-over-batch change in market-maker hedging indicators.",
                txt,
                "Rapid MM flow shifts signal forced hedging. Monitor for gamma flip zones where dealers switch from long to short gamma.",
                anom)

        # =================================================================
        # GROUP 2: POSITIONING & GREEKS
        # Related: Gamma, strikes, OI walls, vanna, dealer greeks
        # =================================================================
        content.append(_mc_group_header(
            "Positioning & Greeks",
            "Strike-level gamma, delta, vanna, and open interest analysis"
        ))

        # Gamma exposure profile
        try:
            _now_et = datetime.now(ET)
            _open_m  = MARKET_OPEN_ET[0]  * 60 + MARKET_OPEN_ET[1]
            _close_m = MARKET_CLOSE_ET[0] * 60 + MARKET_CLOSE_ET[1]
            _cur_m   = _now_et.hour * 60 + _now_et.minute
            _market_open = _open_m <= _cur_m <= _close_m
            _snap_lookback = find_snapshot_by_time_offset(10) if _market_open else None
            _straddle_val  = float(st.get('atm_straddle', 0.0) or 0.0) if st else 0.0
            fig_gamma = create_gamma_chart(df_snap, symbol, spot_raw,
                                           lookback_df=_snap_lookback,
                                           atm_straddle=_straddle_val if _straddle_val > 0 else None)
        except Exception:
            fig_gamma = None
        if fig_gamma is not None:
            content.append(_mc_section_header("Gamma Exposure Profile"))
            txt, anom = _safe_insight(gamma_chart_insight, df_snap, symbol, spot_raw)
            _append_chart(fig_gamma, None,
                "Gamma Exposure (GEX): net dealer gamma at each strike. Green = dealers net long γ (mean-reversion). Red = net short γ (momentum amplifier). Orange = sign flipped vs 10-min ago. Purple = ≥20% change. White dots = previous 10-min slice.",
                txt,
                "Positive GEX = mean-reversion regime, sell spikes. Negative GEX = momentum regime, ride the trend. Key flip level is where GEX crosses zero. White dashed lines = straddle breakeven.",
                anom)

        # Key strike levels
        try:
            fig_strike = create_strike_chart(df_snap, symbol, spot_raw, lookback_df=None)
        except Exception:
            fig_strike = None
        if fig_strike is not None:
            content.append(_mc_section_header("Key Strike Levels"))
            txt, anom = _safe_insight(strike_chart_insight, df_snap, symbol, spot_raw)
            _append_chart(fig_strike, '500px',
                "Key Strikes: highest volume/OI concentrations by strike price, overlaid with spot.",
                txt,
                "Price tends to gravitate toward high-OI strikes near expiry (pin risk). Strikes with extreme volume are likely institutional targets.",
                anom)

        # OI walls & pinning
        try:
            fig_oi = create_oi_walls_chart(df_snap, symbol, spot_raw)
        except Exception:
            fig_oi = None
        if fig_oi is not None:
            content.append(_mc_section_header("OI Walls & Pinning"))
            txt, anom = _safe_insight(oi_walls_insight, df_snap, symbol)
            _append_chart(fig_oi, '400px',
                "OI Walls: open interest by strike showing call/put walls that act as support/resistance.",
                txt,
                "Large call OI walls act as resistance (dealers sell into rallies). Large put OI walls act as support (dealers buy dips). Watch for wall breaches.",
                anom)

        # Vanna exposure
        try:
            fig_vanna = create_vanna_chart(df_snap, symbol, spot_raw)
        except Exception:
            fig_vanna = None
        if fig_vanna is not None:
            content.append(_mc_section_header("Vanna Exposure"))
            txt, anom = _safe_insight(vanna_chart_insight, df_snap, symbol)
            _append_chart(fig_vanna, '400px',
                "Vanna Exposure: sensitivity of delta to changes in implied volatility, by strike.",
                txt,
                "Positive vanna + falling IV = bullish tailwind (dealers buy). Negative vanna + rising IV = selling pressure. Critical near large expiries.",
                anom)

        # Dealer positioning
        try:
            fig_dealer = create_dealer_chart(df_snap, symbol)
        except Exception:
            fig_dealer = None
        if fig_dealer is not None:
            content.append(_mc_section_header("Dealer Positioning"))
            txt, anom = _safe_insight(dealer_chart_insight, df_snap, symbol)
            _append_chart(fig_dealer, '400px',
                "Dealer Greeks: estimated market-maker delta, gamma, vega, and theta across strikes.",
                txt,
                "When dealers are short gamma, expect amplified moves. When long gamma, expect mean-reversion. Theta shows dealers' time-decay P&L.",
                anom)
            content.append(html.Div(
                "Footnote: Vega and Theta are computed from snapshot rows as OI-weighted net greeks (with current DTE filter applied).",
                style={
                    "fontSize": "11px",
                    "color": MC["text_muted"],
                    "margin": "-8px 0 14px 2px",
                    "fontStyle": "italic",
                }
            ))

        # =================================================================
        # GROUP 3: VOLATILITY
        # Related: IV term structure, vol/OI ratio, skew
        # =================================================================
        content.append(_mc_group_header(
            "Volatility",
            "Implied volatility structure, skew, and volume-weighted signals"
        ))

        # IV term structure
        try:
            fig_iv = create_iv_chart(df_snap, symbol)
        except Exception:
            fig_iv = None
        if fig_iv is not None:
            content.append(_mc_section_header("IV Term Structure"))
            txt, anom = _safe_insight(iv_chart_insight, df_snap, symbol)
            _append_chart(fig_iv, '400px',
                "IV Term Structure: implied volatility across expirations. Normal = upward slope (contango). Inverted = near-term fear.",
                txt,
                "Inverted term structure = market pricing an imminent event. Steep contango = complacency. Sell near-term premium in contango, buy protection when inverted.",
                anom)

        # Vol/OI ratio
        try:
            fig_vol_oi = create_vol_oi_chart(df_snap, symbol, spot_raw)
        except Exception:
            fig_vol_oi = None
        if fig_vol_oi is not None:
            content.append(_mc_section_header("Vol/OI Ratio (Live)"))
            txt, anom = _safe_insight(vol_oi_insight, df_snap, symbol)
            _append_chart(fig_vol_oi, '400px',
                "Vol/OI Ratio: today's volume relative to open interest at each strike. High ratio = new positioning.",
                txt,
                "Ratio > 1.0 at a strike = heavy new activity (opening positions). Cluster of high ratios near ATM = institutional re-positioning in progress.",
                anom)

        # =================================================================
        # GROUP 4: MICROSTRUCTURE
        # Related: spread, bid-ask imbalance, trade aggression, trade size
        # =================================================================
        content.append(_mc_group_header(
            "Microstructure",
            "Execution quality, bid-ask dynamics, and trade aggression"
        ))

        try:
            micro_charts = create_microstructure_individual(df_agg, symbol, window_minutes=window)
        except Exception:
            micro_charts = []
        if micro_charts:
            for fig, box in micro_charts:
                content.append(dcc.Graph(figure=fig, style={'height': UNIFORM_CHART_HEIGHT}))
                ins = _insight(box)
                if ins is not None:
                    content.append(ins)

        # =================================================================
        # GROUP 5: EXPIRATION & STRUCTURE
        # Related: DTE concentration, VIX hedging
        # =================================================================
        content.append(_mc_group_header(
            "Expiration & Structure",
            "DTE concentration, term structure effects, and VIX hedging"
        ))

        # Expiration concentration
        try:
            fig_dte = create_dte_concentration_chart(df_snap, symbol)
        except Exception:
            fig_dte = None
        if fig_dte is not None:
            content.append(_mc_section_header("Expiration Concentration"))
            txt, anom = _safe_insight(dte_concentration_insight, df_snap, symbol)
            _append_chart(fig_dte, '400px',
                "Expiration Concentration: volume/OI distribution across 0DTE, 1DTE, and 2-5DTE buckets.",
                txt,
                "High 0DTE concentration = intraday gamma-driven dynamics dominate. More multi-day weight = institutional positioning with lower decay urgency.",
                anom)

        # VIX hedging section
        if symbol in ("VIX", "VIXW"):
            try:
                fig_vix_flow = create_vix_put_flow_chart(df_agg)
            except Exception:
                fig_vix_flow = None
            if fig_vix_flow is not None:
                content.append(_mc_section_header("VIX Put Flow"))
                txt, anom = _safe_insight(vix_put_flow_insight, df_agg)
                _append_chart(fig_vix_flow, '350px',
                    "VIX Put Flow: put volume on VIX options, indicating tail-risk hedging activity.",
                    txt,
                    "Surging VIX put flow = institutions buying downside protection (bullish for VIX). Declining = hedge unwinding (risk-on sentiment).",
                    anom)

            try:
                fig_vix_hedge = create_vix_hedging_chart(df_snap)
            except Exception:
                fig_vix_hedge = None
            if fig_vix_hedge is not None:
                content.append(_mc_section_header("VIX Institutional Hedging"))
                txt, anom = _safe_insight(vix_hedging_insight, df_snap)
                _append_chart(fig_vix_hedge, '400px',
                    "VIX Institutional Hedging: large-size VIX option activity signaling institutional protection levels.",
                    txt,
                    "Watch for large OTM call purchases on VIX — this is the classic crash hedge. Elevated put selling = institutions betting on vol compression.",
                    anom)

        # =================================================================
        # GROUP 6: MODEL PREDICTION (EXPERIMENTAL)
        # Moved to bottom — model is under development, pending validation
        # =================================================================
        content.append(_mc_group_header(
            "Model Prediction (Experimental)",
            "Hybrid51 ensemble ML model — under development, pending live validation"
        ))

        if model_out is None:
            content.append(_prediction_unavailable_card())
        else:
            # 1. Model Production Panel (primary signal card)
            prod_panel = _create_model_production_panel(model_out, symbol, df_agg, pred_history_roll=pred_history_roll)
            if prod_panel is not None:
                content.append(_mc_section_header("Model Production"))
                content.append(prod_panel)
                content.append(_chart_insight_box(
                    "Model Production: Stage 3 output with per-agent bars (no gauge meters).",
                    "",
                    "Read direction + confidence first, then check agent bar agreement. Mixed bars suggest lower conviction."))

            # 2. HUD Strip (compact agent probabilities)
            hud_strip = _create_agent_hud_strip(model_out, symbol, df_agg)
            if hud_strip is not None:
                content.append(hud_strip)

            # 3. Decision Engine + Sizing Guidance side by side
            decision_panel = _create_decision_engine_panel(model_out, pred_history_roll)
            sizing_panel = _create_sizing_guidance(model_out, pred_history_roll, stats=st)
            content.append(html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                children=[decision_panel, sizing_panel],
            ))

            # 4. Price Expected Move Chart (candlestick + forward range)
            fig_em = _create_expected_move_chart(df_agg, symbol, model_out, pred_history_roll)
            if fig_em is not None:
                content.append(_mc_section_header("Price & Expected Move"))
                _append_chart(fig_em, '470px',
                    "Price candlestick with model-projected expected move cone based on ATM straddle.",
                    "",
                    "Cone shows where price could move in the next 90 min. Price breaking the cone edge suggests a volatility regime shift.",
                    compact=False)

            # 5. Model Rollover (from prediction.csv history)
            fig_roll = _create_model_rollover_chart(pred_history_roll)
            if fig_roll is not None:
                content.append(_mc_section_header("Model Rollover Prediction"))
                _append_chart(fig_roll, '360px',
                    "Model Rollover: Stage 3 probability, confidence, and signal strength over time.",
                    "",
                    "Watch for probability crossing 0.5 with rising confidence — signals a directional regime change. Flat confidence = indecisive market.",
                    compact=False)

            # 6. Accumulated Signal Chart
            fig_accum = _create_accumulated_prediction_chart(pred_history_roll)
            if fig_accum is not None:
                content.append(_mc_section_header("Accumulated Signal"))
                _append_chart(fig_accum, '340px',
                    "Accumulated Signal: cumulative sum of (prob - threshold) for Stage 3 over time; agent lines use (prob - 0.5).",
                    "",
                    "Divergence between agents and Stage 3 line signals disagreement. Converging upward = strong bullish conviction across the ensemble.",
                    compact=False)

            # 7. Model Health Panel
            health_panel = _create_model_health_panel(model_out, pred_history_roll)
            if health_panel is not None:
                content.append(_mc_section_header("Model Health"))
                content.append(health_panel)
                content.append(_chart_insight_box(
                    "Model Health: inference latency, feature completeness, warmup status, and history depth.",
                    "",
                    "Quality < 0.7 or missing features > 20 = model operating on degraded data. Reduce position sizing or wait for data quality to recover."))

            # 8. Alert Panel
            alert_panel = _create_alert_panel()
            content.append(_mc_section_header("Alerts & Notifications"))
            content.append(alert_panel)

    else:
        # Cross-symbol views when ALL is selected
        content.append(_mc_group_header(
            "Cross-Symbol Analysis",
            "Comparative gamma, sentiment, and VIX hedging across all tracked symbols"
        ))

        try:
            fig_gamma = create_multi_gamma_chart(df_agg)
        except Exception:
            fig_gamma = None
        if fig_gamma is not None:
            content.append(_mc_section_header("Cross-Symbol Gamma Comparison"))
            content.append(dcc.Graph(figure=fig_gamma, style={'height': '400px'}))
            content.append(_chart_insight_box(
                "Cross-Symbol Gamma: net GEX for each tracked symbol. Compare gamma regimes side by side.",
                "",
                "Symbols with negative gamma are in momentum mode. Positive gamma symbols tend to revert. Divergence between SPX and QQQ gamma signals a rotation."))

        try:
            fig_sent = create_multi_sentiment_chart(df_agg)
        except Exception:
            fig_sent = None
        if fig_sent is not None:
            content.append(_mc_section_header("Cross-Symbol Sentiment"))
            content.append(dcc.Graph(figure=fig_sent, style={'height': '400px'}))
            content.append(_chart_insight_box(
                "Cross-Symbol Sentiment: P/C ratio and flow direction for all symbols.",
                "",
                "Uniform bearish sentiment across all symbols = systemic risk-off. Divergence (e.g. SPX bearish but IWM bullish) = sector rotation."))

        has_vix = not df_agg.empty and "symbol" in df_agg.columns and any(
            s in ("VIX", "VIXW") for s in df_agg["symbol"].unique()
        )
        if has_vix:
            try:
                fig_vix = create_vix_put_flow_chart(df_agg)
            except Exception:
                fig_vix = None
            if fig_vix is not None:
                content.append(_mc_section_header("VIX Institutional Hedging"))
                content.append(dcc.Graph(figure=fig_vix, style={'height': '350px'}))
                content.append(_chart_insight_box(
                    "VIX Hedging: institutional protection activity on VIX options.",
                    "",
                    "Rising VIX put volume with falling equities = fear escalation. VIX call buying = institutions pricing in a vol spike ahead."))

    return html.Div(content), status_html, live_badge, subheader


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
