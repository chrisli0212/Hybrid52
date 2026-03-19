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

MAX_HISTORY    = 200

MC = {
    "bg_dark":       "#0a0a0f",
    "bg_card":       "#12121a",
    "bg_card_hover": "#1a1a28",
    "bg_input":      "#1e1e2e",
    "border":        "rgba(99,102,241,0.15)",
    "border_active": "rgba(99,102,241,0.4)",
    "text":          "#e2e8f0",
    "text_sec":      "#94a3b8",
    "text_muted":    "#64748b",
    "accent":        "#6366f1",
    "accent_glow":   "rgba(99,102,241,0.2)",
    "call":          "#22c55e",
    "put":           "#ef4444",
    "warning":       "#f59e0b",
    "info":          "#3b82f6",
    "grid":          "rgba(99,102,241,0.10)",
    "neutral":       "#6366f1",
}

# ══════════════════════════════════════════════════════════════════════════════
# INLINED FROM theta_dashboard_v3_10.py — ALL chart/data/insight functions
# ══════════════════════════════════════════════════════════════════════════════

# Market hours window (New York / Eastern Time)
# 30 min before open (9:00 AM) to 30 min after close (4:30 PM)
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
    open_minutes = MARKET_OPEN_ET[0] * 60 + MARKET_OPEN_ET[1]   # 540
    close_minutes = MARKET_CLOSE_ET[0] * 60 + MARKET_CLOSE_ET[1]  # 990
    mask = (time_minutes >= open_minutes) & (time_minutes <= close_minutes)
    filtered = df[mask].copy()
    return filtered if not filtered.empty else df  # fallback to unfiltered if nothing passes

# ========================================
# CONFIG - UPDATE THESE PATHS FOR YOUR SETUP
# ========================================

FETCHER_SCRIPT = Path("/workspace/theta_fetching_v5.py")
DATA_DIR = Path("/workspace/daily_data")
AGG_FILE = DATA_DIR / "theta_agg.csv"
SNAPSHOT_FILE = DATA_DIR / "theta_snapshot.csv"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
STATUS_FILE = DATA_DIR / ".fetcher_status"

REFRESH_INTERVAL = 10
MAX_HISTORY = 200

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


# ========================================
# DATA LOADERS
# ========================================

def load_agg_data():
    if not AGG_FILE.exists():
        return None
    try:
        df = pd.read_csv(AGG_FILE, on_bad_lines='skip')
        if df.empty:
            return None
        df['batch_id'] = df['batch_id'].astype(int)
        ts_clean = df['ts'].str.replace(r'\s+[A-Z]{2,5}$', '', regex=True)
        df['_ts_parsed'] = pd.to_datetime(ts_clean, errors='coerce')
        df = df.sort_values(['_ts_parsed', 'symbol'])
        # Filter to market hours only (9:00 AM - 4:30 PM ET)
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
                return df
        except Exception:
            pass
    # Fallback: load the latest snapshot from the snapshots directory
    snaps = list_available_snapshots()
    if snaps:
        try:
            df = pd.read_csv(snaps[-1][2])  # latest = last in sorted list
            return df if not df.empty else None
        except Exception:
            pass
    return None

def list_available_snapshots():
    """Scan snapshot directory and return sorted list of (batch_num, ts_str, filepath) tuples.
    Only includes snapshots within market hours (9:00 AM - 4:30 PM ET)."""
    if not SNAPSHOT_DIR.exists():
        return []
    files = sorted(SNAPSHOT_DIR.glob("snapshot_*.csv"))
    if not files:
        return []
    open_min = MARKET_OPEN_ET[0] * 60 + MARKET_OPEN_ET[1]   # 540
    close_min = MARKET_CLOSE_ET[0] * 60 + MARKET_CLOSE_ET[1]  # 990
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
    # Dealer GEX: calls positive (dealers long call gamma), puts negative (dealers short put gamma)
    if 'cp_sign' in sym.columns:
        sym['gamma_exp'] = sym['gamma_exp'] * sym['cp_sign']
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
        sym['vanna_est'] = sym['gamma'] * sym['implied_vol'] * sym['oi'] * sym['cp_sign']
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
        if col in sym.columns and 'oi' in sym.columns and 'cp_sign' in sym.columns:
            greeks[col.title()] = (sym[col] * sym['oi'] * sym['cp_sign']).sum()
    return greeks if greeks else None

def calculate_metrics(snapshot_df, agg_df, symbol):
    result = {
        'spot': 'N/A', 'net_gex': 'N/A', 'pc_ratio': 'N/A',
        'iv_rank': 'N/A', 'call_premium': 'N/A', 'put_premium': 'N/A',
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
            result['call_premium'] = f"{sym[sym['cp_sign']==1]['volume'].sum():,.0f}"
            result['put_premium'] = f"{sym[sym['cp_sign']==-1]['volume'].sum():,.0f}"
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

def empty_chart(msg, height=350):
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
    for i in range(len(flip_candidates) - 1):
        if flip_candidates.iloc[i]['gamma_exp'] * flip_candidates.iloc[i+1]['gamma_exp'] < 0:
            flip_strike = (flip_candidates.iloc[i]['strike'] + flip_candidates.iloc[i+1]['strike']) / 2
            break

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
        return None, None
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
        return None, None
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
            "<em>Theta</em>: time decay — positive = dealers collect premium as time passes.")

    anomaly = None
    gamma_val = greeks.get('Gamma', 0)
    if gamma_val < 0:
        text += "<br>Dealers have <b>negative net gamma</b> — they are chasing price, amplifying moves."
        anomaly = 'warn'
    return text, anomaly


def multi_gamma_insight(agg_df):
    if agg_df is None or agg_df.empty:
        return None, None
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
        return None, None
    text = ("<b>Cross-Symbol Sentiment:</b> "
            "Compares call vs put volume (or P/C ratio) across symbols. "
            "Uniform bullish/bearish readings suggest broad consensus. "
            "Divergence (e.g., calls on SPX, puts on IWM) suggests rotation or hedged positioning.")
    return text, None


def vix_put_flow_insight(agg_df):
    if agg_df is None or agg_df.empty:
        return None, None
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

def create_gamma_chart(options_df, symbol, spot, lookback_df=None):
    gex = calculate_gex_by_strike(options_df, symbol)
    if gex is None or gex.empty:
        return empty_chart("No GEX data - waiting for snapshot", 400)
    if spot and spot > 0:
        lo, hi = spot * 0.90, spot * 1.10
        gex = gex[(gex['strike'] >= lo) & (gex['strike'] <= hi)]
        if gex.empty:
            return empty_chart("No GEX data in ±10% range", 400)
    colors = [C['call'] if v > 0 else C['put'] for v in gex['gamma_exp']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=gex['strike'], y=gex['gamma_exp'],
        marker=dict(color=colors), name='Current GEX',
        hovertemplate='Strike: %{x}<br>GEX: %{y:,.0f}<extra></extra>'))
    if lookback_df is not None:
        gex_lb = calculate_gex_by_strike(lookback_df, symbol)
        if gex_lb is not None and not gex_lb.empty:
            fig.add_trace(go.Scatter(x=gex_lb['strike'], y=gex_lb['gamma_exp'],
                mode='lines', line=dict(color=C['warning'], width=2, dash='dot'),
                name='Lookback GEX', opacity=0.7))
    if spot and spot > 0:
        fig.add_vline(x=spot, line_dash="dash", line_color=C['warning'],
                      line_width=2, annotation_text=f"Spot ${spot:.0f}")
    fig.update_layout(**base_layout(height=400, xaxis_title="Strike Price",
                                     yaxis_title="Gamma Exposure ($)"))
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
    fig.update_layout(**base_layout(height=400, barmode='overlay',
        xaxis_title="Volume", yaxis_title="Strike", hovermode='y unified'))
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
    fig.update_layout(**base_layout(height=400,
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
    chart_height = 500 if has_premium else 350
    fig.update_xaxes(gridcolor=C['grid'], showgrid=True, type='date', tickformat='%H:%M')
    fig.update_layout(**base_layout(height=chart_height, xaxis_title="Time"))
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
    fig.update_layout(**base_layout(height=400, xaxis_title="Time", xaxis_type="date", xaxis_tickformat="%H:%M",
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
    fig.update_layout(**base_layout(height=400, showlegend=False,
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
    fig.update_layout(**base_layout(height=400, barmode='group',
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
    layout_kw = base_layout(height=400, xaxis_title="Time", xaxis_type="date",
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
    fig.update_layout(**base_layout(height=400))
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
    fig.update_layout(**base_layout(height=400,
        xaxis_title="Strike Price", yaxis_title="Vanna"))
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
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        'Directional (Delta · Gamma)', 'Vol & Time (Vega · Theta)'],
        horizontal_spacing=0.15)
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
    fig.update_layout(**base_layout(height=380, showlegend=False))
    fig.update_yaxes(title_text="Net Position", row=1, col=1, gridcolor=C['grid'])
    fig.update_yaxes(title_text="Net Position", row=1, col=2, gridcolor=C['grid'])
    # Style subplot title text
    for ann in fig.layout.annotations:
        ann.font.color = C['text']
        ann.font.size = 12
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
    fig.update_layout(**base_layout(height=350, barmode='group',
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
    for i in range(len(gex) - 1):
        if gex.iloc[i]['gamma_exp'] * gex.iloc[i+1]['gamma_exp'] < 0:
            flip_strike = (gex.iloc[i]['strike'] + gex.iloc[i+1]['strike']) / 2
            if spot > 0 and abs(flip_strike - spot) / spot < 0.10:
                break  # prefer flip near spot
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
                     fill=False, hline=None, height=280):
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
            fig2.update_layout(height=280, showlegend=True,
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
        fig.update_layout(height=480, paper_bgcolor=C['bg_dark'], plot_bgcolor=C['bg_card'],
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
    layout_kw = base_layout(height=350)
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
        return None, None
    sym_df = _filter_by_window(agg_df, symbol, window_minutes)
    if sym_df.empty:
        return None, None
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
_cached_snap_mtime = None
_cached_filtered_data = {}
_cached_pred_df = None
_cached_pred_mtime = None

# Alert system
_alerts_log = deque(maxlen=50)
_last_alert_state = {}

# Track last non-suppressed prediction time for countdown
_last_live_non_suppressed_ts = None


# ---------------------------------------------------------------------------
# Prediction CSV reading
# ---------------------------------------------------------------------------

def _load_prediction_csv():
    """Load prediction.csv with mtime caching. Returns DataFrame or empty DataFrame."""
    global _cached_pred_df, _cached_pred_mtime
    pred_path = DATA_DIR / "prediction.csv"
    if not pred_path.exists():
        _cached_pred_df = pd.DataFrame()
        _cached_pred_mtime = None
        return _cached_pred_df
    try:
        current_mtime = pred_path.stat().st_mtime
        if _cached_pred_df is not None and current_mtime == _cached_pred_mtime:
            return _cached_pred_df
        df = pd.read_csv(pred_path)
        _cached_pred_df = df
        _cached_pred_mtime = current_mtime
        return df
    except Exception:
        _cached_pred_df = pd.DataFrame()
        _cached_pred_mtime = None
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

    suppressed = str(row_dict.get("suppressed", "False")).strip().lower() in ("true", "1", "yes")
    prob = float(row_dict.get("prob", 0.5) or 0.5)
    confidence = float(row_dict.get("confidence", 0.0) or 0.0)
    signal_strength = float(row_dict.get("signal_strength", 0.0) or 0.0)
    # Evidence-based confidence decomposition (new columns)
    agent_std = float(row_dict.get("agent_std", 0.0) or 0.0)
    consensus_ratio = float(row_dict.get("consensus_ratio", 0.0) or 0.0)
    conf_agreement = float(row_dict.get("conf_agreement", 0.0) or 0.0)
    conf_consensus = float(row_dict.get("conf_consensus", 0.0) or 0.0)
    conf_gate_conviction = float(row_dict.get("conf_gate_conviction", 0.0) or 0.0)
    conf_data_quality = float(row_dict.get("conf_data_quality", 0.0) or 0.0)
    pred = int(float(row_dict.get("pred", 0) or 0))
    threshold = float(row_dict.get("threshold", 0.47) or 0.47)
    reason = str(row_dict.get("reason", "") or "")
    direction = str(row_dict.get("direction", "BULL" if pred == 1 else "BEAR") or "")

    # Build stage2_probs dict from flat agent columns
    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]
    stage2_probs = {}
    for k in agent_keys:
        col = f"agent_{k}_prob"
        stage2_probs[k] = float(row_dict.get(col, 0.5) or 0.5)

    # Build gates dict
    gates = {}
    for k in agent_keys:
        col = f"gate_{k}"
        gates[k] = float(row_dict.get(col, 1.0) or 1.0)

    # Diagnostics sub-dict
    quality_score = float(row_dict.get("quality_score", 0.0) or 0.0)
    feature_completeness = float(row_dict.get("feature_completeness", 0.0) or 0.0)
    warmup_fraction = float(row_dict.get("warmup_fraction", 0.0) or 0.0)
    latency_ms = row_dict.get("latency_ms", None)
    if latency_ms is not None:
        try:
            latency_ms = float(latency_ms)
        except (ValueError, TypeError):
            latency_ms = None
    stage1_missing_count = int(float(row_dict.get("stage1_missing_count", 0) or 0))
    vix_level = float(row_dict.get("vix_level", 0.0) or 0.0)
    spot_price = float(row_dict.get("spot_price", 0.0) or 0.0)

    # Determine vix_valid heuristic: vix level present and reasonable
    vix_valid = 5.0 < vix_level < 80.0

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
        try:
            ts_dt = pd.to_datetime(ts_raw)
        except Exception:
            ts_dt = _now_et_naive()

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
    direction = "UP" if prob >= 0.5 else "DOWN"

    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]
    up_count = sum(1 for k in agent_keys if float(stage2_probs.get(k, 0.5)) >= 0.5)

    prev = _last_alert_state

    # Signal flip
    prev_dir = prev.get("direction")
    if prev_dir is not None and prev_dir != direction and not suppressed and ok:
        _alerts_log.append({
            "ts": now_str, "severity": "high",
            "msg": f"Signal FLIPPED: {prev_dir} -> {direction} (P(up)={prob:.1%})"
        })

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

    direction = "UP" if prob >= 0.5 else "DOWN"
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
            persistent = all(h["prob"] >= 0.5 for h in recent)
        else:
            persistent = all(h["prob"] < 0.5 for h in recent)
        persist_label = f"{sum(1 for h in recent if (h['prob'] >= 0.5) == (direction == 'UP'))}/3 bars {direction}"
    else:
        persistent = False
        persist_label = f"{len(recent)}/3 bars (need 3)"

    # Agent consensus
    if direction == "UP":
        consensus_count = sum(1 for k in agent_keys if float(stage2_probs.get(k, 0.5)) >= 0.5)
    else:
        consensus_count = sum(1 for k in agent_keys if float(stage2_probs.get(k, 0.5)) < 0.5)
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

    direction = "UP" if prob >= 0.5 else "DOWN"
    dir_color = MC["call"] if prob >= 0.5 else MC["put"]

    if suppressed or not ok:
        dir_badge = html.Span("PAUSED SUPPRESSED", style={"color": MC["text_muted"], "fontWeight": 700})
    else:
        dir_icon = "[+]" if prob >= 0.5 else "[-]"
        dir_badge = html.Span(f"{dir_icon} {direction} {prob:.0%}", style={"color": dir_color, "fontWeight": 700})

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

    # Agent votes
    agent_keys = ["A", "B", "C", "K", "T", "Q", "2D"]
    agent_votes = []
    up_count = 0
    for k in agent_keys:
        val = float(stage2_probs.get(k, 0.5))
        is_up = val >= 0.5
        if is_up:
            up_count += 1
        arrow = "^" if is_up else "v"
        color = MC["call"] if is_up else MC["put"]
        if suppressed or not ok:
            color = MC["text_muted"]
        agent_votes.append(html.Span(f"{k}{arrow}", style={
            "color": color, "fontWeight": 600, "fontSize": "12px", "marginRight": "6px"
        }))

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
                if (spot_move >= 0 and prob >= 0.5) or (spot_move < 0 and prob < 0.5):
                    tracking_text = "OK"
                    tracking_color = MC["call"]
                else:
                    tracking_text = "X"
                    tracking_color = MC["put"]

    return html.Div(
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderRadius": "8px", "padding": "10px 16px",
            "marginBottom": "12px",
        },
        children=[
            # Row 1
            html.Div(
                style={"display": "flex", "gap": "16px", "alignItems": "center", "flexWrap": "wrap"},
                children=[
                    html.Span(f"{symbol} {spot_text}", style={
                        "fontSize": "14px", "fontWeight": 700, "color": MC["text"],
                    }),
                    html.Span("|", style={"color": MC["border"]}),
                    dir_badge,
                    html.Span("|", style={"color": MC["border"]}),
                    html.Span("conf: ", style={"color": MC["text_muted"], "fontSize": "12px"}),
                    html.Span(f"{confidence:.0%} {conf_label}", style={
                        "color": conf_badge_color, "fontSize": "12px", "fontWeight": 600,
                        "backgroundColor": _hex_to_rgba(conf_badge_color, 0.13),
                        "padding": "1px 6px", "borderRadius": "4px",
                    }),
                    html.Span("|", style={"color": MC["border"]}),
                    html.Span("regime: ", style={"color": MC["text_muted"], "fontSize": "12px"}),
                    html.Span(regime_text, style={"color": regime_color, "fontSize": "12px", "fontWeight": 600}),
                    html.Span("|", style={"color": MC["border"]}),
                    html.Span(countdown_text, style={"color": MC["text_muted"], "fontSize": "12px"}),
                ]
            ),
            # Row 2
            html.Div(
                style={"display": "flex", "gap": "8px", "alignItems": "center", "marginTop": "6px", "flexWrap": "wrap"},
                children=[
                    *agent_votes,
                    html.Span("|", style={"color": MC["border"]}),
                    html.Span(f"consensus: {up_count}/7", style={
                        "color": MC["text_sec"], "fontSize": "12px",
                    }),
                    html.Span("|", style={"color": MC["border"]}),
                    html.Span("tracking: ", style={"color": MC["text_muted"], "fontSize": "12px"}),
                    html.Span(tracking_text, style={"color": tracking_color, "fontSize": "12px", "fontWeight": 700}),
                ]
            ),
            # Row 3: Confidence decomposition (evidence-based)
            _build_confidence_decomposition_row(model_out, MC),
        ]
    )


# ---------------------------------------------------------------------------
# Position Sizing Guidance
# ---------------------------------------------------------------------------

def _create_sizing_guidance(model_out, pred_history_roll):
    """Position sizing table based on confidence x regime."""
    if not model_out:
        return html.Div(
            style={
                "backgroundColor": MC["bg_card"],
                "border": f"1px solid {MC['border']}",
                "borderRadius": "8px", "padding": "16px", "flex": "1",
            },
            children=[
                html.Div("POSITION SIZING", style={"fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px", "color": MC["accent"]}),
                html.Div("Awaiting model data...", style={"color": MC["text_muted"], "fontSize": "13px", "marginTop": "10px"}),
            ]
        )

    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    confidence = float(model_out.get("confidence", 0.0) or 0.0)
    diagnostics = model_out.get("diagnostics", {}) or {}
    vix_valid = bool(diagnostics.get("vix_valid", False))
    quality = float(diagnostics.get("quality_score", 0.0) or 0.0)
    low_vol = vix_valid and quality > 0.5

    tiers = [
        ("< 55%", "0x", "0x"),
        ("55-65%", "0.5x", "0x"),
        ("65-75%", "1.0x", "0.5x"),
        ("> 75%", "1.25x", "0.75x"),
    ]

    if suppressed or not ok or confidence < 0.55:
        current_mult = "0x"
        current_tier = 0
    elif confidence < 0.65:
        current_mult = "0.5x" if low_vol else "0x"
        current_tier = 1
    elif confidence < 0.75:
        current_mult = "1.0x" if low_vol else "0.5x"
        current_tier = 2
    else:
        current_mult = "1.25x" if low_vol else "0.75x"
        current_tier = 3

    non_supp_count = sum(1 for h in pred_history_roll if not h["suppressed"])

    table_rows = []
    for i, (tier_label, lv, hv) in enumerate(tiers):
        is_active = (i == current_tier)
        bg = _hex_to_rgba(MC["accent"], 0.09) if is_active else "transparent"
        table_rows.append(html.Tr(style={"backgroundColor": bg}, children=[
            html.Td(tier_label, style={"padding": "4px 8px", "fontSize": "12px", "color": MC["text"], "fontWeight": 700 if is_active else 400}),
            html.Td(lv, style={"padding": "4px 8px", "fontSize": "12px", "color": MC["call"], "textAlign": "center"}),
            html.Td(hv, style={"padding": "4px 8px", "fontSize": "12px", "color": MC["warning"], "textAlign": "center"}),
        ]))

    return html.Div(
        style={
            "backgroundColor": MC["bg_card"],
            "border": f"1px solid {MC['border']}",
            "borderRadius": "8px", "padding": "16px", "flex": "1",
        },
        children=[
            html.Div("POSITION SIZING", style={
                "fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px",
                "color": MC["accent"], "marginBottom": "10px",
                "borderBottom": f"1px solid {MC['border']}", "paddingBottom": "6px",
            }),
            html.Div(style={"textAlign": "center", "marginBottom": "12px"}, children=[
                html.Div(current_mult, style={
                    "fontSize": "32px", "fontWeight": 800,
                    "color": MC["call"] if current_mult not in ("0x",) else MC["text_muted"],
                }),
                html.Div("Recommended Size", style={"fontSize": "11px", "color": MC["text_muted"]}),
            ]),
            html.Table(
                style={"width": "100%", "borderCollapse": "collapse"},
                children=[
                    html.Thead(html.Tr(children=[
                        html.Th("Confidence", style={"padding": "4px 8px", "fontSize": "11px", "color": MC["text_muted"], "textAlign": "left", "borderBottom": f"1px solid {MC['border']}"}),
                        html.Th("Low Vol", style={"padding": "4px 8px", "fontSize": "11px", "color": MC["text_muted"], "textAlign": "center", "borderBottom": f"1px solid {MC['border']}"}),
                        html.Th("High Vol", style={"padding": "4px 8px", "fontSize": "11px", "color": MC["text_muted"], "textAlign": "center", "borderBottom": f"1px solid {MC['border']}"}),
                    ])),
                    html.Tbody(table_rows),
                ]
            ),
            html.Div(style={
                "borderTop": f"1px solid {MC['border']}",
                "marginTop": "10px", "paddingTop": "8px",
            }, children=[
                html.Div("CIRCUIT BREAKER", style={"fontSize": "10px", "fontWeight": 700, "color": MC["text_muted"], "letterSpacing": "0.5px"}),
                html.Div(f"Signals today: {non_supp_count}", style={"fontSize": "12px", "color": MC["text_sec"], "marginTop": "3px"}),
                html.Div(f"Regime: {'Low Vol' if low_vol else 'High Vol / Stressed'}", style={
                    "fontSize": "12px",
                    "color": MC["call"] if low_vol else MC["warning"],
                    "marginTop": "2px",
                }),
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

    # Recent accuracy: compare prob direction vs subsequent prob movement
    correct = 0
    total = 0
    non_supp = [h for h in hist if not h["suppressed"]]
    for i in range(len(non_supp) - 1):
        pred_up = non_supp[i]["prob"] >= 0.5
        next_up = non_supp[i + 1]["prob"] >= 0.5
        if pred_up and non_supp[i + 1]["prob"] >= non_supp[i]["prob"]:
            correct += 1
        elif not pred_up and non_supp[i + 1]["prob"] <= non_supp[i]["prob"]:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0.0
    accuracy_text = f"{accuracy:.0%}" if total > 0 else "N/A"
    accuracy_sub = f"({correct}/{total} calls)"

    # Sparkline data from history
    quality_hist = [h.get("confidence", 0.0) for h in hist[-30:] if not h["suppressed"]]

    def _mini_sparkline(values, color, height=35, width=120):
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
            height=height, width=width,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": f"{height}px", "width": f"{width}px"})

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

    acc_color = MC["call"] if accuracy >= 0.55 else (MC["warning"] if accuracy >= 0.45 else MC["put"])
    qual_color = MC["call"] if quality >= 0.6 else (MC["warning"] if quality >= 0.4 else MC["put"])
    comp_color = MC["call"] if completeness >= 0.8 else (MC["warning"] if completeness >= 0.5 else MC["put"])
    lat_color = MC["call"] if (latency is not None and latency < 500) else MC["warning"]

    cards = [
        _health_card("Recent Accuracy", accuracy_text, accuracy_sub,
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
                    "gridTemplateColumns": "1fr 1fr 1fr",
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
    confidence = float(model_out.get("confidence", 0.0) or 0.0)  # No fake fallback
    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    source_state = str(model_out.get("source_state", "UNKNOWN") or "UNKNOWN")
    neutral_mode = suppressed or (not ok)
    bar_color = MC["text_muted"] if neutral_mode else (MC["call"] if stage3_prob >= 0.5 else MC["put"])
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

    s3_arrow = "^" if stage3_prob >= 0.5 else "v"
    s3_delta_text = f"{stage3_prob - 0.5:+.2f}"

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=stage3_prob,
            number={"valueformat": ".2f", "font": {"size": 54}},
            delta={"reference": 0.5, "valueformat": ".2f", "increasing": {"color": MC["call"]}, "decreasing": {"color": MC["put"]}},
            title={"text": f"Stage 3 {s3_arrow} ({source_state})  |  Confidence {confidence*100:.0f}%  |  d0.5: {s3_delta_text}", "font": {"size": 16, "color": MC["text"]}},
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
                "threshold": {"line": {"color": MC["accent"], "width": 3}, "thickness": 0.8, "value": 0.5},
            },
        ),
        row=1,
        col=1,
    )

    for idx, (label, key) in enumerate(agents):
        val = float(stage2_probs.get(key, 0.5))
        row = 2 if idx < 4 else 3
        col = (idx % 4) + 1
        is_up = val >= 0.5
        arrow = "^" if is_up else "v"
        delta_from_half = val - 0.5
        delta_text = f"{delta_from_half:+.2f}"

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
                        {"range": [0, 0.45], "color": ("rgba(148,163,184,0.10)" if neutral_mode else "rgba(239,68,68,0.16)")},
                        {"range": [0.45, 0.55], "color": "rgba(148,163,184,0.12)"},
                        {"range": [0.55, 1.0], "color": ("rgba(148,163,184,0.10)" if neutral_mode else "rgba(16,185,129,0.16)")},
                    ],
                    "threshold": {"line": {"color": MC["accent"], "width": 2}, "thickness": 0.7, "value": 0.5},
                },
            ),
            row=row,
            col=col,
        )

    layout_cfg = base_layout(title="Model Signal Meters", height=760)
    layout_cfg["margin"] = dict(l=30, r=30, t=70, b=30)
    fig.update_layout(**layout_cfg)

    return fig


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
        val = float(stage2_probs.get(k, 0.5))
        confidence_from_half = abs(val - 0.5)
        if val >= 0.5:
            up_agents.append((k, confidence_from_half))
        else:
            down_agents.append((k, confidence_from_half))

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
    """Price + Expected Move chart: candlesticks (left) + widening cone (right).
    Cone uses full ATM straddle scaled over a 90-min horizon so the band is
    prominently visible. Model bias thickens/brightens the favoured side.
    """
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
        avg_iv  = (call_iv + put_iv) / 200.0
        base_em = spot * avg_iv * np.sqrt(1.0 / 252.0)
    else:
        tail    = sdf.tail(min(60, len(sdf)))
        base_em = max(float(tail["spot"].max() - tail["spot"].min()) * 2.0, spot * 0.003)

    # ── WIDER CONE: 90-min horizon ──
    horizon_min = 90
    em_horizon  = base_em * np.sqrt(horizon_min / 390.0)

    suppressed     = bool((model_out or {}).get("suppressed", False))
    ok             = bool((model_out or {}).get("ok", True))
    has_prediction = model_out is not None and not suppressed and ok
    pup            = float((model_out or {}).get("prob",       0.5) or 0.5) if has_prediction else 0.5
    confidence     = float((model_out or {}).get("confidence", 0.0) or 0.0) if has_prediction else 0.0
    stronger_up    = pup > 0.5

    bias_factor = (0.5 + 0.22 * confidence * (1.0 if stronger_up else -1.0)) if has_prediction else 0.5
    up_share    = bias_factor
    dn_share    = 1.0 - bias_factor

    future_x = [now_ts + pd.Timedelta(minutes=i) for i in range(horizon_min + 1)]
    t_frac   = np.linspace(0, 1, horizon_min + 1)
    em_env   = em_horizon * np.sqrt(t_frac)
    up_path  = spot + em_env * up_share * 2
    dn_path  = spot - em_env * dn_share * 2
    mid_path = spot + em_env * (up_share - dn_share)

    base_width      = 2.0
    favored_width   = base_width + confidence * 3.5
    unfavored_width = max(1.0, base_width - confidence * 0.8)
    favored_alpha   = 0.55 + confidence * 0.40
    unfavored_alpha = max(0.25, 0.55 - confidence * 0.25)
    up_width  = favored_width   if stronger_up else unfavored_width
    dn_width  = unfavored_width if stronger_up else favored_width
    up_alpha  = favored_alpha   if stronger_up else unfavored_alpha
    dn_alpha  = unfavored_alpha if stronger_up else favored_alpha
    band_alpha = 0.14 + confidence * 0.18

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]])

    fig.add_trace(go.Candlestick(
        x=ohlc["ts_bucket"],
        open=ohlc["open"], high=ohlc["high"], low=ohlc["low"], close=ohlc["close"],
        name="Price",
        increasing_line_color=MC["call"], decreasing_line_color=MC["put"],
        increasing_fillcolor=MC["call"],  decreasing_fillcolor=MC["put"],
        whiskerwidth=0.5, opacity=0.9,
    ))

    if has_prediction and stronger_up:
        band_fill = f"rgba(34,197,94,{band_alpha:.2f})"
    elif has_prediction:
        band_fill = f"rgba(239,68,68,{band_alpha:.2f})"
    else:
        band_fill = f"rgba(148,163,184,{band_alpha:.2f})"

    fig.add_trace(go.Scatter(
        x=list(future_x) + list(future_x)[::-1],
        y=list(up_path)  + list(dn_path)[::-1],
        fill="toself", mode="lines", name="Expected Range",
        line=dict(color="rgba(148,163,184,0.10)", width=0),
        fillcolor=band_fill, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=future_x, y=list(up_path), mode="lines",
        name=f"+{em_horizon:.2f} upper",
        line=dict(color=f"rgba(34,197,94,{up_alpha:.2f})", width=up_width),
        hovertemplate="%{y:.2f}",
    ))
    fig.add_trace(go.Scatter(
        x=future_x, y=list(dn_path), mode="lines",
        name=f"-{em_horizon:.2f} lower",
        line=dict(color=f"rgba(239,68,68,{dn_alpha:.2f})", width=dn_width),
        hovertemplate="%{y:.2f}",
    ))
    if has_prediction and confidence > 0.3:
        mid_alpha = 0.30 + confidence * 0.45
        mid_color = f"rgba(34,197,94,{mid_alpha:.2f})" if stronger_up else f"rgba(239,68,68,{mid_alpha:.2f})"
        fig.add_trace(go.Scatter(
            x=future_x, y=list(mid_path), mode="lines",
            name=f"Model midline ({'BULL' if stronger_up else 'BEAR'}) {confidence:.0%}",
            line=dict(color=mid_color, width=1.8, dash="dash"),
            hovertemplate="%{y:.2f}",
        ))

    y_low  = float(min(ohlc["low"].min(),  np.nanmin(dn_path)))
    y_high = float(max(ohlc["high"].max(), np.nanmax(up_path)))
    y_pad  = (y_high - y_low) * 0.04
    y_low -= y_pad;  y_high += y_pad

    fig.add_shape(type="line", x0=now_ts, x1=now_ts, y0=y_low, y1=y_high,
                  line=dict(dash="dot", color=MC["accent"], width=1.5))
    fig.add_annotation(x=now_ts, y=y_high, text="NOW", showarrow=False,
                       xanchor="left", yanchor="bottom",
                       font=dict(color=MC["accent"], size=10, family="monospace"),
                       bgcolor="rgba(15,23,42,0.80)", borderpad=2)
    fig.add_hline(y=spot, line_dash="dash", line_color=MC["text_muted"], line_width=0.8,
                  annotation_text=f"{spot:.2f}", annotation_position="right",
                  annotation_font=dict(color=MC["text"], size=11))

    if has_prediction and confidence > 0.2:
        arrow_text  = f"{'▲' if stronger_up else '▼'} {'BULL' if stronger_up else 'BEAR'} {confidence:.0%}"
        arrow_color = MC["call"] if stronger_up else MC["put"]
        arrow_y     = up_path[-1] if stronger_up else dn_path[-1]
        fig.add_annotation(x=future_x[-1], y=arrow_y, text=arrow_text,
                           showarrow=False, xanchor="right", yanchor="middle",
                           font=dict(color=arrow_color, size=12, family="monospace"),
                           bgcolor="rgba(15,23,42,0.85)",
                           bordercolor=arrow_color, borderwidth=1, borderpad=4)

    em_pct      = em_horizon / spot * 100.0 if spot > 0 else 0.0
    range_label = f"90m EM ±{em_horizon:.2f} ({em_pct:.2f}%)"
    if have_straddle:
        range_label += f"  Straddle ${atm_straddle:.2f}"
    if np.isfinite(pc_ratio) and pc_ratio > 0:
        range_label += f"  P/C {pc_ratio:.2f}"
    fig.add_annotation(x=0.5, y=1.0, xref="paper", yref="paper",
                       text=range_label, showarrow=False, xanchor="center", yanchor="bottom",
                       font=dict(color=MC["text_muted"], size=10))

    tracking_status = "--";  tracking_color = MC["text_muted"]
    if has_prediction and len(sdf) >= 5:
        recent_spots = sdf["spot"].tail(10)
        if len(recent_spots) >= 2:
            spot_delta = float(recent_spots.iloc[-1] - recent_spots.iloc[0])
            if (pup > 0.5) == (spot_delta > 0):
                tracking_status = "ON PATH";      tracking_color = MC["call"]
            elif abs(spot_delta) < em_horizon * 0.15:
                tracking_status = "NEUTRAL";      tracking_color = MC["warning"]
            elif abs(spot_delta) > em_horizon * 0.7:
                tracking_status = "INVALIDATED";  tracking_color = MC["put"]
            else:
                tracking_status = "DIVERGING";    tracking_color = MC["warning"]
    if tracking_status != "--":
        fig.add_annotation(
            x=ohlc["ts_bucket"].iloc[len(ohlc) // 2] if len(ohlc) > 0 else now_ts,
            y=y_high, text=tracking_status, showarrow=False,
            xanchor="center", yanchor="top",
            font=dict(color=tracking_color, size=12, family="monospace"),
            bgcolor="rgba(15,23,42,0.80)",
            bordercolor=tracking_color, borderwidth=1, borderpad=3)

    layout_cfg = base_layout(title=f"{symbol} Price & Expected Move", height=470)
    layout_cfg.update({"yaxis": dict(range=[y_low, y_high]),
                        "xaxis": dict(rangeslider={"visible": False}),
                        "hovermode": "x unified"})
    fig.update_layout(layout_cfg)
    return style_axes(fig)


def _create_accumulated_prediction_chart(pred_history_roll):
    """Running cumulative sum of (prob - 0.5) for Stage 3 and all 7 agents.
    Stage 3 = thick solid line with area fill.
    Agents   = thin semi-transparent lines (visually subordinate).
    """
    if len(pred_history_roll) < 2:
        return None

    hist = pred_history_roll
    x    = [h["ts"] for h in hist]

    # Stage 3 cumulative
    s3_deltas = [(h["prob"] - 0.5) if not h.get("suppressed", True) else 0.0 for h in hist]
    s3_cum    = list(np.cumsum(s3_deltas))

    # Per-agent cumulative
    agent_map = [
        ("A",  "agent_A_prob"),
        ("B",  "agent_B_prob"),
        ("C",  "agent_C_prob"),
        ("K",  "agent_K_prob"),
        ("T",  "agent_T_prob"),
        ("Q",  "agent_Q_prob"),
        ("2D", "agent_2D_prob"),
    ]

    fig = go.Figure()

    # Zero reference
    fig.add_hline(y=0, line_dash="dot",
                  line_color=MC["text_muted"], line_width=0.8)

    # Agent lines — thin, muted grey, low opacity
    for label, key in agent_map:
        a_deltas = [(h.get(key, 0.5) - 0.5) if not h.get("suppressed", True) else 0.0
                    for h in hist]
        a_cum = list(np.cumsum(a_deltas))
        fig.add_trace(go.Scatter(
            x=x, y=a_cum,
            mode="lines",
            name=f"S2-{label}",
            line=dict(color="rgba(100,116,139,0.38)", width=1),
            hovertemplate=f"S2-{label}: %{{y:.3f}}<extra></extra>",
        ))

    # Stage 3 — thick solid line + area fill
    final_val  = s3_cum[-1] if s3_cum else 0.0
    is_bull    = final_val >= 0
    s3_color   = MC["call"] if is_bull else MC["put"]
    fill_color = "rgba(34,197,94,0.13)" if is_bull else "rgba(239,68,68,0.13)"

    fig.add_trace(go.Scatter(
        x=x, y=s3_cum,
        mode="lines",
        name="Stage 3 (Cumulative)",
        line=dict(color=s3_color, width=3.5),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate="Stage3 cum: %{y:.3f}<extra></extra>",
    ))

    # Count live (non-suppressed) entries
    n_live = sum(1 for h in hist if not h.get("suppressed", True))

    layout_cfg = base_layout(
        title=f"Accumulated Directional Signal  ({n_live} live bars)", height=340)
    layout_cfg.update({
        "margin": dict(l=55, r=30, t=55, b=40),
        "yaxis":  dict(
            title="Cumulative (prob − 0.5)",
            zeroline=True,
            zerolinecolor="rgba(99,102,241,0.25)",
            zerolinewidth=1,
        ),
        "showlegend": True,
        "legend": dict(
            orientation="h", y=-0.18,
            font=dict(size=10, color=MC["text_muted"]),
        ),
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
    fig.add_hline(y=0.5, line_dash="dot", line_color=MC["text_muted"], secondary_y=False)
    fig.add_hline(y=0.0, line_dash="dot", line_color=MC["text_muted"], secondary_y=True)
    fig.update_yaxes(range=[0, 1], title_text="Probability / Confidence", secondary_y=False)
    fig.update_yaxes(range=[-1, 1], title_text="Strength", secondary_y=True)
    fig.update_layout(**base_layout(title="Stage 3 Rollover Prediction", height=360))
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
    """Thin wrapper around the original loaders + DTE filters with caching."""
    global _cached_agg_df, _cached_snap_df, _cached_agg_mtime, _cached_snap_mtime, _cached_filtered_data

    agg_file = AGG_FILE
    current_agg_mtime = agg_file.stat().st_mtime if agg_file.exists() else 0

    if _cached_agg_df is None or current_agg_mtime != _cached_agg_mtime:
        agg_df = load_agg_data()
        if agg_df is None:
            agg_df = pd.DataFrame()
        _cached_agg_df = agg_df
        _cached_agg_mtime = current_agg_mtime
        _cached_filtered_data.clear()
    else:
        agg_df = _cached_agg_df

    snap_file = SNAPSHOT_FILE
    current_snap_mtime = snap_file.stat().st_mtime if snap_file.exists() else 0

    if _cached_snap_df is None or current_snap_mtime != _cached_snap_mtime:
        snap_df = load_snapshot_data()
        if snap_df is None:
            snap_df = pd.DataFrame()
        _cached_snap_df = snap_df
        _cached_snap_mtime = current_snap_mtime
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
            'fontSize': '9px', 'fontWeight': 600, 'letterSpacing': '1.2px',
            'textTransform': 'uppercase', 'color': MC['text_muted'], 'marginBottom': '4px',
        }),
        html.Div(value, style={
            'fontSize': '20px', 'fontWeight': 700, 'color': color, 'lineHeight': '1.1',
        }),
    ]
    if sub:
        children.append(html.Div(sub, style={'fontSize': '10px', 'color': MC['text_muted'], 'marginTop': '2px'}))
    return html.Div(
        style={
            'background': 'rgba(18,18,26,0.8)', 'backdropFilter': 'blur(12px)',
            'WebkitBackdropFilter': 'blur(12px)',
            'border': f'1px solid {MC["border"]}', 'borderRadius': '10px',
            'padding': '12px 14px', 'flex': '1', 'minWidth': '100px',
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
                html.Span(sym, style={'fontWeight': 700, 'fontSize': '11px', 'color': MC['accent']}),
                html.Span(f"${st['price']:.2f}" if pd.notna(st.get('price')) else "--", style={'fontSize': '12px', 'color': MC['text'], 'fontWeight': 600}),
                html.Span(f"{chg:+.2f}%" if pd.notna(chg) else "--", style={'fontSize': '10px', 'color': chg_color, 'fontWeight': 700}),
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
                'fontSize': '9px', 'fontWeight': 700, 'letterSpacing': '1.5px',
                'color': MC['accent'], 'marginBottom': '10px',
            }),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                html.Span(f"CALL {_fmt_premium_abs(call_p)} ({call_pct:.0f}%)", style={
                    'fontSize': '11px', 'fontWeight': 700, 'color': MC['call'], 'minWidth': '140px',
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
                    'fontSize': '11px', 'fontWeight': 700, 'color': MC['put'],
                    'minWidth': '140px', 'textAlign': 'right',
                }),
            ]),
            html.Div(f"Net: {_fmt_premium(st.get('net_premium', 0.0))} | Total: {_fmt_premium_abs(total)}", style={
                'fontSize': '10px', 'color': MC['text_muted'], 'marginTop': '6px', 'textAlign': 'center',
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
        html.Span(txt, style={'fontWeight': 700, 'fontSize': '13px', 'color': col}),
        html.Span(f" \u2014 {sub}", style={'fontSize': '12px', 'color': MC['text_sec']}),
    ]
    if vix_level and vix_level > 0:
        vc = MC['put'] if vix_level > 25 else (MC['warning'] if vix_level > 18 else MC['call'])
        children.append(html.Span(f"  |  VIX: {vix_level:.1f}", style={'fontSize': '12px', 'color': vc, 'fontWeight': 700}))
    return html.Div(
        style={
            'background': MC['bg_card'],
            'borderLeft': f'4px solid {col}',
            'border': f'1px solid {MC["border"]}',
            'borderRadius': '8px', 'padding': '10px 16px', 'marginBottom': '12px',
        },
        children=children,
    )


def _mc_section_header(title):
    return html.Div(title, style={
        'fontSize': '10px', 'fontWeight': 700, 'letterSpacing': '1.5px',
        'textTransform': 'uppercase', 'color': MC['accent'],
        'borderLeft': f'3px solid {MC["accent"]}',
        'paddingLeft': '12px', 'margin': '28px 0 14px 0',
    })


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
        "fontFamily": "'Inter', 'SF Pro Display', system-ui, -apple-system, sans-serif",
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

            body {{ margin: 0; padding: 0; background: {MC['bg_dark']}; }}

            /* Scrollbar */
            ::-webkit-scrollbar {{ width: 6px; }}
            ::-webkit-scrollbar-track {{ background: {MC['bg_dark']}; }}
            ::-webkit-scrollbar-thumb {{ background: {MC['border']}; border-radius: 3px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: {MC['accent']}; }}

            /* Dropdown overrides */
            .Select-control, .Select-menu-outer, .Select-option, .Select-value, .Select-placeholder {{
              background: {MC['bg_input']} !important;
              color: {MC['text']} !important;
              border-color: {MC['border']} !important;
              font-size: 12px !important;
            }}
            .Select-control:hover {{
              border-color: {MC['border_active']} !important;
            }}
            .Select--single > .Select-control .Select-value,
            .Select--single > .Select-control .Select-placeholder {{
              color: {MC['text']} !important;
            }}
            .Select-arrow-zone .Select-arrow {{
              border-top-color: {MC['text_muted']} !important;
            }}
            .Select-option.is-focused {{
              background: rgba(99,102,241,0.18) !important;
              color: {MC['text']} !important;
            }}
            .Select-option.is-selected {{
              background: rgba(99,102,241,0.28) !important;
              color: {MC['text']} !important;
            }}
            .Select-input input {{
              color: {MC['text']} !important;
            }}
            .Select-menu-outer {{
              border: 1px solid {MC['border']} !important;
            }}

            /* Plotly overrides */
            .js-plotly-plot, .plotly {{
              background: {MC['bg_card']} !important;
              border-radius: 10px;
            }}

            /* Button hover effects */
            button:hover {{ opacity: 0.88; }}
            button:active {{ transform: scale(0.97); }}
            </style>
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
                            "color": MC["text"], "margin": 0, "fontSize": "18px",
                            "fontWeight": 800, "letterSpacing": "2px",
                            "fontFamily": "'Inter', sans-serif",
                        }),
                        html.Div("Intelligence Terminal", style={
                            "color": MC["text_muted"], "fontSize": "10px",
                            "fontWeight": 500, "letterSpacing": "1px", "marginTop": "1px",
                        }),
                    ]),
                ]),
                html.Div(id="live-status", style={"fontSize": "12px"}),
            ],
        ),

        # ── Subheader info bar ──
        html.Div(
            id="subheader",
            style={
                "color": MC["text_muted"], "fontSize": "11px", "padding": "6px 28px",
                "background": MC["bg_dark"], "borderBottom": f"1px solid {MC['border']}",
                "fontFamily": "'JetBrains Mono', monospace", "letterSpacing": "0.3px",
            },
        ),

        # ── Controls Bar ──
        html.Div(
            style={
                "display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap",
                "backgroundColor": MC["bg_card"], "padding": "8px 28px",
                "borderBottom": f"1px solid {MC['border']}", "fontSize": "12px",
            },
            children=[
                html.Div([
                    html.Label("Symbol", style={'marginRight': '5px', 'color': MC['text_muted'], 'fontSize': '10px', 'fontWeight': 600, 'letterSpacing': '0.5px', 'textTransform': 'uppercase'}),
                    dcc.Dropdown(id='symbol-dropdown',
                        options=[{'label': s, 'value': s} for s in ['SPXW', 'SPY', 'QQQ', 'IWM', 'VIX', 'VIXW', 'TLT', 'ALL']],
                        value='SPXW', style={'width': '140px'})
                ]),
                html.Div([
                    html.Label("DTE", style={'marginRight': '5px', 'color': MC['text_muted'], 'fontSize': '10px', 'fontWeight': 600, 'letterSpacing': '0.5px', 'textTransform': 'uppercase'}),
                    dcc.Dropdown(id='dte-dropdown',
                        options=[{'label': '0-1 DTE', 'value': '0_1dte'}, {'label': '0DTE Only', 'value': '0dte'},
                                 {'label': '0-2 DTE', 'value': '0_2dte'}, {'label': 'All DTE', 'value': 'all'}],
                        value='0_1dte', style={'width': '140px'})
                ]),
                html.Div([
                    html.Label("Compare", style={'marginRight': '5px', 'color': MC['text_muted'], 'fontSize': '10px', 'fontWeight': 600, 'letterSpacing': '0.5px', 'textTransform': 'uppercase'}),
                    dcc.Dropdown(id='compare-dropdown',
                        options=[{'label': 'No Compare', 'value': 0}, {'label': 'vs 5m', 'value': 5},
                                 {'label': 'vs 15m', 'value': 15}, {'label': 'vs 30m', 'value': 30},
                                 {'label': 'vs 1h', 'value': 60}, {'label': 'vs 2h', 'value': 120}],
                        value=0, style={'width': '150px'})
                ]),
                html.Div([
                    html.Label("Window", style={'marginRight': '5px', 'color': MC['text_muted'], 'fontSize': '10px', 'fontWeight': 600, 'letterSpacing': '0.5px', 'textTransform': 'uppercase'}),
                    dcc.Dropdown(id='window-dropdown',
                        options=[{'label': 'Full Session', 'value': 'session'}, {'label': '15m', 'value': 15},
                                 {'label': '30m', 'value': 30}, {'label': '45m', 'value': 45}, {'label': '60m', 'value': 60}],
                        value='session', style={'width': '140px'})
                ]),
                html.Div(style={'marginLeft': 'auto', 'display': 'flex', 'gap': '6px'}, children=[
                    html.Button('START', id='btn-start', style={
                        'backgroundColor': MC['call'], 'color': '#fff', 'border': 'none',
                        'padding': '6px 14px', 'borderRadius': '6px', 'cursor': 'pointer',
                        'fontWeight': 700, 'fontSize': '11px', 'letterSpacing': '0.5px',
                    }),
                    html.Button('STOP', id='btn-stop', style={
                        'backgroundColor': MC['put'], 'color': '#fff', 'border': 'none',
                        'padding': '6px 14px', 'borderRadius': '6px', 'cursor': 'pointer',
                        'fontWeight': 700, 'fontSize': '11px', 'letterSpacing': '0.5px',
                    }),
                    html.Button('PAUSE', id='btn-pause', n_clicks=0, style={
                        'backgroundColor': MC['warning'], 'color': '#fff', 'border': 'none',
                        'padding': '6px 14px', 'borderRadius': '6px', 'cursor': 'pointer',
                        'fontWeight': 700, 'fontSize': '11px', 'letterSpacing': '0.5px',
                    }),
                    html.Button('REFRESH', id='btn-refresh', n_clicks=0, style={
                        'backgroundColor': MC['accent'], 'color': '#fff', 'border': 'none',
                        'padding': '6px 14px', 'borderRadius': '6px', 'cursor': 'pointer',
                        'fontWeight': 700, 'fontSize': '11px', 'letterSpacing': '0.5px',
                    }),
                ]),
                html.Div(id='fetcher-status', style={'fontSize': '11px', 'color': MC['text_muted']}),
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
    [Input('btn-start', 'n_clicks'), Input('btn-stop', 'n_clicks')],
    prevent_initial_call=True
)
def manage_fetcher(start_clicks, stop_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-start':
        start_fetcher()
    elif button_id == 'btn-stop':
        stop_fetcher()

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

    # Fetcher status
    status_html = []
    fs = {}
    if is_fetcher_running():
        fs = get_fetcher_status()
        status_html = html.Span([
            html.Span("\u25CF ", style={'color': MC['call'], 'fontSize': '14px'}),
            html.Span("Running ", style={'color': MC['call'], 'fontWeight': 600}),
            html.Span(f"B#{fs.get('batch_id', '?')} | PID {fs.get('pid', '?')}", style={'color': MC['text_muted']})
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
    total_batches = int(df_agg["batch_id"].max()) if (df_agg is not None and not df_agg.empty and "batch_id" in df_agg.columns) else 0
    live_badge = (
        html.Span(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
            html.Span(style={
                'width': '8px', 'height': '8px', 'borderRadius': '50%', 'backgroundColor': MC['call'],
                'boxShadow': f'0 0 8px {MC["call"]}', 'display': 'inline-block',
            }),
            html.Span("LIVE", style={'color': MC['call'], 'fontWeight': 700, 'fontSize': '11px', 'letterSpacing': '1px'}),
            html.Span(f"Batch #{fs.get('batch_id', '?')}", style={'color': MC['text_muted'], 'fontSize': '11px'}),
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

    # ── Ticker Ribbon (all symbols at a glance) ──
    ticker_ribbon = _mc_ticker_ribbon(latest_stats)
    if ticker_ribbon is not None:
        content.append(ticker_ribbon)

    if symbol != 'ALL' and symbol in latest_stats:
        st = latest_stats[symbol]
        price_color = MC['call'] if st['price_change'] >= 0 else MC['put']

        # ── Glassmorphism Stat Cards ──
        # Helper: format stat or show '--' when zero/unavailable
        def _s(key, fmt="{:.2f}", prefix="", suffix=""):
            v = st.get(key, 0.0)
            if v == 0.0 or v is None or (isinstance(v, float) and np.isnan(v)):
                return "--"
            return f"{prefix}{fmt.format(v)}{suffix}"

        content.append(html.Div(
            style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'marginBottom': '14px'},
            children=[
                _mc_metric_card('Price',
                                f"${st['price']:.2f}" if pd.notna(st.get('price')) and st.get('price', 0) != 0 else "--",
                                MC['text'],
                                sub=f"{st['price_change']:+.2f}%" if st.get('price_change', 0) != 0 else "--"),
                _mc_metric_card('P/C Ratio',
                                f"{st['pc_ratio']:.3f}" if st.get('pc_ratio', 0) != 0 else "--",
                                MC['put'] if st.get('pc_ratio', 0) > 1.0 else MC['call'],
                                sub='Bearish' if st.get('pc_ratio', 0) > 1.0 else ('Bullish' if st.get('pc_ratio', 0) > 0 else None)),
                _mc_metric_card('IV Skew',
                                f"{st['iv_skew']:.4f}" if st.get('iv_skew', 0) != 0 else "--",
                                MC['text_sec']),
                _mc_metric_card('Net GEX',
                                (f"{st['net_gamma']/1e6:.1f}M" if abs(st.get('net_gamma', 0)) >= 1e6 else f"{st.get('net_gamma', 0)/1e3:.1f}K") if st.get('net_gamma', 0) != 0 else "--",
                                MC['call'] if st.get('net_gamma', 0) >= 0 else MC['put'],
                                sub='Pos Gamma' if st.get('net_gamma', 0) > 0 else ('Neg Gamma' if st.get('net_gamma', 0) < 0 else None)),
                _mc_metric_card('ATM Straddle',
                                f"${st['atm_straddle']:.2f}" if st.get('atm_straddle', 0) != 0 else "--",
                                MC['warning']),
                _mc_metric_card('Call IV',
                                f"{st['call_iv']:.1f}%" if st.get('call_iv', 0) != 0 else "--",
                                MC['call']),
                _mc_metric_card('Put IV',
                                f"{st['put_iv']:.1f}%" if st.get('put_iv', 0) != 0 else "--",
                                MC['put']),
                _mc_metric_card('Aggression',
                                f"{st['trade_aggression']:+.3f}" if st.get('trade_aggression', 0) != 0 else "--",
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

        # =================================================================
        # MODEL PREDICTION SECTION
        # =================================================================

        if model_out is None:
            # No prediction data available
            content.append(_prediction_unavailable_card())
        else:
            # 1. HUD Strip
            hud_strip = _create_agent_hud_strip(model_out, symbol, df_agg)
            if hud_strip is not None:
                content.append(hud_strip)

            # 2. Decision Engine + Sizing Guidance side by side
            decision_panel = _create_decision_engine_panel(model_out, pred_history_roll)
            sizing_panel = _create_sizing_guidance(model_out, pred_history_roll)
            content.append(html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                children=[decision_panel, sizing_panel],
            ))

            # 3. Price Expected Move Chart (candlestick + forward range)
            fig_em = _create_expected_move_chart(df_agg, symbol, model_out, pred_history_roll)
            if fig_em is not None:
                content.append(_mc_section_header("Price & Expected Move"))
                content.append(dcc.Graph(figure=fig_em, style={'height': '470px'}))

            # 4. Enhanced Signal Meters + Agent Agreement Bar
            fig_meter = _create_signal_meters(model_out)
            if fig_meter is not None:
                content.append(_mc_section_header("Signal Meters"))
                content.append(dcc.Graph(figure=fig_meter, style={'height': '520px'}))
                agreement_bar = _create_agent_agreement_bar(model_out)
                if agreement_bar is not None:
                    content.append(agreement_bar)

            # 5. Model Rollover (from prediction.csv history)
            fig_roll = _create_model_rollover_chart(pred_history_roll)
            if fig_roll is not None:
                content.append(_mc_section_header("Model Rollover Prediction"))
                content.append(dcc.Graph(figure=fig_roll, style={'height': '360px'}))

            # 5b. Accumulated Signal Chart (new)
            fig_accum = _create_accumulated_prediction_chart(pred_history_roll)
            if fig_accum is not None:
                content.append(_mc_section_header("Accumulated Signal"))
                content.append(dcc.Graph(figure=fig_accum, style={'height': '340px'}))

            # 6. Model Health Panel
            health_panel = _create_model_health_panel(model_out, pred_history_roll)
            if health_panel is not None:
                content.append(_mc_section_header("Model Health"))
                content.append(health_panel)

            # 7. Alert Panel
            alert_panel = _create_alert_panel()
            content.append(_mc_section_header("Alerts & Notifications"))
            content.append(alert_panel)

        # =================================================================
        # REST OF DASHBOARD (unchanged chart sections)
        # =================================================================

        # Time-series metrics
        try:
            ts_charts = create_timeseries_individual(df_agg, symbol, window_minutes=window)
        except Exception:
            ts_charts = []
        if ts_charts:
            content.append(_mc_section_header("Time-Series Metrics"))
            for fig, box in ts_charts:
                content.append(dcc.Graph(figure=fig, style={'height': '300px'}))
                ins = _insight(box)
                if ins is not None:
                    content.append(ins)

        # Market microstructure
        try:
            micro_charts = create_microstructure_individual(df_agg, symbol, window_minutes=window)
        except Exception:
            micro_charts = []
        if micro_charts:
            content.append(_mc_section_header("Market Microstructure"))
            for fig, box in micro_charts:
                content.append(dcc.Graph(figure=fig, style={'height': '320px'}))
                ins = _insight(box)
                if ins is not None:
                    content.append(ins)

        # Gamma exposure profile
        try:
            fig_gamma = create_gamma_chart(df_snap, symbol, spot_raw, lookback_df=None)
        except Exception:
            fig_gamma = None
        if fig_gamma is not None:
            content.append(_mc_section_header("Gamma Exposure Profile"))
            content.append(dcc.Graph(figure=fig_gamma, style={'height': '400px'}))
            try:
                text, anomaly = gamma_chart_insight(df_snap, symbol, spot_raw)
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Key strike levels
        try:
            fig_strike = create_strike_chart(df_snap, symbol, spot_raw, lookback_df=None)
        except Exception:
            fig_strike = None
        if fig_strike is not None:
            content.append(_mc_section_header("Key Strike Levels"))
            content.append(dcc.Graph(figure=fig_strike, style={'height': '500px'}))
            try:
                text, anomaly = strike_chart_insight(df_snap, symbol, spot_raw)
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Vol/OI ratio
        try:
            fig_vol_oi = create_vol_oi_chart(df_snap, symbol, spot_raw)
        except Exception:
            fig_vol_oi = None
        if fig_vol_oi is not None:
            content.append(_mc_section_header("Vol/OI Ratio (Live)"))
            content.append(dcc.Graph(figure=fig_vol_oi, style={'height': '400px'}))
            try:
                text, anomaly = vol_oi_insight(df_snap, symbol)
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # IV term structure
        try:
            fig_iv = create_iv_chart(df_snap, symbol)
        except Exception:
            fig_iv = None
        if fig_iv is not None:
            content.append(_mc_section_header("IV Term Structure"))
            content.append(dcc.Graph(figure=fig_iv, style={'height': '400px'}))
            try:
                text, anomaly = iv_chart_insight(df_snap, symbol)
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Vanna exposure
        try:
            fig_vanna = create_vanna_chart(df_snap, symbol, spot_raw)
        except Exception:
            fig_vanna = None
        if fig_vanna is not None:
            content.append(_mc_section_header("Vanna Exposure"))
            content.append(dcc.Graph(figure=fig_vanna, style={'height': '400px'}))
            try:
                text, anomaly = vanna_chart_insight(df_snap, symbol)
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Dealer positioning
        try:
            fig_dealer = create_dealer_chart(df_snap, symbol)
        except Exception:
            fig_dealer = None
        if fig_dealer is not None:
            content.append(_mc_section_header("Dealer Positioning"))
            content.append(dcc.Graph(figure=fig_dealer, style={'height': '400px'}))
            try:
                text, anomaly = dealer_chart_insight(df_snap, symbol)
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # OI walls & pinning
        try:
            fig_oi = create_oi_walls_chart(df_snap, symbol, spot_raw)
        except Exception:
            fig_oi = None
        if fig_oi is not None:
            content.append(_mc_section_header("OI Walls & Pinning"))
            content.append(dcc.Graph(figure=fig_oi, style={'height': '400px'}))
            try:
                text, anomaly = oi_walls_insight(df_snap, symbol)
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Expiration concentration
        try:
            fig_dte = create_dte_concentration_chart(df_snap, symbol)
        except Exception:
            fig_dte = None
        if fig_dte is not None:
            content.append(_mc_section_header("Expiration Concentration"))
            content.append(dcc.Graph(figure=fig_dte, style={'height': '400px'}))
            try:
                text, anomaly = dte_concentration_insight(df_snap, symbol)
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Cumulative volume delta
        try:
            fig_vol = create_cum_vol_delta_chart(df_agg, symbol, window_minutes=window)
        except Exception:
            fig_vol = None
        if fig_vol is not None:
            content.append(_mc_section_header("Cumulative Volume Delta"))
            content.append(dcc.Graph(figure=fig_vol, style={'height': '400px'}))
            try:
                text, anomaly = cum_vol_delta_insight(df_agg, symbol, window_minutes=window)
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Options flow history
        try:
            fig_flow = create_flow_chart(df_agg, symbol)
        except Exception:
            fig_flow = None
        if fig_flow is not None:
            content.append(_mc_section_header("Options Flow History"))
            content.append(dcc.Graph(figure=fig_flow, style={'height': '400px'}))
            try:
                text, anomaly = flow_chart_insight(df_agg, symbol)
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Market maker flow changes
        try:
            fig_mm = create_mm_flow_chart(df_agg, [symbol])
        except Exception:
            fig_mm = None
        if fig_mm is not None:
            content.append(_mc_section_header("Market Maker Flow Changes"))
            content.append(dcc.Graph(figure=fig_mm, style={'height': '350px'}))
            try:
                text, anomaly = mm_flow_insight(df_agg, [symbol])
                ins = _insight(implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # VIX hedging section
        if symbol in ("VIX", "VIXW"):
            try:
                fig_vix_flow = create_vix_put_flow_chart(df_agg)
            except Exception:
                fig_vix_flow = None
            if fig_vix_flow is not None:
                content.append(_mc_section_header("VIX Put Flow"))
                content.append(dcc.Graph(figure=fig_vix_flow, style={'height': '350px'}))
                try:
                    text, anomaly = vix_put_flow_insight(df_agg)
                    ins = _insight(implication_box_html(text, anomaly) if text else "")
                    if ins is not None:
                        content.append(ins)
                except Exception:
                    pass

            try:
                fig_vix_hedge = create_vix_hedging_chart(df_snap)
            except Exception:
                fig_vix_hedge = None
            if fig_vix_hedge is not None:
                content.append(_mc_section_header("VIX Institutional Hedging"))
                content.append(dcc.Graph(figure=fig_vix_hedge, style={'height': '400px'}))
                try:
                    text, anomaly = vix_hedging_insight(df_snap)
                    ins = _insight(implication_box_html(text, anomaly) if text else "")
                    if ins is not None:
                        content.append(ins)
                except Exception:
                    pass

    else:
        # Cross-symbol views when ALL is selected
        try:
            fig_gamma = create_multi_gamma_chart(df_agg)
        except Exception:
            fig_gamma = None
        if fig_gamma is not None:
            content.append(_mc_section_header("Cross-Symbol Gamma Comparison"))
            content.append(dcc.Graph(figure=fig_gamma, style={'height': '400px'}))

        try:
            fig_sent = create_multi_sentiment_chart(df_agg)
        except Exception:
            fig_sent = None
        if fig_sent is not None:
            content.append(_mc_section_header("Cross-Symbol Sentiment"))
            content.append(dcc.Graph(figure=fig_sent, style={'height': '400px'}))

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

    return html.Div(content), status_html, live_badge, subheader


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
