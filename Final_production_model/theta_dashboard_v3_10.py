# ========================================
# THETA OPTIONS INTELLIGENCE DASHBOARD v3.10
# Full Plotly + JupyterLab version
# With chart implications + anomaly warnings
# ========================================
#
# USAGE: In a JupyterLab cell, run:
#   %run theta_dashboard_v3.py
# OR:
#   from theta_dashboard_v3 import run_dashboard
#   run_dashboard()
#

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    from IPython.display import display, HTML
    display(HTML("""
    <div style="background:#1a1a2e;padding:30px;border-radius:12px;text-align:center;">
        <h2 style="color:#ff4757;">Plotly Not Installed</h2>
        <p style="color:#e8eaed;">Run in a cell: <code>!pip install plotly</code></p>
    </div>"""))
    raise SystemExit

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import signal
import os
import time
from datetime import datetime
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# Market hours window (New York / Eastern Time)
# 30 min before open (9:00 AM) to 30 min after close (4:30 PM)
ET = ZoneInfo("America/New_York")
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

FETCHER_SCRIPT = Path("/home/jovyan/theta_fetching_v5.py")
DATA_DIR = Path("/home/jovyan/daily_data")
AGG_FILE = DATA_DIR / "theta_agg.csv"
SNAPSHOT_FILE = DATA_DIR / "theta_snapshot.csv"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
STATUS_FILE = DATA_DIR / ".fetcher_status"

REFRESH_INTERVAL = 20
MAX_HISTORY = 200

# ============================================================================
# DEBUG MODE - Set to True to see filtering diagnostics in console
# ============================================================================
DEBUG_FILTER = False  # Set to True to debug time filtering issues


C = {
    'bg_dark':    '#0f172a',
    'bg_card':    '#1e293b',
    'bg_input':   '#334155',
    'accent':     '#3b82f6',
    'call':       '#10b981',
    'put':        '#ef4444',
    'neutral':    '#3b82f6',
    'warning':    '#f59e0b',
    'text':       '#f1f5f9',
    'text_sec':   '#cbd5e1',
    'text_muted': '#94a3b8',
    'grid':       '#334155',  # must be valid Plotly color
    'border':     'rgba(148,163,184,0.2)',
}

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
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                ts_str = mtime.strftime("%H:%M:%S")
                t_min = mtime.hour * 60 + mtime.minute
                # Can't reliably determine ET from mtime, include it
        except Exception:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
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
            latest_ts = pd.Timestamp(datetime.fromtimestamp(latest_path.stat().st_mtime))
    except Exception:
        latest_ts = pd.Timestamp(datetime.fromtimestamp(latest_path.stat().st_mtime))
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
                snap_ts = pd.Timestamp(datetime.fromtimestamp(fpath.stat().st_mtime))
        except Exception:
            snap_ts = pd.Timestamp(datetime.fromtimestamp(fpath.stat().st_mtime))
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
                first_ts = pd.Timestamp(datetime.fromtimestamp(snaps[0][2].stat().st_mtime))
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
    # gamma_exp already has dealer-perspective sign from enrich_for_ai():
    #   calls = +gamma×OI×spot×100, puts = -gamma×OI×spot×100
    # Do NOT multiply by cp_sign again here.
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


def render_dashboard(selected_symbol='ALL', lookback_batches=0, window_minutes=30, dte_filter='0_1dte'):
    agg_df = load_agg_data()
    if agg_df is not None and '_ts_parsed' not in agg_df.columns:
        agg_df['_ts_parsed'] = pd.NaT
    options_df = load_snapshot_data()

    # Apply DTE filter to snapshot data (v3.9)
    options_df = _apply_dte_filter(options_df, dte_filter)

    # Apply DTE filter to agg data if dte_group column exists (v3.9 fetcher)
    agg_df = _apply_dte_filter_agg(agg_df, dte_filter)

    # lookback_batches is now minutes_ago from the time-based dropdown (v3.8)
    lookback_df = None
    if lookback_batches > 0:
        lookback_df = find_snapshot_by_time_offset(lookback_batches)
        lookback_df = _apply_dte_filter(lookback_df, dte_filter)

    fetcher_alive = is_fetcher_running()
    fs = get_fetcher_status()
    total_batches = int(agg_df['batch_id'].max()) if agg_df is not None else 0

    # Derive last data timestamp from agg data, NOT from datetime.now()
    last_data_ts = None
    data_age_str = ""
    if agg_df is not None and '_ts_parsed' in agg_df.columns and agg_df['_ts_parsed'].notna().any():
        last_data_ts = agg_df['_ts_parsed'].max()
        if pd.notna(last_data_ts):
            age = datetime.now() - last_data_ts.to_pydatetime()
            age_secs = int(age.total_seconds())
            if age_secs < 60:
                data_age_str = f"{age_secs}s ago"
            elif age_secs < 3600:
                data_age_str = f"{age_secs // 60}m ago"
            elif age_secs < 86400:
                data_age_str = f"{age_secs // 3600}h {(age_secs % 3600) // 60}m ago"
            else:
                data_age_str = f"{age_secs // 86400}d {(age_secs % 86400) // 3600}h ago"
    elif agg_df is not None and 'ts' in agg_df.columns:
        try:
            ts_clean = agg_df['ts'].iloc[-1]
            last_data_ts = pd.to_datetime(str(ts_clean).replace(r'[A-Z]{2,5}', ''), errors='coerce')
        except Exception:
            pass

    if last_data_ts and pd.notna(last_data_ts):
        last_update = last_data_ts.strftime("%Y-%m-%d %H:%M:%S") + f" ({data_age_str})"
    else:
        last_update = "No data yet"

    # Count available historical snapshots
    available_snaps = list_available_snapshots()
    snap_count = len(available_snaps)

    if fetcher_alive:
        status_html = f'<span style="color:{C["call"]};">● LIVE</span> <span style="color:{C["text_muted"]};">Batch #{fs["batch_id"]}</span>'
    else:
        if snap_count > 0:
            status_html = f'<span style="color:{C["warning"]};">● REVIEWING</span> <span style="color:{C["text_muted"]};">{snap_count} snapshots available</span>'
        else:
            status_html = f'<span style="color:{C["put"]};">● STOPPED</span>'

    display(HTML(f"""
    <div style="background:{C['bg_dark']};padding:20px 25px;border-radius:10px;
                margin-bottom:12px;font-family:system-ui,sans-serif;
                border:1px solid {C['border']};">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <h1 style="color:{C['text']};margin:0;font-size:1.5em;font-weight:600;">
                Options Intelligence Dashboard</h1>
            <div style="font-size:0.85em;">{status_html}</div>
        </div>
        <div style="color:{C['text_muted']};font-size:0.8em;margin-top:6px;">
            Last data: {last_update} &bull; {total_batches} batches &bull; {snap_count} snapshots
            &bull; Window: {MARKET_OPEN_ET[0]}:{MARKET_OPEN_ET[1]:02d}-{MARKET_CLOSE_ET[0]}:{MARKET_CLOSE_ET[1]:02d} ET
            {f' &bull; Auto-refresh: {REFRESH_INTERVAL}s' if fetcher_alive else ' &bull; Market closed / Fetcher stopped'}
            {f' &bull; Compare: vs {lookback_batches} min ago' if lookback_batches > 0 else ''}{f' &bull; DTE: {dte_filter.replace("_", "-").upper()}' if dte_filter != 'all' else ' &bull; DTE: ALL'}
        </div>
    </div>"""))

    if agg_df is None or agg_df.empty:
        display(HTML(f"""
        <div style="background:{C['bg_card']};padding:50px;border-radius:10px;
                    text-align:center;font-family:sans-serif;">
            <h2 style="color:{C['put']};">No Data Available</h2>
            <p style="color:{C['text']};">
                {'Fetcher running - waiting for first batch...' if fetcher_alive
                 else 'Press START FETCHER to begin collecting data.'}</p>
            <p style="color:{C['text_muted']};font-size:0.8em;">Data dir: {DATA_DIR}</p>
        </div>"""))
        return

    all_symbols = agg_df['symbol'].unique().tolist()
    if selected_symbol != 'ALL':
        symbols_to_show = [selected_symbol] if selected_symbol in all_symbols else all_symbols
    else:
        symbols_to_show = all_symbols

    primary_sym = symbols_to_show[0]
    m = calculate_metrics(options_df, agg_df, primary_sym)

    pc_color = C['call'] if m.get('pc_ratio_raw', 1) < 1 else C['put']
    gex_color = C['call'] if m.get('net_gex_raw', 0) > 0 else C['put']

    display(HTML(f"""
    <div style="background:{C['bg_card']};padding:15px 20px;border-radius:8px;
                margin-bottom:12px;display:grid;
                grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:15px;
                font-family:system-ui,sans-serif;">
        {metric_card_html('Spot Price', m['spot'])}
        {metric_card_html('Net GEX', m['net_gex'], gex_color)}
        {metric_card_html('P/C Ratio', m['pc_ratio'], pc_color)}
        {metric_card_html('IV Skew', m['iv_skew'])}
        {metric_card_html('Call Volume', m['call_premium'], C['call'])}
        {metric_card_html('Put Volume', m['put_premium'], C['put'])}
        {metric_card_html('γ Flip', f"{m['gamma_flip']} {m.get('gamma_flip_dist','')}", C['accent'])}
    </div>"""))

    # === TOP-LEVEL METRIC INSIGHTS ===
    for fn, val in [(pc_ratio_insight, m.get('pc_ratio_raw')),
                    (gex_insight, m.get('net_gex_raw')),
                    (iv_skew_insight, m.get('iv_skew_raw'))]:
        text, anomaly = fn(val)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

    straddle_raw = None
    if agg_df is not None:
        sym_agg = agg_df[agg_df['symbol'] == primary_sym]
        if not sym_agg.empty and 'atm_straddle' in sym_agg.columns:
            straddle_raw = sym_agg.iloc[-1].get('atm_straddle')
    text, anomaly = straddle_insight(straddle_raw, m.get('spot_raw', 0))
    if text:
        display(HTML(implication_box_html(text, anomaly)))

    # === PER-SYMBOL CHARTS ===
    for sym in symbols_to_show:
        spot_raw = 0
        sym_agg = agg_df[agg_df['symbol'] == sym]
        if not sym_agg.empty:
            spot_raw = sym_agg.iloc[-1].get('spot', 0)
            batch_info = f'Batch #{sym_agg.iloc[-1]["batch_id"]} - {sym_agg.iloc[-1].get("ts", "")}'
        else:
            batch_info = ''

        display(HTML(section_header_html(f'{sym}', batch_info)))

        # Time-series (individual charts with insight boxes)
        display(HTML(section_header_html('Time-Series Metrics',
            'P/C ratio, net GEX, IV skew, ATM straddle over time')))
        for _fig, _box in create_timeseries_individual(agg_df, sym, window_minutes=window_minutes):
            display(_fig)
            if _box:
                display(HTML(_box))

        # Market Microstructure
        display(HTML(section_header_html('Market Microstructure',
            'Quote & trade signals: liquidity, urgency, institutional activity')))
        for _fig, _box in create_microstructure_individual(agg_df, sym, window_minutes=window_minutes):
            display(_fig)
            if _box:
                display(HTML(_box))

        # Gamma Exposure
        display(HTML(section_header_html('Gamma Exposure Profile',
            'Dealer gamma positioning by strike')))
        display(create_gamma_chart(options_df, sym, spot_raw, lookback_df))
        text, anomaly = gamma_chart_insight(options_df, sym, spot_raw)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

        # Strike Levels
        display(HTML(section_header_html('Key Strike Levels',
            'Support/resistance from volume concentration')))
        display(create_strike_chart(options_df, sym, spot_raw, lookback_df))
        text, anomaly = strike_chart_insight(options_df, sym, spot_raw)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

        # Vol/OI Ratio (v3.8)
        display(HTML(section_header_html('Vol/OI Ratio (Live)',
            'Volume-to-Open-Interest by strike — fresh positioning detection')))
        display(create_vol_oi_chart(options_df, sym, spot_raw))
        text, anomaly = vol_oi_insight(options_df, sym)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

        # IV Term Structure
        display(HTML(section_header_html('IV Term Structure',
            'Implied volatility across expirations')))
        display(create_iv_chart(options_df, sym))
        text, anomaly = iv_chart_insight(options_df, sym)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

        # Vanna
        display(HTML(section_header_html('Vanna Exposure',
            'Sensitivity of delta to IV changes')))
        display(create_vanna_chart(options_df, sym, spot_raw))
        text, anomaly = vanna_chart_insight(options_df, sym)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

        # Dealer Positioning
        display(HTML(section_header_html('Dealer Positioning',
            'Estimated net dealer greek exposures')))
        display(create_dealer_chart(options_df, sym))
        text, anomaly = dealer_chart_insight(options_df, sym)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

        # OI Walls & Pinning
        display(HTML(section_header_html('OI Walls & Pinning',
            'Open interest concentration — support/resistance/pin targets')))
        display(create_oi_walls_chart(options_df, sym, spot_raw))
        text, anomaly = oi_walls_insight(options_df, sym)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

        # DTE Concentration
        display(HTML(section_header_html('Expiration Concentration',
            'OI/volume distribution across DTE buckets — near-term vs long-dated')))
        display(create_dte_concentration_chart(options_df, sym))
        text, anomaly = dte_concentration_insight(options_df, sym)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

    # === CROSS-SYMBOL CHARTS ===
    if len(all_symbols) > 1:
        display(HTML(section_header_html('Cross-Symbol Gamma Comparison',
            'Net gamma across all tracked symbols')))
        display(create_multi_gamma_chart(agg_df))
        text, anomaly = multi_gamma_insight(agg_df)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

        display(HTML(section_header_html('Cross-Symbol Sentiment',
            'Call vs put volume across symbols')))
        display(create_multi_sentiment_chart(agg_df))
        text, anomaly = multi_sentiment_insight(agg_df)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

    # Cumulative Volume Delta (v3.8)
    display(HTML(section_header_html('Cumulative Volume Delta',
        'Running sum of (call vol − put vol) — directional flow pressure')))
    display(create_cum_vol_delta_chart(agg_df, primary_sym, window_minutes=window_minutes))
    text, anomaly = cum_vol_delta_insight(agg_df, primary_sym, window_minutes=window_minutes)
    if text:
        display(HTML(implication_box_html(text, anomaly)))

    # Options Flow
    display(HTML(section_header_html('Options Flow',
        'Call vs put volume flow over time')))
    display(create_flow_chart(agg_df, primary_sym))
    text, anomaly = flow_chart_insight(agg_df, primary_sym)
    if text:
        display(HTML(implication_box_html(text, anomaly)))
    display(HTML(implication_box_html(
        "<b>Net Premium Bars (right axis):</b> "
        "Green = net call premium dominance, red = net put premium dominance. "
        "When volume is flat but net premium spikes — large block trades are moving the market. "
        "Volume shows activity; premium shows conviction."
    )))

    # MM Flow Changes
    display(HTML(section_header_html('Market Maker Flow Changes',
        'Net positioning changes - institutional momentum')))
    display(create_mm_flow_chart(agg_df, all_symbols[:3]))
    text, anomaly = mm_flow_insight(agg_df, all_symbols[:3])
    if text:
        display(HTML(implication_box_html(text, anomaly)))

    # VIX Hedging — check both agg symbols AND snapshot symbols for VIXW/VIX
    vix_syms = [s for s in all_symbols if s in ('VIX', 'VIXW')]
    has_vix_in_snapshot = False
    if options_df is not None and not options_df.empty and 'symbol' in options_df.columns:
        snap_syms = options_df['symbol'].unique()
        has_vix_in_snapshot = any(s in snap_syms for s in ('VIX', 'VIXW'))
    if vix_syms or has_vix_in_snapshot:
        display(HTML(section_header_html('VIX Institutional Hedging',
            'Large put buying = portfolio protection')))
        display(create_vix_put_flow_chart(agg_df))
        text, anomaly = vix_put_flow_insight(agg_df)
        if text:
            display(HTML(implication_box_html(text, anomaly)))

        display(create_vix_hedging_chart(options_df))
        text, anomaly = vix_hedging_insight(options_df)
        if text:
            display(HTML(implication_box_html(text, anomaly)))


# ========================================
# CONTROL PANEL + AUTO-REFRESH
# ========================================

def run_dashboard():
    """
    Main entry point. In JupyterLab run:
        %run theta_dashboard_v3.py
    Or:
        from theta_dashboard_v3 import run_dashboard
        run_dashboard()
    """
    symbol_dropdown = widgets.Dropdown(
        options=['SPXW', 'SPY', 'QQQ', 'IWM', 'VIX', 'VIXW', 'TLT', 'ALL'],
        value='SPXW', description='Symbol:',
        style={'description_width': '55px'},
        layout=widgets.Layout(width='160px')
    )
    # Time-based compare dropdown (v3.8)
    lookback_dropdown = widgets.Dropdown(
        options=[
            ('No Compare', 0),
            ('vs 5 min ago', 5),
            ('vs 15 min ago', 15),
            ('vs 30 min ago', 30),
            ('vs 1 hr ago', 60),
            ('vs 2 hr ago', 120),
        ],
        value=0, description='Compare:',
        style={'description_width': '65px'},
        layout=widgets.Layout(width='220px')
    )
    window_dropdown = widgets.Dropdown(
        options=[('Full Session', 'session'), ('15 min', 15), ('30 min', 30), 
                 ('45 min', 45), ('60 min', 60)],
        value='session', description='Window:',
        style={'description_width': '60px'},
        layout=widgets.Layout(width='190px')
    )
    # DTE filter dropdown (v3.10 - default 0-1 DTE for credit spread trading)
    dte_dropdown = widgets.Dropdown(
        options=[
            ('0-1 DTE', '0_1dte'),
            ('0DTE Only', '0dte'),
            ('0-2 DTE', '0_2dte'),
            ('All DTE', 'all'),
        ],
        value='0_1dte', description='DTE:',
        style={'description_width': '40px'},
        layout=widgets.Layout(width='160px')
    )

    btn_start = widgets.Button(description='START FETCHER',
        button_style='success', layout=widgets.Layout(width='155px', height='36px'))
    btn_stop = widgets.Button(description='STOP FETCHER',
        button_style='danger', layout=widgets.Layout(width='155px', height='36px'))
    btn_refresh = widgets.Button(description='REFRESH',
        button_style='info', layout=widgets.Layout(width='120px', height='36px'))
    btn_pause = widgets.Button(description='⏸ PAUSE REFRESH',
        button_style='warning', layout=widgets.Layout(width='170px', height='36px'))

    auto_refresh_active = [True]

    status_label = widgets.HTML(
        value=f'<span style="color:{C["text_muted"]};">Initializing...</span>')

    dashboard_output = widgets.Output()

    display(HTML('''<style>
        .jp-OutputArea-output { transition: opacity 0.15s ease; }
        .widget-output { background: #0f172a !important; min-height: 600px; }
    </style>'''))

    def refresh_display():
        with dashboard_output:
            clear_output(wait=True)
            render_dashboard(
                selected_symbol=symbol_dropdown.value,
                lookback_batches=lookback_dropdown.value,
                window_minutes=window_dropdown.value,
                dte_filter=dte_dropdown.value
            )
        if is_fetcher_running():
            fs = get_fetcher_status()
            status_label.value = (
                f'<span style="color:{C["call"]};">Running</span> '
                f'<span style="color:{C["text_muted"]};">Batch #{fs["batch_id"]} | PID {fs["pid"]}</span>')
        else:
            status_label.value = f'<span style="color:{C["put"]};">Stopped</span>'

    def on_start(btn):
        start_fetcher()
        time.sleep(1)
        refresh_display()

    def on_stop(btn):
        stop_fetcher()
        time.sleep(1)
        refresh_display()

    def on_refresh(btn):
        refresh_display()

    def on_pause(btn):
        auto_refresh_active[0] = not auto_refresh_active[0]
        if auto_refresh_active[0]:
            btn_pause.description = '⏸ PAUSE REFRESH'
            btn_pause.button_style = 'warning'
        else:
            btn_pause.description = '▶ RESUME REFRESH'
            btn_pause.button_style = 'success'

    btn_start.on_click(on_start)
    btn_stop.on_click(on_stop)
    btn_refresh.on_click(on_refresh)
    btn_pause.on_click(on_pause)

    row1 = widgets.HBox(
        [symbol_dropdown, dte_dropdown, lookback_dropdown, window_dropdown, btn_start, btn_stop, btn_pause, btn_refresh, status_label],
        layout=widgets.Layout(padding='8px 10px', gap='10px', align_items='center'))

    display(row1)
    display(dashboard_output)
    refresh_display()

    try:
        while True:
            time.sleep(REFRESH_INTERVAL)
            if auto_refresh_active[0]:
                refresh_display()
    except KeyboardInterrupt:
        print("Dashboard stopped. Re-run run_dashboard() to restart.")


if __name__ == "__main__":
    run_dashboard()
