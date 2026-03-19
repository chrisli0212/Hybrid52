import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import os
import time
from datetime import datetime
import sys
import importlib.util
import traceback
from collections import deque

# Load the original script to get its functions and constants
spec = importlib.util.spec_from_file_location("theta_dashboard", "/workspace/theta_dashboard_v3_10.py")
theta_dashboard = importlib.util.module_from_spec(spec)
sys.modules["theta_dashboard"] = theta_dashboard

# Mock IPython modules to avoid errors during import
import unittest.mock
sys.modules['IPython.display'] = unittest.mock.Mock()
sys.modules['ipywidgets'] = unittest.mock.Mock()

spec.loader.exec_module(theta_dashboard)

# Cache variables to avoid reloading data on every refresh
_cached_agg_df = None
_cached_snap_df = None
_cached_agg_mtime = None
_cached_snap_mtime = None
_cached_filtered_data = {}  # Store filtered data by DTE filter
_live_model_service = None
_live_model_init_error = None
_live_model_cache = {"cache_key": None, "result": None}
_model_roll_history = deque(maxlen=360)
_model_roll_last_signature = None
_last_live_non_suppressed_ts = None


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


def _get_live_model_service():
    global _live_model_service, _live_model_init_error
    if _live_model_service is not None:
        return _live_model_service
    if _live_model_init_error is not None:
        return None
    try:
        from live_inference_service import LiveHybrid51InferenceService

        _live_model_service = LiveHybrid51InferenceService()
        return _live_model_service
    except Exception as exc:
        _live_model_init_error = f"{type(exc).__name__}: {exc}"
        return None


def _predict_live_model(agg_df, snap_df):
    global _live_model_cache, _last_live_non_suppressed_ts
    svc = _get_live_model_service()
    if svc is None:
        return {
            "ok": False,
            "suppressed": True,
            "reason": _live_model_init_error or "model_service_unavailable",
            "source_state": "MODEL_INIT_ERROR",
            "cache_hit": False,
        }

    cache_key = (
        _latest_batch_id(agg_df),
        _latest_batch_id(snap_df),
        len(agg_df) if agg_df is not None else 0,
        len(snap_df) if snap_df is not None else 0,
        _latest_time_marker(agg_df),
        _latest_time_marker(snap_df),
    )
    if _live_model_cache.get("cache_key") == cache_key and _live_model_cache.get("result") is not None:
        cached = dict(_live_model_cache["result"])
        cached["source_state"] = "CACHED_SAME_INPUT"
        cached["cache_hit"] = True
        cached["inference_key"] = str(cache_key)
        return cached
    try:
        result = svc.predict_latest(agg_df, snap_df)
    except Exception as exc:
        result = {
            "ok": False,
            "suppressed": True,
            "reason": f"inference_error: {type(exc).__name__}",
            "debug": str(exc),
            "trace": traceback.format_exc(limit=2),
            "source_state": "INFERENCE_ERROR",
            "cache_hit": False,
        }
    result = dict(result or {})
    if "source_state" not in result:
        result["source_state"] = "SUPPRESSED" if bool(result.get("suppressed", False)) else "LIVE_INFERENCE"
    result["cache_hit"] = False
    result["inference_key"] = str(cache_key)
    if bool(result.get("ok", False)) and not bool(result.get("suppressed", False)):
        _last_live_non_suppressed_ts = datetime.now()
    _live_model_cache = {"cache_key": cache_key, "result": result}
    return result


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
    cache_hit = bool(model_out.get("cache_hit", False))
    quality = float(diagnostics.get("quality_score", 0.0) or 0.0)
    latency = diagnostics.get("latency_ms", None)
    warmup_fraction = float(diagnostics.get("warmup_fraction", 0.0) or 0.0)
    completeness = float(diagnostics.get("feature_completeness", 0.0) or 0.0)
    vix_valid = bool(diagnostics.get("vix_valid", False))
    missing_stage1 = int(diagnostics.get("stage1_missing_count", 0) or 0)

    last_live_text = "N/A"
    if _last_live_non_suppressed_ts is not None:
        elapsed = datetime.now() - _last_live_non_suppressed_ts
        secs = int(max(0, elapsed.total_seconds()))
        if secs < 60:
            last_live_text = f"{secs}s ago"
        elif secs < 3600:
            last_live_text = f"{secs // 60}m ago"
        else:
            last_live_text = f"{secs // 3600}h {(secs % 3600) // 60}m ago"

    if suppressed or not ok:
        status_text = "INSUFFICIENT DATA"
        status_color = theta_dashboard.C["warning"]
        signal_text = "Shadow mode suppressed"
        detail = reason or "feature gate triggered"
    else:
        status_text = "LIVE SHADOW SIGNAL"
        status_color = theta_dashboard.C["call"]
        signal_text = "UP" if pred == 1 else "DOWN"
        signal_color = theta_dashboard.C["call"] if pred == 1 else theta_dashboard.C["put"]
        confidence = "HIGH" if abs(prob - 0.5) >= 0.2 else ("MEDIUM" if abs(prob - 0.5) >= 0.1 else "LOW")
        detail = f"P(up): {prob:.3f}  |  Confidence: {confidence}"
    state_line = f"Source: {source_state}" + ("  |  Cache: HIT" if cache_hit else "  |  Cache: MISS")
    gate_line = (
        f"Warmup: {warmup_fraction:.2f}  |  Completeness: {completeness:.2f}  |  "
        f"VIX valid: {'yes' if vix_valid else 'no'}  |  Stage1 missing: {missing_stage1}"
    )
    return html.Div(
        style={
            "backgroundColor": theta_dashboard.C["bg_card"],
            "border": f"1px solid {theta_dashboard.C['border']}",
            "borderLeft": f"4px solid {status_color}",
            "borderRadius": "8px",
            "padding": "12px 14px",
            "minWidth": "290px",
            "maxWidth": "360px",
        },
        children=[
            html.Div(status_text, style={"fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.5px", "color": status_color}),
            html.Div(
                signal_text,
                style={
                    "fontSize": "24px",
                    "fontWeight": 800,
                    "marginTop": "3px",
                    "color": (signal_color if (not suppressed and ok) else theta_dashboard.C["text"]),
                },
            ),
            html.Div(detail, style={"fontSize": "12px", "color": theta_dashboard.C["text_sec"], "marginTop": "3px"}),
            html.Div(state_line, style={"fontSize": "11px", "color": theta_dashboard.C["text_muted"], "marginTop": "4px"}),
            html.Div(gate_line, style={"fontSize": "11px", "color": theta_dashboard.C["text_muted"], "marginTop": "2px"}),
            html.Div(
                f"Quality: {quality:.2f}"
                + (f"  |  Latency: {latency}ms" if latency is not None else "")
                + f"  |  Last live inference: {last_live_text}",
                style={"fontSize": "11px", "color": theta_dashboard.C["text_muted"], "marginTop": "4px"},
            ),
        ],
    )


def _record_model_roll(batch_id, model_out):
    global _model_roll_last_signature
    if not model_out:
        return
    stage2 = dict(model_out.get("stage2_probs", {}) or {})
    signature = (
        batch_id,
        str(model_out.get("source_state", "")),
        str(model_out.get("reason", "")),
        round(float(model_out.get("prob", 0.5) or 0.5), 4),
        round(float(model_out.get("confidence", 0.0) or 0.0), 4),
        round(float(model_out.get("signal_strength", 0.0) or 0.0), 4),
        tuple(sorted((k, round(float(v), 3)) for k, v in stage2.items())),
    )
    if signature == _model_roll_last_signature:
        return
    _model_roll_last_signature = signature

    _model_roll_history.append(
        {
            "batch_id": int(batch_id) if batch_id is not None else -1,
            "ts": datetime.now(),
            "suppressed": bool(model_out.get("suppressed", False)),
            "prob": float(model_out.get("prob", 0.5) or 0.5),
            "confidence": float(model_out.get("confidence", 0.0) or 0.0),
            "strength": float(model_out.get("signal_strength", 0.0) or 0.0),
            "stage2_probs": stage2,
            "source_state": str(model_out.get("source_state", "")),
            "cache_hit": bool(model_out.get("cache_hit", False)),
        }
    )


def _create_expected_move_chart(df_agg: pd.DataFrame, symbol: str, model_out: dict | None):
    if df_agg is None or df_agg.empty or symbol == "ALL" or "symbol" not in df_agg.columns:
        return None
    sdf = df_agg[df_agg["symbol"] == symbol].copy()
    if sdf.empty or "spot" not in sdf.columns:
        return None
    if "_ts_parsed" not in sdf.columns:
        return None

    sdf = sdf[sdf["_ts_parsed"].notna()].copy()
    if sdf.empty:
        return None
    sdf["spot"] = pd.to_numeric(sdf["spot"], errors="coerce")
    sdf = sdf[sdf["spot"].notna()].copy()
    if sdf.empty:
        return None

    sdf = sdf.sort_values("_ts_parsed")
    hist = sdf.tail(min(120, len(sdf))).copy()
    if hist.empty:
        return None

    last_row = sdf.iloc[-1]
    spot = float(last_row.get("spot", hist["spot"].iloc[-1]))
    atm_straddle = float(pd.to_numeric(last_row.get("atm_straddle", np.nan), errors="coerce"))
    if not np.isfinite(atm_straddle) or atm_straddle <= 0:
        # fallback to 1h realized move proxy
        tail = hist.tail(min(60, len(hist)))
        atm_straddle = float(max(0.0, tail["spot"].max() - tail["spot"].min()) / 2.0)
    if not np.isfinite(atm_straddle) or atm_straddle <= 0:
        atm_straddle = max(spot * 0.01, 0.5)

    p_up = float((model_out or {}).get("prob", 0.5) or 0.5)
    p_dn = 1.0 - p_up
    confidence = float((model_out or {}).get("confidence", abs(p_up - 0.5) * 2.0) or 0.0)
    stronger_up = p_up >= p_dn
    up_width = 4 if stronger_up else 2
    dn_width = 4 if not stronger_up else 2
    up_alpha = 0.95 if stronger_up else 0.55
    dn_alpha = 0.95 if not stronger_up else 0.55

    # Build fan over 30-min model horizon with sqrt-time widening.
    horizon_min = 30
    now_ts = hist["_ts_parsed"].iloc[-1]
    future_x = [now_ts + pd.Timedelta(minutes=i) for i in range(horizon_min + 1)]
    t = np.linspace(0, 1, horizon_min + 1)
    em = atm_straddle * np.sqrt(t)
    up_path = spot + em
    dn_path = spot - em

    fig = go.Figure()

    # Historical spot path for context.
    fig.add_trace(
        go.Scatter(
            x=hist["_ts_parsed"],
            y=hist["spot"],
            mode="lines",
            name="Recent Spot",
            line=dict(color=theta_dashboard.C["text_sec"], width=2),
        )
    )

    # Expected-move fan area.
    fig.add_trace(
        go.Scatter(
            x=future_x + future_x[::-1],
            y=list(up_path) + list(dn_path[::-1]),
            fill="toself",
            name="Expected Move Band",
            mode="lines",
            line=dict(color="rgba(148,163,184,0.2)", width=1),
            fillcolor="rgba(148,163,184,0.16)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_x,
            y=up_path,
            mode="lines",
            name=f"UP {p_up*100:.0f}%",
            line=dict(color=f"rgba(16,185,129,{up_alpha})", width=up_width),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_x,
            y=dn_path,
            mode="lines",
            name=f"DOWN {p_dn*100:.0f}%",
            line=dict(color=f"rgba(239,68,68,{dn_alpha})", width=dn_width),
        )
    )
    fig.add_hline(y=spot, line_dash="dash", line_color=theta_dashboard.C["accent"], annotation_text=f"Spot {spot:.2f}")
    # Avoid Plotly add_vline timestamp bug by using add_shape/add_annotation.
    y_min = float(np.nanmin(dn_path))
    y_max = float(np.nanmax(up_path))
    fig.add_shape(
        type="line",
        x0=now_ts,
        x1=now_ts,
        y0=y_min,
        y1=y_max,
        line=dict(dash="dot", color=theta_dashboard.C["warning"], width=1),
    )
    fig.add_annotation(
        x=now_ts,
        y=y_max,
        text="+30m horizon",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color=theta_dashboard.C["warning"], size=10),
        bgcolor="rgba(15,23,42,0.70)",
    )
    fig.add_annotation(
        x=future_x[-1],
        y=up_path[-1] if stronger_up else dn_path[-1],
        text=f"{'UP' if stronger_up else 'DOWN'} favored | Conf {confidence*100:.0f}%",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        bgcolor="rgba(15,23,42,0.75)",
        bordercolor=theta_dashboard.C["border"],
        font=dict(color=theta_dashboard.C["text_sec"], size=11),
    )
    fig.update_layout(
        **theta_dashboard.base_layout(
            title=f"{symbol} Directional Expected Move Fan",
            xaxis_title="Time",
            yaxis_title="Price",
            height=420,
            hovermode="x unified",
        )
    )
    return theta_dashboard.style_axes(fig)


def _create_model_rollover_chart():
    if len(_model_roll_history) < 2:
        return None
    hist = list(_model_roll_history)
    x = [h["ts"] for h in hist]
    prob = [None if h["suppressed"] else h["prob"] for h in hist]
    conf = [None if h["suppressed"] else h["confidence"] for h in hist]
    strength = [None if h["suppressed"] else h["strength"] for h in hist]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=prob, mode="lines+markers", name="Stage3 P(up)", line=dict(color=theta_dashboard.C["accent"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=conf, mode="lines", name="Confidence", line=dict(color=theta_dashboard.C["warning"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=strength, mode="lines", name="Signal Strength", line=dict(color=theta_dashboard.C["call"])), secondary_y=True)
    fig.add_hline(y=0.5, line_dash="dot", line_color=theta_dashboard.C["text_muted"], secondary_y=False)
    fig.add_hline(y=0.0, line_dash="dot", line_color=theta_dashboard.C["text_muted"], secondary_y=True)
    fig.update_yaxes(range=[0, 1], title_text="Probability / Confidence", secondary_y=False)
    fig.update_yaxes(range=[-1, 1], title_text="Strength", secondary_y=True)
    fig.update_layout(**theta_dashboard.base_layout(title="Stage 3 Rollover Prediction", height=360))
    return theta_dashboard.style_axes(fig)


def _create_signal_meters(model_out):
    if not model_out:
        return None
    stage2_probs = dict(model_out.get("stage2_probs", {}) or {})
    stage3_prob = float(model_out.get("prob", 0.5) or 0.5)
    confidence = float(model_out.get("confidence", abs(stage3_prob - 0.5) * 2.0) or 0.0)
    suppressed = bool(model_out.get("suppressed", False))
    ok = bool(model_out.get("ok", False))
    source_state = str(model_out.get("source_state", "UNKNOWN") or "UNKNOWN")
    neutral_mode = suppressed or (not ok)
    bar_color = theta_dashboard.C["text_muted"] if neutral_mode else (theta_dashboard.C["call"] if stage3_prob >= 0.5 else theta_dashboard.C["put"])
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

    # Primary Stage 3 meter (top, full width)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=stage3_prob,
            number={"valueformat": ".2f", "font": {"size": 54}},
            delta={"reference": 0.5, "valueformat": ".2f", "increasing": {"color": theta_dashboard.C["call"]}, "decreasing": {"color": theta_dashboard.C["put"]}},
            title={"text": f"Stage 3 Overall Signal ({source_state})  |  Confidence {confidence*100:.0f}%", "font": {"size": 18, "color": theta_dashboard.C["text"]}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": theta_dashboard.C["text_muted"]},
                "bar": {"color": bar_color, "thickness": 0.42},
                "bgcolor": theta_dashboard.C["bg_card"],
                "borderwidth": 1,
                "bordercolor": theta_dashboard.C["border"],
                "steps": [
                    {"range": [0, 0.4], "color": ("rgba(148,163,184,0.12)" if neutral_mode else "rgba(239,68,68,0.18)")},
                    {"range": [0.4, 0.6], "color": "rgba(148,163,184,0.16)"},
                    {"range": [0.6, 1.0], "color": ("rgba(148,163,184,0.12)" if neutral_mode else "rgba(16,185,129,0.18)")},
                ],
                "threshold": {"line": {"color": theta_dashboard.C["accent"], "width": 3}, "thickness": 0.8, "value": 0.5},
            },
        ),
        row=1,
        col=1,
    )

    # Secondary Stage 2 meters
    for idx, (label, key) in enumerate(agents):
        val = float(stage2_probs.get(key, 0.5))
        row = 2 if idx < 4 else 3
        col = (idx % 4) + 1
        s2_bar = theta_dashboard.C["text_muted"] if neutral_mode else (theta_dashboard.C["call"] if val >= 0.5 else theta_dashboard.C["put"])
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                number={"valueformat": ".2f", "font": {"size": 24}},
                title={"text": label, "font": {"size": 12}},
                gauge={
                    "shape": "angular",
                    "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": theta_dashboard.C["text_muted"]},
                    "bar": {"color": s2_bar, "thickness": 0.3},
                    "bgcolor": theta_dashboard.C["bg_card"],
                    "borderwidth": 1,
                    "bordercolor": theta_dashboard.C["border"],
                    "steps": [
                        {"range": [0, 0.45], "color": ("rgba(148,163,184,0.10)" if neutral_mode else "rgba(239,68,68,0.16)")},
                        {"range": [0.45, 0.55], "color": "rgba(148,163,184,0.12)"},
                        {"range": [0.55, 1.0], "color": ("rgba(148,163,184,0.10)" if neutral_mode else "rgba(16,185,129,0.16)")},
                    ],
                    "threshold": {"line": {"color": theta_dashboard.C["accent"], "width": 2}, "thickness": 0.7, "value": 0.5},
                },
            ),
            row=row,
            col=col,
        )

    layout_cfg = theta_dashboard.base_layout(title="Model Signal Meters", height=760)
    layout_cfg["margin"] = dict(l=30, r=30, t=70, b=30)
    fig.update_layout(**layout_cfg)
    return fig


def load_data(dte_filter="0_1dte"):
    """Thin wrapper around the original loaders + DTE filters with caching."""
    global _cached_agg_df, _cached_snap_df, _cached_agg_mtime, _cached_snap_mtime, _cached_filtered_data
    
    # Check if we need to reload agg data
    agg_file = theta_dashboard.AGG_FILE
    current_agg_mtime = agg_file.stat().st_mtime if agg_file.exists() else 0
    
    if _cached_agg_df is None or current_agg_mtime != _cached_agg_mtime:
        agg_df = theta_dashboard.load_agg_data()
        if agg_df is None:
            agg_df = pd.DataFrame()
        _cached_agg_df = agg_df
        _cached_agg_mtime = current_agg_mtime
        # Clear filtered cache when raw data changes
        _cached_filtered_data.clear()
    else:
        agg_df = _cached_agg_df
    
    # Check if we need to reload snapshot data
    snap_file = theta_dashboard.SNAPSHOT_FILE
    current_snap_mtime = snap_file.stat().st_mtime if snap_file.exists() else 0
    
    if _cached_snap_df is None or current_snap_mtime != _cached_snap_mtime:
        snap_df = theta_dashboard.load_snapshot_data()
        if snap_df is None:
            snap_df = pd.DataFrame()
        _cached_snap_df = snap_df
        _cached_snap_mtime = current_snap_mtime
        # Clear filtered cache when snapshot data changes
        _cached_filtered_data.clear()
    else:
        snap_df = _cached_snap_df
    
    # Check if we already have filtered data for this DTE filter
    cache_key = (dte_filter, _cached_agg_mtime, _cached_snap_mtime)
    if cache_key in _cached_filtered_data:
        return _cached_filtered_data[cache_key]['agg'], _cached_filtered_data[cache_key]['snap']
    
    # Apply DTE filtering
    filtered_agg = agg_df.copy()
    filtered_snap = snap_df.copy()
    
    try:
        filtered_agg = theta_dashboard._apply_dte_filter_agg(filtered_agg, dte_filter)
    except Exception:
        pass
    try:
        filtered_snap = theta_dashboard._apply_dte_filter(filtered_snap, dte_filter)
    except Exception:
        pass
    
    # Cache the filtered data
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
            vol_ratio = 1.0
        net_gamma = float(last.get("net_gex", 0.0) or 0.0)
        stats[sym] = {
            "price": price,
            "price_change": price_change,
            "vol_ratio": vol_ratio,
            "net_gamma": net_gamma,
        }
    return stats

# Setup Dash app
app = dash.Dash(__name__, title="Theta Options Dashboard")


def _insight(box_html: str | None):
    if not box_html:
        return None
    return dcc.Markdown(box_html, dangerously_allow_html=True)


app.layout = html.Div(
    style={
        "backgroundColor": theta_dashboard.C["bg_dark"],
        "minHeight": "100vh",
        "color": theta_dashboard.C["text"],
        "padding": "18px 22px",
        "fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif",
    },
    children=[
        # Global CSS to make Dash controls match the Jupyter theme
        dcc.Markdown(
            f"""
            <style>
            :root {{
              --bg-dark: {theta_dashboard.C['bg_dark']};
              --bg-card: {theta_dashboard.C['bg_card']};
              --bg-input: {theta_dashboard.C['bg_input']};
              --text: {theta_dashboard.C['text']};
              --text-muted: {theta_dashboard.C['text_muted']};
              --border: {theta_dashboard.C['border']};
              --accent: {theta_dashboard.C['accent']};
              --call: {theta_dashboard.C['call']};
              --put: {theta_dashboard.C['put']};
              --warning: {theta_dashboard.C['warning']};
            }}

            /* React-Select (dcc.Dropdown) */
            .Select-control, .Select-menu-outer, .Select-option, .Select-value, .Select-placeholder {{
              background: var(--bg-input) !important;
              color: black !important;
              border-color: rgba(148,163,184,0.25) !important;
            }}
            .Select-control:hover {{
              border-color: rgba(148,163,184,0.45) !important;
            }}
            .Select--single > .Select-control .Select-value, .Select--single > .Select-control .Select-placeholder {{
              color: black !important;
            }}
            .Select-arrow-zone .Select-arrow {{
              border-top-color: #666 !important;
            }}
            .Select-option.is-focused {{
              background: rgba(59,130,246,0.18) !important;
              color: black !important;
            }}
            .Select-option.is-selected {{
              background: rgba(59,130,246,0.28) !important;
              color: black !important;
            }}
            /* Ensure input text is also black */
            .Select-input input {{
              color: black !important;
            }}

            /* Graph background */
            .js-plotly-plot, .plotly {{
              background: var(--bg-card) !important;
            }}
            </style>
            """,
            dangerously_allow_html=True
        ),

        dcc.Store(id="refresh-paused", data=False),

        # Header (like Jupyter)
        html.Div(
            style={
                "background": theta_dashboard.C["bg_dark"],
                "padding": "20px 25px",
                "borderRadius": "10px",
                "marginBottom": "12px",
                "border": f"1px solid {theta_dashboard.C['border']}",
            },
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.H1(
                            "Options Intelligence Dashboard",
                            style={
                                "color": theta_dashboard.C["text"],
                                "margin": 0,
                                "fontSize": "1.5em",
                                "fontWeight": 600,
                            },
                        ),
                        html.Div(id="live-status", style={"fontSize": "0.85em"}),
                    ],
                ),
                html.Div(
                    id="subheader",
                    style={"color": theta_dashboard.C["text_muted"], "fontSize": "0.8em", "marginTop": "6px"},
                ),
            ],
        ),
    
        # Controls (aligned with Jupyter row)
        html.Div(
            style={
                "display": "flex",
                "gap": "10px",
                "marginBottom": "12px",
                "alignItems": "center",
                "flexWrap": "wrap",
                "backgroundColor": theta_dashboard.C["bg_card"],
                "padding": "8px 10px",
                "borderRadius": "8px",
                "border": f"1px solid {theta_dashboard.C['border']}",
                "fontSize": "13px",
            },
            children=[
        html.Div([
            html.Label("Symbol:", style={'marginRight': '6px', 'color': theta_dashboard.C['text_muted']}),
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[{'label': s, 'value': s} for s in ['SPXW', 'SPY', 'QQQ', 'IWM', 'VIX', 'VIXW', 'TLT', 'ALL']],
                value='SPXW',
                style={'width': '160px'}
            )
        ]),
        html.Div([
            html.Label("DTE:", style={'marginRight': '6px', 'color': theta_dashboard.C['text_muted']}),
            dcc.Dropdown(
                id='dte-dropdown',
                options=[
                    {'label': '0-1 DTE', 'value': '0_1dte'},
                    {'label': '0DTE Only', 'value': '0dte'},
                    {'label': '0-2 DTE', 'value': '0_2dte'},
                    {'label': 'All DTE', 'value': 'all'}
                ],
                value='0_1dte',
                style={'width': '160px'}
            )
        ]),
        html.Div([
            html.Label("Compare:", style={'marginRight': '6px', 'color': theta_dashboard.C['text_muted']}),
            dcc.Dropdown(
                id='compare-dropdown',
                options=[
                    {'label': 'No Compare', 'value': 0},
                    {'label': 'vs 5 min ago', 'value': 5},
                    {'label': 'vs 15 min ago', 'value': 15},
                    {'label': 'vs 30 min ago', 'value': 30},
                    {'label': 'vs 1 hr ago', 'value': 60},
                    {'label': 'vs 2 hr ago', 'value': 120}
                ],
                value=0,
                style={'width': '190px'}
            )
        ]),
        html.Div([
            html.Label("Window:", style={'marginRight': '6px', 'color': theta_dashboard.C['text_muted']}),
            dcc.Dropdown(
                id='window-dropdown',
                options=[
                    {'label': 'Full Session', 'value': 'session'},
                    {'label': '15 min', 'value': 15},
                    {'label': '30 min', 'value': 30},
                    {'label': '45 min', 'value': 45},
                    {'label': '60 min', 'value': 60}
                ],
                value='session',
                style={'width': '160px'}
            )
        ]),
        
        html.Button('START FETCHER', id='btn-start', style={'backgroundColor': theta_dashboard.C['call'], 'color': 'white', 'border': 'none', 'padding': '8px 14px', 'borderRadius': '6px', 'cursor': 'pointer', 'fontWeight': 700, 'height': '36px'}),
        html.Button('STOP FETCHER', id='btn-stop', style={'backgroundColor': theta_dashboard.C['put'], 'color': 'white', 'border': 'none', 'padding': '8px 14px', 'borderRadius': '6px', 'cursor': 'pointer', 'fontWeight': 700, 'height': '36px'}),
        html.Button('⏸ PAUSE REFRESH', id='btn-pause', n_clicks=0, style={'backgroundColor': theta_dashboard.C['warning'], 'color': 'white', 'border': 'none', 'padding': '8px 14px', 'borderRadius': '6px', 'cursor': 'pointer', 'fontWeight': 700, 'height': '36px'}),
        html.Button('REFRESH', id='btn-refresh', n_clicks=0, style={'backgroundColor': theta_dashboard.C['accent'], 'color': 'white', 'border': 'none', 'padding': '8px 14px', 'borderRadius': '6px', 'cursor': 'pointer', 'fontWeight': 700, 'height': '36px'}),
        
        html.Div(id='fetcher-status', style={'marginLeft': 'auto', 'fontSize': '0.85em', 'color': theta_dashboard.C['text_muted']})
    ]),
    
        # Auto-refresh interval (10 seconds)
        dcc.Interval(id='interval-update', interval=10*1000, n_intervals=0),
        html.Div(id='action-trigger', style={'display': 'none'}),
    
        # Dashboard Content
        html.Div(id='dashboard-content')
    ]
)


@app.callback(
    [Output("refresh-paused", "data"), Output("btn-pause", "children")],
    Input("btn-pause", "n_clicks"),
    State("refresh-paused", "data"),
    prevent_initial_call=True,
)
def toggle_pause(n_clicks, paused):
    paused = not bool(paused)
    return paused, ("▶ RESUME REFRESH" if paused else "⏸ PAUSE REFRESH")

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
        theta_dashboard.start_fetcher()
    elif button_id == 'btn-stop':
        theta_dashboard.stop_fetcher()
        
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
    ctx = dash.callback_context
    triggered_by = (ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "")
    if paused and triggered_by == "interval-update":
        return no_update, no_update, no_update, no_update

    df_agg, df_snap = load_data(dte_filter=dte)
    model_out = _predict_live_model(df_agg, df_snap)
    latest_batch_id = None
    try:
        latest_batch_id = _latest_batch_id(df_agg)
    except Exception:
        latest_batch_id = None
    _record_model_roll(latest_batch_id, model_out)
    
    status_html = []
    if theta_dashboard.is_fetcher_running():
        fs = theta_dashboard.get_fetcher_status()
        status_html = html.Span([
            html.Span("Running ", style={'color': theta_dashboard.C['call']}),
            html.Span(f"Batch #{fs.get('batch_id', '?')} | PID {fs.get('pid', '?')}", style={'color': theta_dashboard.C['text_muted']})
        ])
    else:
        status_html = html.Span("Stopped", style={'color': theta_dashboard.C['put']})

    # Header status + subheader (match Jupyter)
    all_symbols = df_agg["symbol"].unique().tolist() if (df_agg is not None and not df_agg.empty and "symbol" in df_agg.columns) else []
    snap_count = 0
    try:
        snap_count = len(theta_dashboard.list_available_snapshots())
    except Exception:
        pass
    total_batches = int(df_agg["batch_id"].max()) if (df_agg is not None and not df_agg.empty and "batch_id" in df_agg.columns) else 0
    live_badge = (
        html.Span([
            html.Span("● LIVE", style={"color": theta_dashboard.C["call"]}),
            html.Span(f"  Batch #{fs.get('batch_id', '?')}", style={"color": theta_dashboard.C["text_muted"]}),
        ])
        if theta_dashboard.is_fetcher_running()
        else (html.Span([
            html.Span("● REVIEWING", style={"color": theta_dashboard.C["warning"]}),
            html.Span(f"  {snap_count} snapshots available", style={"color": theta_dashboard.C["text_muted"]}),
        ]) if snap_count > 0 else html.Span([html.Span("● STOPPED", style={"color": theta_dashboard.C["put"]})]))
    )

    last_update = "No data yet"
    if df_agg is not None and not df_agg.empty and "_ts_parsed" in df_agg.columns and df_agg["_ts_parsed"].notna().any():
        try:
            last_ts = df_agg["_ts_parsed"].max()
            if pd.notna(last_ts):
                age = datetime.now() - last_ts.to_pydatetime()
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
    subheader = (
        f"Last data: {last_update} • {total_batches} batches • {snap_count} snapshots • "
        f"Window: {theta_dashboard.MARKET_OPEN_ET[0]}:{theta_dashboard.MARKET_OPEN_ET[1]:02d}-"
        f"{theta_dashboard.MARKET_CLOSE_ET[0]}:{theta_dashboard.MARKET_CLOSE_ET[1]:02d} ET • "
        f"{'Auto-refresh: 10s' if theta_dashboard.is_fetcher_running() else 'Market closed / Fetcher stopped'} • "
        f"{('Compare: vs ' + str(compare) + ' min ago • ') if int(compare) > 0 else ''}"
        f"DTE: {str(dte).replace('_','-').upper() if dte != 'all' else 'ALL'}"
    )
    
    if df_agg.empty and df_snap.empty:
        return html.Div([
            html.H3("Waiting for data...", style={'color': theta_dashboard.C['warning']}),
            html.P("Make sure the fetcher is running and data is being collected in /workspace/daily_data/")
        ]), status_html, live_badge, subheader
    
    content = []
    latest_stats = get_latest_stats(df_agg, df_snap)
    
    header_style = {
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start',
        'backgroundColor': theta_dashboard.C['bg_card'], 'padding': '20px',
        'borderRadius': '8px', 'marginBottom': '20px', 'border': f'1px solid {theta_dashboard.C["border"]}'
    }
    
    header_content = [
        html.Div([
            html.H2(f"Theta Options Intelligence — {symbol}", style={'margin': '0 0 5px 0', 'color': theta_dashboard.C['text']}),
            html.Div(f"DTE Filter: {dte} | Compare: {compare} min | Window: {window}", style={'color': theta_dashboard.C['text_muted']})
        ])
    ]
    
    if symbol != 'ALL' and symbol in latest_stats:
        st = latest_stats[symbol]
        price_color = theta_dashboard.C['call'] if st['price_change'] >= 0 else theta_dashboard.C['put']
        
        stats_boxes = html.Div(style={'display': 'flex', 'gap': '15px'}, children=[
            html.Div(style={'textAlign': 'right'}, children=[
                html.Div("Underlying Price", style={'color': theta_dashboard.C['text_muted'], 'fontSize': '12px', 'textTransform': 'uppercase'}),
                html.Div([
                    html.Span(f"${st['price']:.2f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': theta_dashboard.C['text']}),
                    html.Span(f" ({st['price_change']:+.2f}%)", style={'fontSize': '14px', 'color': price_color, 'marginLeft': '5px'})
                ])
            ]),
            html.Div(style={'width': '1px', 'backgroundColor': theta_dashboard.C['border'], 'margin': '0 10px'}),
            html.Div(style={'textAlign': 'right'}, children=[
                html.Div("Volume Ratio (C/P)", style={'color': theta_dashboard.C['text_muted'], 'fontSize': '12px', 'textTransform': 'uppercase'}),
                html.Div(f"{st['vol_ratio']:.2f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': theta_dashboard.C['call'] if st['vol_ratio'] >= 1 else theta_dashboard.C['put']})
            ]),
            html.Div(style={'width': '1px', 'backgroundColor': theta_dashboard.C['border'], 'margin': '0 10px'}),
            html.Div(style={'textAlign': 'right'}, children=[
                html.Div("Net Gamma", style={'color': theta_dashboard.C['text_muted'], 'fontSize': '12px', 'textTransform': 'uppercase'}),
                html.Div(f"{st['net_gamma']/1e6:.1f}M", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': theta_dashboard.C['call'] if st['net_gamma'] >= 0 else theta_dashboard.C['put']})
            ])
        ])
        header_content.append(stats_boxes)

    model_card = _model_signal_card(model_out)
    if model_card is not None:
        header_content.append(model_card)
        
    content.append(html.Div(style=header_style, children=header_content))
    
    if symbol != 'ALL':
        # Spot price for this symbol (used by several charts)
        spot_raw = 0.0
        if not df_agg.empty and "symbol" in df_agg.columns:
            sym_agg = df_agg[df_agg["symbol"] == symbol]
            if not sym_agg.empty and "spot" in sym_agg.columns:
                spot_raw = float(sym_agg.iloc[-1].get("spot", 0.0) or 0.0)

        fig_em = _create_expected_move_chart(df_agg, symbol, model_out)
        if fig_em is not None:
            content.append(html.H3("Expected Move Overlay", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '20px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_em, style={'height': '430px'}))

        fig_roll = _create_model_rollover_chart()
        if fig_roll is not None:
            content.append(html.H3("Model Rollover Prediction", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_roll, style={'height': '360px'}))

        fig_meter = _create_signal_meters(model_out)
        if fig_meter is not None:
            content.append(html.H3("Signal Meters", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_meter, style={'height': '520px'}))

        # Time-series metrics (P/C, net GEX, IV skew, straddle, etc.)
        try:
            ts_charts = theta_dashboard.create_timeseries_individual(df_agg, symbol, window_minutes=window)
        except Exception:
            ts_charts = []
        if ts_charts:
            content.append(html.H3("Time-Series Metrics", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '20px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            for fig, box in ts_charts:
                content.append(dcc.Graph(figure=fig, style={'height': '300px'}))
                ins = _insight(box)
                if ins is not None:
                    content.append(ins)

        # Market microstructure (spreads, imbalance, trade aggression, avg size)
        try:
            micro_charts = theta_dashboard.create_microstructure_individual(df_agg, symbol, window_minutes=window)
        except Exception:
            micro_charts = []
        if micro_charts:
            content.append(html.H3("Market Microstructure", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            for fig, box in micro_charts:
                content.append(dcc.Graph(figure=fig, style={'height': '320px'}))
                ins = _insight(box)
                if ins is not None:
                    content.append(ins)

        # Gamma exposure profile
        try:
            fig_gamma = theta_dashboard.create_gamma_chart(df_snap, symbol, spot_raw, lookback_df=None)
        except Exception:
            fig_gamma = None
        if fig_gamma is not None:
            content.append(html.H3("Gamma Exposure Profile", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_gamma, style={'height': '400px'}))
            try:
                text, anomaly = theta_dashboard.gamma_chart_insight(df_snap, symbol, spot_raw)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Key strike levels
        try:
            fig_strike = theta_dashboard.create_strike_chart(df_snap, symbol, spot_raw, lookback_df=None)
        except Exception:
            fig_strike = None
        if fig_strike is not None:
            content.append(html.H3("Key Strike Levels", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_strike, style={'height': '500px'}))
            try:
                text, anomaly = theta_dashboard.strike_chart_insight(df_snap, symbol, spot_raw)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Vol/OI ratio
        try:
            fig_vol_oi = theta_dashboard.create_vol_oi_chart(df_snap, symbol, spot_raw)
        except Exception:
            fig_vol_oi = None
        if fig_vol_oi is not None:
            content.append(html.H3("Vol/OI Ratio (Live)", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_vol_oi, style={'height': '400px'}))
            try:
                text, anomaly = theta_dashboard.vol_oi_insight(df_snap, symbol)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # IV term structure
        try:
            fig_iv = theta_dashboard.create_iv_chart(df_snap, symbol)
        except Exception:
            fig_iv = None
        if fig_iv is not None:
            content.append(html.H3("IV Term Structure", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_iv, style={'height': '400px'}))
            try:
                text, anomaly = theta_dashboard.iv_chart_insight(df_snap, symbol)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Vanna exposure
        try:
            fig_vanna = theta_dashboard.create_vanna_chart(df_snap, symbol, spot_raw)
        except Exception:
            fig_vanna = None
        if fig_vanna is not None:
            content.append(html.H3("Vanna Exposure", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_vanna, style={'height': '400px'}))
            try:
                text, anomaly = theta_dashboard.vanna_chart_insight(df_snap, symbol)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Dealer positioning
        try:
            fig_dealer = theta_dashboard.create_dealer_chart(df_snap, symbol)
        except Exception:
            fig_dealer = None
        if fig_dealer is not None:
            content.append(html.H3("Dealer Positioning", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_dealer, style={'height': '400px'}))
            try:
                text, anomaly = theta_dashboard.dealer_chart_insight(df_snap, symbol)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # OI walls & pinning
        try:
            fig_oi = theta_dashboard.create_oi_walls_chart(df_snap, symbol, spot_raw)
        except Exception:
            fig_oi = None
        if fig_oi is not None:
            content.append(html.H3("OI Walls & Pinning", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_oi, style={'height': '400px'}))
            try:
                text, anomaly = theta_dashboard.oi_walls_insight(df_snap, symbol)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Expiration concentration (DTE buckets)
        try:
            fig_dte = theta_dashboard.create_dte_concentration_chart(df_snap, symbol)
        except Exception:
            fig_dte = None
        if fig_dte is not None:
            content.append(html.H3("Expiration Concentration", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_dte, style={'height': '400px'}))
            try:
                text, anomaly = theta_dashboard.dte_concentration_insight(df_snap, symbol)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Cumulative volume delta
        try:
            fig_vol = theta_dashboard.create_cum_vol_delta_chart(df_agg, symbol, window_minutes=window)
        except Exception:
            fig_vol = None
        if fig_vol is not None:
            content.append(html.H3("Cumulative Volume Delta", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_vol, style={'height': '400px'}))
            try:
                text, anomaly = theta_dashboard.cum_vol_delta_insight(df_agg, symbol, window_minutes=window)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Options flow history
        try:
            fig_flow = theta_dashboard.create_flow_chart(df_agg, symbol)
        except Exception:
            fig_flow = None
        if fig_flow is not None:
            content.append(html.H3("Options Flow History", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_flow, style={'height': '400px'}))
            try:
                text, anomaly = theta_dashboard.flow_chart_insight(df_agg, symbol)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Market maker flow changes
        try:
            fig_mm = theta_dashboard.create_mm_flow_chart(df_agg, [symbol])
        except Exception:
            fig_mm = None
        if fig_mm is not None:
            content.append(html.H3("Market Maker Flow Changes", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_mm, style={'height': '350px'}))
            try:
                text, anomaly = theta_dashboard.mm_flow_insight(df_agg, [symbol])
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # VIX hedging section when symbol itself is VIX/VIXW
        if symbol in ("VIX", "VIXW"):
            try:
                fig_vix_flow = theta_dashboard.create_vix_put_flow_chart(df_agg)
            except Exception:
                fig_vix_flow = None
            if fig_vix_flow is not None:
                content.append(html.H3("VIX Put Flow", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
                content.append(dcc.Graph(figure=fig_vix_flow, style={'height': '350px'}))
                try:
                    text, anomaly = theta_dashboard.vix_put_flow_insight(df_agg)
                    ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                    if ins is not None:
                        content.append(ins)
                except Exception:
                    pass

            try:
                fig_vix_hedge = theta_dashboard.create_vix_hedging_chart(df_snap)
            except Exception:
                fig_vix_hedge = None
            if fig_vix_hedge is not None:
                content.append(html.H3("VIX Institutional Hedging", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
                content.append(dcc.Graph(figure=fig_vix_hedge, style={'height': '400px'}))
                try:
                    text, anomaly = theta_dashboard.vix_hedging_insight(df_snap)
                    ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                    if ins is not None:
                        content.append(ins)
                except Exception:
                    pass

    else:
        # Cross-symbol views when ALL is selected
        try:
            fig_gamma = theta_dashboard.create_multi_gamma_chart(df_agg)
        except Exception:
            fig_gamma = None
        if fig_gamma is not None:
            content.append(html.H3("Cross-Symbol Gamma Comparison", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '20px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_gamma, style={'height': '400px'}))

        try:
            fig_sent = theta_dashboard.create_multi_sentiment_chart(df_agg)
        except Exception:
            fig_sent = None
        if fig_sent is not None:
            content.append(html.H3("Cross-Symbol Sentiment", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
            content.append(dcc.Graph(figure=fig_sent, style={'height': '400px'}))

        has_vix = not df_agg.empty and "symbol" in df_agg.columns and any(
            s in ("VIX", "VIXW") for s in df_agg["symbol"].unique()
        )
        if has_vix:
            try:
                fig_vix = theta_dashboard.create_vix_put_flow_chart(df_agg)
            except Exception:
                fig_vix = None
            if fig_vix is not None:
                content.append(html.H3("VIX Institutional Hedging", style={'color': theta_dashboard.C['text_sec'], 'marginTop': '30px', 'borderBottom': f'1px solid {theta_dashboard.C["border"]}', 'paddingBottom': '10px'}))
                content.append(dcc.Graph(figure=fig_vix, style={'height': '350px'}))

    return html.Div(content), status_html, live_badge, subheader

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
