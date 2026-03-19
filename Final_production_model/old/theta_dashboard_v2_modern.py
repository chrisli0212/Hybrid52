"""
Theta Options Intelligence Dashboard — Modern Pro Terminal

Reads three CSV files:
  - daily_data/theta_agg.csv       (market data)
  - daily_data/theta_snapshot.csv  (strike-level data)
  - daily_data/prediction.csv      (model predictions from prediction_service.py)

NO PyTorch dependency. NO model loading. Pure CSV reading + Dash rendering.
Modern Bloomberg/SpotGamma-inspired design with glassmorphism cards.
"""

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

# ---------------------------------------------------------------------------
# Load the original theta_dashboard module for chart-building functions
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("theta_dashboard", "/workspace/theta_dashboard_v3_10.py")
theta_dashboard = importlib.util.module_from_spec(spec)
sys.modules["theta_dashboard"] = theta_dashboard

import unittest.mock
sys.modules['IPython.display'] = unittest.mock.Mock()
sys.modules['ipywidgets'] = unittest.mock.Mock()

spec.loader.exec_module(theta_dashboard)

# ---------------------------------------------------------------------------
# Script-relative paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "daily_data"

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
            ts_dt = datetime.now()

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

    now_str = datetime.now().strftime("%H:%M:%S")
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
        elapsed = (datetime.now() - _last_live_non_suppressed_ts).total_seconds()
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
                        "backgroundColor": f"{conf_badge_color}22",
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
        bg = f"{MC['accent']}18" if is_active else "transparent"
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
            fill="tozeroy", fillcolor=f"{color}18",
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

    layout_cfg = theta_dashboard.base_layout(title="Model Signal Meters", height=760)
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
# Enhanced Expected Move Chart
# ---------------------------------------------------------------------------

def _create_expected_move_chart(df_agg, symbol, model_out, pred_history_roll):
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
        tail = hist.tail(min(60, len(hist)))
        atm_straddle = float(max(0.0, tail["spot"].max() - tail["spot"].min()) / 2.0)
    if not np.isfinite(atm_straddle) or atm_straddle <= 0:
        atm_straddle = max(spot * 0.01, 0.5)

    p_up = float((model_out or {}).get("prob", 0.5) or 0.5)
    p_dn = 1.0 - p_up
    confidence = float((model_out or {}).get("confidence", abs(p_up - 0.5) * 2.0) or 0.0)
    suppressed = bool((model_out or {}).get("suppressed", False))
    ok = bool((model_out or {}).get("ok", True))
    stronger_up = p_up >= p_dn
    up_width = 4 if stronger_up else 2
    dn_width = 4 if not stronger_up else 2
    up_alpha = 0.95 if stronger_up else 0.55
    dn_alpha = 0.95 if not stronger_up else 0.55

    horizon_min = 30
    now_ts = hist["_ts_parsed"].iloc[-1]
    future_x = [now_ts + pd.Timedelta(minutes=i) for i in range(horizon_min + 1)]
    t = np.linspace(0, 1, horizon_min + 1)
    em = atm_straddle * np.sqrt(t)
    up_path = spot + em
    dn_path = spot - em

    fig = go.Figure()

    # Compute tracking status
    tracking_status = "N/A"
    tracking_color = MC["text_muted"]
    if not suppressed and ok and len(hist) >= 5:
        recent_spots = hist["spot"].tail(10)
        if len(recent_spots) >= 2:
            spot_delta = float(recent_spots.iloc[-1]) - float(recent_spots.iloc[0])
            predicted_up = p_up >= 0.5
            moved_up = spot_delta >= 0
            if predicted_up == moved_up:
                tracking_status = "ON PATH"
                tracking_color = MC["call"]
            elif abs(spot_delta) < atm_straddle * 0.1:
                tracking_status = "NEUTRAL"
                tracking_color = MC["warning"]
            elif abs(spot_delta) > atm_straddle * 0.5:
                tracking_status = "INVALIDATED"
                tracking_color = MC["put"]
            else:
                tracking_status = "DIVERGING"
                tracking_color = MC["warning"]

    # Background shading for prediction tracking
    if not suppressed and ok and len(hist) >= 2:
        hist_x = list(hist["_ts_parsed"])
        hist_spots = list(hist["spot"].astype(float))
        anchor = hist_spots[0]
        for i in range(1, len(hist_spots)):
            seg_start = hist_x[i - 1]
            seg_end = hist_x[i]
            predicted_up = p_up >= 0.5
            spot_moved_up = hist_spots[i] >= anchor

            if predicted_up == spot_moved_up:
                shade_color = "rgba(16,185,129,0.06)"
            else:
                shade_color = "rgba(239,68,68,0.06)"

            fig.add_shape(
                type="rect", x0=seg_start, x1=seg_end,
                y0=min(hist_spots) - atm_straddle * 0.1,
                y1=max(hist_spots) + atm_straddle * 0.1,
                fillcolor=shade_color, line_width=0, layer="below",
            )

    # Historical spot path
    fig.add_trace(
        go.Scatter(
            x=hist["_ts_parsed"],
            y=hist["spot"],
            mode="lines",
            name="Recent Spot",
            line=dict(color=MC["text_sec"], width=2),
        )
    )

    # Expected-move fan area
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
    fig.add_hline(y=spot, line_dash="dash", line_color=MC["accent"], annotation_text=f"Spot {spot:.2f}")
    y_min = float(np.nanmin(dn_path))
    y_max = float(np.nanmax(up_path))
    fig.add_shape(
        type="line",
        x0=now_ts, x1=now_ts,
        y0=y_min, y1=y_max,
        line=dict(dash="dot", color=MC["warning"], width=1),
    )
    fig.add_annotation(
        x=now_ts, y=y_max,
        text="+30m horizon",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(color=MC["warning"], size=10),
        bgcolor="rgba(15,23,42,0.70)",
    )
    fig.add_annotation(
        x=future_x[-1],
        y=up_path[-1] if stronger_up else dn_path[-1],
        text=f"{'UP' if stronger_up else 'DOWN'} favored | Conf {confidence*100:.0f}%",
        showarrow=False, xanchor="right", yanchor="bottom",
        bgcolor="rgba(15,23,42,0.75)",
        bordercolor=MC["border"],
        font=dict(color=MC["text_sec"], size=11),
    )

    if tracking_status != "N/A":
        fig.add_annotation(
            x=hist["_ts_parsed"].iloc[len(hist) // 2],
            y=y_max,
            text=tracking_status,
            showarrow=False, xanchor="center", yanchor="top",
            font=dict(color=tracking_color, size=13, family="monospace"),
            bgcolor="rgba(15,23,42,0.80)",
            bordercolor=tracking_color,
            borderwidth=1, borderpad=4,
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


# ---------------------------------------------------------------------------
# Model Rollover Chart (reads from prediction CSV history)
# ---------------------------------------------------------------------------

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
    fig.update_layout(**theta_dashboard.base_layout(title="Stage 3 Rollover Prediction", height=360))
    return theta_dashboard.style_axes(fig)


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
        elapsed = datetime.now() - _last_live_non_suppressed_ts
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

    agg_file = theta_dashboard.AGG_FILE
    current_agg_mtime = agg_file.stat().st_mtime if agg_file.exists() else 0

    if _cached_agg_df is None or current_agg_mtime != _cached_agg_mtime:
        agg_df = theta_dashboard.load_agg_data()
        if agg_df is None:
            agg_df = pd.DataFrame()
        _cached_agg_df = agg_df
        _cached_agg_mtime = current_agg_mtime
        _cached_filtered_data.clear()
    else:
        agg_df = _cached_agg_df

    snap_file = theta_dashboard.SNAPSHOT_FILE
    current_snap_mtime = snap_file.stat().st_mtime if snap_file.exists() else 0

    if _cached_snap_df is None or current_snap_mtime != _cached_snap_mtime:
        snap_df = theta_dashboard.load_snapshot_data()
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
        filtered_agg = theta_dashboard._apply_dte_filter_agg(filtered_agg, dte_filter)
    except Exception:
        pass
    try:
        filtered_snap = theta_dashboard._apply_dte_filter(filtered_snap, dte_filter)
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

MC = {
    "bg_dark": "#0a0a0f",
    "bg_card": "#12121a",
    "bg_card_hover": "#1a1a28",
    "bg_input": "#1e1e2e",
    "border": "rgba(99,102,241,0.15)",
    "border_active": "rgba(99,102,241,0.4)",
    "text": "#e2e8f0",
    "text_sec": "#94a3b8",
    "text_muted": "#64748b",
    "accent": "#6366f1",
    "accent_glow": "rgba(99,102,241,0.2)",
    "call": "#22c55e",
    "put": "#ef4444",
    "warning": "#f59e0b",
    "info": "#3b82f6",
}

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
        _last_live_non_suppressed_ts = datetime.now()

    # Generate alerts from latest prediction
    _generate_alerts(model_out)

    # Fetcher status
    status_html = []
    fs = {}
    if theta_dashboard.is_fetcher_running():
        fs = theta_dashboard.get_fetcher_status()
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
        snap_count = len(theta_dashboard.list_available_snapshots())
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
        if theta_dashboard.is_fetcher_running()
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

    # Prediction status for subheader
    pred_status = "No predictions"
    if model_out:
        if model_out.get("suppressed"):
            pred_status = f"Suppressed: {model_out.get('reason', '')}"
        else:
            pred_status = f"P(up)={model_out['prob']:.1%} | {model_out.get('direction', '?')}"

    subheader = (
        f"Last data: {last_update} | {total_batches} batches | {snap_count} snapshots | "
        f"Window: {theta_dashboard.MARKET_OPEN_ET[0]}:{theta_dashboard.MARKET_OPEN_ET[1]:02d}-"
        f"{theta_dashboard.MARKET_CLOSE_ET[0]}:{theta_dashboard.MARKET_CLOSE_ET[1]:02d} ET | "
        f"{'Auto-refresh: 10s' if theta_dashboard.is_fetcher_running() else 'Market closed / Fetcher stopped'} | "
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

            # 3. Enhanced Expected Move Chart
            fig_em = _create_expected_move_chart(df_agg, symbol, model_out, pred_history_roll)
            if fig_em is not None:
                content.append(_mc_section_header("Expected Move Overlay"))
                content.append(dcc.Graph(figure=fig_em, style={'height': '430px'}))

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
            ts_charts = theta_dashboard.create_timeseries_individual(df_agg, symbol, window_minutes=window)
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
            micro_charts = theta_dashboard.create_microstructure_individual(df_agg, symbol, window_minutes=window)
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
            fig_gamma = theta_dashboard.create_gamma_chart(df_snap, symbol, spot_raw, lookback_df=None)
        except Exception:
            fig_gamma = None
        if fig_gamma is not None:
            content.append(_mc_section_header("Gamma Exposure Profile"))
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
            content.append(_mc_section_header("Key Strike Levels"))
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
            content.append(_mc_section_header("Vol/OI Ratio (Live)"))
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
            content.append(_mc_section_header("IV Term Structure"))
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
            content.append(_mc_section_header("Vanna Exposure"))
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
            content.append(_mc_section_header("Dealer Positioning"))
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
            content.append(_mc_section_header("OI Walls & Pinning"))
            content.append(dcc.Graph(figure=fig_oi, style={'height': '400px'}))
            try:
                text, anomaly = theta_dashboard.oi_walls_insight(df_snap, symbol)
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # Expiration concentration
        try:
            fig_dte = theta_dashboard.create_dte_concentration_chart(df_snap, symbol)
        except Exception:
            fig_dte = None
        if fig_dte is not None:
            content.append(_mc_section_header("Expiration Concentration"))
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
            content.append(_mc_section_header("Cumulative Volume Delta"))
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
            content.append(_mc_section_header("Options Flow History"))
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
            content.append(_mc_section_header("Market Maker Flow Changes"))
            content.append(dcc.Graph(figure=fig_mm, style={'height': '350px'}))
            try:
                text, anomaly = theta_dashboard.mm_flow_insight(df_agg, [symbol])
                ins = _insight(theta_dashboard.implication_box_html(text, anomaly) if text else "")
                if ins is not None:
                    content.append(ins)
            except Exception:
                pass

        # VIX hedging section
        if symbol in ("VIX", "VIXW"):
            try:
                fig_vix_flow = theta_dashboard.create_vix_put_flow_chart(df_agg)
            except Exception:
                fig_vix_flow = None
            if fig_vix_flow is not None:
                content.append(_mc_section_header("VIX Put Flow"))
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
                content.append(_mc_section_header("VIX Institutional Hedging"))
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
            content.append(_mc_section_header("Cross-Symbol Gamma Comparison"))
            content.append(dcc.Graph(figure=fig_gamma, style={'height': '400px'}))

        try:
            fig_sent = theta_dashboard.create_multi_sentiment_chart(df_agg)
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
                fig_vix = theta_dashboard.create_vix_put_flow_chart(df_agg)
            except Exception:
                fig_vix = None
            if fig_vix is not None:
                content.append(_mc_section_header("VIX Institutional Hedging"))
                content.append(dcc.Graph(figure=fig_vix, style={'height': '350px'}))

    return html.Div(content), status_html, live_badge, subheader


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
