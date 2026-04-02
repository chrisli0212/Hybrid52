"""
Agent TQ Validator
Per-block quality checks for the TQ static/sequence features.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from .feature_config import (
    AGENT_TQ_SOURCE_BLOCKS,
    AGENT_TQ_INPUT_DIM,
    AGENT_TQ_INPUT_DIM_HIST,
)

# Cumulative block boundaries in the assembled TQ vector (historical mode)
_HIST_BLOCK_LAYOUT = [
    ("flow_volume",      0,  30),
    ("microstructure",  30,  50),
    ("sentiment_regime",50,  70),
]
_LIVE_BLOCK_LAYOUT = _HIST_BLOCK_LAYOUT + [
    ("phase1_smart_money",    70,  85),
    ("phase1_quote_pressure", 85,  95),  # model clips at 95 via _fix_feat_dim
]


def validate_agent_tq(
    features: np.ndarray,
    historical_mode: Optional[bool] = None,
    raise_on_dim: bool = True,
) -> Dict:
    """
    Validate the Agent-TQ static feature vector.

    Args:
        features:        np.ndarray of shape (70,) hist or (95,) live
        historical_mode: auto-detected from shape if None
        raise_on_dim:    if True, raises on unexpected shape

    Returns:
        dict with per-block stats and alert list
    """
    if features.ndim != 1:
        features = features.flatten()

    D = features.shape[0]
    if historical_mode is None:
        historical_mode = (D == AGENT_TQ_INPUT_DIM_HIST)

    expected_dim = AGENT_TQ_INPUT_DIM_HIST if historical_mode else AGENT_TQ_INPUT_DIM
    if raise_on_dim:
        assert D == expected_dim, (
            f"AgentTQ validator: got dim {D}, expected {expected_dim} "
            f"({'historical' if historical_mode else 'live'} mode)"
        )

    layout = _HIST_BLOCK_LAYOUT if historical_mode else _LIVE_BLOCK_LAYOUT
    alerts: List[str] = []
    block_stats: Dict = {}

    for block_name, s, e in layout:
        if e > D:
            break   # live blocks absent in historical vector
        block = features[s:e]

        nan_rate  = float(np.isnan(block).mean())
        zero_rate = float((block == 0).mean())
        mean_val  = float(np.nanmean(block)) if not np.all(np.isnan(block)) else float("nan")

        block_stats[block_name] = {
            "indices":   (s, e),
            "nan_rate":  nan_rate,
            "zero_rate": zero_rate,
            "mean":      mean_val,
        }

        if nan_rate > 0.05:
            alerts.append(f"AgentTQ [{block_name}]: nan_rate={nan_rate:.1%}")
        if zero_rate > 0.60:
            alerts.append(
                f"AgentTQ [{block_name}]: zero_rate={zero_rate:.1%} — "
                + ("Phase1 features absent in historical mode" if "phase1" in block_name
                   else "check upstream extractor")
            )

    # Microstructure-specific: spread_mean should never be 0 in liquid session
    spread_mean_idx = 30   # first feature of microstructure block
    if D > spread_mean_idx and features[spread_mean_idx] == 0.0:
        alerts.append(
            "AgentTQ [microstructure]: spread_mean=0 — possible pre-market or data gap"
        )

    return {
        "agent":          "TQ",
        "feature_shape":  (D,),
        "historical_mode": historical_mode,
        "block_stats":    block_stats,
        "alerts":         alerts,
        "passed":         len(alerts) == 0,
    }
