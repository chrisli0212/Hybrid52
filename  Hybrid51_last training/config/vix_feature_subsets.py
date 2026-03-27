"""VIX-specific feature schema and optional registration helper."""

from __future__ import annotations

from copy import deepcopy

VIX_FEATURE_NAMES = [
    "vix_level",
    "vix_pct_5m",
    "vix_pct_15m",
    "vix_pct_1h",
    "vix_zscore_15m",
    "vix_percentile_1h",
    "vix_term_slope",
    "vvix_level",
    "vix_vix1d_spread",
    "vix_hilo_range",
]

AGENT_V_CONFIG = {
    "name": "V (VIX Regime)",
    "ranges": [],
    "feat_dim": len(VIX_FEATURE_NAMES),
    "use_backbone": False,
    "separate_input_pipeline": True,
    "input_symbol": "VIXW",
}


def register_vix_agent(agent_feature_subsets: dict) -> dict:
    """Return a copy of feature subset config including Agent V entry."""
    out = deepcopy(agent_feature_subsets)
    out["V"] = dict(AGENT_V_CONFIG)
    return out
