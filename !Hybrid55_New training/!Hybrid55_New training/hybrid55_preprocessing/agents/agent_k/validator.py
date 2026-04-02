"""
Agent K Validator
Per-sub-block quality checks for the GREEK_BY_STRIKE feature block.

Key insight: bucket_greeks has 5 delta-buckets × 13 greeks.
A high zero-rate in deep-ITM or deep-OTM buckets is NORMAL (sparse chains).
A high zero-rate in the ATM bucket is NOT normal and indicates a data gap.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from .feature_config import (
    AGENT_K_INPUT_DIM,
    AGENT_K_SUB_BLOCKS,
    AGENT_K_FEATURE_NAMES,
)

# ATM bucket within bucket_greeks: positions 26–38 (bucket index 2 of 5)
_ATM_BUCKET_SLICE = slice(26, 39)
_GREEKS_PER_BUCKET = 13


def validate_agent_k(features: np.ndarray, raise_on_dim: bool = True) -> Dict:
    """
    Validate Agent K feature vector.

    Args:
        features:     np.ndarray of shape (75,)
        raise_on_dim: raise AssertionError if shape is wrong

    Returns:
        dict with sub-block stats and alert list
    """
    if features.ndim != 1:
        features = features.flatten()

    D = features.shape[0]
    if raise_on_dim:
        assert D == AGENT_K_INPUT_DIM, (
            f"AgentK validator: got dim {D}, expected {AGENT_K_INPUT_DIM}"
        )

    alerts: List[str] = []
    block_stats: Dict = {}

    for block_name, (s, e) in AGENT_K_SUB_BLOCKS.items():
        if e > D:
            break
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
            alerts.append(f"AgentK [{block_name}]: nan_rate={nan_rate:.1%}")

    # ATM bucket: zero-rate > 30% is suspicious (unlike deep-OTM which is expected)
    atm_block = features[_ATM_BUCKET_SLICE]
    atm_zero_rate = float((atm_block == 0).mean())
    if atm_zero_rate > 0.30:
        alerts.append(
            f"AgentK [atm_bucket]: zero_rate={atm_zero_rate:.1%} — "
            "possible empty ATM chain or pre-market snapshot"
        )

    # Deep-ITM/OTM zero-rate info only (not an alert)
    deep_otm_block = features[0:_GREEKS_PER_BUCKET]
    deep_itm_block = features[4 * _GREEKS_PER_BUCKET: 5 * _GREEKS_PER_BUCKET]
    block_stats["deep_otm_zero_rate"] = float((deep_otm_block == 0).mean())
    block_stats["deep_itm_zero_rate"] = float((deep_itm_block == 0).mean())

    # Gamma-squeeze candidate: check if ATM gamma is well above surrounding buckets
    atm_gamma_idx   = 26 + 1   # bucket_greeks[atm bucket, gamma position=1]
    otm_gamma_idx   = 13 + 1   # otm bucket
    itm_gamma_idx   = 39 + 1   # itm bucket
    if D > max(atm_gamma_idx, otm_gamma_idx, itm_gamma_idx):
        atm_g  = features[atm_gamma_idx]
        otm_g  = features[otm_gamma_idx]
        itm_g  = features[itm_gamma_idx]
        block_stats["gamma_peak_ratio"] = (
            float(atm_g / (0.5 * (otm_g + itm_g) + 1e-8))
        )

    return {
        "agent":         "K",
        "feature_shape": (D,),
        "block_stats":   block_stats,
        "alerts":        alerts,
        "passed":        len(alerts) == 0,
    }
