"""
Agent C Validator
Per-block quality checks for the 69-dim Agent C feature vector.

Agent C is a sequence model, so the validator also checks a full
(T, 69) sequence for temporal consistency (e.g. all-zero bars).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from .feature_config import AGENT_C_INPUT_DIM

# Assembled block layout in the 69-dim vector
_BLOCK_LAYOUT = [
    ("gamma_exposure",  0,  30),
    ("vanna_charm",    30,  50),
    ("iv_surface_core",50,  66),
    ("csv_distatm",    66,  69),
]


def validate_agent_c_static(
    features: np.ndarray,
    raise_on_dim: bool = True,
) -> Dict:
    """
    Validate Agent C single-bar feature vector.

    Args:
        features:     np.ndarray of shape (69,)
        raise_on_dim: raise if shape is wrong

    Returns:
        dict with per-block stats and alert list
    """
    if features.ndim != 1:
        features = features.flatten()

    D = features.shape[0]
    if raise_on_dim:
        assert D == AGENT_C_INPUT_DIM, (
            f"AgentC validator: got dim {D}, expected {AGENT_C_INPUT_DIM}"
        )

    alerts: List[str] = []
    block_stats: Dict = {}

    for block_name, s, e in _BLOCK_LAYOUT:
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
            alerts.append(f"AgentC [{block_name}]: nan_rate={nan_rate:.1%}")
        if zero_rate > 0.70 and block_name not in ("csv_distatm",):
            alerts.append(
                f"AgentC [{block_name}]: zero_rate={zero_rate:.1%} — check upstream extractor"
            )

    # Gamma flip level (idx=26) should be a positive float near current spot
    gamma_flip_idx = 25  # index in assembled vector: gamma_exposure[25] = gamma_flip_level
    if D > gamma_flip_idx and features[gamma_flip_idx] == 0.0:
        alerts.append(
            "AgentC [gamma_exposure]: gamma_flip_level=0 — "
            "likely no zero-crossing in gamma curve"
        )

    return {
        "agent":         "C",
        "feature_shape": (D,),
        "block_stats":   block_stats,
        "alerts":        alerts,
        "passed":        len(alerts) == 0,
    }


def validate_agent_c_sequence(
    seq: np.ndarray,
    raise_on_dim: bool = True,
) -> Dict:
    """
    Validate Agent C rolling sequence tensor.

    Args:
        seq:          np.ndarray of shape (T, 69)
        raise_on_dim: raise if shape is wrong

    Returns:
        dict with per-timestep and overall stats
    """
    if seq.ndim != 2:
        raise ValueError(f"AgentC sequence validator expects 2D array, got shape {seq.shape}")

    T, D = seq.shape
    if raise_on_dim:
        assert D == AGENT_C_INPUT_DIM, (
            f"AgentC sequence validator: got dim {D}, expected {AGENT_C_INPUT_DIM}"
        )

    alerts: List[str] = []

    all_zero_bars = [int(t) for t in range(T) if (seq[t] == 0).all()]
    if all_zero_bars:
        alerts.append(
            f"AgentC sequence: {len(all_zero_bars)}/{T} bars are all-zero "
            f"(padding bars: {all_zero_bars[:5]}{'...' if len(all_zero_bars) > 5 else ''})"
        )

    nan_bars = [int(t) for t in range(T) if np.isnan(seq[t]).any()]
    if nan_bars:
        alerts.append(f"AgentC sequence: {len(nan_bars)}/{T} bars contain NaN")

    return {
        "agent":         "C",
        "seq_shape":     (T, D),
        "n_pad_bars":    len(all_zero_bars),
        "n_nan_bars":    len(nan_bars),
        "nan_rate_total": float(np.isnan(seq).mean()),
        "zero_rate_total": float((seq == 0).mean()),
        "alerts":        alerts,
        "passed":        len(alerts) == 0,
    }
