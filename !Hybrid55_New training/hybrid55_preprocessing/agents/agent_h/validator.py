"""
Agent H Validator.

Usage:
    from hybrid55_preprocessing.agents.agent_h.validator import validate_agent_h
    report = validate_agent_h(snapshots)  # snapshots: list of greek DataFrames
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any
import pandas as pd

from .extractor import AgentHExtractor
from .feature_config import AGENT_H_DIM, AGENT_H_SEQ_LEN


def validate_agent_h(
    snapshots: List[pd.DataFrame],
    pad_mode: str = "zero",
) -> Dict[str, Any]:
    extractor = AgentHExtractor()
    tensor = extractor.extract_sequence(snapshots, pad_mode)

    per_timestep_zeros = [(tensor[t] == 0).mean() for t in range(AGENT_H_SEQ_LEN)]
    empty_timesteps = [t for t, z in enumerate(per_timestep_zeros) if z == 1.0]

    return {
        "shape_ok":           tensor.shape == (AGENT_H_SEQ_LEN, AGENT_H_DIM),
        "expected_shape":     (AGENT_H_SEQ_LEN, AGENT_H_DIM),
        "actual_shape":       tensor.shape,
        "n_snapshots_in":     len(snapshots),
        "n_padded":           max(0, AGENT_H_SEQ_LEN - len(snapshots)),
        "nan_count":          int(np.isnan(tensor).sum()),
        "overall_zero_rate":  float((tensor == 0).mean()),
        "per_timestep_zeros": per_timestep_zeros,
        "empty_timesteps":    empty_timesteps,
        "alerts":             extractor.alert_log,
        "tensor":             tensor,
    }
