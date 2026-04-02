"""
Agent 2D Validator.

Usage:
    from hybrid55_preprocessing.agents.agent_2d.validator import validate_agent_2d
    report = validate_agent_2d(snapshots)
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any
import pandas as pd

from .extractor import Agent2DExtractor
from .feature_config import AGENT_2D_SHAPE, AGENT_2D_GREEKS, AGENT_2D_N_TIMESTEPS


def validate_agent_2d(
    snapshots: List[pd.DataFrame],
    normalize: bool = True,
) -> Dict[str, Any]:
    extractor = Agent2DExtractor()
    tensor = extractor.extract_sequence(snapshots, normalize=normalize)

    per_greek_zeros = [
        float((tensor[g] == 0).mean()) for g in range(len(AGENT_2D_GREEKS))
    ]

    return {
        "shape_ok":           tensor.shape == AGENT_2D_SHAPE,
        "expected_shape":     AGENT_2D_SHAPE,
        "actual_shape":       tensor.shape,
        "n_snapshots_in":     len(snapshots),
        "nan_count":          int(np.isnan(tensor).sum()),
        "overall_zero_rate":  float((tensor == 0).mean()),
        "per_greek_zeros":    dict(zip(AGENT_2D_GREEKS, per_greek_zeros)),
        "alerts":             extractor.alert_log,
        "tensor":             tensor,
    }
