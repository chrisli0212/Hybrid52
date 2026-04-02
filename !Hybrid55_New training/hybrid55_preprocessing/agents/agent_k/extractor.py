"""
Agent K Extractor
Extracts GREEK_BY_STRIKE block (master[0:75]) for Agent K.

Agent K uses only the static GREEK_BY_STRIKE block — no sequence input.
The seq argument in AgentK.forward() is accepted for API compat but unused.

Usage:
    extractor = AgentKExtractor()
    features = extractor.extract(flat_features)   # np.ndarray (75,)
"""

from __future__ import annotations

import logging
import numpy as np
from typing import List

from .feature_config import (
    AGENT_K_INPUT_DIM,
    AGENT_K_SLICE_START,
    AGENT_K_SLICE_END,
    AGENT_K_SUB_BLOCKS,
    AGENT_K_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

# Critical greeks: these should virtually never be zero in a liquid options chain
_CRITICAL_FEATURES = {
    "atm_delta",    # ATM delta always ~0.5
    "atm_gamma",    # ATM gamma is peak of gamma curve
    "atm_implied_vol",  # bucket-aggregated IV
}


class AgentKExtractor:
    """
    Extracts the GREEK_BY_STRIKE feature block from master flat features.

    This is a single contiguous slice: master[0:75].
    All computation happens upstream in GreekFeatureExtractor —
    this class only validates, slices, and quality-checks.
    """

    def __init__(self):
        self.alert_log: List[str] = []

    def extract(self, flat_features: np.ndarray) -> np.ndarray:
        """
        Extract Agent K feature vector from master flat array.

        Args:
            flat_features: np.ndarray of shape (311,) or (350,)

        Returns:
            np.ndarray of shape (75,), dtype float32
        """
        self.alert_log.clear()

        if flat_features.shape[0] < AGENT_K_SLICE_END:
            raise ValueError(
                f"AgentK: flat_features too short — "
                f"got {flat_features.shape[0]}, need >= {AGENT_K_SLICE_END}"
            )

        features = flat_features[
            AGENT_K_SLICE_START:AGENT_K_SLICE_END
        ].astype(np.float32)

        assert features.shape == (AGENT_K_INPUT_DIM,), (
            f"AgentK: extraction shape {features.shape} != ({AGENT_K_INPUT_DIM},)"
        )

        self._check_critical(features)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        return features

    def _check_critical(
        self, features: np.ndarray
    ) -> None:
        """Warn if critical greeks (ATM delta, gamma, IV) are zero."""
        for name in _CRITICAL_FEATURES:
            if name in AGENT_K_FEATURE_NAMES:
                idx = AGENT_K_FEATURE_NAMES.index(name)
                if idx < len(features) and features[idx] == 0.0:
                    msg = (
                        f"AgentK: '{name}' (idx={idx}) is 0.0 — "
                        "possible empty chain or pre-market snapshot"
                    )
                    logger.warning(msg)
                    self.alert_log.append(msg)

    def get_feature_names(self) -> List[str]:
        return list(AGENT_K_FEATURE_NAMES)

    @property
    def input_dim(self) -> int:
        return AGENT_K_INPUT_DIM
