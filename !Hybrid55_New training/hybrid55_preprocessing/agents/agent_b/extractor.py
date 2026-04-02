"""
Agent B Extractor
Builds the 311-dim (historical) or 366-dim (live) feature vector
from 1-minute Greek + trade/quote + OHLC snapshots.

Calls shared extractors from extractors/ — does NOT reimplement math.

Usage:
    from hybrid55_preprocessing.agents.agent_b.extractor import AgentBExtractor

    extractor = AgentBExtractor(historical_mode=True)
    features = extractor.extract(greek_df, trade_df, ohlc_df)
    # features.shape == (311,)  [historical]
    # features.shape == (366,)  [live, include_phase1=True]
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from ...extractors.data_validation import filter_dead_columns
from ...extractors.active_chain_filter import filter_active_chain
from .feature_config import (
    AGENT_B_BASE_DIM, AGENT_B_LIVE_DIM, AGENT_B_REGISTRY, AGENT_B_PHASE1_REGISTRY
)

logger = logging.getLogger("hybrid55.agent_b")


class AgentBExtractor:
    """
    Dedicated extractor for Agent B (1-min intraday).
    Output: np.ndarray shape (311,) historical or (366,) live.
    """

    def __init__(self, historical_mode: bool = True):
        """
        Args:
            historical_mode: True = BigQuery training (311 features, Phase 1 disabled)
                             False = live deployment (366 features, Phase 1 enabled)
        """
        self.historical_mode = historical_mode
        self.expected_dim = AGENT_B_BASE_DIM if historical_mode else AGENT_B_LIVE_DIM
        self.alert_log: List[Dict[str, Any]] = []

        # Lazy-import shared extractors (avoids circular imports at package level)
        self._init_extractors()

    def _init_extractors(self):
        """Initialise all shared extractor instances used by Agent B."""
        # These are imported here to keep agent package self-contained
        # Replace with actual imports once shared extractors are ported to extractors/
        self._extractors_ready = False  # set True once shared extractors are ported

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        greek_df: pd.DataFrame,
        trade_df: Optional[pd.DataFrame] = None,
        ohlc_df: Optional[pd.DataFrame] = None,
        open_interest: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extract Agent B features.

        Args:
            greek_df:        1-min Greek snapshot (required)
            trade_df:        Trade/quote snapshot (required for flow/microstructure)
            ohlc_df:         1-min OHLC snapshot (required for ohlc_dynamics)
            open_interest:   Current open interest (for volume anomaly Phase 1)

        Returns:
            np.ndarray of shape (311,) or (366,)
        """
        try:
            features = self._extract_inner(greek_df, trade_df, ohlc_df, open_interest)
        except Exception as e:
            self._alert(f"[AGENT B FAIL] {e}")
            features = np.zeros(self.expected_dim, dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        self._check_zeros(features)

        assert features.shape == (self.expected_dim,), (
            f"Agent B shape mismatch: expected ({self.expected_dim},), got {features.shape}"
        )
        return features

    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names for all blocks."""
        names = []
        registry = AGENT_B_REGISTRY if self.historical_mode else AGENT_B_REGISTRY + AGENT_B_PHASE1_REGISTRY
        for block in registry:
            for i in range(block.size):
                names.append(f"{block.name}_{i:03d}")
        return names

    def get_block_indices(self, block_name: str) -> range:
        """Return index range for a named feature block — useful for debugging."""
        registry = AGENT_B_REGISTRY if self.historical_mode else AGENT_B_REGISTRY + AGENT_B_PHASE1_REGISTRY
        for block in registry:
            if block.name == block_name:
                return block.indices
        raise KeyError(f"Block '{block_name}' not found in Agent B registry")

    # ------------------------------------------------------------------ #
    #  Internal extraction  (wires shared extractors per registry)        #
    # ------------------------------------------------------------------ #

    def _extract_inner(
        self,
        greek_df: pd.DataFrame,
        trade_df: Optional[pd.DataFrame],
        ohlc_df: Optional[pd.DataFrame],
        open_interest: Optional[float],
    ) -> np.ndarray:
        features = np.zeros(self.expected_dim, dtype=np.float32)
        missing_blocks = []

        # Preprocess
        greek_df = filter_dead_columns(greek_df, mode="greek")
        active_df = filter_active_chain(greek_df)
        if active_df.empty:
            self._alert("[AGENT B] Active chain filter returned empty — using unfiltered greek_df")
            active_df = greek_df

        if trade_df is not None:
            trade_df = filter_dead_columns(trade_df, mode="trade")

        # Wire each block from registry
        # TODO: replace _stub_block() with actual shared extractor calls
        # once shared extractors are ported to extractors/ folder.
        # Pattern:
        #   block_feats, failed = self.greek_extractor.safe_extract(active_df)
        #   features[block.start:block.end] = block_feats
        for block in AGENT_B_REGISTRY:
            src = (
                active_df if block.source == "greek_df"
                else (trade_df if block.source == "trade_df" and trade_df is not None else None)
                or (ohlc_df  if block.source == "ohlc_df"  and ohlc_df  is not None else None)
            )
            block_feats, failed = self._stub_block(src, block.size, block.name)
            if failed:
                missing_blocks.append(block.name)
            features[block.start:block.end] = block_feats

        if missing_blocks:
            self._alert(f"[AGENT B] Failed blocks: {missing_blocks}")

        return features

    @staticmethod
    def _stub_block(df, size: int, name: str):
        """
        Stub: returns zeros until shared extractor is wired.
        Replace this with actual extractor call.
        """
        return np.zeros(size, dtype=np.float32), False

    # ------------------------------------------------------------------ #
    #  Alert helpers                                                        #
    # ------------------------------------------------------------------ #

    def _check_zeros(self, features: np.ndarray, threshold: float = 0.50) -> None:
        zero_rate = float((features == 0).mean())
        if zero_rate >= threshold:
            self._alert(
                f"[ZERO ALERT] Agent B: {zero_rate:.1%} zero "
                f"({int(zero_rate * self.expected_dim)}/{self.expected_dim})"
            )

    def _alert(self, msg: str) -> None:
        logger.warning(msg)
        self.alert_log.append({"ts": pd.Timestamp.now().isoformat(), "msg": msg})
