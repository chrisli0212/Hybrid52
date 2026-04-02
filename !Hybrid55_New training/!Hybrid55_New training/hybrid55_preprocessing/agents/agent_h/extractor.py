"""
Agent H Extractor
Builds the (SEQ_LEN, 165) sequence tensor for the LSTM agent.

Each timestep uses the same 165-dim subset of Agent B's blocks.
Rolling windows are assembled here — NOT in the master extractor.

Usage:
    from hybrid55_preprocessing.agents.agent_h.extractor import AgentHExtractor

    extractor = AgentHExtractor()
    # snapshots: list of SEQ_LEN greek DataFrames, oldest first
    tensor = extractor.extract_sequence(snapshots)
    # tensor.shape == (20, 165)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from ...extractors.data_validation import filter_dead_columns
from ...extractors.active_chain_filter import filter_active_chain
from .feature_config import AGENT_H_DIM, AGENT_H_SEQ_LEN

logger = logging.getLogger("hybrid55.agent_h")


class AgentHExtractor:
    """
    Dedicated extractor for Agent H (LSTM sequence).
    Output: np.ndarray shape (SEQ_LEN, AGENT_H_DIM) = (20, 165)
    """

    def __init__(self):
        self.alert_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract_sequence(
        self,
        snapshots: List[pd.DataFrame],
        pad_mode: str = "zero",
    ) -> np.ndarray:
        """
        Build a (SEQ_LEN, AGENT_H_DIM) sequence tensor.

        Args:
            snapshots:  List of Greek DataFrames, oldest-first.
                        Length can be < SEQ_LEN (will be padded).
            pad_mode:   'zero'   — pad missing timesteps with zeros
                        'repeat' — repeat first available snapshot

        Returns:
            np.ndarray of shape (SEQ_LEN, AGENT_H_DIM)
        """
        if not snapshots:
            self._alert("[AGENT H] Empty snapshot list — returning zero tensor")
            return np.zeros((AGENT_H_SEQ_LEN, AGENT_H_DIM), dtype=np.float32)

        # Trim to last SEQ_LEN snapshots
        snapshots = snapshots[-AGENT_H_SEQ_LEN:]
        n_available = len(snapshots)

        tensor = np.zeros((AGENT_H_SEQ_LEN, AGENT_H_DIM), dtype=np.float32)

        for t, snap in enumerate(snapshots):
            tensor[t] = self._extract_single(snap)

        if n_available < AGENT_H_SEQ_LEN:
            if pad_mode == "repeat" and n_available > 0:
                for t in range(n_available, AGENT_H_SEQ_LEN):
                    tensor[t] = tensor[n_available - 1]
            self._alert(
                f"[AGENT H] Short sequence: {n_available}/{AGENT_H_SEQ_LEN} timesteps. "
                f"Padded with '{pad_mode}'."
            )

        self._check_zeros(tensor)

        assert tensor.shape == (AGENT_H_SEQ_LEN, AGENT_H_DIM), (
            f"Agent H shape mismatch: expected ({AGENT_H_SEQ_LEN}, {AGENT_H_DIM}), "
            f"got {tensor.shape}"
        )
        return tensor

    def extract_single_step(self, greek_df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for a single timestep.
        Returns np.ndarray of shape (AGENT_H_DIM,).
        """
        return self._extract_single(greek_df)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _extract_single(self, greek_df: pd.DataFrame) -> np.ndarray:
        """
        Extract 165-dim feature vector for one timestep.
        TODO: wire shared extractors (greek_features, gamma_exposure,
              vanna_charm, iv_surface, time_decay) once ported.
        """
        try:
            greek_df = filter_dead_columns(greek_df, mode="greek")
            active_df = filter_active_chain(greek_df)
            if active_df.empty:
                active_df = greek_df

            # TODO: replace stubs with:
            # greek_block = self.greek_extractor.safe_extract(active_df)   # 75
            # gamma_block = self.gamma_extractor.safe_extract(active_df)   # 30
            # vanna_block = self.vanna_extractor.safe_extract(active_df)   # 20
            # iv_block    = self.iv_extractor.safe_extract(active_df)      # 25
            # time_block  = self.time_extractor.safe_extract(active_df)    # 15
            # return np.concatenate([greek_block, gamma_block, vanna_block,
            #                        iv_block, time_block])

            return np.zeros(AGENT_H_DIM, dtype=np.float32)  # stub

        except Exception as e:
            self._alert(f"[AGENT H] Single step failed: {e}")
            return np.zeros(AGENT_H_DIM, dtype=np.float32)

    def _check_zeros(self, tensor: np.ndarray, threshold: float = 0.80) -> None:
        """Alert if >= threshold of entire tensor is zero (higher bar for sequences)."""
        zero_rate = float((tensor == 0).mean())
        if zero_rate >= threshold:
            self._alert(
                f"[ZERO ALERT] Agent H tensor: {zero_rate:.1%} zero — "
                f"sequence may be mostly empty"
            )

    def _alert(self, msg: str) -> None:
        logger.warning(msg)
        self.alert_log.append({"ts": pd.Timestamp.now().isoformat(), "msg": msg})
