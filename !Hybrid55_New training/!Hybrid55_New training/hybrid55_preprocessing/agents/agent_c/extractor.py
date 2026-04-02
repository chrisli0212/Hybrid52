"""
Agent C Extractor
Assembles the 69-dim sequence feature block for AgentC.

Agent C is a sequence model (CNN + MHA + BiLSTM). This extractor:
  1. Slices gamma_exposure [75:105], vanna_charm [105:125],
     iv_surface_core [125:141], and csv_distatm [273:276] from the master vector.
  2. Builds both a single-bar static vector AND a rolling sequence (B, T, 69).

Usage:
    extractor = AgentCExtractor()
    static = extractor.extract_static(flat_features)      # (69,)
    seq    = extractor.extract_sequence(snapshot_list)    # (T, 69)
"""

from __future__ import annotations

import logging
import numpy as np
from typing import List

from .feature_config import (
    AGENT_C_INPUT_DIM,
    AGENT_C_SOURCE_BLOCKS,
    AGENT_C_IV_SLICE,
    AGENT_C_CSV_SLICE,
    AGENT_C_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

# IV surface block: how many dims to keep from master[125:150]
_IV_KEEP = 16   # first 16 of the 25-dim IV block


class AgentCExtractor:
    """
    Extracts the 69-dim feature block for AgentC from master flat features.

    Block assembly:
        [0:30]  gamma_exposure    master[75:105]
        [30:50] vanna_charm       master[105:125]
        [50:66] iv_surface_core  master[125:141]
        [66:69] csv_distatm      master[273:276]

    seq_len: how many historical bars to stack for AgentC's temporal window
    """

    def __init__(self, seq_len: int = 20):
        self.seq_len = seq_len
        self.alert_log: List[str] = []

    # ── Single-bar static extraction ────────────────────────────────────────
    def extract_static(self, flat_features: np.ndarray) -> np.ndarray:
        """
        Extract Agent C feature vector from master flat array (single bar).

        Args:
            flat_features: np.ndarray of shape (311,) or (350,)

        Returns:
            np.ndarray of shape (69,), dtype float32
        """
        self.alert_log.clear()
        n = flat_features.shape[0]

        # Block 1: gamma_exposure [75:105]
        gamma_block = self._safe_slice(flat_features, 75, 105, "gamma_exposure", n)

        # Block 2: vanna_charm [105:125]
        vanna_block = self._safe_slice(flat_features, 105, 125, "vanna_charm", n)

        # Block 3: iv_surface core [125:141] — 16 of 25 dims
        iv_block = self._safe_slice(flat_features, 125, 141, "iv_surface_core", n)

        # Block 4: csv_distatm [273:276] — 3 dims
        csv_block = self._safe_slice(flat_features, 273, 276, "csv_distatm", n)

        vec = np.concatenate([gamma_block, vanna_block, iv_block, csv_block])

        assert vec.shape == (AGENT_C_INPUT_DIM,), (
            f"AgentC: extraction shape {vec.shape} != ({AGENT_C_INPUT_DIM},)"
        )
        self._check_zeros(vec)
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
        return vec

    # ── Rolling sequence (T × 69) ─────────────────────────────────────────────
    def extract_sequence(
        self,
        snapshot_list: List[np.ndarray],
        pad_value: float = 0.0,
    ) -> np.ndarray:
        """
        Build (T, 69) sequence from a rolling window of master flat snapshots.

        Args:
            snapshot_list: list of flat feature arrays, oldest-first
            pad_value:     constant for missing leading bars

        Returns:
            np.ndarray of shape (seq_len, 69)
        """
        self.alert_log.clear()
        rows: List[np.ndarray] = []
        for snap in snapshot_list[-self.seq_len:]:
            rows.append(self.extract_static(snap))

        n_pad = self.seq_len - len(rows)
        if n_pad > 0:
            pad_row = np.full((AGENT_C_INPUT_DIM,), pad_value, dtype=np.float32)
            rows = [pad_row] * n_pad + rows
            self.alert_log.append(
                f"AgentC: padded {n_pad}/{self.seq_len} leading timesteps"
            )

        seq = np.stack(rows, axis=0)   # (seq_len, 69)
        assert seq.shape == (self.seq_len, AGENT_C_INPUT_DIM), (
            f"AgentC sequence shape {seq.shape} != ({self.seq_len}, {AGENT_C_INPUT_DIM})"
        )
        return seq

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def _safe_slice(
        self,
        arr: np.ndarray,
        s: int,
        e: int,
        name: str,
        n: int,
    ) -> np.ndarray:
        if n < e:
            msg = (
                f"AgentC: block '{name}' [{s}:{e}] out of range "
                f"(flat len={n}) — zero-filling"
            )
            logger.warning(msg)
            self.alert_log.append(msg)
            return np.zeros(e - s, dtype=np.float32)
        return arr[s:e].astype(np.float32)

    def _check_zeros(self, vec: np.ndarray) -> None:
        # Gamma block (first 30): total_gamma at index 20 should not be 0
        if vec[20] == 0.0:
            msg = "AgentC: total_gamma (idx=20) is 0 — empty chain or pre-market"
            logger.warning(msg)
            self.alert_log.append(msg)
        # IV core block (indices 50:66): ATM IV at iv_moneyness_0.00 is index 53
        if vec[53] == 0.0:
            msg = "AgentC: iv_moneyness_0.00 (idx=53) is 0 — check IV surface extractor"
            logger.warning(msg)
            self.alert_log.append(msg)

    def get_feature_names(self) -> List[str]:
        return list(AGENT_C_FEATURE_NAMES)

    @property
    def input_dim(self) -> int:
        return AGENT_C_INPUT_DIM
