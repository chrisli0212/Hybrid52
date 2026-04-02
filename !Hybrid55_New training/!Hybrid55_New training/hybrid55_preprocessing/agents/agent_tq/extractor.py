"""
Agent TQ Extractor
Builds the TQ feature vector (static snapshot + sequence) for AgentTQ.

Agent TQ has THREE input paths:
  1. static  (B, 95) — current-bar flat snapshot
  2. seq     (B, T, 95) — rolling window for Conv1D temporal path
  3. temporal (B, 128) — backbone embedding (provided by backbone, not here)

This extractor produces both `static` and `seq` from master flat features.

Usage:
    extractor = AgentTQExtractor(historical_mode=True)
    static = extractor.extract_static(flat_features)          # (95,) or (70,)
    seq    = extractor.extract_sequence(snapshot_list)        # (T, 95) or (T, 70)
"""

from __future__ import annotations

import logging
import numpy as np
from typing import List, Optional

from .feature_config import (
    AGENT_TQ_INPUT_DIM,
    AGENT_TQ_INPUT_DIM_HIST,
    AGENT_TQ_SOURCE_BLOCKS,
    AGENT_TQ_FEATURE_NAMES_HIST,
    AGENT_TQ_FEATURE_NAMES_LIVE,
)

logger = logging.getLogger(__name__)


class AgentTQExtractor:
    """
    Assembles the TQ feature vector for AgentTQ from the master flat features.

    historical_mode=True:  uses only blocks available in Theta EOD data (70 dims)
    historical_mode=False: full live 95-dim vector (adds Phase 1 smart money + quote pressure)

    AgentTQ._fix_feat_dim() will pad/clip to tq_feat_dim=95 automatically, so
    passing a 70-dim historical vector to the live model is safe — it will be
    zero-padded for the Phase 1 dims.
    """

    def __init__(
        self,
        historical_mode: bool = True,
        seq_len: int = 20,
    ):
        self.historical_mode = historical_mode
        self.seq_len = seq_len
        self.alert_log: List[str] = []

        self._active_blocks = [
            (s, e, name)
            for s, e, name, hist_ok in AGENT_TQ_SOURCE_BLOCKS
            if hist_ok or not historical_mode
        ]
        self._input_dim = (
            AGENT_TQ_INPUT_DIM_HIST if historical_mode else AGENT_TQ_INPUT_DIM
        )
        self._feature_names = (
            AGENT_TQ_FEATURE_NAMES_HIST if historical_mode else AGENT_TQ_FEATURE_NAMES_LIVE
        )

    # ── Static snapshot ───────────────────────────────────────────────────────
    def extract_static(self, flat_features: np.ndarray) -> np.ndarray:
        """
        Extract TQ static snapshot from 311-dim (historical) or 350-dim (live) flat vector.

        Returns: np.ndarray of shape (input_dim,)
        """
        self.alert_log.clear()

        blocks = []
        for s, e, block_name in self._active_blocks:
            if flat_features.shape[0] < e:
                logger.warning(
                    f"AgentTQ: block '{block_name}' [{s}:{e}] out of range "
                    f"(flat_features.shape={flat_features.shape}) — zero-filling"
                )
                self.alert_log.append(f"AgentTQ: {block_name} out of range, zero-filled")
                blocks.append(np.zeros(e - s, dtype=np.float32))
            else:
                blocks.append(flat_features[s:e].astype(np.float32))

        vec = np.concatenate(blocks)
        assert vec.shape == (self._input_dim,), (
            f"AgentTQ static extract: got {vec.shape}, expected ({self._input_dim},)"
        )

        self._check_zeros(vec, context="static")
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
        return vec

    # ── Sequence (rolling window for Conv1D path) ─────────────────────────────
    def extract_sequence(
        self,
        snapshot_list: List[np.ndarray],
        pad_value: float = 0.0,
    ) -> np.ndarray:
        """
        Build (T, input_dim) sequence tensor from a rolling window of flat snapshots.

        Args:
            snapshot_list: list of flat feature arrays (311-dim or 350-dim)
            pad_value:     fill for missing leading timesteps

        Returns:
            np.ndarray of shape (seq_len, input_dim)
        """
        self.alert_log.clear()
        rows: List[np.ndarray] = []

        for snap in snapshot_list[-self.seq_len :]:
            rows.append(self.extract_static(snap))

        n_pad = self.seq_len - len(rows)
        if n_pad > 0:
            pad_row = np.full((self._input_dim,), pad_value, dtype=np.float32)
            rows = [pad_row] * n_pad + rows
            self.alert_log.append(
                f"AgentTQ: padded {n_pad}/{self.seq_len} timesteps"
            )

        seq = np.stack(rows, axis=0)   # (seq_len, input_dim)
        assert seq.shape == (self.seq_len, self._input_dim), (
            f"AgentTQ sequence shape {seq.shape} != "
            f"({self.seq_len}, {self._input_dim})"
        )
        return seq

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _check_zeros(
        self, vec: np.ndarray, context: str = ""
    ) -> None:
        zero_rate = float((vec == 0).mean())
        if zero_rate > 0.40:
            msg = f"AgentTQ [{context}]: {zero_rate:.1%} zeros — check Phase1 availability"
            logger.warning(msg)
            self.alert_log.append(msg)

    def get_feature_names(self) -> List[str]:
        return list(self._feature_names[: self._input_dim])

    @property
    def input_dim(self) -> int:
        return self._input_dim
