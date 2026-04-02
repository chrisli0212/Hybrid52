"""
Agent 2D Extractor
Builds the (N_GREEKS, N_STRIKES, N_TIMESTEPS) = (5, 30, 20) chain tensor.

Calls shared Chain2DProcessor from extractors/ — does NOT reimplement binning.

Usage:
    from hybrid55_preprocessing.agents.agent_2d.extractor import Agent2DExtractor

    extractor = Agent2DExtractor()
    # snapshots: list of N_TIMESTEPS greek DataFrames, oldest first
    tensor = extractor.extract_sequence(snapshots)
    # tensor.shape == (5, 30, 20)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from ...extractors.active_chain_filter import filter_active_chain
from .feature_config import (
    AGENT_2D_SHAPE, AGENT_2D_N_GREEKS, AGENT_2D_N_STRIKES, AGENT_2D_N_TIMESTEPS,
    AGENT_2D_GREEKS, AGENT_2D_DELTA_RANGE, AGENT_2D_NORM_METHOD, AGENT_2D_MIN_BID,
)

logger = logging.getLogger("hybrid55.agent_2d")


class Agent2DExtractor:
    """
    Dedicated extractor for Agent 2D (CNN).
    Output: np.ndarray shape (5, 30, 20)
    """

    def __init__(self):
        self.alert_log: List[Dict[str, Any]] = []
        self._delta_bins = np.linspace(
            AGENT_2D_DELTA_RANGE[0], AGENT_2D_DELTA_RANGE[1],
            AGENT_2D_N_STRIKES + 1
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract_sequence(
        self,
        snapshots: List[pd.DataFrame],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Build (N_GREEKS, N_STRIKES, N_TIMESTEPS) tensor from snapshot list.

        Args:
            snapshots:  List of Greek DataFrames, oldest-first.
                        Trimmed to last N_TIMESTEPS if longer.
            normalize:  Apply per-channel z-score normalization.

        Returns:
            np.ndarray of shape (5, 30, 20)
        """
        if not snapshots:
            self._alert("[AGENT 2D] Empty snapshot list — returning zero tensor")
            return np.zeros(AGENT_2D_SHAPE, dtype=np.float32)

        snapshots = snapshots[-AGENT_2D_N_TIMESTEPS:]
        n_available = len(snapshots)

        tensor = np.zeros(AGENT_2D_SHAPE, dtype=np.float32)

        for t, snap in enumerate(snapshots):
            filtered = filter_active_chain(snap, min_bid=AGENT_2D_MIN_BID)
            if filtered.empty:
                self._alert(f"[AGENT 2D] Timestep {t}: active chain empty, using zero slice")
                continue
            tensor[:, :, t] = self._snapshot_to_slice(filtered)

        if n_available < AGENT_2D_N_TIMESTEPS:
            self._alert(
                f"[AGENT 2D] Short sequence: {n_available}/{AGENT_2D_N_TIMESTEPS} timesteps"
            )

        if normalize:
            tensor = self._normalize(tensor)

        self._check_zeros(tensor)

        assert tensor.shape == AGENT_2D_SHAPE, (
            f"Agent 2D shape mismatch: expected {AGENT_2D_SHAPE}, got {tensor.shape}"
        )
        return tensor

    def extract_single_slice(self, greek_df: pd.DataFrame) -> np.ndarray:
        """
        Extract a single 2D slice (5, 30) from one snapshot.
        Returns shape (N_GREEKS, N_STRIKES).
        """
        filtered = filter_active_chain(greek_df, min_bid=AGENT_2D_MIN_BID)
        if filtered.empty:
            return np.zeros((AGENT_2D_N_GREEKS, AGENT_2D_N_STRIKES), dtype=np.float32)
        return self._snapshot_to_slice(filtered)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _snapshot_to_slice(self, df: pd.DataFrame) -> np.ndarray:
        """Convert one Greek DataFrame to (N_GREEKS, N_STRIKES) array via delta binning."""
        result = np.zeros((AGENT_2D_N_GREEKS, AGENT_2D_N_STRIKES), dtype=np.float32)

        if "delta" not in df.columns or df.empty:
            return result

        delta_vals = df["delta"].values.astype(float)
        bin_idx = np.digitize(delta_vals, self._delta_bins) - 1
        bin_idx = np.clip(bin_idx, 0, AGENT_2D_N_STRIKES - 1)
        bin_series = pd.Series(bin_idx, index=df.index)

        for g_idx, greek in enumerate(AGENT_2D_GREEKS):
            if greek not in df.columns:
                continue
            gb = df.groupby(bin_series)[greek].mean()
            result[g_idx] = gb.reindex(
                range(AGENT_2D_N_STRIKES), fill_value=0.0
            ).values.astype(np.float32)

        return result

    def _normalize(self, tensor: np.ndarray) -> np.ndarray:
        """Per-channel normalization (z-score by default)."""
        out = tensor.copy()
        for g in range(AGENT_2D_N_GREEKS):
            ch  = out[g]
            mu  = ch.mean()
            std = ch.std()
            if AGENT_2D_NORM_METHOD == "zscore" and std > 1e-8:
                out[g] = (ch - mu) / std
            elif AGENT_2D_NORM_METHOD == "minmax":
                lo, hi = ch.min(), ch.max()
                if hi - lo > 1e-8:
                    out[g] = (ch - lo) / (hi - lo)
        return out

    def _check_zeros(self, tensor: np.ndarray, threshold: float = 0.80) -> None:
        zero_rate = float((tensor == 0).mean())
        if zero_rate >= threshold:
            self._alert(
                f"[ZERO ALERT] Agent 2D tensor: {zero_rate:.1%} zero — "
                f"chain may be empty or delta-binning failed"
            )

    def _alert(self, msg: str) -> None:
        logger.warning(msg)
        self.alert_log.append({"ts": pd.Timestamp.now().isoformat(), "msg": msg})
