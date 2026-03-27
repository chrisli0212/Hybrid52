"""
Agent VIX: regime encoder for Stage 3 gating.

This module is intentionally independent from the 325-dim directional feature stack.
It consumes compact VIX regime features (default: 10 dims).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


REGIME_NAMES = ["CALM", "NORMAL", "ELEVATED", "HIGH", "EXTREME"]
NUM_REGIMES = len(REGIME_NAMES)


class AgentVIX(nn.Module):
    """VIX regime encoder with interface-compatible heads."""

    def __init__(
        self,
        vix_feat_dim: int = 10,
        hidden_dim: int = 128,
        regime_emb_dim: int = 32,
        num_regimes: int = NUM_REGIMES,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.vix_feat_dim = vix_feat_dim
        self.regime_emb_dim = regime_emb_dim
        self.num_regimes = num_regimes

        self.input_norm = nn.LayerNorm(vix_feat_dim)
        self.encoder = nn.Sequential(
            nn.Linear(vix_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.regime_emb_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, regime_emb_dim),
        )
        self.regime_classifier = nn.Linear(hidden_dim, num_regimes)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def _truncate(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) > self.vix_feat_dim:
            return x[..., : self.vix_feat_dim]
        return x

    def forward(
        self,
        static: torch.Tensor,
        temporal: torch.Tensor,
        seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep signature compatible with other agents.
        del temporal, seq
        static = self._truncate(static)
        h = self.encoder(self.input_norm(static))
        score = torch.sigmoid(self.score_head(h))
        confidence = torch.sigmoid(self.confidence_head(h))
        regime_logits = self.regime_classifier(h)
        return score, confidence, regime_logits

    def forward_with_regime_emb(
        self,
        vix_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vix_features = self._truncate(vix_features)
        h = self.encoder(self.input_norm(vix_features))
        score = torch.sigmoid(self.score_head(h))
        confidence = torch.sigmoid(self.confidence_head(h))
        regime_logits = self.regime_classifier(h)
        regime_emb = self.regime_emb_head(h)
        return score, confidence, regime_logits, regime_emb

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def vix_level_to_regime_label(vix_level: float) -> int:
    if vix_level < 15.0:
        return 0
    if vix_level < 20.0:
        return 1
    if vix_level < 25.0:
        return 2
    if vix_level < 35.0:
        return 3
    return 4
