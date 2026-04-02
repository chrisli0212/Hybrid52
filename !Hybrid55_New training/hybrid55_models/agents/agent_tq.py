"""
Agent TQ: unified Trade+Quote specialist.

Consumes a combined trade/quote feature block and learns robust microstructure,
spread, and flow dynamics with LayerNorm-first stabilization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AgentTQ(nn.Module):
    def __init__(
        self,
        tq_feat_dim: int = 95,
        temporal_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.tq_feat_dim = tq_feat_dim
        self._temporal_dim = temporal_dim

        self.input_norm = nn.LayerNorm(tq_feat_dim)

        self.static_encoder = nn.Sequential(
            nn.Linear(tq_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
        )

        self.seq_conv = nn.Sequential(
            nn.Conv1d(tq_feat_dim, 96, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(8, 96),
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.imbalance_head = nn.Sequential(
            nn.Linear(tq_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.Tanh(),
        )

        fusion_in = 128 + 64 + 32 + temporal_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
        )
        self.fusion_norm = nn.LayerNorm(64)

        self.score_head = nn.Linear(64, 1)
        self.confidence_head = nn.Linear(64, 1)
        self.signal_head = nn.Linear(64, 3)

    def _fix_feat_dim(self, x: torch.Tensor) -> torch.Tensor:
        d = self.tq_feat_dim
        if x.size(-1) > d:
            return x[..., :d]
        if x.size(-1) < d:
            pad_shape = list(x.shape)
            pad_shape[-1] = d - x.size(-1)
            return torch.cat([x, torch.zeros(*pad_shape, device=x.device)], dim=-1)
        return x

    def forward(
        self,
        static: torch.Tensor,
        temporal: Optional[torch.Tensor],
        seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        bsz = static.size(0)
        static = self.input_norm(self._fix_feat_dim(static))
        static_enc = self.static_encoder(static)
        imbalance = self.imbalance_head(static)

        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        seq = self.input_norm(self._fix_feat_dim(seq))
        conv = self.seq_conv(seq.transpose(1, 2))
        t = conv.size(2)
        w = torch.linspace(0.6, 1.4, t, device=conv.device).view(1, 1, t)
        w = w / w.sum()
        seq_enc = (conv * w).sum(dim=2)

        if temporal is None:
            temporal = torch.zeros(bsz, self._temporal_dim, device=static.device)

        fused = torch.cat([static_enc, seq_enc, imbalance, temporal], dim=-1)
        fused = self.fusion_norm(F.gelu(self.fusion(fused)))

        score = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))
        signal = self.signal_head(fused)
        return score, confidence, signal

