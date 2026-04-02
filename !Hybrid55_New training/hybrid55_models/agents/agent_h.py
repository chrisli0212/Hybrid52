"""
Agent H: OHLC specialist.

Learns from the OHLC dynamics block appended to the flat feature vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AgentH(nn.Module):
    def __init__(
        self,
        input_dim: int = 25,
        temporal_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self._temporal_dim = temporal_dim

        self.input_norm = nn.LayerNorm(input_dim)

        self.static_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
        )

        self.seq_gru = nn.GRU(input_dim, 64, batch_first=True, bidirectional=True)
        self.seq_norm = nn.LayerNorm(128)
        self.attn = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

        self.fusion = nn.Sequential(
            nn.Linear(64 + 128 + temporal_dim, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 64),
        )
        self.fusion_norm = nn.LayerNorm(64)

        self.score_head = nn.Linear(64, 1)
        self.conf_head = nn.Linear(64, 1)

    def _fix_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) > self.input_dim:
            return x[..., : self.input_dim]
        if x.size(-1) < self.input_dim:
            pad = list(x.shape)
            pad[-1] = self.input_dim - x.size(-1)
            return torch.cat([x, torch.zeros(*pad, device=x.device)], dim=-1)
        return x

    def forward(
        self,
        static: torch.Tensor,
        temporal: Optional[torch.Tensor],
        seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        bsz = static.size(0)
        static = self.input_norm(self._fix_dim(static))
        static_enc = self.static_path(static)

        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        seq = self.input_norm(self._fix_dim(seq))
        seq_out, _ = self.seq_gru(seq)
        seq_out = self.seq_norm(seq_out)
        w = torch.softmax(self.attn(seq_out), dim=1)
        seq_enc = (seq_out * w).sum(dim=1)

        if temporal is None:
            temporal = torch.zeros(bsz, self._temporal_dim, device=static.device)
        fused = torch.cat([static_enc, seq_enc, temporal], dim=-1)
        fused = self.fusion_norm(F.gelu(self.fusion(fused)))

        score = torch.sigmoid(self.score_head(fused))
        conf = torch.sigmoid(self.conf_head(fused))
        return score, conf, None

