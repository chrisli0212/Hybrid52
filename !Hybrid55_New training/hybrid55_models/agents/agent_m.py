"""
Agent M: Mixer-based temporal specialist.

Designed to diversify the ensemble with a non-convolutional/non-recurrent
decision head that consumes richer temporal embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentM(nn.Module):
    def __init__(
        self,
        input_dim: int = 146,
        temporal_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.temporal_dim = temporal_dim

        self.static_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 96),
        )

        self.seq_pool = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.GELU(),
            nn.Linear(96, 96),
        )

        self.fusion = nn.Sequential(
            nn.Linear(96 + 96 + temporal_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.score_head = nn.Linear(64, 1)
        self.confidence_head = nn.Linear(64, 1)
        self.signal_head = nn.Linear(64, 5)

    def forward(
        self,
        static: torch.Tensor,
        temporal: torch.Tensor,
        seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        static_n = self.input_norm(static)
        static_repr = self.static_path(static_n)

        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        seq_n = self.input_norm(seq)
        seq_repr = self.seq_pool(seq_n).mean(dim=1)

        if temporal is None:
            temporal = torch.zeros(
                static.size(0), self.temporal_dim, device=static.device, dtype=static.dtype
            )

        fused = self.fusion(torch.cat([static_repr, seq_repr, temporal], dim=-1))
        score = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))
        signal = self.signal_head(fused)
        return score, confidence, signal

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

