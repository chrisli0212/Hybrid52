"""
Agent B: Bidirectional LSTM Sequence Agent
Stacked BiLSTM with attention pooling + static context gating.
input_dim=53  (same Theta Data features as Agent A, processed as time-series)
~180k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentB(nn.Module):
    def __init__(
        self,
        input_dim: int = 53,
        hidden_dim: int = 128,
        num_layers: int = 2,
        temporal_dim: int = 128,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm1 = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.ln1 = nn.LayerNorm(hidden_dim * 2)

        self.lstm2 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )

        self.static_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        fusion_in = hidden_dim + temporal_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.score_head = nn.Linear(32, 1)
        self.confidence_head = nn.Linear(32, 1)
        self._temporal_dim = temporal_dim

    def forward(
        self,
        static: torch.Tensor,
        temporal,
        seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if seq.dim() == 2:
            seq = seq.unsqueeze(1)

        seq = self.input_norm(seq)

        out1, _ = self.lstm1(seq)
        out1 = self.ln1(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.ln2(out2)

        attn_w = self.attn_pool(out2)
        attn_w = torch.softmax(attn_w, dim=1)
        pooled = (out2 * attn_w).sum(dim=1)

        gate = self.static_gate(static)
        pooled = gate * pooled

        if temporal is not None:
            fused_in = torch.cat([pooled, temporal], dim=-1)
        else:
            fused_in = torch.cat([
                pooled,
                torch.zeros(pooled.size(0), self._temporal_dim, device=pooled.device)
            ], dim=-1)

        fused = self.fusion(fused_in)

        score = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))

        return score, confidence, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
