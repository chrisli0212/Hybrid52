"""
Agent A: Neural Baseline Agent
Primary temporal agent with static MLP + causal CNN + backbone fusion.
input_dim=130  (Greeks 0-74 + IV 125-149 + walls 200-213 + CSV-derived 270-285)
~195k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentA(nn.Module):
    def __init__(
        self,
        input_dim: int = 130,     # 130 dims: Greeks(75) + IV(25) + walls(14) + CSV-derived(16)
        temporal_dim: int = 128,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Static feature path
        self.static_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 96)
        )

        # Residual projection
        self.residual_proj = nn.Linear(input_dim, 96)

        # Causal CNN on temporal sequence of 53-d vectors
        self.causal_conv = nn.Sequential(
            nn.Conv1d(input_dim, 48, kernel_size=4, padding=3, groups=1),
            nn.GELU()
        )
        self.gate = nn.Sequential(nn.Linear(96, 1), nn.Sigmoid())

        fusion_dim = 96 + temporal_dim + 96   # static(96) + temporal + gated_cnn(96)
        self.fusion = nn.Linear(fusion_dim, 64)
        self.fusion_norm = nn.LayerNorm(64)

        self._stored_temporal_dim = temporal_dim
        self.score_head = nn.Linear(64, 1)
        self.confidence_head = nn.Linear(64, 1)
        self.signal_head = nn.Linear(64, 5)

    def forward(
        self,
        static: torch.Tensor,
        temporal: torch.Tensor,
        seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        static_out = self.static_path(static) + self.residual_proj(static)

        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        x = self.causal_conv(seq.transpose(1, 2))       # (B, 48, T)
        cnn_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        cnn_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        cnn_out = torch.cat([cnn_max, cnn_avg], dim=-1)  # (B, 96)
        gate_val = self.gate(cnn_out)
        gated_cnn = gate_val * cnn_out

        if temporal is not None:
            fused = torch.cat([static_out, temporal, gated_cnn], dim=-1)
        else:
            # Handle missing temporal (e.g. standalone inference)
            fused = torch.cat([
                static_out,
                torch.zeros(static_out.size(0), self._stored_temporal_dim, device=static_out.device),
                gated_cnn
            ], dim=-1)
        fused = self.fusion_norm(F.gelu(self.fusion(fused)))

        score = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))
        signal = self.signal_head(fused)

        return score, confidence, signal

    def _temporal_dim(self) -> int:
        return self.fusion.in_features - 96 - 96

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
