"""
Agent T: Trade Flow Agent
Analyzes order flow, trade aggression, and market impact from trade/quote data.

Changes vs v1:
  - temporal_dim default 0 → 128 (was causing fusion shape crash when temporal passed)
  - fusion_dim computed correctly at init using actual temporal_dim
  - fusion upgraded from single Linear → MLP (Linear+GELU+Dropout+Linear)
  - no-temporal path uses zero-tensor of correct temporal_dim size
~260k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentT(nn.Module):
    def __init__(
        self,
        trade_feat_dim: int = 25,
        temporal_dim: int = 128,   # fixed: was 0, caused fusion shape crash
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.trade_feat_dim = trade_feat_dim
        self._temporal_dim  = temporal_dim

        # ── Input norm ───────────────────────────────────────────────────
        self.input_norm = nn.LayerNorm(trade_feat_dim)

        # ── Trade flow encoder ──────────────────────────────────────────
        self.flow_encoder = nn.Sequential(
            nn.Linear(trade_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        # ── 1D CNN on trade sequence with recency-weighted pool ──────────────
        self.flow_cnn = nn.Sequential(
            nn.Conv1d(trade_feat_dim, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(8, 64),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # ── Impact detector ────────────────────────────────────────────
        self.impact_net = nn.Sequential(
            nn.Linear(trade_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.Sigmoid(),
        )

        # ── Fusion MLP: correct dim uses actual temporal_dim ────────────────
        fusion_in = 128 + temporal_dim + 32 + 32   # 320 when temporal_dim=128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
        )
        self.fusion_norm = nn.LayerNorm(64)

        # ── Output heads ────────────────────────────────────────────────
        self.score_head      = nn.Linear(64, 1)
        self.confidence_head = nn.Linear(64, 1)
        self.flow_signal_head = nn.Linear(64, 3)   # Buy/Neutral/Sell logits

    def _fix_feat_dim(self, x: torch.Tensor) -> torch.Tensor:
        d = self.trade_feat_dim
        if x.size(-1) > d:
            return x[..., :d]
        if x.size(-1) < d:
            pad_shape = list(x.shape)
            pad_shape[-1] = d - x.size(-1)
            return torch.cat([x, torch.zeros(*pad_shape, device=x.device)], dim=-1)
        return x

    def forward(
        self,
        static: torch.Tensor,            # (B, trade_feat_dim)
        temporal: Optional[torch.Tensor], # (B, temporal_dim) or None
        seq: torch.Tensor,               # (B, T, trade_feat_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        B = static.size(0)

        static = self._fix_feat_dim(static)
        static = self.input_norm(static)

        flow_encoded = self.flow_encoder(static)          # (B, 128)

        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        seq = self._fix_feat_dim(seq)
        seq = self.input_norm(seq)

        cnn_out = self.flow_cnn(seq.transpose(1, 2))      # (B, 32, T)
        T = cnn_out.size(2)
        w = torch.linspace(0.5, 1.5, T, device=cnn_out.device).view(1, 1, T)
        w = w / w.sum()
        flow_temporal = (cnn_out * w).sum(dim=2)           # (B, 32)

        impact_score = self.impact_net(static)             # (B, 32)

        # Always produce temporal_dim-sized tensor
        if temporal is not None:
            t = temporal
        else:
            t = torch.zeros(B, self._temporal_dim, device=static.device)

        fused_in = torch.cat([flow_encoded, t, flow_temporal, impact_score], dim=-1)
        fused    = self.fusion_norm(F.gelu(self.fusion(fused_in)))  # (B, 64)

        score       = torch.sigmoid(self.score_head(fused))
        confidence  = torch.sigmoid(self.confidence_head(fused))
        flow_signal = self.flow_signal_head(fused)         # (B, 3) raw logits

        return score, confidence, flow_signal

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
