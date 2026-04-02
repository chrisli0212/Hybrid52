"""
Agent Q: Quote Dynamics Agent
Analyzes quote updates, spread dynamics, and order book behavior from quote data.

Changes vs v1:
  - temporal_dim default 0 → 128 (was causing fusion shape crash when temporal passed)
  - fusion_dim computed correctly at init using actual temporal_dim
  - fusion upgraded from single Linear → small MLP (Linear+GELU+Dropout+Linear)
  - no-temporal path uses zero-tensor of correct temporal_dim size
~220k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentQ(nn.Module):
    def __init__(
        self,
        quote_feat_dim: int = 35,
        temporal_dim: int = 128,   # fixed: was 0, caused fusion shape crash
        hidden_dim: int = 192,
    ):
        super().__init__()
        self.quote_feat_dim = quote_feat_dim
        self._temporal_dim  = temporal_dim

        # ── Input norm ───────────────────────────────────────────────────
        self.input_norm = nn.LayerNorm(quote_feat_dim)

        # ── Quote pattern encoder with gated residual ─────────────────────
        self.quote_encoder = nn.Sequential(
            nn.Linear(quote_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 96),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.quote_residual = nn.Linear(quote_feat_dim, 96)
        self.encoder_gate   = nn.Sequential(nn.Linear(96, 1), nn.Sigmoid())

        # ── Spread dynamics BiLSTM ──────────────────────────────────────
        self.spread_lstm = nn.LSTM(
            quote_feat_dim, 64, batch_first=True, bidirectional=True
        )
        self.lstm_norm = nn.LayerNorm(128)

        # ── Order book imbalance detector ──────────────────────────────
        self.imbalance_net = nn.Sequential(
            nn.Linear(quote_feat_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.Tanh(),
        )

        # ── Fusion MLP: correct dim uses actual temporal_dim ────────────────
        fusion_in = 96 + temporal_dim + 128 + 24   # 376 when temporal_dim=128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
        )
        self.fusion_norm = nn.LayerNorm(64)

        # ── Output heads ────────────────────────────────────────────────
        self.score_head        = nn.Linear(64, 1)
        self.confidence_head   = nn.Linear(64, 1)
        self.spread_signal_head = nn.Linear(64, 1)

    def _fix_feat_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Clip or zero-pad last dim to quote_feat_dim."""
        d = self.quote_feat_dim
        if x.size(-1) > d:
            return x[..., :d]
        if x.size(-1) < d:
            pad_shape = list(x.shape)
            pad_shape[-1] = d - x.size(-1)
            return torch.cat([x, torch.zeros(*pad_shape, device=x.device)], dim=-1)
        return x

    def forward(
        self,
        static: torch.Tensor,            # (B, quote_feat_dim)
        temporal: Optional[torch.Tensor], # (B, temporal_dim) or None
        seq: torch.Tensor,               # (B, T, quote_feat_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        B = static.size(0)

        # Fix feature dims
        static = self._fix_feat_dim(static)          # (B, 20)
        static = self.input_norm(static)

        # Gated quote encoder
        enc  = self.quote_encoder(static)             # (B, 96)
        res  = self.quote_residual(static)            # (B, 96)
        gate = self.encoder_gate(enc)                 # (B, 1)
        quote_encoded = gate * enc + (1 - gate) * res # (B, 96)

        # Sequence
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        seq = self._fix_feat_dim(seq)                 # (B, T, 20)
        seq = self.input_norm(seq)

        lstm_out, _    = self.spread_lstm(seq)
        spread_features = self.lstm_norm(lstm_out[:, -1, :])  # (B, 128)

        # Imbalance
        imbalance_score = self.imbalance_net(static)  # (B, 24)

        # Temporal: always concat a tensor of temporal_dim size
        if temporal is not None:
            t = temporal
        else:
            t = torch.zeros(B, self._temporal_dim, device=static.device)

        fused_in = torch.cat([quote_encoded, t, spread_features, imbalance_score], dim=-1)
        fused    = self.fusion_norm(F.gelu(self.fusion(fused_in)))  # (B, 64)

        score         = torch.sigmoid(self.score_head(fused))
        confidence    = torch.sigmoid(self.confidence_head(fused))
        spread_signal = torch.tanh(self.spread_signal_head(fused))

        return score, confidence, spread_signal

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
