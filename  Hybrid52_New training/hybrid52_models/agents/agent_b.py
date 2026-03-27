"""
Agent B: Bidirectional LSTM Sequence Agent
Stacked BiLSTM + parallel TCN with attention pooling + static context gating.

Architecture:
  seq   input_dim = 75 (feature_subsets.py B ranges → 75 dims)
  static_dim = 130  (Agent-A static snapshot)
  temporal_dim = 128 (backbone embedding — zero-padded when backbone disabled)

Fix 2026-03-26:
  - fusion_in now conditional: uses actual temporal_dim only when backbone active
  - when use_backbone=False, fusion_in = hidden_dim + 64 (no wasted zero-pad dims)
  - pass use_backbone flag into constructor so fusion is sized correctly at init
~310k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentB(nn.Module):
    def __init__(
        self,
        input_dim:      int = 75,
        static_dim:     int = 130,
        hidden_dim:     int = 128,
        num_layers:     int = 2,
        temporal_dim:   int = 128,
        seq_len:        int = 20,
        n_time_slots:   int = 390,
        time_embed_dim: int = 8,
        use_backbone:   bool = False,   # ← NEW: controls fusion_in size
    ):
        super().__init__()
        self._temporal_dim  = temporal_dim
        self._seq_len       = seq_len
        self._use_backbone  = use_backbone

        lstm_input_dim = input_dim * 2 + time_embed_dim

        self.time_embed = nn.Embedding(n_time_slots, time_embed_dim)
        self.input_norm = nn.LayerNorm(input_dim)

        # BiLSTM stack
        self.lstm1 = nn.LSTM(
            lstm_input_dim, hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.ln1 = nn.LayerNorm(hidden_dim * 2)

        self.lstm2 = nn.LSTM(
            hidden_dim * 2, hidden_dim // 2,
            batch_first=True, bidirectional=True,
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Parallel TCN branch
        self.tcn = nn.Sequential(
            nn.Conv1d(lstm_input_dim, 64, kernel_size=3, padding=2,  dilation=1), nn.GELU(),
            nn.Conv1d(64,            64, kernel_size=3, padding=4,  dilation=2), nn.GELU(),
            nn.Conv1d(64,            64, kernel_size=3, padding=8,  dilation=4), nn.GELU(),
        )
        self.tcn_norm = nn.LayerNorm(64)

        # Attention pool with recency bias
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 48), nn.Tanh(), nn.Linear(48, 1),
        )

        # Static gate
        self.static_gate = nn.Sequential(
            nn.Linear(static_dim, hidden_dim), nn.Sigmoid(),
        )

        # Fusion — only include temporal dims if backbone is active
        fusion_in = hidden_dim + 64 + (temporal_dim if use_backbone else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 96), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(96, 48),        nn.ReLU(),
        )

        self.score_head      = nn.Linear(48, 1)
        self.confidence_head = nn.Linear(48, 1)

    def _add_momentum(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.size(1) > 1:
            delta = seq[:, 1:, :] - seq[:, :-1, :]
            delta = F.pad(delta, (0, 0, 1, 0))
        else:
            delta = torch.zeros_like(seq)
        return torch.cat([seq, delta], dim=-1)

    def _time_of_day_embed(self, seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = seq.shape
        bar_idx = torch.arange(T, device=seq.device).unsqueeze(0).expand(B, -1)
        return self.time_embed(bar_idx)

    def forward(
        self,
        static:  torch.Tensor,           # (B, static_dim)
        temporal: Optional[torch.Tensor], # (B, temporal_dim) or None
        seq:     torch.Tensor,            # (B, T, input_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if seq.dim() == 2:
            seq = seq.unsqueeze(1)

        seq     = self.input_norm(seq)
        seq_m   = self._add_momentum(seq)
        t_emb   = self._time_of_day_embed(seq)
        seq_in  = torch.cat([seq_m, t_emb], dim=-1)

        out1, _ = self.lstm1(seq_in)
        out1    = self.ln1(out1)
        out2, _ = self.lstm2(out1)
        out2    = self.ln2(out2)

        T       = out2.size(1)
        attn_w  = self.attn_pool(out2)
        recency = torch.exp(
            torch.linspace(-1.0, 0.0, T, device=out2.device)
        ).view(1, T, 1)
        attn_w  = torch.softmax(attn_w + recency, dim=1)
        pooled  = (out2 * attn_w).sum(dim=1)

        gate   = self.static_gate(static)
        pooled = gate * pooled

        tcn_out = self.tcn(seq_in.transpose(1, 2))
        tcn_out = tcn_out[:, :, :T].transpose(1, 2)
        tcn_out = self.tcn_norm(tcn_out).mean(dim=1)

        # Fuse — only concat temporal if backbone is active
        if self._use_backbone and temporal is not None:
            fused_in = torch.cat([pooled, tcn_out, temporal], dim=-1)
        else:
            fused_in = torch.cat([pooled, tcn_out], dim=-1)

        fused      = self.fusion(fused_in)
        score      = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))
        return score, confidence, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
