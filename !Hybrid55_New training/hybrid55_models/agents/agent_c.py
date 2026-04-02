"""
Agent C: Multi-Scale Attention Agent
CNN + Multi-head Attention + BiLSTM with attention pooling.

Architecture:
  seq   input_dim = 69 (feature_subsets.py C ranges → 69 dims)
  static_dim = 130  (Agent-A static snapshot — unused in C, kept for API compat)
  temporal_dim = 128 (backbone embedding)

Fix 2026-03-26:
  - dense_head output properly split into score(1) + confidence(1)
  - residual_proj dim aligned to BiLSTM output (192)
  - temporal_weights made dynamic via slicing instead of fixed seq_len
~490k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentC(nn.Module):
    def __init__(
        self,
        input_dim:      int = 69,
        static_dim:     int = 130,
        seq_len:        int = 20,
        embed_dim:      int = 96,
        n_heads:        int = 4,
        temporal_dim:   int = 128,
        n_time_slots:   int = 390,
        time_embed_dim: int = 8,
    ):
        super().__init__()
        self._temporal_dim = temporal_dim
        self._seq_len      = seq_len

        # Temporal recency weights (learnable)
        self.temporal_weights = nn.Parameter(torch.linspace(0.4, 1.6, seq_len))

        # Time-of-day embedding
        self.time_embed = nn.Embedding(n_time_slots, time_embed_dim)

        # Input norm
        self.input_norm = nn.LayerNorm(input_dim)

        # CNN input: input_dim*2 (momentum) + time_embed_dim
        cnn_in = input_dim * 2 + time_embed_dim

        # Linear embedding into CNN space
        self.embedding = nn.Linear(cnn_in, embed_dim)

        # Multi-scale CNN
        self.cnn_local  = nn.Conv1d(embed_dim, 24, kernel_size=3, padding=1)
        self.cnn_medium = nn.Conv1d(embed_dim, 24, kernel_size=5, padding=2)
        self.cnn_long   = nn.Conv1d(embed_dim, 24, kernel_size=7, padding=3)
        self.ln_cnn     = nn.LayerNorm(72)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(72, num_heads=n_heads, batch_first=True)

        # BiLSTM + residual
        self.lstm       = nn.LSTM(72, 96, batch_first=True, bidirectional=True)
        self.ln_lstm    = nn.LayerNorm(192)
        self.residual_proj = nn.Linear(72, 192)

        # Attention pool
        self.pool_attention = nn.Sequential(
            nn.Linear(192, 96), nn.Tanh(), nn.Linear(96, 1),
        )

        # Backbone gate — per-dim gating (192) using temporal embedding
        self.backbone_gate = nn.Sequential(
            nn.Linear(temporal_dim, 192), nn.Sigmoid(),
        )

        # Dense head: fuses pooled(192) + temporal(128) = 320
        fusion_in = 192 + temporal_dim
        self.dense_head = nn.Sequential(
            nn.Linear(fusion_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
        )
        self.score_head      = nn.Linear(64, 1)
        self.confidence_head = nn.Linear(64, 1)

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
        static:   torch.Tensor,           # (B, static_dim) — unused in C
        temporal:  Optional[torch.Tensor], # (B, temporal_dim) or None
        seq:      torch.Tensor,            # (B, T, input_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if seq.dim() == 2:
            seq = seq.unsqueeze(1)

        B, T, _ = seq.shape

        # 1. Input norm
        seq = self.input_norm(seq)

        # 2. Temporal recency weighting (dynamic slice for variable T)
        weights = self.temporal_weights[:T].view(1, T, 1)
        seq     = seq * weights

        # 3. Momentum deltas + time-of-day embed
        seq_m   = self._add_momentum(seq)
        t_emb   = self._time_of_day_embed(seq)
        seq_in  = torch.cat([seq_m, t_emb], dim=-1)

        # 4. Linear embed into CNN space
        x_embed = self.embedding(seq_in)

        # 5. Multi-scale CNN
        x_conv   = x_embed.transpose(1, 2)
        local    = self.cnn_local(x_conv)
        medium   = self.cnn_medium(x_conv)
        long_    = self.cnn_long(x_conv)
        combined = torch.cat([local, medium, long_], dim=1).transpose(1, 2)
        combined = F.relu(self.ln_cnn(combined))

        # 6. Multi-head self-attention
        attn_out, _ = self.attention(combined, combined, combined)

        # 7. BiLSTM with residual
        lstm_out, _ = self.lstm(attn_out)
        lstm_out    = self.ln_lstm(lstm_out + self.residual_proj(attn_out))

        # 8. Attention pool
        pool_w = self.pool_attention(lstm_out)
        pool_w = torch.softmax(pool_w, dim=1)
        pooled = (lstm_out * pool_w).sum(dim=1)

        # 9. Backbone gate
        if temporal is not None:
            gate   = self.backbone_gate(temporal)
            pooled = gate * pooled
            fused_in = torch.cat([pooled, temporal], dim=-1)
        else:
            fused_in = torch.cat([
                pooled,
                torch.zeros(B, self._temporal_dim, device=pooled.device),
            ], dim=-1)

        # 10. Dense head → score + confidence
        fused      = self.dense_head(fused_in)
        score      = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))
        return score, confidence, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
