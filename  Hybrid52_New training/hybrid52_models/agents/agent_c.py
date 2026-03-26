"""
Agent C: Multi-Scale Attention Agent
CNN + Multi-head Attention + BiLSTM with attention pooling.

Architecture:
  seq  input_dim = 37   (34 original + 3 CSV-derived spread_pct dims)
  static_dim    = 56    (Agent-A static snapshot + 3 CSV-derived lambda dims)
  temporal_dim  = 128   (backbone embedding)

Changes vs v1:
  - input_dim 158 → 34 (remove dead zero-padded features)
  - static_dim separated from seq input_dim
  - temporal properly fused into dense head (was ignored)
  - backbone_gate upgraded: per-dim gating (192) instead of scalar (1)
  - momentum deltas appended to seq input
  - learnable time-of-day embedding (390 RTH bars)
  - temporal_weights made dynamic (adapts to actual seq_len)
~490k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentC(nn.Module):
    def __init__(
        self,
        input_dim: int = 39,       # seq feature dim: prior 37 + 2 aux (iv_error/ultima)
        static_dim: int = 64,      # static snapshot dim (Agent A feature set)
        seq_len: int = 20,
        embed_dim: int = 96,
        n_heads: int = 4,
        temporal_dim: int = 128,
        n_time_slots: int = 390,   # RTH 1-min bars
        time_embed_dim: int = 8,
    ):
        super().__init__()
        self._temporal_dim = temporal_dim
        self._seq_len = seq_len

        # ── Temporal recency weights (learnable) ─────────────────────────
        self.temporal_weights = nn.Parameter(torch.linspace(0.4, 1.6, seq_len))

        # ── Time-of-day embedding ─────────────────────────────────────────
        self.time_embed = nn.Embedding(n_time_slots, time_embed_dim)

        # ── Input norm on raw seq ─────────────────────────────────────────
        self.input_norm = nn.LayerNorm(input_dim)

        # after momentum delta + time embed: input_dim*2 + time_embed_dim
        cnn_in = input_dim * 2 + time_embed_dim     # 34*2+8 = 76

        # ── Linear embedding into CNN space ──────────────────────────────
        self.embedding = nn.Linear(cnn_in, embed_dim)

        # ── Multi-scale CNN ──────────────────────────────────────────────
        self.cnn_local  = nn.Conv1d(embed_dim, 24, kernel_size=3, padding=1)
        self.cnn_medium = nn.Conv1d(embed_dim, 24, kernel_size=5, padding=2)
        self.cnn_long   = nn.Conv1d(embed_dim, 24, kernel_size=7, padding=3)
        self.ln_cnn     = nn.LayerNorm(72)

        # ── Multi-head self-attention ──────────────────────────────────────
        self.attention = nn.MultiheadAttention(72, num_heads=n_heads, batch_first=True)

        # ── BiLSTM + residual projection ──────────────────────────────────
        self.lstm         = nn.LSTM(72, 96, batch_first=True, bidirectional=True)
        self.ln_lstm      = nn.LayerNorm(192)
        self.residual_proj = nn.Linear(72, 192)

        # ── Attention pool ────────────────────────────────────────────────
        self.pool_attention = nn.Sequential(
            nn.Linear(192, 96),
            nn.Tanh(),
            nn.Linear(96, 1),
        )

        # ── Backbone gate: per-dim (192) instead of scalar (1) ──────────────
        # Takes temporal embedding as context for gating
        self.backbone_gate = nn.Sequential(
            nn.Linear(temporal_dim, 192),
            nn.Sigmoid(),
        )

        # ── Dense head: fuses pooled(192) + temporal(128) ──────────────────
        fusion_in = 192 + temporal_dim   # 192 + 128 = 320
        self.dense_head = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    # ── helpers ───────────────────────────────────────────────────────────
    def _add_momentum(self, seq: torch.Tensor) -> torch.Tensor:
        """Append 1-bar first-difference. (B, T, D) → (B, T, 2D)"""
        if seq.size(1) > 1:
            delta = seq[:, 1:, :] - seq[:, :-1, :]
            delta = F.pad(delta, (0, 0, 1, 0))
        else:
            delta = torch.zeros_like(seq)
        return torch.cat([seq, delta], dim=-1)

    def _time_of_day_embed(self, seq: torch.Tensor) -> torch.Tensor:
        """Learnable time-of-day embedding per bar. (B, T, embed_dim)"""
        B, T, _ = seq.shape
        bar_idx = torch.arange(T, device=seq.device).unsqueeze(0).expand(B, -1)
        return self.time_embed(bar_idx)

    # ── forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        static: torch.Tensor,       # (B, static_dim=53)  — unused in C, kept for API compat
        temporal: torch.Tensor,     # (B, temporal_dim=128) or None
        seq: torch.Tensor,          # (B, T, input_dim=34)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if seq.dim() == 2:
            seq = seq.unsqueeze(1)

        B, T, _ = seq.shape

        # 1. Input norm
        seq = self.input_norm(seq)                               # (B, T, 34)

        # 2. Temporal recency weighting
        weights = self.temporal_weights[:T].view(1, T, 1)
        seq = seq * weights

        # 3. Momentum deltas + time-of-day embed
        seq_m  = self._add_momentum(seq)                         # (B, T, 68)
        t_emb  = self._time_of_day_embed(seq)                    # (B, T, 8)
        seq_in = torch.cat([seq_m, t_emb], dim=-1)              # (B, T, 76)

        # 4. Linear embed into CNN space
        x_embed = self.embedding(seq_in)                         # (B, T, 96)

        # 5. Multi-scale CNN
        x_conv  = x_embed.transpose(1, 2)                       # (B, 96, T)
        local   = self.cnn_local(x_conv)
        medium  = self.cnn_medium(x_conv)
        long_   = self.cnn_long(x_conv)
        combined = torch.cat([local, medium, long_], dim=1)      # (B, 72, T)
        combined = combined.transpose(1, 2)                      # (B, T, 72)
        combined = F.relu(self.ln_cnn(combined))

        # 6. Multi-head self-attention
        attn_out, _ = self.attention(combined, combined, combined)  # (B, T, 72)

        # 7. Residual + BiLSTM
        residual  = self.residual_proj(combined)                 # (B, T, 192)
        lstm_out, _ = self.lstm(attn_out)                        # (B, T, 192)
        lstm_out  = self.ln_lstm(lstm_out + residual)            # (B, T, 192)

        # 8. Attention pool
        attn_w = self.pool_attention(lstm_out)                   # (B, T, 1)
        attn_w = torch.softmax(attn_w, dim=1)
        pooled = (lstm_out * attn_w).sum(dim=1)                  # (B, 192)

        # 9. Per-dim backbone gate conditioned on temporal
        if temporal is not None and temporal.dim() == 2 and temporal.size(1) > 0:
            gate   = self.backbone_gate(temporal)                # (B, 192)
            pooled = gate * pooled
            fused_in = torch.cat([pooled, temporal], dim=-1)     # (B, 320)
        else:
            fused_in = torch.cat([
                pooled,
                torch.zeros(B, self._temporal_dim, device=pooled.device),
            ], dim=-1)                                           # (B, 320)

        # 10. Dense head
        out = self.dense_head(fused_in)                          # (B, 2)

        score      = torch.sigmoid(out[:, 0:1])
        confidence = torch.sigmoid(out[:, 1:2])

        return score, confidence, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
