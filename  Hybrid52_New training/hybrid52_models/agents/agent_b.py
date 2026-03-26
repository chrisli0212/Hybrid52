"""
Agent B: Bidirectional LSTM Sequence Agent
Stacked BiLSTM + parallel TCN with attention pooling + static context gating.

Architecture:
  seq  input_dim = 36   (34 original + 2 CSV-derived dist_atm dims)
  static_dim    = 56    (Agent-A static snapshot + 3 CSV-derived lambda dims)
  temporal_dim  = 128   (backbone embedding from shared encoder)

New features vs v1:
  - seq_dim separated from static_dim (fixes shape crash)
  - Parallel TCN branch (dilated conv captures local bursts)
  - Explicit momentum deltas appended to seq input (velocity signal)
  - Learnable time-of-day embedding (390 RTH 1-min bars)
  - Recency bias on attention pool (recent bars get higher prior weight)
  - num_layers plumbed into lstm1 (fixes dropout warning)
~310k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentB(nn.Module):
    def __init__(
        self,
        input_dim: int = 40,       # seq feature dim: prior 36 + 4 aux (dual_delta/dual_gamma/d1/d2)
        static_dim: int = 64,      # static snapshot dim (Agent A feature set)
        hidden_dim: int = 128,
        num_layers: int = 2,
        temporal_dim: int = 128,
        seq_len: int = 20,         # number of timesteps in sequence window
        n_time_slots: int = 390,   # RTH 1-min bars: 9:30–16:00
        time_embed_dim: int = 8,
    ):
        super().__init__()
        self._temporal_dim = temporal_dim
        self._seq_len = seq_len

        # ── Momentum delta doubles the feature width ──────────────────────
        lstm_input_dim = input_dim * 2 + time_embed_dim  # raw + delta + time_embed

        # ── Time-of-day embedding ─────────────────────────────────────────
        self.time_embed = nn.Embedding(n_time_slots, time_embed_dim)

        # ── Input norm on raw seq (before delta concat) ───────────────────
        self.input_norm = nn.LayerNorm(input_dim)

        # ── BiLSTM stack ──────────────────────────────────────────────────
        self.lstm1 = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.ln1 = nn.LayerNorm(hidden_dim * 2)

        self.lstm2 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

        # ── Parallel TCN branch (dilated causal convs) ────────────────────
        self.tcn = nn.Sequential(
            nn.Conv1d(lstm_input_dim, 64, kernel_size=3, padding=2,  dilation=1),
            nn.GELU(),
            nn.Conv1d(64,             64, kernel_size=3, padding=4,  dilation=2),
            nn.GELU(),
            nn.Conv1d(64,             64, kernel_size=3, padding=8,  dilation=4),
            nn.GELU(),
        )
        self.tcn_norm = nn.LayerNorm(64)

        # ── Attention pool with recency bias ──────────────────────────────
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1),
        )

        # ── Static gate: uses static_dim, NOT input_dim ───────────────────
        self.static_gate = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # ── Fusion: lstm_pool(hidden_dim) + tcn_pool(64) + temporal ───────
        fusion_in = hidden_dim + 64 + temporal_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, 48),
            nn.ReLU(),
        )

        self.score_head      = nn.Linear(48, 1)
        self.confidence_head = nn.Linear(48, 1)

    # ── helpers ───────────────────────────────────────────────────────────
    def _add_momentum(self, seq: torch.Tensor) -> torch.Tensor:
        """Append 1-bar first-difference to each timestep. (B, T, D) → (B, T, 2D)"""
        if seq.size(1) > 1:
            delta = seq[:, 1:, :] - seq[:, :-1, :]          # (B, T-1, D)
            delta = F.pad(delta, (0, 0, 1, 0))               # zero-pad first bar
        else:
            delta = torch.zeros_like(seq)
        return torch.cat([seq, delta], dim=-1)               # (B, T, 2D)

    def _time_of_day_embed(self, seq: torch.Tensor) -> torch.Tensor:
        """Create learnable time-of-day embeddings for each bar position. (B, T, embed_dim)"""
        B, T, _ = seq.shape
        bar_idx = torch.arange(T, device=seq.device).unsqueeze(0).expand(B, -1)  # (B, T)
        return self.time_embed(bar_idx)                       # (B, T, embed_dim)

    # ── forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        static: torch.Tensor,       # (B, static_dim=53)
        temporal,                   # (B, temporal_dim=128) or None
        seq: torch.Tensor,          # (B, T, input_dim=34)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if seq.dim() == 2:
            seq = seq.unsqueeze(1)                           # (B, 1, D)

        # 1. Input normalisation
        seq = self.input_norm(seq)                           # (B, T, 34)

        # 2. Momentum deltas + time-of-day embed
        seq_m   = self._add_momentum(seq)                    # (B, T, 68)
        t_emb   = self._time_of_day_embed(seq)               # (B, T, 8)
        seq_in  = torch.cat([seq_m, t_emb], dim=-1)         # (B, T, 76) = 34*2+8

        # 3. BiLSTM stack
        out1, _ = self.lstm1(seq_in)
        out1     = self.ln1(out1)                            # (B, T, 256)
        out2, _ = self.lstm2(out1)
        out2     = self.ln2(out2)                            # (B, T, 128)

        # 4. Attention pool with recency bias
        T = out2.size(1)
        attn_w = self.attn_pool(out2)                        # (B, T, 1)
        recency = torch.exp(
            torch.linspace(-1.0, 0.0, T, device=out2.device)
        ).view(1, T, 1)
        attn_w = attn_w + recency
        attn_w = torch.softmax(attn_w, dim=1)
        pooled = (out2 * attn_w).sum(dim=1)                  # (B, 128)

        # 5. Static gate (uses static_dim=53, independent of seq input_dim)
        gate   = self.static_gate(static)                    # (B, 128)
        pooled = gate * pooled                               # (B, 128)

        # 6. Parallel TCN branch
        tcn_out = self.tcn(seq_in.transpose(1, 2))          # (B, 64, T')
        # trim to original T (dilated padding may add extra)
        tcn_out = tcn_out[:, :, :T].transpose(1, 2)         # (B, T, 64)
        tcn_out = self.tcn_norm(tcn_out).mean(dim=1)         # (B, 64) global avg

        # 7. Fuse lstm + tcn + backbone temporal
        if temporal is not None:
            fused_in = torch.cat([pooled, tcn_out, temporal], dim=-1)
        else:
            fused_in = torch.cat([
                pooled,
                tcn_out,
                torch.zeros(pooled.size(0), self._temporal_dim, device=pooled.device),
            ], dim=-1)

        fused = self.fusion(fused_in)                        # (B, 48)

        score      = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))

        return score, confidence, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
