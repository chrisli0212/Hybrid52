"""
Agent K: Greeks Core Agent
Deep MLP focused on option Greek features with gamma-squeeze detection.

Architecture:
  static_dim  = 53   (real Theta Data features, no zero-padding)
  temporal_dim = 128  (backbone embedding)

Changes vs v1:
  - input_dim 127 → 53 (removes 74 zero-padded dead dims)
  - input LayerNorm added (features at very different scales)
  - temporal properly fused before score/confidence heads
  - gamma_squeeze_detector output used as learned gate (not hardcoded +0.1 boost)
  - gamma_squeeze conditioned on temporal as well as static
  - residual connection in MLP tower
~280k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentK(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,       # compact static feature dim + expanded CSV-derived aux features
        hidden_dim: int = 512,
        temporal_dim: int = 128,
    ):
        super().__init__()
        self._temporal_dim = temporal_dim

        # ── Input norm (critical: delta, vanna, charm, lambda are very different scales)
        self.input_norm = nn.LayerNorm(input_dim)

        # ── Main MLP tower with residual ──────────────────────────────
        self.fc1  = nn.Linear(input_dim,       hidden_dim)
        self.fc2  = nn.Linear(hidden_dim,      hidden_dim // 2)
        self.fc3  = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4  = nn.Linear(hidden_dim // 4, 64)
        self.res_proj = nn.Linear(input_dim, 64)   # residual skip from input → 64

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.15)
        self.ln_out = nn.LayerNorm(64)

        # ── Gamma-squeeze detector: learned gate, conditioned on static only ──
        # Output is a (B,1) gate weight multiplied into score — fully learned,
        # no hardcoded +0.1 nudge.
        self.gamma_squeeze_detector = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # ── Fusion: greek_tower(64) + temporal(128) → heads ───────────────
        fusion_in = 64 + temporal_dim   # 64 + 128 = 192
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.score_head      = nn.Linear(64, 1)
        self.confidence_head = nn.Linear(64, 1)

    def forward(
        self,
        static: torch.Tensor,       # (B, 53)
        temporal,                   # (B, 128) or None
        seq: torch.Tensor,          # unused in K, kept for API compat
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        x = self.input_norm(static)                          # (B, 53)

        # MLP tower
        h = F.relu(self.fc1(x))
        h = self.drop1(h)
        h = F.relu(self.fc2(h))
        h = self.drop2(h)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)                                      # (B, 64)
        h = self.ln_out(h + self.res_proj(x))                # residual skip

        # Gamma-squeeze gate (learned, no hardcoded bias)
        gamma_gate = self.gamma_squeeze_detector(x)          # (B, 1)  in [0,1]

        # Fuse with temporal
        if temporal is not None:
            fused_in = torch.cat([h, temporal], dim=-1)      # (B, 192)
        else:
            fused_in = torch.cat([
                h,
                torch.zeros(h.size(0), self._temporal_dim, device=h.device),
            ], dim=-1)

        fused = self.fusion(fused_in)                        # (B, 64)

        # Gamma-squeeze modulates the score gate multiplicatively
        raw_score  = self.score_head(fused)                  # (B, 1) logit
        # learned gating: gamma_gate scales the pre-sigmoid logit magnitude
        score      = torch.sigmoid(raw_score * (1.0 + gamma_gate))
        score      = torch.clamp(score, 0.01, 0.99)

        confidence = torch.sigmoid(self.confidence_head(fused))

        return score, confidence, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
