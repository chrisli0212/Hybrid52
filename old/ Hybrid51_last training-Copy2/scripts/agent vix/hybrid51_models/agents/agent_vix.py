"""
Agent VIX: VIX Regime Gating Agent
Classifies the volatility regime and produces a regime embedding
that gates (modulates) the 7 directional agents in the meta model.

NOT a directional voter — outputs regime state, not UP/DOWN.

Architecture: Static MLP (like Agent K — no temporal backbone needed
because features are pre-computed at multi-timescale).

~50K parameters.

Inputs:
    - ~10 VIX features at 5-min resolution (level, momentum, term structure, vol-of-vol)

Outputs:
    - score: (B, 1) — regime severity score [0=calm, 1=extreme] (for interface compat)
    - confidence: (B, 1) — regime classification confidence
    - signal: (B, 5) — 5-class regime logits [CALM, NORMAL, ELEVATED, HIGH, EXTREME]
    - regime_emb: (B, regime_emb_dim) — dense embedding for gate networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# Regime thresholds (VIX levels)
REGIME_THRESHOLDS = {
    'CALM':     (0.0,  15.0),
    'NORMAL':   (15.0, 20.0),
    'ELEVATED': (20.0, 25.0),
    'HIGH':     (25.0, 35.0),
    'EXTREME':  (35.0, 999.0),
}

REGIME_NAMES = ['CALM', 'NORMAL', 'ELEVATED', 'HIGH', 'EXTREME']
NUM_REGIMES = len(REGIME_NAMES)


class AgentVIX(nn.Module):
    """
    VIX Regime Agent — classifies volatility environment and produces
    regime embedding for per-agent gating in the meta model.

    Args:
        vix_feat_dim: Number of VIX input features (default: 10)
        hidden_dim: Hidden layer width (default: 128)
        regime_emb_dim: Regime embedding dimension for gate networks (default: 32)
        num_regimes: Number of regime classes (default: 5)
        dropout: Dropout rate (default: 0.15)
    """

    def __init__(
        self,
        vix_feat_dim: int = 10,
        hidden_dim: int = 128,
        regime_emb_dim: int = 32,
        num_regimes: int = NUM_REGIMES,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.vix_feat_dim = vix_feat_dim
        self.hidden_dim = hidden_dim
        self.regime_emb_dim = regime_emb_dim
        self.num_regimes = num_regimes

        # Input normalization
        self.input_norm = nn.LayerNorm(vix_feat_dim)

        # ── Regime encoder (main trunk) ──────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(vix_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Regime embedding head ────────────────────────────────────────
        # Dense embedding used by gate networks in the meta model
        self.regime_emb_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, regime_emb_dim),
        )

        # ── Regime classification head ───────────────────────────────────
        # 5-class: CALM / NORMAL / ELEVATED / HIGH / EXTREME
        self.regime_classifier = nn.Linear(hidden_dim, num_regimes)

        # ── Score head (interface compatibility) ─────────────────────────
        # Outputs a regime severity scalar [0, 1] (0=calm, 1=extreme)
        self.score_head = nn.Linear(hidden_dim, 1)

        # ── Confidence head ──────────────────────────────────────────────
        self.confidence_head = nn.Linear(hidden_dim, 1)

        # ── Rule-based prior (hybrid rule + learned, like Agent D) ───────
        # Learnable adjustments to the rule-based VIX regime thresholds
        self.threshold_adjust = nn.Parameter(torch.zeros(num_regimes))

    def _rule_based_regime(self, vix_level: torch.Tensor) -> torch.Tensor:
        """
        Compute rule-based regime probabilities from VIX level.
        Uses soft boundaries (sigmoid transitions) instead of hard thresholds
        to allow gradient flow.

        Args:
            vix_level: (B,) or (B, 1) — raw VIX spot level

        Returns:
            regime_probs: (B, num_regimes) — soft regime probabilities
        """
        if vix_level.dim() == 2:
            vix_level = vix_level.squeeze(-1)

        # Adjusted thresholds (learnable refinement)
        t = self.threshold_adjust
        boundaries = torch.tensor(
            [15.0, 20.0, 25.0, 35.0], device=vix_level.device
        ) + t[:4]  # Only 4 boundaries for 5 regimes

        # Soft regime assignment using sigmoid transitions (width ~1.5 VIX points)
        scale = 1.5
        cum_probs = torch.sigmoid((vix_level.unsqueeze(-1) - boundaries.unsqueeze(0)) / scale)

        # Convert cumulative to per-regime probabilities
        # P(CALM) = 1 - cum[0], P(NORMAL) = cum[0] - cum[1], ..., P(EXTREME) = cum[3]
        regime_probs = torch.zeros(vix_level.size(0), self.num_regimes, device=vix_level.device)
        regime_probs[:, 0] = 1.0 - cum_probs[:, 0]
        for i in range(1, self.num_regimes - 1):
            regime_probs[:, i] = cum_probs[:, i - 1] - cum_probs[:, i]
        regime_probs[:, -1] = cum_probs[:, -1]

        return regime_probs

    def forward(
        self,
        static: torch.Tensor,
        temporal: torch.Tensor,
        seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass (matches IndependentAgent interface).

        Args:
            static: (B, vix_feat_dim) — Last timestep VIX features
            temporal: (B, temporal_dim) or (B, vix_feat_dim) — Not used (no backbone),
                      but accepted for interface compatibility
            seq: (B, seq_len, vix_feat_dim) — VIX feature sequence (not used in MLP mode)

        Returns:
            score: (B, 1) — Regime severity [0=calm, 1=extreme]
            confidence: (B, 1) — Regime confidence
            signal: (B, 5) — Regime classification logits
        """
        # Handle dimension mismatch (if full 325-d vector is passed, slice VIX features)
        if static.size(-1) > self.vix_feat_dim:
            static = static[:, :self.vix_feat_dim]

        # Input normalization
        x = self.input_norm(static)

        # Encode
        h = self.encoder(x)

        # Outputs
        score = torch.sigmoid(self.score_head(h))
        confidence = torch.sigmoid(self.confidence_head(h))
        regime_logits = self.regime_classifier(h)

        return score, confidence, regime_logits

    def forward_with_regime_emb(
        self,
        vix_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extended forward pass that also returns the regime embedding.
        Called by RegimeGatedMetaModel (not by IndependentAgent).

        Args:
            vix_features: (B, vix_feat_dim) — VIX features at 5-min resolution

        Returns:
            score: (B, 1) — Regime severity
            confidence: (B, 1) — Regime confidence
            regime_logits: (B, num_regimes) — Regime classification logits
            regime_emb: (B, regime_emb_dim) — Dense embedding for gate networks
            rule_probs: (B, num_regimes) — Rule-based regime probabilities
        """
        if vix_features.size(-1) > self.vix_feat_dim:
            vix_features = vix_features[:, :self.vix_feat_dim]

        x = self.input_norm(vix_features)
        h = self.encoder(x)

        # Score and confidence
        score = torch.sigmoid(self.score_head(h))
        confidence = torch.sigmoid(self.confidence_head(h))

        # Regime classification
        regime_logits = self.regime_classifier(h)

        # Regime embedding (for gate networks)
        regime_emb = self.regime_emb_head(h)

        # Rule-based prior (uses vix_level = first feature)
        rule_probs = self._rule_based_regime(vix_features[:, 0])

        return score, confidence, regime_logits, regime_emb, rule_probs

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_regime_name(self, regime_idx: int) -> str:
        """Convert regime index to name."""
        return REGIME_NAMES[regime_idx]


def vix_level_to_regime_label(vix_level: float) -> int:
    """
    Convert VIX spot level to regime class index (for labeling).
    Used during data pipeline (build_tier3_vix.py) to create training labels.

    Returns:
        0=CALM, 1=NORMAL, 2=ELEVATED, 3=HIGH, 4=EXTREME
    """
    if vix_level < 15.0:
        return 0
    elif vix_level < 20.0:
        return 1
    elif vix_level < 25.0:
        return 2
    elif vix_level < 35.0:
        return 3
    else:
        return 4


if __name__ == '__main__':
    print("Testing AgentVIX...")
    print("=" * 60)

    model = AgentVIX(vix_feat_dim=10)
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Regime emb head: {sum(p.numel() for p in model.regime_emb_head.parameters()):,}")
    print(f"  Regime classifier: {sum(p.numel() for p in model.regime_classifier.parameters()):,}")
    print(f"  Score head: {sum(p.numel() for p in model.score_head.parameters()):,}")
    print(f"  Confidence head: {sum(p.numel() for p in model.confidence_head.parameters()):,}")

    # Test standard forward (IndependentAgent interface)
    batch_size = 16
    seq_len = 4  # 4 × 5-min = 20-min lookback
    vix_feat_dim = 10

    static = torch.randn(batch_size, vix_feat_dim)
    temporal = torch.randn(batch_size, vix_feat_dim)  # placeholder
    seq = torch.randn(batch_size, seq_len, vix_feat_dim)

    with torch.no_grad():
        score, confidence, regime_logits = model(static, temporal, seq)
        print(f"\nStandard forward:")
        print(f"  score: {score.shape}, confidence: {confidence.shape}, regime_logits: {regime_logits.shape}")

    # Test extended forward (meta model interface)
    with torch.no_grad():
        score, confidence, regime_logits, regime_emb, rule_probs = model.forward_with_regime_emb(static)
        print(f"\nExtended forward (with regime_emb):")
        print(f"  score: {score.shape}")
        print(f"  confidence: {confidence.shape}")
        print(f"  regime_logits: {regime_logits.shape}")
        print(f"  regime_emb: {regime_emb.shape}")
        print(f"  rule_probs: {rule_probs.shape}")
        print(f"  rule_probs sample: {rule_probs[0].numpy()}")

    # Test regime labeling
    print(f"\nRegime labeling:")
    for vix in [12.0, 17.0, 22.0, 30.0, 40.0]:
        label = vix_level_to_regime_label(vix)
        print(f"  VIX={vix:.0f} → {REGIME_NAMES[label]} (label={label})")

    print(f"\n{'='*60}")
    print("✓ AgentVIX tests passed!")
