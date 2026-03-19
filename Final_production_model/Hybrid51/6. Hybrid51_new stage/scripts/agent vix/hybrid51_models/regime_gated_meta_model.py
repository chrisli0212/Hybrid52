"""
Stage 3: Regime-Gated Meta Model

Combines 7 frozen directional agents (A, B, K, C, T, Q, 2D) with a trainable
VIX Regime Agent (V). Per-agent gate networks modulate each directional agent's
output based on the current volatility regime.

Architecture:
    1. Frozen Stage 1 agent checkpoints → 7 directional logit vectors
    2. Agent VIX → regime embedding (32-d)
    3. Per-agent gate networks: gate_i(regime_emb) → sigmoid → scalar weight
    4. Gated fusion: gated_i = softmax(logit_i) × gate_i
    5. Final MLP: concat(gated_0..6, regime_emb) → directional prediction

Key design:
    - Directional agents are FROZEN (no gradient, weights locked)
    - Agent VIX encoder is TRAINABLE (fine-tuned from Stage 1 warm-start)
    - Gate networks are TRAINABLE (learn regime → agent trust mapping)
    - Fusion MLP is TRAINABLE
    - Gradients flow: loss → fusion → gates → VIX agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Local imports (adjust paths as needed for your project structure)
# from .independent_agent import IndependentAgent
# from .agents.agent_vix import AgentVIX


class AgentGateNetwork(nn.Module):
    """
    Per-agent gate: takes regime embedding → outputs scalar gate weight.

    gate_i = sigmoid( Linear(regime_emb) )

    This is a tiny network (~200 params per agent) that learns when to
    trust a specific directional agent based on the volatility regime.
    """

    def __init__(self, regime_emb_dim: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(regime_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, regime_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            regime_emb: (B, regime_emb_dim)
        Returns:
            gate_weight: (B, 1) — in [0, 1]
        """
        return self.gate(regime_emb)


class RegimeGatedMetaModel(nn.Module):
    """
    Stage 3 meta model that fuses 7 frozen directional agents with
    VIX-regime-conditioned gating.

    Args:
        n_agents: Number of directional agents (default: 7)
        agent_output_dim: Output dimension per agent (default: 2 for binary)
        regime_emb_dim: Regime embedding dimension from Agent VIX (default: 32)
        vix_feat_dim: VIX input feature dimension (default: 10)
        gate_hidden_dim: Hidden dimension in gate networks (default: 16)
        fusion_hidden_dim: Hidden dimension in final fusion MLP (default: 128)
        num_classes: Final output classes (default: 2 for binary UP/DOWN)
        dropout: Dropout rate in fusion MLP (default: 0.2)
    """

    AGENT_TYPES = ['A', 'B', 'K', 'C', 'T', 'Q', '2D']

    def __init__(
        self,
        n_agents: int = 7,
        agent_output_dim: int = 2,
        regime_emb_dim: int = 32,
        vix_feat_dim: int = 10,
        gate_hidden_dim: int = 16,
        fusion_hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.agent_output_dim = agent_output_dim
        self.regime_emb_dim = regime_emb_dim
        self.num_classes = num_classes

        # ── Frozen directional agents (loaded separately) ────────────────
        # These are stored as a ModuleList but with requires_grad=False
        # They must be loaded via load_frozen_agents() after __init__
        self.frozen_agents = nn.ModuleList()

        # ── Agent VIX (trainable) ────────────────────────────────────────
        # Import here to avoid circular imports; can also be passed in
        from .agents.agent_vix import AgentVIX
        self.vix_agent = AgentVIX(
            vix_feat_dim=vix_feat_dim,
            regime_emb_dim=regime_emb_dim,
        )

        # ── Per-agent gate networks (trainable) ──────────────────────────
        self.gates = nn.ModuleDict({
            agent_type: AgentGateNetwork(
                regime_emb_dim=regime_emb_dim,
                hidden_dim=gate_hidden_dim,
            )
            for agent_type in self.AGENT_TYPES
        })

        # ── Threshold adjustment (trainable) ─────────────────────────────
        # Dynamic threshold: effective_threshold = base + alpha * risk_score
        self.threshold_alpha = nn.Parameter(torch.tensor(0.0))
        self.base_threshold = nn.Parameter(torch.tensor(0.5))

        # ── Fusion MLP (trainable) ───────────────────────────────────────
        # Input: 7 × agent_output_dim (gated probs) + regime_emb
        fusion_input_dim = n_agents * agent_output_dim + regime_emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes),
        )

        # ── Regime classification auxiliary head ─────────────────────────
        # Auxiliary loss on regime classification (multi-task)
        self.regime_aux_head = nn.Linear(regime_emb_dim, 5)  # 5 regimes

    def load_frozen_agents(self, checkpoint_dir: str, symbol: str, horizon: int):
        """
        Load pre-trained Stage 1 agent checkpoints and freeze them.

        Expects files: {symbol}_agent_{type}_classifier_h{horizon}.pt

        Args:
            checkpoint_dir: Path to directory with .pt files
            symbol: e.g. 'SPXW'
            horizon: e.g. 15
        """
        from .independent_agent import IndependentAgent

        checkpoint_dir = Path(checkpoint_dir)
        self.frozen_agents = nn.ModuleList()

        for agent_type in self.AGENT_TYPES:
            ckpt_path = checkpoint_dir / f"{symbol}_agent_{agent_type}_classifier_h{horizon}.pt"

            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

            # Create agent and load weights
            agent = IndependentAgent(
                agent_type=agent_type,
                feat_dim=325,
                use_feature_subset=True,
                num_classes=self.agent_output_dim,
            )
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            agent.load_state_dict(state_dict)

            # Freeze all parameters
            for param in agent.parameters():
                param.requires_grad = False
            agent.eval()

            self.frozen_agents.append(agent)
            print(f"  Loaded & frozen Agent {agent_type} from {ckpt_path.name}")

        print(f"  Total frozen agents: {len(self.frozen_agents)}")

    def load_vix_warmstart(self, vix_checkpoint_path: str):
        """
        Load warm-started Agent VIX weights from Stage 1 regime training.

        Args:
            vix_checkpoint_path: Path to Agent VIX .pt checkpoint
        """
        state_dict = torch.load(vix_checkpoint_path, map_location='cpu', weights_only=True)
        self.vix_agent.load_state_dict(state_dict)
        print(f"  Loaded VIX warm-start from {vix_checkpoint_path}")

    def forward(
        self,
        sequences_1min: torch.Tensor,
        vix_features_5min: torch.Tensor,
        chain_2d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            sequences_1min: (B, seq_len, 325) — 1-min option chain sequences
            vix_features_5min: (B, vix_feat_dim) — 5-min VIX features (latest bar)
            chain_2d: Optional (B, 5, n_strikes, seq_len) — for Agent 2D

        Returns:
            dict with:
                'logits': (B, num_classes) — final directional prediction
                'regime_logits': (B, 5) — auxiliary regime classification
                'regime_emb': (B, regime_emb_dim) — regime embedding
                'gate_weights': dict[str, (B, 1)] — per-agent gate weights
                'agent_probs': list[(B, num_classes)] — per-agent softmax probs
                'effective_threshold': (B, 1) — dynamic threshold
        """
        batch_size = sequences_1min.size(0)

        # ── 1. Run frozen directional agents ─────────────────────────────
        agent_logits = []
        with torch.no_grad():
            for i, agent in enumerate(self.frozen_agents):
                agent_type = self.AGENT_TYPES[i]
                if agent_type == '2D' and chain_2d is not None:
                    logit = agent(sequences_1min, chain_2d=chain_2d)
                else:
                    logit = agent(sequences_1min)
                agent_logits.append(logit)

        # Convert to probabilities
        agent_probs = [F.softmax(logit, dim=-1) for logit in agent_logits]

        # ── 2. Run Agent VIX ─────────────────────────────────────────────
        (vix_score, vix_confidence, regime_logits,
         regime_emb, rule_probs) = self.vix_agent.forward_with_regime_emb(vix_features_5min)

        # ── 3. Compute per-agent gate weights ────────────────────────────
        gate_weights = {}
        gated_probs = []

        for i, agent_type in enumerate(self.AGENT_TYPES):
            gate_w = self.gates[agent_type](regime_emb)  # (B, 1)
            gate_weights[agent_type] = gate_w

            # Apply gate: scale agent probability by gate weight
            gated_prob = agent_probs[i] * gate_w  # (B, num_classes) × (B, 1)
            gated_probs.append(gated_prob)

        # ── 4. Fuse gated outputs + regime embedding ─────────────────────
        # Concatenate all gated probabilities + regime embedding
        gated_concat = torch.cat(gated_probs, dim=-1)  # (B, n_agents * num_classes)
        fusion_input = torch.cat([gated_concat, regime_emb], dim=-1)

        # Final prediction
        logits = self.fusion(fusion_input)

        # ── 5. Auxiliary outputs ──────────────────────────────────────────
        regime_aux_logits = self.regime_aux_head(regime_emb)

        # Dynamic threshold
        risk_score = vix_score  # Use VIX severity as risk proxy
        effective_threshold = self.base_threshold + self.threshold_alpha * risk_score

        return {
            'logits': logits,
            'regime_logits': regime_aux_logits,
            'regime_emb': regime_emb,
            'gate_weights': gate_weights,
            'agent_probs': agent_probs,
            'effective_threshold': effective_threshold,
            'vix_confidence': vix_confidence,
            'rule_probs': rule_probs,
        }

    def get_gate_summary(self, gate_weights: Dict[str, torch.Tensor]) -> dict:
        """
        Return human-readable gate weight summary (for logging/dashboard).
        """
        summary = {}
        for agent_type, weight in gate_weights.items():
            w = weight.mean().item()
            summary[agent_type] = round(w, 4)
        return summary

    def count_parameters(self) -> dict:
        """Count trainable vs frozen parameters."""
        frozen = sum(p.numel() for p in self.frozen_agents.parameters())
        vix = sum(p.numel() for p in self.vix_agent.parameters())
        gates = sum(p.numel() for p in self.gates.parameters())
        fusion = sum(p.numel() for p in self.fusion.parameters())
        aux = sum(p.numel() for p in self.regime_aux_head.parameters())
        threshold = 2  # alpha + base

        trainable = vix + gates + fusion + aux + threshold
        total = frozen + trainable

        return {
            'frozen_agents': frozen,
            'vix_agent': vix,
            'gates': gates,
            'fusion': fusion,
            'regime_aux_head': aux,
            'threshold_params': threshold,
            'total_trainable': trainable,
            'total_frozen': frozen,
            'total': total,
        }


# ============================================================================
# Loss Function for Stage 3
# ============================================================================

class RegimeGatedLoss(nn.Module):
    """
    Multi-task loss for the RegimeGatedMetaModel.

    Components:
        1. Directional loss: CrossEntropy on final UP/DOWN prediction
        2. Regime auxiliary loss: CrossEntropy on regime classification
        3. Gate regularization: Encourage gates to be decisive (near 0 or 1)

    Args:
        direction_weight: Weight for directional loss (default: 1.0)
        regime_weight: Weight for auxiliary regime loss (default: 0.3)
        gate_reg_weight: Weight for gate entropy regularization (default: 0.01)
    """

    def __init__(
        self,
        direction_weight: float = 1.0,
        regime_weight: float = 0.3,
        gate_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.regime_weight = regime_weight
        self.gate_reg_weight = gate_reg_weight

        self.direction_loss_fn = nn.CrossEntropyLoss()
        self.regime_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        direction_labels: torch.Tensor,
        regime_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            model_output: Output dict from RegimeGatedMetaModel.forward()
            direction_labels: (B,) — 0=DOWN, 1=UP
            regime_labels: (B,) — 0-4 regime class (optional)

        Returns:
            dict with 'total', 'direction', 'regime', 'gate_reg'
        """
        # 1. Directional loss
        direction_loss = self.direction_loss_fn(
            model_output['logits'], direction_labels
        )

        # 2. Regime auxiliary loss (if labels provided)
        regime_loss = torch.tensor(0.0, device=direction_loss.device)
        if regime_labels is not None:
            regime_loss = self.regime_loss_fn(
                model_output['regime_logits'], regime_labels
            )

        # 3. Gate regularization: encourage decisive gates
        # Binary entropy: H = -g*log(g) - (1-g)*log(1-g)
        # Minimizing this pushes gates toward 0 or 1
        gate_reg = torch.tensor(0.0, device=direction_loss.device)
        for agent_type, gate_w in model_output['gate_weights'].items():
            g = gate_w.clamp(1e-6, 1 - 1e-6)
            entropy = -(g * g.log() + (1 - g) * (1 - g).log())
            gate_reg = gate_reg + entropy.mean()
        gate_reg = gate_reg / len(model_output['gate_weights'])

        # Total
        total = (
            self.direction_weight * direction_loss
            + self.regime_weight * regime_loss
            + self.gate_reg_weight * gate_reg
        )

        return {
            'total': total,
            'direction': direction_loss,
            'regime': regime_loss,
            'gate_reg': gate_reg,
        }


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing RegimeGatedMetaModel...")
    print("=" * 60)

    # Create model (without loading frozen agents — just test shapes)
    model = RegimeGatedMetaModel(
        n_agents=7,
        agent_output_dim=2,
        regime_emb_dim=32,
        vix_feat_dim=10,
        num_classes=2,
    )

    params = model.count_parameters()
    print(f"\nParameter counts:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # Test gate networks independently
    batch_size = 8
    regime_emb = torch.randn(batch_size, 32)

    print(f"\nGate network test (regime_emb → gate weights):")
    for agent_type, gate in model.gates.items():
        w = gate(regime_emb)
        print(f"  Agent {agent_type}: gate_weight mean={w.mean().item():.4f}, "
              f"std={w.std().item():.4f}")

    # Test VIX agent
    vix_features = torch.randn(batch_size, 10)
    with torch.no_grad():
        score, conf, logits, emb, rule = model.vix_agent.forward_with_regime_emb(vix_features)
    print(f"\nVIX agent test:")
    print(f"  regime_emb: {emb.shape}")
    print(f"  regime_logits: {logits.shape}")
    print(f"  rule_probs: {rule.shape}")

    # Test loss function
    loss_fn = RegimeGatedLoss()
    mock_output = {
        'logits': torch.randn(batch_size, 2),
        'regime_logits': torch.randn(batch_size, 5),
        'gate_weights': {t: torch.rand(batch_size, 1) for t in model.AGENT_TYPES},
    }
    direction_labels = torch.randint(0, 2, (batch_size,))
    regime_labels = torch.randint(0, 5, (batch_size,))

    losses = loss_fn(mock_output, direction_labels, regime_labels)
    print(f"\nLoss test:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    print(f"\n{'='*60}")
    print("✓ RegimeGatedMetaModel tests passed!")
