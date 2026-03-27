"""
Stage 3 VIX regime-gated probability fusion.

This model gates already-computed per-agent probabilities using a VIX regime
embedding (AgentVIX), then produces the final directional logit.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .agents import AgentVIX


class RegimeGatedProbFusion(nn.Module):
    """Fuse 7 directional agent probabilities with VIX-conditioned gates."""

    def __init__(
        self,
        agent_names: List[str],
        vix_feat_dim: int = 10,
        regime_emb_dim: int = 32,
        gate_hidden_dim: int = 16,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.agent_names = list(agent_names)
        self.n_agents = len(self.agent_names)

        self.vix_agent = AgentVIX(vix_feat_dim=vix_feat_dim, regime_emb_dim=regime_emb_dim)

        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(regime_emb_dim, gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(gate_hidden_dim, 1),
                    nn.Sigmoid(),
                )
                for name in self.agent_names
            }
        )

        # gated probs + summary stats (regime info flows ONLY through gates)
        fusion_in = self.n_agents + 5
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    def forward(
        self,
        agent_probs: torch.Tensor,
        vix_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_probs: (B, n_agents), probabilities from Stage-2 per-agent models
            vix_features: (B, vix_feat_dim), aligned VIX regime feature vectors

        Returns:
            logits: (B,) final directional logits
            gates: (B, n_agents)
            regime_emb: (B, regime_emb_dim)
        """
        _, _, _, regime_emb = self.vix_agent.forward_with_regime_emb(vix_features)

        gate_list = [self.gates[name](regime_emb) for name in self.agent_names]
        gates = torch.cat(gate_list, dim=1)

        gated = gates * agent_probs + (1.0 - gates) * 0.5
        mean_p = gated.mean(dim=1, keepdim=True)
        std_p = gated.std(dim=1, keepdim=True, unbiased=False)
        max_p = gated.max(dim=1, keepdim=True).values
        min_p = gated.min(dim=1, keepdim=True).values
        spread = max_p - min_p

        x = torch.cat([gated, mean_p, std_p, max_p, min_p, spread], dim=1)
        logits = self.fusion(x).squeeze(-1)
        return logits, gates, regime_emb

    @staticmethod
    def gate_summary(gates: torch.Tensor, agent_names: List[str]) -> Dict[str, float]:
        mean_gate = gates.mean(dim=0).detach().cpu().numpy()
        return {name: float(mean_gate[i]) for i, name in enumerate(agent_names)}
