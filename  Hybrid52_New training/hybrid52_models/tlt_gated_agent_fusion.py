"""
Stage 2: TLT-Gated Cross-Symbol Agent Fusion

Each peer symbol's contribution is conditioned by a **learned gate** derived
from a compact TLT macro-context embedding.  No peer priority is hardcoded —
the model learns from training data which peer to trust under each TLT regime.

Input layout  (n_peers=3, tlt_ctx_dim=2 → total 15 dims):
  [logit_SPXW, prob_SPXW,                  (2) SPXW direct signal
   logit_P0,   prob_P0,                    (2) peer 0
   logit_P1,   prob_P1,                    (2) peer 1
   logit_P2,   prob_P2,                    (2) peer 2
   diff_P0,    diff_P1,   diff_P2,         (n_peers) SPXW logit – peer logit
   chain_ctx_logit, chain_ctx_prob,        (2) chain context
   tlt_logit,  tlt_prob]                   (2) frozen TLT same-agent Stage1 output

Gate formula (same calibrated blend as Stage 3 VIX gate):
  gated_peer = gate * prob_peer + (1 - gate) * 0.5
  → high gate  : trust peer fully
  → low gate   : shrink peer toward neutral (0.5)

Design rule: TLT macro embedding flows ONLY through gate networks.
It is never concatenated directly into the fusion MLP input.
This prevents the regime signal from drowning the directional signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TLTGatedAgentFusion(nn.Module):
    """
    Learned TLT-conditioned peer-trust fusion for Stage 2.

    Args:
        n_peers:          Number of directional peer symbols (default 3: SPY/QQQ/IWM)
        tlt_ctx_dim:      TLT context feature dims (default 2: logit + prob)
        tlt_emb_dim:      TLT macro embedding dim (default 16)
        gate_hidden_dim:  Hidden dim in each per-peer gate MLP (default 8)
        fusion_hidden_dim: Hidden dim in final fusion MLP (default 32)
        dropout:          Dropout in fusion MLP (default 0.2)
        has_chain_ctx:    Whether 2 chain-context dims are present between diffs and TLT.
                          True for standard agents (A/B/C/K/T/Q), False for Agent 2D.
    """

    def __init__(
        self,
        n_peers: int = 3,
        tlt_ctx_dim: int = 2,
        tlt_emb_dim: int = 16,
        gate_hidden_dim: int = 8,
        fusion_hidden_dim: int = 32,
        dropout: float = 0.2,
        has_chain_ctx: bool = True,
    ):
        super().__init__()
        self.n_peers = n_peers
        self.tlt_ctx_dim = tlt_ctx_dim
        self.tlt_emb_dim = tlt_emb_dim
        self.has_chain_ctx = has_chain_ctx

        # Pre-compute fixed slice boundaries from n_peers
        self._spxw_slice = slice(0, 2)
        self._peers_start = 2
        self._diffs_start = 2 + 2 * n_peers
        self._chain_start = 2 + 2 * n_peers + n_peers
        # TLT starts after chain context (2 dims) if present, else right after diffs
        self._tlt_start   = self._chain_start + (2 if has_chain_ctx else 0)

        # TLT macro encoder: 2 → 16 with stable LayerNorm
        self.tlt_encoder = nn.Sequential(
            nn.Linear(tlt_ctx_dim, tlt_emb_dim),
            nn.GELU(),
            nn.LayerNorm(tlt_emb_dim),
        )

        # Per-peer gate MLPs — each learns independently from TLT embedding
        self.peer_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tlt_emb_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_peers)
        ])

        # Fusion MLP input:
        #   SPXW logit/prob (2) + gated peer probs (n_peers)
        #   + chain context (2 if standard agent, 0 if Agent 2D)
        #   + summary stats: mean, std, spread (3)
        #   TLT embedding intentionally excluded here
        chain_dims = 2 if has_chain_ctx else 0
        fusion_in = 2 + n_peers + chain_dims + 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    @property
    def n_inputs(self) -> int:
        """Total expected input dimension."""
        return self._tlt_start + self.tlt_ctx_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, n_inputs) — see module docstring for layout

        Returns:
            logits: (B,) directional prediction
            gates:  (B, n_peers) learned TLT-conditioned gate values in [0, 1]
        """
        spxw_logit = x[:, 0:1]
        spxw_prob  = x[:, 1:2]

        # Peer probs: odd indices inside the peers block
        # peer i: logit at 2+2i, prob at 2+2i+1
        peer_probs = [x[:, self._peers_start + 2 * i + 1 : self._peers_start + 2 * i + 2]
                      for i in range(self.n_peers)]

        chain_ctx = (x[:, self._chain_start : self._tlt_start]           # (B, 2)
                     if self.has_chain_ctx else None)
        tlt_ctx   = x[:, self._tlt_start   : self._tlt_start + self.tlt_ctx_dim]  # (B, 2)

        # TLT macro embedding (regime info flows only to gate MLPs)
        tlt_emb = self.tlt_encoder(tlt_ctx)                               # (B, tlt_emb_dim)

        # Per-peer learned gates + calibrated trust blend
        gate_list   = [net(tlt_emb) for net in self.peer_gates]           # list of (B,1)
        gates       = torch.cat(gate_list, dim=1)                         # (B, n_peers)
        peer_p_cat  = torch.cat(peer_probs, dim=1)                        # (B, n_peers)
        gated       = gates * peer_p_cat + (1.0 - gates) * 0.5           # trust-weighted blend

        # Summary statistics over gated peers
        mean_g   = gated.mean(dim=1, keepdim=True)
        std_g    = gated.std(dim=1, keepdim=True, unbiased=False)
        spread_g = gated.max(dim=1, keepdim=True).values - gated.min(dim=1, keepdim=True).values

        # Fusion (TLT emb does NOT enter here — only through gates)
        parts = [spxw_logit, spxw_prob, gated]
        if self.has_chain_ctx and chain_ctx is not None:
            parts.append(chain_ctx)
        parts.extend([mean_g, std_g, spread_g])
        fusion_x = torch.cat(parts, dim=1)
        logits = self.fusion(fusion_x).squeeze(-1)
        return logits, gates

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def gate_summary(gates: torch.Tensor, peer_names: list[str]) -> dict[str, float]:
        """Return mean gate value per peer (for logging/inspection)."""
        mean_g = gates.mean(dim=0).detach().cpu().numpy()
        return {name: float(mean_g[i]) for i, name in enumerate(peer_names)}
