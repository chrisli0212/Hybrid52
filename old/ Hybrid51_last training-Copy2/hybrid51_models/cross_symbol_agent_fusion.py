"""
Stage 2: Cross-Symbol Agent Fusion Model

One small MLP per agent type that combines the same agent's logits
across all 5 symbols (SPXW, SPY, QQQ, IWM, TLT) into a refined SPXW signal.

Input layout:
  Standard agents (A/B/C/K/T/Q) — n_inputs=16:
    [logit_SPXW, prob_SPXW,                 (2)
     logit_SPY,  prob_SPY,                  (2)
     logit_QQQ,  prob_QQQ,                  (2)
     logit_IWM,  prob_IWM,                  (2)
     logit_TLT,  prob_TLT,                  (2)
     diff_SPXW-SPY, diff_SPXW-QQQ,          (2)
     diff_SPXW-IWM, diff_SPXW-TLT,          (2)
     chain_context_logit, chain_context_prob] (2)
    = 16 dims

  Agent 2D cross-symbol — n_inputs=14:
    [logit_SPXW_2D, prob_SPXW_2D,            (2)  ← SPXW unfrozen during training
     logit_SPY_2D,  prob_SPY_2D,             (2)
     logit_QQQ_2D,  prob_QQQ_2D,             (2)
     logit_IWM_2D,  prob_IWM_2D,             (2)
     logit_TLT_2D,  prob_TLT_2D,             (2)
     diff_SPXW-SPY, diff_SPXW-QQQ,           (2)
     diff_SPXW-IWM, diff_SPXW-TLT]           (2)
    = 14 dims
"""

import torch
import torch.nn as nn


class CrossSymbolAgentFusion(nn.Module):
    """
    Lightweight MLP that fuses cross-symbol same-agent logits.

    Args:
        n_inputs: Total input dimension (16 for standard agents, 14 for 2D)
        hidden_dim: Hidden layer width (default: 32)
        dropout: Dropout probability (default: 0.2)
    """

    def __init__(self, n_inputs: int = 16, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.n_inputs = n_inputs

        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_inputs) float tensor
        Returns:
            logits: (batch_size,) float tensor
        """
        return self.net(x).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
