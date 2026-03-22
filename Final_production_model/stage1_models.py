"""
Stage 1 model definitions and checkpoint loader.

Exports:
    BinaryIndependentAgent  — wrapper matching training checkpoint key format
    _build_model_from_ckpt  — reconstruct model from a .pt checkpoint dict
    _Stage1Bundle           — (model, norm_mean, norm_std) named container
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from hybrid51_models.independent_agent import IndependentAgent


class BinaryIndependentAgent(nn.Module):
    """Wraps IndependentAgent as self.base so state-dict keys carry the 'base.' prefix."""

    def __init__(
        self,
        agent_type: str,
        feat_dim: int = 325,
        temporal_dim: int = 128,
        dropout: float = 0.2,
        use_feature_subset: bool = True,
        use_attention_backbone: bool = False,
        use_attention_pool: bool = False,
        cls_input_dim: Optional[int] = None,
        seq_len: int = 20,
    ):
        super().__init__()
        self.base = IndependentAgent(
            agent_type=agent_type,
            feat_dim=feat_dim,
            seq_len=seq_len,
            temporal_dim=temporal_dim,
            dropout=dropout,
            num_classes=5,
            use_feature_subset=use_feature_subset,
            use_attention_backbone=use_attention_backbone,
            use_attention_pool=use_attention_pool,
        )
        if cls_input_dim is None:
            cls_input_dim = (2 + temporal_dim) if self.base.use_backbone else (2 + 32)
        self.base.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, sequences: torch.Tensor, chain_2d: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.base(sequences, chain_2d=chain_2d).squeeze(-1)


def _build_model_from_ckpt(
    ckpt: dict,
    agent_type: str,
    device: torch.device,
    symbol: str = "SPXW",
    seq_len: int = 20,
) -> nn.Module:
    """Reconstruct a BinaryIndependentAgent that exactly matches a saved checkpoint."""
    state = ckpt["model_state_dict"]

    # Hybrid52 production inference uses 325-dim features only (no legacy 650-dim path).
    if "feat_dim" in ckpt:
        feat_dim = int(ckpt["feat_dim"])
        if feat_dim == 650:
            feat_dim = 325
    else:
        feat_dim = 325

    use_subset    = bool(ckpt.get("feature_subset", True))
    use_attn_bb   = bool(ckpt.get("use_attention_backbone", False))
    use_attn_pool = bool(ckpt.get("use_attention_pool", False))
    cls_in_dim    = int(state["base.classifier.0.weight"].shape[1])
    has_static    = "base.static_proj.weight" in state

    model = BinaryIndependentAgent(
        agent_type=agent_type,
        feat_dim=feat_dim,
        seq_len=seq_len,
        use_feature_subset=use_subset,
        use_attention_backbone=use_attn_bb,
        use_attention_pool=use_attn_pool,
        cls_input_dim=cls_in_dim,
    ).to(device)

    if not has_static and hasattr(model.base, "static_proj"):
        del model.base.static_proj

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class _Stage1Bundle:
    """Lightweight container for one (symbol, agent) Stage-1 model + normalisation stats."""

    __slots__ = ("model", "norm_mean", "norm_std", "platt_coef", "platt_intercept")

    def __init__(
        self,
        model: nn.Module,
        norm_mean: Optional[np.ndarray],
        norm_std: Optional[np.ndarray],
        platt_coef: float = 1.0,
        platt_intercept: float = 0.0,
    ):
        self.model = model
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.platt_coef = platt_coef
        self.platt_intercept = platt_intercept
