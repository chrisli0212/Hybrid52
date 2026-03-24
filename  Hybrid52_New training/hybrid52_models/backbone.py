"""
Temporal Backbone - Shared multi-scale temporal feature extractor.
Based on Hybrid-40's proven TemporalBackbone32 architecture.

v2 Changes:
- Replaced BatchNorm1d with LayerNorm for stability with unscaled features

v3 Changes:
- Added AttentionPool: replaces AdaptiveAvgPool1d with learned attention pooling
- TemporalBackbone gains use_attention_pool flag
- TemporalBackboneWithAttention gains use_attention_pool flag
- IndependentAgent wires both via use_attention_backbone / use_attention_pool args
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionPool(nn.Module):
    """
    Learned attention pooling: replaces AdaptiveAvgPool1d.
    Computes a softmax-weighted sum over the time dimension.
    Input:  (B, hidden_dim, seq_len)
    Output: (B, hidden_dim)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)                         # (B, seq_len, hidden_dim)
        weights = torch.softmax(self.score(x_t), dim=1)  # (B, seq_len, 1)
        pooled = (x_t * weights).sum(dim=1)              # (B, hidden_dim)
        return pooled


class TemporalBackbone(nn.Module):
    def __init__(
        self,
        feat_dim: int = 158,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        kernel_sizes: tuple = (3, 5, 7),
        dropout: float = 0.1,
        use_attention_pool: bool = False,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.use_attention_pool = use_attention_pool

        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.dw_convs = nn.ModuleList([
            nn.Conv1d(
                hidden_dim, hidden_dim,
                kernel_size=k,
                padding=k // 2,
                groups=hidden_dim
            )
            for k in kernel_sizes
        ])

        self.pw_combine = nn.Conv1d(hidden_dim * len(kernel_sizes), hidden_dim, kernel_size=1)
        self.ln = nn.LayerNorm(hidden_dim)

        self.pool = AttentionPool(hidden_dim) if use_attention_pool else nn.AdaptiveAvgPool1d(1)

        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)

        x = x.transpose(1, 2)

        scale_outputs = []
        for conv in self.dw_convs:
            scale_out = F.gelu(conv(x))
            scale_outputs.append(scale_out)

        x = torch.cat(scale_outputs, dim=1)

        x = self.pw_combine(x)
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)

        if self.use_attention_pool:
            x = self.pool(x)
        else:
            x = self.pool(x).squeeze(-1)

        x = self.output_proj(x)
        x = self.output_norm(x)
        x = self.dropout(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TemporalBackboneWithAttention(nn.Module):
    """
    TemporalBackbone with multi-head self-attention inserted after input projection,
    before the multi-scale depthwise convolutions. Optionally uses AttentionPool.
    """
    def __init__(
        self,
        feat_dim: int = 158,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_attention_pool: bool = True,
    ):
        super().__init__()

        self.base_backbone = TemporalBackbone(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            embed_dim=hidden_dim,
            dropout=dropout,
            use_attention_pool=use_attention_pool,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.base_backbone.input_proj(x)
        x = self.base_backbone.input_norm(x)
        x = F.gelu(x)

        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)

        x = x.transpose(1, 2)

        scale_outputs = []
        for conv in self.base_backbone.dw_convs:
            scale_out = F.gelu(conv(x))
            scale_outputs.append(scale_out)

        x = torch.cat(scale_outputs, dim=1)
        x = self.base_backbone.pw_combine(x)
        x = x.transpose(1, 2)
        x = self.base_backbone.ln(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)

        if self.base_backbone.use_attention_pool:
            x = self.base_backbone.pool(x)
        else:
            x = self.base_backbone.pool(x).squeeze(-1)

        x = self.output_proj(x)
        x = self.output_norm(x)
        x = self.dropout(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
