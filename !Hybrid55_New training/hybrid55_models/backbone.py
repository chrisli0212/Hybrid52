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
        use_attention_pool: bool = True,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.use_attention_pool = use_attention_pool

        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Causal conv: pad only on the LEFT so bar t never sees bar t+1..t+k
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(
                hidden_dim, hidden_dim,
                kernel_size=k,
                padding=0,          # no built-in padding — we pad manually in forward
                groups=hidden_dim
            )
            for k in kernel_sizes
        ])
        self.dw_conv_pads = [k - 1 for k in kernel_sizes]  # left-only causal pad sizes

        self.pw_combine = nn.Conv1d(hidden_dim * len(kernel_sizes), hidden_dim, kernel_size=1)
        self.ln = nn.LayerNorm(hidden_dim)

        # Last-timestep pool: after causal convs, position -1 has seen all 20 bars
        # AttentionPool still available as option; last-timestep is default
        self.pool = AttentionPool(hidden_dim) if use_attention_pool else None

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
        for conv, pad_size in zip(self.dw_convs, self.dw_conv_pads):
            # Causal: pad left side only so output[t] only sees input[0..t]
            x_padded = F.pad(x, (pad_size, 0))
            scale_out = F.gelu(conv(x_padded))
            scale_outputs.append(scale_out)

        x = torch.cat(scale_outputs, dim=1)

        x = self.pw_combine(x)
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)

        if self.use_attention_pool:
            x = self.pool(x)       # AttentionPool: (B, hidden_dim, T) → (B, hidden_dim)
        else:
            x = x[:, :, -1]        # Last timestep: causal summary of all 20 bars

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
        for conv, pad_size in zip(self.base_backbone.dw_convs, self.base_backbone.dw_conv_pads):
            x_padded = F.pad(x, (pad_size, 0))
            scale_out = F.gelu(conv(x_padded))
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
            x = x[:, :, -1]        # Last timestep after causal convs

        x = self.output_proj(x)
        x = self.output_norm(x)
        x = self.dropout(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

# ============================================================================
# DilatedCausalTCN  —  WaveNet-style residual TCN (replaces TemporalBackbone)
# Receptive field: kernel=2, dilations=[1,2,4,8] → covers 30 bars from 20
# All convolutions are CAUSAL (left-pad only) — no future leakage
# ============================================================================

class CausalResidualBlock(nn.Module):
    """Single dilated causal residual block."""
    def __init__(self, channels: int, kernel_size: int = 2,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation, padding=0)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.skip   = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        x = F.pad(x, (self.causal_pad, 0))
        x = self.conv1(x)                                   # (B, C, T)
        x = F.gelu(self.norm1(x.transpose(1, 2)).transpose(1, 2))
        x = self.dropout(x)
        x = F.pad(x, (self.causal_pad, 0))
        x = self.conv2(x)
        x = F.gelu(self.norm2(x.transpose(1, 2)).transpose(1, 2))
        x = self.dropout(x)
        return x + self.skip(residual)


class DilatedCausalTCN(nn.Module):
    """
    Proper TCN backbone replacing TemporalBackbone.
    - Causal dilated convolutions (no future leakage within the 20-bar window)
    - Residual connections (stable gradients)
    - Last-timestep readout (bar-20 = full causal summary of all 20 bars)
    - Receptive field = 2*(1+2+4+8) = 30 bars with default settings

    Drop-in replacement: same __init__ signature as TemporalBackbone.
    """
    def __init__(
        self,
        feat_dim:   int   = 158,
        hidden_dim: int   = 128,
        embed_dim:  int   = 128,
        dilations:  tuple = (1, 2, 4, 8),
        kernel_size: int  = 2,
        dropout:    float = 0.1,
        use_attention_pool: bool = False,   # kept for API compat, unused
        **kwargs,                           # absorb legacy kwargs
    ):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([
            CausalResidualBlock(hidden_dim, kernel_size, d, dropout)
            for d in dilations
        ])
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)  or  (B, D) — guard for 2-D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.gelu(self.input_norm(self.input_proj(x)))   # (B, T, hidden)
        x = x.transpose(1, 2)                              # (B, hidden, T)
        for block in self.blocks:
            x = block(x)
        x = x[:, :, -1]                    # last timestep — causal summary
        x = self.output_proj(x)
        x = self.output_norm(x)
        return self.dropout_out(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TemporalMixerBackbone(nn.Module):
    """
    Lightweight MLP-Mixer style temporal encoder.
    Uses token-mixing + channel-mixing blocks to provide a
    non-convolutional, non-recurrent mechanism for sequence modeling.
    """
    def __init__(
        self,
        feat_dim: int = 158,
        hidden_dim: int = 192,
        embed_dim: int = 128,
        seq_len: int = 20,
        depth: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.token_mixers = nn.ModuleList()
        self.channel_mixers = nn.ModuleList()
        for _ in range(depth):
            self.token_mixers.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(seq_len, seq_len),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(seq_len, seq_len),
                    nn.Dropout(dropout),
                )
            )
            self.channel_mixers.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout),
                )
            )

        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.size(1) != self.seq_len:
            if x.size(1) > self.seq_len:
                x = x[:, -self.seq_len:, :]
            else:
                pad_len = self.seq_len - x.size(1)
                x = F.pad(x, (0, 0, pad_len, 0))
        h = F.gelu(self.input_norm(self.input_proj(x)))  # (B,T,H)
        for tok, ch in zip(self.token_mixers, self.channel_mixers):
            t = h.transpose(1, 2)  # (B,H,T)
            t = tok[0](h).transpose(1, 2)
            t = tok[1:](t).transpose(1, 2)
            h = h + t
            h = h + ch(h)
        pooled = h.mean(dim=1)
        out = self.output_proj(pooled)
        out = self.output_norm(out)
        return self.dropout_out(out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

