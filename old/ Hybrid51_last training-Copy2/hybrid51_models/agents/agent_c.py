"""
Agent C: Multi-Scale Attention Agent
CNN + Multi-head Attention + BiLSTM with attention pooling.
~461k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentC(nn.Module):
    def __init__(
        self, 
        input_dim: int = 158,
        seq_len: int = 20,
        embed_dim: int = 96,
        n_heads: int = 4
    ):
        super().__init__()
        
        self.temporal_weights = nn.Parameter(torch.linspace(0.4, 1.6, seq_len))
        
        self.embedding = nn.Linear(min(input_dim, 32), embed_dim)
        
        self.cnn_local = nn.Conv1d(embed_dim, 24, kernel_size=3, padding=1)
        self.cnn_medium = nn.Conv1d(embed_dim, 24, kernel_size=5, padding=2)
        self.cnn_long = nn.Conv1d(embed_dim, 24, kernel_size=7, padding=3)
        self.ln_cnn = nn.LayerNorm(72)
        
        self.attention = nn.MultiheadAttention(72, num_heads=n_heads, batch_first=True)
        
        self.lstm = nn.LSTM(72, 96, batch_first=True, bidirectional=True)
        self.ln_lstm = nn.LayerNorm(192)
        
        self.pool_attention = nn.Sequential(
            nn.Linear(192, 96),
            nn.Tanh(),
            nn.Linear(96, 1)
        )
        
        self.residual_proj = nn.Linear(72, 192)
        
        self.dense_head = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(
        self, 
        static: torch.Tensor, 
        temporal: torch.Tensor, 
        seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        
        batch_size, seq_len, feat_dim = seq.shape
        
        weights = self.temporal_weights[:seq_len].view(1, seq_len, 1)
        seq_weighted = seq * weights
        
        x_embed = self.embedding(seq_weighted[:, :, :32])
        
        x_conv = x_embed.transpose(1, 2)
        local = self.cnn_local(x_conv)
        medium = self.cnn_medium(x_conv)
        long = self.cnn_long(x_conv)
        
        combined = torch.cat([local, medium, long], dim=1)
        combined = combined.transpose(1, 2)  # (B, seq_len, 72)
        combined = self.ln_cnn(combined)
        combined = F.relu(combined)
        
        attn_out, _ = self.attention(combined, combined, combined)
        
        residual = self.residual_proj(combined)
        
        lstm_out, _ = self.lstm(attn_out)
        lstm_out = self.ln_lstm(lstm_out + residual)
        
        attn_weights = self.pool_attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (lstm_out * attn_weights).sum(dim=1)
        
        out = self.dense_head(pooled)
        
        score = torch.sigmoid(out[:, 0:1])
        confidence = torch.sigmoid(out[:, 1:2])
        
        return score, confidence, None
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
