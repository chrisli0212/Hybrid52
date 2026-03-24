"""
Agent A: Neural Baseline Agent
Primary temporal agent with static MLP + causal CNN + backbone fusion.
~195k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentA(nn.Module):
    def __init__(
        self, 
        input_dim: int = 158, 
        temporal_dim: int = 128,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Static feature path
        self.static_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 96)
        )
        
        # Residual projection for skip connection
        self.residual_proj = nn.Linear(input_dim, 96)
        
        self.causal_head = nn.Sequential(
            nn.Conv1d(input_dim, 48, kernel_size=4, padding=3, groups=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        fusion_dim = 96 + temporal_dim + 48
        self.fusion = nn.Linear(fusion_dim, 64)
        self.fusion_norm = nn.LayerNorm(64)
        
        self.score_head = nn.Linear(64, 1)
        self.confidence_head = nn.Linear(64, 1)
        self.signal_head = nn.Linear(64, 5)
    
    def forward(
        self, 
        static: torch.Tensor, 
        temporal: torch.Tensor, 
        seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Process static features with residual connection
        static_out = self.static_path(static) + self.residual_proj(static)
        
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        t = self.causal_head(seq.transpose(1, 2)).squeeze(-1)
        
        fused = torch.cat([static_out, temporal, t], dim=-1)
        fused = self.fusion_norm(F.gelu(self.fusion(fused)))
        
        score = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))
        signal = self.signal_head(fused)
        
        return score, confidence, signal
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
