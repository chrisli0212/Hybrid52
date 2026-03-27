"""
Agent K: Greeks Core Agent
Deep network focused on first 127 core Greek features.
~315k parameters
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class AgentK(nn.Module):
    def __init__(self, input_dim: int = 127, hidden_dim: int = 512):
        super().__init__()
        
        self.core_features = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 64)
        )
        
        self.layer_norm = nn.LayerNorm(64)
        
        self.score_head = nn.Linear(64, 1)
        self.confidence_head = nn.Linear(64, 1)
        
        self.gamma_squeeze_detector = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        static: torch.Tensor, 
        temporal: torch.Tensor, 
        seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        core_features = static[:, :self.core_features]
        
        if core_features.size(1) < self.core_features:
            padding = torch.zeros(
                core_features.size(0), 
                self.core_features - core_features.size(1),
                device=core_features.device
            )
            core_features = torch.cat([core_features, padding], dim=1)
        
        out = self.net(core_features)
        out = self.layer_norm(out)
        
        gamma_squeeze = self.gamma_squeeze_detector(core_features)
        
        score = torch.sigmoid(self.score_head(out))
        
        score = score + gamma_squeeze * 0.1
        score = torch.clamp(score, 0.01, 0.99)
        
        confidence = torch.sigmoid(self.confidence_head(out))
        
        return score, confidence, None
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
