"""
Agent B: Bidirectional LSTM Sequence Agent
Stacked BiLSTM for capturing temporal dependencies.
~623k parameters
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class AgentB(nn.Module):
    def __init__(
        self, 
        input_dim: int = 158, 
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.lstm1 = nn.LSTM(
            input_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.lstm2 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.score_head = nn.Linear(32, 1)
        self.confidence_head = nn.Linear(32, 1)
    
    def forward(
        self, 
        static: torch.Tensor, 
        temporal: torch.Tensor, 
        seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        
        out1, _ = self.lstm1(seq)
        out2, _ = self.lstm2(out1)
        
        out = out2[:, -1, :]
        out = self.layer_norm(out)
        
        fused = self.fusion(out)
        
        score = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))
        
        return score, confidence, None
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
