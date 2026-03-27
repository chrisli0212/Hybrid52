"""
Agent T: Trade Flow Agent
Analyzes order flow, trade aggression, and market impact from trade/quote data.
~250k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentT(nn.Module):
    def __init__(
        self, 
        trade_feat_dim: int = 25,
        temporal_dim: int = 0,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.trade_feat_dim = trade_feat_dim
        
        # Input normalization for trade features
        self.input_norm = nn.LayerNorm(trade_feat_dim)
        
        # Trade flow encoder
        self.flow_encoder = nn.Sequential(
            nn.Linear(trade_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        
        # Temporal flow patterns (1D CNN on trade sequence)
        self.flow_cnn = nn.Sequential(
            nn.Conv1d(trade_feat_dim, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(8, 64),  # GroupNorm instead of BatchNorm for stability
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Impact detector (detects large market impact events)
        self.impact_net = nn.Sequential(
            nn.Linear(trade_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        
        # Fusion with backbone temporal features
        fusion_dim = 128 + temporal_dim + 32 + 32
        self.fusion = nn.Linear(fusion_dim, 64)
        self.fusion_norm = nn.LayerNorm(64)
        
        # Output heads
        self.score_head = nn.Linear(64, 1)
        self.confidence_head = nn.Linear(64, 1)
        self.flow_signal_head = nn.Linear(64, 3)  # Buy/Neutral/Sell signal
    
    def forward(
        self, 
        static: torch.Tensor, 
        temporal: Optional[torch.Tensor], 
        seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            static: (B, trade_feat_dim) - Last trade flow features
            temporal: (B, temporal_dim) - Backbone temporal embedding
            seq: (B, seq_len, trade_feat_dim) - Trade flow sequence
            
        Returns:
            score: (B, 1) - Direction score
            confidence: (B, 1) - Confidence score
            flow_signal: (B, 3) - Buy/Neutral/Sell logits
        """
        batch_size = static.size(0)
        
        # Handle case where static has more features than expected
        if static.size(1) > self.trade_feat_dim:
            static = static[:, :self.trade_feat_dim]
        elif static.size(1) < self.trade_feat_dim:
            # Pad if needed
            padding = torch.zeros(
                batch_size, 
                self.trade_feat_dim - static.size(1),
                device=static.device
            )
            static = torch.cat([static, padding], dim=1)
        
        # Apply input normalization
        static = self.input_norm(static)
        
        # Encode static trade flow features
        flow_encoded = self.flow_encoder(static)
        
        # Process sequence with CNN
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        
        # Ensure seq has correct feature dimension
        if seq.size(2) > self.trade_feat_dim:
            seq = seq[:, :, :self.trade_feat_dim]
        elif seq.size(2) < self.trade_feat_dim:
            padding = torch.zeros(
                seq.size(0), seq.size(1), 
                self.trade_feat_dim - seq.size(2),
                device=seq.device
            )
            seq = torch.cat([seq, padding], dim=2)
        
        # Apply input normalization to sequence
        seq = self.input_norm(seq)
        
        flow_temporal = self.flow_cnn(seq.transpose(1, 2)).squeeze(-1)
        
        # Detect market impact
        impact_score = self.impact_net(static)
        
        # Fuse all components
        if temporal is not None:
            fused = torch.cat([flow_encoded, temporal, flow_temporal, impact_score], dim=-1)
        else:
            fused = torch.cat([flow_encoded, flow_temporal, impact_score], dim=-1)
            
        fused = self.fusion_norm(F.gelu(self.fusion(fused)))
        
        # Generate outputs
        score = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))
        flow_signal = self.flow_signal_head(fused)
        
        return score, confidence, flow_signal
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
