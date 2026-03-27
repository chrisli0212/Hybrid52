"""
Agent Q: Quote Dynamics Agent
Analyzes quote updates, spread dynamics, and order book behavior from quote data.
~200k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AgentQ(nn.Module):
    def __init__(
        self, 
        quote_feat_dim: int = 20,
        temporal_dim: int = 0,
        hidden_dim: int = 192
    ):
        super().__init__()
        
        self.quote_feat_dim = quote_feat_dim
        
        # Input normalization for quote features
        self.input_norm = nn.LayerNorm(quote_feat_dim)
        
        # Quote pattern encoder
        self.quote_encoder = nn.Sequential(
            nn.Linear(quote_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 96),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Spread dynamics (BiLSTM for temporal patterns)
        self.spread_lstm = nn.LSTM(
            quote_feat_dim, 
            64, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.1
        )
        self.lstm_norm = nn.LayerNorm(128)
        
        # Order book imbalance detector
        self.imbalance_net = nn.Sequential(
            nn.Linear(quote_feat_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.Tanh()
        )
        
        # Fusion with backbone
        fusion_dim = 96 + temporal_dim + 128 + 24
        self.fusion = nn.Linear(fusion_dim, 64)
        self.fusion_norm = nn.LayerNorm(64)
        
        # Output heads
        self.score_head = nn.Linear(64, 1)
        self.confidence_head = nn.Linear(64, 1)
        self.spread_signal_head = nn.Linear(64, 1)  # Spread direction signal
    
    def forward(
        self, 
        static: torch.Tensor, 
        temporal: Optional[torch.Tensor], 
        seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            static: (B, quote_feat_dim) - Last quote features
            temporal: (B, temporal_dim) - Backbone temporal embedding
            seq: (B, seq_len, quote_feat_dim) - Quote sequence
            
        Returns:
            score: (B, 1) - Direction score
            confidence: (B, 1) - Confidence score
            spread_signal: (B, 1) - Spread direction signal
        """
        batch_size = static.size(0)
        
        # Handle feature dimension mismatches
        if static.size(1) > self.quote_feat_dim:
            static = static[:, :self.quote_feat_dim]
        elif static.size(1) < self.quote_feat_dim:
            padding = torch.zeros(
                batch_size, 
                self.quote_feat_dim - static.size(1),
                device=static.device
            )
            static = torch.cat([static, padding], dim=1)
        
        # Apply input normalization
        static = self.input_norm(static)
        
        # Encode static quote features
        quote_encoded = self.quote_encoder(static)
        
        # Process sequence with BiLSTM
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        
        # Ensure seq has correct feature dimension
        if seq.size(2) > self.quote_feat_dim:
            seq = seq[:, :, :self.quote_feat_dim]
        elif seq.size(2) < self.quote_feat_dim:
            padding = torch.zeros(
                seq.size(0), seq.size(1), 
                self.quote_feat_dim - seq.size(2),
                device=seq.device
            )
            seq = torch.cat([seq, padding], dim=2)
        
        # Apply input normalization to sequence
        seq = self.input_norm(seq)
        
        # BiLSTM for spread dynamics
        lstm_out, _ = self.spread_lstm(seq)
        spread_features = self.lstm_norm(lstm_out[:, -1, :])
        
        # Detect order book imbalance
        imbalance_score = self.imbalance_net(static)
        
        # Fuse all components
        if temporal is not None:
            fused = torch.cat([quote_encoded, temporal, spread_features, imbalance_score], dim=-1)
        else:
            fused = torch.cat([quote_encoded, spread_features, imbalance_score], dim=-1)
            
        fused = self.fusion_norm(F.gelu(self.fusion(fused)))
        
        # Generate outputs
        score = torch.sigmoid(self.score_head(fused))
        confidence = torch.sigmoid(self.confidence_head(fused))
        spread_signal = torch.tanh(self.spread_signal_head(fused))
        
        return score, confidence, spread_signal
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
