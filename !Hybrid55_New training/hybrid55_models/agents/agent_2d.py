"""
Agent 2D: Option Chain Shape Agent (Experimental)
Treats option chain as 2D image (strikes × time × greeks) using CNN.
~200k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Agent2D(nn.Module):
    def __init__(
        self,
        n_greeks: int = 5,
        n_strikes: int = 30,
        n_timesteps: int = 30,
        base_channels: int = 32
    ):
        super().__init__()
        
        self.n_greeks = n_greeks
        self.n_strikes = n_strikes
        self.n_timesteps = n_timesteps
        
        self.conv1 = nn.Conv2d(n_greeks, base_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, base_channels)
        
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, base_channels * 2)
        
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(16, base_channels * 4)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        fc_input_dim = base_channels * 4 * 4 * 4
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        self.smile_detector = nn.Sequential(
            nn.Conv2d(n_greeks, 16, kernel_size=(n_strikes, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * n_timesteps, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.skew_detector = nn.Sequential(
            nn.Conv2d(n_greeks, 16, kernel_size=(3, n_timesteps), padding=(1, 0)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * n_strikes, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        static: torch.Tensor,
        temporal: torch.Tensor,
        seq: torch.Tensor,
        chain_2d: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size = static.size(0)
        device = static.device
        
        if chain_2d is None:
            raise ValueError("Agent2D requires real chain_2d input; received None.")
        
        x = F.relu(self.gn1(self.conv1(chain_2d)))
        x = F.max_pool2d(x, 2) if x.size(2) > 4 and x.size(3) > 4 else x
        
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.max_pool2d(x, 2) if x.size(2) > 4 and x.size(3) > 4 else x
        
        x = F.relu(self.gn3(self.conv3(x)))
        
        x = self.pool(x)
        x = x.view(batch_size, -1)
        
        out = self.fc(x)
        
        score = torch.sigmoid(out[:, 0:1])
        confidence = torch.sigmoid(out[:, 1:2])
        
        try:
            smile_signal = self.smile_detector(chain_2d)
            skew_signal = self.skew_detector(chain_2d)
            
            score = score + (smile_signal - 0.5) * 0.1
            score = score + (skew_signal - 0.5) * 0.1
            score = torch.clamp(score, 0.01, 0.99)
        except Exception as e:
            import warnings
            warnings.warn(f"smile/skew detector failed: {e}", RuntimeWarning)
        
        return score, confidence, None
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Agent2DWithResidual(Agent2D):
    def __init__(
        self,
        n_greeks: int = 5,
        n_strikes: int = 30,
        n_timesteps: int = 30,
        base_channels: int = 32
    ):
        super().__init__(n_greeks, n_strikes, n_timesteps, base_channels)
        
        self.res_conv1 = nn.Conv2d(n_greeks, base_channels * 4, kernel_size=1)
        
    def forward(
        self,
        static: torch.Tensor,
        temporal: torch.Tensor,
        seq: torch.Tensor,
        chain_2d: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size = static.size(0)
        device = static.device
        
        if chain_2d is None:
            raise ValueError("Agent2D requires real chain_2d input; received None.")
        
        residual = self.res_conv1(chain_2d)
        
        x = F.relu(self.gn1(self.conv1(chain_2d)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        
        if x.shape == residual.shape:
            x = x + residual
        else:
            residual = F.adaptive_avg_pool2d(residual, x.shape[2:])
            x = x + residual
        
        x = self.pool(x)
        x = x.view(batch_size, -1)
        
        out = self.fc(x)
        
        score = torch.sigmoid(out[:, 0:1])
        confidence = torch.sigmoid(out[:, 1:2])
        
        return score, confidence, None
