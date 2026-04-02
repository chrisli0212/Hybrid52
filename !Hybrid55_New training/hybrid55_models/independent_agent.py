"""
Phase 17 v2: Independent Agent Model with Feature Subsetting

Changes from v1:
- Feature subsetting: each agent sees only its designated feature subset
- Backbone feat_dim matches agent's subset dimension
- Support for disabling backbone (Agent K uses static MLP only)

Fix 2026-03-26:
- Removed inverse-sigmoid (logit restore) before classifier — score is passed raw
- classifier now receives (score, confidence, temporal_embed) directly
- num_classes forced to 1 for binary mode via BinaryIndependentAgent wrapper
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbone import (
    TemporalBackbone,
    TemporalBackboneWithAttention,
    DilatedCausalTCN,
    TemporalMixerBackbone,
)
from .agents import AgentA, AgentB, AgentC, AgentK, Agent2D, AgentTQ, AgentH, AgentM

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.feature_subsets import AGENT_FEATURE_SUBSETS, TOTAL_FEAT_DIM, get_feature_indices


class IndependentAgent(nn.Module):
    def __init__(
        self,
        agent_type: str = 'A',
        feat_dim: int = TOTAL_FEAT_DIM,
        temporal_dim: int = 128,
        dropout: float = 0.2,
        num_classes: int = 5,
        use_feature_subset: bool = True,
        use_attention_backbone: bool = False,
        use_attention_pool: bool = False,
        use_dilated_tcn: bool = True,   # default ON: DilatedCausalTCN replaces TemporalBackbone
        use_mixer_backbone: bool = False,
    ):
        super().__init__()

        self.agent_type = agent_type.upper()
        self.full_feat_dim = feat_dim
        self.temporal_dim = temporal_dim
        self.num_classes = num_classes
        self.use_feature_subset = use_feature_subset
        self.use_attention_backbone = use_attention_backbone
        self.use_attention_pool = use_attention_pool
        self.use_dilated_tcn = use_dilated_tcn
        self.use_mixer_backbone = use_mixer_backbone

        # Feature subsetting
        if use_feature_subset and self.agent_type in AGENT_FEATURE_SUBSETS:
            subset_config = AGENT_FEATURE_SUBSETS[self.agent_type]
            self.feature_indices = get_feature_indices(self.agent_type)
            if feat_dim == TOTAL_FEAT_DIM * 2 and len(self.feature_indices) > 0:
                self.feature_indices = self.feature_indices + [idx + TOTAL_FEAT_DIM for idx in self.feature_indices]
            self.subset_feat_dim = len(self.feature_indices)
            self.use_backbone = subset_config['use_backbone']
            self.register_buffer('_feat_idx', torch.tensor(self.feature_indices, dtype=torch.long))
        else:
            self.feature_indices = list(range(feat_dim))
            self.subset_feat_dim = feat_dim
            self.use_backbone = True
            self.register_buffer('_feat_idx', torch.arange(feat_dim, dtype=torch.long))

        # Agents B and C need Agent A's static indices separately
        if self.agent_type in ('B', 'C'):
            _static_indices = get_feature_indices('A')
            self.static_feat_dim = len(_static_indices)
            self.register_buffer('_static_idx', torch.tensor(_static_indices, dtype=torch.long))
        else:
            self._static_idx = None
            self.static_feat_dim = self.subset_feat_dim

        # Temporal backbone
        if self.use_backbone:
            if use_mixer_backbone or self.agent_type == 'M':
                self.backbone = TemporalMixerBackbone(
                    feat_dim=self.subset_feat_dim,
                    hidden_dim=192,
                    embed_dim=temporal_dim,
                    seq_len=20,
                    depth=3,
                    dropout=dropout,
                )
            elif use_dilated_tcn:
                # DilatedCausalTCN: causal receptive field 30 bars, residual blocks
                self.backbone = DilatedCausalTCN(
                    feat_dim=self.subset_feat_dim,
                    hidden_dim=128,
                    embed_dim=temporal_dim,
                    dilations=(1, 2, 4, 8),
                    dropout=dropout,
                )
            elif use_attention_backbone:
                self.backbone = TemporalBackboneWithAttention(
                    feat_dim=self.subset_feat_dim,
                    embed_dim=temporal_dim,
                    use_attention_pool=use_attention_pool,
                )
            else:
                self.backbone = TemporalBackbone(
                    feat_dim=self.subset_feat_dim,
                    embed_dim=temporal_dim,
                    use_attention_pool=use_attention_pool,
                )
        else:
            self.backbone = None

        self.agent = self._create_agent()

        # Classifier head
        # Input: score(1) + confidence(1) + temporal_embed OR static_proj(32)
        if self.use_backbone:
            classifier_input_dim = 2 + temporal_dim
            self.static_proj = None
        else:
            static_proj_dim = self.static_feat_dim if self.agent_type in ('B', 'C') else self.subset_feat_dim
            self.static_proj = nn.Linear(static_proj_dim, 32)
            classifier_input_dim = 2 + 32

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _create_agent(self):
        agent_map = {
            'A': AgentA, 'B': AgentB, 'C': AgentC,
            'K': AgentK, '2D': Agent2D, 'TQ': AgentTQ, 'H': AgentH, 'M': AgentM,
        }
        if self.agent_type not in agent_map:
            raise ValueError(f"Unknown agent type: {self.agent_type}.")
        agent_class = agent_map[self.agent_type]

        if self.agent_type == 'A':
            return agent_class(input_dim=self.subset_feat_dim, temporal_dim=self.temporal_dim)
        elif self.agent_type == 'B':
            return agent_class(input_dim=self.subset_feat_dim, static_dim=self.static_feat_dim, hidden_dim=128, temporal_dim=self.temporal_dim)
        elif self.agent_type == 'C':
            return agent_class(input_dim=self.subset_feat_dim, static_dim=self.static_feat_dim, seq_len=20, embed_dim=96, temporal_dim=self.temporal_dim)
        elif self.agent_type == 'K':
            return agent_class(input_dim=self.subset_feat_dim, hidden_dim=512)
        elif self.agent_type == '2D':
            return agent_class(n_greeks=5, n_strikes=20, n_timesteps=20)
        elif self.agent_type == 'TQ':
            return agent_class(tq_feat_dim=self.subset_feat_dim, temporal_dim=self.temporal_dim)
        elif self.agent_type == 'H':
            return agent_class(input_dim=self.subset_feat_dim, temporal_dim=self.temporal_dim)
        elif self.agent_type == 'M':
            return agent_class(input_dim=self.subset_feat_dim, temporal_dim=self.temporal_dim)
        else:
            raise ValueError(f"Initialization not implemented for agent {self.agent_type}")

    def _select_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_feature_subset:
            return x.index_select(-1, self._feat_idx)
        return x

    def forward(self, sequences: torch.Tensor, chain_2d: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert not torch.isnan(sequences).any(), "NaN in input sequences — check data pipeline"

        # Apply feature subsetting
        seq_subset = self._select_features(sequences)

        # Static snapshot (last timestep)
        if self.agent_type in ('B', 'C'):
            static = sequences[:, -1, :].index_select(-1, self._static_idx)
        else:
            static = seq_subset[:, -1, :]

        # Temporal backbone
        if self.backbone is not None:
            temporal_embed = self.backbone(seq_subset)
        else:
            temporal_embed = None

        # Agent-specific forward
        if self.agent_type == '2D':
            score, confidence, signal = self.agent(
                static,
                temporal_embed if temporal_embed is not None else static,
                seq_subset,
                chain_2d=chain_2d,
            )
        elif self.agent_type in ('TQ', 'H'):
            score, confidence, signal = self.agent(static, temporal_embed, seq_subset)
        else:
            if temporal_embed is not None:
                score, confidence, signal = self.agent(static, temporal_embed, seq_subset)
            else:
                score, confidence, signal = self.agent(static, None, seq_subset)

        # Build classifier input — pass score/confidence RAW (no inverse-sigmoid)
        if temporal_embed is not None:
            features = torch.cat([score, confidence, temporal_embed], dim=1)
        elif self.static_proj is not None:
            proj_static = torch.relu(self.static_proj(static))
            features = torch.cat([score, confidence, proj_static], dim=1)
        else:
            features = torch.cat([score, confidence], dim=1)

        logits = self.classifier(features)
        return logits

    def count_parameters(self):
        backbone_params    = sum(p.numel() for p in self.backbone.parameters()) if self.backbone else 0
        agent_params       = sum(p.numel() for p in self.agent.parameters())
        classifier_params  = sum(p.numel() for p in self.classifier.parameters())
        total_params       = sum(p.numel() for p in self.parameters())
        return {
            'backbone':       backbone_params,
            'agent':          agent_params,
            'classifier':     classifier_params,
            'total':          total_params,
            'subset_feat_dim': self.subset_feat_dim,
            'use_backbone':   self.use_backbone,
        }


def create_independent_agent(config):
    agent_type = getattr(config, 'agent_type', config.get('agent_type', 'A'))
    return IndependentAgent(
        agent_type=agent_type,
        feat_dim=getattr(config, 'feat_dim', config.get('feat_dim', TOTAL_FEAT_DIM)),
        temporal_dim=getattr(config, 'temporal_dim', config.get('temporal_dim', 128)),
        dropout=getattr(config, 'dropout', config.get('dropout', 0.2)),
        num_classes=getattr(config, 'num_classes', config.get('num_classes', 5)),
        use_feature_subset=getattr(config, 'use_feature_subset', config.get('use_feature_subset', True)),
        use_mixer_backbone=getattr(config, 'use_mixer_backbone', config.get('use_mixer_backbone', False)),
    )


if __name__ == '__main__':
    print("Testing IndependentAgent v2 with Feature Subsetting...")
    for agent_type in ['A', 'B', 'K', 'C', 'TQ', 'H', 'M']:
        model = IndependentAgent(agent_type=agent_type, feat_dim=TOTAL_FEAT_DIM, use_feature_subset=True)
        dummy = torch.randn(16, 20, TOTAL_FEAT_DIM)
        with torch.no_grad():
            out = model(dummy)
        print(f"Agent {agent_type}: input={dummy.shape} output={out.shape} ✓")
    print("All tests passed!")
