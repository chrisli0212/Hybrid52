"""
Phase 17 v2: Independent Agent Model with Feature Subsetting

Changes from v1:
- Feature subsetting: each agent sees only its designated feature subset
- Backbone feat_dim matches agent's subset dimension (not full 325)
- Agent T/Q still slice their specific features from the subset
- Support for disabling backbone (Agent K uses static MLP only)
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbone import TemporalBackbone, TemporalBackboneWithAttention
from .agents import AgentA, AgentB, AgentC, AgentK, Agent2D, AgentT, AgentQ

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.feature_subsets import AGENT_FEATURE_SUBSETS, TOTAL_FEAT_DIM, get_feature_indices


class IndependentAgent(nn.Module):
    """
    Single agent with backbone for independent decision making.
    v2: Supports feature subsetting per agent for diversity.

    Args:
        agent_type: Type of agent ('A', 'B', 'K', 'C', '2D', 'T', 'Q')
        feat_dim: Full input feature dimension (default: 325)
        temporal_dim: Temporal embedding dimension (default: 128)
        dropout: Dropout rate for classifier (default: 0.2)
        num_classes: Number of output classes (default: 5)
        use_feature_subset: If True, apply per-agent feature subsetting
        trade_feat_start: Start index for trade features in FULL vector (default: 270)
        trade_feat_end: End index for trade features in FULL vector (default: 307)
        quote_feat_start: Start index for quote features in FULL vector (default: 307)
        quote_feat_end: End index for quote features in FULL vector (default: 325)
    """

    def __init__(
        self,
        agent_type: str = 'A',
        feat_dim: int = 325,
        seq_len: int = 20,
        temporal_dim: int = 128,
        dropout: float = 0.2,
        num_classes: int = 5,
        use_feature_subset: bool = True,
        use_attention_backbone: bool = False,
        use_attention_pool: bool = False,
        trade_feat_start: int = 270,
        trade_feat_end: int = 307,
        quote_feat_start: int = 307,
        quote_feat_end: int = 325,
    ):
        super().__init__()

        self.agent_type = agent_type.upper()
        self.full_feat_dim = feat_dim
        self.seq_len = seq_len
        self.temporal_dim = temporal_dim
        self.num_classes = num_classes
        self.use_feature_subset = use_feature_subset
        self.use_attention_backbone = use_attention_backbone
        self.use_attention_pool = use_attention_pool
        self.trade_feat_start = trade_feat_start
        self.trade_feat_end = trade_feat_end
        self.quote_feat_start = quote_feat_start
        self.quote_feat_end = quote_feat_end

        # Feature subsetting
        if use_feature_subset and self.agent_type in AGENT_FEATURE_SUBSETS:
            subset_config = AGENT_FEATURE_SUBSETS[self.agent_type]
            self.feature_indices = get_feature_indices(self.agent_type)
            if feat_dim == TOTAL_FEAT_DIM * 2 and len(self.feature_indices) > 0:
                self.feature_indices = self.feature_indices + [idx + TOTAL_FEAT_DIM for idx in self.feature_indices]
            self.subset_feat_dim = len(self.feature_indices)
            self.use_backbone = subset_config['use_backbone']
            # Register feature indices as buffer (moves with model to GPU)
            self.register_buffer(
                '_feat_idx',
                torch.tensor(self.feature_indices, dtype=torch.long)
            )
        else:
            self.feature_indices = list(range(feat_dim))
            self.subset_feat_dim = feat_dim
            self.use_backbone = True
            self.register_buffer(
                '_feat_idx',
                torch.arange(feat_dim, dtype=torch.long)
            )

        # Determine backbone input dim
        backbone_feat_dim = self.subset_feat_dim

        # Temporal backbone
        if self.use_backbone:
            if use_attention_backbone:
                self.backbone = TemporalBackboneWithAttention(
                    feat_dim=backbone_feat_dim,
                    embed_dim=temporal_dim,
                    use_attention_pool=use_attention_pool,
                )
            else:
                self.backbone = TemporalBackbone(
                    feat_dim=backbone_feat_dim,
                    embed_dim=temporal_dim,
                    use_attention_pool=use_attention_pool,
                )
        else:
            self.backbone = None

        # Create agent based on type
        self.agent = self._create_agent()

        # Classifier head: score + confidence + temporal_embed
        if self.use_backbone:
            classifier_input_dim = 2 + temporal_dim
            self.static_proj = None
        else:
            # Project static features so they don't overwhelm the agent's signal
            self.static_proj = nn.Linear(self.subset_feat_dim, 32)
            classifier_input_dim = 2 + 32

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _create_agent(self):
        """Create agent instance based on agent_type."""
        agent_map = {
            'A': AgentA,
            'B': AgentB,
            'C': AgentC,
            'K': AgentK,
            '2D': Agent2D,
            'T': AgentT,
            'Q': AgentQ,
        }

        if self.agent_type not in agent_map:
            raise ValueError(
                f"Unknown agent type: {self.agent_type}. "
                f"Must be one of {list(agent_map.keys())}"
            )

        agent_class = agent_map[self.agent_type]

        if self.agent_type == 'A':
            return agent_class(
                input_dim=self.subset_feat_dim,
                temporal_dim=self.temporal_dim
            )

        elif self.agent_type == 'B':
            return agent_class(
                input_dim=self.subset_feat_dim,
                hidden_dim=128
            )

        elif self.agent_type == 'C':
            return agent_class(
                input_dim=self.subset_feat_dim,
                seq_len=self.seq_len,
                embed_dim=96
            )

        elif self.agent_type == 'K':
            return agent_class(
                input_dim=self.subset_feat_dim,
                hidden_dim=512
            )

        elif self.agent_type == '2D':
            return agent_class(
                n_greeks=5,
                n_strikes=20,
                n_timesteps=20
            )

        elif self.agent_type == 'T':
            # Use full subset dim if subsetting is active to prevent truncation bug
            trade_dim = self.subset_feat_dim if self.use_feature_subset else (self.trade_feat_end - self.trade_feat_start)
            return agent_class(
                trade_feat_dim=trade_dim,
                temporal_dim=self.temporal_dim if self.use_backbone else 0
            )

        elif self.agent_type == 'Q':
            quote_dim = self.subset_feat_dim if self.use_feature_subset else (self.quote_feat_end - self.quote_feat_start)
            return agent_class(
                quote_feat_dim=quote_dim,
                temporal_dim=self.temporal_dim if self.use_backbone else 0
            )

        else:
            raise ValueError(f"Initialization not implemented for agent {self.agent_type}")

    def _select_features(self, x: torch.Tensor) -> torch.Tensor:
        """Select feature subset from full feature vector."""
        if self.use_feature_subset:
            return x.index_select(-1, self._feat_idx)
        return x

    def forward(self, sequences: torch.Tensor, chain_2d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequences: (batch_size, seq_len, full_feat_dim)
            chain_2d: Optional (batch_size, n_greeks, n_strikes, seq_len) tensor for Agent2D

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = sequences.size(0)

        # Apply feature subsetting
        seq_subset = self._select_features(sequences)

        # Get static features (last timestep)
        static = seq_subset[:, -1, :]  # (B, subset_feat_dim)

        # Extract temporal features from backbone (if applicable)
        if self.backbone is not None:
            temporal_embed = self.backbone(seq_subset)  # (B, temporal_dim)
        else:
            temporal_embed = None

        # Agent-specific processing
        if self.agent_type == 'T':
            if self.use_feature_subset:
                agent_static = static
                agent_seq = seq_subset
            else:
                agent_static = sequences[:, -1, self.trade_feat_start:self.trade_feat_end]
                agent_seq = sequences[:, :, self.trade_feat_start:self.trade_feat_end]
            score, confidence, signal = self.agent(agent_static, temporal_embed, agent_seq)
        elif self.agent_type == 'Q':
            if self.use_feature_subset:
                agent_static = static
                agent_seq = seq_subset
            else:
                agent_static = sequences[:, -1, self.quote_feat_start:self.quote_feat_end]
                agent_seq = sequences[:, :, self.quote_feat_start:self.quote_feat_end]
            score, confidence, signal = self.agent(agent_static, temporal_embed, agent_seq)
        elif self.agent_type == '2D':
            # 2D agent primarily uses chain_2d, with flat sequence inputs kept for interface compatibility
            score, confidence, signal = self.agent(
                static,
                temporal_embed if temporal_embed is not None else static,
                seq_subset,
                chain_2d=chain_2d,
            )
        else:
            # A, B, C, K agents
            if temporal_embed is not None:
                score, confidence, signal = self.agent(static, temporal_embed, seq_subset)
            else:
                # B / K agents without backbone: pass seq/static; agent handles its own temporal processing
                score, confidence, signal = self.agent(static, static, seq_subset)

        # Build classifier input
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
        """Count trainable parameters by component."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters()) if self.backbone else 0
        agent_params = sum(p.numel() for p in self.agent.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'backbone': backbone_params,
            'agent': agent_params,
            'classifier': classifier_params,
            'total': total_params,
            'subset_feat_dim': self.subset_feat_dim,
            'use_backbone': self.use_backbone,
        }


def create_independent_agent(config):
    """
    Factory function to create IndependentAgent from config.
    """
    agent_type = getattr(config, 'agent_type', config.get('agent_type', 'A'))

    model = IndependentAgent(
        agent_type=agent_type,
        feat_dim=getattr(config, 'feat_dim', config.get('feat_dim', 325)),
        seq_len=getattr(config, 'seq_len', config.get('seq_len', 20)),
        temporal_dim=getattr(config, 'temporal_dim', config.get('temporal_dim', 128)),
        dropout=getattr(config, 'dropout', config.get('dropout', 0.2)),
        num_classes=getattr(config, 'num_classes', config.get('num_classes', 5)),
        use_feature_subset=getattr(config, 'use_feature_subset', config.get('use_feature_subset', True)),
    )

    return model


if __name__ == '__main__':
    print("Testing IndependentAgent v2 with Feature Subsetting...")

    for agent_type in ['A', 'B', 'K', 'C', 'T', 'Q']:
        print(f"\n{'='*60}")
        print(f"Testing Agent {agent_type}")
        print(f"{'='*60}")

        model = IndependentAgent(
            agent_type=agent_type,
            feat_dim=325,
            use_feature_subset=True,
        )

        params = model.count_parameters()
        print(f"  Subset feat dim: {params['subset_feat_dim']}")
        print(f"  Use backbone: {params['use_backbone']}")
        print(f"  Backbone params: {params['backbone']:,}")
        print(f"  Agent params: {params['agent']:,}")
        print(f"  Classifier params: {params['classifier']:,}")
        print(f"  Total params: {params['total']:,}")

        batch_size = 16
        seq_len = 20
        dummy_input = torch.randn(batch_size, seq_len, 325)

        with torch.no_grad():
            logits = model(dummy_input)
            print(f"  Input: {dummy_input.shape} → Output: {logits.shape}")

    print(f"\n{'='*60}")
    print("✓ All agent tests passed!")
