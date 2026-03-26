#!/usr/bin/env python3
"""Stage 1 inference helper (single agent).

Loads a trained Stage 1 checkpoint and runs inference on a provided numpy array
or a tier3 dataset split.

This is code-only scaffolding; it does not run unless you execute it.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from hybrid52_models.independent_agent import IndependentAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--np-seq', type=str, default=None, help='Path to .npy sequences (N,T,D)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    agent_type = ckpt.get('agent_type')
    if agent_type is None:
        raise SystemExit('Checkpoint missing agent_type')

    use_subset = bool(ckpt.get('feature_subset', True))
    use_attention_backbone = bool(ckpt.get('use_attention_backbone', False))
    use_attention_pool = bool(ckpt.get('use_attention_pool', False))

    class _BinaryIndependentAgent(nn.Module):
        def __init__(
            self,
            agent_type: str,
            feat_dim: int,
            temporal_dim: int = 128,
            dropout: float = 0.2,
            use_feature_subset: bool = True,
            use_attention_backbone: bool = False,
            use_attention_pool: bool = False,
        ):
            super().__init__()

            self.base = IndependentAgent(
                agent_type=agent_type,
                feat_dim=feat_dim,
                temporal_dim=temporal_dim,
                dropout=dropout,
                num_classes=5,
                use_feature_subset=use_feature_subset,
                use_attention_backbone=use_attention_backbone,
                use_attention_pool=use_attention_pool,
            )

            if self.base.use_backbone:
                classifier_input_dim = 2 + temporal_dim
            else:
                classifier_input_dim = 2 + self.base.subset_feat_dim

            self.base.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1),
            )

        def forward(self, sequences: torch.Tensor) -> torch.Tensor:
            logits = self.base(sequences)
            return logits.squeeze(-1)

    if args.np_seq is None:
        raise SystemExit('Provide --np-seq')

    seqs = np.load(args.np_seq).astype(np.float32)
    feat_dim = int(seqs.shape[2])

    model = _BinaryIndependentAgent(
        agent_type=agent_type,
        feat_dim=feat_dim,
        use_feature_subset=use_subset,
        use_attention_backbone=use_attention_backbone,
        use_attention_pool=use_attention_pool,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    x = torch.from_numpy(seqs).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()

    print('probs shape:', probs.shape)
    print('probs stats:', float(probs.min()), float(probs.mean()), float(probs.max()))


if __name__ == '__main__':
    main()
