#!/usr/bin/env python3
"""
Stage 2 Pre-computation: Chain Context

Runs frozen Stage 1 Agent 2D on SPXW chain_2d data and saves
(logit, prob) for train/val/test as a tiny .npz file.

This chain context (2 values per sample) is appended to each
standard agent's (A/B/C/K/T/Q) cross-symbol input, giving them
a read of "what does the SPXW option chain look like right now?"

Usage:
    python scripts/stage2/precompute_chain_context.py --symbol SPXW --horizon 15
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from hybrid51_utils import ArtifactPaths
from hybrid51_models.independent_agent import IndependentAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PATHS = ArtifactPaths.default()


class BinaryIndependentAgent(nn.Module):
    def __init__(self, agent_type, feat_dim=325, temporal_dim=128, dropout=0.2,
                 use_feature_subset=True, use_attention_backbone=False,
                 use_attention_pool=False, cls_input_dim=None):
        super().__init__()
        self.base = IndependentAgent(
            agent_type=agent_type, feat_dim=feat_dim, temporal_dim=temporal_dim,
            dropout=dropout, num_classes=5,
            use_feature_subset=use_feature_subset,
            use_attention_backbone=use_attention_backbone,
            use_attention_pool=use_attention_pool,
        )
        if cls_input_dim is None:
            cls_input_dim = (2 + temporal_dim) if self.base.use_backbone else (2 + 32)
        self.base.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, sequences, chain_2d=None):
        return self.base(sequences, chain_2d=chain_2d).squeeze(-1)


def _build_model_from_ckpt(ckpt, agent_type: str, device: torch.device):
    """
    Reconstruct BinaryIndependentAgent to exactly match the saved checkpoint,
    handling both old (no static_proj) and new (with static_proj) architectures.
    """
    state = ckpt['model_state_dict']
    feat_dim   = int(ckpt.get('feat_dim', 325))
    use_subset = bool(ckpt.get('feature_subset', True))
    use_attn_bb   = bool(ckpt.get('use_attention_backbone', False))
    use_attn_pool = bool(ckpt.get('use_attention_pool', False))

    cls_in_dim = int(state['base.classifier.0.weight'].shape[1])
    has_static_proj = 'base.static_proj.weight' in state

    model = BinaryIndependentAgent(
        agent_type=agent_type, feat_dim=feat_dim,
        use_feature_subset=use_subset,
        use_attention_backbone=use_attn_bb,
        use_attention_pool=use_attn_pool,
        cls_input_dim=cls_in_dim,
    ).to(device)

    if not has_static_proj and hasattr(model.base, 'static_proj'):
        del model.base.static_proj

    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def _infer_batched(model, sequences, chain_2d, device, batch_size=1024):
    """Run batched inference returning (logits, probs) as float32 numpy arrays."""
    n = len(sequences)
    logits_all = []
    for i in range(0, n, batch_size):
        seq_b = torch.from_numpy(np.array(sequences[i:i + batch_size])).float().to(device)
        c2d_b = torch.from_numpy(np.array(chain_2d[i:i + batch_size])).float().to(device)
        logits_b = model(seq_b, chain_2d=c2d_b).detach().cpu().numpy().astype(np.float32)
        logits_all.append(logits_b)
    logits = np.concatenate(logits_all, axis=0)
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return logits, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='SPXW')
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    PATHS.stage2_cross_results.mkdir(parents=True, exist_ok=True)

    ckpt_path = PATHS.stage1_2d_ckpt(args.symbol, args.horizon)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage1 2D checkpoint not found: {ckpt_path}")

    chain_dir = PATHS.tier3_chain_dir(args.symbol, args.horizon)
    if not chain_dir.exists():
        raise FileNotFoundError(f"Chain-only tier3 not found: {chain_dir}")

    logger.info(f"Loading Stage1 2D checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = _build_model_from_ckpt(ckpt, '2D', device)
    model.eval()
    cls_in = ckpt['model_state_dict']['base.classifier.0.weight'].shape[1]
    logger.info(f"  Model loaded, feat_dim={ckpt.get('feat_dim', 325)}, cls_input_dim={cls_in}")

    norm_mean = np.load(chain_dir / 'norm_mean.npy') if (chain_dir / 'norm_mean.npy').exists() else None
    norm_std = np.load(chain_dir / 'norm_std.npy') if (chain_dir / 'norm_std.npy').exists() else None

    out_path = PATHS.stage2_chain_context(args.symbol, args.horizon)
    result = {}

    for split in ['train', 'val', 'test']:
        seq = np.load(chain_dir / f'{split}_sequences.npy')
        chain = np.load(chain_dir / f'{split}_chain_2d.npy')
        labels = np.load(chain_dir / f'{split}_labels.npy')

        if norm_mean is not None and norm_std is not None:
            seq = (seq - norm_mean) / norm_std

        logger.info(f"  {split}: seq={seq.shape}, chain={chain.shape}")
        logits, probs = _infer_batched(model, seq, chain, device, args.batch_size)
        result[f'{split}_logits'] = logits
        result[f'{split}_probs'] = probs
        result[f'{split}_labels'] = labels.astype(np.int64)
        logger.info(f"    logits mean={logits.mean():.4f} std={logits.std():.4f}")

    np.savez(out_path, **result)
    logger.info(f"Saved chain context → {out_path}")
    logger.info("Done.")


if __name__ == '__main__':
    main()
