#!/usr/bin/env python3
"""Stage 1 inference helper (single agent).

Loads a trained Stage 1 checkpoint and runs inference on a provided numpy array
or a tier3 dataset split.

Fixes applied 2026-03-27:
- Fix 1: z-score norm loaded from checkpoint and applied before model forward
- Fix 2: Platt scaler reconstructed from checkpoint and applied to raw logits
- Fix 3: invert_signal flag read from checkpoint and applied to final probs
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from hybrid55_models.independent_agent import IndependentAgent


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


def _rebuild_platt(ckpt: dict):
    """Reconstruct Platt LogisticRegression from checkpoint dict. Returns None if not saved."""
    if 'platt_scaler_coef' not in ckpt or 'platt_scaler_intercept' not in ckpt:
        return None
    platt = LogisticRegression()
    platt.coef_      = np.asarray(ckpt['platt_scaler_coef'],      dtype=np.float64)
    platt.intercept_ = np.asarray(ckpt['platt_scaler_intercept'], dtype=np.float64).ravel()
    platt.classes_   = np.array([0, 1], dtype=np.int64)
    return platt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',   type=str, required=True,
                        help='Path to .pt checkpoint')
    parser.add_argument('--np-seq', type=str, default=None,
                        help='Path to .npy sequences (N, T, D) — raw, un-normalised')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint ──────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    agent_type           = ckpt.get('agent_type')
    if agent_type is None:
        raise SystemExit('Checkpoint missing agent_type')

    use_subset           = bool(ckpt.get('feature_subset',         True))
    use_attention_bb     = bool(ckpt.get('use_attention_backbone', False))
    use_attention_pool   = bool(ckpt.get('use_attention_pool',     False))
    invert_signal        = bool(ckpt.get('invert_signal',          False))  # Fix 3

    # ── Fix 1: load norm stats ───────────────────────────────
    norm_mean_raw = ckpt.get('norm_mean', None)
    norm_std_raw  = ckpt.get('norm_std',  None)
    if norm_mean_raw is not None:
        norm_mean_t = torch.FloatTensor(norm_mean_raw).to(device)
        norm_std_t  = torch.FloatTensor(norm_std_raw).to(device)
        print(f'Norm stats loaded from checkpoint (dim={len(norm_mean_raw)})')
    else:
        norm_mean_t = norm_std_t = None
        print('WARNING: No norm_mean/norm_std in checkpoint — inference on raw sequences')

    # ── Fix 2: rebuild Platt scaler ──────────────────────────
    platt = _rebuild_platt(ckpt)
    if platt is not None:
        print('Platt scaler loaded from checkpoint')
    else:
        print('WARNING: No Platt scaler in checkpoint — using raw sigmoid probs')

    if args.np_seq is None:
        raise SystemExit('Provide --np-seq path to .npy sequences')

    seqs     = np.load(args.np_seq).astype(np.float32)
    feat_dim = int(seqs.shape[2])

    # ── Build model ──────────────────────────────────────────
    model = _BinaryIndependentAgent(
        agent_type=agent_type,
        feat_dim=feat_dim,
        use_feature_subset=use_subset,
        use_attention_backbone=use_attention_bb,
        use_attention_pool=use_attention_pool,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # ── Inference ────────────────────────────────────────────
    all_logits = []
    batch_size = 2048
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            xb = torch.from_numpy(seqs[i:i + batch_size]).to(device)
            # Fix 1: apply z-score norm
            if norm_mean_t is not None:
                xb = (xb - norm_mean_t) / norm_std_t
            all_logits.append(model(xb))

    raw = torch.cat(all_logits).cpu().numpy()

    # Fix 2: apply Platt scaler
    if platt is not None:
        probs = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    else:
        probs = 1.0 / (1.0 + np.exp(-raw))

    # Fix 3: apply invert_signal
    if invert_signal:
        probs = 1.0 - probs
        print('invert_signal=True — probabilities flipped')

    print(f'probs shape : {probs.shape}')
    print(f'probs stats : min={float(probs.min()):.4f}  '
          f'mean={float(probs.mean()):.4f}  '
          f'max={float(probs.max()):.4f}')


if __name__ == '__main__':
    main()
