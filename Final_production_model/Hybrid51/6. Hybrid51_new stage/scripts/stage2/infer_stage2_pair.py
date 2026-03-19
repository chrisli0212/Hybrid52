#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import sys

import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from hybrid51_utils import ArtifactPaths

PATHS = ArtifactPaths.default()

PAIR_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'TLT']
ALL_AGENTS = ['A', 'B', 'K', 'C', 'T', 'Q']

CROSS_SYMBOL_FEATURE_INDICES = [
    125, 126, 127, 128, 129, 130, 131,
    132, 133, 134,
    137, 138,
    150, 151, 152, 153,
    167, 168, 169,
    250, 251, 252,
    257, 258, 259,
    95, 96, 97, 98,
]
N_CROSS_FEATURES = len(CROSS_SYMBOL_FEATURE_INDICES)


def _load_symbol_split(symbol: str, horizon: int) -> dict:
    data_dir = PATHS.tier3_dir(symbol, horizon)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data not found: {data_dir}")

    return {
        'train_seq': np.load(data_dir / 'train_sequences.npy').astype(np.float32),
        'train_labels': np.load(data_dir / 'train_labels.npy').astype(np.int64),
        'val_seq': np.load(data_dir / 'val_sequences.npy').astype(np.float32),
        'val_labels': np.load(data_dir / 'val_labels.npy').astype(np.int64),
        'test_seq': np.load(data_dir / 'test_sequences.npy').astype(np.float32),
        'test_labels': np.load(data_dir / 'test_labels.npy').astype(np.int64),
    }


def _align_by_min_length(a: dict, b: dict) -> tuple[dict, dict]:
    out_a = dict(a)
    out_b = dict(b)
    for split in ['train', 'val', 'test']:
        seq_key = f"{split}_seq"
        lab_key = f"{split}_labels"
        n = min(len(out_a[seq_key]), len(out_b[seq_key]))
        out_a[seq_key] = out_a[seq_key][:n]
        out_b[seq_key] = out_b[seq_key][:n]
        out_a[lab_key] = out_a[lab_key][:n]
        out_b[lab_key] = out_b[lab_key][:n]
    return out_a, out_b


def _build_cross_features(seq_last_tgt: np.ndarray, seq_last_pair: np.ndarray) -> np.ndarray:
    feat_idx = np.array(CROSS_SYMBOL_FEATURE_INDICES, dtype=np.int64)
    cross = (seq_last_tgt[:, feat_idx] - seq_last_pair[:, feat_idx]).astype(np.float32)
    if cross.shape[1] != N_CROSS_FEATURES:
        raise RuntimeError(f"Expected {N_CROSS_FEATURES} cross features, got {cross.shape[1]}")
    return np.nan_to_num(cross, nan=0.0, posinf=0.0, neginf=0.0)


def _build_pair_features(
    logits_tgt: np.ndarray,
    logits_pair: np.ndarray,
    sequences_tgt: np.ndarray,
    sequences_pair: np.ndarray,
) -> np.ndarray:
    seq_last_tgt = sequences_tgt[:, -1, :]
    seq_last_pair = sequences_pair[:, -1, :]
    cross = _build_cross_features(seq_last_tgt, seq_last_pair)
    return np.concatenate([logits_tgt, logits_pair, cross], axis=1).astype(np.float32)


def _load_stage1_ckpt(symbol: str, agent: str, horizon: int) -> dict:
    ckpt_path = PATHS.stage1_ckpt(symbol, agent, horizon)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage1 ckpt not found: {ckpt_path}")
    import torch
    return torch.load(ckpt_path, map_location='cpu')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='SPXW')
    ap.add_argument('--pair', required=True)
    ap.add_argument('--horizon', type=int, default=15)
    ap.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    ap.add_argument('--stage2-ckpt', default=None)
    ap.add_argument('--stage1-batch', type=int, default=2048)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save-npz', default=None)
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise SystemExit(f"Torch import failed: {e}")

    sys.path.insert(0, str(ROOT))
    from hybrid51_models.independent_agent import IndependentAgent

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

    class _PairFusionModel(nn.Module):
        def __init__(self, n_agents: int = 6, n_cross_features: int = 29, dropout: float = 0.2):
            super().__init__()
            input_dim = n_agents * 2 + n_cross_features
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.GELU(),
                nn.LayerNorm(64),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    stage2_ckpt_path = Path(args.stage2_ckpt) if args.stage2_ckpt else PATHS.stage2_ckpt(args.target, args.pair, args.horizon)
    if not stage2_ckpt_path.exists():
        raise FileNotFoundError(f"Stage2 ckpt not found: {stage2_ckpt_path}")

    logger.info(f"Loading tier3 splits: target={args.target} pair={args.pair} horizon={args.horizon}")
    tgt = _load_symbol_split(args.target, args.horizon)
    pair = _load_symbol_split(args.pair, args.horizon)
    tgt, pair = _align_by_min_length(tgt, pair)

    feat_dim = int(tgt[f"{args.split}_seq"].shape[2])

    logger.info("Loading frozen Stage1 agents...")
    stage1_models_tgt = {}
    stage1_models_pair = {}
    stage1_thresholds = {}

    for agent in ALL_AGENTS:
        ckpt_t = _load_stage1_ckpt(args.target, agent, args.horizon)
        ckpt_p = _load_stage1_ckpt(args.pair, agent, args.horizon)
        use_subset_t = bool(ckpt_t.get('feature_subset', True))
        use_subset_p = bool(ckpt_p.get('feature_subset', True))

        use_attn_backbone_t = bool(ckpt_t.get('use_attention_backbone', False))
        use_attn_pool_t = bool(ckpt_t.get('use_attention_pool', False))
        use_attn_backbone_p = bool(ckpt_p.get('use_attention_backbone', False))
        use_attn_pool_p = bool(ckpt_p.get('use_attention_pool', False))

        m_t = _BinaryIndependentAgent(
            agent_type=agent,
            feat_dim=feat_dim,
            use_feature_subset=use_subset_t,
            use_attention_backbone=use_attn_backbone_t,
            use_attention_pool=use_attn_pool_t,
        ).to(device)
        m_p = _BinaryIndependentAgent(
            agent_type=agent,
            feat_dim=feat_dim,
            use_feature_subset=use_subset_p,
            use_attention_backbone=use_attn_backbone_p,
            use_attention_pool=use_attn_pool_p,
        ).to(device)
        m_t.load_state_dict(ckpt_t['model_state_dict'])
        m_p.load_state_dict(ckpt_p['model_state_dict'])
        m_t.eval()
        m_p.eval()
        stage1_models_tgt[agent] = m_t
        stage1_models_pair[agent] = m_p
        stage1_thresholds[agent] = float(ckpt_t.get('optimal_threshold', 0.5))

    @torch.no_grad()
    def predict_logits(model: nn.Module, sequences: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(sequences).to(device)
        outs = []
        for i in range(0, len(x), args.stage1_batch):
            outs.append(model(x[i:i + args.stage1_batch]).detach().cpu())
        return torch.cat(outs, dim=0).numpy().astype(np.float32)

    seq_t = tgt[f"{args.split}_seq"]
    seq_p = pair[f"{args.split}_seq"]
    y = tgt[f"{args.split}_labels"].astype(np.int64)

    logits_t_list = []
    logits_p_list = []
    for agent in ALL_AGENTS:
        logits_t_list.append(predict_logits(stage1_models_tgt[agent], seq_t).reshape(-1, 1))
        logits_p_list.append(predict_logits(stage1_models_pair[agent], seq_p).reshape(-1, 1))

    logits_t = np.concatenate(logits_t_list, axis=1)
    logits_p = np.concatenate(logits_p_list, axis=1)

    X = _build_pair_features(logits_t, logits_p, seq_t, seq_p)

    ckpt = torch.load(stage2_ckpt_path, map_location='cpu')
    model = _PairFusionModel(n_agents=len(ALL_AGENTS), n_cross_features=N_CROSS_FEATURES, dropout=0.2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X).to(device))).detach().cpu().numpy().astype(np.float32)

    thr = float(ckpt.get('optimal_threshold', 0.5))
    preds = (probs > thr).astype(np.int64)
    acc = float((preds == y).mean())
    tp = float(((preds == 1) & (y == 1)).sum())
    fp = float(((preds == 1) & (y == 0)).sum())
    fn = float(((preds == 0) & (y == 1)).sum())
    f1 = float((2 * tp) / max(1.0, (2 * tp + fp + fn)))

    logger.info(f"Stage2 inference split={args.split} n={len(y)} thr={thr:.2f} acc={acc:.4f} f1={f1:.4f}")

    if args.save_npz:
        out_p = Path(args.save_npz)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_p,
            probs=probs,
            labels=y,
            core_logits=logits_t.astype(np.float32),
            target=args.target,
            pair=args.pair,
            horizon=int(args.horizon),
            split=args.split,
            threshold=thr,
        )
        logger.info(f"Saved: {out_p}")


if __name__ == '__main__':
    main()
