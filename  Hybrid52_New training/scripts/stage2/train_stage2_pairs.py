#!/usr/bin/env python3
"""
Stage 2 v2: Pair Fusion — combines Stage 1 agent predictions across symbol pairs.

Key changes from v1:
- Drops VIXW pair (51.78% high-conf accuracy, worse than random)
- Only uses SPY, QQQ, IWM, TLT pairs with SPXW
- Requires Stage 1 agent diversity (pairwise agreement < 75%) before proceeding
- Uses AdamW + cosine schedule (consistent with Stage 1 v2)

Usage:
    python scripts/stage2/train_stage2_pairs.py --target SPXW --horizon 30
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from hybrid51_utils import ArtifactPaths
from hybrid51_utils.artifacts import DEFAULT_TRAINING_HORIZON_MINUTES

PATHS = ArtifactPaths.default()
STAGE1_RESULTS = PATHS.stage1_results
OUTPUT_ROOT = PATHS.stage2_results

DATA_ROOT = PATHS.data_root

# Dropped VIXW per plan — harmful pair
PAIR_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'TLT']
ALL_AGENTS = ['A', 'B', 'K', 'C', 'T', 'Q']

# Cross-symbol feature indices (from feature_config_v2.py)
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


class PairFusionModel(nn.Module):
    """
    Fuses Stage 1 agent logits from target + correlated symbol.
    Input: 7 agent logits × 2 symbols + 29 cross-symbol features = 43 dims
    """
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.52, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class BinaryIndependentAgent(nn.Module):
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

        sys.path.insert(0, str(ROOT))
        from hybrid51_models.independent_agent import IndependentAgent

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


def _load_symbol_split(symbol: str, horizon: int):
    data_dir = DATA_ROOT / symbol / f"horizon_{horizon}min"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data not found: {data_dir}")

    train_seq = np.load(data_dir / 'train_sequences.npy').astype(np.float32)
    train_labels = np.load(data_dir / 'train_labels.npy').astype(np.int64)
    val_seq = np.load(data_dir / 'val_sequences.npy').astype(np.float32)
    val_labels = np.load(data_dir / 'val_labels.npy').astype(np.int64)
    test_seq = np.load(data_dir / 'test_sequences.npy').astype(np.float32)
    test_labels = np.load(data_dir / 'test_labels.npy').astype(np.int64)

    return {
        'train_seq': train_seq,
        'train_labels': train_labels,
        'val_seq': val_seq,
        'val_labels': val_labels,
        'test_seq': test_seq,
        'test_labels': test_labels,
        'feat_dim': int(train_seq.shape[2]),
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


def _load_stage1_ckpt(symbol: str, agent: str, horizon: int):
    ckpt_path = STAGE1_RESULTS / f'{symbol}_agent{agent}_classifier_h{horizon}.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage1 ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return ckpt


@torch.no_grad()
def _predict_stage1_logits_and_probs(model: nn.Module, sequences: np.ndarray, device: torch.device, batch_size: int):
    x = torch.from_numpy(sequences).to(device)
    logits_out = []
    for i in range(0, len(x), batch_size):
        logits_out.append(model(x[i:i + batch_size]).detach().cpu())
    logits = torch.cat(logits_out, dim=0).numpy().astype(np.float32)
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return logits, probs


def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + eps
    return (num / den).astype(np.float32)


def _build_cross_features(
    probs_tgt: np.ndarray,
    probs_pair: np.ndarray,
    seq_last_tgt: np.ndarray,
    seq_last_pair: np.ndarray,
) -> np.ndarray:
    feat_idx = np.array(CROSS_SYMBOL_FEATURE_INDICES, dtype=np.int64)
    cross = (seq_last_tgt[:, feat_idx] - seq_last_pair[:, feat_idx]).astype(np.float32)
    if cross.shape[1] != N_CROSS_FEATURES:
        raise RuntimeError(f"Expected {N_CROSS_FEATURES} cross features, got {cross.shape[1]}")
    return np.nan_to_num(cross, nan=0.0, posinf=0.0, neginf=0.0)


def _build_pair_features(
    logits_tgt: np.ndarray,
    logits_pair: np.ndarray,
    probs_tgt: np.ndarray,
    probs_pair: np.ndarray,
    sequences_tgt: np.ndarray,
    sequences_pair: np.ndarray,
) -> np.ndarray:
    seq_last_tgt = sequences_tgt[:, -1, :]
    seq_last_pair = sequences_pair[:, -1, :]
    cross = _build_cross_features(probs_tgt, probs_pair, seq_last_tgt, seq_last_pair)
    return np.concatenate([logits_tgt, logits_pair, cross], axis=1).astype(np.float32)


def _evaluate_binary(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    preds = (probs > threshold).astype(np.int64)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1': float(f1_score(labels, preds, average='binary')),
    }


def _sweep_threshold_for_f1(probs: np.ndarray, labels: np.ndarray):
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.30, 0.66, 0.01):
        f1 = f1_score(labels, (probs > thr).astype(np.int64), average='binary')
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1


def check_agent_diversity(agent_predictions: dict, threshold: float = 0.75):
    """
    Check pairwise agreement between agents.
    Returns True if diversity is sufficient (agreement < threshold).
    """
    agents = sorted(agent_predictions.keys())
    agreements = []

    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            p1 = agent_predictions[a1]
            p2 = agent_predictions[a2]
            agree = (p1 == p2).mean()
            agreements.append((a1, a2, agree))
            logger.info(f"  Agreement {a1} vs {a2}: {agree:.1%}")

    avg_agreement = np.mean([a[2] for a in agreements])
    logger.info(f"  Average pairwise agreement: {avg_agreement:.1%} (target: <{threshold:.0%})")

    if avg_agreement > threshold:
        logger.warning(f"  ⚠️ Agent diversity insufficient! Agreement={avg_agreement:.1%} > {threshold:.0%}")
        logger.warning(f"  Stage 2 may not add value. Consider re-training Stage 1 with more diversity.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='SPXW', help='Target symbol')
    parser.add_argument('--pairs', nargs='+', default=None, help='Pair symbols (default: built-in list)')
    parser.add_argument('--all-pairs', action='store_true', help='Train all default pair symbols')
    parser.add_argument('--horizon', type=int, default=DEFAULT_TRAINING_HORIZON_MINUTES)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage1-batch', type=int, default=2048, help='Batch size for frozen Stage1 inference')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*70}")
    logger.info(f"STAGE 2 v2: Pair Fusion | Target={args.target} | Horizon={args.horizon}min")
    logger.info(f"Pair symbols: {PAIR_SYMBOLS} (VIXW dropped)")
    logger.info(f"{'='*70}")

    if args.all_pairs and args.pairs is not None:
        raise SystemExit("Use only one of --all-pairs or --pairs")

    if args.all_pairs:
        pairs = PAIR_SYMBOLS
    elif args.pairs is not None and len(args.pairs) > 0:
        pairs = args.pairs
    else:
        pairs = PAIR_SYMBOLS

    for pair_symbol in pairs:
        logger.info(f"\n{'='*70}")
        logger.info(f"STAGE 2: {args.target} + {pair_symbol} | Horizon={args.horizon}min")
        logger.info(f"{'='*70}")

        tgt = _load_symbol_split(args.target, args.horizon)
        pair = _load_symbol_split(pair_symbol, args.horizon)
        if tgt['feat_dim'] != pair['feat_dim']:
            raise RuntimeError(f"Feature dim mismatch: target={tgt['feat_dim']} pair={pair['feat_dim']}")

        tgt, pair = _align_by_min_length(tgt, pair)
        feat_dim = tgt['feat_dim']

        # Load Stage 1 agents (frozen)
        stage1_models_tgt = {}
        stage1_models_pair = {}
        stage1_thresholds = {}
        for agent in ALL_AGENTS:
            ckpt_t = _load_stage1_ckpt(args.target, agent, args.horizon)
            ckpt_p = _load_stage1_ckpt(pair_symbol, agent, args.horizon)
            use_subset_t = bool(ckpt_t.get('feature_subset', True))
            use_subset_p = bool(ckpt_p.get('feature_subset', True))

            use_attn_backbone_t = bool(ckpt_t.get('use_attention_backbone', False))
            use_attn_pool_t = bool(ckpt_t.get('use_attention_pool', False))
            use_attn_backbone_p = bool(ckpt_p.get('use_attention_backbone', False))
            use_attn_pool_p = bool(ckpt_p.get('use_attention_pool', False))

            m_t = BinaryIndependentAgent(
                agent_type=agent,
                feat_dim=feat_dim,
                use_feature_subset=use_subset_t,
                use_attention_backbone=use_attn_backbone_t,
                use_attention_pool=use_attn_pool_t,
            ).to(device)
            m_p = BinaryIndependentAgent(
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

        # Stage1 diversity check on target val split
        logger.info("\nStep 1: Checking Stage 1 agent diversity (target val split)...")
        agent_val_preds = {}
        for agent in ALL_AGENTS:
            _, p_val = _predict_stage1_logits_and_probs(stage1_models_tgt[agent], tgt['val_seq'], device, args.stage1_batch)
            agent_val_preds[agent] = (p_val > stage1_thresholds[agent]).astype(np.int64)
        check_agent_diversity(agent_val_preds, threshold=0.75)

        # Build Stage2 design matrix using frozen Stage1 predictions
        logger.info("\nStep 2: Building frozen Stage1 predictions...")
        def build_split(split: str):
            seq_t = tgt[f"{split}_seq"]
            seq_p = pair[f"{split}_seq"]
            y = tgt[f"{split}_labels"].astype(np.float32)

            logits_t_list, probs_t_list = [], []
            logits_p_list, probs_p_list = [], []
            for agent in ALL_AGENTS:
                lt, pt = _predict_stage1_logits_and_probs(stage1_models_tgt[agent], seq_t, device, args.stage1_batch)
                lp, pp = _predict_stage1_logits_and_probs(stage1_models_pair[agent], seq_p, device, args.stage1_batch)
                logits_t_list.append(lt.reshape(-1, 1))
                probs_t_list.append(pt.reshape(-1, 1))
                logits_p_list.append(lp.reshape(-1, 1))
                probs_p_list.append(pp.reshape(-1, 1))

            logits_t = np.concatenate(logits_t_list, axis=1)
            probs_t = np.concatenate(probs_t_list, axis=1)
            logits_p = np.concatenate(logits_p_list, axis=1)
            probs_p = np.concatenate(probs_p_list, axis=1)

            X = _build_pair_features(logits_t, logits_p, probs_t, probs_p, seq_t, seq_p)
            return X, y, probs_t, probs_p

        X_train, y_train, _, _ = build_split('train')
        X_val, y_val, _, _ = build_split('val')
        X_test, y_test, _, _ = build_split('test')

        logger.info(f"Pair features: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

        model = PairFusionModel(n_agents=len(ALL_AGENTS), n_cross_features=N_CROSS_FEATURES, dropout=0.2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
        train_up_frac = float(np.mean(y_train))
        criterion = BinaryFocalLoss(gamma=2.0, alpha=train_up_frac, label_smoothing=0.05)

        X_val_t = torch.from_numpy(X_val).to(device)
        best_state = None
        best_val_score = -1.0

        logger.info("\nStep 3: Training Stage 2 pair fusion...")
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.item())
            sched.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t).detach().cpu().numpy().astype(np.float32)
                val_probs = (1.0 / (1.0 + np.exp(-val_logits))).astype(np.float32)
                val_metrics = _evaluate_binary(val_probs, y_val.astype(np.int64), threshold=0.5)

            val_score = float(val_metrics['accuracy']) + float(val_metrics['f1'])
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0 or epoch == args.epochs - 1:
                logger.info(
                    f"  Ep {epoch+1:03d}: loss={total_loss/ max(1,len(train_loader)):.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final eval + save artifacts
        model.eval()
        with torch.no_grad():
            test_logits = model(torch.from_numpy(X_test).to(device)).detach().cpu().numpy().astype(np.float32)
        test_probs = (1.0 / (1.0 + np.exp(-test_logits))).astype(np.float32)

        val_logits = model(torch.from_numpy(X_val).to(device)).detach().cpu().numpy().astype(np.float32)
        val_probs = (1.0 / (1.0 + np.exp(-val_logits))).astype(np.float32)
        opt_thr, opt_val_f1 = _sweep_threshold_for_f1(val_probs, y_val.astype(np.int64))
        test_metrics = _evaluate_binary(test_probs, y_test.astype(np.int64), threshold=opt_thr)

        logger.info(f"\nBest val_f1={opt_val_f1:.4f} @thr={opt_thr:.2f}")
        logger.info(f"Test: acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        ckpt_path = OUTPUT_ROOT / f'{args.target}_{pair_symbol}_h{args.horizon}_pair_fusion.pt'
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'target': args.target,
                'pair': pair_symbol,
                'horizon': args.horizon,
                'agents': ALL_AGENTS,
                'optimal_threshold': opt_thr,
                'val_f1': float(opt_val_f1),
                'test_metrics': test_metrics,
            },
            ckpt_path,
        )

        probs_path = OUTPUT_ROOT / f'{args.target}_{pair_symbol}_h{args.horizon}_pair_probs.npz'
        np.savez(
            probs_path,
            val_probs=val_probs,
            val_labels=y_val.astype(np.int64),
            test_probs=test_probs,
            test_labels=y_test.astype(np.int64),
            val_core_logits=X_val[:, :len(ALL_AGENTS)].astype(np.float32),
            test_core_logits=X_test[:, :len(ALL_AGENTS)].astype(np.float32),
        )

        logger.info(f"Saved Stage2 ckpt: {ckpt_path}")
        logger.info(f"Saved Stage2 probs: {probs_path}")


if __name__ == '__main__':
    main()
