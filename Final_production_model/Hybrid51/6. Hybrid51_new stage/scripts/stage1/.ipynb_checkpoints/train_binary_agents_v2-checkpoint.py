#!/usr/bin/env python3
"""
Stage 1 v2: Train individual agents on binary UP/DOWN target.

Key improvements over v1:
- BinaryFocalLoss with label smoothing (replaces BCEWithLogitsLoss)
- AdamW optimizer with weight_decay=0.01 (from Adam with 1e-5)
- CosineAnnealingWarmRestarts scheduler (from ReduceLROnPlateau)
- Gradient clipping at 1.0 (from 5.0)
- Feature normalization using train-split z-score stats
- Feature subsetting per agent for diversity
- Threshold optimization on val set after training
- Gradient accumulation for effective batch size 2048
- 80 epochs with patience=15 (from 25/7)
- Saves per-agent optimal threshold

Usage:
    python scripts/stage1/train_binary_agents_v2.py --symbol SPXW
    python scripts/stage1/train_binary_agents_v2.py --symbol SPXW --agents A B K --horizon 15
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
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from hybrid51_models.independent_agent import IndependentAgent
from config.feature_subsets import AGENT_FEATURE_SUBSETS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_ROOT = Path("/workspace/data/tier3_binary_v2")
OUTPUT_ROOT = Path("/workspace/Hybrid51/6. Hybrid51_new stage/results/stage1")
RETURN_SCALE = 10000.0

ALL_AGENTS = ['A', 'B', 'K', 'C', 'T', 'Q']
HORIZONS = [5, 15, 30]
ALL_SYMBOLS = ['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT']


# ============================================================================
# Data Augmentation
# ============================================================================

def mixup_sequences(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Mixup augmentation for sequences.
    Mixes pairs of samples: x_mix = lam*x_i + (1-lam)*x_j
    Returns mixed x and both label sets + lambda for soft loss.
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """Add Gaussian noise to sequence tensor during training."""
    return x + sigma * torch.randn_like(x)


# ============================================================================
# Focal Loss
# ============================================================================

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss with label smoothing.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Addresses class imbalance and focuses on hard examples.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.52, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits)
        # Binary cross entropy per sample
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal modulation
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


class SoftF1Loss(nn.Module):
    """Differentiable approximation of F1 loss."""
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        return 1 - f1


# ============================================================================
# Model Wrapper
# ============================================================================

class BinaryIndependentAgent(nn.Module):
    """
    Wraps IndependentAgent with binary output head.
    Supports both classification (BCE/focal) and regression (Huber).
    """
    def __init__(self, agent_type, feat_dim=325, temporal_dim=128, dropout=0.2,
                 mode='classifier', use_feature_subset=True,
                 use_attention_backbone=False, use_attention_pool=False):
        super().__init__()
        self.mode = mode
        self.agent_type = agent_type

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

        # Replace classifier head with binary/regression head
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

    def forward(self, sequences):
        """Returns (B,) raw logit/return prediction."""
        logits = self.base(sequences)
        return logits.squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Training
# ============================================================================

def normalize_data(sequences, norm_mean, norm_std):
    """Apply z-score normalization using training statistics."""
    # sequences: (N, seq_len, feat_dim)
    return (sequences - norm_mean) / norm_std


def train_one_model(model, train_loader, val_seq, val_targets, val_labels, device,
                    mode='classifier', epochs=80, lr=3e-4, patience=15,
                    accum_steps=4, f1_weight=0.3,
                    use_mixup=False, mixup_alpha=0.2,
                    noise_sigma=0.0, positive_class_prior=0.52):
    """Train a single model with focal loss, cosine schedule, gradient accumulation."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    if mode == 'classifier':
        focal_loss = BinaryFocalLoss(gamma=2.0, alpha=positive_class_prior, label_smoothing=0.05)
        soft_f1 = SoftF1Loss()
    else:
        criterion = nn.HuberLoss(delta=1.0)

    val_seq_t = torch.FloatTensor(val_seq).to(device)

    best_f1 = 0
    best_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        optimizer.zero_grad()
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device).float()

            # Gaussian noise augmentation
            if noise_sigma > 0:
                xb = add_gaussian_noise(xb, sigma=noise_sigma)

            if mode == 'classifier':
                if use_mixup:
                    xb_mix, ya, yb_mix, lam = mixup_sequences(xb, yb, alpha=mixup_alpha)
                    pred = model(xb_mix)
                    focal_part = mixup_loss(focal_loss, pred, ya, yb_mix, lam)
                    f1_part = mixup_loss(soft_f1, pred, ya, yb_mix, lam)
                    loss = focal_part + f1_weight * f1_part
                else:
                    pred = model(xb)
                    loss = focal_loss(pred, yb) + f1_weight * soft_f1(pred, yb)
            else:
                pred = model(xb)
                loss = criterion(pred, yb)

            # Gradient accumulation
            loss = loss / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            n_batches += 1

        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds_list = []
            bs = 2048
            for i in range(0, len(val_seq_t), bs):
                preds_list.append(model(val_seq_t[i:i+bs]))
            raw_output = torch.cat(preds_list).cpu().numpy()

        if mode == 'classifier':
            probs = 1 / (1 + np.exp(-raw_output))
            dir_preds = (probs > 0.5).astype(int)
        else:
            dir_preds = (raw_output > 0).astype(int)

        acc = accuracy_score(val_labels, dir_preds)
        f1 = f1_score(val_labels, dir_preds, average='binary')
        avg_loss = total_loss / n_batches
        current_lr = optimizer.param_groups[0]['lr']

        # Track best by F1 (primary) with acc as tiebreaker
        improved = False
        if f1 > best_f1 + 0.001 or (abs(f1 - best_f1) < 0.001 and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = True

        if not improved:
            patience_counter += 1

        marker = ' *' if improved else ''
        if epoch % 5 == 0 or improved or epoch == epochs - 1:
            logger.info(f"    Ep {epoch+1:3d}: loss={avg_loss:.4f} acc={acc:.4f} f1={f1:.4f} lr={current_lr:.6f}{marker}")

        if patience_counter >= patience:
            logger.info(f"    Early stop at epoch {epoch+1} (best F1={best_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc, best_f1


def optimize_threshold(model, val_seq, val_labels, device, mode='classifier'):
    """Sweep thresholds on validation set to maximize F1."""
    model.eval()
    val_seq_t = torch.FloatTensor(val_seq).to(device)

    with torch.no_grad():
        outputs = []
        bs = 2048
        for i in range(0, len(val_seq_t), bs):
            outputs.append(model(val_seq_t[i:i+bs]))
        raw_output = torch.cat(outputs).cpu().numpy()

    if mode == 'classifier':
        probs = 1 / (1 + np.exp(-raw_output))
    else:
        probs = raw_output

    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.30, 0.66, 0.01):
        if mode == 'classifier':
            preds = (probs > threshold).astype(int)
        else:
            preds = (probs > threshold).astype(int)

        f1 = f1_score(val_labels, preds, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logger.info(f"    Optimal threshold: {best_threshold:.2f} (val F1={best_f1:.4f})")
    return best_threshold, best_f1


def evaluate_model(model, test_seq, test_labels, test_returns, device,
                   mode='classifier', threshold=0.5):
    """Evaluate on test set with optimized threshold."""
    model.eval()
    test_seq_t = torch.FloatTensor(test_seq).to(device)

    with torch.no_grad():
        outputs = []
        bs = 2048
        for i in range(0, len(test_seq_t), bs):
            outputs.append(model(test_seq_t[i:i+bs]))
        raw_output = torch.cat(outputs).cpu().numpy()

    if mode == 'classifier':
        probs = 1 / (1 + np.exp(-raw_output))
        preds = (probs > threshold).astype(int)
        confidence = np.abs(probs - 0.5) * 2
        brier = brier_score_loss(test_labels, probs)
    else:
        preds = (raw_output > 0).astype(int)
        abs_mag = np.abs(raw_output)
        if abs_mag.max() > abs_mag.min():
            confidence = (abs_mag - abs_mag.min()) / (abs_mag.max() - abs_mag.min())
        else:
            confidence = np.zeros_like(abs_mag)
        brier = None

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average='binary')
    try:
        auc = roc_auc_score(test_labels, raw_output)
    except:
        auc = 0.5

    ic, _ = spearmanr(raw_output, test_returns)
    if np.isnan(ic):
        ic = 0.0

    # Confidence buckets
    conf_buckets = {}
    for thr in [0.0, 0.2, 0.4, 0.6, 0.8]:
        mask = confidence >= thr
        if mask.sum() > 50:
            conf_buckets[f"conf>={thr:.1f}"] = {
                'accuracy': round(float(accuracy_score(test_labels[mask], preds[mask])), 4),
                'f1': round(float(f1_score(test_labels[mask], preds[mask], average='binary')), 4),
                'coverage': round(float(mask.mean()), 4),
                'n': int(mask.sum()),
            }

    return {
        'accuracy': round(float(acc), 4),
        'f1': round(float(f1), 4),
        'auc': round(float(auc), 4),
        'ic': round(float(ic), 4),
        'brier': round(float(brier), 6) if brier is not None else None,
        'threshold': round(float(threshold), 3),
        'confidence_buckets': conf_buckets,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='SPXW', help='Single symbol (default: SPXW)')
    parser.add_argument('--symbols', nargs='+', default=None, help='Train over multiple symbols (e.g. SPXW SPY QQQ)')
    parser.add_argument('--all-symbols', action='store_true', help='Train over default symbol universe')
    parser.add_argument('--agents', nargs='+', default=ALL_AGENTS)
    parser.add_argument('--horizon', type=int, default=15, help='Single horizon in minutes')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--accum-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--f1-weight', type=float, default=0.3, help='Weight for soft-F1 loss')
    parser.add_argument('--no-feature-subset', action='store_true', help='Disable feature subsetting')
    parser.add_argument('--use-attention-backbone', action='store_true',
                        help='Use TemporalBackboneWithAttention (multi-head self-attention before convs)')
    parser.add_argument('--use-attention-pool', action='store_true',
                        help='Use learned AttentionPool instead of AdaptiveAvgPool1d')
    parser.add_argument('--use-mixup', action='store_true',
                        help='Enable Mixup augmentation during training')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Mixup alpha (Beta distribution parameter, default: 0.2)')
    parser.add_argument('--noise-sigma', type=float, default=0.0,
                        help='Gaussian noise sigma for augmentation (0=disabled, suggested: 0.02)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.all_symbols and args.symbols is not None:
        raise SystemExit("Use only one of --all-symbols or --symbols")

    if args.all_symbols:
        symbols = ALL_SYMBOLS
    elif args.symbols is not None and len(args.symbols) > 0:
        symbols = args.symbols
    else:
        symbols = [args.symbol]

    horizon = args.horizon

    for symbol in symbols:
        data_dir = DATA_ROOT / symbol / f"horizon_{horizon}min"
        if not data_dir.exists():
            logger.warning(f"[{symbol}] Data not found, skipping: {data_dir}")
            continue

        logger.info(f"{'='*70}")
        logger.info(f"STAGE 1 v2: {symbol} | Horizon={horizon}min | Feature Subsetting={'OFF' if args.no_feature_subset else 'ON'}")
        logger.info(f"{'='*70}")

        # Load data
        train_seq = np.load(data_dir / 'train_sequences.npy')
        train_labels = np.load(data_dir / 'train_labels.npy')
        train_returns = np.load(data_dir / 'train_returns.npy')
        val_seq = np.load(data_dir / 'val_sequences.npy')
        val_labels = np.load(data_dir / 'val_labels.npy')
        val_returns = np.load(data_dir / 'val_returns.npy')
        test_seq = np.load(data_dir / 'test_sequences.npy')
        test_labels = np.load(data_dir / 'test_labels.npy')
        test_returns = np.load(data_dir / 'test_returns.npy')

        feat_dim = train_seq.shape[2]

        # Load and apply normalization stats
        norm_mean_path = data_dir / 'norm_mean.npy'
        norm_std_path = data_dir / 'norm_std.npy'
        if norm_mean_path.exists() and norm_std_path.exists():
            norm_mean = np.load(norm_mean_path)
            norm_std = np.load(norm_std_path)
            logger.info("Applying z-score normalization from training stats")
            train_seq = normalize_data(train_seq, norm_mean, norm_std)
            val_seq = normalize_data(val_seq, norm_mean, norm_std)
            test_seq = normalize_data(test_seq, norm_mean, norm_std)
        else:
            logger.warning("No normalization stats found — training without z-score norm")

        train_returns_scaled = train_returns * RETURN_SCALE

        logger.info(f"Data: train={len(train_seq):,} val={len(val_seq):,} test={len(test_seq):,} feat_dim={feat_dim}")

        all_results = {}

        for agent_type in args.agents:
            logger.info(f"\n  --- Agent {agent_type} ---")
            subset_info = AGENT_FEATURE_SUBSETS.get(agent_type, {})
            logger.info(f"  Subset: {subset_info.get('name', 'N/A')} ({subset_info.get('feat_dim', feat_dim)} dims)")

            for mode in ['classifier']:  # Focus on classifier first
                logger.info(f"  [{agent_type}] Mode: {mode}")

                try:
                    model = BinaryIndependentAgent(
                        agent_type=agent_type,
                        feat_dim=feat_dim,
                        temporal_dim=128,
                        dropout=0.2,
                        mode=mode,
                        use_feature_subset=not args.no_feature_subset,
                        use_attention_backbone=args.use_attention_backbone,
                        use_attention_pool=args.use_attention_pool,
                    ).to(device)

                    n_params = model.count_parameters()
                    logger.info(f"    Params: {n_params:,}")

                    # Prepare targets
                    if mode == 'classifier':
                        targets = train_labels.astype(np.float32)
                        positive_class_prior = float(np.mean(targets))
                    else:
                        targets = train_returns_scaled
                        positive_class_prior = 0.52

                    train_ds = TensorDataset(
                        torch.FloatTensor(train_seq),
                        torch.FloatTensor(targets),
                    )
                    train_loader = DataLoader(
                        train_ds, batch_size=args.batch_size,
                        shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=True,
                    )

                    # Train
                    model, val_acc, val_f1 = train_one_model(
                        model, train_loader, val_seq,
                        val_returns if mode == 'regressor' else val_labels.astype(np.float32),
                        val_labels, device,
                        mode=mode, epochs=args.epochs, lr=args.lr,
                        patience=args.patience, accum_steps=args.accum_steps,
                        f1_weight=args.f1_weight,
                        use_mixup=args.use_mixup,
                        mixup_alpha=args.mixup_alpha,
                        noise_sigma=args.noise_sigma,
                        positive_class_prior=positive_class_prior,
                    )

                    # Optimize threshold
                    opt_threshold, opt_val_f1 = optimize_threshold(
                        model, val_seq, val_labels, device, mode=mode
                    )

                    # Evaluate on test with optimized threshold
                    test_metrics = evaluate_model(
                        model, test_seq, test_labels, test_returns, device,
                        mode=mode, threshold=opt_threshold,
                    )
                    logger.info(f"    Test: acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} "
                                f"auc={test_metrics['auc']:.4f} ic={test_metrics['ic']:.4f} "
                                f"brier={test_metrics['brier']:.6f} "
                                f"thr={test_metrics['threshold']:.3f}")

                    # Save checkpoint
                    ckpt_path = OUTPUT_ROOT / f'{symbol}_agent{agent_type}_{mode}_h{horizon}.pt'
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'test_metrics': test_metrics,
                        'agent_type': agent_type,
                        'horizon': horizon,
                        'mode': mode,
                        'n_params': n_params,
                        'optimal_threshold': opt_threshold,
                        'positive_class_prior': positive_class_prior,
                        'feature_subset': not args.no_feature_subset,
                        'subset_feat_dim': model.base.subset_feat_dim,
                        'use_attention_backbone': args.use_attention_backbone,
                        'use_attention_pool': args.use_attention_pool,
                        'use_mixup': args.use_mixup,
                        'noise_sigma': args.noise_sigma,
                    }, ckpt_path)

                    key = f"agent_{agent_type}_{mode}"
                    all_results[key] = test_metrics
                    all_results[key]['n_params'] = n_params

                except Exception as e:
                    logger.error(f"    FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[f"agent_{agent_type}_{mode}"] = {'error': str(e)}

                try:
                    del model
                except:
                    pass
                torch.cuda.empty_cache()

        # =========================================================================
        # Summary table
        # =========================================================================
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY: {symbol} | Horizon={horizon}min")
        logger.info(f"{'='*80}")
        logger.info(f"{'Agent':>6} {'Mode':>12} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'IC':>8} {'Thr':>6}")
        logger.info("-" * 70)

        for agent_type in args.agents:
            for mode in ['classifier']:
                rkey = f"agent_{agent_type}_{mode}"
                r = all_results.get(rkey, {})
                if 'error' in r:
                    logger.info(f"{agent_type:>6} {mode:>12} {'FAILED':>10}")
                elif 'accuracy' in r:
                    logger.info(f"{agent_type:>6} {mode:>12} "
                                f"{r['accuracy']:>10.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} "
                                f"{r['ic']:>8.4f} {r.get('threshold', 0.5):>6.3f}")

        logger.info(f"\nRandom baseline: accuracy=0.5000, F1=0.5000")

        # Save results
        result_path = OUTPUT_ROOT / f'{symbol}_h{horizon}_results.json'
        with open(result_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {result_path}")


if __name__ == '__main__':
    main()
