#!/usr/bin/env python3
"""
Stage 1 — Train Agent VIX (warm-start) on regime classification.

This trains Agent VIX standalone as a 5-class regime classifier using
VIX 5-min features. The trained weights serve as a warm-start for
Stage 3 (RegimeGatedMetaModel), where the VIX agent is fine-tuned
end-to-end with the gating mechanism.

Loss: CrossEntropy + 0.3 × Regime-aware Focal Loss
Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
Training: 60 epochs, patience=12

Usage:
    python scripts/stage1/train_vix_agent.py
    python scripts/stage1/train_vix_agent.py --data-root /workspace/data/tier3_vix_v4/VIXW
    python scripts/stage1/train_vix_agent.py --epochs 80 --lr 1e-4
"""

import argparse
import json
import logging
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Defaults
# ============================================================================

DATA_ROOT = Path("/workspace/data/tier3_vix_v4/VIXW")
OUTPUT_ROOT = Path("/workspace/Hybrid51/6. Hybrid51_new stage/results/stage1_vix")

NUM_REGIMES = 5
REGIME_NAMES = ['CALM', 'NORMAL', 'ELEVATED', 'HIGH', 'EXTREME']
VIX_FEAT_DIM = 10


# ============================================================================
# Dataset
# ============================================================================

class VIXRegimeDataset(Dataset):
    """Dataset for VIX regime classification."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: (N, lookback, vix_feat_dim) — normalized VIX feature sequences
            labels: (N,) — regime class labels (0-4)
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance.
    Regime classes (especially EXTREME) are often rare.
    """
    class_counts = np.bincount(labels, minlength=NUM_REGIMES)
    class_weights = 1.0 / np.maximum(class_counts.astype(np.float64), 1.0)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )
    return sampler


# ============================================================================
# Focal Loss (for class imbalance)
# ============================================================================

class RegimeFocalLoss(nn.Module):
    """
    Multi-class Focal Loss for regime classification.
    Addresses class imbalance (EXTREME regime is rare).
    """

    def __init__(self, gamma: float = 2.0, alpha: np.ndarray = None, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer('alpha', torch.from_numpy(alpha).float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        num_classes = logits.size(-1)
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / num_classes
            targets_one_hot = torch.full_like(logits, smooth)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth)
        else:
            targets_one_hot = F.one_hot(targets, num_classes).float()

        # Focal modulation
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        focal_weight = (1 - probs) ** self.gamma
        loss = -(focal_weight * targets_one_hot * log_probs).sum(dim=-1)

        # Alpha weighting (per-class)
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            loss = loss * alpha_weight

        return loss.mean()


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        # Agent VIX uses last timestep as static, full seq as seq
        static = features[:, -1, :]  # (B, vix_feat_dim)
        temporal = static             # placeholder (no backbone)
        seq = features                 # (B, lookback, vix_feat_dim)

        # Forward
        score, confidence, regime_logits = model(static, temporal, seq)

        # Loss
        loss = criterion(regime_logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        all_preds.extend(regime_logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        static = features[:, -1, :]
        temporal = static
        seq = features

        score, confidence, regime_logits = model(static, temporal, seq)
        loss = criterion(regime_logits, labels)

        total_loss += loss.item() * features.size(0)
        probs = F.softmax(regime_logits, dim=-1)
        all_preds.extend(regime_logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ============================================================================
# Main Training
# ============================================================================

def train_vix_agent(
    data_root: Path = DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    patience: int = 12,
    device: str = 'auto',
):
    """
    Train Agent VIX on regime classification (Stage 1 warm-start).
    """
    t0 = time.time()

    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # ── Load data ────────────────────────────────────────────────────────
    logger.info(f"Loading data from {data_root}")
    train_features = np.load(data_root / "train_vix_features.npy")
    train_labels = np.load(data_root / "train_vix_labels.npy")
    val_features = np.load(data_root / "val_vix_features.npy")
    val_labels = np.load(data_root / "val_vix_labels.npy")
    test_features = np.load(data_root / "test_vix_features.npy")
    test_labels = np.load(data_root / "test_vix_labels.npy")

    logger.info(f"Train: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}")

    # Class distribution
    train_dist = np.bincount(train_labels, minlength=NUM_REGIMES)
    logger.info(f"Train regime distribution: {dict(zip(REGIME_NAMES, train_dist.tolist()))}")

    # ── Datasets and loaders ─────────────────────────────────────────────
    train_dataset = VIXRegimeDataset(train_features, train_labels)
    val_dataset = VIXRegimeDataset(val_features, val_labels)
    test_dataset = VIXRegimeDataset(test_features, test_labels)

    # Balanced sampling for training
    train_sampler = get_balanced_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    from hybrid51_models.agents.agent_vix import AgentVIX

    vix_feat_dim = train_features.shape[-1]
    model = AgentVIX(vix_feat_dim=vix_feat_dim).to(device)
    total_params = model.count_parameters()
    logger.info(f"Agent VIX parameters: {total_params:,}")

    # ── Loss ─────────────────────────────────────────────────────────────
    # Class-inverse alpha weights for focal loss
    alpha = 1.0 / np.maximum(train_dist.astype(np.float64), 1.0)
    alpha = alpha / alpha.sum()  # normalize
    criterion = RegimeFocalLoss(gamma=2.0, alpha=alpha, label_smoothing=0.05)

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ── Training loop ────────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    output_root.mkdir(parents=True, exist_ok=True)
    best_model_path = output_root / "vix_agent_best.pt"

    logger.info(f"Training for {epochs} epochs (patience={patience})...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} F1={train_f1:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} F1={val_f1:.4f} | "
            f"LR={current_lr:.2e}"
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'lr': current_lr,
        })

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  → New best! F1={val_f1:.4f} saved to {best_model_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # ── Evaluate on test set ─────────────────────────────────────────────
    logger.info("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    test_loss, test_acc, test_f1, test_preds, test_labels_np, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    logger.info(f"\nTest Results (best epoch {best_epoch}):")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  F1 (macro): {test_f1:.4f}")
    logger.info(f"  Loss: {test_loss:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels_np, test_preds, labels=range(NUM_REGIMES))
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  {REGIME_NAMES}")
    for i, row in enumerate(cm):
        logger.info(f"  {REGIME_NAMES[i]:>10}: {row.tolist()}")

    # Classification report
    report = classification_report(
        test_labels_np, test_preds,
        target_names=REGIME_NAMES,
        zero_division=0,
        output_dict=True,
    )

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        'model': 'AgentVIX',
        'vix_feat_dim': vix_feat_dim,
        'best_epoch': best_epoch,
        'total_epochs': len(history),
        'total_params': total_params,
        'test_accuracy': test_acc,
        'test_f1_macro': test_f1,
        'test_loss': test_loss,
        'per_class_report': report,
        'confusion_matrix': cm.tolist(),
        'regime_names': REGIME_NAMES,
        'train_distribution': dict(zip(REGIME_NAMES, train_dist.tolist())),
        'best_val_f1': best_val_f1,
        'training_time_sec': round(time.time() - t0, 1),
        'history': history,
        'hyperparameters': {
            'lr': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'patience': patience,
            'focal_gamma': 2.0,
            'label_smoothing': 0.05,
        },
    }

    results_path = output_root / "vix_agent_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Best model saved to {best_model_path}")
    logger.info(f"Total training time: {results['training_time_sec']}s")

    # Print summary
    print("\n" + "=" * 60)
    print("Agent VIX Training Summary (Stage 1 Warm-Start)")
    print("=" * 60)
    print(f"  Parameters:   {total_params:,}")
    print(f"  Best epoch:   {best_epoch}")
    print(f"  Val F1:       {best_val_f1:.4f}")
    print(f"  Test Acc:     {test_acc:.4f}")
    print(f"  Test F1:      {test_f1:.4f}")
    print(f"  Model:        {best_model_path}")
    print(f"  Results:      {results_path}")
    print("=" * 60)

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Agent VIX (Stage 1 warm-start)")
    parser.add_argument('--data-root', type=str, default=str(DATA_ROOT))
    parser.add_argument('--output-root', type=str, default=str(OUTPUT_ROOT))
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    train_vix_agent(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )
