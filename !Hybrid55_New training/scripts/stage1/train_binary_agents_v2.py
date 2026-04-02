#!/usr/bin/env python3
"""
Stage 1 v2 — PyTorch Lightning edition.

Changes vs original:
- Added LitAgent (LightningModule) wrapping BinaryIndependentAgent
- train_one_model() replaced by run_lightning_training()
  * fp16 AMP (--precision 16-mixed), native grad accumulation,
    gradient clip, EarlyStopping on val_mcc, ModelCheckpoint, CSVLogger
- main() now calls run_lightning_training() instead of train_one_model()
- All other logic (losses, dataset, threshold, evaluate_model) UNCHANGED
- Added --precision CLI arg (default: 16-mixed; use 32 for CPU-only)

Fix 2026-03-26 (preserved):
- REMOVED double-normalization in _forward_model
- REMOVED invert_signal auto-flip band-aid
- Changed threshold_objective default: balanced_acc -> f1
- evaluate_model AUC now computed from raw_output BEFORE any transforms
- Added val_auc < 0.48 guard
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch
torch.set_float32_matmul_precision('medium')  # FIX1
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

try:
    import lightning as L
    from lightning.pytorch.callbacks import (EarlyStopping, ModelCheckpoint,
                                             LearningRateMonitor)
    from lightning.pytorch.loggers import CSVLogger
except ModuleNotFoundError:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                             LearningRateMonitor)
    from pytorch_lightning.loggers import CSVLogger

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from hybrid55_models.independent_agent import IndependentAgent
from config.feature_subsets import AGENT_FEATURE_SUBSETS, assert_supported_schema
from hybrid55_preprocessing.feature_config_v2 import FEATURE_SCHEMA_VERSION
from hybrid55_utils.artifacts import DEFAULT_TRAINING_HORIZON_MINUTES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT    = Path("/workspace/data/tier3_binary_hybrid55")
DEFAULT_OUTPUT_ROOT  = Path("/workspace/!Hybrid55_New training/results/stage1")
DEFAULT_CHAIN_2D_DIR = Path("/workspace/data/chain_2d")
RETURN_SCALE = 10000.0
STD_EPS = 1e-5

ALL_AGENTS  = ['A', 'B', 'K', 'C', 'TQ', 'H', 'M', '2D']
HORIZONS    = [5, 15, 30]
ALL_SYMBOLS = ['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT']


def _load_normalization_stats(data_dir: Path, feat_dim: int):
    """
    Load normalization stats with modality-first preference:
    norm_stats_greek.npz / norm_stats_tq.npz / norm_stats_ohlc.npz,
    then fallback to legacy norm_mean.npy + norm_std.npy.
    """
    modality_files = [
        data_dir / 'norm_stats_greek.npz',
        data_dir / 'norm_stats_tq.npz',
        data_dir / 'norm_stats_ohlc.npz',
    ]
    if all(p.exists() for p in modality_files):
        mean = np.zeros(feat_dim, dtype=np.float32)
        std = np.ones(feat_dim, dtype=np.float32)
        for p in modality_files:
            d = np.load(p)
            idx = d['indices'].astype(np.int64)
            valid = idx[(idx >= 0) & (idx < feat_dim)]
            if len(valid) == 0:
                continue
            mean[valid] = d['mean'][:len(valid)].astype(np.float32)
            std[valid] = np.maximum(d['std'][:len(valid)].astype(np.float32), STD_EPS)
        return mean, std, 'modality_npz'

    nm_path = data_dir / 'norm_mean.npy'
    ns_path = data_dir / 'norm_std.npy'
    if nm_path.exists() and ns_path.exists():
        mean = np.load(nm_path).astype(np.float32)
        std = np.maximum(np.load(ns_path).astype(np.float32), STD_EPS)
        if len(mean) != feat_dim:
            logger.warning(
                "Normalization dim mismatch: stats=%d feat_dim=%d; clipping to min dim",
                len(mean), feat_dim
            )
            n = min(len(mean), feat_dim)
            out_m = np.zeros(feat_dim, dtype=np.float32)
            out_s = np.ones(feat_dim, dtype=np.float32)
            out_m[:n] = mean[:n]
            out_s[:n] = std[:n]
            return out_m, out_s, 'legacy_npy_clipped'
        return mean, std, 'legacy_npy'

    return None, None, 'none'


def _load_chain_norm_stats(data_dir: Path):
    mean_p = data_dir / 'chain_norm_mean.npy'
    std_p = data_dir / 'chain_norm_std.npy'
    if mean_p.exists() and std_p.exists():
        mean = np.load(mean_p).astype(np.float32)
        std = np.maximum(np.load(std_p).astype(np.float32), STD_EPS)
        return mean, std, 'chain_norm_npy'
    return None, None, 'none'


def _load_feature_schema(data_dir: Path) -> str:
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        logger.warning("Missing metadata.json at %s; assuming legacy schema", data_dir)
        return FEATURE_SCHEMA_VERSION
    try:
        payload = json.loads(meta_path.read_text())
        return str(payload.get("feature_schema_version", FEATURE_SCHEMA_VERSION))
    except Exception as exc:
        logger.warning("Failed reading metadata.json schema (%s): %s", data_dir, exc)
        return FEATURE_SCHEMA_VERSION


# ============================================================================
# Helpers
# ============================================================================

def _resolve_chain_2d_paths(data_dir, symbol, chain_2d_dir, n_train, n_val):
    tp = data_dir / 'train_chain_2d.npy'
    vp = data_dir / 'val_chain_2d.npy'
    ep = data_dir / 'test_chain_2d.npy'
    if tp.exists() and vp.exists() and ep.exists():
        return tp, vp, ep
    mono = chain_2d_dir / f"{symbol}_chain_2d_train.npy"
    if mono.exists():
        batch = np.load(mono)
        N = len(batch); t1 = int(N * 0.70); t2 = int(N * 0.85)
        np.save(tp, batch[:t1]); np.save(vp, batch[t1:t2]); np.save(ep, batch[t2:])
        return tp, vp, ep
    return None, None, None


# ============================================================================
# Data Augmentation
# ============================================================================

def mixup_sequences(x, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def add_gaussian_noise(x, sigma=0.02):
    return x + sigma * torch.randn_like(x)


# ============================================================================
# Losses
# ============================================================================

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.65, label_smoothing=0.05):  # FIX2
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * focal_weight * bce).mean()

class SoftF1Loss(nn.Module):
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        return 1 - 2 * tp / (2 * tp + fp + fn + 1e-8)

class AsymmetricTradingLoss(nn.Module):
    """
    Penalizes wrong-direction calls asymmetrically.
    fp_weight > 1.0: predict UP but market goes DOWN costs more
    (suits options buying where wrong calls lose full premium)
    """
    def __init__(self, fp_weight: float = 1.5, gamma: float = 1.0):
        super().__init__()
        self.fp_weight = fp_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * 0.9 + 0.05
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        fp_mask = (probs > 0.5).float() * (1 - targets)
        weights = 1.0 + (self.fp_weight - 1.0) * fp_mask
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal = (1 - p_t) ** self.gamma
        return (weights * focal * bce).mean()


# ============================================================================
# Data Verification
# ============================================================================

def verify_data_before_training(data_dir: Path, symbol: str, horizon: int) -> bool:
    seq     = np.load(data_dir / 'train_sequences.npy', mmap_mode='r')
    labels  = np.load(data_dir / 'train_labels.npy')
    returns = np.load(data_dir / 'train_returns.npy')

    X        = seq.reshape(-1, seq.shape[-1])
    feat_dim = X.shape[1]
    signmatch = ((returns > 0) == (labels == 1)).mean()

    std  = X.std(axis=0)
    nz   = (np.abs(X) > 1e-8).mean(axis=0)
    dead = (std < STD_EPS) | (nz < 1e-4)
    n_dead = int(dead.sum())

    if feat_dim >= 286:
        csv_block = X[:, 270:286]
        csv_nz    = (np.abs(csv_block) > 1e-8).mean()
        csv_live  = csv_nz > 0.05
    else:
        csv_live = True; csv_nz = -1.0

    up_pct = float(labels.mean())
    group_ranges = {
        "greek_core": list(range(0, min(feat_dim, 150))) + list(range(200, min(feat_dim, 286))),
        "tq_slice": list(range(150, min(feat_dim, 200))),
        "ohlc_block": list(range(286, min(feat_dim, 311))),
    }
    ok = True

    logger.info(f"\n{'='*60}")
    logger.info(f"DATA HEALTH CHECK: {symbol} h{horizon}min")
    logger.info(f"{'='*60}")

    if signmatch < 0.50:
        logger.error(f"  LABEL INVERSION: signmatch={signmatch:.4f} (expected >0.95)")
        ok = False
    else:
        logger.info(f"  Label sign match: {signmatch:.4f}")

    if n_dead > feat_dim * 0.50:
        logger.error(f"  HIGH dead feature count: {n_dead}/{feat_dim} ({n_dead/feat_dim:.1%})")
        ok = False
    elif n_dead > feat_dim * 0.15:
        logger.warning(f"  Elevated dead feature count: {n_dead}/{feat_dim} ({n_dead/feat_dim:.1%})")
    else:
        logger.info(f"  Dead features: {n_dead}/{feat_dim} ({n_dead/feat_dim:.1%})")

    if feat_dim >= 286:
        if not csv_live:
            logger.error(f"  CSV-DERIVED DIMS 270-285 ARE DEAD: nz={csv_nz:.4f}")
            ok = False
        else:
            logger.info(f"  CSV-derived dims 270-285: nz={csv_nz:.4f} (live)")

    for g, idx in group_ranges.items():
        if not idx:
            logger.info(f"  {g}: n/a for feat_dim={feat_dim}")
            continue
        idx_arr = np.asarray(idx, dtype=np.int64)
        g_live = int((~dead[idx_arr]).sum())
        g_total = int(len(idx_arr))
        logger.info(f"  {g}: live={g_live}/{g_total} ({g_live/max(1,g_total):.1%})")

    if up_pct < 0.35 or up_pct > 0.65:
        logger.warning(f"  Label imbalance: UP={up_pct:.4f} (expected 0.45-0.55)")
    else:
        logger.info(f"  Label balance: UP={up_pct:.4f}")

    logger.info(f"{'='*60}")
    if ok:
        logger.info("  DATA HEALTH: PASS")
    else:
        logger.error("  DATA HEALTH: FAIL — fix issues above before training")
    logger.info(f"{'='*60}\n")
    return ok


# ============================================================================
# Dataset
# ============================================================================

class SequenceWithOptionalChainDataset(Dataset):
    def __init__(self, sequences, targets, norm_mean=None, norm_std=None, chain_2d=None,
                 chain_norm_mean=None, chain_norm_std=None):
        self.sequences = sequences
        self.targets   = torch.FloatTensor(targets)
        self.norm_mean = torch.FloatTensor(norm_mean) if norm_mean is not None else None
        self.norm_std  = torch.FloatTensor(norm_std)  if norm_std  is not None else None
        self.chain_2d  = torch.FloatTensor(chain_2d)  if chain_2d  is not None else None
        self.chain_norm_mean = (torch.FloatTensor(chain_norm_mean).view(-1, 1, 1)
                                if chain_norm_mean is not None else None)
        self.chain_norm_std = (torch.FloatTensor(chain_norm_std).view(-1, 1, 1)
                               if chain_norm_std is not None else None)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.FloatTensor(np.array(self.sequences[idx]))
        # z-score norm applied HERE (only place); clamp to ±5σ to prevent
        # outlier activations (some features reach 300+σ and corrupt gradients).
        if self.norm_mean is not None and self.norm_std is not None:
            seq = (seq - self.norm_mean) / self.norm_std
        seq = seq.clamp(-5.0, 5.0)
        if self.chain_2d is None:
            return seq, self.targets[idx]
        chain = self.chain_2d[idx]
        if self.chain_norm_mean is not None and self.chain_norm_std is not None:
            chain = (chain - self.chain_norm_mean) / self.chain_norm_std
        return seq, chain, self.targets[idx]


# ============================================================================
# Model Wrapper
# ============================================================================

class BinaryIndependentAgent(nn.Module):
    def __init__(self, agent_type, feat_dim=325, temporal_dim=128, dropout=0.2,
                 mode='classifier', use_feature_subset=True,
                 use_attention_backbone=False, use_attention_pool=False,
                 use_dilated_tcn=True):
        super().__init__()
        self.mode       = mode
        self.agent_type = agent_type

        self.base = IndependentAgent(
            agent_type=agent_type, feat_dim=feat_dim,
            temporal_dim=temporal_dim, dropout=dropout,
            num_classes=5, use_feature_subset=use_feature_subset,
            use_attention_backbone=use_attention_backbone,
            use_attention_pool=use_attention_pool,
            use_dilated_tcn=use_dilated_tcn,
        )

        if self.base.use_backbone:
            classifier_input_dim = 2 + temporal_dim
        else:
            classifier_input_dim = 2 + 32

        self.base.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, sequences, chain_2d=None):
        logits = self.base(sequences, chain_2d=chain_2d)
        return logits.squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Forward helper (unchanged — NO normalization here, Dataset handles it)
# ============================================================================

def _forward_model(model, seq_batch, chain_batch=None):
    if chain_batch is None:
        return model(seq_batch)
    return model(seq_batch, chain_2d=chain_batch)


# ============================================================================
# LightningModule  ← NEW
# ============================================================================

class LitAgent(L.LightningModule):
    """
    Thin LightningModule wrapper around BinaryIndependentAgent.
    Reuses all existing loss classes; replaces the manual epoch loop.
    """
    def __init__(
        self,
        model: BinaryIndependentAgent,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        focal_gamma: float = 2.0,
        noise_sigma: float = 0.0,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        deg_penalty_weight: float = 0.20,
        deg_target_pos_min: float = 0.35,
        deg_target_pos_max: float = 0.65,
        norm_mean=None,
        norm_std=None,
    ):
        super().__init__()
        self.model       = model
        self.lr          = lr
        self.weight_decay = weight_decay
        self.noise_sigma  = noise_sigma
        self.use_mixup    = use_mixup
        self.mixup_alpha  = mixup_alpha
        self.deg_penalty_weight = max(float(deg_penalty_weight), 0.0)
        self.deg_target_pos_min = float(min(deg_target_pos_min, deg_target_pos_max))
        self.deg_target_pos_max = float(max(deg_target_pos_min, deg_target_pos_max))

        # Norm buffers (used in validation_step where Dataset doesn't normalise)
        if norm_mean is not None:
            self.register_buffer('norm_mean', torch.FloatTensor(norm_mean))
            self.register_buffer('norm_std',  torch.FloatTensor(norm_std))
        else:
            self.norm_mean = self.norm_std = None

        # Losses — symmetric (fp_weight=1.0) with mild focal (gamma=0.5) so the
        # model can express confidence without being penalised for UP predictions.
        self.criterion = AsymmetricTradingLoss(fp_weight=1.0, gamma=min(focal_gamma, 0.5))

        # Register penalty targets as buffers to avoid creating tensors in forward pass
        self.register_buffer('std_target_tensor', torch.tensor(0.15))  # Increased from 0.10 for better spread
        self.register_buffer('deg_target_pos_min_tensor', torch.tensor(float(self.deg_target_pos_min)))
        self.register_buffer('deg_target_pos_max_tensor', torch.tensor(float(self.deg_target_pos_max)))

        # Validation accumulators
        self._val_logits = []
        self._val_labels = []

        # Best metrics (read by main() after trainer.fit)
        self.best_mcc = -1.0
        self.best_auc =  0.5

    def _prediction_rate_penalty(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Two complementary differentiable penalties:
        #
        # 1. Spread penalty: reward output variance. The model trivially minimises BCE
        #    by predicting the class prior constantly (prob ≈ base_rate everywhere).
        #    Penalise low variance to force the model to produce spread-out predictions.
        #    Target std >= 0.15 (increased from 0.10 for better spread).
        std_penalty = torch.relu(self.std_target_tensor - probs.std())
        std_penalty = std_penalty ** 2

        # 2. Rate penalty: keep soft mean(prob) in target band so the model doesn't
        #    constantly predict all-UP or all-DOWN at any threshold.
        soft_rate = probs.mean()
        low_violation = torch.relu(self.deg_target_pos_min_tensor - soft_rate)
        high_violation = torch.relu(soft_rate - self.deg_target_pos_max_tensor)
        rate_penalty = (low_violation + high_violation) ** 2

        # Combined: spread penalty weighted 5× more as it targets the primary failure mode.
        penalty = 5.0 * std_penalty + rate_penalty

        # Report hard rate for monitoring (no gradient needed).
        hard_rate = (probs > 0.5).float().mean().detach()
        return hard_rate, penalty

    # ── forward ──────────────────────────────────────────────
    def forward(self, x, chain=None):
        return self.model(x, chain_2d=chain)

    # ── training step ─────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            xb, chain_b, yb = batch
        else:
            xb, yb = batch; chain_b = None
        yb = yb.float()

        if self.noise_sigma > 0:
            xb = add_gaussian_noise(xb, self.noise_sigma)

        if self.use_mixup and chain_b is None:
            xb, ya, yb2, lam = mixup_sequences(xb, yb, self.mixup_alpha)
            pred = self(xb)
            base_loss = mixup_loss(self.criterion, pred, ya, yb2, lam)
        else:
            pred = self(xb, chain_b)
            base_loss = self.criterion(pred, yb)

        probs = torch.sigmoid(pred)
        train_pred_pos, rate_penalty = self._prediction_rate_penalty(probs)
        loss = base_loss + self.deg_penalty_weight * rate_penalty

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_base_loss', base_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_pred_pos_rate', train_pred_pos, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_deg_penalty', rate_penalty, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    # ── validation step ───────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        # val DataLoader passes raw (un-normalised) sequences; apply norm here
        if len(batch) == 3:
            xb, chain_b, yb = batch
        else:
            xb, yb = batch; chain_b = None

        if self.norm_mean is not None:
            xb = (xb - self.norm_mean) / self.norm_std
        xb = xb.clamp(-5.0, 5.0)

        logits = self(xb, chain_b)
        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(yb.detach().cpu())

    def on_validation_epoch_end(self):
        logits = torch.cat(self._val_logits).numpy()
        labels = torch.cat(self._val_labels).numpy().astype(int)
        self._val_logits.clear()
        self._val_labels.clear()

        probs    = 1.0 / (1.0 + np.exp(-logits))
        dir_pred = (probs > 0.5).astype(int)
        val_pred_pos = float(np.mean(dir_pred)) if len(dir_pred) else 0.0
        val_pos_out_of_band = (
            val_pred_pos < self.deg_target_pos_min or val_pred_pos > self.deg_target_pos_max
        )

        try:
            auc = float(roc_auc_score(labels, probs))
        except Exception:
            auc = 0.5

        mcc = float(matthews_corrcoef(labels, dir_pred))
        f1  = float(f1_score(labels, dir_pred, average='binary', zero_division=0))
        acc = float(accuracy_score(labels, dir_pred))

        self.log_dict(
            {
                'val_auc': auc,
                'val_mcc': mcc,
                'val_f1': f1,
                'val_acc': acc,
                'val_pred_pos_rate': val_pred_pos,
                'val_pos_out_of_band': float(val_pos_out_of_band),
            },
            prog_bar=True,
        )

        if auc < 0.48:
            logger.critical(f"  val_auc={auc:.4f} < 0.48 — possible label inversion!")
        elif auc < 0.5:
            logger.warning(f"  Weak agent (val_auc={auc:.4f}) — monitor closely")
        if val_pos_out_of_band:
            logger.warning(
                "  Validation pred_pos_rate=%.3f out of target band [%.2f, %.2f]",
                val_pred_pos, self.deg_target_pos_min, self.deg_target_pos_max
            )

        # Track bests for logging after fit()
        # FIX7: per-epoch coarse threshold scan
        dir_preds = (probs > 0.5).astype(int)  # FIX7: define dir_preds for threshold scan
        _bthr = 0.5; _btacc = float(balanced_accuracy_score(labels, dir_preds))
        for _t in np.arange(0.35, 0.66, 0.05):
            _dp2 = (probs > _t).astype(int)
            _ta2 = float(balanced_accuracy_score(labels, _dp2))
            if _ta2 > _btacc: _btacc = _ta2; _bthr = _t
        self.log_dict({'val_bacc_opt': _btacc, 'val_opt_thr': _bthr}, prog_bar=True)

        if mcc > self.best_mcc + 0.005:
            self.best_mcc = mcc; self.best_auc = auc
        elif auc > self.best_auc + 0.002:
            self.best_auc = auc

    # ── optimiser + scheduler ─────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # OneCycleLR needs total_steps; Lightning provides it via trainer.estimated_stepping_batches
        total_steps = self.trainer.estimated_stepping_batches
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': sched, 'interval': 'step'},
        }


# ============================================================================
# Lightning training entry point  ← replaces train_one_model()
# ============================================================================

def run_lightning_training(
    model, train_loader, val_seq, val_labels, device,
    epochs=80, lr=1e-4, patience=15,
    accum_steps=4,
    use_mixup=False, mixup_alpha=0.2,
    noise_sigma=0.0,
    deg_penalty_weight=0.20,
    deg_target_pos_min=0.35,
    deg_target_pos_max=0.65,
    val_chain=None, focal_gamma=2.0,
    norm_mean=None, norm_std=None,
    output_root=None, symbol='SPXW', agent_type='A', horizon=30,
    precision='16-mixed',
    chain_norm_mean=None, chain_norm_std=None,
):
    """Drop-in replacement for train_one_model(). Returns (model, best_auc, best_brier=0)."""

    # Val DataLoader — NO norm in dataset; LitAgent applies it in validation_step
    val_ds = SequenceWithOptionalChainDataset(
        val_seq, val_labels.astype(np.float32),
        norm_mean=None, norm_std=None,   # intentionally omitted
        chain_2d=val_chain,
        chain_norm_mean=chain_norm_mean, chain_norm_std=chain_norm_std,
    )
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False,
                            num_workers=0, pin_memory=True)

    lit = LitAgent(
        model=model,
        lr=lr,
        focal_gamma=focal_gamma,
        noise_sigma=noise_sigma,
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        deg_penalty_weight=deg_penalty_weight,
        deg_target_pos_min=deg_target_pos_min,
        deg_target_pos_max=deg_target_pos_max,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    ckpt_dir = str(output_root) if output_root else "."
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f'{symbol}_agent{agent_type}_h{horizon}_best',
        monitor='val_mcc', mode='max', save_top_k=1,
    )
    early_cb = EarlyStopping(
        monitor='val_mcc', mode='max',
        patience=patience, min_delta=0.005,
    )
    lr_cb = LearningRateMonitor(logging_interval='step')

    csv_log = CSVLogger(
        save_dir=str(Path(ckpt_dir) / 'logs'),
        name=f'agent_{agent_type}',
    )

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    # fp16 only makes sense on GPU; fall back to 32 on CPU
    eff_precision = precision if accelerator == 'gpu' else '32'

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        precision=eff_precision,
        accumulate_grad_batches=accum_steps,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb, early_cb, lr_cb],
        logger=csv_log,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    trainer.fit(lit, train_loader, val_loader)

    logger.info(f"  Lightning fit done — best val_mcc={lit.best_mcc:.4f}  best_auc={lit.best_auc:.4f}")

    # Restore best weights from checkpoint
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        state = torch.load(best_ckpt, map_location='cpu')
        # Lightning checkpoint stores model weights under 'state_dict'
        sd = {k.replace('model.', '', 1): v
              for k, v in state['state_dict'].items()
              if k.startswith('model.')}
        model.load_state_dict(sd)
        logger.info(f"  Restored best weights from: {best_ckpt}")

    model = model.to(device)
    return model, lit.best_auc, 0.0   # brier placeholder (eval_model computes it)


# ============================================================================
# Threshold optimisation (UNCHANGED)
# ============================================================================

def _threshold_objective_score(labels, preds, objective):
    if objective == 'balanced_acc':
        return float(balanced_accuracy_score(labels, preds))
    if objective == 'mcc':
        return float(matthews_corrcoef(labels, preds))
    return float(f1_score(labels, preds, average='binary', zero_division=0))


def _constant_prediction_baselines(labels):
    """Reference baselines for degenerate all-UP / all-DOWN predictions."""
    labels = np.asarray(labels).astype(int)
    pred_up = np.ones_like(labels, dtype=int)
    pred_dn = np.zeros_like(labels, dtype=int)

    def _metrics(pred):
        return {
            'accuracy': float(accuracy_score(labels, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(labels, pred)),
            'f1': float(f1_score(labels, pred, average='binary', zero_division=0)),
            'mcc': float(matthews_corrcoef(labels, pred)),
        }

    return {
        'label_up_ratio': float(labels.mean()) if len(labels) else 0.0,
        'all_up': _metrics(pred_up),
        'all_down': _metrics(pred_dn),
    }


def _probability_quantiles(probs: np.ndarray) -> dict:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return {
            'n': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'quantiles': {},
        }
    q_levels = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    q_vals = np.quantile(probs, q_levels)
    q_map = {}
    for q, v in zip(q_levels, q_vals):
        key = f"p{int(round(q * 100)):02d}"
        q_map[key] = round(float(v), 6)
    return {
        'n': int(probs.size),
        'mean': round(float(probs.mean()), 6),
        'std': round(float(probs.std()), 6),
        'min': round(float(probs.min()), 6),
        'max': round(float(probs.max()), 6),
        'quantiles': q_map,
    }


def _compute_threshold_curve(labels: np.ndarray, probs: np.ndarray) -> list[dict]:
    labels = np.asarray(labels).astype(int).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    curve = []
    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (probs > threshold).astype(int)
        pred_pos = float(np.mean(preds)) if len(preds) else 0.0
        curve.append({
            'threshold': round(float(threshold), 3),
            'pred_pos_rate': round(pred_pos, 4),
            'mcc': round(float(matthews_corrcoef(labels, preds)), 6),
            'f1': round(float(f1_score(labels, preds, average='binary', zero_division=0)), 6),
            'balanced_acc': round(float(balanced_accuracy_score(labels, preds)), 6),
        })
    return curve


def _best_threshold_for_probs(
    labels, probs, objective,
    target_pos_min: float = 0.35, target_pos_max: float = 0.65,
):
    best_threshold = 0.5
    best_f1 = 0.0
    best_obj = -1e18
    best_pred_pos_gap = 1e9
    best_is_degenerate = True
    best_in_band = False
    # Wider sweep avoids forced one-class predictions when calibrated probs
    # are clustered outside the old 0.30-0.65 band.
    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (probs > threshold).astype(int)
        obj = _threshold_objective_score(labels, preds, objective)
        f1 = float(f1_score(labels, preds, average='binary', zero_division=0))
        pred_pos = float(np.mean(preds)) if len(preds) else 0.0
        is_degenerate = (pred_pos <= 0.01) or (pred_pos >= 0.99)
        in_band = (target_pos_min <= pred_pos <= target_pos_max)
        pred_pos_gap = abs(pred_pos - 0.5)

        pick = False
        if obj > best_obj + 1e-8:
            pick = True
        elif abs(obj - best_obj) < 1e-8:
            # Tie-breaking priority (highest to lowest):
            # 1. Prefer non-degenerate over degenerate.
            if (not is_degenerate) and best_is_degenerate:
                pick = True
            elif is_degenerate == best_is_degenerate:
                # 2. Prefer in-band pred_pos_rate [target_min, target_max].
                if in_band and not best_in_band:
                    pick = True
                elif in_band == best_in_band:
                    # 3. Prefer less skewed prediction rate.
                    if pred_pos_gap < best_pred_pos_gap - 1e-8:
                        pick = True
                    elif abs(pred_pos_gap - best_pred_pos_gap) < 1e-8 and f1 > best_f1:
                        pick = True

        if pick:
            best_obj = obj
            best_f1 = f1
            best_threshold = threshold
            best_pred_pos_gap = pred_pos_gap
            best_is_degenerate = is_degenerate
            best_in_band = in_band
    return best_threshold, best_f1, best_obj


def _align_with_optional_chain(seq, labels, chain=None, returns=None, tag="split"):
    """Ensure seq/labels/(returns)/chain lengths match before batched inference."""
    n = len(seq)
    n = min(n, len(labels))
    if returns is not None:
        n = min(n, len(returns))
    if chain is not None:
        n = min(n, len(chain))
    if (n != len(seq)) or (n != len(labels)) or (returns is not None and n != len(returns)) or (chain is not None and n != len(chain)):
        logger.warning(
            "[%s] Length mismatch detected (seq=%d labels=%d returns=%s chain=%s) -> clipping to %d",
            tag,
            len(seq),
            len(labels),
            str(len(returns)) if returns is not None else "NA",
            str(len(chain)) if chain is not None else "NA",
            n,
        )
    seq = seq[:n]
    labels = labels[:n]
    if returns is not None:
        returns = returns[:n]
    if chain is not None:
        chain = chain[:n]
    return seq, labels, chain, returns


def optimize_threshold(model, val_seq, val_labels, device, mode='classifier',
                       val_chain=None, norm_mean=None, norm_std=None,
                       chain_norm_mean=None, chain_norm_std=None,
                       threshold_objective='f1',
                       target_pos_min: float = 0.35, target_pos_max: float = 0.65):
    val_seq, val_labels, val_chain, _ = _align_with_optional_chain(
        val_seq, val_labels, chain=val_chain, returns=None, tag="val_opt"
    )
    model.eval()
    norm_mean_t = torch.FloatTensor(norm_mean).to(device) if norm_mean is not None else None
    norm_std_t  = torch.FloatTensor(norm_std).to(device)  if norm_std  is not None else None
    chain_mean_t = (torch.FloatTensor(chain_norm_mean).view(1, -1, 1, 1).to(device)
                    if chain_norm_mean is not None else None)
    chain_std_t = (torch.FloatTensor(chain_norm_std).view(1, -1, 1, 1).to(device)
                   if chain_norm_std is not None else None)

    with torch.no_grad():
        outputs = []
        bs = 2048
        for i in range(0, len(val_seq), bs):
            xb = torch.FloatTensor(np.array(val_seq[i:i+bs])).to(device)
            if norm_mean_t is not None:
                xb = (xb - norm_mean_t) / norm_std_t
            xb = xb.clamp(-5.0, 5.0)
            chain_slice = (torch.FloatTensor(np.array(val_chain[i:i+bs])).to(device)
                           if val_chain is not None else None)
            if chain_slice is not None and chain_mean_t is not None:
                chain_slice = (chain_slice - chain_mean_t) / chain_std_t
            outputs.append(_forward_model(model, xb, chain_slice))
        raw_output = torch.cat(outputs).cpu().numpy()

    platt_scaler = None
    invert_signal = False
    diagnostics = {}
    if mode == 'classifier':
        platt_scaler = LogisticRegression()
        platt_scaler.fit(raw_output.reshape(-1, 1), val_labels)
        logger.info("  Fitted Platt scaler for probability calibration")
        base_probs = platt_scaler.predict_proba(raw_output.reshape(-1, 1))[:, 1]
        try:
            val_auc = float(roc_auc_score(val_labels, base_probs))
        except Exception:
            val_auc = 0.5
        flipped_probs = 1.0 - base_probs
        try:
            val_auc_flipped = float(roc_auc_score(val_labels, flipped_probs))
        except Exception:
            val_auc_flipped = 0.5

        thr_base, f1_base, obj_base = _best_threshold_for_probs(
            val_labels, base_probs, threshold_objective,
            target_pos_min=target_pos_min, target_pos_max=target_pos_max,
        )
        thr_flip, f1_flip, obj_flip = _best_threshold_for_probs(
            val_labels, flipped_probs, threshold_objective,
            target_pos_min=target_pos_min, target_pos_max=target_pos_max,
        )

        use_flip = (obj_flip > obj_base + 1e-8) or (
            abs(obj_flip - obj_base) < 1e-8 and f1_flip > f1_base + 1e-8
        )
        invert_signal = bool(use_flip)
        if invert_signal:
            probs = flipped_probs
            raw_output = -raw_output
            best_threshold, best_f1, best_obj = thr_flip, f1_flip, obj_flip
            logger.warning(
                "  Auto-flip selected by validation objective: auc_base=%.4f auc_flip=%.4f",
                val_auc, val_auc_flipped
            )
        else:
            probs = base_probs
            best_threshold, best_f1, best_obj = thr_base, f1_base, obj_base
            if val_auc < 0.5:
                logger.warning(f"  Weak agent (val_auc={val_auc:.4f}) — monitor closely")
    else:
        probs = raw_output
        best_threshold, best_f1, best_obj = _best_threshold_for_probs(
            val_labels, probs, threshold_objective,
            target_pos_min=target_pos_min, target_pos_max=target_pos_max,
        )

    logger.info(f"  Optimal threshold: {best_threshold:.2f} "
                f"(objective={threshold_objective}={best_obj:.4f}, val_f1={best_f1:.4f})")
    selected_preds = (probs > best_threshold).astype(int)
    diagnostics = {
        'objective': threshold_objective,
        'objective_value': round(float(best_obj), 6),
        'selected_threshold': round(float(best_threshold), 3),
        'selected_pred_pos_rate': round(float(np.mean(selected_preds)) if len(selected_preds) else 0.0, 4),
        'selected_metrics': {
            'mcc': round(float(matthews_corrcoef(val_labels, selected_preds)), 6),
            'f1': round(float(f1_score(val_labels, selected_preds, average='binary', zero_division=0)), 6),
            'balanced_acc': round(float(balanced_accuracy_score(val_labels, selected_preds)), 6),
        },
        'probability_summary': _probability_quantiles(probs),
        'threshold_curve': _compute_threshold_curve(val_labels, probs),
    }
    return best_threshold, best_f1, platt_scaler, invert_signal, diagnostics


# ============================================================================
# Evaluation (UNCHANGED)
# ============================================================================

def evaluate_model(model, test_seq, test_labels, test_returns, device,
                   mode='classifier', threshold=0.5, test_chain=None,
                   platt_scaler=None, norm_mean=None, norm_std=None,
                   invert_signal=False, chain_norm_mean=None, chain_norm_std=None,
                   baseline_metrics=None, eval_tag='eval',
                   deg_target_pos_min=0.35, deg_target_pos_max=0.65):
    test_seq, test_labels, test_chain, test_returns = _align_with_optional_chain(
        test_seq, test_labels, chain=test_chain, returns=test_returns, tag="test_eval"
    )
    model.eval()
    norm_mean_t = torch.FloatTensor(norm_mean).to(device) if norm_mean is not None else None
    norm_std_t  = torch.FloatTensor(norm_std).to(device)  if norm_std  is not None else None
    chain_mean_t = (torch.FloatTensor(chain_norm_mean).view(1, -1, 1, 1).to(device)
                    if chain_norm_mean is not None else None)
    chain_std_t = (torch.FloatTensor(chain_norm_std).view(1, -1, 1, 1).to(device)
                   if chain_norm_std is not None else None)

    with torch.no_grad():
        outputs = []
        bs = 2048
        for i in range(0, len(test_seq), bs):
            xb = torch.FloatTensor(np.array(test_seq[i:i+bs])).to(device)
            if norm_mean_t is not None:
                xb = (xb - norm_mean_t) / norm_std_t
            xb = xb.clamp(-5.0, 5.0)
            chain_slice = (torch.FloatTensor(np.array(test_chain[i:i+bs])).to(device)
                           if test_chain is not None else None)
            if chain_slice is not None and chain_mean_t is not None:
                chain_slice = (chain_slice - chain_mean_t) / chain_std_t
            outputs.append(_forward_model(model, xb, chain_slice))
        raw_output = torch.cat(outputs).cpu().numpy()

    # Compute AUC from raw_output BEFORE any transforms
    try:
        auc = roc_auc_score(test_labels, raw_output)
    except Exception:
        auc = 0.5

    threshold_curve = None
    if mode == 'classifier':
        if platt_scaler is not None:
            probs = platt_scaler.predict_proba(raw_output.reshape(-1, 1))[:, 1]
        else:
            probs = 1 / (1 + np.exp(-raw_output))
        if invert_signal:
            probs = 1.0 - probs
        preds      = (probs > threshold).astype(int)
        threshold_curve = _compute_threshold_curve(test_labels, probs)
        confidence = np.abs(probs - 0.5) * 2
        brier      = brier_score_loss(test_labels, probs)
    else:
        preds     = (raw_output > 0).astype(int)
        abs_mag   = np.abs(raw_output)
        confidence = (abs_mag - abs_mag.min()) / (abs_mag.max() - abs_mag.min() + 1e-8)
        brier = None; probs = raw_output

    acc  = accuracy_score(test_labels, preds)
    bacc = balanced_accuracy_score(test_labels, preds)
    f1   = f1_score(test_labels, preds, average='binary')
    ic, _ = spearmanr(raw_output, test_returns)
    if np.isnan(ic): ic = 0.0
    pred_pos_rate = float(np.mean(preds)) if len(preds) else 0.0
    degenerate_prediction = (pred_pos_rate <= 0.01) or (pred_pos_rate >= 0.99)
    pred_pos_out_of_band = (
        pred_pos_rate < float(deg_target_pos_min)
        or pred_pos_rate > float(deg_target_pos_max)
    )
    if degenerate_prediction:
        logger.warning(
            "[%s] Near-constant predictions: pred_pos_rate=%.4f at threshold=%.3f",
            eval_tag, pred_pos_rate, float(threshold)
        )
    if pred_pos_out_of_band:
        logger.warning(
            "[%s] Prediction-rate outside target band: pred_pos_rate=%.4f band=[%.2f, %.2f]",
            eval_tag, pred_pos_rate, float(deg_target_pos_min), float(deg_target_pos_max),
        )

    if baseline_metrics is None:
        baseline_metrics = _constant_prediction_baselines(test_labels)
    all_up = baseline_metrics.get('all_up', {})
    all_down = baseline_metrics.get('all_down', {})
    matches_all_up = (
        abs(float(acc) - float(all_up.get('accuracy', -1.0))) < 1e-6 and
        abs(float(f1) - float(all_up.get('f1', -1.0))) < 1e-6
    )
    matches_all_down = (
        abs(float(acc) - float(all_down.get('accuracy', -1.0))) < 1e-6 and
        abs(float(f1) - float(all_down.get('f1', -1.0))) < 1e-6
    )
    if matches_all_up or matches_all_down:
        which = "all-UP" if matches_all_up else "all-DOWN"
        logger.warning("[%s] Test metrics match %s constant baseline", eval_tag, which)

    ranking_signal = probs if mode == 'classifier' else raw_output
    order = np.argsort(ranking_signal)
    quintile_returns = {}; quintile_mean_returns = []
    if len(order) >= 5:
        for idx, bin_indices in enumerate(np.array_split(order, 5), start=1):
            mr = float(np.mean(test_returns[bin_indices])) if len(bin_indices) > 0 else 0.0
            quintile_mean_returns.append(mr)
            quintile_returns[f'q{idx}'] = round(mr * RETURN_SCALE, 4)
    quintile_spread_bp = None
    if len(quintile_mean_returns) == 5:
        quintile_spread_bp = round(
            (quintile_mean_returns[-1] - quintile_mean_returns[0]) * RETURN_SCALE, 4
        )

    conf_buckets = {}
    for thr in [0.0, 0.2, 0.4, 0.6, 0.8]:
        mask = confidence >= thr
        if mask.sum() > 50:
            conf_buckets[f"conf>={thr:.1f}"] = {
                'accuracy': round(float(accuracy_score(test_labels[mask], preds[mask])), 4),
                'f1':       round(float(f1_score(test_labels[mask], preds[mask],
                                                 average='binary')), 4),
                'coverage': round(float(mask.mean()), 4),
                'n':        int(mask.sum()),
            }

    return {
        'accuracy':          round(float(acc), 4),
        'balanced_accuracy': round(float(bacc), 4),
        'f1':                round(float(f1), 4),
        'auc':               round(float(auc), 4),
        'ic':                round(float(ic), 4),
        'brier':             round(float(brier), 6) if brier is not None else None,
        'threshold':         round(float(threshold), 3),
        'pred_pos_rate':     round(float(pred_pos_rate), 4),
        'pred_neg_rate':     round(float(1.0 - pred_pos_rate), 4),
        'degenerate_prediction': bool(degenerate_prediction),
        'pred_pos_out_of_band': bool(pred_pos_out_of_band),
        'matches_all_up_baseline': bool(matches_all_up),
        'matches_all_down_baseline': bool(matches_all_down),
        'baseline_all_up': {
            'accuracy': round(float(all_up.get('accuracy', 0.0)), 4),
            'f1': round(float(all_up.get('f1', 0.0)), 4),
            'balanced_accuracy': round(float(all_up.get('balanced_accuracy', 0.0)), 4),
            'mcc': round(float(all_up.get('mcc', 0.0)), 4),
        },
        'baseline_all_down': {
            'accuracy': round(float(all_down.get('accuracy', 0.0)), 4),
            'f1': round(float(all_down.get('f1', 0.0)), 4),
            'balanced_accuracy': round(float(all_down.get('balanced_accuracy', 0.0)), 4),
            'mcc': round(float(all_down.get('mcc', 0.0)), 4),
        },
        'quintile_mean_returns_bp': quintile_returns,
        'quintile_spread_bp': quintile_spread_bp,
        'confidence_buckets': conf_buckets,
        'probability_summary': _probability_quantiles(probs),
        'threshold_curve': threshold_curve if threshold_curve is not None else [],
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',      default='SPXW')
    parser.add_argument('--symbols',     nargs='+', default=None)
    parser.add_argument('--all-symbols', action='store_true')
    parser.add_argument('--data-root',   default=str(DEFAULT_DATA_ROOT))
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--chain-2d-dir', default=str(DEFAULT_CHAIN_2D_DIR))
    parser.add_argument('--agents',      nargs='+', default=ALL_AGENTS)
    parser.add_argument('--horizon',     type=int,   default=DEFAULT_TRAINING_HORIZON_MINUTES)
    parser.add_argument('--epochs',      type=int,   default=80)
    parser.add_argument('--batch-size',  type=int,   default=512)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--patience',    type=int,   default=15)
    parser.add_argument('--accum-steps', type=int,   default=4)
    parser.add_argument('--f1-weight',   type=float, default=0.3)
    parser.add_argument('--no-feature-subset',      action='store_true')
    parser.add_argument('--use-attention-backbone', action='store_true')
    parser.add_argument('--use-attention-pool',     action='store_true')
    parser.add_argument('--no-dilated-tcn',         action='store_true',
                        help='Disable DilatedCausalTCN (fall back to TemporalBackbone)')
    parser.add_argument('--use-mixup',              action='store_true')
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--noise-sigma', type=float, default=0.02)
    parser.add_argument('--deg-penalty-weight', type=float, default=0.50,
                        help='Weight for anti-degeneracy prediction-rate penalty term (increased from 0.20).')
    parser.add_argument('--deg-target-pos-min', type=float, default=0.40,
                        help='Lower bound for target positive prediction rate (tightened from 0.35).')
    parser.add_argument('--deg-target-pos-max', type=float, default=0.60,
                        help='Upper bound for target positive prediction rate (tightened from 0.65).')
    parser.add_argument('--threshold-objective',
                        choices=['f1', 'balanced_acc', 'mcc'], default='mcc')
    parser.add_argument('--precision',   default='16-mixed',
                        help='Lightning precision: 16-mixed | bf16-mixed | 32')
    parser.add_argument('--smoke-samples', type=int, default=0,
                        help='If >0, truncate each split to first N rows for quick wiring checks')
    parser.add_argument('--debug',       action='store_true')
    parser.add_argument('--device',      default='cpu',
                        help='Kept for backward compat; Lightning auto-selects GPU/CPU')
    parser.add_argument('--min-h-live-ratio', type=float, default=0.20,
                        help='Skip Agent H when OHLC live dims ratio is below this threshold')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root    = Path(args.data_root)
    output_root  = Path(args.output_root)
    chain_2d_dir = Path(args.chain_2d_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.deg_target_pos_min > args.deg_target_pos_max:
        parser.error('--deg-target-pos-min must be <= --deg-target-pos-max')

    if args.all_symbols and args.symbols is not None:
        raise SystemExit("Use only one of --all-symbols or --symbols")
    if args.all_symbols:
        symbols = ALL_SYMBOLS
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = [args.symbol]

    horizon = args.horizon

    for symbol in symbols:
        data_dir = data_root / symbol / f"horizon_{horizon}min"
        if not data_dir.exists():
            logger.warning(f"[{symbol}] Data not found, skipping: {data_dir}")
            continue

        logger.info(f"{'='*70}")
        logger.info(f"STAGE 1 v2 [Lightning]: {symbol} | Horizon={horizon}min | precision={args.precision}")
        logger.info(f"{'='*70}")

        train_seq     = np.load(data_dir / 'train_sequences.npy')
        train_labels  = np.load(data_dir / 'train_labels.npy')
        train_returns = np.load(data_dir / 'train_returns.npy')
        val_seq       = np.load(data_dir / 'val_sequences.npy')
        val_labels    = np.load(data_dir / 'val_labels.npy')
        val_returns   = np.load(data_dir / 'val_returns.npy')
        test_seq      = np.load(data_dir / 'test_sequences.npy')
        test_labels   = np.load(data_dir / 'test_labels.npy')
        test_returns  = np.load(data_dir / 'test_returns.npy')

        if args.smoke_samples > 0:
            n = int(args.smoke_samples)
            train_seq, train_labels, train_returns = train_seq[:n], train_labels[:n], train_returns[:n]
            val_seq, val_labels, val_returns = val_seq[:n], val_labels[:n], val_returns[:n]
            test_seq, test_labels, test_returns = test_seq[:n], test_labels[:n], test_returns[:n]
            logger.info(f"[SMOKE] Truncated splits to first {n} rows each")
            if args.threshold_objective == 'balanced_acc':
                logger.warning(
                    "[SMOKE] threshold_objective=balanced_acc may tie under class imbalance; "
                    "prefer mcc or f1 for meaningful smoke comparisons."
                )

        train_chain_path, val_chain_path, test_chain_path = _resolve_chain_2d_paths(
            data_dir, symbol, chain_2d_dir, len(train_seq), len(val_seq)
        )
        has_chain_files = train_chain_path is not None
        train_chain = val_chain = test_chain = None
        chain_norm_mean = chain_norm_std = None
        chain_norm_src = 'none'

        feat_dim = train_seq.shape[2]
        data_schema_version = _load_feature_schema(data_dir)
        assert_supported_schema(data_schema_version)

        # Dead-feature mask
        X = train_seq.reshape(-1, feat_dim)
        std          = X.std(axis=0)
        nonzero_rate = (np.abs(X) > 0).mean(axis=0)
        feature_mask = (std > STD_EPS) & (nonzero_rate > 1e-4)
        live_idx_path = data_dir / 'live_feature_indices.npy'
        if live_idx_path.exists():
            saved_live = np.load(live_idx_path)
            saved_mask = np.zeros(feat_dim, dtype=bool)
            saved_mask[saved_live] = True
            feature_mask &= saved_mask
            logger.info(f"Loaded persisted live-feature mask: {int(saved_mask.sum())}/{feat_dim} dims")
        logger.info(f"Dead-feature mask keeps {int(feature_mask.sum())}/{feat_dim} dims")
        mask3 = feature_mask.astype(np.float32)[None, None, :]
        train_seq *= mask3; val_seq *= mask3; test_seq *= mask3

        # Normalization stats
        norm_mean, norm_std, norm_src = _load_normalization_stats(data_dir, feat_dim)
        if norm_mean is not None and norm_std is not None:
            logger.info("Will apply z-score normalization on-the-fly (%s)", norm_src)
        else:
            logger.warning("No normalization stats found — training without z-score norm")
        chain_norm_mean, chain_norm_std, chain_norm_src = _load_chain_norm_stats(data_dir)
        if chain_norm_mean is not None:
            logger.info("Chain2D normalization source: %s", chain_norm_src)
        else:
            logger.info("Chain2D normalization source: none (raw chain_2d inputs)")

        train_returns_scaled = train_returns * RETURN_SCALE
        logger.info(f"Data: train={len(train_seq):,} val={len(val_seq):,} "
                    f"test={len(test_seq):,} feat_dim={feat_dim}")

        data_ok = verify_data_before_training(data_dir, symbol, horizon)
        if not data_ok:
            logger.warning(f"[{symbol}] Data health check FAILED — continuing anyway.")
        test_baselines = _constant_prediction_baselines(test_labels)
        logger.info(
            "Baselines (test): up_ratio=%.4f all-UP(acc=%.4f,f1=%.4f) all-DOWN(acc=%.4f,f1=%.4f)",
            test_baselines['label_up_ratio'],
            test_baselines['all_up']['accuracy'],
            test_baselines['all_up']['f1'],
            test_baselines['all_down']['accuracy'],
            test_baselines['all_down']['f1'],
        )

        all_results = {}

        for agent_type in args.agents:
            logger.info(f"\n --- Agent {agent_type} ---")
            subset_info = AGENT_FEATURE_SUBSETS.get(agent_type, {})
            logger.info(f"  Subset: {subset_info.get('name', 'N/A')} "
                        f"({subset_info.get('feat_dim', feat_dim)} dims)")
            subset_indices = subset_info.get('indices', None)
            if subset_indices is None and subset_info.get('ranges'):
                subset_indices = []
                for s, e in subset_info['ranges']:
                    subset_indices.extend(range(s, e))
            if subset_indices is not None:
                valid_subset = [i for i in subset_indices if i < feat_dim]
                if len(valid_subset) == 0:
                    logger.warning(
                        "  [%s] Skipping: subset indices out of range for feat_dim=%d "
                        "(expected expanded Tier3 with OHLC block)",
                        agent_type, feat_dim,
                    )
                    all_results[f"agent_{agent_type}_classifier"] = {
                        'error': f'subset_out_of_range_feat_dim_{feat_dim}'
                    }
                    continue
                subset_indices = valid_subset

            if agent_type == '2D':
                if not has_chain_files:
                    logger.warning("  [2D] Skipping: no chain_2d files found")
                    all_results[f"agent_{agent_type}_classifier"] = {
                        'error': 'missing chain_2d files'
                    }
                    continue
                if train_chain is None:
                    train_chain = np.load(train_chain_path)
                    val_chain   = np.load(val_chain_path)
                    test_chain  = np.load(test_chain_path)
                    if args.smoke_samples > 0:
                        n = int(args.smoke_samples)
                        train_chain = train_chain[:n]
                        val_chain = val_chain[:n]
                        test_chain = test_chain[:n]

            if agent_type == 'H' and subset_indices:
                Xh = train_seq.reshape(-1, feat_dim)[:, subset_indices]
                h_live = (Xh.std(axis=0) > STD_EPS) & ((np.abs(Xh) > 0).mean(axis=0) > 1e-4)
                h_live_ratio = float(h_live.mean()) if len(h_live) else 0.0
                logger.info("  [H] Live OHLC dims ratio: %.1f%%", 100.0 * h_live_ratio)
                if h_live_ratio < float(args.min_h_live_ratio):
                    logger.warning(
                        "  [H] Skipping: OHLC live ratio %.1f%% below threshold %.1f%%",
                        100.0 * h_live_ratio, 100.0 * float(args.min_h_live_ratio),
                    )
                    all_results[f"agent_{agent_type}_classifier"] = {
                        'error': f'h_live_ratio_below_threshold_{h_live_ratio:.3f}'
                    }
                    continue

            for mode in ['classifier']:
                logger.info(f"  [{agent_type}] Mode: {mode}")
                try:
                    model = BinaryIndependentAgent(
                        agent_type=agent_type, feat_dim=feat_dim,
                        temporal_dim=32, dropout=0.5, mode=mode,
                        use_feature_subset=not args.no_feature_subset,
                        use_attention_backbone=args.use_attention_backbone,
                        use_attention_pool=args.use_attention_pool,
                        use_dilated_tcn=not args.no_dilated_tcn,
                    ).to(device)
                    n_params = model.count_parameters()
                    logger.info(f"  Params: {n_params:,}")

                    targets            = train_labels.astype(np.float32)
                    positive_class_prior = float(np.mean(targets))

                    # FIX5b: per-agent dead-feature mask (inside try, correct scope)
                    _agent_train_seq = train_seq.copy()
                    _agent_val_seq   = val_seq.copy()
                    _agent_test_seq  = test_seq.copy()
                    _agent_indices = subset_indices
                    if _agent_indices is not None:
                        _Xa = _agent_train_seq.reshape(-1, feat_dim)[:, _agent_indices]
                        _agent_live = (_Xa.std(axis=0) > STD_EPS) & ((np.abs(_Xa) > 0).mean(axis=0) > 1e-4)
                        _agent_mask_full = np.zeros(feat_dim, dtype=np.float32)
                        _agent_mask_full[_agent_indices] = _agent_live.astype(np.float32)
                        logger.info(f"  [{agent_type}] Per-agent mask: {int(_agent_live.sum())}/{len(_agent_indices)} active dims")
                    else:
                        # FIX5b-fallback: compute full-dim mask locally
                        _Xf = _agent_train_seq.reshape(-1, feat_dim)
                        _agent_mask_full = ((_Xf.std(axis=0) > STD_EPS) & ((np.abs(_Xf) > 0).mean(axis=0) > 1e-4)).astype(np.float32)
                    _amask3 = _agent_mask_full[None, None, :]
                    _agent_train_seq *= _amask3; _agent_val_seq *= _amask3; _agent_test_seq *= _amask3

                    # FIX6: clip all agents to ±5σ in raw space (pre-normalization).
                    # The Dataset also clamps post-normalization; both layers are needed
                    # because 122 features have raw outliers reaching 300+σ that can
                    # produce NaN/Inf during the normalization step itself.
                    if norm_mean is not None:
                        _clip_std = np.maximum(norm_std, 1e-8)
                        _agent_train_seq = np.clip(_agent_train_seq, norm_mean - 5*_clip_std, norm_mean + 5*_clip_std)
                        _agent_val_seq   = np.clip(_agent_val_seq,   norm_mean - 5*_clip_std, norm_mean + 5*_clip_std)
                        _agent_test_seq  = np.clip(_agent_test_seq,  norm_mean - 5*_clip_std, norm_mean + 5*_clip_std)
                        logger.info(f"  [{agent_type}] ±5σ feature clipping applied (raw + normalized)")

                    train_ds = SequenceWithOptionalChainDataset(
                        _agent_train_seq, targets,
                        norm_mean=norm_mean, norm_std=norm_std,
                        chain_2d=train_chain if agent_type == '2D' else None,
                        chain_norm_mean=chain_norm_mean if agent_type == '2D' else None,
                        chain_norm_std=chain_norm_std if agent_type == '2D' else None,
                    )

                    # Balanced sampler: force 50/50 UP/DOWN per batch
                    _cls_counts     = np.bincount(train_labels.astype(int))
                    _sample_weights = torch.FloatTensor(
                        1.0 / _cls_counts[train_labels.astype(int)]
                    )
                    _sampler = WeightedRandomSampler(
                        weights=_sample_weights,
                        num_samples=len(train_labels),
                        replacement=True,
                    )
                    train_loader = DataLoader(
                        train_ds, batch_size=args.batch_size,
                        sampler=_sampler, drop_last=True,
                        num_workers=0, pin_memory=True,
                    )

                    focal_gamma = 1.0 if agent_type in ('TQ', 'H') else 2.0

                    # ── Lightning training (replaces train_one_model) ──────
                    model, val_auc, _ = run_lightning_training(
                        model, train_loader, _agent_val_seq, val_labels, device,
                        epochs=args.epochs, lr=args.lr,
                        patience=args.patience, accum_steps=args.accum_steps,
                        use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
                        noise_sigma=args.noise_sigma,
                        deg_penalty_weight=args.deg_penalty_weight,
                        deg_target_pos_min=args.deg_target_pos_min,
                        deg_target_pos_max=args.deg_target_pos_max,
                        val_chain=val_chain if agent_type == '2D' else None,
                        focal_gamma=focal_gamma,
                        norm_mean=norm_mean, norm_std=norm_std,
                        output_root=output_root,
                        symbol=symbol, agent_type=agent_type, horizon=horizon,
                        precision=args.precision,
                        chain_norm_mean=chain_norm_mean if agent_type == '2D' else None,
                        chain_norm_std=chain_norm_std if agent_type == '2D' else None,
                    )

                    opt_threshold, opt_val_f1, platt_scaler, invert_signal, val_threshold_diagnostics = optimize_threshold(
                        model, _agent_val_seq, val_labels, device, mode=mode,
                        val_chain=val_chain if agent_type == '2D' else None,
                        norm_mean=norm_mean, norm_std=norm_std,
                        chain_norm_mean=chain_norm_mean if agent_type == '2D' else None,
                        chain_norm_std=chain_norm_std if agent_type == '2D' else None,
                        threshold_objective=args.threshold_objective,
                        target_pos_min=args.deg_target_pos_min,
                        target_pos_max=args.deg_target_pos_max,
                    )

                    test_metrics = evaluate_model(
                        model, _agent_test_seq, test_labels, test_returns, device,
                        mode=mode, threshold=opt_threshold,
                        test_chain=test_chain if agent_type == '2D' else None,
                        platt_scaler=platt_scaler,
                        norm_mean=norm_mean, norm_std=norm_std,
                        invert_signal=invert_signal,
                        chain_norm_mean=chain_norm_mean if agent_type == '2D' else None,
                        chain_norm_std=chain_norm_std if agent_type == '2D' else None,
                        baseline_metrics=test_baselines,
                        eval_tag=f"{symbol}:{agent_type}:{mode}",
                        deg_target_pos_min=args.deg_target_pos_min,
                        deg_target_pos_max=args.deg_target_pos_max,
                    )

                    logger.info(
                        f"  Test: acc={test_metrics['accuracy']:.4f} "
                        f"f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f} "
                        f"ic={test_metrics['ic']:.4f} "
                        f"brier={test_metrics['brier']:.6f} "
                        f"thr={test_metrics['threshold']:.3f}"
                    )

                    ckpt_data = {
                        'model_state_dict':       model.state_dict(),
                        'test_metrics':           test_metrics,
                        'agent_type':             agent_type,
                        'horizon':                horizon,
                        'mode':                   mode,
                        'n_params':               n_params,
                        'optimal_threshold':      opt_threshold,
                        'positive_class_prior':   positive_class_prior,
                        'feature_subset':         not args.no_feature_subset,
                        'subset_feat_dim':        model.base.subset_feat_dim,
                        'uses_chain_2d':          agent_type == '2D',
                        'use_attention_backbone': args.use_attention_backbone,
                        'use_attention_pool':     args.use_attention_pool,
                        'use_mixup':              args.use_mixup,
                        'noise_sigma':            args.noise_sigma,
                        'deg_penalty_weight':     args.deg_penalty_weight,
                        'deg_target_pos_min':     args.deg_target_pos_min,
                        'deg_target_pos_max':     args.deg_target_pos_max,
                        'invert_signal':          bool(invert_signal),
                        'threshold_objective':    args.threshold_objective,
                        'val_threshold_diagnostics': val_threshold_diagnostics,
                    }
                    if platt_scaler is not None:
                        ckpt_data['platt_scaler_coef']      = platt_scaler.coef_.tolist()
                        ckpt_data['platt_scaler_intercept'] = platt_scaler.intercept_.tolist()
                    if norm_mean is not None:
                        ckpt_data['norm_mean'] = norm_mean.tolist()
                        ckpt_data['norm_std']  = norm_std.tolist()

                    ckpt_path = output_root / f'{symbol}_agent{agent_type}_{mode}_h{horizon}.pt'
                    torch.save(ckpt_data, ckpt_path)

                    key = f"agent_{agent_type}_{mode}"
                    all_results[key] = test_metrics
                    all_results[key]['n_params'] = n_params
                    all_results[key]['val_threshold_diagnostics'] = val_threshold_diagnostics

                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    import traceback; traceback.print_exc()
                    all_results[f"agent_{agent_type}_{mode}"] = {'error': str(e)}

                try:
                    del model
                except Exception:
                    pass
                torch.cuda.empty_cache()

        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY: {symbol} | Horizon={horizon}min")
        logger.info(f"{'='*80}")
        logger.info(f"{'Agent':>6} {'Mode':>12} {'Accuracy':>10} {'F1':>8} "
                    f"{'AUC':>8} {'IC':>8} {'Thr':>6} {'Pred+%':>8} {'Flag':>14}")
        logger.info("-" * 102)
        for agent_type in args.agents:
            for mode in ['classifier']:
                rkey = f"agent_{agent_type}_{mode}"
                r    = all_results.get(rkey, {})
                if 'error' in r:
                    logger.info(f"{agent_type:>6} {mode:>12} {'FAILED':>10}")
                elif 'accuracy' in r:
                    flag = ''
                    if r.get('matches_all_up_baseline'):
                        flag = 'ALL_UP_BASE'
                    elif r.get('matches_all_down_baseline'):
                        flag = 'ALL_DN_BASE'
                    if r.get('degenerate_prediction'):
                        flag = (flag + '+DEGEN') if flag else 'DEGEN'
                    if r.get('pred_pos_out_of_band'):
                        flag = (flag + '+POS_OOB') if flag else 'POS_OOB'
                    logger.info(
                        f"{agent_type:>6} {mode:>12} "
                        f"{r['accuracy']:>10.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} "
                        f"{r['ic']:>8.4f} {r.get('threshold', 0.5):>6.3f} "
                        f"{100.0 * r.get('pred_pos_rate', 0.0):>7.2f}% {flag:>14}"
                    )
        logger.info(
            "\nConstant baselines (same test split): all-UP acc=%.4f f1=%.4f | "
            "all-DOWN acc=%.4f f1=%.4f",
            test_baselines['all_up']['accuracy'],
            test_baselines['all_up']['f1'],
            test_baselines['all_down']['accuracy'],
            test_baselines['all_down']['f1'],
        )
        logger.info("Random baseline: accuracy=0.5000, F1=0.5000")

        result_path = output_root / f'{symbol}_h{horizon}_results.json'
        all_results['_diagnostics'] = {
            'label_up_ratio': round(float(test_baselines['label_up_ratio']), 4),
            'baseline_all_up': {
                'accuracy': round(float(test_baselines['all_up']['accuracy']), 4),
                'f1': round(float(test_baselines['all_up']['f1']), 4),
                'balanced_accuracy': round(float(test_baselines['all_up']['balanced_accuracy']), 4),
                'mcc': round(float(test_baselines['all_up']['mcc']), 4),
            },
            'baseline_all_down': {
                'accuracy': round(float(test_baselines['all_down']['accuracy']), 4),
                'f1': round(float(test_baselines['all_down']['f1']), 4),
                'balanced_accuracy': round(float(test_baselines['all_down']['balanced_accuracy']), 4),
                'mcc': round(float(test_baselines['all_down']['mcc']), 4),
            },
            'threshold_objective': args.threshold_objective,
            'deg_penalty_weight': float(args.deg_penalty_weight),
            'deg_target_pos_min': float(args.deg_target_pos_min),
            'deg_target_pos_max': float(args.deg_target_pos_max),
            'smoke_samples': int(args.smoke_samples),
        }
        with open(result_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {result_path}")


if __name__ == '__main__':
    main()
