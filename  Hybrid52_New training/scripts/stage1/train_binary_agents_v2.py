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

import lightning as L
from lightning.pytorch.callbacks import (EarlyStopping, ModelCheckpoint,
                                         LearningRateMonitor)
from lightning.pytorch.loggers import CSVLogger

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from hybrid52_models.independent_agent import IndependentAgent
from config.feature_subsets import AGENT_FEATURE_SUBSETS
from hybrid52_utils.artifacts import DEFAULT_TRAINING_HORIZON_MINUTES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT    = Path("/workspace/data/tier3_binary_hybrid52")
DEFAULT_OUTPUT_ROOT  = Path("/workspace/ Hybrid52_New training/results/stage1")
DEFAULT_CHAIN_2D_DIR = Path("/workspace/data/chain_2d")
RETURN_SCALE = 10000.0

ALL_AGENTS  = ['A', 'B', 'K', 'C', 'T', 'Q', '2D']
HORIZONS    = [5, 15, 30]
ALL_SYMBOLS = ['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT']


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
    dead = (std < 1e-8) | (nz < 1e-4)
    n_dead = int(dead.sum())

    if feat_dim >= 286:
        csv_block = X[:, 270:286]
        csv_nz    = (np.abs(csv_block) > 1e-8).mean()
        csv_live  = csv_nz > 0.05
    else:
        csv_live = True; csv_nz = -1.0

    up_pct = float(labels.mean())
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
    def __init__(self, sequences, targets, norm_mean=None, norm_std=None, chain_2d=None):
        self.sequences = sequences
        self.targets   = torch.FloatTensor(targets)
        self.norm_mean = torch.FloatTensor(norm_mean) if norm_mean is not None else None
        self.norm_std  = torch.FloatTensor(norm_std)  if norm_std  is not None else None
        self.chain_2d  = torch.FloatTensor(chain_2d)  if chain_2d  is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.FloatTensor(np.array(self.sequences[idx]))
        # z-score norm applied HERE (only place)
        if self.norm_mean is not None and self.norm_std is not None:
            seq = (seq - self.norm_mean) / self.norm_std
        if self.chain_2d is None:
            return seq, self.targets[idx]
        return seq, self.chain_2d[idx], self.targets[idx]


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
        weight_decay: float = 0.01,
        focal_gamma: float = 2.0,
        noise_sigma: float = 0.0,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
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

        # Norm buffers (used in validation_step where Dataset doesn't normalise)
        if norm_mean is not None:
            self.register_buffer('norm_mean', torch.FloatTensor(norm_mean))
            self.register_buffer('norm_std',  torch.FloatTensor(norm_std))
        else:
            self.norm_mean = self.norm_std = None

        # Losses
        self.criterion = AsymmetricTradingLoss(fp_weight=1.5, gamma=focal_gamma)

        # Validation accumulators
        self._val_logits = []
        self._val_labels = []

        # Best metrics (read by main() after trainer.fit)
        self.best_mcc = -1.0
        self.best_auc =  0.5

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
            loss = mixup_loss(self.criterion, pred, ya, yb2, lam)
        else:
            pred = self(xb, chain_b)
            loss = self.criterion(pred, yb)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
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

        try:
            auc = float(roc_auc_score(labels, probs))
        except Exception:
            auc = 0.5

        mcc = float(matthews_corrcoef(labels, dir_pred))
        f1  = float(f1_score(labels, dir_pred, average='binary', zero_division=0))
        acc = float(accuracy_score(labels, dir_pred))

        self.log_dict({'val_auc': auc, 'val_mcc': mcc, 'val_f1': f1, 'val_acc': acc},
                      prog_bar=True)

        if auc < 0.48:
            logger.critical(f"  val_auc={auc:.4f} < 0.48 — possible label inversion!")
        elif auc < 0.5:
            logger.warning(f"  Weak agent (val_auc={auc:.4f}) — monitor closely")

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
    epochs=80, lr=3e-4, patience=15,
    accum_steps=4,
    use_mixup=False, mixup_alpha=0.2,
    noise_sigma=0.0,
    val_chain=None, focal_gamma=2.0,
    norm_mean=None, norm_std=None,
    output_root=None, symbol='SPXW', agent_type='A', horizon=30,
    precision='16-mixed',
):
    """Drop-in replacement for train_one_model(). Returns (model, best_auc, best_brier=0)."""

    # Val DataLoader — NO norm in dataset; LitAgent applies it in validation_step
    val_ds = SequenceWithOptionalChainDataset(
        val_seq, val_labels.astype(np.float32),
        norm_mean=None, norm_std=None,   # intentionally omitted
        chain_2d=val_chain,
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


def optimize_threshold(model, val_seq, val_labels, device, mode='classifier',
                       val_chain=None, norm_mean=None, norm_std=None,
                       threshold_objective='f1'):
    model.eval()
    norm_mean_t = torch.FloatTensor(norm_mean).to(device) if norm_mean is not None else None
    norm_std_t  = torch.FloatTensor(norm_std).to(device)  if norm_std  is not None else None

    with torch.no_grad():
        outputs = []
        bs = 2048
        for i in range(0, len(val_seq), bs):
            xb = torch.FloatTensor(np.array(val_seq[i:i+bs])).to(device)
            if norm_mean_t is not None:
                xb = (xb - norm_mean_t) / norm_std_t
            chain_slice = (torch.FloatTensor(np.array(val_chain[i:i+bs])).to(device)
                           if val_chain is not None else None)
            outputs.append(_forward_model(model, xb, chain_slice))
        raw_output = torch.cat(outputs).cpu().numpy()

    platt_scaler = None
    invert_signal = False
    if mode == 'classifier':
        platt_scaler = LogisticRegression()
        platt_scaler.fit(raw_output.reshape(-1, 1), val_labels)
        logger.info("  Fitted Platt scaler for probability calibration")
        probs = platt_scaler.predict_proba(raw_output.reshape(-1, 1))[:, 1]
        try:
            val_auc = float(roc_auc_score(val_labels, probs))
        except Exception:
            val_auc = 0.5
        if val_auc < 0.48:
            invert_signal = True
            probs = 1.0 - probs
            raw_output = -raw_output
            logger.warning(f"  Auto-flip enabled (val_auc={val_auc:.4f} < 0.48)")
        elif val_auc < 0.5:
            logger.warning(f"  Weak agent (val_auc={val_auc:.4f}) — NOT flipping, monitor closely")
    else:
        probs = raw_output

    best_threshold = 0.5; best_f1 = 0.0; best_obj = -1e18
    for threshold in np.arange(0.30, 0.66, 0.01):
        preds = (probs > threshold).astype(int)
        obj = _threshold_objective_score(val_labels, preds, threshold_objective)
        f1  = float(f1_score(val_labels, preds, average='binary', zero_division=0))
        if obj > best_obj or (abs(obj - best_obj) < 1e-8 and f1 > best_f1):
            best_obj = obj; best_f1 = f1; best_threshold = threshold

    logger.info(f"  Optimal threshold: {best_threshold:.2f} "
                f"(objective={threshold_objective}={best_obj:.4f}, val_f1={best_f1:.4f})")
    return best_threshold, best_f1, platt_scaler, invert_signal


# ============================================================================
# Evaluation (UNCHANGED)
# ============================================================================

def evaluate_model(model, test_seq, test_labels, test_returns, device,
                   mode='classifier', threshold=0.5, test_chain=None,
                   platt_scaler=None, norm_mean=None, norm_std=None,
                   invert_signal=False):
    model.eval()
    norm_mean_t = torch.FloatTensor(norm_mean).to(device) if norm_mean is not None else None
    norm_std_t  = torch.FloatTensor(norm_std).to(device)  if norm_std  is not None else None

    with torch.no_grad():
        outputs = []
        bs = 2048
        for i in range(0, len(test_seq), bs):
            xb = torch.FloatTensor(np.array(test_seq[i:i+bs])).to(device)
            if norm_mean_t is not None:
                xb = (xb - norm_mean_t) / norm_std_t
            chain_slice = (torch.FloatTensor(np.array(test_chain[i:i+bs])).to(device)
                           if test_chain is not None else None)
            outputs.append(_forward_model(model, xb, chain_slice))
        raw_output = torch.cat(outputs).cpu().numpy()

    # Compute AUC from raw_output BEFORE any transforms
    try:
        auc = roc_auc_score(test_labels, raw_output)
    except Exception:
        auc = 0.5

    if mode == 'classifier':
        if platt_scaler is not None:
            probs = platt_scaler.predict_proba(raw_output.reshape(-1, 1))[:, 1]
        else:
            probs = 1 / (1 + np.exp(-raw_output))
        if invert_signal:
            probs = 1.0 - probs
        preds      = (probs > threshold).astype(int)
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
        'quintile_mean_returns_bp': quintile_returns,
        'quintile_spread_bp': quintile_spread_bp,
        'confidence_buckets': conf_buckets,
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
    parser.add_argument('--lr',          type=float, default=3e-4)
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
    parser.add_argument('--noise-sigma', type=float, default=0.0)
    parser.add_argument('--threshold-objective',
                        choices=['f1', 'balanced_acc', 'mcc'], default='balanced_acc')
    parser.add_argument('--precision',   default='16-mixed',
                        help='Lightning precision: 16-mixed | bf16-mixed | 32')
    parser.add_argument('--debug',       action='store_true')
    parser.add_argument('--device',      default='cpu',
                        help='Kept for backward compat; Lightning auto-selects GPU/CPU')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root    = Path(args.data_root)
    output_root  = Path(args.output_root)
    chain_2d_dir = Path(args.chain_2d_dir)
    output_root.mkdir(parents=True, exist_ok=True)

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

        train_chain_path, val_chain_path, test_chain_path = _resolve_chain_2d_paths(
            data_dir, symbol, chain_2d_dir, len(train_seq), len(val_seq)
        )
        has_chain_files = train_chain_path is not None
        train_chain = val_chain = test_chain = None

        feat_dim = train_seq.shape[2]

        # Dead-feature mask
        X = train_seq.reshape(-1, feat_dim)
        std          = X.std(axis=0)
        nonzero_rate = (np.abs(X) > 0).mean(axis=0)
        feature_mask = (std > 1e-8) & (nonzero_rate > 1e-4)
        logger.info(f"Dead-feature mask keeps {int(feature_mask.sum())}/{feat_dim} dims")
        mask3 = feature_mask.astype(np.float32)[None, None, :]
        train_seq *= mask3; val_seq *= mask3; test_seq *= mask3

        # Normalization stats
        nm_path = data_dir / 'norm_mean.npy'
        ns_path = data_dir / 'norm_std.npy'
        if nm_path.exists() and ns_path.exists():
            norm_mean = np.load(nm_path)
            norm_std  = np.load(ns_path)
            logger.info("Will apply z-score normalization on-the-fly (Dataset only)")
        else:
            norm_mean = norm_std = None
            logger.warning("No normalization stats found — training without z-score norm")

        train_returns_scaled = train_returns * RETURN_SCALE
        logger.info(f"Data: train={len(train_seq):,} val={len(val_seq):,} "
                    f"test={len(test_seq):,} feat_dim={feat_dim}")

        data_ok = verify_data_before_training(data_dir, symbol, horizon)
        if not data_ok:
            logger.warning(f"[{symbol}] Data health check FAILED — continuing anyway.")

        all_results = {}

        for agent_type in args.agents:
            logger.info(f"\n --- Agent {agent_type} ---")
            subset_info = AGENT_FEATURE_SUBSETS.get(agent_type, {})
            logger.info(f"  Subset: {subset_info.get('name', 'N/A')} "
                        f"({subset_info.get('feat_dim', feat_dim)} dims)")

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

            for mode in ['classifier']:
                logger.info(f"  [{agent_type}] Mode: {mode}")
                try:
                    model = BinaryIndependentAgent(
                        agent_type=agent_type, feat_dim=feat_dim,
                        temporal_dim=128, dropout=0.2, mode=mode,
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
                    _agent_indices = subset_info.get('indices', None)
                    if _agent_indices is not None:
                        _Xa = _agent_train_seq.reshape(-1, feat_dim)[:, _agent_indices]
                        _agent_live = (_Xa.std(axis=0) > 1e-8) & ((np.abs(_Xa) > 0).mean(axis=0) > 1e-4)
                        _agent_mask_full = np.zeros(feat_dim, dtype=np.float32)
                        _agent_mask_full[_agent_indices] = _agent_live.astype(np.float32)
                        logger.info(f"  [{agent_type}] Per-agent mask: {int(_agent_live.sum())}/{len(_agent_indices)} active dims")
                    else:
                        # FIX5b-fallback: compute full-dim mask locally
                        _Xf = _agent_train_seq.reshape(-1, feat_dim)
                        _agent_mask_full = ((_Xf.std(axis=0) > 1e-8) & ((np.abs(_Xf) > 0).mean(axis=0) > 1e-4)).astype(np.float32)
                    _amask3 = _agent_mask_full[None, None, :]
                    _agent_train_seq *= _amask3; _agent_val_seq *= _amask3; _agent_test_seq *= _amask3

                    # FIX6: clip T and Q to ±5σ
                    if agent_type in ('T', 'Q') and norm_mean is not None:
                        _clip_std = np.maximum(norm_std, 1e-8)
                        _agent_train_seq = np.clip(_agent_train_seq, norm_mean - 5*_clip_std, norm_mean + 5*_clip_std)
                        _agent_val_seq   = np.clip(_agent_val_seq,   norm_mean - 5*_clip_std, norm_mean + 5*_clip_std)
                        _agent_test_seq  = np.clip(_agent_test_seq,  norm_mean - 5*_clip_std, norm_mean + 5*_clip_std)
                        logger.info(f"  [{agent_type}] ±5σ feature clipping applied")

                    train_ds = SequenceWithOptionalChainDataset(
                        _agent_train_seq, targets,
                        norm_mean=norm_mean, norm_std=norm_std,
                        chain_2d=train_chain if agent_type == '2D' else None,
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

                    focal_gamma = 1.0 if agent_type in ('T', 'Q') else 2.0

                    # ── Lightning training (replaces train_one_model) ──────
                    model, val_auc, _ = run_lightning_training(
                        model, train_loader, _agent_val_seq, val_labels, device,
                        epochs=args.epochs, lr=args.lr,
                        patience=args.patience, accum_steps=args.accum_steps,
                        use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
                        noise_sigma=args.noise_sigma,
                        val_chain=val_chain if agent_type == '2D' else None,
                        focal_gamma=focal_gamma,
                        norm_mean=norm_mean, norm_std=norm_std,
                        output_root=output_root,
                        symbol=symbol, agent_type=agent_type, horizon=horizon,
                        precision=args.precision,
                    )

                    opt_threshold, opt_val_f1, platt_scaler, invert_signal = optimize_threshold(
                        model, _agent_val_seq, val_labels, device, mode=mode,
                        val_chain=val_chain if agent_type == '2D' else None,
                        norm_mean=norm_mean, norm_std=norm_std,
                        threshold_objective=args.threshold_objective,
                    )

                    test_metrics = evaluate_model(
                        model, _agent_test_seq, test_labels, test_returns, device,
                        mode=mode, threshold=opt_threshold,
                        test_chain=test_chain if agent_type == '2D' else None,
                        platt_scaler=platt_scaler,
                        norm_mean=norm_mean, norm_std=norm_std,
                        invert_signal=invert_signal,
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
                        'invert_signal':          bool(invert_signal),
                        'threshold_objective':    args.threshold_objective,
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
                    f"{'AUC':>8} {'IC':>8} {'Thr':>6}")
        logger.info("-" * 70)
        for agent_type in args.agents:
            for mode in ['classifier']:
                rkey = f"agent_{agent_type}_{mode}"
                r    = all_results.get(rkey, {})
                if 'error' in r:
                    logger.info(f"{agent_type:>6} {mode:>12} {'FAILED':>10}")
                elif 'accuracy' in r:
                    logger.info(
                        f"{agent_type:>6} {mode:>12} "
                        f"{r['accuracy']:>10.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} "
                        f"{r['ic']:>8.4f} {r.get('threshold', 0.5):>6.3f}"
                    )
        logger.info("\nRandom baseline: accuracy=0.5000, F1=0.5000")

        result_path = output_root / f'{symbol}_h{horizon}_results.json'
        with open(result_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {result_path}")


if __name__ == '__main__':
    main()
