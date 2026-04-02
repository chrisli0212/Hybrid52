"""
LightningModule wrapper around BinaryIndependentAgent.
Drops into train_binary_agents_v2.py with minimal changes:
  - replace manual train loop with trainer.fit()
  - keeps ALL existing loss / metric / threshold logic unchanged
"""
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

# Re-use existing loss classes from trainer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.stage1.train_binary_agents_v2 import (
    BinaryFocalLoss, SoftF1Loss, AsymmetricTradingLoss,
    SequenceWithOptionalChainDataset, add_gaussian_noise, mixup_sequences, mixup_loss
)


class LitAgent(L.LightningModule):
    def __init__(
        self,
        model,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        f1_weight: float = 0.3,
        focal_gamma: float = 2.0,
        positive_class_prior: float = 0.52,
        noise_sigma: float = 0.0,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        norm_mean=None,
        norm_std=None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.f1_weight = f1_weight
        self.noise_sigma = noise_sigma
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # Register norm buffers
        if norm_mean is not None:
            self.register_buffer('norm_mean', torch.FloatTensor(norm_mean))
            self.register_buffer('norm_std',  torch.FloatTensor(norm_std))
        else:
            self.norm_mean = self.norm_std = None

        # Losses - use symmetric loss to avoid prediction bias
        self.focal_loss = AsymmetricTradingLoss(fp_weight=1.0, gamma=focal_gamma)
        self.soft_f1    = SoftF1Loss()

        # Validation tracking
        self._val_preds  = []
        self._val_labels = []
        self.best_mcc  = -1.0
        self.best_auc  = 0.5

    def _normalize(self, x):
        if self.norm_mean is not None:
            return (x - self.norm_mean) / self.norm_std
        return x

    def forward(self, x, chain=None):
        return self.model(x, chain_2d=chain)

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
            loss = mixup_loss(self.focal_loss, pred, ya, yb2, lam) + \
                   self.f1_weight * mixup_loss(self.soft_f1, pred, ya, yb2, lam)
        else:
            pred = self(xb, chain_b)
            loss = self.focal_loss(pred, yb) + self.f1_weight * self.soft_f1(pred, yb)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            xb, chain_b, yb = batch
        else:
            xb, yb = batch; chain_b = None
        xb = self._normalize(xb)
        logits = self(xb, chain_b)
        self._val_preds.append(logits.cpu())
        self._val_labels.append(yb.cpu())

    def on_validation_epoch_end(self):
        logits = torch.cat(self._val_preds).numpy()
        labels = torch.cat(self._val_labels).numpy().astype(int)
        self._val_preds.clear(); self._val_labels.clear()

        probs    = 1 / (1 + np.exp(-logits))
        dir_pred = (probs > 0.5).astype(int)

        try:
            auc = float(roc_auc_score(labels, probs))
        except:
            auc = 0.5
        mcc = float(matthews_corrcoef(labels, dir_pred))
        f1  = float(f1_score(labels, dir_pred, average='binary', zero_division=0))

        self.log_dict({'val_auc': auc, 'val_mcc': mcc, 'val_f1': f1}, prog_bar=True)

        # Track best (MCC primary, AUC secondary)
        if mcc > self.best_mcc + 0.005:
            self.best_mcc = mcc; self.best_auc = auc
        elif auc > self.best_auc + 0.002:
            self.best_auc = auc

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2, eta_min=1e-6
        )
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}
