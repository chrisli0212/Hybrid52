#!/usr/bin/env python3
"""Train Agent VIX warm-start checkpoint on tier3_vix_v4 data."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from hybrid52_models.agents import AgentVIX


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VIXDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _sampler(y: np.ndarray, n_cls: int) -> WeightedRandomSampler:
    cnt = np.bincount(y, minlength=n_cls).astype(np.float64)
    w = 1.0 / np.maximum(cnt, 1.0)
    sw = w[y]
    return WeightedRandomSampler(weights=torch.from_numpy(sw).double(), num_samples=len(y), replacement=True)


@torch.no_grad()
def _eval(model: AgentVIX, loader: DataLoader, crit: nn.Module, device: torch.device):
    model.eval()
    total = 0.0
    n = 0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        s, c, logits = model(x[:, -1, :], x[:, -1, :], x)
        del s, c
        loss = crit(logits, y)
        total += float(loss.item()) * len(y)
        n += len(y)
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return total / max(n, 1), acc, f1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/workspace/data/tier3_vix_v4/VIXW")
    p.add_argument("--output-root", default="/workspace/ Hybrid52_New training/results/stage1_vix")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    t0 = time.time()
    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tr_x = np.load(data_root / "train_vix_features.npy")
    tr_y = np.load(data_root / "train_vix_labels.npy")
    va_x = np.load(data_root / "val_vix_features.npy")
    va_y = np.load(data_root / "val_vix_labels.npy")
    te_x = np.load(data_root / "test_vix_features.npy")
    te_y = np.load(data_root / "test_vix_labels.npy")

    n_cls = int(max(tr_y.max(), va_y.max(), te_y.max()) + 1)
    feat_dim = tr_x.shape[-1]

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    logger.info("Device: %s", device)

    tr_loader = DataLoader(VIXDataset(tr_x, tr_y), batch_size=args.batch_size, sampler=_sampler(tr_y, n_cls), num_workers=2, pin_memory=True)
    va_loader = DataLoader(VIXDataset(va_x, va_y), batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    te_loader = DataLoader(VIXDataset(te_x, te_y), batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    model = AgentVIX(vix_feat_dim=feat_dim, num_regimes=n_cls).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        tp, tl = [], []

        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            _, _, logits = model(x[:, -1, :], x[:, -1, :], x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * len(y)
            n += len(y)
            tp.extend(logits.argmax(dim=1).detach().cpu().numpy())
            tl.extend(y.detach().cpu().numpy())

        tr_loss = total / max(n, 1)
        tr_acc = float(accuracy_score(tl, tp))
        tr_f1 = float(f1_score(tl, tp, average="macro", zero_division=0))

        va_loss, va_acc, va_f1 = _eval(model, va_loader, crit, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1, "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1})

        logger.info("Ep %3d | train loss=%.4f acc=%.4f f1=%.4f | val loss=%.4f acc=%.4f f1=%.4f", epoch, tr_loss, tr_acc, tr_f1, va_loss, va_acc, va_f1)

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc, te_f1 = _eval(model, te_loader, crit, device)

    ckpt = out_root / "vix_agent_best.pt"
    torch.save(model.state_dict(), ckpt)

    res = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test_loss": te_loss,
        "test_acc": te_acc,
        "test_f1_macro": te_f1,
        "feat_dim": feat_dim,
        "num_regimes": n_cls,
        "params": model.count_parameters(),
        "train_size": int(len(tr_y)),
        "val_size": int(len(va_y)),
        "test_size": int(len(te_y)),
        "train_time_sec": round(time.time() - t0, 1),
        "history": history,
    }
    (out_root / "vix_agent_results.json").write_text(json.dumps(res, indent=2))
    logger.info("Saved warm-start checkpoint: %s", ckpt)


if __name__ == "__main__":
    main()
