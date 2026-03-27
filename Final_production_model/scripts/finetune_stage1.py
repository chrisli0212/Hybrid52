"""
finetune_stage1.py — head-only fine-tune on 2026 data
"""

import argparse
import sys, logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

BASE = Path("/workspace/Final_production_model")
sys.path.insert(0, str(BASE))

from stage1_models import _build_model_from_ckpt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path("/workspace/data/tier3_binary_2026_duckdb")
HORIZON   = "horizon_30min"
MODEL_DIR = BASE / "models/stage1"
EPOCHS    = 3
LR        = 1e-4
BATCH     = 256

SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
AGENTS  = [
    ("agentA", "A"),
    ("agentB", "B"),
    ("agentC", "C"),
    ("agentK", "K"),
    ("agentT", "T"),
    ("agentQ", "Q"),
]

log.info(f"Device: {DEVICE}")

def load_data(symbol: str, data_root: Path, horizon_dir: str):
    """Load pre-split train/val sequences and labels."""
    d = data_root / symbol / horizon_dir
    X_tr  = np.load(d / "train_sequences.npy").astype(np.float32)
    y_tr  = np.load(d / "train_labels.npy").astype(np.float32)
    X_val = np.load(d / "val_sequences.npy").astype(np.float32)
    y_val = np.load(d / "val_labels.npy").astype(np.float32)
    return X_tr, y_tr, X_val, y_val

def make_loaders(X_tr, y_tr, X_val, y_val):
    log.info(f"  Train: {X_tr.shape}  Val: {X_val.shape}  UP%={y_tr.mean()*100:.1f}%")
    tr_ld = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                       batch_size=BATCH, shuffle=True, pin_memory=True)
    val_ld = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                        batch_size=BATCH, shuffle=False, pin_memory=True)
    return tr_ld, val_ld

def finetune_agent(symbol: str, agt_name: str, agt_type: str, data_root: Path, horizon_dir: str, model_dir: Path):
    log.info(f"\n{'='*55}")
    log.info(f"  {symbol} / {agt_name}")

    ckpt_path = model_dir / f"{symbol}_{agt_name}.pt"
    if not ckpt_path.exists():
        log.warning(f"  SKIP — checkpoint not found: {ckpt_path}")
        return None

    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = _build_model_from_ckpt(ckpt, agent_type=agt_type, device=DEVICE)
    model.train()

    # Freeze all, then unfreeze only base.classifier
    for p in model.parameters():
        p.requires_grad_(False)
    unfrozen = 0
    for name, param in model.named_parameters():
        if name.startswith("base.classifier."):
            param.requires_grad_(True)
            unfrozen += param.numel()

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"  Frozen: {frozen:,} | Trainable heads: {unfrozen:,}")

    if unfrozen == 0:
        log.warning(f"  SKIP — no trainable params")
        return None

    # Load data
    try:
        X_tr, y_tr, X_val, y_val = load_data(symbol, data_root=data_root, horizon_dir=horizon_dir)
    except FileNotFoundError as e:
        log.warning(f"  SKIP — {e}")
        return None

    tr_ld, val_ld = make_loaders(X_tr, y_tr, X_val, y_val)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logit = model(Xb, chain_2d=None)
            loss  = criterion(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for Xb, yb in val_ld:
                logit = model(Xb.to(DEVICE), chain_2d=None)
                scores.append(torch.sigmoid(logit).cpu().numpy())
                labels.append(yb.numpy())
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        val_auc = roc_auc_score(labels, scores)
        val_acc = ((scores >= 0.5) == labels.astype(bool)).mean() * 100
        log.info(f"  Epoch {epoch}/{EPOCHS}  loss={total_loss/len(tr_ld):.4f}  "
                 f"val_auc={val_auc:.4f}  val_acc={val_acc:.2f}%")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Save — preserve all original ckpt keys
    ckpt["model_state_dict"] = best_state
    ckpt["finetune_val_auc_2026"] = best_auc
    torch.save(ckpt, str(ckpt_path))
    log.info(f"  ✅ Saved  val_auc={best_auc:.4f}  → {ckpt_path.name}")
    return best_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head-only Stage1 finetune")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT),
                        help="Tier3 root containing symbol/horizon_30min tensors")
    parser.add_argument("--horizon-dir", type=str, default=HORIZON,
                        help="Tier3 horizon folder name (default: horizon_30min)")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR),
                        help="Directory containing Stage1 checkpoints (e.g., models/stage1)")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    horizon_dir = args.horizon_dir
    model_dir = Path(args.model_dir)
    log.info(f"Finetune data_root={data_root} horizon={horizon_dir}")
    log.info(f"Finetune model_dir={model_dir}")

    for sym in SYMBOLS:
        for agt_name, agt_type in AGENTS:
            finetune_agent(sym, agt_name, agt_type, data_root=data_root, horizon_dir=horizon_dir, model_dir=model_dir)
