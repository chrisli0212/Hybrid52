"""
finetune_backbone.py — partial backbone unfreeze on 2026 data
Unfreezes: lstm-equivalent (dw_convs + pw_combine) + output_proj + classifier
Keeps frozen: input_proj + input_norm (feature projection layer)
"""

import argparse
import sys, logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

PRODROOT  = Path("/workspace/Final_production_model")
sys.path.insert(0, str(PRODROOT))

from stage1_models import _build_model_from_ckpt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path("/workspace/data/tier3_binary_2026_duckdb")
HORIZON   = "horizon_30min"
MODEL_DIR = PRODROOT / "models/stage1"
EPOCHS    = 12
LR_HEAD   = 1e-4
LR_BONE   = 1e-5   # 10x smaller for backbone layers
BATCH     = 256
MIN_SAVE_AUC = 0.51
MIN_IMPROVE_AUC = 0.002

# Only fine-tune active agents (gate≈1.0), skip C/T/2D
SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
AGENTS  = [("agentA","A"), ("agentB","B"), ("agentK","K"), ("agentQ","Q")]

log.info(f"Device: {DEVICE}")

def load_data(symbol, data_root: Path, horizon_dir: str):
    d = data_root / symbol / horizon_dir
    X_tr  = np.load(d / "train_sequences.npy").astype(np.float32)
    y_tr  = np.load(d / "train_labels.npy").astype(np.float32)
    X_val = np.load(d / "val_sequences.npy").astype(np.float32)
    y_val = np.load(d / "val_labels.npy").astype(np.float32)
    norm_mean = np.load(d / "norm_mean.npy").astype(np.float32)
    norm_std = np.load(d / "norm_std.npy").astype(np.float32)
    norm_std = np.where(norm_std < 1e-6, 1.0, norm_std).astype(np.float32)
    return X_tr, y_tr, X_val, y_val, norm_mean, norm_std

def make_loaders(X_tr, y_tr, X_val, y_val, batch_size):
    log.info(f"  Train={X_tr.shape}  Val={X_val.shape}  UP%={y_tr.mean()*100:.1f}%")
    tr  = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                     batch_size=batch_size, shuffle=True, pin_memory=True)
    val = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                     batch_size=batch_size, shuffle=False, pin_memory=True)
    return tr, val


def _normalize_batch(Xb, norm_mean_t, norm_std_t):
    Xb = (Xb - norm_mean_t) / torch.clamp(norm_std_t, min=1e-6)
    return torch.clamp(Xb, -10.0, 10.0)


@torch.no_grad()
def _eval_auc(model, loader, norm_mean_t, norm_std_t):
    model.eval()
    scores, labs = [], []
    for Xb, yb in loader:
        Xb = _normalize_batch(Xb.to(DEVICE), norm_mean_t, norm_std_t)
        logit = model(Xb, chain_2d=None)
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labs.append(yb.numpy())
    scores = np.concatenate(scores)
    labs = np.concatenate(labs)
    return float(roc_auc_score(labs, scores))

def get_param_groups(model):
    """
    Frozen:   base.backbone.input_proj / input_norm
    Backbone: base.backbone.dw_convs, pw_combine, ln, pool, output_proj, output_norm
    Head:     base.classifier
    """
    frozen, backbone_params, head_params = [], [], []
    for name, param in model.named_parameters():
        if "input_proj" in name or "input_norm" in name:
            param.requires_grad_(False)
            frozen.append(name)
        elif "classifier" in name:
            param.requires_grad_(True)
            head_params.append(param)
        else:
            param.requires_grad_(True)
            backbone_params.append(param)

    log.info(f"  Frozen: {len(frozen)} params groups")
    log.info(f"  Backbone trainable: {sum(p.numel() for p in backbone_params):,}")
    log.info(f"  Head trainable:     {sum(p.numel() for p in head_params):,}")
    return backbone_params, head_params

def finetune(symbol, agt_name, agt_type, *, data_root: Path, horizon_dir: str, model_dir: Path,
             epochs, lr_backbone, lr_head, batch_size, min_save_auc, min_improve_auc):
    log.info(f"\n{'='*55}")
    log.info(f"  {symbol} / {agt_name}")

    ckpt_path = model_dir / f"{symbol}_{agt_name}.pt"
    if not ckpt_path.exists():
        log.warning(f"  SKIP — not found: {ckpt_path}")
        return

    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = _build_model_from_ckpt(ckpt, agent_type=agt_type, device=DEVICE)

    backbone_params, head_params = get_param_groups(model)
    if not backbone_params and not head_params:
        log.warning("  SKIP — no trainable params")
        return

    try:
        X_tr, y_tr, X_val, y_val, norm_mean, norm_std = load_data(symbol, data_root=data_root, horizon_dir=horizon_dir)
    except FileNotFoundError as e:
        log.warning(f"  SKIP — {e}")
        return

    tr_ld, val_ld = make_loaders(X_tr, y_tr, X_val, y_val, batch_size=batch_size)
    norm_mean_t = torch.from_numpy(norm_mean).to(DEVICE)
    norm_std_t = torch.from_numpy(norm_std).to(DEVICE)

    baseline_auc = _eval_auc(model, val_ld, norm_mean_t, norm_std_t)
    log.info(f"  Baseline val_auc(before finetune)={baseline_auc:.4f}")

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": head_params,     "lr": lr_head, "weight_decay": 1e-4},
    ])
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            Xb = _normalize_batch(Xb, norm_mean_t, norm_std_t)
            optimizer.zero_grad()
            logit = model(Xb, chain_2d=None)
            loss  = criterion(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                backbone_params + head_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        scores, labs = [], []
        with torch.no_grad():
            for Xb, yb in val_ld:
                Xb = _normalize_batch(Xb.to(DEVICE), norm_mean_t, norm_std_t)
                logit = model(Xb, chain_2d=None)
                scores.append(torch.sigmoid(logit).cpu().numpy())
                labs.append(yb.numpy())
        scores = np.concatenate(scores)
        labs   = np.concatenate(labs)
        auc = roc_auc_score(labs, scores)
        acc = ((scores >= 0.5) == labs.astype(bool)).mean() * 100
        log.info(f"  Epoch {epoch}/{epochs}  "
                 f"loss={total_loss/len(tr_ld):.4f}  "
                 f"val_auc={auc:.4f}  val_acc={acc:.2f}%")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    delta_auc = best_auc - baseline_auc
    log.info(f"  Best val_auc(after finetune)={best_auc:.4f}  delta={delta_auc:+.4f}")
    save_floor = max(min_save_auc, baseline_auc + min_improve_auc)
    if best_auc > save_floor:   # save only if better than random floor and baseline by margin
        ckpt["model_state_dict"] = best_state
        ckpt["finetune_val_auc_2026"] = best_auc
        ckpt["finetune_val_auc_2026_baseline"] = baseline_auc
        torch.save(ckpt, str(ckpt_path))
        log.info(f"  ✅ Saved  val_auc={best_auc:.4f} → {ckpt_path.name}")
    else:
        log.warning(
            "  ⚠️  checkpoint NOT overwritten "
            f"(best_auc={best_auc:.4f}, required>{save_floor:.4f}, "
            f"baseline={baseline_auc:.4f}, min_improve={min_improve_auc:.4f})"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partial Stage1 backbone finetune on 1-year dataset")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT),
                        help="Tier3 root containing symbol/horizon_30min tensors")
    parser.add_argument("--horizon-dir", type=str, default=HORIZON,
                        help="Tier3 horizon folder name (default: horizon_30min)")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR),
                        help="Directory containing Stage1 checkpoints (e.g., models/stage1)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr-backbone", type=float, default=LR_BONE)
    parser.add_argument("--lr-head", type=float, default=LR_HEAD)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--min-save-auc", type=float, default=MIN_SAVE_AUC)
    parser.add_argument("--min-improve-auc", type=float, default=MIN_IMPROVE_AUC)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    horizon_dir = args.horizon_dir
    log.info(f"Finetune data_root={data_root} horizon={horizon_dir}")
    log.info(f"Finetune model_dir={model_dir}")

    for sym in SYMBOLS:
        for agt_name, agt_type in AGENTS:
            finetune(
                sym,
                agt_name,
                agt_type,
                data_root=data_root,
                horizon_dir=horizon_dir,
                model_dir=model_dir,
                epochs=args.epochs,
                lr_backbone=args.lr_backbone,
                lr_head=args.lr_head,
                batch_size=args.batch_size,
                min_save_auc=args.min_save_auc,
                min_improve_auc=args.min_improve_auc,
            )
