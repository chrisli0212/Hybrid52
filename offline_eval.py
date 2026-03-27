"""
Offline Stage-1 accuracy evaluation.
Uses production .pt checkpoints + freshly computed norm files
against test_sequences / test_labels from tier3_binary_2026_v1.
"""
import sys, os
sys.path.insert(0, "/workspace/Final_production_model")

import numpy as np
import torch
from pathlib import Path
from stage1_models import _build_model_from_ckpt

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR  = Path("/workspace/Final_production_model/models/stage1")
DATA_ROOT  = Path("/workspace/data/tier3_binary_2026_v1")
NORM_ROOT  = Path("/workspace/data/tier3_binary_v5")   # freshly synced norms
HORIZON    = "horizon_30min"
AGENTS     = ["agentA", "agentB", "agentC", "agentK", "agentQ", "agentT", "agent2D"]
SYMBOLS    = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
BATCH_SIZE = 512

def evaluate_agent(sym, agent_name):
    ckpt_path = MODEL_DIR / f"{sym}_{agent_name}.pt"
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = _build_model_from_ckpt(ckpt, agent_type=agent_name.replace("agent",""), device=DEVICE, symbol=sym)
    model.eval()

    # Load norm
    nm = torch.tensor(np.load(NORM_ROOT / sym / HORIZON / "norm_mean.npy"), device=DEVICE)
    ns = torch.tensor(np.load(NORM_ROOT / sym / HORIZON / "norm_std.npy"),  device=DEVICE)

    # Load test split
    d = DATA_ROOT / sym / HORIZON
    X  = np.load(d / "test_sequences.npy")   # (N, 20, 325)
    y  = np.load(d / "test_labels.npy")       # (N,)

    # Load chain_2d for agent2D
    chain = None
    if agent_name == "agent2D":
        c = np.load(d / "test_chain_2d.npy")  # (N, 5, 20, 20)
        chain = torch.tensor(c, dtype=torch.float32)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    # Normalize
    X_t = (X_t - nm) / torch.clamp(ns, min=1e-6)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(X_t), BATCH_SIZE):
            xb = X_t[i:i+BATCH_SIZE].to(DEVICE)
            cb = chain[i:i+BATCH_SIZE].to(DEVICE) if chain is not None else None
            logits = model(xb, chain_2d=cb)
            probs  = torch.sigmoid(logits).cpu()
            preds  = (probs >= 0.5).long()
            all_preds.append(preds)
            all_probs.append(probs)

    preds = torch.cat(all_preds)
    probs = torch.cat(all_probs)
    acc   = (preds == y_t).float().mean().item()

    # High-confidence subset (prob >= 0.6 or <= 0.4)
    conf_mask = (probs >= 0.6) | (probs <= 0.4)
    conf_acc  = (preds[conf_mask] == y_t[conf_mask]).float().mean().item() if conf_mask.sum() > 0 else float('nan')
    conf_pct  = conf_mask.float().mean().item()

    return acc, conf_acc, conf_pct, len(y)

print(f"\n{'Symbol':<8} {'Agent':<10} {'N_test':>8} {'Acc':>8} {'ConfAcc':>10} {'Conf%':>8}")
print("-" * 60)
for sym in SYMBOLS:
    for agent in AGENTS:
        res = evaluate_agent(sym, agent)
        if res is None:
            continue
        acc, cacc, cpct, n = res
        flag = "  <<<" if acc >= 0.56 else ""
        print(f"{sym:<8} {agent:<10} {n:>8,} {acc:>8.1%} {cacc:>10.1%} {cpct:>8.1%}{flag}")
    print()

