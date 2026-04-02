#!/usr/bin/env python3
"""
Compare Stage-1 test metrics when applying different Tier3 norm files to the SAME raw sequences.

Typical use:
  - Sequences from: /workspace/data/tier3_binary_2026_duckdb/SPXW/horizon_30min
  - "Same-build" norm: norm_mean/std in that folder (CORRECT for those sequences)
  - "Foreign" norm: e.g. /workspace/data/tier3_binary_v5/SPXW/horizon_30min (WRONG if applied to 2026 sequences)

This does NOT retrain; it only evaluates an existing checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
STAGE1 = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE1))

from hybrid55_utils.artifacts import DEFAULT_TRAINING_HORIZON_MINUTES
from train_binary_agents_v2 import BinaryIndependentAgent, evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _dead_feature_mask(train_seq: np.ndarray, feat_dim: int) -> np.ndarray:
    X = train_seq.reshape(-1, feat_dim)
    eps_std = 1e-5
    min_nonzero = 1e-4
    std = X.std(axis=0)
    nonzero_rate = (np.abs(X) > 0).mean(axis=0)
    return (std > eps_std) & (nonzero_rate > min_nonzero)


def _load_platt(ckpt: dict) -> LogisticRegression | None:
    if "platt_scaler_coef" not in ckpt or "platt_scaler_intercept" not in ckpt:
        return None
    lr = LogisticRegression()
    lr.coef_ = np.asarray(ckpt["platt_scaler_coef"], dtype=np.float64)
    lr.intercept_ = np.asarray(ckpt["platt_scaler_intercept"], dtype=np.float64).ravel()
    lr.classes_ = np.array([0, 1], dtype=np.int64)
    return lr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True, help="Tier3 root (e.g. tier3_binary_2026_duckdb)")
    p.add_argument("--symbol", type=str, default="SPXW")
    p.add_argument("--horizon", type=int, default=DEFAULT_TRAINING_HORIZON_MINUTES)
    p.add_argument(
        "--compare-norm-root",
        type=str,
        default=None,
        help="Optional second Tier3 root to load foreign norm_mean/std (e.g. tier3_binary_v5)",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_root) / args.symbol / f"horizon_{args.horizon}min"
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent_type = ckpt.get("agent_type", "A")
    use_subset = ckpt.get("feature_subset", True)
    use_attn_bb = ckpt.get("use_attention_backbone", False)
    use_attn_pool = ckpt.get("use_attention_pool", False)

    train_seq = np.load(data_dir / "train_sequences.npy")
    test_seq = np.load(data_dir / "test_sequences.npy")
    test_labels = np.load(data_dir / "test_labels.npy")
    test_returns = np.load(data_dir / "test_returns.npy")
    feat_dim = train_seq.shape[2]

    mask = _dead_feature_mask(train_seq, feat_dim).astype(np.float32)
    m3 = mask[None, None, :]
    test_seq = test_seq * m3

    test_chain = None
    if agent_type == "2D":
        tcp = data_dir / "test_chain_2d.npy"
        if not tcp.exists():
            raise FileNotFoundError("Agent 2D requires test_chain_2d.npy")
        test_chain = np.load(tcp)

    nm_same = np.load(data_dir / "norm_mean.npy")
    ns_same = np.load(data_dir / "norm_std.npy")

    foreign_dir = None
    nm_foreign = ns_foreign = None
    if args.compare_norm_root:
        foreign_dir = Path(args.compare_norm_root) / args.symbol / f"horizon_{args.horizon}min"
        if foreign_dir.is_dir():
            nm_foreign = np.load(foreign_dir / "norm_mean.npy")
            ns_foreign = np.load(foreign_dir / "norm_std.npy")
        else:
            logger.warning("compare-norm path missing: %s", foreign_dir)

    model = BinaryIndependentAgent(
        agent_type=agent_type,
        feat_dim=feat_dim,
        temporal_dim=128,
        dropout=0.2,
        mode="classifier",
        use_feature_subset=use_subset,
        use_attention_backbone=use_attn_bb,
        use_attention_pool=use_attn_pool,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    threshold = float(ckpt.get("optimal_threshold", 0.5))
    invert_signal = bool(ckpt.get("invert_signal", False))
    platt = _load_platt(ckpt)

    def run_tag(norm_mean: np.ndarray | None, norm_std: np.ndarray | None, tag: str) -> dict:
        metrics = evaluate_model(
            model,
            test_seq,
            test_labels,
            test_returns,
            device,
            mode="classifier",
            threshold=threshold,
            test_chain=test_chain,
            platt_scaler=platt,
            norm_mean=norm_mean,
            norm_std=norm_std,
            invert_signal=invert_signal,
        )
        metrics["tag"] = tag
        return metrics

    results = []
    results.append(run_tag(nm_same, ns_same, f"norm_from_same_build:{data_dir}"))

    if nm_foreign is not None and ns_foreign is not None:
        results.append(
            run_tag(nm_foreign, ns_foreign, f"norm_from_foreign:{foreign_dir}")
        )

    print("\n" + "=" * 80)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Eval split:  test | n={len(test_labels):,} | symbol={args.symbol} h={args.horizon}")
    print(f"Sequences + dead-feature mask from: {data_dir}")
    print("=" * 80)
    for r in results:
        bacc = r.get("balanced_accuracy", None)
        bacc_s = f"{bacc:.4f}" if bacc is not None else "n/a"
        print(
            f"\n[{r['tag']}]"
            f"\n  accuracy={r['accuracy']:.4f}  balanced_accuracy={bacc_s}"
            f"\n  f1={r['f1']:.4f}  auc={r['auc']:.4f}  thr={r['threshold']:.4f}  invert={invert_signal}"
        )
    print("=" * 80 + "\n")

    out = {
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_dir),
        "compare_norm_dir": str(foreign_dir) if foreign_dir else None,
        "rows": results,
    }
    out_path = data_dir / f"eval_norm_compare_{agent_type}_h{args.horizon}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
