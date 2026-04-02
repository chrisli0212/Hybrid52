#!/usr/bin/env python3
"""
Audit Stage1 checkpoint normalization compatibility against configured tier3 root.

Checks per (symbol, agent):
- checkpoint has norm_mean/norm_std
- filesystem norm files exist under tier3 root
- lengths match
- max/mean absolute difference
- invert_signal flag visibility
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
AGENTS = ["A", "B", "C", "K", "T", "Q", "2D"]


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _arr_or_none(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    try:
        return np.asarray(v, dtype=np.float32).reshape(-1)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Audit Stage1 checkpoint-vs-tier3 norm compatibility")
    p.add_argument("--config", default="/workspace/Final_production_model/config/production_config.json")
    p.add_argument("--model-root", default="/workspace/Final_production_model/models/stage1")
    p.add_argument("--tier3-root", default=None, help="Override tier3 root (else from config)")
    p.add_argument("--horizon", type=int, default=None, help="Override horizon (else from config)")
    args = p.parse_args()

    cfg = _load_config(Path(args.config))
    horizon = int(args.horizon or cfg.get("model_info", {}).get("horizon_minutes", 30))
    tier3_root = Path(args.tier3_root or cfg.get("data_paths", {}).get("tier3_binary_root", ""))
    model_root = Path(args.model_root)
    hdir = f"horizon_{horizon}min"

    print(f"tier3_root={tier3_root}")
    print(f"model_root={model_root}")
    print(f"horizon={hdir}")

    total = 0
    mismatch = 0
    missing_fs = 0
    missing_ckpt_norm = 0
    invert_true = 0

    for sym in SYMBOLS:
        fs_nm_path = tier3_root / sym / hdir / "norm_mean.npy"
        fs_ns_path = tier3_root / sym / hdir / "norm_std.npy"
        fs_nm = np.load(fs_nm_path) if fs_nm_path.exists() else None
        fs_ns = np.load(fs_ns_path) if fs_ns_path.exists() else None
        if fs_nm is None or fs_ns is None:
            missing_fs += 1

        for ag in AGENTS:
            total += 1
            ckpt_path = model_root / f"{sym}_agent{ag}.pt"
            if not ckpt_path.exists():
                print(f"{sym}/{ag}: missing checkpoint")
                continue

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            inv = bool(ckpt.get("invert_signal", False))
            if inv:
                invert_true += 1
            ck_nm = _arr_or_none(ckpt.get("norm_mean"))
            ck_ns = _arr_or_none(ckpt.get("norm_std"))
            if ck_nm is None or ck_ns is None:
                missing_ckpt_norm += 1
                print(f"{sym}/{ag}: ckpt norms missing | invert_signal={inv}")
                continue

            if fs_nm is None or fs_ns is None:
                print(f"{sym}/{ag}: fs norms missing | ckpt_len={len(ck_nm)} invert_signal={inv}")
                continue

            n = min(len(ck_nm), len(fs_nm))
            mean_abs_nm = float(np.mean(np.abs(ck_nm[:n] - fs_nm[:n])))
            mean_abs_ns = float(np.mean(np.abs(ck_ns[:n] - fs_ns[:n])))
            len_ok = (len(ck_nm) == len(fs_nm)) and (len(ck_ns) == len(fs_ns))
            same = len_ok and (mean_abs_nm < 1e-8) and (mean_abs_ns < 1e-8)
            if not same:
                mismatch += 1
            print(
                f"{sym}/{ag}: len_ckpt={len(ck_nm)} len_fs={len(fs_nm)} "
                f"mean_abs_nm={mean_abs_nm:.6g} mean_abs_ns={mean_abs_ns:.6g} "
                f"invert_signal={inv} match={same}"
            )

    print("\nSUMMARY")
    print(f"total_pairs={total}")
    print(f"mismatch_pairs={mismatch}")
    print(f"missing_fs_symbol_groups={missing_fs}")
    print(f"missing_ckpt_norm_pairs={missing_ckpt_norm}")
    print(f"invert_signal_true_pairs={invert_true}")


if __name__ == "__main__":
    main()
