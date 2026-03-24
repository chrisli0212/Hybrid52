#!/usr/bin/env python3
"""
split_chain_2d.py  —  Split a monolithic chain_2d .npy into train/val/test.

Run this ONCE after build_chain_2d.py finishes, before starting Stage 1 training.

Usage:
    python scripts/split_chain_2d.py \\
        --src  /workspace/data/chain_2d/SPXW_chain_2d_train.npy \\
        --dst  /workspace/data/tier3_binary_v5/SPXW/horizon_30min \\
        --train 0.70 --val 0.15

Outputs (written to --dst):
    train_chain_2d.npy
    val_chain_2d.npy
    test_chain_2d.npy
"""

import argparse
import sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src',   required=True, help='Monolithic .npy from build_chain_2d.py')
    p.add_argument('--dst',   required=True, help='Destination dir (tier3 horizon dir)')
    p.add_argument('--train', type=float, default=0.70, help='Train fraction (default 0.70)')
    p.add_argument('--val',   type=float, default=0.15, help='Val fraction   (default 0.15)')
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"ERROR: source file not found: {src}", file=sys.stderr)
        print("Run build_chain_2d.py first.", file=sys.stderr)
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src} ...")
    batch = np.load(src)
    N  = len(batch)
    t1 = int(N * args.train)
    t2 = int(N * (args.train + args.val))

    splits = {
        'train_chain_2d.npy': batch[:t1],
        'val_chain_2d.npy':   batch[t1:t2],
        'test_chain_2d.npy':  batch[t2:],
    }

    for fname, arr in splits.items():
        out = dst / fname
        np.save(out, arr)
        print(f"  {out}  shape={arr.shape}")

    print(f"\nDone. Total={N}  train={t1}  val={t2-t1}  test={N-t2}")


if __name__ == '__main__':
    main()
