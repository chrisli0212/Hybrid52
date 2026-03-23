#!/usr/bin/env python3
"""
Sync production normalization files from canonical per-symbol training stats.

This script intentionally avoids deriving norms from live snapshots, because
Stage-1 models were trained with precomputed per-symbol tier3 train-split stats.

Usage:
    python compute_production_norms.py
    python compute_production_norms.py --source-root /workspace/data/tier3_binary_v4
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
NORM_FILES = ("norm_mean.npy", "norm_std.npy")


def _sha16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _root_has_complete_norms(root: Path, symbols: list[str], horizon_dir: str) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for symbol in symbols:
        for fname in NORM_FILES:
            p = root / symbol / horizon_dir / fname
            if not p.exists():
                missing.append(str(p))
    return len(missing) == 0, missing


def _discover_source_roots(search_root: Path, symbols: list[str], horizon_dir: str) -> list[Path]:
    """
    Discover candidate roots under search_root matching:
      <root>/SPXW/<horizon_dir>/norm_mean.npy
    then validate full symbol coverage for both norm files.
    """
    if not search_root.exists():
        return []

    candidates: set[Path] = set()
    for norm_mean in search_root.rglob("norm_mean.npy"):
        # Expect: <root>/SPXW/horizon_XXmin/norm_mean.npy
        if norm_mean.parent.name != horizon_dir:
            continue
        symbol_dir = norm_mean.parent.parent
        if symbol_dir.name != "SPXW":
            continue
        root = symbol_dir.parent
        ok, _ = _root_has_complete_norms(root, symbols, horizon_dir)
        if ok:
            candidates.add(root)

    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        "--src",
        dest="source_root",
        type=str,
        default=None,
        help="Canonical tier3 root containing per-symbol norm files (optional)",
    )
    parser.add_argument(
        "--output-dir",
        "--dst",
        dest="output_dir",
        type=str,
        default="/workspace/data/tier3_binary_v5",
        help="Destination root used by production inference",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Horizon in minutes (used in horizon_<N>min path)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols to sync normalization files for",
    )
    parser.add_argument(
        "--search-root",
        type=str,
        default="/workspace",
        help="Root path to scan when --source-root is missing/invalid",
    )
    args = parser.parse_args()

    requested_source = Path(args.source_root).resolve() if args.source_root else None
    output_root = Path(args.output_dir)
    search_root = Path(args.search_root)
    horizon_dir = f"horizon_{int(args.horizon)}min"
    symbols = [s.upper() for s in args.symbols]

    logger.info("=" * 70)
    logger.info("SYNCING PRODUCTION NORMALIZATION FILES")
    logger.info("=" * 70)
    logger.info(f"Requested source root: {requested_source}")
    logger.info(f"Search root: {search_root}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Horizon: {horizon_dir}")
    logger.info(f"Symbols: {symbols}")

    selected_source: Path | None = None
    discovery_candidates: list[Path] = []

    # 1) Prefer explicit source when valid.
    if requested_source is not None:
        ok, missing = _root_has_complete_norms(requested_source, symbols, horizon_dir)
        if ok:
            selected_source = requested_source
        else:
            logger.warning(f"Requested source is missing required files: {requested_source}")
            for m in missing[:5]:
                logger.warning(f"  missing: {m}")
            if len(missing) > 5:
                logger.warning(f"  ... and {len(missing) - 5} more")

    # 2) Fall back to auto-discovery under /workspace (or configured search root).
    if selected_source is None:
        discovery_candidates = _discover_source_roots(search_root, symbols, horizon_dir)
        if discovery_candidates:
            selected_source = discovery_candidates[0]
            logger.info(f"Auto-discovered source root: {selected_source}")
            if len(discovery_candidates) > 1:
                logger.warning("Multiple valid source roots found; selected the first by stable ordering.")
                for alt in discovery_candidates[1:]:
                    logger.warning(f"  alternate: {alt}")

    if selected_source is None:
        logger.error("No valid canonical norm source root found.")
        logger.error(
            "Provide --source-root explicitly or place per-symbol norm files under "
            "<root>/<SYMBOL>/" + horizon_dir + "/norm_{mean,std}.npy"
        )
        return 1

    copied = []
    failed = []

    for symbol in symbols:
        src_dir = selected_source / symbol / horizon_dir
        dst_dir = output_root / symbol / horizon_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        for fname in NORM_FILES:
            src = src_dir / fname
            dst = dst_dir / fname
            if not src.exists():
                logger.error(f"Missing source file: {src}")
                failed.append(str(src))
                continue

            try:
                arr = np.load(src)
                if arr.ndim != 1:
                    raise ValueError(f"{fname} must be 1D, got shape {arr.shape}")
                if arr.shape[0] != 325:
                    raise ValueError(f"{fname} expected length 325, got {arr.shape[0]}")

                shutil.copy2(src, dst)
                copied.append({
                    "symbol": symbol,
                    "file": fname,
                    "shape": list(arr.shape),
                    "sha16": _sha16(arr),
                })
                logger.info(f"  Copied {symbol}/{fname}  shape={arr.shape} sha16={_sha16(arr)}")
            except Exception as e:
                logger.error(f"Failed to process {src}: {e}")
                failed.append(f"{src}: {e}")

    metadata = {
        "mode": "copy_from_canonical_tier3",
        "requested_source_root": str(requested_source) if requested_source is not None else None,
        "selected_source_root": str(selected_source),
        "search_root": str(search_root),
        "discovery_candidates": [str(p) for p in discovery_candidates],
        "output_root": str(output_root),
        "horizon": horizon_dir,
        "symbols": symbols,
        "copied_files": copied,
        "failed": failed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    meta_path = output_root / "normalization_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata: {meta_path}")

    if failed:
        logger.error("Completed with errors. Some norm files were not synced.")
        return 1

    logger.info("All normalization files synced successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
