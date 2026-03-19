#!/usr/bin/env python3

import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hybrid51_utils import ArtifactPaths

PATHS = ArtifactPaths.default()

ALL_AGENTS = ["A", "B", "K", "C", "T", "Q"]
DEFAULT_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
DEFAULT_PAIRS = ["SPY", "QQQ", "IWM", "TLT"]


def _try_import(mod: str) -> tuple[bool, str]:
    try:
        importlib.import_module(mod)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _check_tier3_symbol(symbol: str, horizon: int) -> list[str]:
    errs: list[str] = []
    d = PATHS.tier3_dir(symbol, horizon)
    if not d.exists():
        return [f"Missing tier3 directory: {d}"]

    needed = [
        "train_sequences.npy",
        "train_labels.npy",
        "val_sequences.npy",
        "val_labels.npy",
        "test_sequences.npy",
        "test_labels.npy",
    ]
    for fn in needed:
        p = d / fn
        if not p.exists():
            errs.append(f"Missing tier3 file: {p}")
    return errs


def _check_stage1_ckpts(symbols: list[str], horizon: int) -> list[str]:
    errs: list[str] = []
    for sym in symbols:
        for agent in ALL_AGENTS:
            p = PATHS.stage1_ckpt(sym, agent, horizon)
            if not p.exists():
                errs.append(f"Missing Stage1 ckpt: {p}")
    return errs


def _check_stage2_outputs(target: str, pairs: list[str], horizon: int) -> list[str]:
    errs: list[str] = []
    for pair in pairs:
        p_npz = PATHS.stage2_probs(target, pair, horizon)
        p_pt = PATHS.stage2_ckpt(target, pair, horizon)
        if not p_npz.exists():
            errs.append(f"Missing Stage2 probs: {p_npz}")
        if not p_pt.exists():
            errs.append(f"Missing Stage2 ckpt: {p_pt}")
    return errs


def _check_stage3_outputs(target: str, horizon: int) -> list[str]:
    errs: list[str] = []
    p_metrics = PATHS.stage3_metrics(target, horizon)
    if not p_metrics.exists():
        errs.append(f"Missing Stage3 metrics: {p_metrics}")

    p_joblib = PATHS.stage3_logreg_model(target, horizon)
    p_mlp = PATHS.stage3_mlp_model(target, horizon)
    if not p_joblib.exists() and not p_mlp.exists():
        errs.append(f"Missing Stage3 model (expected one of): {p_joblib} or {p_mlp}")
    return errs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--for", dest="for_stage", choices=["phase0", "stage1", "stage2", "stage3"], required=True)
    parser.add_argument("--target", default="SPXW")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--pairs", nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=15)
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS
    pairs = args.pairs if args.pairs else DEFAULT_PAIRS

    errors: list[str] = []
    warnings: list[str] = []

    req = {
        "phase0": ["duckdb", "numpy", "pandas"],
        "stage1": ["torch", "numpy", "sklearn", "scipy"],
        "stage2": ["torch", "numpy", "sklearn"],
        "stage3": ["numpy", "sklearn", "joblib"],
    }

    for mod in req[args.for_stage]:
        ok, msg = _try_import(mod)
        if not ok:
            errors.append(f"Import failed: {mod} ({msg})")

    if args.for_stage in ("phase0", "stage1", "stage2"):
        for sym in symbols:
            errors.extend(_check_tier3_symbol(sym, args.horizon))

    if args.for_stage in ("stage2", "stage3"):
        errors.extend(_check_stage1_ckpts([args.target] + pairs, args.horizon))

    if args.for_stage == "stage3":
        errors.extend(_check_stage2_outputs(args.target, pairs, args.horizon))

    if args.for_stage == "stage3":
        errors.extend(_check_stage3_outputs(args.target, args.horizon))

    if not PATHS.data_root.exists():
        warnings.append(f"DATA_ROOT not found: {PATHS.data_root}")

    print("=" * 80)
    print("Hybrid51 Preflight")
    print(f"Root      : {PATHS.root}")
    print(f"For stage : {args.for_stage}")
    print(f"Horizon   : {args.horizon}")
    print(f"Target    : {args.target}")
    print(f"Symbols   : {symbols}")
    print(f"Pairs     : {pairs}")
    print("=" * 80)

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"- {w}")
        print("")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"- {e}")
        sys.exit(2)

    print("OK")


if __name__ == "__main__":
    main()
