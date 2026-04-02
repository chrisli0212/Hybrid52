#!/usr/bin/env python3
"""
CSV-first Phase0 runner (no DuckDB required for equities path).

Pipeline:
  1) preflight_historical_data_1yr.py
  2) /workspace/csv_to_tier1.py                    (CSV -> tier1 parquet)
  3) build_tier2.py                                (tier1 -> tier2 minutes)
  4) build_tier3_binary.py                         (tier2 -> tier3 binary)
  5) build_vixw_minutes_from_raw.py + build_tier3_vix.py (optional VIX path)
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


PHASE0_DIR = Path(__file__).resolve().parent
DEFAULT_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]


def _check_dependencies(required: list[str], optional: list[str]) -> None:
    missing_required = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing_required.append(pkg)
    if missing_required:
        raise RuntimeError(
            "Missing required python packages: "
            + ", ".join(missing_required)
            + ". Install before running pipeline."
        )
    missing_optional = []
    for pkg in optional:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing_optional.append(pkg)
    if missing_optional:
        print(f"[WARN] Optional packages missing: {', '.join(missing_optional)}")


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 120)
    print("RUN:", " ".join(str(c) for c in cmd))
    print("=" * 120)
    subprocess.run(cmd, check=True)


def _tier1_done(tier1_root: Path, symbol: str) -> bool:
    greek = tier1_root / "greek" / f"symbol={symbol}"
    tq = tier1_root / "tradequote" / f"symbol={symbol}"
    return greek.exists() and any(greek.glob("*.parquet")) and tq.exists() and any(tq.glob("*.parquet"))


def _tier2_done(tier2_root: Path, symbol: str) -> bool:
    return (tier2_root / f"{symbol}_minutes.parquet").exists()


def _tier3_done(tier3_root: Path, symbol: str, horizons: list[int]) -> bool:
    for h in horizons:
        p = tier3_root / symbol / f"horizon_{h}min" / "train_sequences.npy"
        if not p.exists():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase0 from historical_data_1yr without DuckDB")
    parser.add_argument("--csv-root", default="/workspace/historical_data_1yr")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--tier1-root", default="/workspace/data/tier1_2026_csv_v1")
    parser.add_argument("--tier2-root", default="/workspace/data/tier2_minutes_2026_csv_v1")
    parser.add_argument("--tier3-root", default="/workspace/data/tier3_binary_2026_csv_v1")
    parser.add_argument("--horizons", type=int, nargs="+", default=[30])
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--return-threshold", type=float, default=0.0003)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-preflight", action="store_true")
    parser.add_argument("--include-vix", action="store_true")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]
    csv_root = Path(args.csv_root)
    tier1_root = Path(args.tier1_root)
    tier2_root = Path(args.tier2_root)
    tier3_root = Path(args.tier3_root)

    _check_dependencies(required=["numpy", "pandas", "pyarrow"], optional=["duckdb"])

    preflight_py = PHASE0_DIR / "preflight_historical_data_1yr.py"
    build_tier2_py = PHASE0_DIR / "build_tier2.py"
    build_tier3_py = PHASE0_DIR / "build_tier3_binary.py"
    build_vixw_py = PHASE0_DIR / "build_vixw_minutes_from_raw.py"
    build_tier3_vix_py = PHASE0_DIR / "build_tier3_vix.py"
    csv_to_tier1_py = Path("/workspace/csv_to_tier1.py")

    if not csv_root.exists():
        raise FileNotFoundError(f"CSV root not found: {csv_root}")
    if not csv_to_tier1_py.exists():
        raise FileNotFoundError(f"CSV->tier1 bridge missing: {csv_to_tier1_py}")

    print("=" * 70)
    print("PHASE0 CSV-FIRST (NO-DUCKDB EQUITIES)")
    print("CSV root:", csv_root)
    print("Symbols:", symbols)
    print("Tier1:", tier1_root)
    print("Tier2:", tier2_root)
    print("Tier3:", tier3_root)
    print("Horizons:", args.horizons)
    print("Seq len:", args.seq_len)
    print("Return threshold:", args.return_threshold)
    print("Workers:", args.workers)
    print("Force:", args.force)
    print("=" * 70)

    # 1) Preflight
    preflight_cmd = [
        sys.executable, str(preflight_py),
        "--csv-root", str(csv_root),
        "--symbols", *symbols,
    ]
    if args.strict_preflight:
        preflight_cmd.append("--strict")
    _run(preflight_cmd)

    # 2) CSV -> Tier1
    if args.force or any(not _tier1_done(tier1_root, s) for s in symbols):
        _run([
            sys.executable, str(csv_to_tier1_py),
            "--csv-dir", str(csv_root),
            "--out-root", str(tier1_root),
            "--symbols", *symbols,
            *(["--overwrite"] if args.force else []),
        ])
    else:
        print("SKIP Tier1: all requested symbols already present.")

    # 3) Tier1 -> Tier2
    for sym in symbols:
        if args.force or not _tier2_done(tier2_root, sym):
            _run([
                sys.executable, str(build_tier2_py),
                "--symbol", sym,
                "--tier1-root", str(tier1_root),
                "--output-root", str(tier2_root),
                "--workers", str(args.workers),
            ])
        else:
            print(f"SKIP Tier2: {sym} already present.")

    # 4) Tier2 -> Tier3
    for sym in symbols:
        if args.force or not _tier3_done(tier3_root, sym, args.horizons):
            _run([
                sys.executable, str(build_tier3_py),
                "--symbol", sym,
                "--tier2-root", str(tier2_root),
                "--output-root", str(tier3_root),
                "--horizons", *[str(h) for h in args.horizons],
                "--seq-len", str(args.seq_len),
                "--return-threshold", str(args.return_threshold),
                "--split-mode", "calendar",
            ])
        else:
            print(f"SKIP Tier3: {sym} already present.")

    # 5) Optional VIX path from raw CSV
    if args.include_vix:
        _run([sys.executable, str(build_vixw_py)])
        _run([
            sys.executable, str(build_tier3_vix_py),
            "--tier2-root", str(tier2_root),
            "--output-root", "/workspace/data/tier3_vix_v4",
            "--symbol", "VIXW",
            "--bootstrap-from-raw",
            "--resolution", "1min",
        ])

    print("\nDONE: CSV-first phase0 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
