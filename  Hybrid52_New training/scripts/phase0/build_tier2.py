#!/usr/bin/env python3
"""
Phase 0 Step 3: Build tier2 minute bars from tier1 parquets using MasterFeatureExtractor v2.

Combines Greek + TQ tier1 data → 325-dim feature vectors per minute bar.
This is a thin wrapper that delegates to the existing tier2_reprocess.py logic
but points at the v2 tier1 output paths.

Usage:
    python scripts/phase0/build_tier2.py --symbol SPXW
    python scripts/phase0/build_tier2.py --all-symbols
"""

import argparse
import sys
from pathlib import Path
import os

# Add the preprocessing module to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'hybrid51_preprocessing'))

# Also add stage3 preprocessing path as fallback
STAGE3_ROOT = Path("/workspace/Hybrid51/5. hybrid51_stage3")
sys.path.insert(0, str(STAGE3_ROOT))
sys.path.insert(0, str(STAGE3_ROOT / 'hybrid51_preprocessing'))

# Paths — defaults
TIER1_GREEK_ROOT = Path("/workspace/data/tier1_v2/greek")
TIER1_TQ_ROOT = Path("/workspace/data/tier1_v2/tradequote")
OUTPUT_ROOT = Path("/workspace/data/tier2_minutes_v3")

ALL_SYMBOLS = ['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT']  # Dropped VIXW per plan


def main():
    parser = argparse.ArgumentParser(description='Build tier2 from tier1 v2')
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--all-symbols', action='store_true')
    parser.add_argument('--continue-remaining', action='store_true',
                        help='If set with --symbol, continue with the remaining symbols in ALL_SYMBOLS after the specified symbol')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers (memory intensive)')
    parser.add_argument('--tier1-root', type=str, default=None,
                        help='Tier1 root containing greek/ and tradequote/ (e.g. /workspace/data/tier1_2026_v1)')
    parser.add_argument('--output-root', type=str, default=str(OUTPUT_ROOT),
                        help='Tier2 output root (default: /workspace/data/tier2_minutes_v3)')
    args = parser.parse_args()

    if not args.symbol and not args.all_symbols:
        parser.error('Either --symbol or --all-symbols required')

    if args.all_symbols:
        symbols = ALL_SYMBOLS
    else:
        if args.continue_remaining:
            if args.symbol in ALL_SYMBOLS:
                start_idx = ALL_SYMBOLS.index(args.symbol)
                symbols = ALL_SYMBOLS[start_idx:]
            else:
                symbols = [args.symbol] + [s for s in ALL_SYMBOLS if s != args.symbol]
        else:
            symbols = [args.symbol]

    if args.tier1_root:
        tier1_root = Path(args.tier1_root)
        tier1_greek_root = tier1_root / 'greek'
        tier1_tq_root = tier1_root / 'tradequote'
    else:
        tier1_greek_root = TIER1_GREEK_ROOT
        tier1_tq_root = TIER1_TQ_ROOT

    output_root = Path(args.output_root)

    print(f"Tier2 Build v3")
    print(f"  Greek root: {tier1_greek_root}")
    print(f"  TQ root:    {tier1_tq_root}")
    print(f"  Output:     {output_root}")
    print(f"  Symbols:    {symbols}")

    # Delegate to the existing tier2_reprocess logic
    # We import it here to allow the path setup above to take effect
    try:
        from hybrid51_preprocessing.master_extractor_v2 import MasterFeatureExtractor
        from hybrid51_preprocessing.chain_2d import Chain2DProcessor
        from hybrid51_preprocessing.feature_config_v2 import TOTAL_FEATURES
        print(f"  Feature dim: {TOTAL_FEATURES}")
    except ImportError as e:
        print(f"ERROR: Could not import preprocessing modules: {e}")
        print("Make sure hybrid51_preprocessing is available.")
        sys.exit(1)

    # Reuse the tier2_reprocess module.
    # Important: this must be imported as a normal module (not via importlib-from-file)
    # so multiprocessing workers can import it when pickling functions.
    data_build_dir = STAGE3_ROOT / 'scripts' / 'data_build'
    sys.path.insert(0, str(data_build_dir))
    import importlib
    tier2_mod = importlib.import_module('tier2_reprocess')

    # Override paths after executing (module code sets its own defaults)
    tier2_mod.TIER1_GREEK_ROOT = tier1_greek_root
    tier2_mod.TIER1_TQ_ROOT = tier1_tq_root
    tier2_mod.OUTPUT_ROOT = output_root
    tier2_mod.ALL_SYMBOLS = symbols

    # Propagate worker preference to the underlying multiprocessing pool (see tier2_reprocess.py)
    os.environ['TIER2_WORKERS'] = str(args.workers)

    # Process each symbol
    for sym in symbols:
        greek_dir = tier1_greek_root / f"symbol={sym}"
        tq_dir = tier1_tq_root / f"symbol={sym}"

        if not greek_dir.exists():
            print(f"  SKIP {sym}: no Greek data at {greek_dir}")
            continue

        print(f"\n  Processing {sym}...")
        try:
            tier2_mod.process_symbol(sym)
        except Exception as e:
            print(f"  ERROR processing {sym}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTier2 build complete. Output: {output_root}")


if __name__ == '__main__':
    main()
