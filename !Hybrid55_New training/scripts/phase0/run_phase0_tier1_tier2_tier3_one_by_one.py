#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


PHASE0_DIR = Path(__file__).resolve().parent

def _run(cmd: list[str]) -> None:
    print('\n' + '=' * 120)
    print('RUN:', ' '.join(str(c) for c in cmd))
    print('=' * 120)
    subprocess.run(cmd, check=True)


def _tier1_done(tier1_root: Path, symbol: str) -> bool:
    sym_dir = tier1_root / symbol
    if not sym_dir.exists():
        return False
    has_greek = any(sym_dir.glob('*_greek.parquet'))
    has_tq = any(sym_dir.glob('*_tq.parquet'))
    has_split_tq = any(sym_dir.glob('*_trade.parquet')) and any(sym_dir.glob('*_quote.parquet'))
    return has_greek and (has_tq or has_split_tq)


def _tier2_done(tier2_root: Path, symbol: str) -> bool:
    return (tier2_root / f'{symbol}_minutes.parquet').exists()


def _tier3_done(tier3_root: Path, symbol: str, horizons: list[int]) -> bool:
    sym_dir = tier3_root / symbol
    if not sym_dir.exists():
        return False
    for h in horizons:
        p = sym_dir / f'horizon_{h}min' / 'train_sequences.npy'
        if not p.exists():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', default=['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT'])
    parser.add_argument('--source-root', type=str, default='/workspace/data/theta_data_3year')
    parser.add_argument('--tier1-root', type=str, default='/workspace/data/tier1_hybrid55')
    parser.add_argument('--tier2-root', type=str, default='/workspace/data/tier2_minutes_hybrid55')
    parser.add_argument('--tier3-root', type=str, default='/workspace/data/tier3_binary_hybrid55')
    parser.add_argument('--workers', type=int, default=4)
    # Tier3 can build multiple horizons; production training uses 30 min (`horizon_30min/`).
    parser.add_argument('--horizons', type=int, nargs='+', default=[5, 15, 30])
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--return-threshold', type=float, default=0.0003)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    tier1_root = Path(args.tier1_root)
    source_root = Path(args.source_root)
    tier2_root = Path(args.tier2_root)
    tier3_root = Path(args.tier3_root)

    extract_tier1_py = PHASE0_DIR / 'extract_tier1_joined.py'
    build_tier2_py = PHASE0_DIR / 'build_tier2_fast.py'  # direct MasterFeatureExtractor, no tier2_reprocess chain
    build_tier3_py = PHASE0_DIR / 'build_tier3_binary.py'

    symbols = [s.upper() for s in args.symbols]

    print('=' * 70)
    print('PHASE0 ONE-BY-ONE: Tier1 -> Tier2 -> Tier3')
    print('Symbols:', symbols)
    print('Source root:', source_root)
    print('Tier1:', tier1_root)
    print('Tier2:', tier2_root)
    print('Tier3:', tier3_root)
    print('Workers:', args.workers)
    print('Horizons:', args.horizons)
    print('Seq len:', args.seq_len)
    print('Return threshold:', args.return_threshold)
    print('Force:', args.force)
    print('=' * 70)

    for sym in symbols:
        sym_source = source_root / sym
        if not sym_source.exists():
            raise FileNotFoundError(f'source symbol folder not found for {sym}: {sym_source}')

        print('\n' + '#' * 120)
        print(f'SYMBOL: {sym}  (source={sym_source})')
        print('#' * 120)

        if args.force or not _tier1_done(tier1_root, sym):
            # Tier1 default: per-date *_trade.parquet + *_quote.parquet (see extract_tier1_joined.py).
            # Optional legacy *_tq.parquet: run extract_tier1_joined.py with --write-combined-tq.
            _run([
                sys.executable,
                str(extract_tier1_py),
                '--source-root',
                str(source_root),
                '--symbols',
                sym,
                '--output-root',
                str(tier1_root),
            ])
        else:
            print(f'SKIP Tier1: already present at {tier1_root} for {sym}')

        if args.force or not _tier2_done(tier2_root, sym):
            _run([
                sys.executable,
                str(build_tier2_py),
                '--symbol',
                sym,
                '--tier1-root',
                str(tier1_root),
                '--output-root',
                str(tier2_root),
                '--workers',
                str(args.workers),
            ])
        else:
            print(f'SKIP Tier2: already present at {tier2_root} for {sym}')

        if args.force or not _tier3_done(tier3_root, sym, args.horizons):
            _run([
                sys.executable,
                str(build_tier3_py),
                '--symbol',
                sym,
                '--tier2-root',
                str(tier2_root),
                '--output-root',
                str(tier3_root),
                '--horizons',
                *[str(h) for h in args.horizons],
                '--seq-len',
                str(args.seq_len),
                '--return-threshold',
                str(args.return_threshold),
            ])
        else:
            print(f'SKIP Tier3: already present at {tier3_root} for {sym}')

    print('\nDONE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
