#!/usr/bin/env python3
"""
Phase 0 Step 4: Build binary UP/DOWN sequences from tier2 minute bars.

Key improvements over v1:
- Binary labels (UP/DOWN) for multiple horizons (5, 15, 30 min)
- Computes and saves per-feature z-score normalization stats from TRAINING split only
- Identifies and logs zero-variance features
- Builds chain_2d data for Agent 2D
- Chronological 60/20/20 split (no shuffle — financial time series)
- Saves raw returns for regression mode
- Flat sample filter: removes near-zero return samples (|return| < threshold)
- Temporal derivative features: appends delta channels to capture momentum

Usage:
    python scripts/phase0/build_tier3_binary.py --symbol SPXW
    python scripts/phase0/build_tier3_binary.py --symbol SPXW --horizons 5 15 30
    python scripts/phase0/build_tier3_binary.py --symbol SPXW --return-threshold 0.0003
    python scripts/phase0/build_tier3_binary.py --symbol SPXW --add-delta-features
    python scripts/phase0/build_tier3_binary.py --symbol SPXW --strip-zero-variance
"""

import argparse
import json
import logging
import gc
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import duckdb
from numpy.lib.format import open_memmap

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TIER2_ROOT = Path("/workspace/data/tier2_minutes_v4")
OUTPUT_ROOT = Path("/workspace/data/tier3_binary_v4")

ALL_SYMBOLS = ['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT']
FEAT_DIM = 325
SEQ_LEN = 20
HORIZONS = [5, 15, 30]  # minutes
RETURN_THRESHOLD = 0.0003  # |return| < 0.03% → flat sample, filtered out by default
BATCH_SIZE = 4096
CHAIN_INPUT_STRIKES = 30
CHAIN_OUTPUT_STRIKES = 20
CHAIN_STRIKE_START = max(0, (CHAIN_INPUT_STRIKES - CHAIN_OUTPUT_STRIKES) // 2)
TQ_FEAT_START = 270
TQ_FEAT_END = 325


def add_delta_features(sequences: np.ndarray) -> np.ndarray:
    """
    Append temporal derivative (delta) channels to each sequence.
    Optimized for memory to avoid creating multiple massive arrays at once.
    """
    import gc
    N, seq_len, feat_dim = sequences.shape
    
    # We allocate the final target array directly
    out = np.zeros((N, seq_len, feat_dim * 2), dtype=np.float32)
    
    # Copy original sequences into the first half
    out[:, :, :feat_dim] = sequences
    
    # Compute deltas in-place directly into the second half
    # Delta for t=0 is zeros (already initialized to zero)
    # Delta for t=1..seq_len-1 is seq[t] - seq[t-1]
    out[:, 1:, feat_dim:] = sequences[:, 1:, :] - sequences[:, :-1, :]
    
    # Delete original array to recover memory immediately
    # Note: caller still needs to reassign the reference or delete their copy
    gc.collect()
    return out


def compute_normalization_stats(train_features: np.ndarray) -> dict:
    """
    Compute per-feature mean and std from training data only.
    Returns dict with 'mean', 'std', 'zero_variance_mask', 'useful_features'.
    """
    mean = np.mean(train_features, axis=0).astype(np.float32)
    std = np.std(train_features, axis=0).astype(np.float32)

    # Identify zero-variance features (std < 1e-8)
    zero_var_mask = std < 1e-8
    n_zero = zero_var_mask.sum()
    n_useful = (~zero_var_mask).sum()

    # Replace zero std with 1.0 to avoid division by zero during normalization
    std_safe = std.copy()
    std_safe[zero_var_mask] = 1.0

    logger.info(f"  Normalization: {n_useful}/{len(mean)} useful features, {n_zero} zero-variance")

    if len(mean) == FEAT_DIM:
        # Log which feature groups have zero variance (only valid for the base 325-dim layout)
        group_ranges = {
            'Core Greeks (0-49)': (0, 50),
            'IV Surface (50-99)': (50, 100),
            'Term Structure (100-127)': (100, 128),
            'Flow & Volume (128-149)': (128, 150),
            'Microstructure (150-179)': (150, 180),
            'Sentiment/Regime (180-209)': (180, 210),
            'Cross-Strike-Time (210-239)': (210, 240),
            'Gamma Exposure (240-269)': (240, 270),
            'Smart Money (270-284)': (270, 285),
            'Volume Anomaly (285-296)': (285, 297),
            'Trade Conditions (297-306)': (297, 307),
            'Quote Pressure (307-324)': (307, 325),
        }

        for name, (start, end) in group_ranges.items():
            group_zero = zero_var_mask[start:end].sum()
            group_total = end - start
            if group_zero > 0:
                logger.info(f"    {name}: {group_zero}/{group_total} zero-variance")

    return {
        'mean': mean,
        'std': std_safe,
        'zero_variance_mask': zero_var_mask,
        'n_useful': int(n_useful),
        'n_zero': int(n_zero),
    }


def _iter_batches(indices: np.ndarray, batch_size: int = BATCH_SIZE):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def _build_sequence_batch(all_features: np.ndarray, sample_indices: np.ndarray, seq_len: int,
                          add_delta: bool) -> np.ndarray:
    offsets = np.arange(seq_len, dtype=np.int64)
    batch = all_features[sample_indices[:, None] + offsets[None, :]]
    if add_delta:
        batch = add_delta_features(batch)
    return batch


def _build_chain_batch(all_chain: np.ndarray, sample_indices: np.ndarray, seq_len: int) -> np.ndarray:
    offsets = np.arange(seq_len, dtype=np.int64)
    strike_slice = slice(CHAIN_STRIKE_START, CHAIN_STRIKE_START + CHAIN_OUTPUT_STRIKES)
    batch = all_chain[sample_indices[:, None] + offsets[None, :], :, strike_slice]
    return np.transpose(batch, (0, 2, 3, 1))


def _compute_normalization_stats_batched(all_features: np.ndarray, train_indices: np.ndarray,
                                         seq_len: int, add_delta: bool) -> dict:
    feat_dim = FEAT_DIM * 2 if add_delta else FEAT_DIM
    feat_sum = np.zeros(feat_dim, dtype=np.float64)
    feat_sq_sum = np.zeros(feat_dim, dtype=np.float64)
    total_count = 0

    for batch_indices in _iter_batches(train_indices):
        batch_seq = _build_sequence_batch(all_features, batch_indices, seq_len, add_delta)
        flat = batch_seq.reshape(-1, batch_seq.shape[2])
        flat64 = flat.astype(np.float64, copy=False)
        feat_sum += flat64.sum(axis=0)
        feat_sq_sum += np.square(flat64).sum(axis=0)
        total_count += flat.shape[0]
        del batch_seq, flat, flat64

    mean = (feat_sum / total_count).astype(np.float32)
    var = np.maximum((feat_sq_sum / total_count) - np.square(mean.astype(np.float64)), 0.0)
    std = np.sqrt(var).astype(np.float32)

    zero_var_mask = std < 1e-8
    n_zero = int(zero_var_mask.sum())
    n_useful = int((~zero_var_mask).sum())

    std_safe = std.copy()
    std_safe[zero_var_mask] = 1.0

    logger.info(f"  Normalization: {n_useful}/{len(mean)} useful features, {n_zero} zero-variance")

    return {
        'mean': mean,
        'std': std_safe,
        'zero_variance_mask': zero_var_mask,
        'n_useful': n_useful,
        'n_zero': n_zero,
    }


def build_binary_sequences(symbol: str, horizons: list, seq_len: int = SEQ_LEN,
                           return_threshold: float = 0.0,
                           add_delta: bool = False,
                           strip_zero_variance: bool = False):
    """Build binary UP/DOWN sequences for one symbol across multiple horizons."""
    t0 = time.time()

    # Load tier2 minute bars
    minute_file = TIER2_ROOT / f"{symbol}_minutes.parquet"
    if not minute_file.exists():
        logger.error(f"{symbol}: Tier2 not found at {minute_file}")
        return None

    import pyarrow.parquet as pq

    logger.info(f"{symbol}: Reading features via PyArrow to conserve memory")
    
    try:
        parquet_file = pq.ParquetFile(minute_file)
        N_ROWS = parquet_file.metadata.num_rows
        logger.info(f"{symbol}: Loaded {N_ROWS:,} minute bars")

        table_feat = pq.read_table(minute_file, columns=['features'])
        all_features = np.zeros((N_ROWS, FEAT_DIM), dtype=np.float32)
        idx = 0
        for chunk in table_feat.column('features').chunks:
            arr = chunk.to_numpy(zero_copy_only=False)
            for row in arr:
                all_features[idx] = row[:FEAT_DIM]
                idx += 1
        
        del table_feat
        gc.collect()

        table_price = pq.read_table(minute_file, columns=['underlying_price'])
        all_prices = np.zeros(N_ROWS, dtype=np.float64)
        idx = 0
        for chunk in table_price.column('underlying_price').chunks:
            arr = chunk.to_numpy(zero_copy_only=False)
            for row in arr:
                if row is not None:
                    all_prices[idx] = row
                idx += 1
                
        del table_price
        gc.collect()

        schema_names = parquet_file.schema_arrow.names
        has_chain = 'chain_2d' in schema_names
        if has_chain:
            logger.info(f"{symbol}: Loading chain_2d sequences...")
            table_chain = pq.read_table(minute_file, columns=['chain_2d'])
            all_chain = np.zeros((N_ROWS, 5, CHAIN_INPUT_STRIKES), dtype=np.float32)
            idx = 0
            for chunk in table_chain.column('chain_2d').chunks:
                arr = chunk.to_numpy(zero_copy_only=False)
                for row in arr:
                    if row is not None:
                        all_chain[idx] = np.array(row, dtype=np.float32).reshape(5, CHAIN_INPUT_STRIKES)
                    idx += 1
            del table_chain
            gc.collect()
        else:
            logger.warning(f"{symbol}: No chain_2d column — Agent 2D will use synthetic chains")
            all_chain = None

    except Exception as e:
        logger.error(f"Error loading {symbol}: {e}")
        return None
        
    gc.collect()

    # Clean NaN/Inf
    nan_count = np.isnan(all_features).sum()
    inf_count = np.isinf(all_features).sum()
    if nan_count > 0 or inf_count > 0:
        logger.warning(f"{symbol}: Replacing {nan_count} NaN, {inf_count} Inf with 0")
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

    tq_active_mask = np.any(np.abs(all_features[:, TQ_FEAT_START:TQ_FEAT_END]) > 1e-8, axis=1)
    tq_feature_coverage = float(tq_active_mask.mean()) if len(tq_active_mask) > 0 else 0.0
    logger.info(f"{symbol}: TQ feature coverage={tq_feature_coverage * 100:.1f}% of minutes")

    max_horizon = max(horizons)
    n_samples = len(all_features) - seq_len - max_horizon
    if n_samples <= 0:
        logger.error(f"{symbol}: Not enough data for sequences (need {seq_len + max_horizon}, have {len(all_features)})")
        return None

    logger.info(f"{symbol}: Preparing {n_samples:,} samples (seq_len={seq_len})")

    final_feat_dim = FEAT_DIM * 2 if add_delta else FEAT_DIM
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)
    if add_delta:
        logger.info(f"{symbol}: Delta features enabled → {final_feat_dim} dims (was {FEAT_DIM})")

    results = {}

    for horizon in horizons:
        logger.info(f"\n{symbol}: Building horizon={horizon}min labels...")

        current_prices = all_prices[seq_len - 1:seq_len - 1 + n_samples]
        future_prices = all_prices[seq_len - 1 + horizon:seq_len - 1 + horizon + n_samples]
        returns = np.zeros(n_samples, dtype=np.float32)
        valid = current_prices > 0
        returns[valid] = ((future_prices[valid] - current_prices[valid]) / current_prices[valid]).astype(np.float32)
        labels = (returns > 0).astype(np.int64)

        if return_threshold > 0:
            keep_mask = np.abs(returns) >= return_threshold
            n_filtered = n_samples - keep_mask.sum()
            pct_filtered = 100.0 * n_filtered / n_samples
            logger.info(f"  Flat filter (|ret|>={return_threshold:.4f}): "
                        f"removed {n_filtered:,} ({pct_filtered:.1f}%) flat samples, "
                        f"{keep_mask.sum():,} remain")
            keep_idx = np.flatnonzero(keep_mask)
            n_filtered_total = len(keep_idx)
            train_end_f = int(0.6 * n_filtered_total)
            val_end_f = int(0.8 * n_filtered_total)
        else:
            keep_idx = np.arange(n_samples, dtype=np.int64)
            train_end_f = train_end
            val_end_f = val_end

        train_idx = keep_idx[:train_end_f]
        val_idx = keep_idx[train_end_f:val_end_f]
        test_idx = keep_idx[val_end_f:]

        if len(train_idx) == 0:
            logger.error(f"{symbol} h{horizon}: no training samples after filtering")
            continue

        norm_stats = _compute_normalization_stats_batched(all_features, train_idx, seq_len, add_delta)

        if strip_zero_variance and norm_stats['n_zero'] > 0:
            logger.info(
                f"  {symbol} h{horizon}: strip_zero_variance enabled — keeping feat_dim={final_feat_dim} and saving mask only (no dim removal)"
            )

        out_dir = OUTPUT_ROOT / symbol / f"horizon_{horizon}min"
        out_dir.mkdir(parents=True, exist_ok=True)

        split_specs = [
            ('train', train_idx),
            ('val', val_idx),
            ('test', test_idx),
        ]

        split_sizes = {}
        for split_name, split_idx in split_specs:
            split_sizes[split_name] = len(split_idx)
            np.save(out_dir / f'{split_name}_labels.npy', labels[split_idx])
            np.save(out_dir / f'{split_name}_returns.npy', returns[split_idx])

            if len(split_idx) == 0:
                np.save(
                    out_dir / f'{split_name}_sequences.npy',
                    np.empty((0, seq_len, final_feat_dim), dtype=np.float32)
                )
                if all_chain is not None:
                    np.save(
                        out_dir / f'{split_name}_chain_2d.npy',
                        np.empty((0, 5, CHAIN_OUTPUT_STRIKES, seq_len), dtype=np.float32)
                    )
                continue

            seq_mem = open_memmap(
                out_dir / f'{split_name}_sequences.npy',
                mode='w+',
                dtype=np.float32,
                shape=(len(split_idx), seq_len, final_feat_dim),
            )

            if all_chain is not None:
                chain_mem = open_memmap(
                    out_dir / f'{split_name}_chain_2d.npy',
                    mode='w+',
                    dtype=np.float32,
                    shape=(len(split_idx), 5, CHAIN_OUTPUT_STRIKES, seq_len),
                )
            else:
                chain_mem = None

            write_offset = 0
            for batch_indices in _iter_batches(split_idx):
                batch_seq = _build_sequence_batch(all_features, batch_indices, seq_len, add_delta)
                seq_mem[write_offset:write_offset + len(batch_indices)] = batch_seq

                if chain_mem is not None:
                    chain_mem[write_offset:write_offset + len(batch_indices)] = _build_chain_batch(all_chain, batch_indices, seq_len)

                write_offset += len(batch_indices)
                del batch_seq

            del seq_mem
            if chain_mem is not None:
                del chain_mem
            gc.collect()

        np.save(out_dir / 'norm_mean.npy', norm_stats['mean'])
        np.save(out_dir / 'norm_std.npy', norm_stats['std'])
        np.save(out_dir / 'zero_variance_mask.npy', norm_stats['zero_variance_mask'])

        up_pct = 100 * labels[train_idx].mean()
        logger.info(f"  {symbol} h{horizon}: train={split_sizes['train']:,} val={split_sizes['val']:,} test={split_sizes['test']:,}")
        logger.info(f"  UP/DOWN split: {up_pct:.1f}% UP, {100-up_pct:.1f}% DOWN")
        logger.info(f"  Saved to: {out_dir}")

        metadata = {
            'symbol': symbol,
            'horizon_min': horizon,
            'seq_len': seq_len,
            'feat_dim': final_feat_dim,
            'n_samples': n_samples,
            'n_samples_after_filter': int(keep_mask.sum()) if return_threshold > 0 else n_samples,
            'return_threshold': return_threshold,
            'add_delta_features': add_delta,
            'strip_zero_variance': strip_zero_variance,
            'train_size': split_sizes['train'],
            'val_size': split_sizes['val'],
            'test_size': split_sizes['test'],
            'up_pct_train': round(float(up_pct), 2),
            'useful_features': norm_stats['n_useful'],
            'zero_variance_features': norm_stats['n_zero'],
            'has_chain_2d': all_chain is not None,
            'chain_input_strikes': CHAIN_INPUT_STRIKES if all_chain is not None else 0,
            'chain_output_strikes': CHAIN_OUTPUT_STRIKES if all_chain is not None else 0,
            'chain_strike_start': CHAIN_STRIKE_START if all_chain is not None else None,
            'tq_feature_coverage': round(tq_feature_coverage, 4),
        }
        with open(out_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        results[f'h{horizon}'] = metadata

        del returns, labels, keep_idx, train_idx, val_idx, test_idx
        gc.collect()

    del all_features, all_prices
    if all_chain is not None:
        del all_chain
    gc.collect()
    
    elapsed = time.time() - t0
    logger.info(f"\n{symbol}: All horizons complete in {elapsed:.1f}s")
    return results


def main():
    global TIER2_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description='Build tier3 binary sequences')
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--all-symbols', action='store_true')
    parser.add_argument('--horizons', type=int, nargs='+', default=HORIZONS)
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN)
    parser.add_argument('--tier2-root', type=str, default=str(TIER2_ROOT),
                        help='Tier2 minute-bar root (default: /workspace/data/tier2_minutes_v3)')
    parser.add_argument('--output-root', type=str, default=str(OUTPUT_ROOT),
                        help='Tier3 output root (default: /workspace/data/tier3_binary_v2)')
    parser.add_argument('--return-threshold', type=float, default=RETURN_THRESHOLD,
                        help='Filter flat samples where |return| < threshold (default: 0.0003). Set 0 to disable.')
    parser.add_argument('--add-delta-features', action='store_true',
                        help='Append temporal derivative (delta) channels, doubling feat_dim')
    parser.add_argument('--strip-zero-variance', action='store_true',
                        help='Compute and save zero-variance mask (does not change feat_dim)')
    args = parser.parse_args()

    TIER2_ROOT = Path(args.tier2_root)
    OUTPUT_ROOT = Path(args.output_root)

    if not args.symbol and not args.all_symbols:
        parser.error('Either --symbol or --all-symbols required')

    symbols = ALL_SYMBOLS if args.all_symbols else [args.symbol]

    print("=" * 70)
    print("PHASE 0 STEP 4: Build Tier3 Binary Sequences")
    print(f"  Symbols:            {symbols}")
    print(f"  Horizons:           {args.horizons}")
    print(f"  Seq len:            {args.seq_len}")
    print(f"  Tier2 root:         {TIER2_ROOT}")
    print(f"  Output root:        {OUTPUT_ROOT}")
    print(f"  Feat dim:           {FEAT_DIM}")
    print(f"  Return threshold:   {args.return_threshold} ({'disabled' if args.return_threshold == 0 else 'enabled'})")
    print(f"  Delta features:     {'ON' if args.add_delta_features else 'OFF'}")
    print(f"  Zero-var mask:      {'ON' if args.strip_zero_variance else 'OFF'}")
    print("=" * 70)

    all_results = {}
    for sym in symbols:
        result = build_binary_sequences(
            sym, args.horizons, args.seq_len,
            return_threshold=args.return_threshold,
            add_delta=args.add_delta_features,
            strip_zero_variance=args.strip_zero_variance,
        )
        if result:
            all_results[sym] = result

    # Save global summary
    summary_path = OUTPUT_ROOT / 'build_summary.json'
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
