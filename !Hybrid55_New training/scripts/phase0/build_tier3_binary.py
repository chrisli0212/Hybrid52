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
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from hybrid55_preprocessing.feature_config_v2 import (
    TOTAL_FEATURES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_GROUPS,
    FeatureGroup,
)
from hybrid55_preprocessing.quality_checks import DataQualityChecker, validate_preprocessed_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TIER2_ROOT = Path("/workspace/data/tier2_minutes_hybrid55")
OUTPUT_ROOT = Path("/workspace/data/tier3_binary_hybrid55")

ALL_SYMBOLS = ['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT']
FEAT_DIM = TOTAL_FEATURES
SEQ_LEN = 20
HORIZONS = [5, 15, 30]  # minutes
RETURN_THRESHOLD = 0.0003  # |return| < 0.03% → flat sample, filtered out by default
BATCH_SIZE = 4096
STD_EPS = 1e-5
CHAIN_INPUT_STRIKES = 30
CHAIN_OUTPUT_STRIKES = 20
CHAIN_STRIKE_START = max(0, (CHAIN_INPUT_STRIKES - CHAIN_OUTPUT_STRIKES) // 2)
CSV_FEAT_START = 270
CSV_FEAT_END = 286
_GREEK_END = FEATURE_GROUPS[FeatureGroup.IV_SURFACE].end_idx
_TQ_END = FEATURE_GROUPS[FeatureGroup.CSV_DERIVED].end_idx
_OHLC_END = FEATURE_GROUPS[FeatureGroup.OHLC_DYNAMICS].end_idx
MODALITY_RANGES = {
    "greek": [(0, _GREEK_END)],
    "tq": [(_GREEK_END, _TQ_END)],
    "ohlc": [(_TQ_END, _OHLC_END)],
}
FEATURE_GROUP_DIAGNOSTIC_RANGES = {
    "greek_core": [(0, _GREEK_END)],
    "tq_slice": [(_GREEK_END, _TQ_END)],
    "ohlc_block": [(_TQ_END, _OHLC_END)],
}
# Sparse modalities can be informative even with low variance in short windows.
# Threshold raised from 0.002 to 0.05: features that are >95% zero provide negligible
# signal and were slipping through as "live" despite being functionally dead.
SPARSE_KEEP_MIN_NONZERO_RATIO = {
    "tq": 0.05,    # keep if at least 5% non-zero across train sequence timesteps
    "ohlc": 0.05,
}


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

    # Identify near-zero variance features from training split
    zero_var_mask = std < STD_EPS
    n_zero = zero_var_mask.sum()
    n_useful = (~zero_var_mask).sum()

    # Floor std to keep z-score normalization numerically stable
    std_safe = np.maximum(std, STD_EPS)

    logger.info(f"  Normalization: {n_useful}/{len(mean)} useful features, {n_zero} zero-variance")

    if len(mean) == FEAT_DIM:
        # Log zero-variance counts for current Hybrid55 modality bands.
        group_ranges = {
            'Greek core (0-149)': (0, 150),
            'TQ slice (150-285)': (150, 286),
            'OHLC block (286-310)': (286, 311),
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
    feat_nonzero_count = np.zeros(feat_dim, dtype=np.float64)
    total_count = 0

    for batch_indices in _iter_batches(train_indices):
        batch_seq = _build_sequence_batch(all_features, batch_indices, seq_len, add_delta)
        flat = batch_seq.reshape(-1, batch_seq.shape[2])
        flat64 = flat.astype(np.float64, copy=False)
        feat_sum += flat64.sum(axis=0)
        feat_sq_sum += np.square(flat64).sum(axis=0)
        feat_nonzero_count += (np.abs(flat64) > 1e-12).sum(axis=0)
        total_count += flat.shape[0]
        del batch_seq, flat, flat64

    mean = (feat_sum / total_count).astype(np.float32)
    var = np.maximum((feat_sq_sum / total_count) - np.square(mean.astype(np.float64)), 0.0)
    std = np.sqrt(var).astype(np.float32)
    nonzero_ratio = (feat_nonzero_count / np.maximum(total_count, 1)).astype(np.float32)

    zero_var_mask = std < STD_EPS
    n_zero = int(zero_var_mask.sum())
    n_useful = int((~zero_var_mask).sum())

    std_safe = np.maximum(std, STD_EPS)

    logger.info(f"  Normalization: {n_useful}/{len(mean)} useful features, {n_zero} zero-variance")

    return {
        'mean': mean,
        'std': std_safe,
        'zero_variance_mask': zero_var_mask,
        'nonzero_ratio': nonzero_ratio,
        'n_useful': n_useful,
        'n_zero': n_zero,
    }


def _apply_sparse_modality_keep_mask(norm_stats: dict, feat_dim: int) -> dict:
    """
    Relax strict zero-variance masking for sparse modalities (TQ/OHLC).
    Keeps channels that are genuinely sparse but not fully dead.
    """
    zmask = norm_stats['zero_variance_mask'].copy()
    nz = norm_stats.get('nonzero_ratio', None)
    if nz is None:
        return norm_stats
    for mod_name, min_ratio in SPARSE_KEEP_MIN_NONZERO_RATIO.items():
        ranges = MODALITY_RANGES.get(mod_name, [])
        idx = _modality_indices(feat_dim, ranges)
        if idx.size == 0:
            continue
        resurrect = zmask[idx] & (nz[idx] >= float(min_ratio))
        if np.any(resurrect):
            zmask[idx[resurrect]] = False
    norm_stats['zero_variance_mask'] = zmask
    norm_stats['n_zero'] = int(zmask.sum())
    norm_stats['n_useful'] = int((~zmask).sum())
    logger.info(
        "  Sparse-aware mask: %d/%d useful features, %d zero-variance",
        norm_stats['n_useful'],
        len(zmask),
        norm_stats['n_zero'],
    )
    return norm_stats


def _modality_indices(feat_dim: int, ranges: list[tuple[int, int]]) -> np.ndarray:
    idx = []
    for start, end in ranges:
        s = max(0, int(start))
        e = min(int(end), feat_dim)
        if e > s:
            idx.extend(range(s, e))
    return np.asarray(idx, dtype=np.int64)


def _save_modality_norm_stats(out_dir: Path, norm_stats: dict, feat_dim: int) -> dict:
    """Persist per-modality normalization files for greek / tq / ohlc."""
    saved = {}
    for name, ranges in MODALITY_RANGES.items():
        idx = _modality_indices(feat_dim, ranges)
        if idx.size == 0:
            continue
        np.savez(
            out_dir / f"norm_stats_{name}.npz",
            indices=idx,
            mean=norm_stats["mean"][idx],
            std=norm_stats["std"][idx],
            zero_variance_mask=norm_stats["zero_variance_mask"][idx],
        )
        live_idx = idx[~norm_stats["zero_variance_mask"][idx]]
        np.save(out_dir / f"live_feature_indices_{name}.npy", live_idx)
        saved[name] = int(idx.size)
    return saved


def _group_live_zero_counts(zero_var_mask: np.ndarray, feat_dim: int) -> dict:
    out = {}
    for name, ranges in FEATURE_GROUP_DIAGNOSTIC_RANGES.items():
        idx = _modality_indices(feat_dim, ranges)
        if idx.size == 0:
            out[name] = {"total": 0, "live": 0, "zero": 0}
            continue
        zero = int(zero_var_mask[idx].sum())
        total = int(idx.size)
        out[name] = {"total": total, "live": total - zero, "zero": zero}
    return out


def build_binary_sequences(symbol: str, horizons: list, seq_len: int = SEQ_LEN,
                           return_threshold: float = 0.0,
                           add_delta: bool = False,
                           strip_zero_variance: bool = False,
                           split_mode: str = 'calendar'):
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
        ts_col = 'datetime' if 'datetime' in schema_names else ('timestamp' if 'timestamp' in schema_names else None)
        if ts_col is not None:
            table_dt = pq.read_table(minute_file, columns=[ts_col])
            dt_vals = []
            for chunk in table_dt.column(ts_col).chunks:
                dt_vals.append(chunk.to_pandas())
            # Keep naive timestamps as-is; store as int64 ns for stable joins later.
            all_timestamps = pd.to_datetime(pd.concat(dt_vals, ignore_index=True), errors='coerce').values.astype('datetime64[ns]')
            if len(all_timestamps) != N_ROWS:
                raise RuntimeError(f"{symbol}: datetime length mismatch ({len(all_timestamps)} != {N_ROWS})")
            del table_dt, dt_vals
            gc.collect()
        else:
            logger.warning(f"{symbol}: No datetime column in tier2; using synthetic monotonic timestamps")
            all_timestamps = np.arange(N_ROWS, dtype=np.int64).astype('datetime64[ns]')
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
                        all_chain[idx] = np.array(row, dtype=np.float32).reshape(5, CHAIN_INPUT_STRIKES)  # 5×30 already
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

    # Enhanced NaN/Inf detection and replacement with quality metrics
    nan_count = np.isnan(all_features).sum()
    inf_count = np.isinf(all_features).sum()
    total_elements = all_features.size

    if nan_count > 0 or inf_count > 0:
        nan_pct = 100.0 * nan_count / total_elements
        inf_pct = 100.0 * inf_count / total_elements

        if nan_pct > 5.0 or inf_pct > 5.0:
            logger.error(f"{symbol}: HIGH NaN/Inf contamination - NaN: {nan_count} ({nan_pct:.2f}%), "
                        f"Inf: {inf_count} ({inf_pct:.2f}%)")
        else:
            logger.warning(f"{symbol}: Replacing {nan_count} NaN ({nan_pct:.3f}%), "
                          f"{inf_count} Inf ({inf_pct:.3f}%) with 0")

        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Run quality check before sequence building
    try:
        quality_checker = DataQualityChecker(
            missing_threshold=0.05,
            zero_threshold=0.95,
            constant_threshold=1e-10
        )
        quality_report = quality_checker.check_features(all_features)

        logger.info(f"{symbol}: Pre-sequence quality - overall={quality_report.overall_quality:.2%}, "
                   f"zero={quality_report.zero_pct:.2%}")

        if quality_report.errors:
            logger.error(f"{symbol}: Pre-sequence quality errors: {'; '.join(quality_report.errors[:3])}")

        n_high_zero_features = sum(1 for fm in quality_report.feature_metrics if fm.zero_pct > 0.95)
        if n_high_zero_features > FEAT_DIM * 0.2:
            logger.warning(f"{symbol}: {n_high_zero_features}/{FEAT_DIM} features are >95% zero")
    except Exception as e:
        logger.warning(f"{symbol}: Pre-sequence quality check failed: {e}")

    tq_idx = _modality_indices(FEAT_DIM, MODALITY_RANGES.get("tq", []))
    csv_idx = np.arange(CSV_FEAT_START, min(CSV_FEAT_END, FEAT_DIM), dtype=np.int64)
    tq_active_mask = (
        np.any(np.abs(all_features[:, tq_idx]) > 1e-8, axis=1)
        if tq_idx.size > 0
        else np.zeros(len(all_features), dtype=bool)
    )
    csv_active_mask = (
        np.any(np.abs(all_features[:, csv_idx]) > 1e-8, axis=1)
        if csv_idx.size > 0
        else np.zeros(len(all_features), dtype=bool)
    )
    tq_feature_coverage = float(tq_active_mask.mean()) if len(tq_active_mask) > 0 else 0.0
    csv_feature_coverage = float(csv_active_mask.mean()) if len(csv_active_mask) > 0 else 0.0
    logger.info(f"{symbol}: TQ feature coverage={tq_feature_coverage * 100:.1f}% of minutes")
    logger.info(f"{symbol}: CSV tail coverage={csv_feature_coverage * 100:.1f}% of minutes")

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
            if split_mode == 'after_filter':
                keep_idx = np.flatnonzero(keep_mask)
                n_filtered_total = len(keep_idx)
                train_end_f = int(0.6 * n_filtered_total)
                val_end_f = int(0.8 * n_filtered_total)
                train_idx = keep_idx[:train_end_f]
                val_idx = keep_idx[train_end_f:val_end_f]
                test_idx = keep_idx[val_end_f:]
            else:
                # Calendar-stable split: keep original temporal boundaries, then filter inside each bucket.
                train_idx = np.flatnonzero(keep_mask[:train_end])
                val_idx = train_end + np.flatnonzero(keep_mask[train_end:val_end])
                test_idx = val_end + np.flatnonzero(keep_mask[val_end:])
                keep_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0).astype(np.int64)
        else:
            keep_idx = np.arange(n_samples, dtype=np.int64)
            train_idx = keep_idx[:train_end]
            val_idx = keep_idx[train_end:val_end]
            test_idx = keep_idx[val_end:]

        if len(train_idx) == 0:
            logger.error(f"{symbol} h{horizon}: no training samples after filtering")
            continue

        final_feat_dim = FEAT_DIM * 2 if add_delta else FEAT_DIM  # reset each horizon
        norm_stats = _compute_normalization_stats_batched(all_features, train_idx, seq_len, add_delta)
        norm_stats = _apply_sparse_modality_keep_mask(norm_stats, final_feat_dim)

        out_dir = OUTPUT_ROOT / symbol / f"horizon_{horizon}min"
        out_dir.mkdir(parents=True, exist_ok=True)

        live_idx = np.where(~norm_stats['zero_variance_mask'])[0]
        np.save(out_dir / 'live_feature_indices.npy', live_idx)
        if strip_zero_variance and norm_stats['n_zero'] > 0:
            final_feat_dim = len(live_idx)
            logger.info(f"  {symbol} h{horizon}: strip_zero_variance — keeping {final_feat_dim} live features")
            # Use a per-horizon view — do NOT mutate all_features (other horizons still need full 286 dims)
            horizon_features = all_features[:, live_idx]
            norm_stats['mean'] = norm_stats['mean'][live_idx]
            norm_stats['std']  = norm_stats['std'][live_idx]
        else:
            horizon_features = all_features

        split_specs = [
            ('train', train_idx),
            ('val', val_idx),
            ('test', test_idx),
        ]

        split_sizes = {}
        chain_norm_sum = None
        chain_norm_sq_sum = None
        chain_norm_count = 0
        for split_name, split_idx in split_specs:
            split_sizes[split_name] = len(split_idx)
            np.save(out_dir / f'{split_name}_labels.npy', labels[split_idx])
            np.save(out_dir / f'{split_name}_returns.npy', returns[split_idx])
            # Save timestamp + source-index lineage so downstream stages can align cross-symbol by time.
            split_src_idx = (split_idx + (seq_len - 1)).astype(np.int64)
            split_ts_ns = all_timestamps[split_src_idx].astype('datetime64[ns]').astype(np.int64)
            np.save(out_dir / f'{split_name}_source_indices.npy', split_src_idx)
            np.save(out_dir / f'{split_name}_timestamps.npy', split_ts_ns)

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
                batch_seq = _build_sequence_batch(horizon_features, batch_indices, seq_len, add_delta)
                seq_mem[write_offset:write_offset + len(batch_indices)] = batch_seq

                if chain_mem is not None:
                    batch_chain = _build_chain_batch(all_chain, batch_indices, seq_len)
                    chain_mem[write_offset:write_offset + len(batch_indices)] = batch_chain
                    if split_name == 'train':
                        # Per-channel stats over N x strikes x timesteps for Agent2D normalization.
                        ch = batch_chain.astype(np.float64, copy=False)
                        if chain_norm_sum is None:
                            chain_norm_sum = np.zeros(ch.shape[1], dtype=np.float64)
                            chain_norm_sq_sum = np.zeros(ch.shape[1], dtype=np.float64)
                        chain_norm_sum += ch.sum(axis=(0, 2, 3))
                        chain_norm_sq_sum += np.square(ch).sum(axis=(0, 2, 3))
                        chain_norm_count += int(ch.shape[0] * ch.shape[2] * ch.shape[3])

                write_offset += len(batch_indices)
                del batch_seq
                if chain_mem is not None:
                    try:
                        del batch_chain
                    except Exception:
                        pass

            del seq_mem
            if chain_mem is not None:
                del chain_mem
            gc.collect()

        np.save(out_dir / 'norm_mean.npy', norm_stats['mean'])
        np.save(out_dir / 'norm_std.npy', norm_stats['std'])
        np.save(out_dir / 'zero_variance_mask.npy', norm_stats['zero_variance_mask'])
        # Keep modality/group accounting aligned with the saved feature layout
        # (post-strip when strip_zero_variance is enabled).
        stats_feat_dim = int(len(norm_stats['mean']))
        modality_norm_dims = _save_modality_norm_stats(out_dir, norm_stats, stats_feat_dim)
        feature_group_live_counts = _group_live_zero_counts(norm_stats['zero_variance_mask'], stats_feat_dim)
        logger.info(
            "  Group live dims: greek=%d/%d tq=%d/%d ohlc=%d/%d",
            feature_group_live_counts["greek_core"]["live"], feature_group_live_counts["greek_core"]["total"],
            feature_group_live_counts["tq_slice"]["live"], feature_group_live_counts["tq_slice"]["total"],
            feature_group_live_counts["ohlc_block"]["live"], feature_group_live_counts["ohlc_block"]["total"],
        )

        has_chain_norm = False
        if all_chain is not None and chain_norm_sum is not None and chain_norm_count > 0:
            chain_mean = (chain_norm_sum / chain_norm_count).astype(np.float32)
            chain_var = np.maximum((chain_norm_sq_sum / chain_norm_count) - np.square(chain_mean.astype(np.float64)), 0.0)
            chain_std = np.maximum(np.sqrt(chain_var).astype(np.float32), STD_EPS)
            np.save(out_dir / 'chain_norm_mean.npy', chain_mean)
            np.save(out_dir / 'chain_norm_std.npy', chain_std)
            has_chain_norm = True

        train_up = float(labels[train_idx].mean()) if len(train_idx) else 0.0
        val_up = float(labels[val_idx].mean()) if len(val_idx) else 0.0
        test_up = float(labels[test_idx].mean()) if len(test_idx) else 0.0
        up_pct = 100 * train_up
        logger.info(f"  {symbol} h{horizon}: train={split_sizes['train']:,} val={split_sizes['val']:,} test={split_sizes['test']:,}")
        logger.info(f"  UP ratio: train={train_up:.4f} val={val_up:.4f} test={test_up:.4f}")
        logger.info(f"  Saved to: {out_dir}")

        metadata = {
            'symbol': symbol,
            'horizon_min': horizon,
            'feature_schema_version': FEATURE_SCHEMA_VERSION,
            'seq_len': seq_len,
            'feat_dim': final_feat_dim,
            'n_samples': n_samples,
            'n_samples_after_filter': int(keep_mask.sum()) if return_threshold > 0 else n_samples,
            'return_threshold': return_threshold,
            'split_mode': split_mode,
            'add_delta_features': add_delta,
            'strip_zero_variance': strip_zero_variance,
            'train_size': split_sizes['train'],
            'val_size': split_sizes['val'],
            'test_size': split_sizes['test'],
            'up_pct_train': round(float(up_pct), 2),
            'up_ratio_train': round(train_up, 6),
            'up_ratio_val': round(val_up, 6),
            'up_ratio_test': round(test_up, 6),
            'majority_baseline_acc_train': round(max(train_up, 1.0 - train_up), 6),
            'majority_baseline_acc_val': round(max(val_up, 1.0 - val_up), 6),
            'majority_baseline_acc_test': round(max(test_up, 1.0 - test_up), 6),
            'useful_features': norm_stats['n_useful'],
            'zero_variance_features': norm_stats['n_zero'],
            'has_chain_2d': all_chain is not None,
            'chain_input_strikes': CHAIN_INPUT_STRIKES if all_chain is not None else 0,
            'chain_output_strikes': CHAIN_OUTPUT_STRIKES if all_chain is not None else 0,
            'chain_strike_start': CHAIN_STRIKE_START if all_chain is not None else None,
            'tq_feature_coverage': round(tq_feature_coverage, 4),
            'csv_feature_coverage': round(csv_feature_coverage, 4),
            'modality_norm_dims': modality_norm_dims,
            'feature_group_live_counts': feature_group_live_counts,
            'has_chain_norm': has_chain_norm,
        }
        with open(out_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save quality report for sequences
        try:
            quality_report_path = out_dir / 'quality_report.json'
            from hybrid55_preprocessing.quality_checks import save_quality_report

            # Load a sample of the sequences for quality check
            X_train_sample = np.load(out_dir / 'X_train.npy', mmap_mode='r')
            sample_size = min(10000, len(X_train_sample))
            X_sample = X_train_sample[:sample_size]

            quality_checker = DataQualityChecker(
                missing_threshold=0.05,
                zero_threshold=0.95,
                constant_threshold=1e-10
            )
            quality_dict = quality_checker.check_sequence(X_sample,
                chain_2d=np.load(out_dir / 'chain_2d_train.npy', mmap_mode='r')[:sample_size]
                if (out_dir / 'chain_2d_train.npy').exists() else None)

            with open(quality_report_path, 'w') as qf:
                json.dump(quality_dict, qf, indent=2)

            logger.info(f"  Quality report saved to {quality_report_path}")
        except Exception as e:
            logger.warning(f"  Failed to save quality report: {e}")

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
    parser.add_argument('--split-mode', choices=['calendar', 'after_filter'], default='calendar',
                        help="Split policy: 'calendar' keeps 60/20/20 time boundaries then applies flat filter; "
                             "'after_filter' reproduces legacy split over filtered samples.")
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
    print(f"  Feature schema:     {FEATURE_SCHEMA_VERSION}")
    print(f"  Return threshold:   {args.return_threshold} ({'disabled' if args.return_threshold == 0 else 'enabled'})")
    print(f"  Delta features:     {'ON' if args.add_delta_features else 'OFF'}")
    print(f"  Zero-var mask:      {'ON' if args.strip_zero_variance else 'OFF'}")
    print(f"  Split mode:         {args.split_mode}")
    print("=" * 70)

    all_results = {}
    for sym in symbols:
        result = build_binary_sequences(
            sym, args.horizons, args.seq_len,
            return_threshold=args.return_threshold,
            add_delta=args.add_delta_features,
            strip_zero_variance=args.strip_zero_variance,
            split_mode=args.split_mode,
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
