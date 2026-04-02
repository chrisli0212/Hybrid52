#!/usr/bin/env python3
"""
Phase 0 — Build VIX 5-min regime features from tier2 VIXW minute bars.

Pipeline:
    1. Load VIXW 1-min parquet from tier2
    2. Resample to 5-min OHLCV bars (session-safe)
    3. Compute ~10 VIX regime features per 5-min bar
    4. Build aligned sequences for training
    5. Create regime labels from VIX level thresholds
    6. Chronological 60/20/20 split
    7. Z-score normalize using training split stats
    8. Save to tier3_vix_v4/{SYMBOL}/

Usage:
    python scripts/phase0/build_tier3_vix.py
    python scripts/phase0/build_tier3_vix.py --tier2-root /workspace/data/tier2_minutes_v4
    python scripts/phase0/build_tier3_vix.py --lookback 4 --output-root /workspace/data/tier3_vix_v4
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Defaults
# ============================================================================

TIER2_ROOT = Path("/workspace/data/tier2_minutes_v4")
OUTPUT_ROOT = Path("/workspace/data/tier3_vix_v4")

VIX_SYMBOL = "VIXW"       # Source symbol in tier2
VIX_FEAT_DIM = 10          # Number of VIX features
RESAMPLE_FREQ = "5min"     # 5-minute bars
LOOKBACK = 4               # 4 × 5-min = 20-min lookback window
MARKET_OPEN = "09:30"      # ET
MARKET_CLOSE = "16:15"     # ET (VIX trades until 16:15)

# Regime thresholds
REGIME_THRESHOLDS = [15.0, 20.0, 25.0, 35.0]
REGIME_NAMES = ['CALM', 'NORMAL', 'ELEVATED', 'HIGH', 'EXTREME']


# ============================================================================
# Resampling
# ============================================================================

def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-min bars to 5-min bars, session-safe.

    Expects columns: datetime (or index), open, high, low, close, volume
    If 'close' column is named 'spot' or 'price', renames it.
    """
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index or 'datetime' column")

    # Handle common column name variations
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ('spot', 'price', 'last') and 'close' not in df.columns:
            col_map[col] = 'close'
    if col_map:
        df = df.rename(columns=col_map)

    # Ensure required columns exist
    required = ['close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Add session date for session-safe resampling
    df['session_date'] = df.index.date

    # Build OHLCV rules based on available columns
    agg_rules = {'close': 'last'}
    if 'open' in df.columns:
        agg_rules['open'] = 'first'
    if 'high' in df.columns:
        agg_rules['high'] = 'max'
    if 'low' in df.columns:
        agg_rules['low'] = 'min'
    if 'volume' in df.columns:
        agg_rules['volume'] = 'sum'

    # Session-safe resample
    df_5min = (
        df.groupby('session_date')
          .resample(RESAMPLE_FREQ, label='left', closed='left')
          .agg(agg_rules)
          .droplevel(0)  # drop session_date from multi-index
          .dropna(subset=['close'])
    )

    # Fill missing OHLC from close if not present
    if 'open' not in df_5min.columns:
        df_5min['open'] = df_5min['close']
    if 'high' not in df_5min.columns:
        df_5min['high'] = df_5min['close']
    if 'low' not in df_5min.columns:
        df_5min['low'] = df_5min['close']

    logger.info(f"Resampled {len(df)} 1-min bars → {len(df_5min)} 5-min bars")
    return df_5min


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_vix_features(df_5min: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ~10 VIX regime features from 5-min OHLCV bars.

    Returns DataFrame with VIX_FEAT_DIM columns, same index as input.
    """
    features = pd.DataFrame(index=df_5min.index)

    close = df_5min['close'].values
    open_ = df_5min['open'].values
    high = df_5min['high'].values
    low = df_5min['low'].values

    # 0: vix_level — current VIX spot
    features['vix_level'] = close

    # 1: vix_pct_5m — 5-min percentage change
    features['vix_pct_5m'] = np.where(
        open_ != 0,
        (close - open_) / np.abs(open_),
        0.0
    )

    # 2: vix_pct_15m — 15-min pct change (3 bars back)
    close_3back = pd.Series(close, index=df_5min.index).shift(3).values
    features['vix_pct_15m'] = np.where(
        (close_3back != 0) & ~np.isnan(close_3back),
        (close - close_3back) / np.abs(close_3back),
        0.0
    )

    # 3: vix_pct_1h — 1-hour pct change (12 bars back)
    close_12back = pd.Series(close, index=df_5min.index).shift(12).values
    features['vix_pct_1h'] = np.where(
        (close_12back != 0) & ~np.isnan(close_12back),
        (close - close_12back) / np.abs(close_12back),
        0.0
    )

    # 4: vix_zscore_15m — z-score vs 20-bar rolling window
    close_series = pd.Series(close, index=df_5min.index)
    roll_mean = close_series.rolling(20, min_periods=5).mean()
    roll_std = close_series.rolling(20, min_periods=5).std()
    features['vix_zscore_15m'] = np.where(
        roll_std > 0.001,
        (close_series - roll_mean) / roll_std,
        0.0
    )

    # 5: vix_percentile_1h — percentile rank in 12-bar window
    features['vix_percentile_1h'] = close_series.rolling(12, min_periods=3).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / max(len(x) - 1, 1),
        raw=False
    ).fillna(0.5)

    # 6: vix_term_slope — proxy: rate of change over 6 bars (~30 min)
    # True term structure (M2-M1)/M1 requires VX futures data; use slope proxy
    close_6back = close_series.shift(6).values
    features['vix_term_slope'] = np.where(
        (close_6back != 0) & ~np.isnan(close_6back),
        (close - close_6back) / np.abs(close_6back),
        0.0
    )

    # 7: vvix_level — vol-of-vol proxy: rolling std of 5-min returns
    returns_5m = close_series.pct_change().fillna(0)
    features['vvix_level'] = returns_5m.rolling(20, min_periods=5).std().fillna(0)

    # 8: vix_vix1d_spread — intraday spread proxy: current vs session open
    session_open = close_series.groupby(close_series.index.date).transform('first')
    features['vix_vix1d_spread'] = np.where(
        session_open > 0,
        (close_series - session_open) / session_open,
        0.0
    )

    # 9: vix_hilo_range — 5-min high-low range (normalized)
    features['vix_hilo_range'] = np.where(
        open_ != 0,
        (high - low) / np.abs(open_),
        0.0
    )

    # Clean NaN/Inf
    features = features.replace([np.inf, -np.inf], 0.0)
    features = features.fillna(0.0)
    features = features.astype(np.float32)

    logger.info(f"Extracted {len(features.columns)} VIX features for {len(features)} bars")
    return features


# ============================================================================
# Regime Labeling
# ============================================================================

def assign_regime_labels(vix_levels: np.ndarray) -> np.ndarray:
    """
    Assign regime class labels from VIX levels.
    Returns integer array: 0=CALM, 1=NORMAL, 2=ELEVATED, 3=HIGH, 4=EXTREME
    """
    labels = np.zeros(len(vix_levels), dtype=np.int64)
    for i, threshold in enumerate(REGIME_THRESHOLDS):
        labels[vix_levels >= threshold] = i + 1
    return labels


# ============================================================================
# Sequence Building
# ============================================================================

def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    lookback: int = LOOKBACK,
) -> tuple:
    """
    Build rolling sequences of shape (N, lookback, feat_dim).

    Args:
        features: (T, feat_dim) — feature matrix
        labels: (T,) — regime labels per timestep
        lookback: Number of lookback steps

    Returns:
        sequences: (N, lookback, feat_dim)
        seq_labels: (N,) — label of the last timestep in each sequence
    """
    T, feat_dim = features.shape
    N = T - lookback + 1
    if N <= 0:
        raise ValueError(f"Not enough data: {T} bars with lookback={lookback}")

    sequences = np.zeros((N, lookback, feat_dim), dtype=np.float32)
    seq_labels = np.zeros(N, dtype=np.int64)

    for i in range(N):
        sequences[i] = features[i : i + lookback]
        seq_labels[i] = labels[i + lookback - 1]

    logger.info(f"Built {N} sequences of shape ({lookback}, {feat_dim})")
    return sequences, seq_labels


# ============================================================================
# Normalization
# ============================================================================

def compute_normalization_stats(train_features: np.ndarray) -> dict:
    """
    Compute per-feature mean and std from TRAINING data only.
    """
    # Flatten sequences to (N*lookback, feat_dim) for stats
    if train_features.ndim == 3:
        flat = train_features.reshape(-1, train_features.shape[-1])
    else:
        flat = train_features

    mean = np.mean(flat, axis=0).astype(np.float32)
    std = np.std(flat, axis=0).astype(np.float32)

    # Prevent division by zero
    zero_var = std < 1e-8
    std[zero_var] = 1.0

    return {
        'mean': mean,
        'std': std,
        'zero_variance_count': int(zero_var.sum()),
    }


def apply_normalization(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalize."""
    return ((data - mean) / std).astype(np.float32)


# ============================================================================
# Main Pipeline
# ============================================================================

def build_vix_tier3(
    tier2_root: Path = TIER2_ROOT,
    output_root: Path = OUTPUT_ROOT,
    symbol: str = VIX_SYMBOL,
    lookback: int = LOOKBACK,
):
    """Main pipeline: load → resample → extract → sequence → split → normalize → save."""

    t0 = time.time()
    logger.info(f"Building VIX tier3 from {tier2_root / f'{symbol}_minutes.parquet'}")

    # ── 1. Load tier2 1-min data ─────────────────────────────────────────
    parquet_path = tier2_root / f"{symbol}_minutes.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Tier2 file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} 1-min bars from {parquet_path.name}")

    # ── 2. Resample to 5-min ─────────────────────────────────────────────
    df_5min = resample_to_5min(df)

    # ── 3. Extract VIX features ──────────────────────────────────────────
    features_df = extract_vix_features(df_5min)
    features = features_df.values.astype(np.float32)
    vix_levels = df_5min['close'].values.astype(np.float32)

    # ── 4. Create regime labels ──────────────────────────────────────────
    labels = assign_regime_labels(vix_levels)
    regime_dist = {REGIME_NAMES[i]: int((labels == i).sum()) for i in range(len(REGIME_NAMES))}
    logger.info(f"Regime distribution: {regime_dist}")

    # ── 5. Build sequences ───────────────────────────────────────────────
    sequences, seq_labels = build_sequences(features, labels, lookback=lookback)
    N = len(sequences)

    # ── 6. Chronological 60/20/20 split ──────────────────────────────────
    train_end = int(N * 0.6)
    val_end = int(N * 0.8)

    train_seq, train_labels = sequences[:train_end], seq_labels[:train_end]
    val_seq, val_labels = sequences[train_end:val_end], seq_labels[train_end:val_end]
    test_seq, test_labels = sequences[val_end:], seq_labels[val_end:]

    logger.info(f"Split: train={len(train_seq)}, val={len(val_seq)}, test={len(test_seq)}")

    # ── 7. Normalize using training stats ────────────────────────────────
    norm_stats = compute_normalization_stats(train_seq)
    mean, std = norm_stats['mean'], norm_stats['std']
    logger.info(f"Zero-variance features: {norm_stats['zero_variance_count']}/{VIX_FEAT_DIM}")

    train_seq = apply_normalization(train_seq, mean, std)
    val_seq = apply_normalization(val_seq, mean, std)
    test_seq = apply_normalization(test_seq, mean, std)

    # ── 8. Save ──────────────────────────────────────────────────────────
    out_dir = output_root / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "train_vix_features.npy", train_seq)
    np.save(out_dir / "train_vix_labels.npy", train_labels)
    np.save(out_dir / "val_vix_features.npy", val_seq)
    np.save(out_dir / "val_vix_labels.npy", val_labels)
    np.save(out_dir / "test_vix_features.npy", test_seq)
    np.save(out_dir / "test_vix_labels.npy", test_labels)
    np.save(out_dir / "vix_norm_mean.npy", mean)
    np.save(out_dir / "vix_norm_std.npy", std)

    # Metadata
    metadata = {
        'symbol': symbol,
        'resample_freq': RESAMPLE_FREQ,
        'lookback': lookback,
        'vix_feat_dim': VIX_FEAT_DIM,
        'feature_names': list(features_df.columns),
        'num_regimes': len(REGIME_NAMES),
        'regime_names': REGIME_NAMES,
        'regime_thresholds': REGIME_THRESHOLDS,
        'regime_distribution': regime_dist,
        'split': {
            'train': len(train_seq),
            'val': len(val_seq),
            'test': len(test_seq),
        },
        'zero_variance_features': norm_stats['zero_variance_count'],
        'source_bars_1min': len(df),
        'bars_5min': len(df_5min),
        'sequences': N,
        'build_time_sec': round(time.time() - t0, 1),
    }

    with open(out_dir / "vix_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved to {out_dir}/")
    logger.info(f"Build time: {metadata['build_time_sec']}s")

    # Print summary
    print("\n" + "=" * 60)
    print("VIX Tier3 Build Summary")
    print("=" * 60)
    print(f"  Source:     {parquet_path}")
    print(f"  1-min bars: {len(df):,}")
    print(f"  5-min bars: {len(df_5min):,}")
    print(f"  Sequences:  {N:,}")
    print(f"  Shape:      ({lookback}, {VIX_FEAT_DIM})")
    print(f"  Train/Val/Test: {len(train_seq):,} / {len(val_seq):,} / {len(test_seq):,}")
    print(f"  Regimes:    {regime_dist}")
    print(f"  Output:     {out_dir}")
    print("=" * 60)

    return metadata


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build VIX tier3 regime features")
    parser.add_argument('--tier2-root', type=str, default=str(TIER2_ROOT))
    parser.add_argument('--output-root', type=str, default=str(OUTPUT_ROOT))
    parser.add_argument('--symbol', type=str, default=VIX_SYMBOL)
    parser.add_argument('--lookback', type=int, default=LOOKBACK,
                        help="Lookback window in 5-min bars (default: 4 = 20 min)")
    args = parser.parse_args()

    build_vix_tier3(
        tier2_root=Path(args.tier2_root),
        output_root=Path(args.output_root),
        symbol=args.symbol,
        lookback=args.lookback,
    )
