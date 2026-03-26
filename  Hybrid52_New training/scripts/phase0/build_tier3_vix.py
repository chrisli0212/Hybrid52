#!/usr/bin/env python3
"""Build tier3 VIX regime features from VIXW minute parquet."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


TIER2_ROOT = Path("/workspace/data/tier2_minutes_v4")
OUTPUT_ROOT = Path("/workspace/data/tier3_vix_expanded")
VIX_SYMBOL = "VIXW"
VIX_FEAT_DIM = 10
RESAMPLE_FREQ = "5min"
LOOKBACK = 4
RAW_VIXW_DIR = Path("/workspace/historical_data_1yr/VIXW")
RAW_VIXW_TQ_DIR = Path("/workspace/historical_data_1yr/VIXW/OI")
REGIME_THRESHOLDS = [15.0, 20.0, 25.0, 35.0]
REGIME_NAMES = ["CALM", "NORMAL", "ELEVATED", "HIGH", "EXTREME"]
DEFAULT_RESOLUTION = "1min"


def _resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Need datetime index or datetime column")

    if "close" not in df.columns:
        if "underlying_price" in df.columns:
            df["close"] = df["underlying_price"]
        else:
            raise ValueError("Missing close/underlying_price")

    if "open" not in df.columns:
        df["open"] = df["close"]
    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    if "volume" not in df.columns:
        if "trade_count" in df.columns:
            df["volume"] = df["trade_count"]
        else:
            df["volume"] = 0.0

    df["session_date"] = df.index.date
    out = (
        df.groupby("session_date")
        .resample(RESAMPLE_FREQ, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .droplevel(0)
        .dropna(subset=["close"])
    )
    return out


def _extract_features(df_bars: pd.DataFrame, resolution: str = "5min") -> pd.DataFrame:
    """Extract VIX regime features. Window sizes adapt to bar resolution."""
    close = df_bars["close"].astype(float)
    open_ = df_bars["open"].astype(float)
    high = df_bars["high"].astype(float)
    low = df_bars["low"].astype(float)

    # Adapt rolling/shift sizes: base is 5-min bars
    m = 1 if resolution == "5min" else 5  # multiplier for 1-min bars

    feat = pd.DataFrame(index=df_bars.index)
    feat["vix_level"] = close
    feat["vix_pct_5m"] = np.where(open_ != 0, (close - open_) / np.abs(open_), 0.0)
    s15 = 3 * m   # 15 min
    feat["vix_pct_15m"] = np.where(close.shift(s15) != 0, (close - close.shift(s15)) / np.abs(close.shift(s15)), 0.0)
    s60 = 12 * m  # 60 min
    feat["vix_pct_1h"] = np.where(close.shift(s60) != 0, (close - close.shift(s60)) / np.abs(close.shift(s60)), 0.0)

    r100 = 20 * m  # 100-min rolling window
    roll_mean = close.rolling(r100, min_periods=5 * m).mean()
    roll_std = close.rolling(r100, min_periods=5 * m).std()
    feat["vix_zscore_15m"] = np.where(roll_std > 1e-6, (close - roll_mean) / roll_std, 0.0)

    r60 = 12 * m   # 60-min rolling window
    feat["vix_percentile_1h"] = close.rolling(r60, min_periods=3 * m).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / max(len(x) - 1, 1), raw=False
    )
    s30 = 6 * m    # 30 min
    feat["vix_term_slope"] = np.where(close.shift(s30) != 0, (close - close.shift(s30)) / np.abs(close.shift(s30)), 0.0)
    ret = close.pct_change().fillna(0.0)
    feat["vvix_level"] = ret.rolling(r100, min_periods=5 * m).std().fillna(0.0)
    sess_open = close.groupby(close.index.date).transform("first")
    feat["vix_vix1d_spread"] = np.where(sess_open > 0, (close - sess_open) / sess_open, 0.0)
    feat["vix_hilo_range"] = np.where(open_ != 0, (high - low) / np.abs(open_), 0.0)

    feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    return feat


def _prepare_1min_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare 1-min OHLCV bars from raw minute parquet."""
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Need datetime index or datetime column")

    if "close" not in df.columns:
        if "underlying_price" in df.columns:
            df["close"] = df["underlying_price"]
        else:
            raise ValueError("Missing close/underlying_price")

    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = df.get("trade_count", 0.0)

    return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def _assign_labels(vix_levels: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(vix_levels), dtype=np.int64)
    for i, thr in enumerate(REGIME_THRESHOLDS):
        labels[vix_levels >= thr] = i + 1
    return labels


def _build_sequences(features: np.ndarray, labels: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    n = features.shape[0] - lookback + 1
    if n <= 0:
        raise ValueError("Not enough bars for lookback")
    x = np.zeros((n, lookback, features.shape[1]), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        x[i] = features[i : i + lookback]
        y[i] = labels[i + lookback - 1]
    return x, y


def _norm_stats(train_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    flat = train_seq.reshape(-1, train_seq.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    zmask = std < 1e-8
    std[zmask] = 1.0
    return mean, std, int(zmask.sum())


def _align_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def build_vix_tier3(
    tier2_root: Path,
    output_root: Path,
    symbol: str,
    lookback: int,
    bootstrap_from_raw: bool,
    resolution: str = DEFAULT_RESOLUTION,
) -> dict:
    t0 = time.time()
    src = tier2_root / f"{symbol}_minutes.parquet"
    source_mode = "existing_tier2"

    if not src.exists() and bootstrap_from_raw:
        logger.info("Tier2 VIXW minute parquet missing, bootstrapping from raw...")
        from scripts.phase0.build_vixw_minutes_from_raw import build_vixw_minutes

        build_vixw_minutes(
            RAW_VIXW_DIR,
            RAW_VIXW_TQ_DIR,
            src,
        )
        source_mode = "bootstrapped_from_raw"

    if not src.exists():
        raise FileNotFoundError(f"Missing source parquet: {src}")

    df = pd.read_parquet(src)

    if resolution == "1min":
        df_bars = _prepare_1min_bars(df)
        feat_df = _extract_features(df_bars, resolution="1min")
        close_arr = df_bars["close"].to_numpy(dtype=np.float32)
        bars_desc = f"{len(df_bars)} 1-min bars"
    else:
        df_bars = _resample_to_5min(df)
        feat_df = _extract_features(df_bars, resolution="5min")
        close_arr = df_bars["close"].to_numpy(dtype=np.float32)
        bars_desc = f"{len(df_bars)} 5-min bars"

    logger.info(f"  Source mode={source_mode} | Resolution={resolution}: {bars_desc}")
    labels = _assign_labels(close_arr)

    seq, seq_labels = _build_sequences(feat_df.to_numpy(dtype=np.float32), labels, lookback=lookback)
    n = len(seq)
    i1 = int(n * 0.6)
    i2 = int(n * 0.8)

    tr_x, va_x, te_x = seq[:i1], seq[i1:i2], seq[i2:]
    tr_y, va_y, te_y = seq_labels[:i1], seq_labels[i1:i2], seq_labels[i2:]

    mean, std, zero_var = _norm_stats(tr_x)
    tr_x = _align_norm(tr_x, mean, std)
    va_x = _align_norm(va_x, mean, std)
    te_x = _align_norm(te_x, mean, std)

    out = output_root / symbol
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_vix_features.npy", tr_x)
    np.save(out / "train_vix_labels.npy", tr_y)
    np.save(out / "val_vix_features.npy", va_x)
    np.save(out / "val_vix_labels.npy", va_y)
    np.save(out / "test_vix_features.npy", te_x)
    np.save(out / "test_vix_labels.npy", te_y)
    np.save(out / "vix_norm_mean.npy", mean)
    np.save(out / "vix_norm_std.npy", std)

    regime_dist = {REGIME_NAMES[i]: int((seq_labels == i).sum()) for i in range(len(REGIME_NAMES))}
    metadata = {
        "symbol": symbol,
        "source": str(src),
        "source_mode": source_mode,
        "raw_source_hint": str(RAW_VIXW_DIR),
        "resolution": resolution,
        "lookback": lookback,
        "vix_feat_dim": VIX_FEAT_DIM,
        "feature_names": list(feat_df.columns),
        "regime_names": REGIME_NAMES,
        "regime_thresholds": REGIME_THRESHOLDS,
        "regime_distribution": regime_dist,
        "source_bars_1min": int(len(df)),
        "bars_used": int(len(df_bars)),
        "sequences": int(n),
        "train_size": int(len(tr_x)),
        "val_size": int(len(va_x)),
        "test_size": int(len(te_x)),
        "zero_variance_features": zero_var,
        "build_time_sec": round(time.time() - t0, 1),
    }
    (out / "vix_metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Saved VIX tier3 -> %s", out)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VIX tier3 features")
    parser.add_argument("--tier2-root", default=str(TIER2_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--symbol", default=VIX_SYMBOL)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--resolution", choices=["1min", "5min"], default=DEFAULT_RESOLUTION)
    parser.add_argument("--bootstrap-from-raw", action="store_true")
    args = parser.parse_args()

    md = build_vix_tier3(
        tier2_root=Path(args.tier2_root),
        output_root=Path(args.output_root),
        symbol=args.symbol,
        lookback=args.lookback,
        bootstrap_from_raw=args.bootstrap_from_raw,
        resolution=args.resolution,
    )
    print(json.dumps(md, indent=2))
