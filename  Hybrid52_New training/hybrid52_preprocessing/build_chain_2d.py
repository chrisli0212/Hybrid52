"""
build_chain_2d.py  —  Standalone CLI builder for chain_2d .npy training files.

Reads Theta Data option chain CSVs (historical greek snapshots) from a directory,
builds rolling (n_greeks, n_strikes, n_timesteps) tensors, normalises per sample,
and writes:
    <OUTPUT_DIR>/
        SPXW_chain_2d_train.npy      shape (N, 5, 30, 20)  float32
        SPXW_chain_2d_timestamps.npy shape (N,)            str
        SPXW_chain_2d_meta.json      build config + stats

Usage:
    python3 build_chain_2d.py \\
        --raw_dir  /workspace/data/raw/options \\
        --out_dir  /workspace/data/chain_2d \\
        --symbol   SPXW \\
        --n_strikes 30 \\
        --n_timesteps 20 \\
        --min_bid 0.05

Integration:
    The output .npy file is consumed by:
      - Hybrid51Dataset  (training_pipeline.py)  via  load_processed_data()
      - Chain2DProcessor.build_from_csv_dir()   (chain_2d.py)
      - MasterFeatureExtractor                  (master_extractor_v2.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ── column name aliases from Theta Data exports ───────────────────────────────
GREEK_COL_ALIASES: dict[str, list[str]] = {
    "delta":        ["delta", "Delta"],
    "gamma":        ["gamma", "Gamma"],
    "vega":         ["vega",  "Vega"],
    "theta":        ["theta", "Theta"],
    "implied_vol":  ["implied_vol", "iv", "impliedVol", "implied_volatility", "IV"],
    "strike":       ["strike", "Strike", "strike_price"],
    "bid":          ["bid",   "Bid",   "bid_price"],
}

DEFAULT_GREEKS = ["delta", "gamma", "vega", "theta", "implied_vol"]


def _resolve_col(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Return the actual column name for a canonical greek name, or None."""
    for alias in GREEK_COL_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return None


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename aliased columns to canonical names in-place."""
    rename_map: dict[str, str] = {}
    for canonical, aliases in GREEK_COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias != canonical:
                rename_map[alias] = canonical
                break
    return df.rename(columns=rename_map)


def _filter_active_chain(df: pd.DataFrame, min_bid: float = 0.05) -> pd.DataFrame:
    """Drop contracts with no real liquidity."""
    if "bid" in df.columns:
        df = df[df["bid"] >= min_bid]
    return df


def _snapshot_to_slice(
    df: pd.DataFrame,
    delta_bins: np.ndarray,
    n_strikes: int,
    greeks: List[str],
) -> np.ndarray:
    """
    Convert one snapshot DataFrame → (n_greeks, n_strikes) slice.
    Contracts are binned by delta into n_strikes delta buckets; each bucket
    gets the mean greek value of all contracts that fall in it.
    """
    n_greeks = len(greeks)
    result = np.zeros((n_greeks, n_strikes), dtype=np.float32)

    if df is None or df.empty or "delta" not in df.columns:
        return result

    delta_vals = df["delta"].values.astype(float)
    bin_idx = np.digitize(delta_vals, delta_bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_strikes - 1)
    bin_series = pd.Series(bin_idx, index=df.index)

    for g_idx, greek in enumerate(greeks):
        if greek not in df.columns:
            continue
        gb = df.groupby(bin_series)[greek].mean()
        result[g_idx, :] = gb.reindex(range(n_strikes), fill_value=0.0).values.astype(np.float32)

    return result


def _normalise_sample(chain_3d: np.ndarray) -> np.ndarray:
    """Z-score per greek channel independently."""
    out = chain_3d.copy()
    for g in range(chain_3d.shape[0]):
        ch = chain_3d[g]
        mu  = ch.mean()
        std = ch.std() + 1e-8
        out[g] = (ch - mu) / std
    return out


def _load_raw_files(raw_dir: Path, symbol: str) -> pd.DataFrame:
    """Load all parquet/csv files matching symbol in raw_dir."""
    files = sorted(raw_dir.glob(f"{symbol}*.parquet")) + \
            sorted(raw_dir.glob(f"{symbol}*.csv"))
    if not files:
        # Try case-insensitive glob
        files = sorted(raw_dir.glob("*.parquet")) + sorted(raw_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No parquet/csv files found in {raw_dir} for symbol '{symbol}'.\n"
            f"Run: find /workspace -name '*.parquet' | head -20"
        )

    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            dfs.append(_normalise_columns(df))
            print(f"  loaded {f.name}  rows={len(df):,}")
        except Exception as e:
            print(f"  SKIP  {f.name}: {e}")

    if not dfs:
        raise RuntimeError("All files failed to load.")

    return pd.concat(dfs, ignore_index=True)


def _find_timestamp_col(df: pd.DataFrame) -> str:
    candidates = ["timestamp", "ts", "datetime", "date", "time", "Timestamp"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"No timestamp column found. Columns: {list(df.columns)[:20]}\n"
        "Set --ts_col manually with the correct column name."
    )


def build_chain_2d(
    raw_dir: str | Path,
    out_dir: str | Path,
    symbol: str = "SPXW",
    n_strikes: int = 30,
    n_timesteps: int = 20,
    greeks: Optional[List[str]] = None,
    delta_range: tuple[float, float] = (-0.9, 0.9),
    min_bid: float = 0.05,
    ts_col: Optional[str] = None,
) -> Path:
    """
    Core builder.  Returns path to saved .npy file.
    Called by CLI and by Chain2DProcessor.build_from_csv_dir().
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    greeks = greeks or DEFAULT_GREEKS
    n_greeks = len(greeks)

    print(f"\n── Loading raw files from {raw_dir} ──")
    df_all = _load_raw_files(raw_dir, symbol)
    print(f"Total rows: {len(df_all):,}  columns: {list(df_all.columns)[:10]}")

    # Resolve timestamp column
    if ts_col is None:
        ts_col = _find_timestamp_col(df_all)
    df_all["_ts"] = pd.to_datetime(df_all[ts_col])
    df_all = df_all.sort_values("_ts").reset_index(drop=True)
    df_all["_ts_1min"] = df_all["_ts"].dt.floor("1min")

    timestamps_1min = sorted(df_all["_ts_1min"].unique())
    print(f"Unique 1-min bars: {len(timestamps_1min)}")

    if len(timestamps_1min) < n_timesteps + 1:
        raise ValueError(
            f"Not enough 1-min bars ({len(timestamps_1min)}) to build windows of {n_timesteps}."
        )

    # Delta bins
    delta_bins = np.linspace(delta_range[0], delta_range[1], n_strikes + 1)

    # Build per-bar snapshots
    print("Building per-bar snapshots...")
    snapshots: list[Optional[pd.DataFrame]] = []
    for ts in timestamps_1min:
        snap = df_all[df_all["_ts_1min"] == ts].copy()
        snap = _filter_active_chain(snap, min_bid)
        snapshots.append(snap if len(snap) > 0 else None)

    # Rolling windows
    print(f"Building {len(snapshots) - n_timesteps} rolling windows...")
    all_tensors: list[np.ndarray] = []
    all_ts: list[str] = []
    skipped = 0

    for i in range(n_timesteps, len(snapshots)):
        window = snapshots[i - n_timesteps: i]
        chain_3d = np.zeros((n_greeks, n_strikes, n_timesteps), dtype=np.float32)

        for t, snap in enumerate(window):
            if snap is not None and len(snap) > 0:
                chain_3d[:, :, t] = _snapshot_to_slice(snap, delta_bins, n_strikes, greeks)

        # Skip windows that are entirely zero (no data)
        if chain_3d.sum() == 0:
            skipped += 1
            continue

        chain_3d = _normalise_sample(chain_3d)
        all_tensors.append(chain_3d)
        all_ts.append(str(timestamps_1min[i]))

    if not all_tensors:
        raise RuntimeError(
            "All windows were empty (all-zero). Check --min_bid threshold or column names."
        )

    batch    = np.stack(all_tensors, axis=0)          # (N, n_greeks, n_strikes, n_timesteps)
    ts_array = np.array(all_ts)

    out_npy = out_dir / f"{symbol}_chain_2d_train.npy"
    out_ts  = out_dir / f"{symbol}_chain_2d_timestamps.npy"
    np.save(out_npy, batch)
    np.save(out_ts,  ts_array)

    meta = {
        "symbol":       symbol,
        "n_samples":    int(len(batch)),
        "shape":        list(batch.shape),
        "greeks":       greeks,
        "n_strikes":    n_strikes,
        "n_timesteps":  n_timesteps,
        "delta_range":  list(delta_range),
        "min_bid":      min_bid,
        "skipped_empty_windows": skipped,
        "stats": {
            "mean":  float(batch.mean()),
            "std":   float(batch.std()),
            "min":   float(batch.min()),
            "max":   float(batch.max()),
            "nonzero_samples": int((batch.sum(axis=(1, 2, 3)) != 0).sum()),
        },
    }
    meta_path = out_dir / f"{symbol}_chain_2d_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅  {out_npy}  shape={batch.shape}")
    print(f"✅  {out_ts}")
    print(f"✅  {meta_path}")
    print(f"   mean={batch.mean():.4f}  std={batch.std():.4f}  "
          f"non-zero={meta['stats']['nonzero_samples']}/{len(batch)}  "
          f"skipped={skipped}")

    return out_npy


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chain_2d .npy files from Theta Data CSVs.")
    p.add_argument("--raw_dir",     required=True,  help="Dir containing raw option chain CSV/parquet files")
    p.add_argument("--out_dir",     required=True,  help="Output directory for .npy files")
    p.add_argument("--symbol",      default="SPXW", help="Underlying symbol prefix (default: SPXW)")
    p.add_argument("--n_strikes",   type=int, default=30,  help="Number of delta-bucketed strike bins")
    p.add_argument("--n_timesteps", type=int, default=20,  help="Rolling window length in 1-min bars")
    p.add_argument("--min_bid",     type=float, default=0.05, help="Min bid to keep contract (liquidity filter)")
    p.add_argument("--ts_col",      default=None, help="Timestamp column name (auto-detected if omitted)")
    p.add_argument("--greeks",      nargs="+", default=None,
                   help="Greek columns to use (default: delta gamma vega theta implied_vol)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        build_chain_2d(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            symbol=args.symbol,
            n_strikes=args.n_strikes,
            n_timesteps=args.n_timesteps,
            greeks=args.greeks,
            min_bid=args.min_bid,
            ts_col=args.ts_col,
        )
    except Exception as e:
        print(f"\n❌  ERROR: {e}", file=sys.stderr)
        sys.exit(1)
