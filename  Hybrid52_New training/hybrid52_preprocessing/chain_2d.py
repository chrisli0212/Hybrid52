"""
Hybrid51 Option Chain 2D Processor (Task 10)
Creates 2D tensor representation of option chain for Agent-2D CNN.
Shape: (n_greeks, n_strikes, n_timesteps) = (5, 30, 20)

Added in v2:
  - build_from_csv_dir() classmethod: wraps build_chain_2d() CLI builder so
    training_pipeline.py / MasterFeatureExtractor can call it directly in one line.
  - load_from_npy() classmethod: load a pre-built .npy and return (batch, timestamps).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .feature_config import CHAIN_2D_CONFIG


class Chain2DProcessor:
    def __init__(
        self,
        n_greeks: int = 5,
        n_strikes: int = 30,
        n_timesteps: int = 20,
        greeks: List[str] = None,
        delta_range: Tuple[float, float] = (-0.9, 0.9)
    ):
        self.n_greeks = n_greeks
        self.n_strikes = n_strikes
        self.n_timesteps = n_timesteps
        self.greeks = greeks or ['delta', 'gamma', 'vega', 'theta', 'implied_vol']
        self.delta_range = delta_range

        self.delta_bins = np.linspace(delta_range[0], delta_range[1], n_strikes + 1)

    # ── constructors ──────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls) -> 'Chain2DProcessor':
        return cls(
            n_greeks=CHAIN_2D_CONFIG['n_greeks'],
            n_strikes=CHAIN_2D_CONFIG['n_strikes'],
            n_timesteps=CHAIN_2D_CONFIG['n_timesteps'],
            greeks=CHAIN_2D_CONFIG['greeks'],
            delta_range=CHAIN_2D_CONFIG['delta_range']
        )

    @classmethod
    def build_from_csv_dir(
        cls,
        raw_dir: str,
        out_dir: str,
        symbol: str = "SPXW",
        min_bid: float = 0.05,
        ts_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build chain_2d .npy files from raw Theta Data CSVs and return
        (batch, timestamps) so training_pipeline.py can use them directly.

        Example::

            batch, ts = Chain2DProcessor.build_from_csv_dir(
                raw_dir="/workspace/data/raw/options",
                out_dir="/workspace/data/chain_2d",
            )
            # batch.shape == (N, 5, 30, 20)
        """
        from .build_chain_2d import build_chain_2d

        proc = cls.from_config()
        out_npy = build_chain_2d(
            raw_dir=raw_dir,
            out_dir=out_dir,
            symbol=symbol,
            n_strikes=proc.n_strikes,
            n_timesteps=proc.n_timesteps,
            greeks=proc.greeks,
            delta_range=proc.delta_range,
            min_bid=min_bid,
            ts_col=ts_col,
        )
        return cls.load_from_npy(out_dir, symbol)

    @classmethod
    def load_from_npy(
        cls,
        out_dir: str,
        symbol: str = "SPXW",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a pre-built chain_2d .npy file.

        Returns:
            batch      (N, n_greeks, n_strikes, n_timesteps)  float32
            timestamps (N,)                                   str
        """
        p = Path(out_dir)
        batch = np.load(p / f"{symbol}_chain_2d_train.npy")
        ts    = np.load(p / f"{symbol}_chain_2d_timestamps.npy")
        return batch, ts

    # ── online (real-time) helpers ────────────────────────────────────────────
    def get_delta_bin(self, delta: float) -> int:
        if delta <= self.delta_range[0]:
            return 0
        if delta >= self.delta_range[1]:
            return self.n_strikes - 1
        bin_idx = np.searchsorted(self.delta_bins, delta) - 1
        return max(0, min(self.n_strikes - 1, bin_idx))

    def snapshot_to_slice(self, df: pd.DataFrame) -> np.ndarray:
        slice_2d = np.zeros((self.n_greeks, self.n_strikes), dtype=np.float32)

        if 'delta' not in df.columns or df.empty:
            return slice_2d

        delta_vals = df['delta'].values
        delta_bin  = np.digitize(delta_vals, self.delta_bins) - 1
        delta_bin  = np.clip(delta_bin, 0, self.n_strikes - 1)
        bin_series = pd.Series(delta_bin, index=df.index)

        for greek_idx, greek in enumerate(self.greeks):
            if greek not in df.columns:
                continue
            gb = df.groupby(bin_series)[greek].mean()
            slice_2d[greek_idx, :] = gb.reindex(range(self.n_strikes), fill_value=0.0).values

        return slice_2d

    def process_sequence(
        self,
        snapshots: List[pd.DataFrame],
        pad_mode: str = 'zero'
    ) -> np.ndarray:
        chain_3d = np.zeros(
            (self.n_greeks, self.n_strikes, self.n_timesteps),
            dtype=np.float32
        )

        n_available = min(len(snapshots), self.n_timesteps)

        for t in range(n_available):
            snapshot_idx = len(snapshots) - n_available + t
            chain_3d[:, :, t] = self.snapshot_to_slice(snapshots[snapshot_idx])

        if n_available < self.n_timesteps and pad_mode == 'repeat':
            for t in range(n_available, self.n_timesteps):
                chain_3d[:, :, t] = chain_3d[:, :, n_available - 1]

        return chain_3d

    def normalize(
        self,
        chain_3d: np.ndarray,
        method: str = 'zscore'
    ) -> np.ndarray:
        chain_norm = chain_3d.copy()

        for greek_idx in range(self.n_greeks):
            greek_data = chain_norm[greek_idx]

            if method == 'zscore':
                mean = greek_data.mean()
                std  = greek_data.std()
                if std > 1e-10:
                    chain_norm[greek_idx] = (greek_data - mean) / std
            elif method == 'minmax':
                min_val = greek_data.min()
                max_val = greek_data.max()
                if max_val - min_val > 1e-10:
                    chain_norm[greek_idx] = (greek_data - min_val) / (max_val - min_val)
            elif method == 'robust':
                median = np.median(greek_data)
                q1  = np.percentile(greek_data, 25)
                q3  = np.percentile(greek_data, 75)
                iqr = q3 - q1
                if iqr > 1e-10:
                    chain_norm[greek_idx] = (greek_data - median) / iqr

        return chain_norm

    def extract_batch(
        self,
        all_snapshots: List[List[pd.DataFrame]],
        normalize: bool = True,
        normalize_method: str = 'zscore'
    ) -> np.ndarray:
        n_samples = len(all_snapshots)
        batch = np.zeros(
            (n_samples, self.n_greeks, self.n_strikes, self.n_timesteps),
            dtype=np.float32
        )

        for i, snapshots in enumerate(all_snapshots):
            chain_3d = self.process_sequence(snapshots)
            if normalize:
                chain_3d = self.normalize(chain_3d, normalize_method)
            batch[i] = chain_3d

        return batch

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        return (self.n_greeks, self.n_strikes, self.n_timesteps)

    @property
    def total_elements(self) -> int:
        return self.n_greeks * self.n_strikes * self.n_timesteps


# ── convenience functions (unchanged) ────────────────────────────────────────
def create_chain_2d_from_df(
    df: pd.DataFrame,
    n_strikes: int = 30,
    greeks: List[str] = None
) -> np.ndarray:
    processor = Chain2DProcessor(
        n_greeks=len(greeks) if greeks else 5,
        n_strikes=n_strikes,
        n_timesteps=1,
        greeks=greeks
    )
    return processor.snapshot_to_slice(df)


def visualize_chain_2d(chain_2d: np.ndarray, greek_names: List[str] = None) -> str:
    n_greeks, n_strikes = chain_2d.shape[:2]

    if greek_names is None:
        greek_names = CHAIN_2D_CONFIG['greeks']

    lines = []
    for greek_idx in range(n_greeks):
        name    = greek_names[greek_idx] if greek_idx < len(greek_names) else f"greek_{greek_idx}"
        values  = chain_2d[greek_idx]
        lines.append(
            f"{name}: min={values.min():.4f}  max={values.max():.4f}  mean={values.mean():.4f}"
        )

    return "\n".join(lines)
