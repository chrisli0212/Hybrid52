from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class ZScoreStats:
    mean: np.ndarray
    std: np.ndarray
    zero_variance_mask: np.ndarray

    def save(self, out_dir: Path, prefix: str = '') -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f'{prefix}norm_mean.npy', self.mean)
        np.save(out_dir / f'{prefix}norm_std.npy', self.std)
        np.save(out_dir / f'{prefix}zero_variance_mask.npy', self.zero_variance_mask)

    @staticmethod
    def load(in_dir: Path, prefix: str = '') -> 'ZScoreStats':
        mean = np.load(in_dir / f'{prefix}norm_mean.npy')
        std = np.load(in_dir / f'{prefix}norm_std.npy')
        zero = np.load(in_dir / f'{prefix}zero_variance_mask.npy')
        return ZScoreStats(mean=mean, std=std, zero_variance_mask=zero)


def compute_zscore_stats(train_features_flat: np.ndarray, eps: float = 1e-8) -> ZScoreStats:
    """Compute per-feature z-score stats from flattened training features.

    Args:
        train_features_flat: shape (N, D)
    """
    mean = train_features_flat.mean(axis=0).astype(np.float32)
    std = train_features_flat.std(axis=0).astype(np.float32)
    zero_mask = std < eps
    std_safe = std.copy()
    std_safe[zero_mask] = 1.0
    return ZScoreStats(mean=mean, std=std_safe, zero_variance_mask=zero_mask)


def apply_zscore(seqs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalization to sequences.

    seqs: (N, T, D) or (T, D)
    mean/std: (D,)
    """
    return (seqs - mean) / std
