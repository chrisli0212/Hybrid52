from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Tier3BinaryPaths:
    root: Path
    symbol: str
    horizon_min: int

    @property
    def dir(self) -> Path:
        return self.root / self.symbol / f'horizon_{self.horizon_min}min'


class NumpySequenceDataset(Dataset):
    """Dataset wrapping numpy arrays for (seq, label) with optional z-score norm."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray,
                 norm_mean: Optional[np.ndarray] = None,
                 norm_std: Optional[np.ndarray] = None):
        self.sequences = sequences.astype(np.float32)
        self.labels    = labels.astype(np.float32)
        self.norm_mean = norm_mean.astype(np.float32) if norm_mean is not None else None
        self.norm_std  = norm_std.astype(np.float32)  if norm_std  is not None else None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = torch.from_numpy(self.sequences[idx])
        if self.norm_mean is not None and self.norm_std is not None:
            seq = (seq - torch.from_numpy(self.norm_mean)) / torch.from_numpy(self.norm_std)
        return seq, torch.tensor(self.labels[idx])


def load_tier3_binary(paths: Tier3BinaryPaths):
    d = paths.dir
    train_seq = np.load(d / 'train_sequences.npy')
    train_labels = np.load(d / 'train_labels.npy')
    train_returns = np.load(d / 'train_returns.npy')

    val_seq = np.load(d / 'val_sequences.npy')
    val_labels = np.load(d / 'val_labels.npy')
    val_returns = np.load(d / 'val_returns.npy')

    test_seq = np.load(d / 'test_sequences.npy')
    test_labels = np.load(d / 'test_labels.npy')
    test_returns = np.load(d / 'test_returns.npy')

    return (train_seq, train_labels, train_returns), (val_seq, val_labels, val_returns), (test_seq, test_labels, test_returns)
