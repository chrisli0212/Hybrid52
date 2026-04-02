"""
Hybrid51 Training Pipeline Integration (Task 14)
PyTorch Dataset and DataLoader for training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import json

from .feature_config_v2 import TOTAL_FEATURES, CHAIN_2D_CONFIG
from .sequence_pipeline import SequenceConfig, SequenceCreator
from .master_extractor import MasterFeatureExtractor
from .quality_checks import validate_preprocessed_data


class Hybrid51Dataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        chain_2d: Optional[np.ndarray] = None,
        normalize: bool = True,
        augment: bool = False
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.chain_2d = torch.from_numpy(chain_2d).float() if chain_2d is not None else None
        
        self.normalize = normalize
        self.augment = augment
        
        if normalize:
            self._compute_normalization_stats()
            self._apply_normalization()
    
    def _compute_normalization_stats(self):
        X_flat = self.X.view(-1, self.X.shape[-1])
        self.mean = X_flat.mean(dim=0, keepdim=True)
        raw_std  = X_flat.std(dim=0, keepdim=True)
        # Mask dead features: std < 1e-6 → set std=1.0 so they normalise to 0 cleanly
        self.dead_mask = (raw_std < 1e-6).squeeze()   # [feat_dim] bool
        self.std = raw_std.clone()
        self.std[self.std < 1e-6] = 1.0               # dead → (x-mean)/1 = 0
    
    def _apply_normalization(self):
        original_shape = self.X.shape
        X_flat = self.X.view(-1, original_shape[-1])
        X_flat = (X_flat - self.mean) / self.std
        self.X = X_flat.view(original_shape)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'features': self.X[idx],
            'label': self.y[idx]
        }
        
        if self.chain_2d is not None:
            item['chain_2d'] = self.chain_2d[idx]
        
        return item
    
    @property
    def n_features(self) -> int:
        return self.X.shape[-1]
    
    @property
    def seq_length(self) -> int:
        return self.X.shape[1] if self.X.ndim == 3 else 1
    
    @property
    def n_classes(self) -> int:
        return len(torch.unique(self.y))


class BalancedSampler(Sampler):
    def __init__(self, labels: torch.Tensor, replacement: bool = True):
        self.labels = labels
        self.replacement = replacement
        
        unique_labels = torch.unique(labels)
        self.n_classes = len(unique_labels)
        
        self.label_indices = {
            int(label): (labels == label).nonzero(as_tuple=True)[0].tolist()
            for label in unique_labels
        }
        
        self.min_class_size = min(len(indices) for indices in self.label_indices.values())
        self.n_samples = self.min_class_size * self.n_classes
    
    def __iter__(self) -> Iterator[int]:
        indices = []
        
        for label, label_indices in self.label_indices.items():
            if self.replacement:
                sampled = np.random.choice(label_indices, self.min_class_size, replace=True)
            else:
                sampled = np.random.choice(label_indices, self.min_class_size, replace=False)
            indices.extend(sampled.tolist())
        
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        return self.n_samples


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    chain_2d_train: Optional[np.ndarray] = None,
    chain_2d_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    balanced: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = Hybrid51Dataset(X_train, y_train, chain_2d_train, normalize=True)
    # Val uses TRAIN stats — no leakage
    val_dataset = Hybrid51Dataset(X_val, y_val, chain_2d_val, normalize=False)
    val_dataset.mean = train_dataset.mean
    val_dataset.std  = train_dataset.std
    val_dataset._apply_normalization()
    
    if balanced:
        train_sampler = BalancedSampler(train_dataset.y)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def save_processed_data(
    output_dir: str,
    X: np.ndarray,
    y: np.ndarray,
    chain_2d: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / 'X.npy', X)
    np.save(output_path / 'y.npy', y)
    
    if chain_2d is not None:
        np.save(output_path / 'chain_2d.npy', chain_2d)
    
    meta = metadata or {}
    meta.update({
        'n_samples': len(y),
        'n_features': X.shape[-1],
        'seq_length': X.shape[1] if X.ndim == 3 else 1,
        'has_chain_2d': chain_2d is not None,
        'label_distribution': {
            str(label): int(count)
            for label, count in zip(*np.unique(y, return_counts=True))
        }
    })
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    report = validate_preprocessed_data(X, chain_2d)
    
    with open(output_path / 'quality_report.json', 'w') as f:
        json.dump({
            'overall_quality': report.overall_quality,
            'missing_pct': report.missing_pct,
            'warnings': report.warnings[:20],
            'errors': report.errors
        }, f, indent=2)


def load_processed_data(
    data_dir: str
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    data_path = Path(data_dir)
    
    X = np.load(data_path / 'X.npy')
    y = np.load(data_path / 'y.npy')
    
    chain_2d_path = data_path / 'chain_2d.npy'
    chain_2d = np.load(chain_2d_path) if chain_2d_path.exists() else None
    
    meta_path = data_path / 'metadata.json'
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    return X, y, chain_2d, metadata


class PreprocessingPipeline:
    def __init__(
        self,
        symbol: str,
        output_dir: str,
        sequence_config: SequenceConfig = None
    ):
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.sequence_config = sequence_config or SequenceConfig()
        
        self.extractor = MasterFeatureExtractor(
            include_chain_2d=self.sequence_config.include_chain_2d
        )
        self.sequence_creator = SequenceCreator(
            config=self.sequence_config,
            extractor=self.extractor
        )
    
    def process(
        self,
        snapshots: List,
        labels: List[int],
        timestamps: List[str] = None,
        validate: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        sequences = self.sequence_creator.create_sequences_from_dfs(
            snapshots, labels, timestamps
        )
        
        X, chain_2d, y = self.sequence_creator.sequences_to_arrays(sequences)
        
        if validate:
            report = validate_preprocessed_data(X, chain_2d)
            if report.overall_quality < 0.8:
                raise ValueError(f"Data quality too low: {report.overall_quality:.2%}")
        
        return X, chain_2d, y
    
    def save(
        self,
        X: np.ndarray,
        y: np.ndarray,
        chain_2d: Optional[np.ndarray] = None,
        split: str = 'train'
    ):
        output_path = self.output_dir / self.symbol / split
        save_processed_data(
            str(output_path),
            X, y, chain_2d,
            metadata={
                'symbol': self.symbol,
                'split': split,
                'sequence_config': {
                    'sequence_length': self.sequence_config.sequence_length,
                    'prediction_horizon': self.sequence_config.prediction_horizon,
                    'stride': self.sequence_config.stride
                }
            }
        )
