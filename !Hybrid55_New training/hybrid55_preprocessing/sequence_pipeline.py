"""
Hybrid51 Sequence Creation Pipeline (Task 12)
Creates rolling window sequences for temporal learning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass

from .feature_config_v2 import TOTAL_FEATURES, CHAIN_2D_CONFIG
from .master_extractor import MasterFeatureExtractor


@dataclass
class SequenceConfig:
    sequence_length: int = 20
    prediction_horizon: int = 1
    stride: int = 1
    min_quality: float = 0.8
    include_chain_2d: bool = True


@dataclass
class Sequence:
    features: np.ndarray
    chain_2d: Optional[np.ndarray]
    label: int
    timestamp: str
    quality_score: float


class SequenceCreator:
    def __init__(
        self,
        config: SequenceConfig = None,
        extractor: MasterFeatureExtractor = None
    ):
        self.config = config or SequenceConfig()
        self.extractor = extractor or MasterFeatureExtractor(
            include_chain_2d=self.config.include_chain_2d
        )
    
    def create_sequences_from_dfs(
        self,
        snapshots: List[pd.DataFrame],
        labels: List[int],
        timestamps: List[str] = None
    ) -> List[Sequence]:
        n_snapshots = len(snapshots)
        seq_len = self.config.sequence_length
        horizon = self.config.prediction_horizon
        stride = self.config.stride
        
        if n_snapshots < seq_len + horizon:
            return []
        
        sequences = []
        
        for start_idx in range(0, n_snapshots - seq_len - horizon + 1, stride):
            end_idx = start_idx + seq_len
            label_idx = end_idx + horizon - 1
            
            seq_snapshots = snapshots[start_idx:end_idx]
            
            features_list = []
            quality_scores = []
            
            for snap in seq_snapshots:
                result = self.extractor.extract(snap)
                features_list.append(result.features)
                quality_scores.append(result.quality_score)
            
            avg_quality = np.mean(quality_scores)
            if avg_quality < self.config.min_quality:
                continue
            
            features = np.stack(features_list, axis=0)
            
            chain_2d = None
            if self.config.include_chain_2d and self.extractor.chain_processor:
                chain_2d = self.extractor.chain_processor.process_sequence(seq_snapshots)
            
            label = labels[label_idx]
            timestamp = timestamps[label_idx] if timestamps else str(label_idx)
            
            sequences.append(Sequence(
                features=features,
                chain_2d=chain_2d,
                label=label,
                timestamp=timestamp,
                quality_score=avg_quality
            ))
        
        return sequences
    
    def sequences_to_arrays(
        self,
        sequences: List[Sequence]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        n_sequences = len(sequences)
        seq_len = self.config.sequence_length
        
        X = np.zeros((n_sequences, seq_len, TOTAL_FEATURES), dtype=np.float32)
        y = np.zeros(n_sequences, dtype=np.int64)
        
        chain_2d = None
        if self.config.include_chain_2d and sequences[0].chain_2d is not None:
            n_greeks = CHAIN_2D_CONFIG['n_greeks']
            n_strikes = CHAIN_2D_CONFIG['n_strikes']
            n_timesteps = CHAIN_2D_CONFIG['n_timesteps']
            chain_2d = np.zeros((n_sequences, n_greeks, n_strikes, n_timesteps), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            X[i] = seq.features
            y[i] = seq.label
            if chain_2d is not None and seq.chain_2d is not None:
                chain_2d[i] = seq.chain_2d
        
        return X, chain_2d, y


class StreamingSequenceCreator:
    def __init__(
        self,
        config: SequenceConfig = None,
        extractor: MasterFeatureExtractor = None
    ):
        self.config = config or SequenceConfig()
        self.extractor = extractor or MasterFeatureExtractor(
            include_chain_2d=self.config.include_chain_2d
        )
        
        self.buffer: List[pd.DataFrame] = []
        self.feature_buffer: List[np.ndarray] = []
        self.label_buffer: List[int] = []
        self.timestamp_buffer: List[str] = []
        self.quality_buffer: List[float] = []
    
    def add_snapshot(
        self,
        df: pd.DataFrame,
        label: int,
        timestamp: str = None
    ):
        result = self.extractor.extract(df)
        
        self.buffer.append(df)
        self.feature_buffer.append(result.features)
        self.label_buffer.append(label)
        self.timestamp_buffer.append(timestamp or str(len(self.buffer)))
        self.quality_buffer.append(result.quality_score)
        
        max_buffer = self.config.sequence_length + self.config.prediction_horizon + 10
        if len(self.buffer) > max_buffer:
            self.buffer.pop(0)
            self.feature_buffer.pop(0)
            self.label_buffer.pop(0)
            self.timestamp_buffer.pop(0)
            self.quality_buffer.pop(0)
    
    def can_create_sequence(self) -> bool:
        return len(self.buffer) >= self.config.sequence_length + self.config.prediction_horizon
    
    def create_latest_sequence(self) -> Optional[Sequence]:
        if not self.can_create_sequence():
            return None
        
        seq_len = self.config.sequence_length
        horizon = self.config.prediction_horizon
        
        start_idx = len(self.buffer) - seq_len - horizon
        end_idx = start_idx + seq_len
        label_idx = end_idx + horizon - 1
        
        features = np.stack(self.feature_buffer[start_idx:end_idx], axis=0)
        quality_scores = self.quality_buffer[start_idx:end_idx]
        avg_quality = np.mean(quality_scores)
        
        if avg_quality < self.config.min_quality:
            return None
        
        chain_2d = None
        if self.config.include_chain_2d and self.extractor.chain_processor:
            chain_2d = self.extractor.chain_processor.process_sequence(
                self.buffer[start_idx:end_idx]
            )
        
        return Sequence(
            features=features,
            chain_2d=chain_2d,
            label=self.label_buffer[label_idx],
            timestamp=self.timestamp_buffer[label_idx],
            quality_score=avg_quality
        )
    
    def reset(self):
        self.buffer.clear()
        self.feature_buffer.clear()
        self.label_buffer.clear()
        self.timestamp_buffer.clear()
        self.quality_buffer.clear()


def create_sequences_from_csv(
    csv_path: str,
    label_col: str,
    timestamp_col: str = 'timestamp',
    config: SequenceConfig = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    df = pd.read_csv(csv_path)
    
    timestamps = df.groupby(timestamp_col).first().index.tolist()
    
    snapshots = [
        df[df[timestamp_col] == ts].copy()
        for ts in timestamps
    ]
    
    labels = [
        df[df[timestamp_col] == ts][label_col].iloc[0]
        for ts in timestamps
    ]
    
    creator = SequenceCreator(config=config)
    sequences = creator.create_sequences_from_dfs(snapshots, labels, timestamps)
    
    return creator.sequences_to_arrays(sequences)
