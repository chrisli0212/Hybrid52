"""
Hybrid51 Master Feature Extractor (Task 11)
Orchestrates all feature extraction modules to produce 270 features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .feature_config_v2 import TOTAL_FEATURES, FEATURE_GROUPS, FeatureGroup, get_feature_names
from .data_validation import get_excluded_columns, get_usable_greek_columns
from .greek_features import GreekFeatureExtractor
from .gamma_exposure import GammaExposureExtractor
from .iv_surface import IVSurfaceExtractor
from .flow_volume import FlowVolumeExtractor
from .microstructure import MicrostructureExtractor
from .walls_positioning import WallsPositioningExtractor
from .cross_strike_time import CrossStrikeExtractor, TimeDecayExtractor
from .sentiment_regime import SentimentRegimeExtractor
from .chain_2d import Chain2DProcessor


@dataclass
class ExtractionResult:
    features: np.ndarray
    chain_2d: Optional[np.ndarray] = None
    quality_score: float = 1.0
    missing_groups: List[str] = None
    
    def __post_init__(self):
        if self.missing_groups is None:
            self.missing_groups = []


class MasterFeatureExtractor:
    def __init__(
        self,
        include_chain_2d: bool = True,
        normalize: bool = True
    ):
        self.include_chain_2d = include_chain_2d
        self.normalize = normalize
        
        self.greek_extractor = GreekFeatureExtractor()
        self.gamma_extractor = GammaExposureExtractor()
        self.iv_extractor = IVSurfaceExtractor()
        self.flow_extractor = FlowVolumeExtractor()
        self.micro_extractor = MicrostructureExtractor()
        self.walls_extractor = WallsPositioningExtractor()
        self.cross_extractor = CrossStrikeExtractor()
        self.time_extractor = TimeDecayExtractor()
        self.sentiment_extractor = SentimentRegimeExtractor()
        
        if include_chain_2d:
            self.chain_processor = Chain2DProcessor.from_config()
        else:
            self.chain_processor = None
        
        self.excluded_columns = set(get_excluded_columns())
        self.feature_names = get_feature_names()
        
        self._normalization_stats = None
    
    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=[c for c in self.excluded_columns if c in df.columns], errors='ignore')
        return df
    
    def extract_flat_features(self, df: pd.DataFrame) -> np.ndarray:
        features = np.zeros(TOTAL_FEATURES, dtype=np.float32)
        
        df = self.preprocess_df(df)
        
        idx = 0
        
        greek_features = self.greek_extractor.extract(df)
        features[idx:idx + len(greek_features)] = greek_features
        idx += 75
        
        gamma_features = self.gamma_extractor.extract(df)
        features[idx:idx + len(gamma_features)] = gamma_features
        idx += 50
        
        iv_features = self.iv_extractor.extract(df)
        features[idx:idx + len(iv_features)] = iv_features
        idx += 25
        
        flow_features = self.flow_extractor.extract(df)
        features[idx:idx + len(flow_features)] = flow_features
        idx += 30
        
        micro_features = self.micro_extractor.extract(df)
        features[idx:idx + len(micro_features)] = micro_features
        idx += 20
        
        walls_features = self.walls_extractor.extract(df)
        features[idx:idx + len(walls_features)] = walls_features
        idx += 20
        
        cross_features = self.cross_extractor.extract(df)
        features[idx:idx + len(cross_features)] = cross_features
        idx += 15
        
        time_features = self.time_extractor.extract(df)
        features[idx:idx + len(time_features)] = time_features
        idx += 15
        
        sentiment_features = self.sentiment_extractor.extract(df)
        features[idx:idx + len(sentiment_features)] = sentiment_features
        idx += 20
        
        assert idx == TOTAL_FEATURES, f"Expected {TOTAL_FEATURES}, got {idx}"
        
        return features
    
    def extract(
        self,
        df: pd.DataFrame,
        historical_snapshots: Optional[List[pd.DataFrame]] = None
    ) -> ExtractionResult:
        flat_features = self.extract_flat_features(df)
        
        chain_2d = None
        if self.include_chain_2d and self.chain_processor is not None:
            if historical_snapshots is not None and len(historical_snapshots) > 0:
                snapshots = historical_snapshots + [df]
                chain_2d = self.chain_processor.process_sequence(snapshots)
            else:
                chain_2d = self.chain_processor.snapshot_to_slice(df)
                chain_2d = chain_2d[:, :, np.newaxis]
        
        nan_count = np.isnan(flat_features).sum()
        quality_score = 1.0 - (nan_count / TOTAL_FEATURES)
        
        flat_features = np.nan_to_num(flat_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return ExtractionResult(
            features=flat_features,
            chain_2d=chain_2d,
            quality_score=quality_score
        )
    
    def extract_batch(
        self,
        dfs: List[pd.DataFrame],
        historical_snapshots_list: Optional[List[List[pd.DataFrame]]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
        n_samples = len(dfs)
        
        flat_features = np.zeros((n_samples, TOTAL_FEATURES), dtype=np.float32)
        quality_scores = []
        
        for i, df in enumerate(dfs):
            hist = historical_snapshots_list[i] if historical_snapshots_list else None
            result = self.extract(df, hist)
            flat_features[i] = result.features
            quality_scores.append(result.quality_score)
        
        chain_2d_batch = None
        if self.include_chain_2d and self.chain_processor is not None:
            if historical_snapshots_list is not None:
                all_snapshots = [
                    (hist if hist else []) + [df]
                    for hist, df in zip(historical_snapshots_list, dfs)
                ]
                chain_2d_batch = self.chain_processor.extract_batch(all_snapshots)
        
        return flat_features, chain_2d_batch, quality_scores
    
    def fit_normalization(self, features: np.ndarray):
        self._normalization_stats = {
            'mean': features.mean(axis=0),
            'std': features.std(axis=0) + 1e-10
        }
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self._normalization_stats is None:
            return features
        
        return (features - self._normalization_stats['mean']) / self._normalization_stats['std']
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names
    
    def get_feature_group_indices(self, group: FeatureGroup) -> range:
        return FEATURE_GROUPS[group].indices
    
    def set_sentiment_context(
        self,
        historical_iv: Optional[pd.Series] = None,
        market_data: Optional[Dict] = None,
        vix_level: Optional[float] = None
    ):
        self.sentiment_extractor.set_context(historical_iv, market_data, vix_level)
    
    @property
    def n_flat_features(self) -> int:
        return TOTAL_FEATURES
    
    @property
    def chain_2d_shape(self) -> Optional[Tuple[int, int, int]]:
        if self.chain_processor:
            return self.chain_processor.output_shape
        return None


def extract_all_features(
    df: pd.DataFrame,
    include_chain_2d: bool = True
) -> ExtractionResult:
    extractor = MasterFeatureExtractor(include_chain_2d=include_chain_2d)
    return extractor.extract(df)
