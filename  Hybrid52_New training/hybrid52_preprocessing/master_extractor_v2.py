"""
Hybrid52 Master Feature Extractor (historical mode).

Produces 286 flat features:
- 270 base extractors
- 16 CSV-derived enrichments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .feature_config_v2 import TOTAL_FEATURES, FEATURE_GROUPS, FeatureGroup, get_feature_names, HISTORICAL_MODE
from .data_validation import get_excluded_columns, get_trade_quote_excluded_columns
from .greek_features import GreekFeatureExtractor
from .csv_derived import CsvDerivedExtractor
from .gamma_exposure import GammaExposureExtractor
from .iv_surface import IVSurfaceExtractor
from .flow_volume import FlowVolumeExtractor
from .microstructure import MicrostructureExtractor
from .walls_positioning import WallsPositioningExtractor
from .cross_strike_time import CrossStrikeExtractor, TimeDecayExtractor
from .sentiment_regime import SentimentRegimeExtractor
from .smart_money import SmartMoneyDetector
from .volume_anomaly import VolumeAnomalyDetector
from .trade_conditions import TradeConditionAnalyzer
from .quote_pressure import QuotePressureAnalyzer
from .chain_2d import Chain2DProcessor


@dataclass
class ExtractionResult:
    features: np.ndarray
    chain_2d: Optional[np.ndarray] = None
    quality_score: float = 1.0
    missing_groups: List[str] = None
    phase1_enabled: bool = False
    
    def __post_init__(self):
        if self.missing_groups is None:
            self.missing_groups = []


class MasterFeatureExtractor:
    def __init__(
        self,
        include_chain_2d: bool = True,
        include_phase1: bool = True,
        normalize: bool = True
    ):
        """
        Initialize master feature extractor.
        
        Args:
            include_chain_2d: Include 2D chain tensor for Agent-2D CNN
            include_phase1: Include Phase 1 advanced trade/quote features (disabled in historical mode)
            normalize: Apply normalization to features
        """
        self.include_chain_2d = include_chain_2d
        self.include_phase1 = include_phase1 and not HISTORICAL_MODE
        self.csv_derived = CsvDerivedExtractor()
        self.normalize = normalize
        
        # Original extractors (270 features)
        self.greek_extractor = GreekFeatureExtractor()
        self.gamma_extractor = GammaExposureExtractor()
        self.iv_extractor = IVSurfaceExtractor()
        self.flow_extractor = FlowVolumeExtractor()
        self.micro_extractor = MicrostructureExtractor()
        self.walls_extractor = WallsPositioningExtractor()
        self.cross_extractor = CrossStrikeExtractor()
        self.time_extractor = TimeDecayExtractor()
        self.sentiment_extractor = SentimentRegimeExtractor()
        
        # Phase 1 extractors (55 features)
        if self.include_phase1:
            self.smart_money_detector = SmartMoneyDetector()
            self.volume_anomaly_detector = VolumeAnomalyDetector()
            self.trade_condition_analyzer = TradeConditionAnalyzer()
            self.quote_pressure_analyzer = QuotePressureAnalyzer()
        
        if include_chain_2d:
            self.chain_processor = Chain2DProcessor.from_config()
        else:
            self.chain_processor = None
        
        # Excluded columns
        self.excluded_greek_columns = set(get_excluded_columns())
        self.excluded_trade_columns = set(get_trade_quote_excluded_columns())
        
        self.feature_names = get_feature_names()
        self._normalization_stats = None
    
    def preprocess_greek_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove excluded Greek columns (speed, vera)."""
        return df.drop(columns=[c for c in self.excluded_greek_columns if c in df.columns], errors='ignore')
    
    def preprocess_trade_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove excluded trade/quote columns (ext_condition1-4, bid_condition, ask_condition)."""
        return df.drop(columns=[c for c in self.excluded_trade_columns if c in df.columns], errors='ignore')
    
    def extract_flat_features(
        self,
        greek_df: pd.DataFrame,
        trade_df: Optional[pd.DataFrame] = None,
        open_interest: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract all flat features.
        
        Args:
            greek_df: Greek data (required for all 270 base features)
            trade_df: Trade/quote data (required for Phase 1 features)
            open_interest: Current open interest (for volume anomaly detection)
        
        Returns:
            Feature array of shape (270 + optional phase1 features,)
        """
        n_features = 325 if self.include_phase1 else 270
        features = np.zeros(n_features, dtype=np.float32)
        
        greek_df = self.preprocess_greek_df(greek_df)
        
        idx = 0
        
        # Original 270 features
        greek_features = self.greek_extractor.extract(greek_df)
        features[idx:idx + 75] = greek_features
        idx += 75
        
        gamma_features = self.gamma_extractor.extract(greek_df)
        features[idx:idx + 50] = gamma_features
        idx += 50
        
        iv_features = self.iv_extractor.extract(greek_df)
        features[idx:idx + 25] = iv_features
        idx += 25
        
        flow_features = self.flow_extractor.extract(greek_df)
        features[idx:idx + 30] = flow_features
        idx += 30
        
        micro_features = self.micro_extractor.extract(greek_df)
        features[idx:idx + 20] = micro_features
        idx += 20
        
        walls_features = self.walls_extractor.extract(greek_df)
        features[idx:idx + 20] = walls_features
        idx += 20
        
        cross_features = self.cross_extractor.extract(greek_df)
        features[idx:idx + 15] = cross_features
        idx += 15
        
        time_features = self.time_extractor.extract(greek_df)
        features[idx:idx + 15] = time_features
        idx += 15
        
        sentiment_features = self.sentiment_extractor.extract(greek_df)
        features[idx:idx + 20] = sentiment_features
        idx += 20
        
        assert idx == 270, f"Expected 270 base features, got {idx}"
        
        # Phase 1 features (55 additional)
        if self.include_phase1 and trade_df is not None:
            trade_df = self.preprocess_trade_df(trade_df)
            
            smart_money_features = self.smart_money_detector.extract(trade_df)
            features[idx:idx + 15] = smart_money_features
            idx += 15
            
            volume_anomaly_features = self.volume_anomaly_detector.extract(
                trade_df, 
                open_interest=open_interest
            )
            features[idx:idx + 12] = volume_anomaly_features
            idx += 12
            
            trade_condition_features = self.trade_condition_analyzer.extract(trade_df)
            features[idx:idx + 10] = trade_condition_features
            idx += 10
            
            quote_pressure_features = self.quote_pressure_analyzer.extract(trade_df)
            features[idx:idx + 18] = quote_pressure_features
            idx += 18
            
            assert idx == 325, f"Expected 325 total features, got {idx}"
        
        return features
    
    def extract(
        self,
        greek_df: pd.DataFrame,
        trade_df: Optional[pd.DataFrame] = None,
        historical_snapshots: Optional[List[pd.DataFrame]] = None,
        open_interest: Optional[float] = None
    ) -> ExtractionResult:
        """
        Extract all features from Greek and trade/quote data.
        
        Args:
            greek_df: Greek data snapshot
            trade_df: Trade/quote data snapshot (required for Phase 1)
            historical_snapshots: Historical Greek snapshots for 2D chain
            open_interest: Current open interest
        
        Returns:
            ExtractionResult with features, chain_2d, and quality metrics
        """
        flat_features = self.extract_flat_features(greek_df, trade_df, open_interest)
        
        chain_2d = None
        if self.include_chain_2d and self.chain_processor is not None:
            if historical_snapshots is not None and len(historical_snapshots) > 0:
                snapshots = historical_snapshots + [greek_df]
                chain_2d = self.chain_processor.process_sequence(snapshots)
            else:
                chain_2d = self.chain_processor.snapshot_to_slice(greek_df)
                chain_2d = chain_2d[:, :, np.newaxis]
        
        n_features = len(flat_features)
        nan_count = np.isnan(flat_features).sum()
        quality_score = 1.0 - (nan_count / n_features)
        
        flat_features = np.nan_to_num(flat_features, nan=0.0, posinf=0.0, neginf=0.0)
        csv_feats = self.csv_derived.extract(greek_df)
        flat_features = np.concatenate([flat_features, csv_feats])

        return ExtractionResult(
            features=flat_features,
            chain_2d=chain_2d,
            quality_score=quality_score,
            phase1_enabled=self.include_phase1
        )
    
    def extract_batch(
        self,
        greek_dfs: List[pd.DataFrame],
        trade_dfs: Optional[List[pd.DataFrame]] = None,
        historical_snapshots_list: Optional[List[List[pd.DataFrame]]] = None,
        open_interests: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
        """
        Batch extraction for multiple snapshots.
        
        Args:
            greek_dfs: List of Greek data snapshots
            trade_dfs: List of trade/quote snapshots (same length as greek_dfs)
            historical_snapshots_list: Historical snapshots for each sample
            open_interests: Open interest values for each sample
        
        Returns:
            (features_array, chain_2d_array, quality_scores)
        """
        n_samples = len(greek_dfs)
        feature_rows: List[np.ndarray] = []
        quality_scores = []
        
        for i, greek_df in enumerate(greek_dfs):
            trade_df = trade_dfs[i] if trade_dfs else None
            hist = historical_snapshots_list[i] if historical_snapshots_list else None
            oi = open_interests[i] if open_interests else None
            
            result = self.extract(greek_df, trade_df, hist, oi)
            feature_rows.append(result.features.astype(np.float32, copy=False))
            quality_scores.append(result.quality_score)

        flat_features = np.vstack(feature_rows) if feature_rows else np.zeros((n_samples, TOTAL_FEATURES), dtype=np.float32)
        
        chain_2d_batch = None
        if self.include_chain_2d and self.chain_processor is not None:
            if historical_snapshots_list is not None:
                all_snapshots = [
                    (hist if hist else []) + [greek_df]
                    for hist, greek_df in zip(historical_snapshots_list, greek_dfs)
                ]
                chain_2d_batch = self.chain_processor.extract_batch(all_snapshots)
        
        return flat_features, chain_2d_batch, quality_scores
    
    def fit_normalization(self, features: np.ndarray):
        """Compute normalization statistics from training data."""
        self._normalization_stats = {
            'mean': features.mean(axis=0),
            'std': features.std(axis=0) + 1e-10
        }
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        if self._normalization_stats is None:
            return features
        
        return (features - self._normalization_stats['mean']) / self._normalization_stats['std']
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        return self.feature_names[:self.n_flat_features]
    
    def get_feature_group_indices(self, group: FeatureGroup) -> range:
        """Get index range for a feature group."""
        return FEATURE_GROUPS[group].indices
    
    def set_sentiment_context(
        self,
        historical_iv: Optional[pd.Series] = None,
        market_data: Optional[Dict] = None,
        vix_level: Optional[float] = None
    ):
        """Set historical context for sentiment/regime features."""
        self.sentiment_extractor.set_context(historical_iv, market_data, vix_level)
    
    def set_volume_anomaly_context(
        self,
        symbol: str,
        strike: float,
        historical_volume_mean: float,
        historical_volume_std: float
    ):
        """Set historical volume statistics for anomaly detection."""
        if self.include_phase1:
            self.volume_anomaly_detector.set_historical_context(
                symbol, strike, historical_volume_mean, historical_volume_std
            )
    
    @property
    def n_flat_features(self) -> int:
        """Total number of flat features."""
        return TOTAL_FEATURES
    
    @property
    def chain_2d_shape(self) -> Optional[Tuple[int, int, int]]:
        """Shape of 2D chain tensor."""
        if self.chain_processor:
            return self.chain_processor.output_shape
        return None
    
    @property
    def n_base_features(self) -> int:
        """Number of original base features."""
        return 270
    
    @property
    def n_phase1_features(self) -> int:
        """Number of Phase 1 features."""
        return 55 if self.include_phase1 else 0


def extract_all_features(
    greek_df: pd.DataFrame,
    trade_df: Optional[pd.DataFrame] = None,
    include_chain_2d: bool = True,
    include_phase1: bool = True
) -> ExtractionResult:
    """
    Convenience function to extract all features.
    
    Args:
        greek_df: Greek data
        trade_df: Trade/quote data (required for Phase 1)
        include_chain_2d: Include 2D chain
        include_phase1: Include Phase 1 advanced features
    
    Returns:
        ExtractionResult with all features
    """
    extractor = MasterFeatureExtractor(
        include_chain_2d=include_chain_2d,
        include_phase1=include_phase1
    )
    return extractor.extract(greek_df, trade_df)
