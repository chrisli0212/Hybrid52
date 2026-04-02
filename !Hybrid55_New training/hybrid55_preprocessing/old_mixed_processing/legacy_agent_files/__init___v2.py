"""
Hybrid51 Preprocessing Package - UPDATED WITH PHASE 1
Feature engineering pipeline for options data with 327 features (270 + 57 Phase 1).

Phase 1 Additions:
- Smart Money Detection (15 features)
- Volume Anomaly Detection (12 features)
- Trade Condition Analysis (10 features)
- Quote Pressure & Exchange Routing (20 features)
"""

# Feature Configuration
from .feature_config_v2 import (
    TOTAL_FEATURES,
    FEATURE_GROUPS,
    FeatureGroup,
    DELTA_BUCKETS,
    GREEKS_FOR_BUCKETING,
    ATM_GREEKS,
    MONEYNESS_LEVELS,
    DTE_BUCKETS,
    CHAIN_2D_CONFIG,
    get_feature_names,
)

# Data Validation
from .data_validation import (
    get_usable_greek_columns,
    get_excluded_columns,
    get_trade_quote_excluded_columns,
    get_metadata_columns,
    get_feature_columns_by_group,
    validate_greek_columns,
)

# Base Feature Extractors (270 features)
from .greek_features import GreekFeatureExtractor
from .gamma_exposure import GammaExposureExtractor
from .iv_surface import IVSurfaceExtractor
from .flow_volume import FlowVolumeExtractor
from .microstructure import MicrostructureExtractor
from .walls_positioning import WallsPositioningExtractor
from .cross_strike_time import CrossStrikeExtractor, TimeDecayExtractor
from .sentiment_regime import SentimentRegimeExtractor

# Phase 1 Feature Extractors (57 features)
from .smart_money import SmartMoneyDetector
from .volume_anomaly import VolumeAnomalyDetector
from .trade_conditions import TradeConditionAnalyzer
from .quote_pressure import QuotePressureAnalyzer

# Master Extractor
from .master_extractor_v2 import (
    MasterFeatureExtractor,
    ExtractionResult,
    extract_all_features,
)

# Sequence Pipeline
from .sequence_pipeline import (
    SequenceConfig,
    Sequence,
    SequenceCreator,
    StreamingSequenceCreator,
)

# 2D Chain Processor
from .chain_2d import Chain2DProcessor

# Quality Checks
from .quality_checks import (
    DataQualityChecker,
    DataQualityReport,
    validate_preprocessed_data,
    print_quality_summary,
)

# Training Pipeline
from .training_pipeline import (
    Hybrid51Dataset,
    BalancedSampler,
    create_data_loaders,
    save_processed_data,
    load_processed_data,
    PreprocessingPipeline,
)

__all__ = [
    # Configuration
    'TOTAL_FEATURES',
    'FEATURE_GROUPS',
    'FeatureGroup',
    'DELTA_BUCKETS',
    'GREEKS_FOR_BUCKETING',
    'ATM_GREEKS',
    'MONEYNESS_LEVELS',
    'DTE_BUCKETS',
    'CHAIN_2D_CONFIG',
    'get_feature_names',
    
    # Data Validation
    'get_usable_greek_columns',
    'get_excluded_columns',
    'get_trade_quote_excluded_columns',
    'get_metadata_columns',
    'get_feature_columns_by_group',
    'validate_greek_columns',
    
    # Base Extractors
    'GreekFeatureExtractor',
    'GammaExposureExtractor',
    'IVSurfaceExtractor',
    'FlowVolumeExtractor',
    'MicrostructureExtractor',
    'WallsPositioningExtractor',
    'CrossStrikeExtractor',
    'TimeDecayExtractor',
    'SentimentRegimeExtractor',
    
    # Phase 1 Extractors
    'SmartMoneyDetector',
    'VolumeAnomalyDetector',
    'TradeConditionAnalyzer',
    'QuotePressureAnalyzer',
    
    # Master Extractor
    'MasterFeatureExtractor',
    'ExtractionResult',
    'extract_all_features',
    
    # Sequence Pipeline
    'SequenceConfig',
    'Sequence',
    'SequenceCreator',
    'StreamingSequenceCreator',
    
    # 2D Chain
    'Chain2DProcessor',
    
    # Quality
    'DataQualityChecker',
    'DataQualityReport',
    'validate_preprocessed_data',
    'print_quality_summary',
    
    # Training
    'Hybrid51Dataset',
    'BalancedSampler',
    'create_data_loaders',
    'save_processed_data',
    'load_processed_data',
    'PreprocessingPipeline',
]

__version__ = '0.2.0'  # Updated for Phase 1 integration
