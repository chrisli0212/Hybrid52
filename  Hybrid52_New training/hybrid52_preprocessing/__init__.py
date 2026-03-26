"""
Hybrid51 Preprocessing Package
Feature engineering pipeline for options data with 270 features.
"""

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

from .data_validation import (
    get_usable_greek_columns,
    get_excluded_columns,
    get_metadata_columns,
    get_feature_columns_by_group,
    validate_greek_columns,
)

from .master_extractor import (
    MasterFeatureExtractor,
    ExtractionResult,
    extract_all_features,
)

from .sequence_pipeline import (
    SequenceConfig,
    Sequence,
    SequenceCreator,
    StreamingSequenceCreator,
)

from .chain_2d import Chain2DProcessor

from .quality_checks import (
    DataQualityChecker,
    DataQualityReport,
    validate_preprocessed_data,
    print_quality_summary,
)

try:
    from .training_pipeline import (
        Hybrid51Dataset,
        BalancedSampler,
        create_data_loaders,
        save_processed_data,
        load_processed_data,
        PreprocessingPipeline,
    )
except (ImportError, OSError):
    pass

__all__ = [
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
    'get_usable_greek_columns',
    'get_excluded_columns',
    'get_metadata_columns',
    'get_feature_columns_by_group',
    'validate_greek_columns',
    'MasterFeatureExtractor',
    'ExtractionResult',
    'extract_all_features',
    'SequenceConfig',
    'Sequence',
    'SequenceCreator',
    'StreamingSequenceCreator',
    'Chain2DProcessor',
    'DataQualityChecker',
    'DataQualityReport',
    'validate_preprocessed_data',
    'print_quality_summary',
    'Hybrid51Dataset',
    'BalancedSampler',
    'create_data_loaders',
    'save_processed_data',
    'load_processed_data',
    'PreprocessingPipeline',
]

__version__ = '0.1.0'
