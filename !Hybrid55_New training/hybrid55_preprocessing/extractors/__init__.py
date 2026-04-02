"""
Shared raw feature extractors.
These are pure computation modules — no agent-specific logic, no feature counts,
no assertions. Each returns a numpy array whose length is determined by the extractor.
Agent assemblers select and combine these outputs.
"""

from .data_validation import (
    get_usable_greek_columns,
    get_excluded_columns,
    get_trade_quote_excluded_columns,
    filter_dead_columns,
)

from .gamma_exposure import (
    extract_gamma_features,
    extract_vanna_charm_features,
    extract_gamma_exposure_all,
)

from .iv_surface import extract_iv_surface_features

from .flow_volume import extract_flow_volume_features

from .greek_features import (
    extract_greek_features,
    extract_greek_features_batch,
)

from .cross_strike_time import (
    extract_cross_strike_features,
    extract_time_decay_features,
)

__all__ = [
    # Data validation
    "get_usable_greek_columns",
    "get_excluded_columns",
    "get_trade_quote_excluded_columns",
    "filter_dead_columns",
    # Gamma exposure
    "extract_gamma_features",
    "extract_vanna_charm_features",
    "extract_gamma_exposure_all",
    # IV surface
    "extract_iv_surface_features",
    # Flow volume
    "extract_flow_volume_features",
    # Greek features
    "extract_greek_features",
    "extract_greek_features_batch",
    # Cross strike/time
    "extract_cross_strike_features",
    "extract_time_decay_features",
]
