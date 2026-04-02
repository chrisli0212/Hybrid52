"""
Hybrid55 historical feature configuration.

Defines 311 flat features:
- 286 base flat features
- 25 OHLC dynamics features

Dead/constant raw tier1 columns are filtered upstream before extraction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import IntEnum


class FeatureGroup(IntEnum):
    GREEK_BY_STRIKE = 0
    GAMMA_EXPOSURE = 1
    VANNA_CHARM = 2
    IV_SURFACE = 3
    FLOW_VOLUME = 4
    MICROSTRUCTURE = 5
    WALLS_POSITIONING = 6
    CROSS_STRIKE = 7
    TIME_DECAY = 8
    SENTIMENT_REGIME = 9
    CSV_DERIVED = 10
    OHLC_DYNAMICS = 11


@dataclass
class FeatureGroupConfig:
    name: str
    start_idx: int
    num_features: int
    description: str
    subgroups: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    @property
    def end_idx(self) -> int:
        return self.start_idx + self.num_features
    
    @property
    def indices(self) -> range:
        return range(self.start_idx, self.end_idx)


FEATURE_GROUPS = {
    FeatureGroup.GREEK_BY_STRIKE: FeatureGroupConfig(
        name="greek_by_strike",
        start_idx=0,
        num_features=75,
        description="Greeks aggregated by delta buckets + ATM + skew",
        subgroups={
            "bucket_greeks": (0, 65),
            "atm_greeks": (65, 72),
            "skew_metrics": (72, 75),
        }
    ),
    FeatureGroup.GAMMA_EXPOSURE: FeatureGroupConfig(
        name="gamma_exposure",
        start_idx=75,
        num_features=30,
        description="SpotGamma-inspired gamma exposure analysis",
        subgroups={
            "gamma_by_strike": (75, 95),
            "net_gamma": (95, 99),
            "dealer_positioning": (99, 102),
            "gamma_zones": (102, 105),
        }
    ),
    FeatureGroup.VANNA_CHARM: FeatureGroupConfig(
        name="vanna_charm",
        start_idx=105,
        num_features=20,
        description="Higher-order Greek exposures",
        subgroups={
            "vanna_by_bucket": (105, 110),
            "charm_by_bucket": (110, 115),
            "net_exposures": (115, 121),
            "cross_greek_ratios": (121, 125),
        }
    ),
    FeatureGroup.IV_SURFACE: FeatureGroupConfig(
        name="iv_surface",
        start_idx=125,
        num_features=25,
        description="Implied volatility surface features",
        subgroups={
            "iv_by_moneyness": (125, 132),
            "iv_term_structure": (132, 137),
            "vol_skew_metrics": (137, 142),
            "iv_percentiles": (142, 145),
            "put_call_iv_diff": (145, 150),
        }
    ),
    FeatureGroup.FLOW_VOLUME: FeatureGroupConfig(
        name="flow_volume",
        start_idx=150,
        num_features=30,
        description="Unusual Whales-inspired flow tracking",
        subgroups={
            "call_put_ratios": (150, 154),
            "volume_by_aggression": (154, 160),
            "size_distribution": (160, 164),
            "premium_metrics": (164, 167),
            "flow_direction": (167, 170),
            "time_weighted_flow": (170, 174),
            "dark_pool_vs_lit": (174, 176),
            "trade_metrics": (176, 180),
        }
    ),
    FeatureGroup.MICROSTRUCTURE: FeatureGroupConfig(
        name="microstructure",
        start_idx=180,
        num_features=20,
        description="Market microstructure features",
        subgroups={
            "bid_ask_spread": (180, 184),
            "order_book_imbalance": (184, 189),
            "quote_intensity": (189, 192),
            "trade_velocity": (192, 194),
            "effective_spreads": (194, 196),
            "price_impact": (196, 200),
        }
    ),
    FeatureGroup.WALLS_POSITIONING: FeatureGroupConfig(
        name="walls_positioning",
        start_idx=200,
        num_features=20,
        description="Put/Call walls and dealer positioning",
        subgroups={
            "max_gamma_strikes": (200, 202),
            "max_oi_strikes": (202, 206),
            "wall_distances": (206, 214),
            "dealer_positioning": (214, 220),
        }
    ),
    FeatureGroup.CROSS_STRIKE: FeatureGroupConfig(
        name="cross_strike",
        start_idx=220,
        num_features=15,
        description="Cross-strike distribution metrics",
        subgroups={
            "oi_volume_dist": (220, 226),
            "greek_concentrations": (226, 229),
            "strike_clustering": (229, 232),
            "liquidity_gradient": (232, 235),
        }
    ),
    FeatureGroup.TIME_DECAY: FeatureGroupConfig(
        name="time_decay",
        start_idx=235,
        num_features=15,
        description="Time decay and expiration features",
        subgroups={
            "dte_buckets": (235, 240),
            "decay_accelerations": (240, 245),
            "time_concentrations": (245, 248),
            "calendar_proximity": (248, 250),
        }
    ),
    FeatureGroup.SENTIMENT_REGIME: FeatureGroupConfig(
        name="sentiment_regime",
        start_idx=250,
        num_features=20,
        description="Market sentiment and regime indicators",
        subgroups={
            "sentiment_scores": (250, 253),
            "volatility_regime": (253, 257),
            "trend_stress": (257, 263),
            "correlation_metrics": (263, 267),
            "vix_relative": (267, 270),
        }
    ),
    
    FeatureGroup.CSV_DERIVED: FeatureGroupConfig(
        name="csv_derived",
        start_idx=270,
        num_features=16,
        description="CSV-native enrichments and first-wave derived features",
        subgroups={
            "lambda_features": (270, 273),
            "dist_atm_features": (273, 275),
            "spread_pct_features": (275, 278),
            "chain_aux_features": (278, 284),
            "oi_features": (284, 286),
        }
    ),
    FeatureGroup.OHLC_DYNAMICS: FeatureGroupConfig(
        name="ohlc_dynamics",
        start_idx=286,
        num_features=25,
        description="1-minute OHLC chain dynamics (coverage-aware)",
        subgroups={
            "coverage_and_geometry": (286, 292),
            "volume_profiles": (292, 302),
            "cross_strike": (302, 307),
            "patterns": (307, 311),
        },
    ),
}


TOTAL_FEATURES = 311
FEATURE_SCHEMA_VERSION = "hybrid55_v1_live_raw_guarded_311"
HISTORICAL_MODE = True

# Locked dead raw tier1 fields discovered by audit.
# These are intentionally excluded upstream and should not drive model training.
DEAD_RAW_TIER1_FIELDS = {
    "speed",
    "vera",
    "zomma",
    "dual_gamma",
    "iv_error",
    "endpoint",
    "batch_id",
    "ts",
}

DELTA_BUCKETS = [
    ("deep_otm", 0.0, 0.2),
    ("otm", 0.2, 0.4),
    ("atm", 0.4, 0.6),
    ("itm", 0.6, 0.8),
    ("deep_itm", 0.8, 1.0),
]

GREEKS_FOR_BUCKETING = [
    'delta', 'gamma', 'vega', 'theta', 'lambda',
    'vanna', 'charm',
    'implied_vol', 'open_interest', 'moneyness',
    'bid', 'ask', 'mid',
]

ATM_GREEKS = ['delta', 'gamma', 'vega', 'theta', 'lambda', 'vanna', 'charm']

MONEYNESS_LEVELS = [-0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30]

DTE_BUCKETS = [
    ("0_7d", 0, 7),
    ("8_30d", 8, 30),
    ("31_60d", 31, 60),
    ("61_90d", 61, 90),
    ("90plus", 91, 365),
]

CHAIN_2D_CONFIG = {
    "n_greeks": 5,
    "n_strikes": 30,
    "n_timesteps": 20,
    "greeks": ['delta', 'gamma', 'vega', 'theta', 'implied_vol'],
    "delta_range": (-0.9, 0.9),
}


def get_feature_names() -> List[str]:
    """Get all 286 historical-mode feature names."""
    names = []
    
    # Original 270 features (unchanged)
    for bucket_name, _, _ in DELTA_BUCKETS:
        for greek in GREEKS_FOR_BUCKETING:
            names.append(f"{bucket_name}_{greek}")
    
    for greek in ATM_GREEKS:
        names.append(f"atm_{greek}")
    
    names.extend(["call_put_iv_diff", "vol_term_slope", "put_skew_intensity"])
    
    for i in range(10):
        names.append(f"gamma_strike_above_{i}")
    for i in range(10):
        names.append(f"gamma_strike_below_{i}")
    
    names.extend(["total_gamma", "call_gamma", "put_gamma", "net_gamma"])
    names.extend(["dealer_gamma_estimate", "gamma_flip_level", "dist_to_gamma_flip"])
    names.extend(["below_gamma_flip", "above_gamma_flip", "gamma_zone_strength"])
    
    for bucket_name, _, _ in DELTA_BUCKETS:
        names.append(f"{bucket_name}_vanna")
    for bucket_name, _, _ in DELTA_BUCKETS:
        names.append(f"{bucket_name}_charm")
    
    names.extend(["total_vanna", "call_vanna", "put_vanna",
                  "total_charm", "call_charm", "put_charm"])
    names.extend(["vanna_gamma_ratio", "charm_theta_ratio", 
                  "vanna_vega_ratio", "net_vanna_strength"])
    
    for level in MONEYNESS_LEVELS:
        names.append(f"iv_moneyness_{level:.0%}".replace("-", "neg").replace("%", "pct"))
    
    names.extend(["iv_1w", "iv_1m", "iv_2m", "iv_3m", "iv_6m"])
    names.extend(["put_skew", "call_skew", "term_skew", "smile_curvature", "skew_asymmetry"])
    names.extend(["iv_p25", "iv_p50", "iv_p75"])
    names.extend(["pc_iv_diff_1w", "pc_iv_diff_1m", "pc_iv_diff_2m", 
                  "pc_iv_diff_3m", "pc_iv_diff_6m"])
    
    names.extend(["call_put_vol_ratio", "call_put_oi_ratio", 
                  "call_put_premium_ratio", "net_cp_bias"])
    names.extend(["passive_volume", "aggressive_volume", "sweep_volume",
                  "aggression_ratio", "call_aggression", "put_aggression"])
    names.extend(["small_trade_vol", "medium_trade_vol", "large_trade_vol", "block_trade_vol"])
    names.extend(["total_premium", "avg_premium", "vwap_premium"])
    names.extend(["buy_volume", "sell_volume", "buy_sell_imbalance"])
    names.extend(["flow_1m", "flow_5m", "flow_15m", "flow_30m"])
    names.extend(["dark_pool_pct", "lit_pct"])
    names.extend(["trade_count", "avg_trade_size", "trade_velocity", "vol_concentration"])
    
    names.extend(["spread_mean", "spread_pct", "spread_rolling", "spread_std"])
    names.extend(["bid_ask_imbalance", "tob_imbalance", "depth_imbalance",
                  "imbalance_vol", "sustained_imbalance"])
    names.extend(["quote_frequency", "cancel_rate", "improvement_rate"])
    names.extend(["trades_per_min", "volume_per_min"])
    names.extend(["effective_spread", "realized_spread"])
    names.extend(["temp_impact", "short_impact", "size_impact_corr", "impact_asymmetry"])
    
    names.extend(["call_max_gamma_strike", "put_max_gamma_strike"])
    names.extend(["call_max_oi_strike", "put_max_oi_strike",
                  "call_oi_at_max", "put_oi_at_max"])
    names.extend(["dist_to_call_wall", "dist_to_put_wall",
                  "call_wall_strength", "put_wall_strength",
                  "combined_wall_strength", "pinning_prob",
                  "breakout_prob", "wall_asymmetry"])
    names.extend(["dealer_net_delta", "dealer_net_gamma",
                  "dealer_long_short_ratio", "dealer_hedging_pressure",
                  "dealer_gamma_demand", "dealer_vega_exposure"])
    
    names.extend(["oi_concentration", "vol_concentration_gini",
                  "oi_skewness", "vol_skewness",
                  "oi_vol_corr", "oi_vol_divergence"])
    names.extend(["gamma_concentration", "vanna_concentration", "charm_concentration"])
    names.extend(["active_strikes", "strike_uniformity", "atm_clustering"])
    names.extend(["liquidity_decay", "otm_put_liquidity", "otm_call_liquidity"])
    
    names.extend(["oi_0_7d", "oi_8_30d", "oi_31_60d", "oi_61_90d", "oi_90plus"])
    names.extend(["theta_accel", "gamma_accel", "charm_accel",
                  "weighted_theta", "weighted_charm"])
    names.extend(["near_term_oi_conc", "near_term_vol_conc", "near_term_gamma_conc"])
    names.extend(["days_to_major_exp", "days_to_opex"])
    
    names.extend(["cp_sentiment", "premium_sentiment", "flow_sentiment"])
    names.extend(["iv_percentile", "iv_expansion", "iv_contraction", "vol_regime"])
    names.extend(["momentum_1d", "momentum_5d", "momentum_20d",
                  "trend_strength", "stress_indicator", "fear_indicator"])
    names.extend(["spx_corr", "vix_corr", "sector_corr", "beta_to_spx"])
    names.extend(["iv_vix_ratio", "relative_iv", "vix_term_impact"])
    
    # CSV-derived features (dims 270-285) — 16 features
    names.extend([
        'lambda_mean', 'lambda_atm', 'lambda_skew',
        'dist_atm_mean', 'dist_atm_weighted',
        'spread_pct_mean', 'spread_pct_atm', 'spread_pct_skew',
        'dte_mean', 'dte_std',
        'cp_sign_mean', 'mid_mean',
        'spread_atm', 'iv_std',
        'oi_mean', 'oi_put_call_ratio',
    ])

    # OHLC dynamics features (dims 286-310) — 25 features
    names.extend([
        "ohlc_active_ratio",
        "ohlc_range_pct",
        "ohlc_body_pct",
        "ohlc_upper_shadow",
        "ohlc_lower_shadow",
        "ohlc_close_position",
        "ohlc_volume_weighted_return",
        "ohlc_cp_vol_ratio",
        "ohlc_volume_gini",
        "ohlc_high_vol_strike_dist",
        "ohlc_volume_skew",
        "ohlc_trade_fragmentation",
        "ohlc_atm_volume_share",
        "ohlc_otm_put_volume_share",
        "ohlc_total_volume_log",
        "ohlc_volume_momentum",
        "ohlc_avg_range_dispersion",
        "ohlc_call_put_range_ratio",
        "ohlc_vwap_moneyness",
        "ohlc_high_low_corr",
        "ohlc_close_open_skew",
        "ohlc_doji_pct",
        "ohlc_hammer_pct",
        "ohlc_shooting_star_pct",
        "ohlc_gap_pct",
    ])
    
    assert len(names) == TOTAL_FEATURES, f"Expected {TOTAL_FEATURES}, got {len(names)}"
    
    return names
