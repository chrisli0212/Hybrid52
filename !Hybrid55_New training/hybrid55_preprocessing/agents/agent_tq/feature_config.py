"""
Agent TQ Feature Configuration
Agent TQ: Unified Trade+Quote Specialist (Conv1D + static MLP + imbalance head)

Input:
    static: (B, 95) — flat TQ snapshot
    seq:    (B, T, 95) — rolling window of TQ snapshots for Conv1D path

Source blocks in master 311-dim vector:
    microstructure    [180:200]  20 dims
    flow_volume       [150:180]  30 dims  (already covers trade metrics)
    sentiment_regime  [250:270]  20 dims
    phase1_smart_money          [295:310] 15 dims (live mode only)
    phase1_quote_pressure       [332:350] 18 dims (live mode only)

Historical mode (HISTORICAL_MODE=True): 50 dims (micro + flow subset)
Live mode: 95 dims (adds phase1 blocks)
"""

from typing import List, Tuple

# ── Dimension constants ───────────────────────────────────────────────────────
AGENT_TQ_INPUT_DIM: int = 95        # full live-mode dim (matches AgentTQ default)
AGENT_TQ_INPUT_DIM_HIST: int = 70   # historical mode (no Phase 1 TQ blocks)

# ── Source slices in master flat vector ──────────────────────────────────────
# Each tuple: (start, end, block_name, available_in_historical)
AGENT_TQ_SOURCE_BLOCKS: List[Tuple[int, int, str, bool]] = [
    (150, 180, "flow_volume",       True),   # 30 dims
    (180, 200, "microstructure",    True),   # 20 dims
    (250, 270, "sentiment_regime",  True),   # 20 dims
    (295, 310, "phase1_smart_money",  False), # 15 dims — live only
    (332, 350, "phase1_quote_pressure", False), # 18 dims — live only  (within 350-dim live vector)
]

# ── Feature names ─────────────────────────────────────────────────────────────
_FLOW_VOLUME_FEATS = [
    "call_put_vol_ratio", "call_put_oi_ratio", "call_put_premium_ratio", "net_cp_bias",
    "passive_volume", "aggressive_volume", "sweep_volume", "aggression_ratio",
    "call_aggression", "put_aggression",
    "small_trade_vol", "medium_trade_vol", "large_trade_vol", "block_trade_vol",
    "total_premium", "avg_premium", "vwap_premium",
    "buy_volume", "sell_volume", "buy_sell_imbalance",
    "flow_1m", "flow_5m", "flow_15m", "flow_30m",
    "dark_pool_pct", "lit_pct",
    "trade_count", "avg_trade_size", "trade_velocity", "vol_concentration",
]
_MICROSTRUCTURE_FEATS = [
    "spread_mean", "spread_pct", "spread_rolling", "spread_std",
    "bid_ask_imbalance", "tob_imbalance", "depth_imbalance", "imbalance_vol", "sustained_imbalance",
    "quote_frequency", "cancel_rate", "improvement_rate",
    "trades_per_min", "volume_per_min",
    "effective_spread", "realized_spread",
    "temp_impact", "short_impact", "size_impact_corr", "impact_asymmetry",
]
_SENTIMENT_REGIME_FEATS = [
    "cp_sentiment", "premium_sentiment", "flow_sentiment",
    "iv_percentile", "iv_expansion", "iv_contraction", "vol_regime",
    "momentum_1d", "momentum_5d", "momentum_20d",
    "trend_strength", "stress_indicator", "fear_indicator",
    "spx_corr", "vix_corr", "sector_corr", "beta_to_spx",
    "iv_vix_ratio", "relative_iv", "vix_term_impact",
]
_SMART_MONEY_FEATS = [
    "sm_sweep_ratio", "sm_block_premium", "sm_delta_weighted_flow",
    "sm_iv_timing", "sm_cross_strike_pattern",
    "sm_early_session_flow", "sm_late_session_flow",
    "sm_call_whale_ratio", "sm_put_whale_ratio",
    "sm_aggressor_ratio", "sm_continuation_score",
    "sm_reversal_score", "sm_institutional_proxy",
    "sm_dark_pool_flow", "sm_composite_score",
]
_QUOTE_PRESSURE_FEATS = [
    "qp_bid_lift_rate", "qp_ask_hit_rate", "qp_cancel_replace_ratio",
    "qp_depth_refresh_rate", "qp_spread_compression",
    "qp_call_pressure", "qp_put_pressure", "qp_net_pressure",
    "qp_top_book_stability", "qp_queue_imbalance",
    "qp_routing_lit_pct", "qp_routing_dark_pct",
    "qp_hidden_order_proxy", "qp_price_improvement_rate",
    "qp_effective_spread_ratio", "qp_short_term_alpha",
    "qp_quote_stuffing_indicator", "qp_composite_pressure",
]

AGENT_TQ_FEATURE_NAMES_HIST: List[str] = (
    _FLOW_VOLUME_FEATS      # 30
    + _MICROSTRUCTURE_FEATS # 20
    + _SENTIMENT_REGIME_FEATS  # 20
)  # = 70

AGENT_TQ_FEATURE_NAMES_LIVE: List[str] = (
    AGENT_TQ_FEATURE_NAMES_HIST
    + _SMART_MONEY_FEATS     # +15
    + _QUOTE_PRESSURE_FEATS  # +18 → but TQ model clips at tq_feat_dim=95 anyway
)  # = 70 + 25 = 95 (model clips to 95 internally via _fix_feat_dim)

assert len(AGENT_TQ_FEATURE_NAMES_HIST) == AGENT_TQ_INPUT_DIM_HIST, (
    f"Agent TQ hist dim mismatch: {len(AGENT_TQ_FEATURE_NAMES_HIST)} != {AGENT_TQ_INPUT_DIM_HIST}"
)
assert len(AGENT_TQ_FEATURE_NAMES_LIVE) >= AGENT_TQ_INPUT_DIM, (
    f"Agent TQ live dim too short: {len(AGENT_TQ_FEATURE_NAMES_LIVE)} < {AGENT_TQ_INPUT_DIM}"
)
