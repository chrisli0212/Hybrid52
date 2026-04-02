"""
Agent A Feature Configuration.

Agent A: EOD Theta historical snapshot agent.
Data source: Theta historical CSVs (greek + OI join).
Input dim: 53

Exclusions (zero/sparse in EOD chain):
  open, high, low, close, count  — always 0.0
  volume, vwap                   — zero >95% of contracts
  bid_exchange, ask_exchange     — categorical, not float
"""

AGENT_A_GROUPS = {
    "atm_greeks": [
        "atm_delta", "atm_gamma", "atm_vega", "atm_theta",
        "atm_vanna", "atm_charm", "atm_implied_vol",
        "atm_spread", "atm_spread_pct",
    ],
    "gex_vanna_charm": [
        "total_gex", "call_gex", "put_gex", "net_gex", "gex_flip_dist",
        "total_vanna_exp", "net_vanna_exp",
        "total_charm_exp", "net_charm_exp",
    ],
    "oi_structure": [
        "call_oi_total", "put_oi_total", "put_call_oi_ratio",
        "oi_at_atm", "oi_skew",
        "max_call_oi_strike", "max_put_oi_strike",
        "dist_to_call_wall", "dist_to_put_wall", "wall_asymmetry",
    ],
    "iv_surface": [
        "iv_atm", "iv_25d_call", "iv_25d_put", "iv_skew_25d",
        "iv_10d_put", "iv_term_slope", "iv_smile_curv",
    ],
    "liquidity": [
        "mean_spread_pct", "spread_std", "pct_liquid_strikes",
    ],
    "quote_imbalance": [
        "bid_size_total", "ask_size_total", "quote_imbalance",
        "call_quote_imb", "put_quote_imb",
    ],
    "delta_bucketed_oi": [
        "oi_deep_otm_put", "oi_otm_put", "oi_atm",
        "oi_otm_call", "oi_deep_otm_call", "net_delta_exposure",
    ],
    "dte_structure": [
        "dte_min", "dte_wtd_avg", "frac_0dte", "frac_7d",
    ],
}

# Flat ordered list of all features
AGENT_A_FEATURES: list = []
for _group_feats in AGENT_A_GROUPS.values():
    AGENT_A_FEATURES.extend(_group_feats)

AGENT_A_DIM: int = len(AGENT_A_FEATURES)  # must == 53

# Columns from Theta historical CSV required by this agent
REQUIRED_GREEK_COLS = [
    "strike", "right", "cp_sign", "dte",
    "bid", "ask", "bid_size", "ask_size", "spread", "spread_pct", "mid",
    "delta", "gamma", "vega", "theta", "vanna", "charm", "implied_vol",
    "underlying_price", "moneyness", "dist_atm_pct",
]

# Columns from Theta OI CSV (joined on symbol+expiration+strike+right)
REQUIRED_OI_COLS = ["open_interest"]

# Columns to DROP before extraction (zero/sparse/categorical in EOD snapshot)
DROP_COLS = [
    "open", "high", "low", "close", "count",
    "volume", "vwap",
    "bid_exchange", "ask_exchange",
]

assert AGENT_A_DIM == 53, f"Agent A: expected 53 features, got {AGENT_A_DIM}"
