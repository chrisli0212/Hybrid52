"""
Agent A Feature Configuration — grounded in Theta Data availability.

Theta historical CSV provides (per contract per snapshot):
  bid, ask, bid_size, ask_size, spread, spread_pct, mid
  delta, gamma, vega, theta, vanna, charm, implied_vol, lambda
  strike, dte, moneyness, dist_atm_pct, underlying_price, cp_sign

Theta OI CSV provides (joined on symbol+expiration+strike+right):
  open_interest

EXCLUDED (zero/constant in EOD chain snapshot):
  open, high, low, close, count  → always 0.0
  volume, vwap                   → ~0 for 95%+ of strikes
  bid_exchange, ask_exchange     → categorical exchange codes, not float features
"""

AGENT_A_FEATURE_GROUPS = {
    "atm_greeks": [                        # 9 features — single ATM slice
        "atm_delta", "atm_gamma", "atm_vega", "atm_theta",
        "atm_vanna", "atm_charm", "atm_implied_vol",
        "atm_spread", "atm_spread_pct",
    ],
    "gex_vanna_charm": [                   # 9 features — OI-weighted cross-strike
        "total_gex", "call_gex", "put_gex", "net_gex", "gex_flip_dist",
        "total_vanna_exp", "net_vanna_exp",
        "total_charm_exp", "net_charm_exp",
    ],
    "oi_structure": [                      # 10 features — from OI file
        "call_oi_total", "put_oi_total", "put_call_oi_ratio",
        "oi_at_atm", "oi_skew",
        "max_call_oi_strike", "max_put_oi_strike",
        "dist_to_call_wall", "dist_to_put_wall", "wall_asymmetry",
    ],
    "iv_surface": [                        # 7 features
        "iv_atm", "iv_25d_call", "iv_25d_put", "iv_skew_25d",
        "iv_10d_put", "iv_term_slope", "iv_smile_curv",
    ],
    "liquidity": [                         # 3 features
        "mean_spread_pct", "spread_std", "pct_liquid_strikes",
    ],
    "quote_imbalance": [                   # 5 features
        "bid_size_total", "ask_size_total", "quote_imbalance",
        "call_quote_imb", "put_quote_imb",
    ],
    "delta_bucketed_oi": [                 # 6 features
        "oi_deep_otm_put", "oi_otm_put", "oi_atm",
        "oi_otm_call", "oi_deep_otm_call", "net_delta_exposure",
    ],
    "dte_structure": [                     # 4 features
        "dte_min", "dte_wtd_avg", "frac_0dte", "frac_7d",
    ],
}

AGENT_A_FEATURES = []
for group_feats in AGENT_A_FEATURE_GROUPS.values():
    AGENT_A_FEATURES.extend(group_feats)

AGENT_A_INPUT_DIM = len(AGENT_A_FEATURES)  # 53

# Columns from Theta historical CSV used in extraction
THETA_HIST_REQUIRED = [
    "strike", "right", "cp_sign", "dte",
    "bid", "ask", "bid_size", "ask_size", "spread", "spread_pct", "mid",
    "delta", "gamma", "vega", "theta", "vanna", "charm", "implied_vol",
    "underlying_price", "moneyness", "dist_atm_pct",
]
# lambda is available but clip to [-500, 500] before use
THETA_HIST_CLIPPED = {"lambda": (-500, 500)}

# Columns from Theta OI CSV (joined on symbol+expiration+strike+right)
THETA_OI_REQUIRED = ["open_interest"]

# Columns to DROP (zero/sparse/categorical in EOD snapshot)
THETA_HIST_DROP = [
    "open", "high", "low", "close", "count",  # always 0 in EOD chain snapshot
    "volume", "vwap",                           # zero for >95% of contracts
    "bid_exchange", "ask_exchange",             # categorical, not float
]

assert AGENT_A_INPUT_DIM == 53, f"Expected 53, got {AGENT_A_INPUT_DIM}"
