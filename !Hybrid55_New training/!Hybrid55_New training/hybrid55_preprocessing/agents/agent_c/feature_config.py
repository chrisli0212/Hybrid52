"""
Agent C Feature Configuration
Agent C: Multi-Scale Attention Agent (CNN + MultiheadAttention + BiLSTM)

Input:
    seq:    (B, T, 69) — intraday rolling window, time-major
    static: accepted for API compat, unused inside AgentC

Source blocks in master 311-dim flat vector (T slices per bar):
    gamma_exposure   [ 75:105]  30 dims
    vanna_charm      [105:125]  20 dims
    iv_surface       [125:150]  25 dims
    csv_derived      [270:286]  16 dims  (lambda, distatm, spreadpct, chain-aux, OI)
    NOTE: total = 30+20+25+16 = 91 dims, but AgentC uses a curated subset of 69
          that excludes redundant sub-features within each block (see below).

The 69-dim selection (kept features per block):
    gamma_exposure:  30 dims kept  (full block)
    vanna_charm:     20 dims kept  (full block)
    iv_surface:      16 dims kept  (ivbymoneyness[7] + ivtermstructure[5] + volskewmetrics[4])
    csv_derived:      3 dims kept  (distatm_mean, distatm_weighted, iv_std)
    = 30 + 20 + 16 + 3 = 69
"""

from typing import List, Tuple

# ── Dimension constant ─────────────────────────────────────────────────────────
AGENT_C_INPUT_DIM: int = 69   # matches AgentC(input_dim=69) default

# ── Source slices: (start, end, block_name, keep_indices_within_block) ───
# keep_indices=None means keep ALL dims from that block
AGENT_C_SOURCE_BLOCKS: List[Tuple[int, int, str]] = [
    (75,  105, "gamma_exposure"),   # 30 dims — full block
    (105, 125, "vanna_charm"),      # 20 dims — full block
    (125, 141, "iv_surface_core"), # 16 dims — iv[0:7] + ivterm[0:5] + volskew[0:4]
    (273, 276, "csv_distatm"),      # 3 dims  — distatm_mean, distatm_weighted, iv_std (at 275)
]

# Exact master vector positions for the curated IV block (125:141)
# master[125:132] = ivbymoneyness (7 dims)
# master[132:137] = ivtermstructure (5 dims)
# master[137:141] = volskewmetrics first 4 (put_skew, call_skew, term_skew, smile_curv)
AGENT_C_IV_SLICE = (125, 141)     # 16 dims from iv_surface block

# Exact master positions for csv_distatm (3 dims)
# master[273] = distatm_mean
# master[274] = distatm_weighted
# master[275] = spreadpct_mean  <- used as iv_std proxy here
AGENT_C_CSV_SLICE = (273, 276)    # 3 dims

# ── Feature names (69 total) ────────────────────────────────────────────────
_GAMMA_FEATS = [
    f"gamma_strike_above_{i}" for i in range(10)
] + [
    f"gamma_strike_below_{i}" for i in range(10)
] + [
    "total_gamma", "call_gamma", "put_gamma", "net_gamma",
    "dealer_gamma_estimate", "gamma_flip_level", "dist_to_gamma_flip",
    "below_gamma_flip", "above_gamma_flip", "gamma_zone_strength",
]  # 30 dims

_VANNA_FEATS = [
    f"{b}_vanna" for b in ["deep_otm", "otm", "atm", "itm", "deep_itm"]
] + [
    f"{b}_charm" for b in ["deep_otm", "otm", "atm", "itm", "deep_itm"]
] + [
    "total_vanna", "call_vanna", "put_vanna",
    "total_charm", "call_charm", "put_charm",
    "vanna_gamma_ratio", "charm_theta_ratio", "vanna_vega_ratio", "net_vanna_strength",
]  # 20 dims

_IV_CORE_FEATS = [
    f"iv_moneyness_{lvl:.2f}".replace("-", "neg").replace(".", "p")
    for lvl in [-0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30]
] + [
    "iv_1w", "iv_1m", "iv_2m", "iv_3m", "iv_6m",
] + [
    "put_skew", "call_skew", "term_skew", "smile_curvature",
]  # 16 dims

_CSV_FEATS = ["distatm_mean", "distatm_weighted", "spreadpct_mean"]  # 3 dims

AGENT_C_FEATURE_NAMES: List[str] = (
    _GAMMA_FEATS       # 30
    + _VANNA_FEATS      # 20
    + _IV_CORE_FEATS    # 16
    + _CSV_FEATS        # 3
)  # = 69

assert len(AGENT_C_FEATURE_NAMES) == AGENT_C_INPUT_DIM, (
    f"Agent C feature count mismatch: {len(AGENT_C_FEATURE_NAMES)} != {AGENT_C_INPUT_DIM}"
)
