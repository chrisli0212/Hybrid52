"""
Agent K Feature Configuration
Agent K: Greeks Core Agent (deep MLP with gamma-squeeze detector)

Input:
    static: (B, 75) — GREEK_BY_STRIKE full block [0:75]
    seq:    not used by AgentK (accepted for API compat, can be zeros)

Source block in master 311-dim flat vector:
    GREEK_BY_STRIKE  [0:75]  75 dims
        ├─ bucket_greeks  [0:65]  5 delta-buckets × 13 greeks each
        ├─ atm_greeks     [65:72] 7 ATM-slice greeks
        └─ skew_metrics   [72:75] 3 call/put/vol skew metrics
"""

from typing import List

# ── Dimension constant ─────────────────────────────────────────────────────────
AGENT_K_INPUT_DIM: int = 75   # GREEK_BY_STRIKE full block, master[0:75]

# ── Source slice in master flat vector ──────────────────────────────────────
AGENT_K_SLICE_START: int = 0
AGENT_K_SLICE_END: int = 75

# ── Sub-block layout for validation ───────────────────────────────────────
AGENT_K_SUB_BLOCKS = {
    "bucket_greeks": (0,  65),   # 5 buckets × 13 greeks
    "atm_greeks":    (65, 72),   # 7 ATM-slice greeks
    "skew_metrics":  (72, 75),   # callputivdiff, voltermslope, putskewintensity
}

# ── Delta bucket definitions (must match feature_config_v2.DELTA_BUCKETS) ───
_DELTA_BUCKETS = [
    "deep_otm",  # |delta| in [0.0, 0.2]
    "otm",       # |delta| in [0.2, 0.4]
    "atm",       # |delta| in [0.4, 0.6]
    "itm",       # |delta| in [0.6, 0.8]
    "deep_itm",  # |delta| in [0.8, 1.0]
]
_GREEKS_PER_BUCKET = [
    "delta", "gamma", "vega", "theta", "lambda",
    "vanna", "charm", "implied_vol",
    "open_interest", "moneyness",
    "bid", "ask", "mid",
]
_ATM_GREEKS = ["delta", "gamma", "vega", "theta", "lambda", "vanna", "charm"]

# ── Ordered feature names for Agent K ──────────────────────────────────────
AGENT_K_FEATURE_NAMES: List[str] = [
    f"{bucket}_{greek}"
    for bucket in _DELTA_BUCKETS
    for greek in _GREEKS_PER_BUCKET
] + [f"atm_{g}" for g in _ATM_GREEKS] + [
    "callputivdiff", "voltermslope", "putskewintensity"
]

assert len(AGENT_K_FEATURE_NAMES) == AGENT_K_INPUT_DIM, (
    f"Agent K feature count mismatch: {len(AGENT_K_FEATURE_NAMES)} != {AGENT_K_INPUT_DIM}"
)
