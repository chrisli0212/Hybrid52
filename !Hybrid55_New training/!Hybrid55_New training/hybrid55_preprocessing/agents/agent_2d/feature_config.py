"""
Agent 2D Feature Configuration.

Agent 2D: CNN agent consuming the option chain as a 2D image tensor.
Input shape: (N_GREEKS, N_STRIKES, N_TIMESTEPS)

CRITICAL: these constants must match the CNN input_channels, height, width.
Change ONLY here. Both build_chain_2d.py and the live assembler read from here.
"""

from typing import List, Tuple

# Shape constants — single source of truth
AGENT_2D_N_GREEKS:     int = 5
AGENT_2D_N_STRIKES:    int = 30
AGENT_2D_N_TIMESTEPS:  int = 20

AGENT_2D_SHAPE: Tuple[int, int, int] = (
    AGENT_2D_N_GREEKS,
    AGENT_2D_N_STRIKES,
    AGENT_2D_N_TIMESTEPS,
)

# Greeks included in the 2D tensor (order matters — matches CNN channel order)
AGENT_2D_GREEKS: List[str] = [
    "delta",
    "gamma",
    "vega",
    "theta",
    "implied_vol",
]

assert len(AGENT_2D_GREEKS) == AGENT_2D_N_GREEKS, (
    f"Agent 2D: greek list length {len(AGENT_2D_GREEKS)} != N_GREEKS {AGENT_2D_N_GREEKS}"
)

# Delta range for strike binning
AGENT_2D_DELTA_RANGE: Tuple[float, float] = (-0.9, 0.9)

# Liquidity filter applied before chain building
AGENT_2D_MIN_BID: float = 0.05

# Normalization method per channel ('zscore' | 'minmax' | 'robust')
AGENT_2D_NORM_METHOD: str = "zscore"
