"""
Shared active chain filter.
All agents should call this BEFORE running any extractor.
Filters to liquid, delta-range contracts only.

Rules:
  - Liquidity: bid != 0 AND ask != 0 AND vega != 0
  - Calls: delta in [0.20, 0.90]
  - Puts:  delta in [-0.90, -0.20]
"""

import pandas as pd

CALL_DELTA_LO, CALL_DELTA_HI = 0.20, 0.90
PUT_DELTA_LO, PUT_DELTA_HI = -0.90, -0.20


def filter_active_chain(df: pd.DataFrame, min_bid: float = 0.05) -> pd.DataFrame:
    """
    Filter a Greek snapshot DataFrame to active liquid contracts only.

    Args:
        df: raw Greek snapshot (one timestamp)
        min_bid: minimum bid price for liquidity (default 0.05)

    Returns:
        Filtered DataFrame. May be empty — caller should handle gracefully.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Step 1: Liquidity filter
    for col in ("bid", "ask", "vega"):
        if col in out.columns:
            out = out[out[col].fillna(0) != 0]
    if "bid" in out.columns:
        out = out[out["bid"] >= min_bid]

    if out.empty:
        return out

    # Step 2: Delta range filter
    if "delta" not in out.columns:
        return out

    if "right" not in out.columns:
        # Infer from delta sign
        calls = (out["delta"] >= CALL_DELTA_LO) & (out["delta"] <= CALL_DELTA_HI)
        puts  = (out["delta"] >= PUT_DELTA_LO)  & (out["delta"] <= PUT_DELTA_HI)
        return out[calls | puts]

    right = out["right"].astype(str).str.strip().str.lower()
    is_call = right.isin(("call", "c"))
    is_put  = right.isin(("put",  "p"))

    call_ok = is_call & (out["delta"] >= CALL_DELTA_LO) & (out["delta"] <= CALL_DELTA_HI)
    put_ok  = is_put  & (out["delta"] >= PUT_DELTA_LO)  & (out["delta"] <= PUT_DELTA_HI)

    return out[call_ok | put_ok]
