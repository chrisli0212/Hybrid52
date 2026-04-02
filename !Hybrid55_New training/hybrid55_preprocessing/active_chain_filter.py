"""
Active contract filter for option chain data.

Filters to liquid/active contracts only:
- Liquidity: (bid != 0) & (ask != 0) & (vega != 0)
- Delta range: calls [0.2, 0.9], puts [-0.9, -0.2]

Apply after groupby(timestamp), before feature extraction and chain building.
See plan: 1-Min Sequential Data at Scale + 2D Agent Streaming.
"""

import pandas as pd
from typing import Optional


# Delta range for active liquid strikes (calls 0.2-0.9, puts -0.9 to -0.2)
CALL_DELTA_LO, CALL_DELTA_HI = 0.2, 0.9
PUT_DELTA_LO, PUT_DELTA_HI = -0.9, -0.2


def filter_active_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a 1-min Greek DataFrame to active contracts only.

    - Liquidity first: drop rows where bid == 0 or ask == 0 or vega == 0.
    - Delta range: keep calls with delta in [0.2, 0.9], puts with delta in [-0.9, -0.2].
    Handles 'right' column values 'call'/'C' and 'put'/'P'.

    Returns filtered DataFrame (may be empty). Caller should skip timestamps
    that have zero rows after filter.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()

    # Liquidity: bid, ask, vega all non-zero
    for col in ("bid", "ask", "vega"):
        if col in out.columns:
            out = out[out[col].fillna(0) != 0]

    if out.empty:
        return out

    # Delta range: calls [0.2, 0.9], puts [-0.9, -0.2]
    if "delta" not in out.columns:
        return out

    if "right" not in out.columns:
        # Infer from delta: positive = call, negative = put
        calls = (out["delta"] >= 0) & (out["delta"] >= CALL_DELTA_LO) & (out["delta"] <= CALL_DELTA_HI)
        puts = (out["delta"] < 0) & (out["delta"] >= PUT_DELTA_LO) & (out["delta"] <= PUT_DELTA_HI)
        out = out[calls | puts]
        return out

    right = out["right"].astype(str).str.strip().str.lower()
    is_call = right.isin(("call", "c"))
    is_put = right.isin(("put", "p"))

    call_ok = is_call & (out["delta"] >= CALL_DELTA_LO) & (out["delta"] <= CALL_DELTA_HI)
    put_ok = is_put & (out["delta"] >= PUT_DELTA_LO) & (out["delta"] <= PUT_DELTA_HI)
    out = out[call_ok | put_ok]

    return out
