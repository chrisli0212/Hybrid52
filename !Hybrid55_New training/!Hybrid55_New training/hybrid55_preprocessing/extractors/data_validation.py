"""
Shared data validation & column filtering.
Single source of truth for dead/constant/metadata column lists.
All agents import from here — never duplicate these lists.
"""

from typing import List, Set

# Columns confirmed always-zero in Theta Data exports
DEAD_COLUMNS: Set[str] = {
    "speed", "vera", "zomma", "dual_gamma", "iv_error",
    "endpoint", "batch_id", "ts",
}

# Always-constant trade/quote fields
DEAD_TRADE_QUOTE_COLUMNS: Set[str] = {
    "ext_condition1", "ext_condition2", "ext_condition3", "ext_condition4",
    "bid_condition", "ask_condition",
    "endpoint", "batch_id", "ts",
}

METADATA_COLUMNS: Set[str] = {
    "symbol", "expiration", "strike", "right",
    "timestamp", "trade_date", "underlying_timestamp",
}


def get_excluded_columns() -> List[str]:
    """Greek columns that are always zero — exclude from all agents."""
    return sorted(DEAD_COLUMNS)


def get_trade_quote_excluded_columns() -> List[str]:
    """Trade/quote columns that are always constant — exclude from all agents."""
    return sorted(DEAD_TRADE_QUOTE_COLUMNS)


def get_usable_greek_columns() -> List[str]:
    """Core greek columns confirmed active in Theta Data."""
    return [
        "delta", "gamma", "vega", "theta", "lambda",
        "rho", "epsilon",
        "vanna", "charm", "vomma", "veta", "color",
        "dual_delta", "d1", "d2", "ultima",
        "implied_vol",
        "bid", "ask", "underlying_price",
        "open_interest", "moneyness", "dist_atm_pct",
        "mid", "spread", "spread_pct", "lambda_ratio",
        "dte_int", "cp_sign",
    ]


def get_metadata_columns() -> List[str]:
    return sorted(METADATA_COLUMNS)


def filter_dead_columns(df, mode: str = "greek"):
    """
    Drop dead/constant columns from a DataFrame.

    Args:
        df: input DataFrame
        mode: 'greek' to drop DEAD_COLUMNS, 'trade' to drop DEAD_TRADE_QUOTE_COLUMNS

    Returns:
        DataFrame with dead columns removed
    """
    import pandas as pd
    dead = DEAD_COLUMNS if mode == "greek" else DEAD_TRADE_QUOTE_COLUMNS
    to_drop = [c for c in dead if c in df.columns]
    return df.drop(columns=to_drop, errors="ignore")
