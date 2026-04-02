"""
Agent A Feature Extractor
Builds the 53-d feature vector from Theta Data historical + OI CSVs.
Only uses columns that are actually populated in Theta EOD snapshots.

Usage:
    from hybrid52_preprocessing.extract_agent_a_features import extract_agent_a_snapshot
    feats = extract_agent_a_snapshot(hist_df, oi_df, underlying_price)
"""

import numpy as np
import pandas as pd
from typing import Optional
from .feature_config_agent_a import AGENT_A_FEATURES, AGENT_A_INPUT_DIM, THETA_HIST_DROP


def _filter_liquid(df: pd.DataFrame, min_bid: float = 0.05) -> pd.DataFrame:
    return df[df["bid"] >= min_bid].copy()


def _atm_slice(df: pd.DataFrame, spot: float) -> pd.Series:
    """Return the single row closest to ATM."""
    df = df.copy()
    df["_atm_dist"] = (df["strike"] - spot).abs()
    return df.nsmallest(1, "_atm_dist").iloc[0]


def _gex(df: pd.DataFrame) -> float:
    """Gamma Exposure = sum(gamma * OI * 100)"""
    if "open_interest" not in df.columns:
        return 0.0
    return float((df["gamma"] * df["open_interest"] * 100).sum())


def extract_agent_a_snapshot(
    hist_df: pd.DataFrame,
    oi_df: Optional[pd.DataFrame],
    underlying_price: Optional[float] = None,
) -> np.ndarray:
    """
    Args:
        hist_df: Theta historical chain snapshot DataFrame (one timestamp)
        oi_df:   Theta OI DataFrame for the same day (joined on strike+right)
        underlying_price: SPX spot price (taken from hist_df if None)

    Returns:
        np.ndarray of shape (53,) — Agent A feature vector
    """
    # Drop zero/constant/categorical columns
    hist_df = hist_df.drop(columns=[c for c in THETA_HIST_DROP if c in hist_df.columns])

    # Clip lambda to avoid extreme values for deep OTM
    if "lambda" in hist_df.columns:
        hist_df["lambda"] = hist_df["lambda"].clip(-500, 500)

    spot = underlying_price or float(hist_df["underlying_price"].iloc[0])

    # Join OI
    if oi_df is not None:
        join_cols = ["strike", "right"]
        available = [c for c in join_cols if c in oi_df.columns]
        oi_sub = oi_df[available + ["open_interest"]].copy()
        hist_df = hist_df.merge(oi_sub, on=available, how="left")
        hist_df["open_interest"] = hist_df["open_interest"].fillna(0)
    else:
        hist_df["open_interest"] = 0

    calls = _filter_liquid(hist_df[hist_df["right"] == "CALL"])
    puts  = _filter_liquid(hist_df[hist_df["right"] == "PUT"])
    all_liq = _filter_liquid(hist_df)

    # ── ATM greeks ──────────────────────────────────────────────────────────
    atm_row = _atm_slice(all_liq, spot)
    atm_greeks = [
        atm_row.get("delta", 0), atm_row.get("gamma", 0),
        atm_row.get("vega", 0),  atm_row.get("theta", 0),
        atm_row.get("vanna", 0), atm_row.get("charm", 0),
        atm_row.get("implied_vol", 0),
        atm_row.get("spread", 0), atm_row.get("spread_pct", 0),
    ]

    # ── GEX / Vanna / Charm exposures ────────────────────────────────────────
    call_gex  = _gex(calls)
    put_gex   = _gex(puts)
    net_gex   = call_gex - put_gex
    total_gex = call_gex + put_gex

    # GEX flip: strike where cumulative GEX crosses zero
    gex_by_strike = hist_df.groupby("strike").apply(
        lambda g: (g["gamma"] * g["open_interest"] * 100 * g.get("cp_sign", 1)).sum()
    ).sort_index()
    cumgex = gex_by_strike.cumsum()
    flip_strikes = gex_by_strike.index[(cumgex.shift(1, fill_value=0) * cumgex) < 0]
    gex_flip_dist = float((flip_strikes[0] - spot) / spot) if len(flip_strikes) > 0 else 0.0

    def _greek_exp(df, greek):
        if "open_interest" not in df.columns or greek not in df.columns:
            return 0.0
        return float((df[greek] * df["open_interest"] * 100).sum())

    total_vanna = _greek_exp(hist_df, "vanna")
    net_vanna   = _greek_exp(calls, "vanna") - _greek_exp(puts, "vanna")
    total_charm = _greek_exp(hist_df, "charm")
    net_charm   = _greek_exp(calls, "charm") - _greek_exp(puts, "charm")

    gex_feats = [total_gex, call_gex, put_gex, net_gex, gex_flip_dist,
                 total_vanna, net_vanna, total_charm, net_charm]

    # ── OI structure ─────────────────────────────────────────────────────────
    call_oi = float(calls["open_interest"].sum())
    put_oi  = float(puts["open_interest"].sum())
    pc_oi_ratio = put_oi / (call_oi + 1e-8)

    atm_mask = hist_df["dist_atm_pct"].abs() <= 1.0
    oi_at_atm = float(hist_df.loc[atm_mask, "open_interest"].sum())

    # OI skew: weighted avg strike vs spot
    total_oi = float(hist_df["open_interest"].sum()) + 1e-8
    oi_skew = float(((hist_df["strike"] - spot) * hist_df["open_interest"]).sum() / (total_oi * spot))

    call_oi_by_strike = calls.groupby("strike")["open_interest"].sum()
    put_oi_by_strike  = puts.groupby("strike")["open_interest"].sum()
    max_call_k = float(call_oi_by_strike.idxmax()) if len(call_oi_by_strike) > 0 else spot
    max_put_k  = float(put_oi_by_strike.idxmax())  if len(put_oi_by_strike) > 0  else spot
    dist_call_wall = (max_call_k - spot) / spot
    dist_put_wall  = (spot - max_put_k) / spot
    wall_asym = dist_call_wall - dist_put_wall

    oi_feats = [call_oi, put_oi, pc_oi_ratio, oi_at_atm, oi_skew,
                max_call_k, max_put_k, dist_call_wall, dist_put_wall, wall_asym]

    # ── IV surface ───────────────────────────────────────────────────────────
    def iv_at_delta(df, target_delta, tol=0.05):
        mask = (df["delta"].abs() - target_delta).abs() <= tol
        sub = df[mask]
        return float(sub["implied_vol"].mean()) if len(sub) > 0 else float(atm_row.get("implied_vol", 0))

    iv_atm     = float(atm_row.get("implied_vol", 0))
    iv_25c     = iv_at_delta(calls, 0.25)
    iv_25p     = iv_at_delta(puts,  0.25)
    iv_skew_25 = iv_25p - iv_25c
    iv_10p     = iv_at_delta(puts, 0.10)

    # Term structure: DTE≤7 vs DTE>7
    near = _filter_liquid(hist_df[hist_df["dte"] <= 7])
    far  = _filter_liquid(hist_df[hist_df["dte"] > 7])
    iv_near = float(near["implied_vol"].mean()) if len(near) > 0 else iv_atm
    iv_far  = float(far["implied_vol"].mean())  if len(far) > 0  else iv_atm
    iv_term_slope = iv_far - iv_near

    iv_smile_curv = 0.5 * (iv_25c + iv_25p) - iv_atm

    iv_feats = [iv_atm, iv_25c, iv_25p, iv_skew_25, iv_10p, iv_term_slope, iv_smile_curv]

    # ── Liquidity ─────────────────────────────────────────────────────────────
    mean_spread_pct  = float(all_liq["spread_pct"].mean()) if len(all_liq) > 0 else 0.0
    spread_std       = float(all_liq["spread_pct"].std())  if len(all_liq) > 0 else 0.0
    pct_liquid       = len(all_liq) / (len(hist_df) + 1e-8)
    liq_feats = [mean_spread_pct, spread_std, pct_liquid]

    # ── Quote book imbalance ──────────────────────────────────────────────────
    bid_sz_total = float(hist_df["bid_size"].sum())
    ask_sz_total = float(hist_df["ask_size"].sum())
    denom = bid_sz_total + ask_sz_total + 1e-8
    quote_imb  = (bid_sz_total - ask_sz_total) / denom
    call_qimb  = float(
        (calls["bid_size"].sum() - calls["ask_size"].sum()) /
        (calls["bid_size"].sum() + calls["ask_size"].sum() + 1e-8)
    ) if len(calls) > 0 else 0.0
    put_qimb   = float(
        (puts["bid_size"].sum() - puts["ask_size"].sum()) /
        (puts["bid_size"].sum() + puts["ask_size"].sum() + 1e-8)
    ) if len(puts) > 0 else 0.0
    imb_feats = [bid_sz_total, ask_sz_total, quote_imb, call_qimb, put_qimb]

    # ── Delta-bucketed OI ─────────────────────────────────────────────────────
    def oi_in_delta_bucket(df, lo, hi):
        mask = (df["delta"].abs() >= lo) & (df["delta"].abs() < hi)
        return float(df.loc[mask, "open_interest"].sum())

    oi_dp_deep = oi_in_delta_bucket(puts, 0.0, 0.2)
    oi_dp_otm  = oi_in_delta_bucket(puts, 0.2, 0.4)
    oi_atm_b   = oi_in_delta_bucket(hist_df, 0.4, 0.6)
    oi_dc_otm  = oi_in_delta_bucket(calls, 0.2, 0.4)
    oi_dc_deep = oi_in_delta_bucket(calls, 0.0, 0.2)
    net_delta_exp = float((hist_df["delta"] * hist_df["open_interest"]).sum())
    delta_feats = [oi_dp_deep, oi_dp_otm, oi_atm_b, oi_dc_otm, oi_dc_deep, net_delta_exp]

    # ── DTE structure ────────────────────────────────────────────────────────
    dte_min = float(hist_df["dte"].min())
    dte_wtd = float(
        (hist_df["dte"] * hist_df["open_interest"]).sum() /
        (hist_df["open_interest"].sum() + 1e-8)
    )
    frac_0dte = float(hist_df.loc[hist_df["dte"] == 0, "open_interest"].sum() / (total_oi))
    frac_7d   = float(hist_df.loc[hist_df["dte"] <= 7, "open_interest"].sum() / (total_oi))
    dte_feats = [dte_min, dte_wtd, frac_0dte, frac_7d]

    # ── Assemble and validate ────────────────────────────────────────────────
    feat_vec = (
        atm_greeks + gex_feats + oi_feats + iv_feats +
        liq_feats + imb_feats + delta_feats + dte_feats
    )
    assert len(feat_vec) == AGENT_A_INPUT_DIM, \
        f"Feature count mismatch: {len(feat_vec)} != {AGENT_A_INPUT_DIM}"

    arr = np.array(feat_vec, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    return arr
