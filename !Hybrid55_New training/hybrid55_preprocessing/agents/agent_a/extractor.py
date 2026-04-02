"""
Agent A Extractor
Builds the 53-dim feature vector from Theta Data historical + OI CSVs.

Only uses columns populated in Theta EOD snapshots.
Calls shared extractors from extractors/ — does NOT reimplement math.

Usage:
    from hybrid55_preprocessing.agents.agent_a.extractor import AgentAExtractor

    extractor = AgentAExtractor()
    features = extractor.extract(hist_df, oi_df, underlying_price)
    # features.shape == (53,)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from ...extractors.data_validation import filter_dead_columns
from ...extractors.active_chain_filter import filter_active_chain
from .feature_config import AGENT_A_DIM, AGENT_A_FEATURES, DROP_COLS

logger = logging.getLogger("hybrid55.agent_a")


class AgentAExtractor:
    """
    Dedicated extractor for Agent A (EOD Theta historical).
    Output: np.ndarray shape (53,)
    """

    def __init__(self):
        self.alert_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        hist_df: pd.DataFrame,
        oi_df: Optional[pd.DataFrame] = None,
        underlying_price: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extract Agent A features.

        Args:
            hist_df: Theta historical chain snapshot (one timestamp)
            oi_df:   Theta OI DataFrame for same day (joined on strike+right)
            underlying_price: SPX spot price (taken from hist_df if None)

        Returns:
            np.ndarray of shape (53,)
        """
        try:
            features = self._extract_inner(hist_df, oi_df, underlying_price)
        except Exception as e:
            self._alert(f"[AGENT A FAIL] {e}")
            features = np.zeros(AGENT_A_DIM, dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        self._check_zeros(features)

        assert features.shape == (AGENT_A_DIM,), (
            f"Agent A shape mismatch: expected ({AGENT_A_DIM},), got {features.shape}"
        )
        return features

    def get_feature_names(self) -> List[str]:
        return list(AGENT_A_FEATURES)

    # ------------------------------------------------------------------ #
    #  Internal extraction                                                 #
    # ------------------------------------------------------------------ #

    def _extract_inner(
        self,
        hist_df: pd.DataFrame,
        oi_df: Optional[pd.DataFrame],
        underlying_price: Optional[float],
    ) -> np.ndarray:
        # Drop dead/categorical columns
        hist_df = hist_df.drop(
            columns=[c for c in DROP_COLS if c in hist_df.columns], errors="ignore"
        )
        hist_df = filter_dead_columns(hist_df, mode="greek")

        # Clip lambda to avoid extreme values for deep OTM
        if "lambda" in hist_df.columns:
            hist_df = hist_df.copy()
            hist_df["lambda"] = hist_df["lambda"].clip(-500, 500)

        spot = underlying_price or float(hist_df["underlying_price"].iloc[0])

        # Join OI
        if oi_df is not None:
            join_cols = [c for c in ("strike", "right") if c in oi_df.columns]
            oi_sub = oi_df[join_cols + ["open_interest"]].copy()
            hist_df = hist_df.merge(oi_sub, on=join_cols, how="left")
            hist_df["open_interest"] = hist_df["open_interest"].fillna(0)
        else:
            hist_df["open_interest"] = 0

        # Apply active chain filter (shared)
        active_df = filter_active_chain(hist_df)
        if active_df.empty:
            self._alert("[AGENT A] Active chain filter returned empty DataFrame — using unfiltered")
            active_df = hist_df.copy()

        calls = active_df[active_df["delta"] > 0].copy() if "delta" in active_df.columns else pd.DataFrame()
        puts  = active_df[active_df["delta"] < 0].copy() if "delta" in active_df.columns else pd.DataFrame()

        # --- ATM greeks (9) ---
        atm_row = self._atm_slice(active_df, spot)
        atm_greeks = [
            atm_row.get("delta", 0),       atm_row.get("gamma", 0),
            atm_row.get("vega", 0),        atm_row.get("theta", 0),
            atm_row.get("vanna", 0),       atm_row.get("charm", 0),
            atm_row.get("implied_vol", 0),
            atm_row.get("spread", 0),      atm_row.get("spread_pct", 0),
        ]

        # --- GEX / Vanna / Charm exposures (9) ---
        call_gex   = self._gex(calls)
        put_gex    = self._gex(puts)
        net_gex    = call_gex - put_gex
        total_gex  = call_gex + put_gex
        gex_flip_dist = self._gex_flip_dist(hist_df, spot)
        total_vanna = self._greek_exp(hist_df, "vanna")
        net_vanna   = self._greek_exp(calls, "vanna") - self._greek_exp(puts, "vanna")
        total_charm = self._greek_exp(hist_df, "charm")
        net_charm   = self._greek_exp(calls, "charm") - self._greek_exp(puts, "charm")
        gex_feats = [total_gex, call_gex, put_gex, net_gex, gex_flip_dist,
                     total_vanna, net_vanna, total_charm, net_charm]

        # --- OI structure (10) ---
        call_oi    = float(calls["open_interest"].sum()) if len(calls) > 0 else 0.0
        put_oi     = float(puts["open_interest"].sum())  if len(puts)  > 0 else 0.0
        pc_oi      = put_oi / (call_oi + 1e-8)
        atm_mask   = hist_df["dist_atm_pct"].abs() <= 1.0 if "dist_atm_pct" in hist_df.columns else pd.Series(False, index=hist_df.index)
        oi_at_atm  = float(hist_df.loc[atm_mask, "open_interest"].sum())
        total_oi   = float(hist_df["open_interest"].sum()) + 1e-8
        oi_skew    = float(((hist_df["strike"] - spot) * hist_df["open_interest"]).sum() / (total_oi * spot))
        c_oi_by_k  = calls.groupby("strike")["open_interest"].sum() if len(calls) > 0 else pd.Series(dtype=float)
        p_oi_by_k  = puts.groupby("strike")["open_interest"].sum()  if len(puts)  > 0 else pd.Series(dtype=float)
        max_call_k = float(c_oi_by_k.idxmax()) if len(c_oi_by_k) > 0 else spot
        max_put_k  = float(p_oi_by_k.idxmax())  if len(p_oi_by_k)  > 0 else spot
        dist_call_wall = (max_call_k - spot) / spot
        dist_put_wall  = (spot - max_put_k)  / spot
        wall_asym      = dist_call_wall - dist_put_wall
        oi_feats = [call_oi, put_oi, pc_oi, oi_at_atm, oi_skew,
                    max_call_k, max_put_k, dist_call_wall, dist_put_wall, wall_asym]

        # --- IV surface (7) ---
        iv_feats = self._iv_surface(active_df, calls, puts, atm_row)

        # --- Liquidity (3) ---
        mean_sp  = float(active_df["spread_pct"].mean()) if "spread_pct" in active_df.columns and len(active_df) > 0 else 0.0
        sp_std   = float(active_df["spread_pct"].std())  if "spread_pct" in active_df.columns and len(active_df) > 1 else 0.0
        pct_liq  = len(active_df) / (len(hist_df) + 1e-8)
        liq_feats = [mean_sp, sp_std, pct_liq]

        # --- Quote imbalance (5) ---
        bid_sz  = float(hist_df["bid_size"].sum()) if "bid_size" in hist_df.columns else 0.0
        ask_sz  = float(hist_df["ask_size"].sum()) if "ask_size" in hist_df.columns else 0.0
        denom   = bid_sz + ask_sz + 1e-8
        q_imb   = (bid_sz - ask_sz) / denom
        c_qimb  = self._quote_imb(calls)
        p_qimb  = self._quote_imb(puts)
        imb_feats = [bid_sz, ask_sz, q_imb, c_qimb, p_qimb]

        # --- Delta-bucketed OI (6) ---
        delta_feats = self._delta_bucketed_oi(hist_df, calls, puts, spot)

        # --- DTE structure (4) ---
        dte_col  = "dte_int" if "dte_int" in hist_df.columns else ("dte" if "dte" in hist_df.columns else None)
        if dte_col:
            dte_s    = pd.to_numeric(hist_df[dte_col], errors="coerce").fillna(0)
            dte_min  = float(dte_s.min())
            dte_wtd  = float((dte_s * hist_df["open_interest"]).sum() / (total_oi))
            frac_0   = float(hist_df.loc[hist_df[dte_col] == 0, "open_interest"].sum() / total_oi)
            frac_7   = float(hist_df.loc[hist_df[dte_col] <= 7, "open_interest"].sum() / total_oi)
        else:
            dte_min = dte_wtd = frac_0 = frac_7 = 0.0
        dte_feats = [dte_min, dte_wtd, frac_0, frac_7]

        # --- Assemble ---
        feat_vec = (
            atm_greeks + gex_feats + oi_feats + iv_feats +
            liq_feats + imb_feats + delta_feats + dte_feats
        )
        return np.array(feat_vec, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Shared helpers (private, Agent A specific formulas)                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _atm_slice(df: pd.DataFrame, spot: float) -> dict:
        if df.empty or "strike" not in df.columns:
            return {}
        df2 = df.copy()
        df2["_dist"] = (df2["strike"] - spot).abs()
        row = df2.nsmallest(1, "_dist").iloc[0]
        return row.to_dict()

    @staticmethod
    def _gex(df: pd.DataFrame) -> float:
        if df.empty or "gamma" not in df.columns or "open_interest" not in df.columns:
            return 0.0
        return float((df["gamma"] * df["open_interest"] * 100).sum())

    @staticmethod
    def _gex_flip_dist(df: pd.DataFrame, spot: float) -> float:
        if "gamma" not in df.columns or "open_interest" not in df.columns:
            return 0.0
        gex_by_k = df.groupby("strike").apply(
            lambda g: (g["gamma"] * g["open_interest"] * 100).sum()
        ).sort_index()
        cumgex = gex_by_k.cumsum()
        flips  = gex_by_k.index[(cumgex.shift(1, fill_value=0) * cumgex) < 0]
        if len(flips) == 0:
            return 0.0
        return float((flips[0] - spot) / spot)

    @staticmethod
    def _greek_exp(df: pd.DataFrame, greek: str) -> float:
        if df.empty or greek not in df.columns or "open_interest" not in df.columns:
            return 0.0
        return float((df[greek] * df["open_interest"] * 100).sum())

    @staticmethod
    def _iv_at_delta(df: pd.DataFrame, target: float, tol: float = 0.05) -> float:
        if df.empty or "delta" not in df.columns or "implied_vol" not in df.columns:
            return 0.0
        mask = (df["delta"].abs() - target).abs() <= tol
        sub  = df[mask]
        return float(sub["implied_vol"].mean()) if len(sub) > 0 else 0.0

    def _iv_surface(self, active_df, calls, puts, atm_row) -> list:
        iv_atm   = float(atm_row.get("implied_vol", 0))
        iv_25c   = self._iv_at_delta(calls, 0.25)
        iv_25p   = self._iv_at_delta(puts,  0.25)
        iv_skew  = iv_25p - iv_25c
        iv_10p   = self._iv_at_delta(puts,  0.10)
        dte_col  = "dte_int" if "dte_int" in active_df.columns else ("dte" if "dte" in active_df.columns else None)
        if dte_col:
            near_iv = float(active_df.loc[active_df[dte_col] <= 7,  "implied_vol"].mean()) if "implied_vol" in active_df.columns else iv_atm
            far_iv  = float(active_df.loc[active_df[dte_col] >  7,  "implied_vol"].mean()) if "implied_vol" in active_df.columns else iv_atm
            near_iv = near_iv if not np.isnan(near_iv) else iv_atm
            far_iv  = far_iv  if not np.isnan(far_iv)  else iv_atm
        else:
            near_iv = far_iv = iv_atm
        iv_term  = far_iv - near_iv
        iv_curve = 0.5 * (iv_25c + iv_25p) - iv_atm
        return [iv_atm, iv_25c, iv_25p, iv_skew, iv_10p, iv_term, iv_curve]

    @staticmethod
    def _delta_bucketed_oi(df, calls, puts, spot) -> list:
        def oi_in_bucket(src, lo, hi):
            if src.empty or "delta" not in src.columns or "open_interest" not in src.columns:
                return 0.0
            mask = (src["delta"].abs() >= lo) & (src["delta"].abs() < hi)
            return float(src.loc[mask, "open_interest"].sum())
        oi_dp_deep = oi_in_bucket(puts,  0.0, 0.2)
        oi_dp_otm  = oi_in_bucket(puts,  0.2, 0.4)
        oi_atm_b   = oi_in_bucket(df,    0.4, 0.6)
        oi_dc_otm  = oi_in_bucket(calls, 0.2, 0.4)
        oi_dc_deep = oi_in_bucket(calls, 0.0, 0.2)
        net_delta  = float((df["delta"] * df["open_interest"]).sum()) if "delta" in df.columns and "open_interest" in df.columns else 0.0
        return [oi_dp_deep, oi_dp_otm, oi_atm_b, oi_dc_otm, oi_dc_deep, net_delta]

    @staticmethod
    def _quote_imb(df: pd.DataFrame) -> float:
        if df.empty or "bid_size" not in df.columns or "ask_size" not in df.columns:
            return 0.0
        b = df["bid_size"].sum()
        a = df["ask_size"].sum()
        return float((b - a) / (b + a + 1e-8))

    # ------------------------------------------------------------------ #
    #  Alert helpers                                                        #
    # ------------------------------------------------------------------ #

    def _check_zeros(self, features: np.ndarray, threshold: float = 0.50) -> None:
        zero_rate = float((features == 0).mean())
        if zero_rate >= threshold:
            self._alert(
                f"[ZERO ALERT] Agent A: {zero_rate:.1%} zero "
                f"({int(zero_rate * AGENT_A_DIM)}/{AGENT_A_DIM})"
            )

    def _alert(self, msg: str) -> None:
        import logging
        logging.getLogger("hybrid55.agent_a").warning(msg)
        self.alert_log.append({"ts": pd.Timestamp.now().isoformat(), "msg": msg})
