"""
OHLC feature extractor for 1-minute option bars.

Produces 25 sparse-safe chain-level features from open/high/low/close/volume/count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_mean(x: pd.Series) -> float:
    if x is None or len(x) == 0:
        return 0.0
    v = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(v.mean()) if len(v) else 0.0


def _safe_std(x: pd.Series) -> float:
    if x is None or len(x) == 0:
        return 0.0
    v = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(v.std()) if len(v) > 1 else 0.0


class OhlcFeatureExtractor:
    n_features = 25

    feature_names = [
        "ohlc_active_ratio",
        "ohlc_range_pct",
        "ohlc_body_pct",
        "ohlc_upper_shadow",
        "ohlc_lower_shadow",
        "ohlc_close_position",
        "ohlc_volume_weighted_return",
        "ohlc_cp_vol_ratio",
        "ohlc_volume_gini",
        "ohlc_high_vol_strike_dist",
        "ohlc_volume_skew",
        "ohlc_trade_fragmentation",
        "ohlc_atm_volume_share",
        "ohlc_otm_put_volume_share",
        "ohlc_total_volume_log",
        "ohlc_volume_momentum",
        "ohlc_avg_range_dispersion",
        "ohlc_call_put_range_ratio",
        "ohlc_vwap_moneyness",
        "ohlc_high_low_corr",
        "ohlc_close_open_skew",
        "ohlc_doji_pct",
        "ohlc_hammer_pct",
        "ohlc_shooting_star_pct",
        "ohlc_gap_pct",
    ]

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        x = np.sort(np.asarray(values, dtype=np.float64))
        if np.allclose(x.sum(), 0.0):
            return 0.0
        n = len(x)
        idx = np.arange(1, n + 1, dtype=np.float64)
        return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)

    def extract(self, ohlc_df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(self.n_features, dtype=np.float32)
        if ohlc_df is None or len(ohlc_df) == 0:
            return out

        df = ohlc_df.copy()
        for c in ("open", "high", "low", "close", "volume", "count", "strike"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)

        valid = df[
            (df.get("open", 0) > 0)
            & (df.get("high", 0) > 0)
            & (df.get("low", 0) > 0)
            & (df.get("close", 0) > 0)
        ].copy()
        if len(valid) == 0:
            return out

        out[0] = float(len(valid) / max(1, len(df)))  # active ratio

        rng = (valid["high"] - valid["low"]).clip(lower=0.0)
        open_safe = valid["open"].replace(0, np.nan)
        body = valid["close"] - valid["open"]
        out[1] = _safe_mean(rng / open_safe)
        out[2] = _safe_mean(body / open_safe)
        out[3] = _safe_mean((valid["high"] - np.maximum(valid["open"], valid["close"])) / rng.replace(0, np.nan))
        out[4] = _safe_mean((np.minimum(valid["open"], valid["close"]) - valid["low"]) / rng.replace(0, np.nan))
        out[5] = _safe_mean((valid["close"] - valid["low"]) / rng.replace(0, np.nan))

        vol = valid.get("volume", pd.Series(np.ones(len(valid)), index=valid.index)).fillna(0).clip(lower=0)
        vol_sum = float(vol.sum())
        weights = (vol / vol_sum) if vol_sum > 0 else pd.Series(np.ones(len(valid)) / len(valid), index=valid.index)

        body_pct = (body / open_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[6] = float((body_pct * weights).sum())

        right = valid.get("right", pd.Series([""] * len(valid), index=valid.index)).astype(str).str.upper()
        call_vol = float(vol[right == "C"].sum())
        put_vol = float(vol[right == "P"].sum())
        out[7] = float(call_vol / max(put_vol, 1e-6))

        out[8] = self._gini(vol.to_numpy(dtype=np.float64))

        if "strike" in valid.columns and "underlying_price" in valid.columns:
            idx = int(np.argmax(vol.to_numpy(dtype=np.float64)))
            strike = float(valid.iloc[idx]["strike"])
            und = float(pd.to_numeric(valid["underlying_price"], errors="coerce").dropna().mean() or 0.0)
            out[9] = float(abs(strike - und) / max(abs(und), 1e-6))

        out[10] = float(pd.Series(vol).skew()) if len(vol) > 2 else 0.0
        cnt = valid.get("count", pd.Series(np.zeros(len(valid)), index=valid.index)).fillna(0).clip(lower=0)
        out[11] = float(vol_sum / max(float(cnt.sum()), 1.0))
        out[12] = float(call_vol + put_vol) / max(vol_sum, 1e-6)
        if "moneyness" in valid.columns:
            m = pd.to_numeric(valid["moneyness"], errors="coerce")
            put_otm = vol[(right == "P") & (m < 1.0)].sum()
            out[13] = float(put_otm / max(vol_sum, 1e-6))
            out[18] = float((m.fillna(1.0) * weights).sum())
        out[14] = float(np.log1p(vol_sum))

        # Intra-minute proxy; true 5-bar momentum is handled downstream by sequence model
        out[15] = float(vol.std() / max(vol.mean(), 1e-6)) if len(vol) > 1 else 0.0

        range_pct = (rng / open_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[16] = _safe_std(range_pct)
        call_range = _safe_mean(range_pct[right == "C"])
        put_range = _safe_mean(range_pct[right == "P"])
        out[17] = float(call_range / max(put_range, 1e-6))

        hi = pd.to_numeric(valid["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        lo = pd.to_numeric(valid["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if len(hi) > 2 and np.std(hi) > 1e-12 and np.std(lo) > 1e-12:
            out[19] = float(np.corrcoef(hi, lo)[0, 1])
        out[20] = float(pd.Series(body_pct).skew()) if len(body_pct) > 2 else 0.0

        doji = (np.abs(body) < 0.1 * rng).astype(float)
        hammer = ((valid["close"] > valid["open"]) & ((valid["open"] - valid["low"]) > 2 * np.abs(body))).astype(float)
        star = ((valid["open"] > valid["close"]) & ((valid["high"] - valid["open"]) > 2 * np.abs(body))).astype(float)
        out[21] = float(doji.mean())
        out[22] = float(hammer.mean())
        out[23] = float(star.mean())

        # Gap proxy inside minute snapshot; true inter-bar gap can be learned in temporal model.
        out[24] = _safe_mean(np.abs(valid["open"] - valid["close"]) / valid["close"].replace(0, np.nan))

        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

