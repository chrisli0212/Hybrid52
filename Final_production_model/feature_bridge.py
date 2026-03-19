"""
Feature Bridge — extracts live features from theta CSV data for Stage-1 inference.

Responsibilities:
  - Maintain per-symbol rolling sequence history (deque, maxlen=SEQ_LEN)
  - Produce (1, T, 325) sequence tensors and (1, 5, S, T) chain-2D tensors
  - Extract 10-dim VIX feature vector from VIXW options data
  - Report warmup fraction and VIX spot level

Exports:
    FeatureBridge
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from hybrid51_preprocessing.master_extractor_v2 import MasterFeatureExtractor, ExtractionResult

ALL_SYMBOLS   = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
SEQ_LEN       = 20
STRIKE_BINS   = 20
FEAT_DIM      = 325


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _col_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns or df[col].dropna().empty:
        return default
    return float(df[col].astype(float).mean())


def _col_std(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns or len(df[col].dropna()) < 2:
        return default
    return float(df[col].astype(float).std())


def _col_sum(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns or df[col].dropna().empty:
        return default
    return float(df[col].astype(float).sum())


def _col_pct(df: pd.DataFrame, col: str, pct: float, default: float = 0.0) -> float:
    if col not in df.columns or df[col].dropna().empty:
        return default
    return float(np.nanpercentile(df[col].astype(float).values, pct))


# ---------------------------------------------------------------------------
# FeatureBridge
# ---------------------------------------------------------------------------

class FeatureBridge:
    """
    Feature bridge using the original Hybrid51 master extractors for the
    correct 325-dim feature layout matching training.

    Call sequence each tick:
        quality = bridge.update_history(snap_df)   # update rolling histories
        vix_f   = bridge.build_vix_features(...)   # build 10-dim VIX vector
        seq, ch = bridge.get_stage1_tensors(sym)   # fetch tensors for Stage-1
    """

    def __init__(self, seq_len: int = SEQ_LEN, strike_bins: int = STRIKE_BINS):
        self.seq_len     = seq_len
        self.strike_bins = strike_bins

        self.extractor = MasterFeatureExtractor(
            include_chain_2d=False,
            include_phase1=True,
            normalize=False,
        )

        self._seq_history:   Dict[str, deque] = {s: deque(maxlen=seq_len) for s in ALL_SYMBOLS}
        self._chain_history: Dict[str, deque] = {s: deque(maxlen=seq_len) for s in ALL_SYMBOLS}

        self._vix_spot_history: deque = deque(maxlen=512)
        self._vix_meta_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=512))

        self._last_batch_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------

    def _adapt_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        rename_map = {"bid_quote": "bid", "ask_quote": "ask"}
        for old, new in rename_map.items():
            if old in out.columns and new not in out.columns:
                out.rename(columns={old: new}, inplace=True)
        if "right" in out.columns:
            out["right"] = out["right"].astype(str).str.upper().replace({"CALL": "C", "PUT": "P"})
        num_cols = [
            "strike", "delta", "gamma", "theta", "vega", "implied_vol",
            "bid", "ask", "volume", "oi", "moneyness", "mid", "spread",
            "spread_pct", "gamma_exp", "vega_exp", "theta_exp", "delta_exp",
            "vanna", "charm", "dist_atm_pct", "atm_strike", "count",
        ]
        for c in num_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def _split_calls_puts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if "right" not in df.columns:
            return df, pd.DataFrame()
        return df[df["right"] == "C"], df[df["right"] == "P"]

    def _delta_buckets(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if "delta" not in df.columns or df.empty:
            e = pd.DataFrame()
            return {"deep_otm": e, "otm": e, "atm": e, "itm": e, "deep_itm": e}
        d = df["delta"].abs()
        return {
            "deep_otm": df[d < 0.1],
            "otm":      df[(d >= 0.1) & (d < 0.3)],
            "atm":      df[(d >= 0.3) & (d < 0.7)],
            "itm":      df[(d >= 0.7) & (d < 0.9)],
            "deep_itm": df[d >= 0.9],
        }

    def _dte_buckets(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if "dte" not in df.columns or df.empty:
            e = pd.DataFrame()
            return {"0d": e, "1d": e, "2_5d": e, "6_14d": e, "15d_plus": e}
        dte = pd.to_numeric(df["dte"], errors="coerce").fillna(0)
        return {
            "0d":       df[dte < 1],
            "1d":       df[(dte >= 1) & (dte < 2)],
            "2_5d":     df[(dte >= 2) & (dte < 6)],
            "6_14d":    df[(dte >= 6) & (dte < 15)],
            "15d_plus": df[dte >= 15],
        }

    def _build_trade_df(self, df: pd.DataFrame) -> pd.DataFrame:
        t = df.copy()
        if "timestamp_quote" in t.columns and "quote_timestamp" not in t.columns:
            t["quote_timestamp"] = t["timestamp_quote"]
        if "timestamp_trade" in t.columns and "trade_timestamp" not in t.columns:
            t["trade_timestamp"] = t["timestamp_trade"]
        if "spot" in t.columns and "underlying_price" not in t.columns:
            t["underlying_price"] = pd.to_numeric(t["spot"], errors="coerce")
        if "size" not in t.columns:
            if "count" in t.columns:
                t["size"] = pd.to_numeric(t["count"], errors="coerce").fillna(0)
            elif "volume" in t.columns:
                t["size"] = pd.to_numeric(t["volume"], errors="coerce").fillna(0)
        # Parse timestamp columns to datetime so smart_money detector can call .diff().dt.total_seconds()
        for ts_col in ("trade_timestamp", "quote_timestamp"):
            if ts_col in t.columns and not pd.api.types.is_datetime64_any_dtype(t[ts_col]):
                t[ts_col] = pd.to_datetime(t[ts_col], errors="coerce")
        return t

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_325_features(self, snap_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Extract a 325-dim feature vector from one symbol's snapshot rows.
        Returns (feature_vector, quality_score ∈ [0, 1]).
        """
        if snap_df is None or snap_df.empty:
            return np.zeros(FEAT_DIM, dtype=np.float32), 0.0

        df = self._adapt_columns(snap_df)
        if df.empty:
            return np.zeros(FEAT_DIM, dtype=np.float32), 0.0

        trade_df = self._build_trade_df(df)

        open_interest: Optional[float] = None
        if "oi" in df.columns:
            oi_val = pd.to_numeric(df["oi"], errors="coerce").sum()
            if np.isfinite(oi_val) and oi_val > 0:
                open_interest = float(oi_val)

        try:
            result: ExtractionResult = self.extractor.extract(
                greek_df=df,
                trade_df=trade_df,
                historical_snapshots=None,
                open_interest=open_interest,
            )
            n_base = int(np.count_nonzero(result.features[:270]))
            n_p1   = int(np.count_nonzero(result.features[270:]))
            completeness = (n_base + n_p1) / FEAT_DIM
            return result.features, min(1.0, completeness)
        except Exception:
            return np.zeros(FEAT_DIM, dtype=np.float32), 0.0

    def extract_chain_slice(self, snap_df: pd.DataFrame) -> np.ndarray:
        """
        Extract chain-2D slice: shape (5, strike_bins).
        Channels: [delta, gamma, theta, vega, implied_vol].
        """
        result = np.zeros((5, self.strike_bins), dtype=np.float32)
        if snap_df is None or snap_df.empty:
            return result

        df = self._adapt_columns(snap_df)
        channels = ["delta", "gamma", "theta", "vega", "implied_vol"]
        if any(c not in df.columns for c in channels) or "atm_strike" not in df.columns:
            return result

        atm = _col_mean(df, "atm_strike")
        if atm <= 0:
            return result

        df = df.copy()
        # Filter to calls only — consistent with training data convention for 2D chain input
        snap_calls = df[df["cp_sign"] == 1].copy() if "cp_sign" in df.columns else df.copy()
        if snap_calls.empty:
            snap_calls = df.copy()
        snap_calls["_dist"] = (snap_calls["strike"].astype(float) - atm).abs()
        df = snap_calls.sort_values("_dist").head(self.strike_bins)

        for ch_i, ch_name in enumerate(channels):
            vals = df[ch_name].fillna(0).astype(float).values[: self.strike_bins]
            result[ch_i, : len(vals)] = vals

        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def update_history(self, snap_df: pd.DataFrame) -> Dict[str, float]:
        """
        Process one snapshot tick: extract features and push to rolling histories.
        Returns per-symbol quality scores.
        """
        if snap_df is None or snap_df.empty:
            return {}

        snap_by_sym: Dict[str, pd.DataFrame] = {}
        if "symbol" in snap_df.columns:
            for sym, sdf in snap_df.groupby("symbol"):
                snap_by_sym[str(sym)] = sdf

        quality_scores: Dict[str, float] = {}
        for symbol in ALL_SYMBOLS:
            sdf = snap_by_sym.get(symbol, pd.DataFrame())
            vec, q = self.extract_325_features(sdf)
            quality_scores[symbol] = q
            self._seq_history[symbol].append(vec)
            self._chain_history[symbol].append(self.extract_chain_slice(sdf))

        return quality_scores

    def get_stage1_tensors(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ready-to-use tensors for Stage-1 inference.

        sequence tensor : (1, seq_len, 325)
        chain tensor    : (1, 5, strike_bins, seq_len)

        Short histories are forward-filled with the oldest available frame.
        """
        hist = list(self._seq_history[symbol])
        if not hist:
            hist = [np.zeros(FEAT_DIM, dtype=np.float32)]
        while len(hist) < self.seq_len:
            hist.insert(0, hist[0].copy())
        seq_tensor = np.stack(hist[-self.seq_len :], axis=0).astype(np.float32)[np.newaxis]

        chain_hist = list(self._chain_history[symbol])
        if not chain_hist:
            chain_hist = [np.zeros((5, self.strike_bins), dtype=np.float32)]
        while len(chain_hist) < self.seq_len:
            chain_hist.insert(0, chain_hist[0].copy())
        chain = np.stack(chain_hist[-self.seq_len :], axis=0).transpose(1, 2, 0).astype(np.float32)
        chain_tensor = chain[np.newaxis]

        return seq_tensor, chain_tensor

    # ------------------------------------------------------------------
    # VIX features
    # ------------------------------------------------------------------

    def build_vix_features(
        self,
        agg_df: pd.DataFrame,
        snap_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Build 10-dim VIX feature vector. Returns shape (1, 10).

        Primary source : VIXW rows in snap_df (per-contract greeks + spot).
        Fallback        : VIXW row in agg_df (aggregated iv_skew / pc_ratio).
        """
        vix_row: Dict[str, Any] = {}

        if snap_df is not None and not snap_df.empty and "symbol" in snap_df.columns:
            vixw = snap_df[snap_df["symbol"] == "VIXW"]
            if not vixw.empty:
                if "spot" in vixw.columns:
                    sv = _safe_float(vixw["spot"].dropna().iloc[-1] if not vixw["spot"].dropna().empty else 0.0)
                    if sv > 0:
                        vix_row["spot"] = sv
                if "implied_vol" in vixw.columns and "right" in vixw.columns:
                    vixw = vixw.copy()
                    vixw["right"] = vixw["right"].astype(str).str.upper().replace({"CALL": "C", "PUT": "P"})
                    c_iv = vixw[vixw["right"] == "C"]["implied_vol"].astype(float)
                    p_iv = vixw[vixw["right"] == "P"]["implied_vol"].astype(float)
                    if not c_iv.empty:
                        vix_row["call_iv"] = float(c_iv.mean())
                    if not p_iv.empty:
                        vix_row["put_iv"] = float(p_iv.mean())
                    if "call_iv" in vix_row and "put_iv" in vix_row:
                        vix_row["iv_skew"] = vix_row["put_iv"] - vix_row["call_iv"]
                if "volume" in vixw.columns and "right" in vixw.columns:
                    vixw2 = vixw.copy()
                    vixw2["right"] = vixw2["right"].astype(str).str.upper().replace({"CALL": "C", "PUT": "P"})
                    c_vol = float(vixw2[vixw2["right"] == "C"]["volume"].astype(float).sum())
                    p_vol = float(vixw2[vixw2["right"] == "P"]["volume"].astype(float).sum())
                    if c_vol > 0:
                        vix_row["pc_ratio"] = p_vol / c_vol

        if not agg_df.empty and "symbol" in agg_df.columns:
            agg_vix = agg_df[agg_df["symbol"] == "VIXW"]
            if not agg_vix.empty:
                row = agg_vix.iloc[-1].to_dict()
                for k in ("spot", "iv_skew", "call_iv", "put_iv", "pc_ratio"):
                    if k not in vix_row or vix_row.get(k, 0.0) == 0.0:
                        v = _safe_float(row.get(k, 0.0))
                        if v != 0.0:
                            vix_row[k] = v

        vix_spot = _safe_float(vix_row.get("spot", 0.0))
        self._vix_spot_history.append(vix_spot)
        for k in ("iv_skew", "call_iv", "put_iv", "pc_ratio"):
            self._vix_meta_history[k].append(_safe_float(vix_row.get(k, 0.0)))

        s = np.asarray(list(self._vix_spot_history), dtype=np.float32)
        if s.size == 0:
            s = np.zeros(1, dtype=np.float32)

        def pct_change(lb: int) -> float:
            if s.size <= lb:
                return 0.0
            base = float(s[-lb - 1])
            return float((s[-1] - base) / base) if base != 0 else 0.0

        def zscore(w: int) -> float:
            tail = s[-w:] if s.size >= w else s
            std = float(np.std(tail))
            return float((tail[-1] - np.mean(tail)) / std) if std > 1e-8 else 0.0

        def percentile(w: int) -> float:
            tail = s[-w:] if s.size >= w else s
            return float(np.sum(tail <= tail[-1]) / tail.size) if tail.size > 1 else 0.5

        iv_skew  = float(self._vix_meta_history["iv_skew"][-1])  if self._vix_meta_history["iv_skew"]  else 0.0
        call_iv  = float(self._vix_meta_history["call_iv"][-1])  if self._vix_meta_history["call_iv"]  else 0.0
        put_iv   = float(self._vix_meta_history["put_iv"][-1])   if self._vix_meta_history["put_iv"]   else 0.0
        pc_ratio = float(self._vix_meta_history["pc_ratio"][-1]) if self._vix_meta_history["pc_ratio"] else 0.0

        hilo = s[-12:] if s.size >= 12 else s
        vix_hilo_range = float((np.max(hilo) - np.min(hilo)) / max(1e-6, np.mean(hilo)))

        vix_vec = np.array([
            vix_spot,
            pct_change(5),
            pct_change(15),
            pct_change(60),
            zscore(15),
            percentile(60),
            0.5 * iv_skew,   # = iv_skew + 0.5*(call_iv-put_iv) simplified; skew-weighted IV differential
            float(np.std(s[-20:]) if s.size >= 3 else 0.0),
            pc_ratio - 1.0,
            vix_hilo_range,
        ], dtype=np.float32)

        vix_vec = np.nan_to_num(vix_vec, nan=0.0, posinf=0.0, neginf=0.0)
        vix_vec = np.clip(vix_vec, -1e6, 1e6)
        return vix_vec.reshape(1, -1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def warmup_fraction(self) -> float:
        return float(np.mean([len(self._seq_history[s]) / self.seq_len for s in ALL_SYMBOLS]))

    @property
    def vix_level(self) -> float:
        return float(self._vix_spot_history[-1]) if self._vix_spot_history else 0.0
