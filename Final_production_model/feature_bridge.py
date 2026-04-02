"""
Feature Bridge — extracts live features from theta CSV data for Stage-1 inference.

Responsibilities:
  - Maintain per-symbol rolling sequence history (deque, maxlen=SEQ_LEN)
  - Produce (1, T, 325) sequence tensors and (1, 5, S, T) chain-2D tensors
  - Extract 10-dim VIX feature vector from VIXW options data
  - Report warmup fraction and VIX spot level

Training vs production cadence (Hybrid51 tier3):
  Tier-2/3 training builds sequences from **1-minute** bars: each LSTM step is one
  minute, so SEQ_LEN=20 spans **20 minutes** of wall time. Live fetch/poll may run
  every 10s; without gating, each tick appends a step and the same 20 slots cover
  only ~200s. Prefer ``history_align_to_exchange_minute`` + ``history_minute_aggregate``
  to build one bar per NY minute (mean or last of sub-minute fetches), matching
  tier-2 minute training. Alternatively set ``history_min_interval_seconds`` (e.g. 60)
  for a rolling wall-clock gate while the fetcher still writes at higher frequency.

VIX 10-vector (``build_vix_features``):
  Lookbacks like ``pct_change(60)`` / ``percentile(60)`` count **deque samples**,
  not calendar minutes. At one sample per minute they match ~60 minutes; at one
  sample per 10s they match ~10 minutes. Use ``history_min_interval_seconds=60`` to
  align VIX history steps with training-style minute spacing.

Exports:
    FeatureBridge
"""

from __future__ import annotations

import time
from datetime import datetime
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from hybrid51_preprocessing.master_extractor_v2 import MasterFeatureExtractor, ExtractionResult

ALL_SYMBOLS   = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
SEQ_LEN       = 20
STRIKE_BINS   = 20
FEAT_DIM      = 325
EXCHANGE_TZ   = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _ny_minute_key() -> Tuple[int, int, int, int, int]:
    t = datetime.now(EXCHANGE_TZ)
    return (t.year, t.month, t.day, t.hour, t.minute)


def _contract_keys_greek(df: pd.DataFrame) -> List[str]:
    return [k for k in ("symbol", "expiration", "strike", "right") if k in df.columns]


def _contract_keys_tq(df: pd.DataFrame) -> List[str]:
    return [k for k in ("symbol", "expiration", "strike", "right", "endpoint") if k in df.columns]


def aggregate_minute_snapshots(
    dfs: List[pd.DataFrame],
    mode: str,
    kind: str,
) -> pd.DataFrame:
    """
    Combine multiple 10s (or sub-minute) snapshots into one training-style bar.

    ``mode``:
      - ``last``: last row per contract key (close-of-minute style on irregular keys).
      - ``mean``: groupby contract key, numeric columns mean, other columns last.
    """
    clean = [d.copy() for d in dfs if d is not None and not d.empty]
    if not clean:
        return pd.DataFrame()
    if len(clean) == 1:
        return clean[0]

    mode = (mode or "mean").strip().lower()
    cat = pd.concat(clean, ignore_index=True)
    keys = _contract_keys_greek(cat) if kind == "greek" else _contract_keys_tq(cat)
    if not keys:
        return clean[-1]

    if mode == "last":
        return cat.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    num_cols = cat.select_dtypes(include=[np.number]).columns.tolist()
    skip_mean = set(keys) | {"batch_id"}
    mean_cols = [c for c in num_cols if c not in skip_mean]
    last_cols = [c for c in cat.columns if c not in keys and c not in mean_cols]
    agg_map: Dict[str, str] = {c: "mean" for c in mean_cols}
    for c in last_cols:
        agg_map[c] = "last"
    if not agg_map:
        return clean[-1]
    try:
        out = cat.groupby(keys, dropna=False, as_index=False).agg(agg_map)
        return out
    except Exception:
        return clean[-1]


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

        Call sequence each tick (when history advanced):
        quality, advanced = bridge.update_history(greek_df, trade_quote_df=...)
        vix_f   = bridge.build_vix_features(...)   # build 10-dim VIX vector
        seq, ch = bridge.get_stage1_tensors(sym)   # fetch tensors for Stage-1
    """

    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        strike_bins: int = STRIKE_BINS,
        history_min_interval_seconds: float = 0.0,
        history_align_to_exchange_minute: bool = False,
        history_minute_aggregate: str = "mean",
    ):
        self.seq_len     = seq_len
        self.strike_bins = strike_bins
        # 0 = append on every update_history (legacy); 60 ≈ tier3 1-minute step spacing.
        self.history_min_interval_seconds = max(0.0, float(history_min_interval_seconds))
        self._last_hist_push_monotonic: Optional[float] = None

        self.history_align_to_exchange_minute = bool(history_align_to_exchange_minute)
        aggr = str(history_minute_aggregate or "mean").strip().lower()
        self.history_minute_aggregate = aggr if aggr in ("last", "mean") else "mean"
        self._minute_bucket_key: Optional[Tuple[int, int, int, int, int]] = None
        self._minute_greek_snaps: List[pd.DataFrame] = []
        self._minute_tq_snaps: List[pd.DataFrame] = []

        self.extractor = MasterFeatureExtractor(
            include_chain_2d=False,
            include_phase1=True,
            normalize=False,
        )

        self._seq_history:   Dict[str, deque] = {s: deque(maxlen=seq_len) for s in ALL_SYMBOLS}
        self._chain_history: Dict[str, deque] = {s: deque(maxlen=seq_len) for s in ALL_SYMBOLS}

        self._vix_spot_history: deque = deque(maxlen=512)
        self._vix_meta_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=512))
        self._last_quality_scores: Dict[str, float] = {}
        self._last_nonzero_density_scores: Dict[str, float] = {}

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

    def extract_325_features(
        self,
        greek_df: pd.DataFrame,
        trade_quote_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Extract a 325-dim feature vector from one symbol's snapshot rows.
        Returns (feature_vector, quality_score ∈ [0, 1]).
        """
        if greek_df is None or greek_df.empty:
            return np.zeros(FEAT_DIM, dtype=np.float32), 0.0

        df_greek = self._adapt_columns(greek_df)
        if df_greek.empty:
            return np.zeros(FEAT_DIM, dtype=np.float32), 0.0

        # Never substitute greek rows for trade/quote — training used separate streams; phase-1 stays zero without real T/Q.
        if trade_quote_df is not None and not trade_quote_df.empty:
            df_trade_src = self._adapt_columns(trade_quote_df)
            trade_df = self._build_trade_df(df_trade_src) if not df_trade_src.empty else None
        else:
            trade_df = None

        open_interest: Optional[float] = None
        if "oi" in df_greek.columns:
            oi_val = pd.to_numeric(df_greek["oi"], errors="coerce").sum()
            if np.isfinite(oi_val) and oi_val > 0:
                open_interest = float(oi_val)

        try:
            result: ExtractionResult = self.extractor.extract(
                greek_df=df_greek,
                trade_df=trade_df,
                historical_snapshots=None,
                open_interest=open_interest,
            )
            # Use extractor-native quality score (NaN/inf hygiene) as completeness.
            # Non-zero density is tracked separately as a diagnostic.
            quality = float(np.clip(result.quality_score, 0.0, 1.0))
            return result.features, quality
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
        if any(c not in df.columns for c in channels):
            return result

        if "atm_strike" not in df.columns:
            # Fallback for model-greeks stream that may not carry atm_strike:
            # infer nearest strike from current spot/underlying_price.
            spot_col = "underlying_price" if "underlying_price" in df.columns else ("spot" if "spot" in df.columns else None)
            if spot_col is None or "strike" not in df.columns:
                return result
            spot_vals = pd.to_numeric(df[spot_col], errors="coerce").dropna()
            strike_vals = pd.to_numeric(df["strike"], errors="coerce").dropna()
            if spot_vals.empty or strike_vals.empty:
                return result
            spot_now = float(spot_vals.iloc[-1])
            nearest = float(strike_vals.iloc[(strike_vals - spot_now).abs().argsort().iloc[0]])
            df = df.copy()
            df["atm_strike"] = nearest

        atm = _col_mean(df, "atm_strike")
        if atm <= 0:
            return result

        df = df.copy()
        # Filter to calls only — consistent with training data convention for 2D chain input
        if "cp_sign" in df.columns:
            snap_calls = df[df["cp_sign"] == 1].copy()
        elif "right" in df.columns:
            snap_calls = df[df["right"] == "C"].copy()
        else:
            snap_calls = df.copy()
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

    def _flush_history_step(
        self,
        greek_df: pd.DataFrame,
        trade_quote_df: Optional[pd.DataFrame],
    ) -> Tuple[Dict[str, float], bool]:
        """Append one aggregated timestep to seq/chain deques (internal)."""
        now = time.monotonic()
        greek_by_sym: Dict[str, pd.DataFrame] = {}
        if "symbol" in greek_df.columns:
            for sym, sdf in greek_df.groupby("symbol"):
                greek_by_sym[str(sym)] = sdf

        trade_by_sym: Dict[str, pd.DataFrame] = {}
        if trade_quote_df is not None and not trade_quote_df.empty and "symbol" in trade_quote_df.columns:
            for sym, sdf in trade_quote_df.groupby("symbol"):
                trade_by_sym[str(sym)] = sdf

        quality_scores: Dict[str, float] = {}
        nonzero_density_scores: Dict[str, float] = {}
        for symbol in ALL_SYMBOLS:
            sdf_greek = greek_by_sym.get(symbol, pd.DataFrame())
            sdf_trade = trade_by_sym.get(symbol, pd.DataFrame())
            vec, q = self.extract_325_features(sdf_greek, sdf_trade)
            quality_scores[symbol] = q
            nonzero_density_scores[symbol] = float(np.count_nonzero(vec) / FEAT_DIM)
            self._seq_history[symbol].append(vec)
            self._chain_history[symbol].append(self.extract_chain_slice(sdf_greek))

        self._last_quality_scores = quality_scores
        self._last_nonzero_density_scores = nonzero_density_scores
        self._last_hist_push_monotonic = now
        return quality_scores, True

    def _update_history_exchange_minute(
        self,
        greek_df: pd.DataFrame,
        trade_quote_df: Optional[pd.DataFrame],
    ) -> Tuple[Dict[str, float], bool]:
        """
        Buffer sub-minute fetches; on NY minute rollover, aggregate and flush one bar.
        """
        mk = _ny_minute_key()
        tq = trade_quote_df if trade_quote_df is not None else pd.DataFrame()
        tq_snap = tq.copy() if not tq.empty else pd.DataFrame()

        if self._minute_bucket_key is not None and mk < self._minute_bucket_key:
            self._minute_bucket_key = None
            self._minute_greek_snaps = []
            self._minute_tq_snaps = []

        if self._minute_bucket_key is None:
            self._minute_bucket_key = mk
            self._minute_greek_snaps = [greek_df.copy()]
            self._minute_tq_snaps = [tq_snap]
            return dict(self._last_quality_scores), False

        if mk == self._minute_bucket_key:
            self._minute_greek_snaps.append(greek_df.copy())
            self._minute_tq_snaps.append(tq_snap)
            return dict(self._last_quality_scores), False

        g_agg = aggregate_minute_snapshots(
            self._minute_greek_snaps, self.history_minute_aggregate, "greek"
        )
        t_nonempty = [x for x in self._minute_tq_snaps if not x.empty]
        t_agg = (
            aggregate_minute_snapshots(t_nonempty, self.history_minute_aggregate, "tq")
            if t_nonempty
            else pd.DataFrame()
        )

        self._minute_bucket_key = mk
        self._minute_greek_snaps = [greek_df.copy()]
        self._minute_tq_snaps = [tq_snap]

        if g_agg.empty:
            return dict(self._last_quality_scores), False

        t_out: Optional[pd.DataFrame] = t_agg if not t_agg.empty else None
        return self._flush_history_step(g_agg, t_out)

    def update_history(
        self,
        greek_df: pd.DataFrame,
        trade_quote_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, float], bool]:
        """
        Process one snapshot tick: extract features and push to rolling histories.

        Returns (quality_scores, advanced). If ``history_align_to_exchange_minute``,
        buffers snapshots until the NY minute rolls over, then aggregates (mean/last)
        into one step. If only ``history_min_interval_seconds`` > 0 (no exchange
        align), returns (last scores, False) when called inside that wall interval.
        """
        if greek_df is None or greek_df.empty:
            return {}, False

        if self.history_align_to_exchange_minute:
            return self._update_history_exchange_minute(greek_df, trade_quote_df)

        now = time.monotonic()
        if self.history_min_interval_seconds > 0 and self._last_hist_push_monotonic is not None:
            if (now - self._last_hist_push_monotonic) < self.history_min_interval_seconds:
                return dict(self._last_quality_scores), False

        return self._flush_history_step(greek_df, trade_quote_df)

    def get_stage1_tensors(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ready-to-use tensors for Stage-1 inference.

        sequence tensor : (1, seq_len, 325)
        chain tensor    : (1, 5, strike_bins, seq_len)

        Partial history (e.g. 3 real minutes of updates, then 17 missing): the
        **oldest** available frame is repeated at the **past** end of the window
        until length ``seq_len`` is reached; the **newest** timesteps are the
        actual observations. So the LSTM always receives a full length-20 tensor
        while using all data collected so far, up to 20 steps of real history.
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

        Appends one sample to internal VIX deques per call. Indices such as 60 in
        ``pct_change(60)`` / ``percentile(60)`` refer to **samples in the deque**;
        wall-clock span equals 60 × (time between calls). Pair with
        ``history_min_interval_seconds=60`` so VIX steps align with tier3 training.
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

    @property
    def nonzero_density(self) -> float:
        if not self._last_nonzero_density_scores:
            return 0.0
        return float(np.mean(list(self._last_nonzero_density_scores.values())))

    @property
    def nonzero_density_by_symbol(self) -> Dict[str, float]:
        return dict(self._last_nonzero_density_scores)
