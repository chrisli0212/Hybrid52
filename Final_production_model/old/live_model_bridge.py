from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from live_training_feature_port import LiveTrainingFeaturePort


REQUIRED_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
DEFAULT_SEQ_LEN = 20
DEFAULT_STRIKE_BINS = 20


@dataclass
class BridgeOutput:
    stage1_sequences: Dict[str, np.ndarray]
    stage1_chain_2d: Dict[str, np.ndarray]
    vix_features: np.ndarray
    diagnostics: Dict[str, Any]


class LiveFeatureBridge:
    """
    Build live tensors from dashboard CSV data.

    The trained model expects offline tensorized features. This bridge creates a
    deterministic online approximation with strict shapes and quality diagnostics.
    """

    def __init__(self, seq_len: int = DEFAULT_SEQ_LEN, strike_bins: int = DEFAULT_STRIKE_BINS):
        self.seq_len = int(seq_len)
        self.strike_bins = int(strike_bins)
        self.feature_port = LiveTrainingFeaturePort()

        self._seq_history: Dict[str, deque[np.ndarray]] = {
            s: deque(maxlen=self.seq_len) for s in REQUIRED_SYMBOLS
        }
        self._chain_history: Dict[str, deque[np.ndarray]] = {
            s: deque(maxlen=self.seq_len) for s in REQUIRED_SYMBOLS
        }
        self._vix_spot_history: deque[float] = deque(maxlen=512)
        self._vix_meta_history: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=512))

        self._last_batch_id: int | None = None
        self._last_output: BridgeOutput | None = None

    def build_from_dataframes(self, agg_df: pd.DataFrame, snap_df: pd.DataFrame) -> BridgeOutput:
        if agg_df is None:
            agg_df = pd.DataFrame()
        if snap_df is None:
            snap_df = pd.DataFrame()

        latest_batch = self._extract_latest_batch_id(agg_df, snap_df)
        if self._last_output is not None and latest_batch is not None and latest_batch == self._last_batch_id:
            return self._last_output

        agg_latest = self._latest_rows_by_symbol(agg_df)
        snap_by_symbol = self._split_snapshot_by_symbol(snap_df)

        stage1_sequences: Dict[str, np.ndarray] = {}
        stage1_chain_2d: Dict[str, np.ndarray] = {}

        quality_scores = []
        symbols_ready = []

        for symbol in REQUIRED_SYMBOLS:
            sdf = snap_by_symbol.get(symbol, pd.DataFrame())
            adapted = self.feature_port.adapt_snapshot_columns(sdf)
            vec325, q_score = self.feature_port.extract_feature_vector(adapted)
            quality_scores.append(q_score)
            self._seq_history[symbol].append(vec325)
            seq_tensor = self._stack_seq(symbol)

            chain_30 = self.feature_port.extract_chain_slice30(adapted)
            chain_slice = self.feature_port.center_crop_30_to_20(chain_30)
            self._chain_history[symbol].append(chain_slice)
            chain_tensor = self._stack_chain(symbol)

            stage1_sequences[symbol] = seq_tensor
            stage1_chain_2d[symbol] = chain_tensor
            symbols_ready.append(symbol)

        vix_features = self._build_vix_features(agg_latest)
        completeness = float(np.mean(quality_scores)) if quality_scores else 0.0
        warmup_fraction = float(
            np.mean(
                [
                    len(self._seq_history[s]) / self.seq_len
                    for s in REQUIRED_SYMBOLS
                ]
            )
        )
        warmup_ready = warmup_fraction >= 1.0
        vix_valid = bool(np.isfinite(vix_features).all())

        diagnostics = {
            "batch_id": latest_batch,
            "symbols_ready": symbols_ready,
            "feature_completeness": completeness,
            "warmup_fraction": warmup_fraction,
            "warmup_ready": warmup_ready,
            "vix_valid": vix_valid,
            "quality_score": float(0.6 * completeness + 0.4 * min(1.0, warmup_fraction)),
            "suppression_reason": self._suppression_reason(completeness, warmup_fraction, vix_valid),
        }

        out = BridgeOutput(
            stage1_sequences=stage1_sequences,
            stage1_chain_2d=stage1_chain_2d,
            vix_features=vix_features,
            diagnostics=diagnostics,
        )
        self._last_output = out
        self._last_batch_id = latest_batch
        return out

    def _suppression_reason(self, completeness: float, warmup_fraction: float, vix_valid: bool) -> str | None:
        if not vix_valid:
            return "vix_features_invalid"
        if completeness < 0.6:
            return "low_feature_completeness"
        if warmup_fraction < 0.35:
            return "insufficient_temporal_warmup"
        return None

    def _extract_latest_batch_id(self, agg_df: pd.DataFrame, snap_df: pd.DataFrame) -> int | None:
        for df in (agg_df, snap_df):
            if not df.empty and "batch_id" in df.columns:
                try:
                    return int(pd.to_numeric(df["batch_id"], errors="coerce").dropna().max())
                except Exception:
                    continue
        return None

    def _latest_rows_by_symbol(self, agg_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        if agg_df.empty or "symbol" not in agg_df.columns:
            return {}
        work = agg_df.copy()
        if "batch_id" in work.columns:
            work["_batch_num"] = pd.to_numeric(work["batch_id"], errors="coerce")
            work = work.sort_values("_batch_num")
        out: Dict[str, Dict[str, Any]] = {}
        for symbol, sdf in work.groupby("symbol"):
            if len(sdf) > 0:
                out[str(symbol)] = sdf.iloc[-1].to_dict()
        return out

    def _split_snapshot_by_symbol(self, snap_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if snap_df.empty or "symbol" not in snap_df.columns:
            return {}
        return {str(s): sdf.copy() for s, sdf in snap_df.groupby("symbol")}

    def _safe_num(self, value: Any, default: float = 0.0) -> Tuple[float, bool]:
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(v):
            return float(default), False
        return float(v), True

    def _stack_seq(self, symbol: str) -> np.ndarray:
        hist = list(self._seq_history[symbol])
        if not hist:
            hist = [np.zeros(325, dtype=np.float32)]
        while len(hist) < self.seq_len:
            hist.insert(0, hist[0].copy())
        seq = np.stack(hist[-self.seq_len :], axis=0).astype(np.float32)
        return seq[None, :, :]

    def _stack_chain(self, symbol: str) -> np.ndarray:
        hist = list(self._chain_history[symbol])
        if not hist:
            hist = [np.zeros((5, self.strike_bins), dtype=np.float32)]
        while len(hist) < self.seq_len:
            hist.insert(0, hist[0].copy())
        # hist: [T, C, S] -> [C, S, T]
        chain = np.stack(hist[-self.seq_len :], axis=0).transpose(1, 2, 0).astype(np.float32)
        return chain[None, :, :, :]

    def _build_vix_features(self, agg_latest: Dict[str, Dict[str, Any]]) -> np.ndarray:
        vix_row = agg_latest.get("VIXW", {})
        vix_spot, _ = self._safe_num(vix_row.get("spot", np.nan), default=0.0)
        self._vix_spot_history.append(vix_spot)
        for k in ("iv_skew", "call_iv", "put_iv", "pc_ratio"):
            v, _ = self._safe_num(vix_row.get(k, np.nan), default=0.0)
            self._vix_meta_history[k].append(v)

        s = np.asarray(self._vix_spot_history, dtype=np.float32)
        if s.size == 0:
            s = np.zeros(1, dtype=np.float32)

        def pct_change(lookback: int) -> float:
            if s.size <= lookback:
                return 0.0
            base = float(s[-lookback - 1])
            if base == 0:
                return 0.0
            return float((s[-1] - base) / base)

        def zscore(window: int) -> float:
            tail = s[-window:] if s.size >= window else s
            std = float(np.std(tail))
            if std < 1e-8:
                return 0.0
            return float((tail[-1] - np.mean(tail)) / std)

        def percentile(window: int) -> float:
            tail = s[-window:] if s.size >= window else s
            if tail.size <= 1:
                return 0.5
            rank = float(np.sum(tail <= tail[-1]) / tail.size)
            return rank

        iv_skew = float(self._vix_meta_history["iv_skew"][-1]) if self._vix_meta_history["iv_skew"] else 0.0
        call_iv = float(self._vix_meta_history["call_iv"][-1]) if self._vix_meta_history["call_iv"] else 0.0
        put_iv = float(self._vix_meta_history["put_iv"][-1]) if self._vix_meta_history["put_iv"] else 0.0
        pc_ratio = float(self._vix_meta_history["pc_ratio"][-1]) if self._vix_meta_history["pc_ratio"] else 0.0

        hilo_window = s[-12:] if s.size >= 12 else s
        vix_hilo_range = float((np.max(hilo_window) - np.min(hilo_window)) / max(1e-6, np.mean(hilo_window)))

        # 10-dim order expected by model config.vix_feature_subsets
        vix_vec = np.array(
            [
                vix_spot,  # vix_level
                pct_change(5),  # vix_pct_5m (proxy by batches)
                pct_change(15),  # vix_pct_15m
                pct_change(60),  # vix_pct_1h
                zscore(15),  # vix_zscore_15m
                percentile(60),  # vix_percentile_1h
                iv_skew + 0.5 * (call_iv - put_iv),  # vix_term_slope proxy
                float(np.std(s[-20:]) if s.size >= 3 else 0.0),  # vvix_level proxy
                (pc_ratio - 1.0),  # vix_vix1d_spread proxy
                vix_hilo_range,  # vix_hilo_range
            ],
            dtype=np.float32,
        )
        vix_vec = np.nan_to_num(vix_vec, nan=0.0, posinf=0.0, neginf=0.0)
        vix_vec = np.clip(vix_vec, -1e6, 1e6)
        return vix_vec.reshape(1, -1)
