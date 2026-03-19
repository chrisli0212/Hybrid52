#!/usr/bin/env python3
"""
Prediction Service — Decoupled Hybrid51 h30 VIX-Gated Live Inference

Reads theta CSV data, runs full 3-stage inference, writes predictions to CSV.
Runs on a loop every 30 seconds alongside the theta fetcher.

Usage:
    python prediction_service.py                         # Defaults
    python prediction_service.py --data-dir ./daily_data  # Custom data dir
    python prediction_service.py --interval 30            # Custom polling interval
    python prediction_service.py --device cpu              # Force CPU
    python prediction_service.py --once                    # Single prediction, don't loop
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths — all relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "models"
CONFIG_PATH = SCRIPT_DIR / "config" / "production_config.json"
DEFAULT_DATA_DIR = SCRIPT_DIR / "daily_data"

# Add parent to path for hybrid51_models imports
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hybrid51_models.independent_agent import IndependentAgent  # noqa: E402
from hybrid51_models.cross_symbol_agent_fusion import CrossSymbolAgentFusion  # noqa: E402
from hybrid51_models.regime_gated_meta_model import RegimeGatedProbFusion  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prediction_service")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALL_SYMBOLS = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
ALL_AGENTS = ["A", "B", "C", "K", "T", "Q", "2D"]
STANDARD_PEER_SYMBOLS = ["SPY", "QQQ", "IWM"]
AGENT_2D_PEER_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT"]
SEQ_LEN = 20
STRIKE_BINS = 20
FEAT_DIM = 325

# Prediction CSV header
PRED_CSV_COLUMNS = [
    "batch_id", "ts",
    "prob", "pred", "threshold", "confidence", "signal_strength", "direction",
    "agent_A_prob", "agent_B_prob", "agent_C_prob", "agent_K_prob",
    "agent_T_prob", "agent_Q_prob", "agent_2D_prob",
    "gate_A", "gate_B", "gate_C", "gate_K", "gate_T", "gate_Q", "gate_2D",
    "quality_score", "feature_completeness", "warmup_fraction", "latency_ms",
    "stage1_missing_count", "suppressed", "reason",
    "vix_level", "spot_price",
    # New: confidence decomposition (evidence-based)
    "agent_std", "consensus_ratio",
    "conf_agreement", "conf_consensus", "conf_gate_conviction", "conf_data_quality",
]


# ═══════════════════════════════════════════════════════════════════════════
# BinaryIndependentAgent wrapper — matches training checkpoint format
# ═══════════════════════════════════════════════════════════════════════════
class BinaryIndependentAgent(nn.Module):
    """Wraps IndependentAgent as self.base → state dict keys have 'base.' prefix."""

    def __init__(
        self,
        agent_type: str,
        feat_dim: int = 325,
        temporal_dim: int = 128,
        dropout: float = 0.2,
        use_feature_subset: bool = True,
        use_attention_backbone: bool = False,
        use_attention_pool: bool = False,
        cls_input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.base = IndependentAgent(
            agent_type=agent_type,
            feat_dim=feat_dim,
            temporal_dim=temporal_dim,
            dropout=dropout,
            num_classes=5,
            use_feature_subset=use_feature_subset,
            use_attention_backbone=use_attention_backbone,
            use_attention_pool=use_attention_pool,
        )
        if cls_input_dim is None:
            cls_input_dim = (2 + temporal_dim) if self.base.use_backbone else (2 + 32)
        self.base.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, sequences: torch.Tensor, chain_2d: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.base(sequences, chain_2d=chain_2d).squeeze(-1)


def _build_model_from_ckpt(ckpt: dict, agent_type: str, device: torch.device, symbol: str = "SPXW") -> nn.Module:
    """Reconstruct BinaryIndependentAgent matching saved checkpoint exactly."""
    state = ckpt["model_state_dict"]
    if "feat_dim" in ckpt:
        feat_dim = int(ckpt["feat_dim"])
    elif "base._feat_idx" in state and state["base._feat_idx"].numel() > 0:
        feat_dim = 650 if int(state["base._feat_idx"].max().item()) >= 325 else 325
    else:
        feat_dim = 325 if symbol == "SPXW" else 650

    use_subset = bool(ckpt.get("feature_subset", True))
    use_attn_bb = bool(ckpt.get("use_attention_backbone", False))
    use_attn_pool = bool(ckpt.get("use_attention_pool", False))
    cls_in_dim = int(state["base.classifier.0.weight"].shape[1])
    has_static_proj = "base.static_proj.weight" in state

    model = BinaryIndependentAgent(
        agent_type=agent_type,
        feat_dim=feat_dim,
        use_feature_subset=use_subset,
        use_attention_backbone=use_attn_bb,
        use_attention_pool=use_attn_pool,
        cls_input_dim=cls_in_dim,
    ).to(device)

    if not has_static_proj and hasattr(model.base, "static_proj"):
        del model.base.static_proj

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Self-Contained Feature Bridge
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert any value to float safely."""
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


class FeatureBridge:
    """
    Self-contained feature bridge that builds 325-dim vectors from theta_snapshot.csv.
    No dependency on hybrid51_preprocessing or master_extractor_v2.
    """

    def __init__(self, seq_len: int = SEQ_LEN, strike_bins: int = STRIKE_BINS):
        self.seq_len = seq_len
        self.strike_bins = strike_bins

        # Rolling history per symbol
        self._seq_history: Dict[str, deque] = {s: deque(maxlen=seq_len) for s in ALL_SYMBOLS}
        self._chain_history: Dict[str, deque] = {s: deque(maxlen=seq_len) for s in ALL_SYMBOLS}

        # VIX rolling history
        self._vix_spot_history: deque = deque(maxlen=512)
        self._vix_meta_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=512))

        self._last_batch_id: Optional[int] = None

    def _adapt_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to match expected format."""
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        rename_map = {"bid_quote": "bid", "ask_quote": "ask"}
        for old, new in rename_map.items():
            if old in out.columns and new not in out.columns:
                out.rename(columns={old: new}, inplace=True)
        if "right" in out.columns:
            out["right"] = out["right"].astype(str).str.upper().replace({"CALL": "C", "PUT": "P"})
        # Ensure numeric
        num_cols = ["strike", "delta", "gamma", "theta", "vega", "implied_vol",
                    "bid", "ask", "volume", "oi", "moneyness", "mid", "spread",
                    "spread_pct", "gamma_exp", "vega_exp", "theta_exp", "delta_exp",
                    "vanna", "charm", "dist_atm_pct", "atm_strike", "count"]
        for c in num_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def _split_calls_puts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into calls and puts."""
        if "right" not in df.columns:
            return df, pd.DataFrame()
        calls = df[df["right"] == "C"]
        puts = df[df["right"] == "P"]
        return calls, puts

    def _delta_buckets(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split by delta into 5 buckets: deep_otm, otm, atm, itm, deep_itm."""
        if "delta" not in df.columns or df.empty:
            empty = pd.DataFrame()
            return {"deep_otm": empty, "otm": empty, "atm": empty, "itm": empty, "deep_itm": empty}
        d = df["delta"].abs()
        return {
            "deep_otm": df[d < 0.1],
            "otm": df[(d >= 0.1) & (d < 0.3)],
            "atm": df[(d >= 0.3) & (d < 0.7)],
            "itm": df[(d >= 0.7) & (d < 0.9)],
            "deep_itm": df[d >= 0.9],
        }

    def _dte_buckets(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split by DTE into buckets."""
        if "dte" not in df.columns or df.empty:
            empty = pd.DataFrame()
            return {"0d": empty, "1d": empty, "2_5d": empty, "6_14d": empty, "15d_plus": empty}
        dte = pd.to_numeric(df["dte"], errors="coerce").fillna(0)
        return {
            "0d": df[dte < 1],
            "1d": df[(dte >= 1) & (dte < 2)],
            "2_5d": df[(dte >= 2) & (dte < 6)],
            "6_14d": df[(dte >= 6) & (dte < 15)],
            "15d_plus": df[dte >= 15],
        }

    def extract_325_features(self, snap_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Extract 325-dim feature vector from snapshot data for one symbol.
        Returns (feature_vector, quality_score).
        """
        vec = np.zeros(FEAT_DIM, dtype=np.float32)
        if snap_df is None or snap_df.empty:
            return vec, 0.0

        df = self._adapt_columns(snap_df)
        if df.empty:
            return vec, 0.0

        calls, puts = self._split_calls_puts(df)
        n_filled = 0

        # ── dims 0-49: Core Greeks ──
        # aggregate delta, gamma, theta, vega, rho by 5 delta buckets for calls and puts
        idx = 0
        for sub_df, _label in [(calls, "call"), (puts, "put")]:
            buckets = self._delta_buckets(sub_df)
            for _bname, bdf in buckets.items():
                for greek in ["delta", "gamma", "theta", "vega"]:
                    val = _col_mean(bdf, greek)
                    vec[idx] = val
                    if val != 0.0:
                        n_filled += 1
                    idx += 1
                # rho placeholder (not in snapshot, use 0)
                vec[idx] = 0.0
                idx += 1
        # idx should be 50 (2 sides × 5 buckets × 5 greeks)

        # ── dims 50-99: IV Surface ──
        idx = 50
        atm_iv = _col_mean(df[df["delta"].abs().between(0.3, 0.7)] if "delta" in df.columns else df, "implied_vol")
        vec[idx] = atm_iv; idx += 1; n_filled += (1 if atm_iv != 0 else 0)

        # IV skew: OTM put IV - OTM call IV
        otm_call_iv = _col_mean(calls[calls["delta"].abs().between(0.1, 0.3)] if "delta" in calls.columns and not calls.empty else calls, "implied_vol")
        otm_put_iv = _col_mean(puts[puts["delta"].abs().between(0.1, 0.3)] if "delta" in puts.columns and not puts.empty else puts, "implied_vol")
        vec[idx] = otm_put_iv - otm_call_iv; idx += 1

        # Moneyness-level IVs (10 quantile bins)
        if "moneyness" in df.columns and "implied_vol" in df.columns:
            m = df[["moneyness", "implied_vol"]].dropna()
            if len(m) >= 5:
                for q in np.linspace(0, 100, 10):
                    pct_m = np.nanpercentile(m["moneyness"].values, q)
                    near = m[(m["moneyness"] - pct_m).abs() < 0.05]
                    vec[idx] = _col_mean(near, "implied_vol")
                    idx += 1
                    n_filled += 1
            else:
                idx += 10
        else:
            idx += 10

        # IV stats
        vec[idx] = _col_std(df, "implied_vol"); idx += 1
        vec[idx] = _col_pct(df, "implied_vol", 25); idx += 1
        vec[idx] = _col_pct(df, "implied_vol", 75); idx += 1
        vec[idx] = _col_pct(df, "implied_vol", 90); idx += 1

        # Term structure by DTE
        dte_bkts = self._dte_buckets(df)
        for _dname, ddf in dte_bkts.items():
            vec[idx] = _col_mean(ddf, "implied_vol")
            idx += 1
        # Fill rest of dim 50-99
        while idx < 100:
            vec[idx] = 0.0
            idx += 1

        # ── dims 100-127: Term Structure ──
        idx = 100
        for _dname, ddf in dte_bkts.items():
            vec[idx] = _col_mean(ddf, "implied_vol"); idx += 1
            vec[idx] = _col_mean(ddf, "theta"); idx += 1
            vec[idx] = _col_std(ddf, "implied_vol"); idx += 1
        # Decay profiles
        if "theta" in df.columns and "dte" in df.columns:
            m = df[["dte", "theta"]].dropna()
            if len(m) >= 3:
                for q in [10, 25, 50, 75, 90]:
                    dte_q = np.nanpercentile(m["dte"].values, q)
                    near = m[(m["dte"] - dte_q).abs() < 1]
                    vec[idx] = _col_mean(near, "theta")
                    idx += 1
            else:
                idx += 5
        else:
            idx += 5
        while idx < 128:
            vec[idx] = 0.0; idx += 1

        # ── dims 128-149: Flow & Volume ──
        idx = 128
        vec[idx] = _col_sum(calls, "volume"); idx += 1; n_filled += 1
        vec[idx] = _col_sum(puts, "volume"); idx += 1
        call_vol = _col_sum(calls, "volume")
        put_vol = _col_sum(puts, "volume")
        total_vol = call_vol + put_vol
        vec[idx] = (put_vol / total_vol) if total_vol > 0 else 0.5; idx += 1
        # Premium = volume × mid
        if "mid" in calls.columns and "volume" in calls.columns:
            call_prem = float((calls["mid"].fillna(0) * calls["volume"].fillna(0)).sum())
        else:
            call_prem = 0.0
        if "mid" in puts.columns and "volume" in puts.columns:
            put_prem = float((puts["mid"].fillna(0) * puts["volume"].fillna(0)).sum())
        else:
            put_prem = 0.0
        vec[idx] = call_prem; idx += 1
        vec[idx] = put_prem; idx += 1
        total_prem = call_prem + put_prem
        vec[idx] = (put_prem / total_prem) if total_prem > 0 else 0.5; idx += 1

        # Volume ratios by delta bucket
        for _bname, bdf in self._delta_buckets(df).items():
            bc = bdf[bdf["right"] == "C"] if "right" in bdf.columns else bdf
            bp = bdf[bdf["right"] == "P"] if "right" in bdf.columns else pd.DataFrame()
            cv = _col_sum(bc, "volume")
            pv = _col_sum(bp, "volume")
            vec[idx] = (pv / (cv + pv)) if (cv + pv) > 0 else 0.5; idx += 1

        while idx < 150:
            vec[idx] = 0.0; idx += 1

        # ── dims 150-179: Microstructure ──
        idx = 150
        vec[idx] = _col_mean(df, "spread"); idx += 1; n_filled += 1
        vec[idx] = _col_std(df, "spread"); idx += 1
        vec[idx] = _col_mean(df, "spread_pct"); idx += 1
        vec[idx] = _col_pct(df, "spread", 50); idx += 1
        vec[idx] = _col_pct(df, "spread", 90); idx += 1

        # Bid-ask imbalance
        if "bid" in df.columns and "ask" in df.columns:
            bid_vals = df["bid"].fillna(0).astype(float)
            ask_vals = df["ask"].fillna(0).astype(float)
            total = bid_vals + ask_vals
            imb = np.where(total > 0, (bid_vals - ask_vals) / total, 0.0)
            vec[idx] = float(np.nanmean(imb)); idx += 1
            vec[idx] = float(np.nanstd(imb)); idx += 1
        else:
            idx += 2

        # Spread by delta bucket
        for _bname, bdf in self._delta_buckets(df).items():
            vec[idx] = _col_mean(bdf, "spread"); idx += 1

        while idx < 180:
            vec[idx] = 0.0; idx += 1

        # ── dims 180-209: Sentiment/Regime ──
        idx = 180
        # PCR (put-call ratio)
        vec[idx] = (put_vol / call_vol) if call_vol > 0 else 1.0; idx += 1; n_filled += 1
        # IV percentile (approx)
        vec[idx] = atm_iv; idx += 1
        # IV mean, std
        vec[idx] = _col_mean(df, "implied_vol"); idx += 1
        vec[idx] = _col_std(df, "implied_vol"); idx += 1
        # Skew metrics
        vec[idx] = otm_put_iv - atm_iv; idx += 1
        vec[idx] = otm_call_iv - atm_iv; idx += 1
        # Volume momentum proxies
        vec[idx] = _col_sum(df, "volume"); idx += 1
        vec[idx] = _col_sum(df, "count") if "count" in df.columns else 0.0; idx += 1
        # Regime indicators
        if "implied_vol" in df.columns:
            iv_vals = df["implied_vol"].dropna().astype(float)
            vec[idx] = float(iv_vals.skew()) if len(iv_vals) >= 3 else 0.0; idx += 1
            vec[idx] = float(iv_vals.kurtosis()) if len(iv_vals) >= 4 else 0.0; idx += 1
        else:
            idx += 2

        while idx < 210:
            vec[idx] = 0.0; idx += 1

        # ── dims 210-239: Cross-Strike-Time ──
        idx = 210
        if "strike" in df.columns and "implied_vol" in df.columns:
            strikes = df["strike"].dropna().astype(float)
            ivs = df["implied_vol"].dropna().astype(float)
            if len(strikes) >= 5:
                corr = float(np.corrcoef(strikes.values[:min(len(strikes), len(ivs))],
                                         ivs.values[:min(len(strikes), len(ivs))])[0, 1])
                vec[idx] = corr if np.isfinite(corr) else 0.0
            idx += 1
        else:
            idx += 1

        if "delta" in df.columns and "implied_vol" in df.columns:
            d_vals = df["delta"].dropna().astype(float)
            iv_vals = df["implied_vol"].dropna().astype(float)
            if len(d_vals) >= 5:
                n = min(len(d_vals), len(iv_vals))
                corr = float(np.corrcoef(d_vals.values[:n], iv_vals.values[:n])[0, 1])
                vec[idx] = corr if np.isfinite(corr) else 0.0
            idx += 1
        else:
            idx += 1

        while idx < 240:
            vec[idx] = 0.0; idx += 1

        # ── dims 240-269: Gamma Exposure ──
        idx = 240
        if "gamma_exp" in df.columns:
            gex_calls = _col_sum(calls, "gamma_exp")
            gex_puts = _col_sum(puts, "gamma_exp")
            vec[idx] = gex_calls + gex_puts; idx += 1; n_filled += 1  # net GEX
            vec[idx] = gex_calls; idx += 1
            vec[idx] = gex_puts; idx += 1
            vec[idx] = _col_std(df, "gamma_exp"); idx += 1
            # GEX concentration
            if "strike" in df.columns:
                gex_by_strike = df.groupby("strike")["gamma_exp"].sum()
                if len(gex_by_strike) > 0:
                    max_gex_strike = gex_by_strike.abs().idxmax()
                    atm = _col_mean(df, "atm_strike")
                    vec[idx] = float(max_gex_strike) if np.isfinite(max_gex_strike) else 0.0; idx += 1
                    vec[idx] = float(max_gex_strike - atm) if atm != 0 else 0.0; idx += 1
                else:
                    idx += 2
            else:
                idx += 2
        else:
            idx += 6

        while idx < 270:
            vec[idx] = 0.0; idx += 1

        # ── dims 270-284: Smart Money ──
        idx = 270
        if "volume" in df.columns:
            vol_vals = df["volume"].fillna(0).astype(float)
            vol_mean = vol_vals.mean()
            vol_std = vol_vals.std()
            if vol_std > 0:
                large_mask = vol_vals > (vol_mean + 2 * vol_std)
                vec[idx] = float(large_mask.sum()); idx += 1; n_filled += 1
                vec[idx] = float(vol_vals[large_mask].sum()) if large_mask.any() else 0.0; idx += 1
            else:
                idx += 2
        else:
            idx += 2
        while idx < 285:
            vec[idx] = 0.0; idx += 1

        # ── dims 285-296: Volume Anomaly ──
        idx = 285
        if "volume" in df.columns:
            vol_vals = df["volume"].fillna(0).astype(float)
            vol_mean = vol_vals.mean()
            vol_std = vol_vals.std()
            vec[idx] = float(vol_vals.max() - vol_mean) / max(vol_std, 1e-6); idx += 1; n_filled += 1
            for q in [75, 90, 95, 99]:
                vec[idx] = float(np.nanpercentile(vol_vals.values, q)); idx += 1
        else:
            idx += 5
        while idx < 297:
            vec[idx] = 0.0; idx += 1

        # ── dims 297-306: Trade Conditions ──
        idx = 297
        # Sweep ratio proxy: high volume / total volume
        if "volume" in df.columns:
            vol_vals = df["volume"].fillna(0).astype(float)
            total = vol_vals.sum()
            high = vol_vals[vol_vals > vol_vals.quantile(0.9)].sum() if len(vol_vals) > 5 else 0.0
            vec[idx] = float(high / total) if total > 0 else 0.0; idx += 1; n_filled += 1
        else:
            idx += 1
        while idx < 307:
            vec[idx] = 0.0; idx += 1

        # ── dims 307-324: Quote Pressure ──
        idx = 307
        if "bid" in df.columns and "ask" in df.columns:
            bid_vals = df["bid"].fillna(0).astype(float)
            ask_vals = df["ask"].fillna(0).astype(float)
            # Size imbalance
            total_ba = bid_vals + ask_vals
            imb = np.where(total_ba > 0, (bid_vals - ask_vals) / total_ba, 0.0)
            vec[idx] = float(np.nanmean(imb)); idx += 1; n_filled += 1
            vec[idx] = float(np.nanstd(imb)); idx += 1
            # Call vs put bid-ask
            if not calls.empty and "bid" in calls.columns:
                cb = calls["bid"].fillna(0).astype(float)
                ca = calls["ask"].fillna(0).astype(float)
                ct = cb + ca
                vec[idx] = float(np.nanmean(np.where(ct > 0, (cb - ca) / ct, 0.0))); idx += 1
            else:
                idx += 1
            if not puts.empty and "bid" in puts.columns:
                pb = puts["bid"].fillna(0).astype(float)
                pa = puts["ask"].fillna(0).astype(float)
                pt = pb + pa
                vec[idx] = float(np.nanmean(np.where(pt > 0, (pb - pa) / pt, 0.0))); idx += 1
            else:
                idx += 1
        else:
            idx += 4
        while idx < 325:
            vec[idx] = 0.0; idx += 1

        # Quality score: fraction of non-zero features
        quality = n_filled / FEAT_DIM
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return vec, min(1.0, quality)

    def extract_chain_slice(self, snap_df: pd.DataFrame) -> np.ndarray:
        """
        Extract chain_2d slice: 5 channels × strike_bins strikes.
        Channels: [delta, gamma, theta, vega, implied_vol]
        """
        result = np.zeros((5, self.strike_bins), dtype=np.float32)
        if snap_df is None or snap_df.empty:
            return result

        df = self._adapt_columns(snap_df)
        channels = ["delta", "gamma", "theta", "vega", "implied_vol"]
        missing = [c for c in channels if c not in df.columns]
        if missing or "atm_strike" not in df.columns:
            return result

        atm = _col_mean(df, "atm_strike")
        if atm <= 0:
            return result

        # Sort by distance to ATM
        df = df.copy()
        df["_dist"] = (df["strike"].astype(float) - atm).abs()
        df = df.sort_values("_dist").head(self.strike_bins)

        for ch_i, ch_name in enumerate(channels):
            vals = df[ch_name].fillna(0).astype(float).values[:self.strike_bins]
            result[ch_i, :len(vals)] = vals

        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def update_history(self, snap_df: pd.DataFrame) -> Dict[str, float]:
        """
        Process one snapshot: extract features, update rolling histories.
        Returns quality scores per symbol.
        """
        if snap_df is None or snap_df.empty:
            return {}

        snap_by_sym = {}
        if "symbol" in snap_df.columns:
            for sym, sdf in snap_df.groupby("symbol"):
                snap_by_sym[str(sym)] = sdf

        quality_scores = {}
        for symbol in ALL_SYMBOLS:
            sdf = snap_by_sym.get(symbol, pd.DataFrame())
            vec, q = self.extract_325_features(sdf)
            quality_scores[symbol] = q
            self._seq_history[symbol].append(vec)

            chain_slice = self.extract_chain_slice(sdf)
            self._chain_history[symbol].append(chain_slice)

        return quality_scores

    def get_stage1_tensors(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get (sequence_tensor, chain_tensor) for a symbol.
        sequence: (1, seq_len, 325)
        chain: (1, 5, strike_bins, seq_len)
        """
        # Sequence tensor
        hist = list(self._seq_history[symbol])
        if not hist:
            hist = [np.zeros(FEAT_DIM, dtype=np.float32)]
        while len(hist) < self.seq_len:
            hist.insert(0, hist[0].copy())
        seq = np.stack(hist[-self.seq_len:], axis=0).astype(np.float32)
        seq_tensor = seq[np.newaxis, :, :]  # (1, T, 325)

        # Chain tensor
        chain_hist = list(self._chain_history[symbol])
        if not chain_hist:
            chain_hist = [np.zeros((5, self.strike_bins), dtype=np.float32)]
        while len(chain_hist) < self.seq_len:
            chain_hist.insert(0, chain_hist[0].copy())
        # chain_hist: list of (5, S) → stack → (T, 5, S) → transpose → (5, S, T)
        chain = np.stack(chain_hist[-self.seq_len:], axis=0).transpose(1, 2, 0).astype(np.float32)
        chain_tensor = chain[np.newaxis, :, :, :]  # (1, 5, S, T)

        return seq_tensor, chain_tensor

    def build_vix_features(self, agg_df: pd.DataFrame) -> np.ndarray:
        """
        Build 10-dim VIX feature vector from agg CSV.
        Returns shape (1, 10).
        """
        vix_row = {}
        if not agg_df.empty and "symbol" in agg_df.columns:
            vix_rows = agg_df[agg_df["symbol"] == "VIXW"]
            if not vix_rows.empty:
                vix_row = vix_rows.iloc[-1].to_dict()

        vix_spot = _safe_float(vix_row.get("spot", 0.0))
        self._vix_spot_history.append(vix_spot)

        for k in ("iv_skew", "call_iv", "put_iv", "pc_ratio"):
            v = _safe_float(vix_row.get(k, 0.0))
            self._vix_meta_history[k].append(v)

        s = np.asarray(list(self._vix_spot_history), dtype=np.float32)
        if s.size == 0:
            s = np.zeros(1, dtype=np.float32)

        def pct_change(lookback: int) -> float:
            if s.size <= lookback:
                return 0.0
            base = float(s[-lookback - 1])
            return float((s[-1] - base) / base) if base != 0 else 0.0

        def zscore(window: int) -> float:
            tail = s[-window:] if s.size >= window else s
            std = float(np.std(tail))
            return float((tail[-1] - np.mean(tail)) / std) if std > 1e-8 else 0.0

        def percentile(window: int) -> float:
            tail = s[-window:] if s.size >= window else s
            return float(np.sum(tail <= tail[-1]) / tail.size) if tail.size > 1 else 0.5

        iv_skew = float(self._vix_meta_history["iv_skew"][-1]) if self._vix_meta_history["iv_skew"] else 0.0
        call_iv = float(self._vix_meta_history["call_iv"][-1]) if self._vix_meta_history["call_iv"] else 0.0
        put_iv = float(self._vix_meta_history["put_iv"][-1]) if self._vix_meta_history["put_iv"] else 0.0
        pc_ratio = float(self._vix_meta_history["pc_ratio"][-1]) if self._vix_meta_history["pc_ratio"] else 0.0

        hilo_window = s[-12:] if s.size >= 12 else s
        vix_hilo_range = float((np.max(hilo_window) - np.min(hilo_window)) / max(1e-6, np.mean(hilo_window)))

        vix_vec = np.array([
            vix_spot,
            pct_change(5),
            pct_change(15),
            pct_change(60),
            zscore(15),
            percentile(60),
            iv_skew + 0.5 * (call_iv - put_iv),
            float(np.std(s[-20:]) if s.size >= 3 else 0.0),
            (pc_ratio - 1.0),
            vix_hilo_range,
        ], dtype=np.float32)

        vix_vec = np.nan_to_num(vix_vec, nan=0.0, posinf=0.0, neginf=0.0)
        vix_vec = np.clip(vix_vec, -1e6, 1e6)
        return vix_vec.reshape(1, -1)

    @property
    def warmup_fraction(self) -> float:
        return float(np.mean([len(self._seq_history[s]) / self.seq_len for s in ALL_SYMBOLS]))

    @property
    def vix_level(self) -> float:
        return float(self._vix_spot_history[-1]) if self._vix_spot_history else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 Bundle
# ═══════════════════════════════════════════════════════════════════════════

class _Stage1Bundle:
    __slots__ = ("model", "norm_mean", "norm_std")
    def __init__(self, model: nn.Module, norm_mean: Optional[np.ndarray], norm_std: Optional[np.ndarray]):
        self.model = model
        self.norm_mean = norm_mean
        self.norm_std = norm_std


# ═══════════════════════════════════════════════════════════════════════════
# Prediction Service
# ═══════════════════════════════════════════════════════════════════════════

class PredictionService:
    """
    Loads all stage1 + stage2 + stage3 models once, then runs inference
    on each new batch of theta data, writing results to prediction.csv.
    """

    def __init__(self, data_dir: Path, device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)

        # Load config
        if CONFIG_PATH.exists():
            self.config = json.loads(CONFIG_PATH.read_text())
        else:
            logger.warning(f"Config not found at {CONFIG_PATH}, using defaults")
            self.config = {}

        self.threshold = float(
            self.config.get("architecture", {}).get("stage3", {}).get("threshold", 0.47)
        )

        # Models
        self.stage1: Dict[str, Dict[str, _Stage1Bundle]] = {s: {} for s in ALL_SYMBOLS}
        self.stage2: Dict[str, Tuple[nn.Module, dict]] = {}
        self.stage3_model: Optional[RegimeGatedProbFusion] = None
        self.stage3_agent_order: List[str] = list(ALL_AGENTS)

        # Feature bridge
        self.bridge = FeatureBridge()

        # Tracking
        self._last_batch_id: Optional[int] = None
        self._pred_csv_path = self.data_dir / "prediction.csv"

        # Load all models
        self._load_all_models()

    def _load_norm_stats(self, symbol: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Try to load normalization stats. Return (None, None) if not available."""
        horizon = int(self.config.get("model_info", {}).get("horizon_minutes", 30))
        data_root = self.config.get("data_paths", {}).get("tier3_binary_root", "")
        if not data_root:
            return None, None
        d = Path(data_root) / symbol / f"horizon_{horizon}min"
        nm_path = d / "norm_mean.npy"
        ns_path = d / "norm_std.npy"
        if nm_path.exists() and ns_path.exists():
            try:
                return np.load(nm_path), np.load(ns_path)
            except Exception:
                pass
        return None, None

    def _load_all_models(self) -> None:
        """Load all stage1, stage2, stage3 models."""
        logger.info("Loading models...")
        t0 = time.perf_counter()

        # Stage 1
        loaded_s1 = 0
        for symbol in ALL_SYMBOLS:
            nm, ns = self._load_norm_stats(symbol)
            for agent in ALL_AGENTS:
                ckpt_path = MODEL_DIR / f"stage1/{symbol}_agent{agent}.pt"
                if not ckpt_path.exists():
                    continue
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    model = _build_model_from_ckpt(ckpt, agent_type=agent, device=self.device, symbol=symbol)
                    self.stage1[symbol][agent] = _Stage1Bundle(model=model, norm_mean=nm, norm_std=ns)
                    loaded_s1 += 1
                except Exception as e:
                    logger.warning(f"  Failed to load stage1 {symbol}/{agent}: {e}")
        logger.info(f"  Stage1: {loaded_s1}/35 models loaded")

        # Stage 2
        loaded_s2 = 0
        for agent in ALL_AGENTS:
            ckpt_path = MODEL_DIR / f"stage2/agent{agent}_fusion.pt"
            if not ckpt_path.exists():
                continue
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                n_inputs = int(ckpt["n_inputs"])
                fusion = CrossSymbolAgentFusion(n_inputs=n_inputs, hidden_dim=32, dropout=0.2).to(self.device)
                sd_key = "fusion_state_dict" if agent == "2D" else "model_state_dict"
                fusion.load_state_dict(ckpt[sd_key], strict=True)
                fusion.eval()
                self.stage2[agent] = (fusion, ckpt)
                loaded_s2 += 1
            except Exception as e:
                logger.warning(f"  Failed to load stage2 {agent}: {e}")
        logger.info(f"  Stage2: {loaded_s2}/7 models loaded")

        # Stage 3
        ckpt3_path = MODEL_DIR / "stage3/stage3_vix_gated.pt"
        if ckpt3_path.exists():
            try:
                ckpt3 = torch.load(ckpt3_path, map_location="cpu", weights_only=False)
                self.stage3_model = RegimeGatedProbFusion(
                    agent_names=ckpt3["agent_names"],
                    vix_feat_dim=int(ckpt3["vix_feat_dim"]),
                    regime_emb_dim=int(ckpt3["regime_emb_dim"]),
                    fusion_hidden_dim=int(ckpt3["fusion_hidden_dim"]),
                    dropout=float(ckpt3["dropout"]),
                ).to(self.device)
                self.stage3_model.load_state_dict(ckpt3["model_state_dict"], strict=True)
                self.stage3_model.eval()
                self.threshold = float(ckpt3.get("threshold", self.threshold))
                self.stage3_agent_order = list(ckpt3["agent_names"])
                logger.info("  Stage3: loaded")
            except Exception as e:
                logger.warning(f"  Failed to load stage3: {e}")
        else:
            logger.warning(f"  Stage3 checkpoint not found: {ckpt3_path}")

        elapsed = time.perf_counter() - t0
        logger.info(f"Models loaded in {elapsed:.1f}s")

    def _read_csv_safe(self, path: Path) -> pd.DataFrame:
        """Read CSV, return empty DataFrame on error."""
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _get_latest_batch_id(self, agg_df: pd.DataFrame) -> Optional[int]:
        if agg_df.empty or "batch_id" not in agg_df.columns:
            return None
        try:
            return int(pd.to_numeric(agg_df["batch_id"], errors="coerce").dropna().max())
        except Exception:
            return None

    def _get_spot_price(self, agg_df: pd.DataFrame) -> float:
        """Get SPXW spot price from agg data."""
        if agg_df.empty or "symbol" not in agg_df.columns:
            return 0.0
        spxw = agg_df[agg_df["symbol"] == "SPXW"]
        if spxw.empty or "spot" not in spxw.columns:
            return 0.0
        return _safe_float(spxw.iloc[-1]["spot"])

    @torch.no_grad()
    def _stage1_predict(self, seq: np.ndarray, chain: Optional[np.ndarray], bundle: _Stage1Bundle) -> Tuple[float, float]:
        """Run a single Stage1 prediction. Returns (logit, prob)."""
        x = torch.from_numpy(seq.astype(np.float32)).to(self.device)
        if bundle.norm_mean is not None and bundle.norm_std is not None:
            nm = torch.from_numpy(bundle.norm_mean.astype(np.float32)).to(self.device)
            ns = torch.from_numpy(bundle.norm_std.astype(np.float32)).to(self.device)
            x = (x - nm) / torch.clamp(ns, min=1e-6)

        c = None
        if chain is not None:
            c = torch.from_numpy(chain.astype(np.float32)).to(self.device)

        logits = bundle.model(x, chain_2d=c).detach().cpu().numpy().reshape(-1)
        logit = float(logits[-1])
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        return logit, prob

    # ═══════════════════════════════════════════════════════════════════════
    # Evidence-Based Confidence Computation
    # ═══════════════════════════════════════════════════════════════════════
    #
    # Replaces the fake formula: confidence = abs(prob - threshold) * 2
    #
    # Based on ensemble uncertainty research (arxiv 2509.14386):
    #   - Agent disagreement (variance) converges to aleatoric uncertainty
    #   - Binary supervision alone CANNOT produce calibrated confidence
    #   - Ensemble disagreement is the gold standard for confidence
    #
    # Four signals, all derived from actual model internals:
    #   1. Agent Agreement  (0.40) — std of 7 stage2 agent probabilities
    #   2. Consensus Ratio  (0.20) — fraction of agents on same side as pred
    #   3. Gate Conviction  (0.20) — gate-weighted |agent_prob - 0.5|
    #   4. Data Quality     (0.20) — feature completeness + warmup
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_confidence(
        agent_probs: np.ndarray,   # shape (7,) stage2 probs
        gate_weights: np.ndarray,  # shape (7,) stage3 gate weights
        pred: int,                 # 0 or 1
        feature_completeness: float,
        warmup_frac: float,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute evidence-based confidence from actual model signals.

        Returns:
            confidence: float in [0.0, 1.0]
            signal_strength: float in [-1.0, 1.0]  (sign=direction, magnitude=confidence)
            details: dict with decomposition for CSV/dashboard transparency
        """
        n_agents = len(agent_probs)

        # ── Signal 1: Agent Agreement (weight 0.40) ──
        # Low std dev = agents agree = high confidence
        # Max practical std for 7 binary classifiers ≈ 0.20
        agent_std = float(np.std(agent_probs))
        MAX_PRACTICAL_STD = 0.20
        conf_agreement = float(max(0.0, 1.0 - agent_std / MAX_PRACTICAL_STD))

        # ── Signal 2: Consensus Ratio (weight 0.20) ──
        # How many agents agree with the final prediction direction
        threshold_mid = 0.5  # each agent's own decision boundary
        if pred == 1:
            agreeing = np.sum(agent_probs > threshold_mid)
        else:
            agreeing = np.sum(agent_probs <= threshold_mid)
        consensus_ratio = float(agreeing / n_agents)
        # Rescale: 50% agreement = 0 confidence, 100% = 1.0
        conf_consensus = float(np.clip((consensus_ratio - 0.5) * 2.0, 0.0, 1.0))

        # ── Signal 3: Gate-Weighted Conviction (weight 0.20) ──
        # How strongly the *trusted* agents feel about their predictions
        # |prob - 0.5| measures how far from uncertain each agent is
        agent_convictions = np.abs(agent_probs - 0.5)
        gate_sum = float(np.sum(gate_weights))
        if gate_sum > 1e-8:
            gate_normalized = gate_weights / gate_sum
            gate_weighted_conv = float(np.dot(gate_normalized, agent_convictions))
        else:
            gate_weighted_conv = float(np.mean(agent_convictions))
        # Max practical conviction per agent = 0.5 (prob at 0 or 1)
        # Typical range 0.05-0.25; scale so 0.25 → 1.0
        conf_gate_conviction = float(np.clip(gate_weighted_conv / 0.25, 0.0, 1.0))

        # ── Signal 4: Data Quality (weight 0.20) ──
        # feature_completeness: fraction of 325 features filled (from bridge)
        # warmup_fraction: how much of SEQ_LEN history we have
        conf_data_quality = float(np.clip(
            0.7 * feature_completeness + 0.3 * min(1.0, warmup_frac),
            0.0, 1.0
        ))

        # ── Final Confidence ──
        confidence = float(
            0.40 * conf_agreement +
            0.20 * conf_consensus +
            0.20 * conf_gate_conviction +
            0.20 * conf_data_quality
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # ── Signal Strength (directional confidence) ──
        raw_direction = 1.0 if pred == 1 else -1.0
        signal_strength = float(np.clip(raw_direction * confidence, -1.0, 1.0))

        details = {
            "agent_std": round(agent_std, 6),
            "consensus_ratio": round(consensus_ratio, 4),
            "conf_agreement": round(conf_agreement, 4),
            "conf_consensus": round(conf_consensus, 4),
            "conf_gate_conviction": round(conf_gate_conviction, 4),
            "conf_data_quality": round(conf_data_quality, 4),
        }
        return confidence, signal_strength, details

    @torch.no_grad()
    def _run_inference(self, agg_df: pd.DataFrame, snap_df: pd.DataFrame) -> Dict[str, Any]:
        """Run full 3-stage inference pipeline. Returns prediction dict."""
        t0 = time.perf_counter()

        # Update feature bridge with new snapshot
        quality_scores = self.bridge.update_history(snap_df)
        vix_features = self.bridge.build_vix_features(agg_df)
        warmup_frac = self.bridge.warmup_fraction
        feature_completeness = float(np.mean(list(quality_scores.values()))) if quality_scores else 0.0
        # Quality score: same weights as conf_data_quality for consistency
        quality_score = float(0.7 * feature_completeness + 0.3 * min(1.0, warmup_frac))
        vix_valid = bool(np.isfinite(vix_features).all())

        # Check suppression
        suppression_reason = None
        if not vix_valid:
            suppression_reason = "vix_features_invalid"
        elif feature_completeness < 0.01:
            suppression_reason = "no_snapshot_data"
        elif warmup_frac < 0.35:
            n = int(warmup_frac * SEQ_LEN)
            suppression_reason = f"warmup_{n}_of_{SEQ_LEN}"

        if suppression_reason:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return self._suppressed_result(
                suppression_reason, quality_score, feature_completeness,
                warmup_frac, latency_ms, self.bridge.vix_level,
            )

        # ── STAGE 1 ──
        stage1_logits: Dict[str, Dict[str, float]] = {s: {} for s in ALL_SYMBOLS}
        stage1_probs: Dict[str, Dict[str, float]] = {s: {} for s in ALL_SYMBOLS}
        missing_stage1 = 0

        for symbol in ALL_SYMBOLS:
            seq_t, chain_t = self.bridge.get_stage1_tensors(symbol)
            for agent in ALL_AGENTS:
                bundle = self.stage1.get(symbol, {}).get(agent)
                if bundle is None:
                    missing_stage1 += 1
                    continue
                try:
                    lg, pb = self._stage1_predict(
                        seq_t, chain_t if agent == "2D" else None, bundle
                    )
                    stage1_logits[symbol][agent] = lg
                    stage1_probs[symbol][agent] = pb
                except Exception as e:
                    logger.debug(f"Stage1 {symbol}/{agent} failed: {e}")
                    missing_stage1 += 1

        # ── STAGE 2 ──
        stage2_probs: Dict[str, float] = {}
        for agent in ALL_AGENTS:
            if agent not in self.stage2:
                stage2_probs[agent] = 0.5
                continue
            fusion, ckpt = self.stage2[agent]
            try:
                X = self._build_stage2_design_matrix(agent, ckpt, stage1_logits, stage1_probs)
                n_inputs = int(ckpt["n_inputs"])
                # Pad or truncate if shape mismatch
                if X.shape[1] != n_inputs:
                    if X.shape[1] < n_inputs:
                        X = np.concatenate([X, np.zeros((1, n_inputs - X.shape[1]), dtype=np.float32)], axis=1)
                    else:
                        X = X[:, :n_inputs]
                logits = fusion(torch.from_numpy(X).to(self.device)).detach().cpu().numpy().reshape(-1)
                prob = float(1.0 / (1.0 + np.exp(-float(logits[-1]))))
                stage2_probs[agent] = prob
            except Exception as e:
                logger.debug(f"Stage2 {agent} failed: {e}")
                stage2_probs[agent] = 0.5

        # ── STAGE 3 ──
        if self.stage3_model is None:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return self._suppressed_result(
                "stage3_not_loaded", quality_score, feature_completeness,
                warmup_frac, latency_ms, self.bridge.vix_level,
            )

        agent_cols = [stage2_probs.get(a, 0.5) for a in self.stage3_agent_order]
        agent_mat = np.asarray(agent_cols, dtype=np.float32).reshape(1, -1)
        vix_feat = vix_features.astype(np.float32)
        if vix_feat.shape[1] != 10:
            vix_feat = np.pad(vix_feat, ((0, 0), (0, max(0, 10 - vix_feat.shape[1]))),
                              constant_values=0.0)[:, :10]

        logits3, gates3, _ = self.stage3_model(
            torch.from_numpy(agent_mat).to(self.device),
            torch.from_numpy(vix_feat).to(self.device),
        )
        prob = float(torch.sigmoid(logits3).detach().cpu().numpy().reshape(-1)[-1])
        pred = int(prob > self.threshold)
        gates = gates3.detach().cpu().numpy().reshape(-1)
        gates_map = {a: float(gates[i]) for i, a in enumerate(self.stage3_agent_order)}

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        direction = "BULL" if pred == 1 else "BEAR"

        # ── Evidence-based confidence (replaces fake abs(prob-threshold)*2) ──
        agent_prob_values = np.array([stage2_probs.get(a, 0.5) for a in ALL_AGENTS], dtype=np.float64)
        gate_values = np.array([gates_map.get(a, 0.0) for a in ALL_AGENTS], dtype=np.float64)

        confidence, signal_strength, conf_details = self._compute_confidence(
            agent_prob_values, gate_values, pred, feature_completeness, warmup_frac
        )

        return {
            "prob": prob,
            "pred": pred,
            "threshold": self.threshold,
            "confidence": confidence,
            "signal_strength": signal_strength,
            "direction": direction,
            "stage2_probs": {a: float(stage2_probs.get(a, 0.5)) for a in ALL_AGENTS},
            "gates": gates_map,
            "quality_score": quality_score,
            "feature_completeness": feature_completeness,
            "warmup_fraction": warmup_frac,
            "latency_ms": latency_ms,
            "stage1_missing_count": missing_stage1,
            "suppressed": False,
            "reason": "",
            "vix_level": self.bridge.vix_level,
            # Confidence decomposition
            "agent_std": conf_details["agent_std"],
            "consensus_ratio": conf_details["consensus_ratio"],
            "conf_agreement": conf_details["conf_agreement"],
            "conf_consensus": conf_details["conf_consensus"],
            "conf_gate_conviction": conf_details["conf_gate_conviction"],
            "conf_data_quality": conf_details["conf_data_quality"],
        }

    def _build_stage2_design_matrix(
        self, agent: str, ckpt: dict,
        stage1_logits: Dict[str, Dict[str, float]],
        stage1_probs: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """Build the design matrix for a Stage2 fusion model."""
        if agent == "2D":
            active_peers = ckpt.get("active_peers", AGENT_2D_PEER_SYMBOLS)
            spxw_logit = stage1_logits["SPXW"].get("2D", 0.0)
            spxw_prob = stage1_probs["SPXW"].get("2D", 0.5)
            parts: List[float] = [spxw_logit, spxw_prob]
            peer_logits_only: List[float] = []
            for sym in active_peers:
                peer_l = stage1_logits.get(sym, {}).get("2D", 0.0)
                peer_p = stage1_probs.get(sym, {}).get("2D", 0.5)
                parts.extend([peer_l, peer_p])
                peer_logits_only.append(peer_l)
            for peer_l in peer_logits_only:
                parts.append(spxw_logit - peer_l)
        else:
            syms = ckpt.get("symbols", ["SPXW"] + STANDARD_PEER_SYMBOLS)
            peer_syms = [s for s in syms if s != "SPXW"]
            spxw_logit = stage1_logits["SPXW"].get(agent, 0.0)
            parts = [spxw_logit, stage1_probs["SPXW"].get(agent, 0.5)]
            for sym in peer_syms:
                parts.append(stage1_logits.get(sym, {}).get(agent, 0.0))
                parts.append(stage1_probs.get(sym, {}).get(agent, 0.5))
            for sym in peer_syms:
                parts.append(spxw_logit - stage1_logits.get(sym, {}).get(agent, 0.0))
            # Chain context proxy (SPXW 2D logit/prob)
            ctx_logit = stage1_logits["SPXW"].get("2D", 0.0)
            ctx_prob = stage1_probs["SPXW"].get("2D", 0.5)
            parts.extend([ctx_logit, ctx_prob])

        return np.asarray(parts, dtype=np.float32).reshape(1, -1)

    def _suppressed_result(
        self, reason: str, quality: float, completeness: float,
        warmup: float, latency: float, vix: float,
    ) -> Dict[str, Any]:
        return {
            "prob": 0.5,
            "pred": 0,
            "threshold": self.threshold,
            "confidence": 0.0,
            "signal_strength": 0.0,
            "direction": "SUPPRESSED",
            "stage2_probs": {a: 0.5 for a in ALL_AGENTS},
            "gates": {a: 0.0 for a in ALL_AGENTS},
            "quality_score": quality,
            "feature_completeness": completeness,
            "warmup_fraction": warmup,
            "latency_ms": latency,
            "stage1_missing_count": 0,
            "suppressed": True,
            "reason": reason,
            "vix_level": vix,
            # Confidence decomposition: all zero when suppressed
            "agent_std": 0.0,
            "consensus_ratio": 0.0,
            "conf_agreement": 0.0,
            "conf_consensus": 0.0,
            "conf_gate_conviction": 0.0,
            "conf_data_quality": 0.0,
        }

    def _write_prediction_row(self, batch_id: int, result: Dict[str, Any], spot_price: float) -> None:
        """Append one row to prediction.csv."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
        s2 = result["stage2_probs"]
        g = result["gates"]

        row = {
            "batch_id": batch_id,
            "ts": ts,
            "prob": round(result["prob"], 6),
            "pred": result["pred"],
            "threshold": round(result["threshold"], 4),
            "confidence": round(result["confidence"], 6),
            "signal_strength": round(result["signal_strength"], 6),
            "direction": result["direction"],
            "agent_A_prob": round(s2.get("A", 0.5), 6),
            "agent_B_prob": round(s2.get("B", 0.5), 6),
            "agent_C_prob": round(s2.get("C", 0.5), 6),
            "agent_K_prob": round(s2.get("K", 0.5), 6),
            "agent_T_prob": round(s2.get("T", 0.5), 6),
            "agent_Q_prob": round(s2.get("Q", 0.5), 6),
            "agent_2D_prob": round(s2.get("2D", 0.5), 6),
            "gate_A": round(g.get("A", 0.5), 6),
            "gate_B": round(g.get("B", 0.5), 6),
            "gate_C": round(g.get("C", 0.5), 6),
            "gate_K": round(g.get("K", 0.5), 6),
            "gate_T": round(g.get("T", 0.5), 6),
            "gate_Q": round(g.get("Q", 0.5), 6),
            "gate_2D": round(g.get("2D", 0.5), 6),
            "quality_score": round(result["quality_score"], 4),
            "feature_completeness": round(result["feature_completeness"], 4),
            "warmup_fraction": round(result["warmup_fraction"], 4),
            "latency_ms": round(result["latency_ms"], 2),
            "stage1_missing_count": result["stage1_missing_count"],
            "suppressed": result["suppressed"],
            "reason": result["reason"],
            "vix_level": round(result["vix_level"], 4),
            "spot_price": round(spot_price, 2),
            # Confidence decomposition
            "agent_std": round(result.get("agent_std", 0.0), 6),
            "consensus_ratio": round(result.get("consensus_ratio", 0.0), 4),
            "conf_agreement": round(result.get("conf_agreement", 0.0), 4),
            "conf_consensus": round(result.get("conf_consensus", 0.0), 4),
            "conf_gate_conviction": round(result.get("conf_gate_conviction", 0.0), 4),
            "conf_data_quality": round(result.get("conf_data_quality", 0.0), 4),
        }

        write_header = not self._pred_csv_path.exists()
        with open(self._pred_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PRED_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def tick(self) -> bool:
        """
        Check for new data and run inference if available.
        Returns True if a prediction was made.
        """
        agg_df = self._read_csv_safe(self.data_dir / "theta_agg.csv")
        snap_df = self._read_csv_safe(self.data_dir / "theta_snapshot.csv")

        batch_id = self._get_latest_batch_id(agg_df)
        if batch_id is None:
            return False

        if batch_id == self._last_batch_id:
            return False

        self._last_batch_id = batch_id
        spot_price = self._get_spot_price(agg_df)

        try:
            result = self._run_inference(agg_df, snap_df)
            self._write_prediction_row(batch_id, result, spot_price)

            status = "SUPPRESSED" if result["suppressed"] else result["direction"]
            prob_str = f"{result['prob']:.3f}" if not result["suppressed"] else "---"
            logger.info(
                f"Batch {batch_id}: {status} prob={prob_str} "
                f"conf={result['confidence']:.3f} "
                f"latency={result['latency_ms']:.0f}ms "
                f"quality={result['quality_score']:.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"Inference error at batch {batch_id}: {e}")
            self._write_prediction_row(batch_id, self._suppressed_result(
                f"error: {str(e)[:80]}", 0.0, 0.0, 0.0, 0.0, 0.0,
            ), spot_price)
            return True

    def run_loop(self, interval: int = 30) -> None:
        """Run prediction loop at given interval (seconds)."""
        logger.info(f"Starting prediction loop (interval={interval}s)")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Output: {self._pred_csv_path}")
        logger.info(f"  Threshold: {self.threshold}")

        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
            time.sleep(interval)


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid51 Prediction Service — reads theta CSVs, writes prediction.csv"
    )
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Directory containing theta_agg.csv and theta_snapshot.csv")
    parser.add_argument("--interval", type=int, default=30,
                        help="Polling interval in seconds (default: 30)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: auto-detect)")
    parser.add_argument("--once", action="store_true",
                        help="Run a single prediction and exit")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Handle graceful shutdown
    shutdown = False
    def _sig_handler(sig, frame):
        nonlocal shutdown
        shutdown = True
        logger.info("Received shutdown signal")
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    service = PredictionService(data_dir=Path(args.data_dir), device=device)

    if args.once:
        service.tick()
    else:
        service.run_loop(interval=args.interval)


if __name__ == "__main__":
    main()
