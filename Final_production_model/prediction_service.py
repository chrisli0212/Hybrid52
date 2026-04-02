#!/usr/bin/env python3
"""
Prediction Service — Hybrid51 h30 Live Inference Orchestrator

Reads theta CSV data, runs the full 3-stage inference pipeline, and writes
results to prediction.csv.  Runs on a polling loop alongside theta_fetching.

Dedicated modules (imported below) own each subsystem:
    stage1_models   — BinaryIndependentAgent, _build_model_from_ckpt, _Stage1Bundle
    feature_bridge  — FeatureBridge (325-dim features, VIX 10-dim, chain-2D)
    confidence      — compute_confidence (evidence-based, 4-component)

Usage:
    python prediction_service.py                          # defaults
    python prediction_service.py --data-dir ./daily_data  # custom data dir
    python prediction_service.py --interval 10            # custom poll (sec)
    python prediction_service.py --device cpu             # force CPU
    python prediction_service.py --once                   # single prediction
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
MODEL_DIR   = SCRIPT_DIR / "models"
CONFIG_PATH = SCRIPT_DIR / "config" / "production_config.json"
DEFAULT_DATA_DIR = SCRIPT_DIR / "daily_data"

# Ensure local packages are importable
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

HYBRID51_DIR = SCRIPT_DIR.parent / "Hybrid51" / "6. Hybrid51_new stage"
if str(HYBRID51_DIR) not in sys.path:
    sys.path.insert(1, str(HYBRID51_DIR))

# ---------------------------------------------------------------------------
# Sub-module imports  (each owns one responsibility)
# ---------------------------------------------------------------------------
from stage1_models import BinaryIndependentAgent, _build_model_from_ckpt, _Stage1Bundle  # noqa: E402
from feature_bridge import FeatureBridge, _safe_float                                    # noqa: E402
from confidence import compute_confidence                                                 # noqa: E402
from data_ingestion_endpoint_loader import EndpointBatchLoader                            # noqa: E402

from hybrid51_models.cross_symbol_agent_fusion import CrossSymbolAgentFusion  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prediction_service")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_SYMBOLS          = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
ALL_AGENTS           = ["A", "B", "C", "K", "T", "Q", "2D"]
STANDARD_PEER_SYMBOLS = ["SPY", "QQQ", "IWM"]
AGENT_2D_PEER_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT"]
SEQ_LEN = 20

PRED_CSV_COLUMNS = [
    "batch_id", "ts",
    "prob", "pred", "threshold", "confidence", "signal_strength", "direction",
    "agent_A_prob", "agent_B_prob", "agent_C_prob", "agent_K_prob",
    "agent_T_prob", "agent_Q_prob", "agent_2D_prob",
    "gate_A", "gate_B", "gate_C", "gate_K", "gate_T", "gate_Q", "gate_2D",
    "quality_score", "feature_completeness", "warmup_fraction", "latency_ms",
    "feature_nonzero_density", "tq_corr", "stage1_tq_corr",
    "chain_ready_2d", "stage1_2d_success", "stage2_2d_fallback", "gate_2d_pinned",
    "stage1_missing_count", "stage2_failed_agents", "suppressed", "reason",
    "vix_level", "spot_price",
    "agent_std", "consensus_ratio",
    "conf_agreement", "conf_consensus", "conf_gate_conviction", "conf_data_quality",
]


# ═══════════════════════════════════════════════════════════════════════════
# PredictionService
# ═══════════════════════════════════════════════════════════════════════════

class PredictionService:
    """
    Orchestrates the full Hybrid51 inference pipeline.

    Loads all stage-1 / stage-2 / stage-3 models once at startup, then on
    each tick: reads the latest theta CSVs, runs the 3-stage pipeline, and
    appends one row to prediction.csv.

    Tier-3 training uses **1-minute** feature sequences (20 steps ≈ 20 minutes).
    The fetcher may still run every ~10s. For tier-2 **minute** training parity,
    set ``inference.history_align_to_exchange_minute`` (and optionally
    ``history_minute_aggregate``: ``mean`` or ``last``) so each LSTM step is one
    **NY** minute bar built from sub-minute snapshots. Alternatively
    ``history_min_interval_seconds`` gates deque advances on a rolling wall-clock
    interval without calendar alignment.

    Shorter lookbacks: ``FeatureBridge.get_stage1_tensors`` pads the past with
    the oldest frame so the LSTM always sees SEQ_LEN steps; ``inference.min_warmup_fraction``
    (default ``1/SEQ_LEN``) allows the first prediction as soon as one real
    step exists, then quality improves as more minutes accrue toward 20.
    """

    def __init__(self, data_dir: Path, device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.device   = torch.device(device)

        self.config: Dict[str, Any] = {}
        if CONFIG_PATH.exists():
            self.config = json.loads(CONFIG_PATH.read_text())
        else:
            logger.warning(f"Config not found at {CONFIG_PATH}, using defaults")

        stage3_cfg = self.config.get("architecture", {}).get("stage3", {}) or {}
        # Preserve whether threshold was explicitly set in config so we can
        # deterministically choose between config and checkpoint defaults.
        self._stage3_threshold_explicit = "threshold" in stage3_cfg
        self.threshold = float(stage3_cfg.get("threshold", 0.36))
        self.stage3_method = str(
            self.config.get("architecture", {}).get("stage3", {}).get("method", "logreg")
        ).strip().lower()

        # Model containers
        self.stage1: Dict[str, Dict[str, _Stage1Bundle]] = {s: {} for s in ALL_SYMBOLS}
        self.stage2: Dict[str, Tuple[nn.Module, dict]]   = {}
        self.stage3_model: Optional[Any]                 = None
        self.stage3_vg: Optional[nn.Module]              = None   # VIX regime-gated fusion
        self.stage3_vg_threshold: float                  = 0.47
        self.stage3_agent_order: List[str]               = list(ALL_AGENTS)
        self._suppression_reason_counts: Dict[str, int]  = {}
        self._agent_prob_history: Dict[str, deque]       = {
            a: deque(maxlen=120) for a in ALL_AGENTS
        }
        self._gate_history: Dict[str, deque] = {
            a: deque(maxlen=120) for a in ALL_AGENTS
        }
        self._last_variance_warn_ts: float               = 0.0
        self._last_corr_warn_ts: float                   = 0.0
        self._last_gate_warn_ts: float                   = 0.0

        # Feature bridge (325-dim extractor + rolling histories)
        inf_cfg = self.config.get("inference", {})
        _hist_iv = float(inf_cfg.get("history_min_interval_seconds", 0) or 0)
        _hist_align = bool(inf_cfg.get("history_align_to_exchange_minute", False))
        _hist_aggr = str(inf_cfg.get("history_minute_aggregate", "mean")).strip().lower()
        _hist_iv_use = 0.0 if _hist_align else _hist_iv
        if _hist_align and _hist_iv > 0:
            logger.info(
                "  FeatureBridge: history_min_interval_seconds ignored "
                "(history_align_to_exchange_minute=true uses NY minute boundaries)"
            )
        self.bridge = FeatureBridge(
            history_min_interval_seconds=_hist_iv_use,
            history_align_to_exchange_minute=_hist_align,
            history_minute_aggregate=_hist_aggr,
        )
        if _hist_align:
            logger.info(
                f"  FeatureBridge: NY-minute aggregation mode={_hist_aggr!r} "
                "(10s+ fetches pooled per ET minute for tier3 parity)"
            )
        elif _hist_iv > 0:
            logger.info(
                f"  FeatureBridge: history_min_interval_seconds={_hist_iv} "
                "(deque + VIX steps aligned to tier3 1m spacing)"
            )
        # Minimum rolling depth before allowing inference (fraction of SEQ_LEN).
        # Default 1/SEQ_LEN => first real step after startup; tensors still padded to SEQ_LEN.
        self._min_warmup_fraction = float(
            self.config.get("inference", {}).get("min_warmup_fraction", 1.0 / float(SEQ_LEN))
        )
        self._min_warmup_fraction = float(np.clip(self._min_warmup_fraction, 0.0, 1.0))
        logger.info(
            f"  Inference: min_warmup_fraction={self._min_warmup_fraction} "
            f"(suppress below this fraction of {SEQ_LEN} real timesteps)"
        )
        self.endpoint_loader = EndpointBatchLoader(self.data_dir)

        # Change-detection state
        self._last_batch_id: Optional[int] = None
        self._last_agg_mtime: float        = 0.0
        self._stale_count: int             = 0
        self._last_input_signature: Optional[str] = None
        self._duplicate_skip_count: int = 0
        self._pred_csv_path                = self.data_dir / "prediction.csv"

        self._load_all_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_norm_stats(self, symbol: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        horizon = int(self.config.get("model_info", {}).get("horizon_minutes", 30))
        sub     = Path(symbol) / f"horizon_{horizon}min"

        # Build search candidates from explicit roots only (env override + config).
        # Avoid silent fallback to stale hardcoded datasets.
        candidates: list[Path] = []
        env_root_h51 = os.getenv("HYBRID51_DATA_ROOT", "").strip()
        env_root_h52 = os.getenv("HYBRID52_DATA_ROOT", "").strip()
        cfg_root = str(self.config.get("data_paths", {}).get("tier3_binary_root", "")).strip()
        for root in [env_root_h51, env_root_h52, cfg_root]:
            if root:
                p = Path(root) / sub
                if p not in candidates:
                    candidates.append(p)
        if env_root_h51:
            logger.info(f"  {symbol}: HYBRID51_DATA_ROOT={env_root_h51}")
        if env_root_h52:
            logger.info(f"  {symbol}: HYBRID52_DATA_ROOT={env_root_h52}")

        for d in candidates:
            nm_path, ns_path = d / "norm_mean.npy", d / "norm_std.npy"
            if nm_path.exists() and ns_path.exists():
                try:
                    nm, ns = np.load(nm_path), np.load(ns_path)
                    logger.info(f"  {symbol}: normalisation loaded from {d}")
                    return nm, ns
                except Exception as e:
                    logger.warning(f"  {symbol}: failed to load normalisation from {d} — {e}")

        logger.error(
            f"  {symbol}: *** NORMALISATION FILES NOT FOUND *** — "
            f"searched {[str(c) for c in candidates]}.  "
            f"Model will produce DEGRADED outputs without z-score normalisation!"
        )
        return None, None

    def _load_all_models(self) -> None:
        logger.info("Loading models...")
        t0 = time.perf_counter()

        # ── Stage 1 ───────────────────────────────────────────────────────
        loaded_s1 = 0
        inverted_s1 = 0
        for symbol in ALL_SYMBOLS:
            nm, ns = self._load_norm_stats(symbol)
            for agent in ALL_AGENTS:
                path = MODEL_DIR / f"stage1/{symbol}_agent{agent}.pt"
                if not path.exists():
                    continue
                try:
                    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
                    model = _build_model_from_ckpt(
                        ckpt,
                        agent_type=agent,
                        device=self.device,
                        symbol=symbol,
                        seq_len=SEQ_LEN,
                    )
                    # Per-symbol Platt scaling — calibrates raw logits using
                    # coefficients fitted during training for each (symbol, agent).
                    platt_c = float(np.array(ckpt.get("platt_scaler_coef", [[1.0]])).flatten()[0])
                    platt_i = float(np.array(ckpt.get("platt_scaler_intercept", [0.0])).flatten()[0])
                    invert_signal = bool(ckpt.get("invert_signal", False))
                    if invert_signal:
                        inverted_s1 += 1
                    # If filesystem norms are missing, fall back to checkpoint-embedded norms.
                    if (nm is None or ns is None) and ("norm_mean" in ckpt) and ("norm_std" in ckpt):
                        try:
                            nm_ckpt = np.asarray(ckpt["norm_mean"], dtype=np.float32).reshape(-1)
                            ns_ckpt = np.asarray(ckpt["norm_std"], dtype=np.float32).reshape(-1)
                            if nm_ckpt.shape == ns_ckpt.shape and nm_ckpt.ndim == 1:
                                nm, ns = nm_ckpt, ns_ckpt
                                logger.warning(
                                    f"  {symbol}/{agent}: using checkpoint-embedded normalization "
                                    f"(len={nm_ckpt.shape[0]})"
                                )
                        except Exception as e:
                            logger.warning(f"  {symbol}/{agent}: failed to parse ckpt norm fallback: {e}")
                    self.stage1[symbol][agent] = _Stage1Bundle(
                        model=model, norm_mean=nm, norm_std=ns,
                        platt_coef=platt_c, platt_intercept=platt_i,
                        invert_signal=invert_signal, agent_type=agent,
                    )
                    loaded_s1 += 1
                except Exception as e:
                    logger.warning(f"  Failed to load stage1 {symbol}/{agent}: {e}")
        logger.info(f"  Stage1: {loaded_s1}/35 models loaded")
        logger.info(f"  Stage1: invert_signal enabled for {inverted_s1} models")

        # ── Stage 2 ───────────────────────────────────────────────────────
        loaded_s2 = 0
        for agent in ALL_AGENTS:
            path = MODEL_DIR / f"stage2/agent{agent}_fusion.pt"
            if not path.exists():
                continue
            try:
                ckpt     = torch.load(path, map_location="cpu", weights_only=False)
                n_inputs = int(ckpt["n_inputs"])
                fusion   = CrossSymbolAgentFusion(n_inputs=n_inputs, hidden_dim=32, dropout=0.2).to(self.device)
                sd_key   = "fusion_state_dict" if agent == "2D" else "model_state_dict"
                fusion.load_state_dict(ckpt[sd_key], strict=True)
                fusion.eval()
                self.stage2[agent] = (fusion, ckpt)
                if (
                    agent == "2D"
                    and ckpt.get("target") == "SPXW"
                    and "spxw_2d_state_dict" in ckpt
                ):
                    spxw_bundle = self.stage1.get("SPXW", {}).get("2D")
                    if spxw_bundle is not None:
                        try:
                            spxw_bundle.model.load_state_dict(
                                ckpt["spxw_2d_state_dict"],
                                strict=True,
                            )
                            spxw_bundle.model.eval()
                            logger.info(
                                "  Stage2 agent2D: loaded fine-tuned SPXW 2D backbone "
                                "from stage2 checkpoint"
                            )
                        except Exception as e:
                            logger.warning(
                                "  Stage2 agent2D: failed to load spxw_2d_state_dict "
                                f"({e}) — using original Stage1 SPXW 2D weights"
                            )
                loaded_s2 += 1
            except Exception as e:
                logger.warning(f"  Failed to load stage2 agent{agent}: {e}")
        logger.info(f"  Stage2: {loaded_s2}/7 models loaded")
        self._log_stage2_input_compatibility()

        # ── Stage 3 (LogReg joblib) ───────────────────────────────────────
        path3 = MODEL_DIR / "stage3/stage3_logreg.joblib"
        if path3.exists():
            try:
                self.stage3_model = joblib.load(path3)
                logger.info("  Stage3: loaded (LogReg)")
            except Exception as e:
                logger.warning(f"  Failed to load stage3: {e}")
        else:
            logger.warning(f"  Stage3 model not found: {path3}")

        # ── Stage 3 VIX regime-gated fusion (uses learned per-agent gates) ──
        path3_vg = MODEL_DIR / "stage3/stage3_vix_gated.pt"
        if path3_vg.exists():
            try:
                from hybrid51_models.regime_gated_meta_model import RegimeGatedProbFusion
                vg_ckpt = torch.load(path3_vg, map_location=self.device, weights_only=False)
                self.stage3_vg = RegimeGatedProbFusion(
                    agent_names=vg_ckpt["agent_names"],
                    vix_feat_dim=vg_ckpt["vix_feat_dim"],
                    regime_emb_dim=vg_ckpt["regime_emb_dim"],
                    fusion_hidden_dim=vg_ckpt["fusion_hidden_dim"],
                    dropout=vg_ckpt["dropout"],
                ).to(self.device)
                self.stage3_vg.load_state_dict(vg_ckpt["model_state_dict"], strict=True)
                self.stage3_vg.eval()
                self.stage3_vg_threshold = float(vg_ckpt.get("threshold", 0.47))
                # Align gate order to the checkpoint's trained agent ordering.
                # Avoids silent misalignment if checkpoint was trained with a
                # different agent sequence than ALL_AGENTS.
                if "agent_names" in vg_ckpt:
                    self.stage3_agent_order = list(vg_ckpt["agent_names"])
                logger.info("  Stage3-VG: loaded (VIX regime-gated fusion)")
            except Exception as e:
                logger.warning(f"  Failed to load stage3 VIX-gated: {e}")

        # ── Normalization health check ─────────────────────────────────
        syms_missing_norm = [
            s for s in ALL_SYMBOLS
            if self.stage1.get(s) and any(
                b.norm_mean is None for b in self.stage1[s].values()
            )
        ]
        if syms_missing_norm:
            logger.error(
                f"  *** CRITICAL: normalisation missing for {syms_missing_norm}. "
                f"Stage-1 outputs will be UNRELIABLE (model trained on z-scored features). "
                f"Fix the tier3_binary_root path in config/production_config.json "
                f"or ensure norm_mean.npy / norm_std.npy exist."
            )

        logger.info(f"Models loaded in {time.perf_counter() - t0:.1f}s")

    def _expected_stage2_input_dim(self, agent: str, ckpt: dict) -> int:
        """
        Expected input width from the current runtime feature builder.
        Used only for startup compatibility logging.
        """
        if agent == "2D":
            active_peers = ckpt.get("active_peers", AGENT_2D_PEER_SYMBOLS)
            return 2 + 2 * len(active_peers) + len(active_peers)
        syms = ckpt.get("symbols", ["SPXW"] + STANDARD_PEER_SYMBOLS)
        peer_syms = [s for s in syms if s != "SPXW"]
        return 2 + 2 * len(peer_syms) + len(peer_syms) + 2

    def _log_stage2_input_compatibility(self) -> None:
        for agent in ALL_AGENTS:
            entry = self.stage2.get(agent)
            if entry is None:
                logger.warning(f"  Stage2 agent{agent}: model missing (will fallback to 0.5)")
                continue
            _, ckpt = entry
            expected = self._expected_stage2_input_dim(agent, ckpt)
            actual = int(ckpt.get("n_inputs", -1))
            if expected != actual:
                logger.warning(
                    f"  Stage2 agent{agent}: input design mismatch runtime={expected} ckpt={actual} "
                    "(runtime padding/truncation may reduce variance)"
                )
            else:
                logger.info(f"  Stage2 agent{agent}: input design verified n_inputs={actual}")

    def _record_agent_variance(self, stage2_probs: Dict[str, float]) -> None:
        for agent in ALL_AGENTS:
            self._agent_prob_history[agent].append(float(stage2_probs.get(agent, 0.5)))

        # Emit at most once every 5 minutes to avoid noisy logs.
        now = time.time()
        if now - self._last_variance_warn_ts < 300:
            return

        low_var_agents: List[str] = []
        for agent, hist in self._agent_prob_history.items():
            if len(hist) < 30:
                continue
            std = float(np.std(np.asarray(hist, dtype=np.float32)))
            if std < 0.005:
                low_var_agents.append(f"{agent}(std={std:.4f})")

        if low_var_agents:
            logger.warning(
                "Stage2 low-variance agents over recent window: "
                + ", ".join(low_var_agents)
            )
            self._last_variance_warn_ts = now

    def _rolling_tq_corr(self, min_points: int = 30) -> float:
        t_hist = self._agent_prob_history["T"]
        q_hist = self._agent_prob_history["Q"]
        n = min(len(t_hist), len(q_hist))
        if n < min_points:
            return 0.0
        t = np.asarray(list(t_hist)[-n:], dtype=np.float32)
        q = np.asarray(list(q_hist)[-n:], dtype=np.float32)
        if float(np.std(t)) < 1e-8 or float(np.std(q)) < 1e-8:
            return 0.0
        corr = float(np.corrcoef(t, q)[0, 1])
        return corr if np.isfinite(corr) else 0.0

    def _record_tq_correlation_warning(self, tq_corr: float) -> None:
        now = time.time()
        if now - self._last_corr_warn_ts < 300:
            return
        if abs(tq_corr) >= 0.9:
            logger.warning(f"Stage2 high T/Q correlation detected: corr={tq_corr:.4f}")
            self._last_corr_warn_ts = now

    def _record_gate_diagnostics(self, gates: Dict[str, float]) -> bool:
        for agent in ALL_AGENTS:
            self._gate_history[agent].append(float(gates.get(agent, 1.0)))

        g_hist = self._gate_history["2D"]
        p_hist = self._agent_prob_history["2D"]
        n = min(len(g_hist), len(p_hist))
        if n < 30:
            return False

        g = np.asarray(list(g_hist)[-n:], dtype=np.float32)
        p = np.asarray(list(p_hist)[-n:], dtype=np.float32)
        gate_mean = float(np.mean(g))
        gate_std = float(np.std(g))
        prob_std = float(np.std(p))
        pinned = gate_mean < 1e-3 and gate_std < 1e-3 and prob_std > 0.01

        now = time.time()
        if pinned and (now - self._last_gate_warn_ts >= 300):
            logger.warning(
                "Gate2D pinned near zero while agent2D varies "
                f"(gate_mean={gate_mean:.6f}, gate_std={gate_std:.6f}, prob_std={prob_std:.6f})"
            )
            self._last_gate_warn_ts = now
        return pinned

    @staticmethod
    def _compute_stage1_tq_corr(stage1_raw_logits: Dict[str, Dict[str, float]]) -> float:
        t_vals: List[float] = []
        q_vals: List[float] = []
        for sym in ALL_SYMBOLS:
            row = stage1_raw_logits.get(sym, {})
            if "T" in row and "Q" in row:
                t_vals.append(float(row["T"]))
                q_vals.append(float(row["Q"]))
        if len(t_vals) < 3:
            return 0.0
        t = np.asarray(t_vals, dtype=np.float32)
        q = np.asarray(q_vals, dtype=np.float32)
        if float(np.std(t)) < 1e-8 or float(np.std(q)) < 1e-8:
            return 0.0
        corr = float(np.corrcoef(t, q)[0, 1])
        return corr if np.isfinite(corr) else 0.0

    def _build_input_signature(
        self,
        batch_id: int,
        agg_df: pd.DataFrame,
        greek_df: pd.DataFrame,
        trade_quote_df: pd.DataFrame,
    ) -> str:
        """
        Hash semantic input content so rewritten files with unchanged data
        do not produce duplicate prediction rows.
        """
        hasher = hashlib.sha1()
        hasher.update(str(int(batch_id)).encode("utf-8"))

        def _update_df(df: pd.DataFrame, preferred_cols: List[str]) -> None:
            if df is None or df.empty:
                hasher.update(b"empty")
                return
            cols = [c for c in preferred_cols if c in df.columns]
            if not cols:
                cols = list(df.columns[: min(8, len(df.columns))])
            slim = df[cols].copy()
            sort_cols = [c for c in ("symbol", "ts", "strike", "right", "dte_group") if c in slim.columns]
            if sort_cols:
                slim = slim.sort_values(sort_cols, kind="stable")
            hashed = pd.util.hash_pandas_object(slim.fillna(0), index=False).values
            hasher.update(hashed.tobytes())

        _update_df(agg_df, ["symbol", "ts", "batch_id", "spot", "call_vol", "put_vol", "pc_ratio"])
        _update_df(
            greek_df,
            ["symbol", "ts", "strike", "right", "spot", "delta", "gamma", "theta", "vega", "implied_vol"],
        )
        _update_df(
            trade_quote_df,
            ["symbol", "ts", "strike", "right", "bid_size", "ask_size", "volume", "price", "bid", "ask"],
        )
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Stage-1 single prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _stage1_predict(
        self,
        seq: np.ndarray,
        chain: Optional[np.ndarray],
        bundle: _Stage1Bundle,
    ) -> Tuple[float, float, float]:
        """Returns (raw_logit, calibrated_logit, calibrated_prob) for one (symbol, agent) pair.

        Applies per-symbol Platt scaling to the raw model logit before
        converting to probability.  This uses the calibration coefficients
        fitted during training (stored in each checkpoint), which stretch
        the output range and make each symbol's agent more sensitive to
        changes in the underlying features.
        """
        x = torch.from_numpy(seq.astype(np.float32)).to(self.device)
        if bundle.norm_mean is not None and bundle.norm_std is not None:
            # Hybrid52 parity: T and Q apply raw ±5σ clipping before z-score.
            if bundle.agent_type in ("T", "Q"):
                clip_std = np.maximum(bundle.norm_std.astype(np.float32), 1e-8)
                low = bundle.norm_mean.astype(np.float32) - 5.0 * clip_std
                high = bundle.norm_mean.astype(np.float32) + 5.0 * clip_std
                seq_clipped = np.clip(seq.astype(np.float32, copy=True), low, high)
                x = torch.from_numpy(seq_clipped).to(self.device)
            nm = torch.from_numpy(bundle.norm_mean.astype(np.float32)).to(self.device)
            ns = torch.from_numpy(bundle.norm_std.astype(np.float32)).to(self.device)
            x  = (x - nm) / torch.clamp(ns, min=1e-6)
            x  = torch.clamp(x, -10.0, 10.0)

        c = torch.from_numpy(chain.astype(np.float32)).to(self.device) if chain is not None else None
        logits = bundle.model(x, chain_2d=c).detach().cpu().numpy().reshape(-1)
        raw_logit = float(np.clip(logits[-1], -88.0, 88.0))

        # Platt scaling: calibrated_logit = coef * raw_logit + intercept
        # Trained per (symbol, agent) pair — stretches weak-signal symbols
        # (e.g. TLT coef~0.01 compresses noise, SPXW coef~1.05 preserves signal).
        calibrated_logit = float(np.clip(
            bundle.platt_coef * raw_logit + bundle.platt_intercept,
            -88.0, 88.0,
        ))
        calibrated_prob = float(1.0 / (1.0 + np.exp(-calibrated_logit)))
        if bundle.invert_signal:
            calibrated_prob = 1.0 - calibrated_prob
            p = float(np.clip(calibrated_prob, 1e-6, 1.0 - 1e-6))
            calibrated_logit = float(np.clip(np.log(p / (1.0 - p)), -88.0, 88.0))
        return raw_logit, calibrated_logit, calibrated_prob

    @staticmethod
    def _chain_has_real_data(chain: Optional[np.ndarray]) -> bool:
        """
        Return True when the 2D chain tensor contains usable non-zero values.
        """
        if chain is None:
            return False
        arr = np.asarray(chain)
        if arr.size == 0:
            return False
        finite = np.isfinite(arr)
        if not finite.any():
            return False
        return bool(np.any(np.abs(arr[finite]) > 1e-8))

    # ------------------------------------------------------------------
    # Stage-2 design matrix
    # ------------------------------------------------------------------

    def _build_stage2_design_matrix(
        self,
        agent: str,
        ckpt: dict,
        stage1_logits: Dict[str, Dict[str, float]],
        stage1_probs:  Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """
        Build the (1, n_inputs) design matrix for one Stage-2 fusion model.

        Standard agents (A/B/C/K/T/Q) — 13 dims:
            [spxw_l, spxw_p]                  (2)
            [peer_l, peer_p] × 3 peers        (6)
            [spxw_l − peer_l] × 3 peers       (3)
            [ctx_l, ctx_p]  ← SPXW 2D proxy   (2)

        Agent 2D — 14 dims:
            [spxw_l, spxw_p]                  (2)
            [peer_l, peer_p] × 4 peers        (8)
            [spxw_l − peer_l] × 4 peers       (4)
        """
        if agent == "2D":
            active_peers  = ckpt.get("active_peers", AGENT_2D_PEER_SYMBOLS)
            spxw_logit    = stage1_logits["SPXW"].get("2D", 0.0)
            spxw_prob     = stage1_probs["SPXW"].get("2D", 0.5)
            parts: List[float] = [spxw_logit, spxw_prob]
            peer_logits: List[float] = []
            for sym in active_peers:
                pl = stage1_logits.get(sym, {}).get("2D", 0.0)
                pp = stage1_probs.get(sym, {}).get("2D", 0.5)
                parts.extend([pl, pp])
                peer_logits.append(pl)
            for pl in peer_logits:
                parts.append(spxw_logit - pl)
        else:
            syms      = ckpt.get("symbols", ["SPXW"] + STANDARD_PEER_SYMBOLS)
            peer_syms = [s for s in syms if s != "SPXW"]
            spxw_logit = stage1_logits["SPXW"].get(agent, 0.0)
            parts = [spxw_logit, stage1_probs["SPXW"].get(agent, 0.5)]
            for sym in peer_syms:
                parts.append(stage1_logits.get(sym, {}).get(agent, 0.0))
                parts.append(stage1_probs.get(sym, {}).get(agent, 0.5))
            for sym in peer_syms:
                parts.append(spxw_logit - stage1_logits.get(sym, {}).get(agent, 0.0))
            # Chain-context proxy: SPXW Agent-2D Stage-1 outputs
            parts.extend([
                stage1_logits["SPXW"].get("2D", 0.0),
                stage1_probs["SPXW"].get("2D", 0.5),
            ])

        return np.asarray(parts, dtype=np.float32).reshape(1, -1)

    # ------------------------------------------------------------------
    # Full 3-stage inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_inference(
        self,
        agg_df: pd.DataFrame,
        greek_df: pd.DataFrame,
        trade_quote_df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()

        quality_scores, hist_advanced = self.bridge.update_history(
            greek_df, trade_quote_df=trade_quote_df
        )
        if not hist_advanced:
            return None
        vix_snap = trade_quote_df if trade_quote_df is not None and not trade_quote_df.empty else greek_df
        vix_features = self.bridge.build_vix_features(agg_df, snap_df=vix_snap)
        warmup_frac         = self.bridge.warmup_fraction
        # Persist warmup progress across service/dashboard restarts:
        # if today's prediction rows already reached sequence length, unlock immediately.
        persisted_rows_today = self._prediction_rows_today()
        persisted_warmup_frac = min(1.0, persisted_rows_today / float(SEQ_LEN))
        warmup_frac_effective = max(float(warmup_frac), float(persisted_warmup_frac))
        feature_completeness = float(np.mean(list(quality_scores.values()))) if quality_scores else 0.0
        feature_nonzero_density = float(self.bridge.nonzero_density)
        quality_score       = float(0.7 * feature_completeness + 0.3 * min(1.0, warmup_frac_effective))
        vix_valid           = bool(np.isfinite(vix_features).all())

        if warmup_frac_effective >= 1.0 and self.bridge.vix_level == 0.0:
            logger.warning("VIX level is 0.0 after full warmup — VIXW data may be absent.")

        # Suppression checks
        suppression_reason: Optional[str] = None
        if not vix_valid:
            suppression_reason = "vix_features_invalid"
        elif feature_completeness < 0.01:
            suppression_reason = "no_snapshot_data"
        elif warmup_frac_effective < self._min_warmup_fraction:
            suppression_reason = (
                f"warmup_{int(round(warmup_frac_effective * SEQ_LEN))}_of_{SEQ_LEN}"
            )
        elif self.bridge.vix_level == 0.0 and warmup_frac_effective >= 1.0:
            suppression_reason = "vix_level_zero"

        if suppression_reason:
            return self._suppressed_result(
                suppression_reason, quality_score, feature_completeness,
                warmup_frac_effective, round((time.perf_counter() - t0) * 1000.0, 2),
                self.bridge.vix_level, feature_nonzero_density=feature_nonzero_density,
            )

        # ── STAGE 1 ───────────────────────────────────────────────────────
        # Keep both raw and calibrated Stage-1 logits:
        # - raw logits match Stage-2 training feature distribution
        # - calibrated logits/probs preserve post-Platt semantics
        stage1_logits: Dict[str, Dict[str, float]] = {s: {} for s in ALL_SYMBOLS}
        stage1_raw_logits: Dict[str, Dict[str, float]] = {s: {} for s in ALL_SYMBOLS}
        stage1_probs:  Dict[str, Dict[str, float]] = {s: {} for s in ALL_SYMBOLS}
        missing_s1 = 0
        has_real_chain_data = False
        chain_ready_2d = False
        stage1_2d_success = 0

        for symbol in ALL_SYMBOLS:
            seq_t, chain_t = self.bridge.get_stage1_tensors(symbol)
            symbol_chain_ready = self._chain_has_real_data(chain_t)
            has_real_chain_data = has_real_chain_data or symbol_chain_ready
            chain_ready_2d = chain_ready_2d or symbol_chain_ready
            for agent in ALL_AGENTS:
                bundle = self.stage1.get(symbol, {}).get(agent)
                if bundle is None:
                    missing_s1 += 1
                    continue
                if agent == "2D" and not symbol_chain_ready:
                    # During bridge warmup, chain tensors can be all-zeros; skip
                    # Agent2D stage-1 inference to avoid synthetic-chain fallback.
                    missing_s1 += 1
                    continue
                try:
                    raw_lg, cal_lg, pb = self._stage1_predict(
                        seq_t, chain_t if agent == "2D" else None, bundle
                    )
                    stage1_raw_logits[symbol][agent] = raw_lg
                    stage1_logits[symbol][agent] = cal_lg
                    stage1_probs[symbol][agent]  = pb
                    if agent == "2D":
                        stage1_2d_success += 1
                except Exception as e:
                    logger.debug(f"Stage1 {symbol}/{agent} failed: {e}")
                    missing_s1 += 1

        stage1_tq_corr = self._compute_stage1_tq_corr(stage1_raw_logits)
        if abs(stage1_tq_corr) >= 0.95:
            logger.warning(f"Stage1 T/Q raw-logit correlation high: corr={stage1_tq_corr:.4f}")

        # ── STAGE 2 ───────────────────────────────────────────────────────
        stage2_probs: Dict[str, float] = {}
        failed_s2:    List[str]        = []
        stage2_2d_fallback = False

        for agent in ALL_AGENTS:
            if agent == "2D" and not has_real_chain_data:
                logger.info("Stage2 agent2D skipped: no real chain_2d data yet (warmup/empty)")
                stage2_probs[agent] = 0.5
                failed_s2.append(agent)
                stage2_2d_fallback = True
                continue
            if agent not in self.stage2:
                logger.warning(f"Stage2 agent {agent} not loaded — using 0.5 fallback")
                stage2_probs[agent] = 0.5
                failed_s2.append(agent)
                if agent == "2D":
                    stage2_2d_fallback = True
                continue
            fusion, ckpt = self.stage2[agent]
            try:
                X        = self._build_stage2_design_matrix(agent, ckpt, stage1_raw_logits, stage1_probs)
                n_inputs = int(ckpt["n_inputs"])
                if X.shape[1] != n_inputs:
                    logger.warning(
                        f"Stage2 agent{agent}: design matrix shape {X.shape[1]} != "
                        f"expected {n_inputs} — {'padding' if X.shape[1] < n_inputs else 'truncating'}"
                    )
                    if X.shape[1] < n_inputs:
                        X = np.concatenate(
                            [X, np.zeros((1, n_inputs - X.shape[1]), dtype=np.float32)], axis=1
                        )
                    else:
                        X = X[:, :n_inputs]
                logit = fusion(torch.from_numpy(X).to(self.device)).detach().cpu().numpy().reshape(-1)[-1]
                stage2_probs[agent] = float(1.0 / (1.0 + np.exp(-float(np.clip(logit, -88.0, 88.0)))))
            except Exception as e:
                logger.warning(f"Stage2 agent{agent} inference failed — using 0.5 fallback: {e}")
                stage2_probs[agent] = 0.5
                failed_s2.append(agent)
                if agent == "2D":
                    stage2_2d_fallback = True

        if failed_s2:
            logger.warning(f"Stage2 failed agents: {failed_s2}")
        self._record_agent_variance(stage2_probs)
        tq_corr = self._rolling_tq_corr()
        self._record_tq_correlation_warning(tq_corr)

        # ── STAGE 3 ───────────────────────────────────────────────────────
        if self.stage3_model is None and self.stage3_vg is None:
            return self._suppressed_result(
                "stage3_not_loaded", quality_score, feature_completeness,
                warmup_frac, round((time.perf_counter() - t0) * 1000.0, 2),
                self.bridge.vix_level, feature_nonzero_density=feature_nonzero_density,
            )

        agent_probs_ordered = np.array(
            [stage2_probs.get(a, 0.5) for a in self.stage3_agent_order],
            dtype=np.float32,
        )
        meta_feat = np.concatenate([
            agent_probs_ordered,
            [
                float(agent_probs_ordered.mean()),
                float(agent_probs_ordered.std()),
                float(agent_probs_ordered.max() - agent_probs_ordered.min()),
                # Keep the 0.5-centered agree_up meta feature for model compatibility.
                float(np.mean(agent_probs_ordered > 0.5)),
                float(agent_probs_ordered.max()),
                float(agent_probs_ordered.min()),
            ],
        ]).reshape(1, -1).astype(np.float32)

        prob = 0.5
        stage3_threshold = self.threshold
        gates = {a: 1.0 for a in self.stage3_agent_order}

        # VIX regime-gated gates and (optionally) probability inference.
        # Reuse the already-computed vix_features; do NOT call build_vix_features()
        # again, because it mutates internal rolling history.
        vg_logits = None
        if self.stage3_vg is not None:
            try:
                vg_agent_t = torch.from_numpy(agent_probs_ordered).unsqueeze(0).to(self.device)
                vg_vix_t = torch.from_numpy(vix_features.astype(np.float32)).to(self.device)
                with torch.no_grad():
                    vg_logits_t, vg_gates, _ = self.stage3_vg(vg_agent_t, vg_vix_t)
                vg_logits = float(vg_logits_t.detach().cpu().numpy().reshape(-1)[-1])
                gate_arr_vg = vg_gates.detach().cpu().numpy().flatten()
                gates = {a: float(gate_arr_vg[i]) for i, a in enumerate(self.stage3_agent_order)}
            except Exception as e:
                logger.debug(f"VG inference failed, falling back: {e}")

        # Final Stage-3 probability source:
        # - method=vix_gated: use VG logits as primary model output
        # - otherwise: use LogReg meta model
        if self.stage3_method == "vix_gated" and vg_logits is not None:
            prob = float(1.0 / (1.0 + np.exp(-float(np.clip(vg_logits, -88.0, 88.0)))))
            # Deterministic threshold priority:
            # 1) explicit config threshold (user override),
            # 2) VG checkpoint threshold,
            # 3) generic service threshold.
            if self._stage3_threshold_explicit:
                stage3_threshold = float(self.threshold)
            else:
                stage3_threshold = float(self.stage3_vg_threshold or self.threshold)
        elif self.stage3_model is not None:
            prob = float(self.stage3_model.predict_proba(meta_feat)[0, 1])
            stage3_threshold = self.threshold
        elif vg_logits is not None:
            prob = float(1.0 / (1.0 + np.exp(-float(np.clip(vg_logits, -88.0, 88.0)))))
            stage3_threshold = float(self.stage3_vg_threshold or self.threshold)
        else:
            return self._suppressed_result(
                "stage3_inference_unavailable", quality_score, feature_completeness,
                warmup_frac, round((time.perf_counter() - t0) * 1000.0, 2),
                self.bridge.vix_level, feature_nonzero_density=feature_nonzero_density,
            )

        pred = int(prob >= stage3_threshold)

        latency_ms  = round((time.perf_counter() - t0) * 1000.0, 2)
        direction   = "BULL" if pred == 1 else "BEAR"

        agent_prob_arr = np.array([stage2_probs.get(a, 0.5) for a in ALL_AGENTS], dtype=np.float64)
        gate_arr       = np.array([gates.get(a, 1.0) for a in ALL_AGENTS], dtype=np.float64)
        confidence, signal_strength, conf_details = compute_confidence(
            agent_prob_arr, gate_arr, pred, feature_completeness, warmup_frac_effective
        )
        gate_2d_pinned = self._record_gate_diagnostics(gates)

        return {
            "prob":                 prob,
            "pred":                 pred,
            "threshold":            stage3_threshold,
            "confidence":           confidence,
            "signal_strength":      signal_strength,
            "direction":            direction,
            "stage2_probs":         {a: float(stage2_probs.get(a, 0.5)) for a in ALL_AGENTS},
            "gates":                gates,
            "quality_score":        quality_score,
            "feature_completeness": feature_completeness,
            "feature_nonzero_density": feature_nonzero_density,
            "warmup_fraction":      warmup_frac_effective,
            "latency_ms":           latency_ms,
            "tq_corr":              tq_corr,
            "stage1_tq_corr":       stage1_tq_corr,
            "chain_ready_2d":       int(chain_ready_2d),
            "stage1_2d_success":    int(stage1_2d_success),
            "stage2_2d_fallback":   int(stage2_2d_fallback),
            "gate_2d_pinned":       int(gate_2d_pinned),
            "stage1_missing_count": missing_s1,
            "stage2_failed_agents": ",".join(failed_s2),
            "suppressed":           False,
            "reason":               "",
            "vix_level":            self.bridge.vix_level,
            "agent_std":            conf_details["agent_std"],
            "consensus_ratio":      conf_details["consensus_ratio"],
            "conf_agreement":       conf_details["conf_agreement"],
            "conf_consensus":       conf_details["conf_consensus"],
            "conf_gate_conviction": conf_details["conf_gate_conviction"],
            "conf_data_quality":    conf_details["conf_data_quality"],
        }

    # ------------------------------------------------------------------
    # Suppressed result template
    # ------------------------------------------------------------------

    def _suppressed_result(
        self,
        reason: str,
        quality: float,
        completeness: float,
        warmup: float,
        latency: float,
        vix: float,
        feature_nonzero_density: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "prob":                 0.5,
            "pred":                 0,
            "threshold":            self.threshold,
            "confidence":           0.0,
            "signal_strength":      0.0,
            "direction":            "SUPPRESSED",
            "stage2_probs":         {a: 0.5 for a in ALL_AGENTS},
            "gates":                {a: 0.0 for a in ALL_AGENTS},
            "quality_score":        quality,
            "feature_completeness": completeness,
            "feature_nonzero_density": feature_nonzero_density,
            "warmup_fraction":      warmup,
            "latency_ms":           latency,
            "tq_corr":              0.0,
            "stage1_tq_corr":       0.0,
            "chain_ready_2d":       0,
            "stage1_2d_success":    0,
            "stage2_2d_fallback":   1,
            "gate_2d_pinned":       0,
            "stage1_missing_count": 0,
            "stage2_failed_agents": "",
            "suppressed":           True,
            "reason":               reason,
            "vix_level":            vix,
            "agent_std":            0.0,
            "consensus_ratio":      0.0,
            "conf_agreement":       0.0,
            "conf_consensus":       0.0,
            "conf_gate_conviction": 0.0,
            "conf_data_quality":    0.0,
        }

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _read_csv_safe(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _prediction_rows_today(self) -> int:
        """Count prediction.csv rows for the current local date."""
        path = self._pred_csv_path
        if not path.exists():
            return 0
        try:
            df = pd.read_csv(path, usecols=["ts"], on_bad_lines="skip")
            if df.empty or "ts" not in df.columns:
                return 0
            ts = pd.to_datetime(
                df["ts"].astype(str).str.replace(r"\s+[A-Z]{2,5}$", "", regex=True),
                errors="coerce",
            )
            today = datetime.now().date()
            return int((ts.dt.date == today).sum())
        except Exception:
            return 0

    def _get_latest_batch_id(self, agg_df: pd.DataFrame) -> Optional[int]:
        if agg_df.empty or "batch_id" not in agg_df.columns:
            return None
        try:
            return int(pd.to_numeric(agg_df["batch_id"], errors="coerce").dropna().max())
        except Exception:
            return None

    def _get_spot_price(self, agg_df: pd.DataFrame) -> float:
        if agg_df.empty or "symbol" not in agg_df.columns:
            return 0.0
        spxw = agg_df[agg_df["symbol"] == "SPXW"]
        if spxw.empty or "spot" not in spxw.columns:
            return 0.0
        return _safe_float(spxw.iloc[-1]["spot"])

    def _write_prediction_row(self, batch_id: int, result: Dict[str, Any], spot: float) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
        s2  = result["stage2_probs"]
        g   = result["gates"]

        row = {
            "batch_id":             batch_id,
            "ts":                   ts,
            "prob":                 round(result["prob"],             6),
            "pred":                 result["pred"],
            "threshold":            round(result["threshold"],        4),
            "confidence":           round(result["confidence"],       6),
            "signal_strength":      round(result["signal_strength"],  6),
            "direction":            result["direction"],
            "agent_A_prob":         round(s2.get("A",  0.5), 6),
            "agent_B_prob":         round(s2.get("B",  0.5), 6),
            "agent_C_prob":         round(s2.get("C",  0.5), 6),
            "agent_K_prob":         round(s2.get("K",  0.5), 6),
            "agent_T_prob":         round(s2.get("T",  0.5), 6),
            "agent_Q_prob":         round(s2.get("Q",  0.5), 6),
            "agent_2D_prob":        round(s2.get("2D", 0.5), 6),
            "gate_A":               round(g.get("A",  0.5), 6),
            "gate_B":               round(g.get("B",  0.5), 6),
            "gate_C":               round(g.get("C",  0.5), 6),
            "gate_K":               round(g.get("K",  0.5), 6),
            "gate_T":               round(g.get("T",  0.5), 6),
            "gate_Q":               round(g.get("Q",  0.5), 6),
            "gate_2D":              round(g.get("2D", 0.5), 6),
            "quality_score":        round(result["quality_score"],        4),
            "feature_completeness": round(result["feature_completeness"], 4),
            "feature_nonzero_density": round(result.get("feature_nonzero_density", 0.0), 4),
            "warmup_fraction":      round(result["warmup_fraction"],      4),
            "latency_ms":           round(result["latency_ms"],           2),
            "tq_corr":              round(result.get("tq_corr", 0.0), 6),
            "stage1_tq_corr":       round(result.get("stage1_tq_corr", 0.0), 6),
            "chain_ready_2d":       int(result.get("chain_ready_2d", 0)),
            "stage1_2d_success":    int(result.get("stage1_2d_success", 0)),
            "stage2_2d_fallback":   int(result.get("stage2_2d_fallback", 0)),
            "gate_2d_pinned":       int(result.get("gate_2d_pinned", 0)),
            "stage1_missing_count": result["stage1_missing_count"],
            "stage2_failed_agents": result.get("stage2_failed_agents", ""),
            "suppressed":           result["suppressed"],
            "reason":               result["reason"],
            "vix_level":            round(result["vix_level"],  4),
            "spot_price":           round(spot,                 2),
            "agent_std":            round(result.get("agent_std",            0.0), 6),
            "consensus_ratio":      round(result.get("consensus_ratio",      0.0), 4),
            "conf_agreement":       round(result.get("conf_agreement",       0.0), 4),
            "conf_consensus":       round(result.get("conf_consensus",       0.0), 4),
            "conf_gate_conviction": round(result.get("conf_gate_conviction", 0.0), 4),
            "conf_data_quality":    round(result.get("conf_data_quality",    0.0), 4),
        }

        fieldnames = list(PRED_CSV_COLUMNS)
        write_header = not self._pred_csv_path.exists()
        if not write_header:
            try:
                with open(self._pred_csv_path, "r", newline="") as rf:
                    first_line = rf.readline().strip()
                if first_line:
                    existing_cols = [c.strip() for c in first_line.split(",") if c.strip()]
                    if existing_cols:
                        fieldnames = existing_cols
            except Exception:
                fieldnames = list(PRED_CSV_COLUMNS)

        row_out = {k: row.get(k, "") for k in fieldnames}
        with open(self._pred_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row_out)

    # ------------------------------------------------------------------
    # Main polling loop
    # ------------------------------------------------------------------

    def tick(self) -> bool:
        """
        Check for new data and run inference if a new batch is available.
        Returns True if a prediction row was written.
        """
        batch_id, endpoint_frames, agg_df = self.endpoint_loader.load_latest()
        greek_df = endpoint_frames.get("greeks", pd.DataFrame()) if endpoint_frames else pd.DataFrame()
        trade_quote_df = endpoint_frames.get("trade_quote", pd.DataFrame()) if endpoint_frames else pd.DataFrame()

        # Backward compatibility: never feed the same merged snapshot as both greek and trade/quote.
        if batch_id is None:
            agg_path = self.data_dir / "theta_agg.csv"
            agg_df = self._read_csv_safe(agg_path)
            greek_df = self._read_csv_safe(self.data_dir / "theta_model_greeks.csv")
            trade_quote_df = self._read_csv_safe(self.data_dir / "theta_model_trade_quote.csv")
            if greek_df.empty or "batch_id" not in greek_df.columns:
                greek_df = self._read_csv_safe(self.data_dir / "theta_snapshot.csv")
                trade_quote_df = pd.DataFrame()
            batch_id = self._get_latest_batch_id(agg_df)
            if batch_id is None and not greek_df.empty and "batch_id" in greek_df.columns:
                b = pd.to_numeric(greek_df["batch_id"], errors="coerce").dropna()
                if not b.empty:
                    batch_id = int(b.max())
            if batch_id is None:
                return False

        input_signature = self._build_input_signature(batch_id, agg_df, greek_df, trade_quote_df)
        if batch_id == self._last_batch_id and input_signature == self._last_input_signature:
            self._duplicate_skip_count += 1
            if self._duplicate_skip_count == 1 or self._duplicate_skip_count % 10 == 0:
                logger.info(
                    f"Skipping duplicate input for batch {batch_id} "
                    f"(count={self._duplicate_skip_count})"
                )
            return False

        spot = self._get_spot_price(agg_df)
        try:
            result = self._run_inference(agg_df, greek_df, trade_quote_df)
            if result is None:
                return False
            self._stale_count = 0
            self._duplicate_skip_count = 0
            self._last_batch_id = batch_id
            self._last_input_signature = input_signature
            self._last_agg_mtime = time.time()

            if result.get("suppressed", False):
                reason = str(result.get("reason", "") or "unknown")
                self._suppression_reason_counts[reason] = self._suppression_reason_counts.get(reason, 0) + 1
                cnt = self._suppression_reason_counts[reason]
                if cnt == 1 or cnt % 5 == 0:
                    logger.warning(f"Suppressed reason '{reason}' count={cnt}")
            self._write_prediction_row(batch_id, result, spot)
            status   = "SUPPRESSED" if result["suppressed"] else result["direction"]
            prob_str = f"{result['prob']:.3f}" if not result["suppressed"] else "---"
            logger.info(
                f"Batch {batch_id}: {status} prob={prob_str} "
                f"conf={result['confidence']:.3f} "
                f"latency={result['latency_ms']:.0f}ms "
                f"quality={result['quality_score']:.2f} "
                f"tq_corr={float(result.get('tq_corr', 0.0)):.3f} "
                f"gate2d_pinned={int(result.get('gate_2d_pinned', 0))} "
                f"chain2d={int(result.get('chain_ready_2d', 0))}"
            )
            return True
        except Exception as e:
            logger.error(f"Inference error at batch {batch_id}: {e}")
            self._write_prediction_row(batch_id, self._suppressed_result(
                f"error: {str(e)[:80]}", 0.0, 0.0, 0.0, 0.0, 0.0,
            ), spot)
            return True

    def run_loop(self, interval: int = 10) -> None:
        logger.info(f"Starting prediction loop (interval={interval}s)")
        logger.info(f"  Data dir : {self.data_dir}")
        logger.info(f"  Output   : {self._pred_csv_path}")
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
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid51 Prediction Service")
    p.add_argument("--data-dir",  default=str(DEFAULT_DATA_DIR))
    p.add_argument("--interval",  type=int,   default=10)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--once",      action="store_true")
    p.add_argument("--vix-root",  default=None)
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    service = PredictionService(data_dir=Path(args.data_dir), device=args.device)

    def _shutdown(sig, frame):
        logger.info("Signal received — shutting down")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    if args.once:
        service.tick()
    else:
        service.run_loop(interval=args.interval)


if __name__ == "__main__":
    main()
