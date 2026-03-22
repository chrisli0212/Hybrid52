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
import json
import logging
import os
import signal
import sys
import time
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
    """

    def __init__(self, data_dir: Path, device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.device   = torch.device(device)

        self.config: Dict[str, Any] = {}
        if CONFIG_PATH.exists():
            self.config = json.loads(CONFIG_PATH.read_text())
        else:
            logger.warning(f"Config not found at {CONFIG_PATH}, using defaults")

        self.threshold = float(
            self.config.get("architecture", {}).get("stage3", {}).get("threshold", 0.36)
        )

        # Model containers
        self.stage1: Dict[str, Dict[str, _Stage1Bundle]] = {s: {} for s in ALL_SYMBOLS}
        self.stage2: Dict[str, Tuple[nn.Module, dict]]   = {}
        self.stage3_model: Optional[Any]                 = None
        self.stage3_agent_order: List[str]               = list(ALL_AGENTS)

        # Feature bridge (325-dim extractor + rolling histories)
        self.bridge = FeatureBridge()

        # Change-detection state
        self._last_batch_id: Optional[int] = None
        self._last_agg_mtime: float        = 0.0
        self._stale_count: int             = 0
        self._pred_csv_path                = self.data_dir / "prediction.csv"

        self._load_all_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_norm_stats(self, symbol: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        horizon = int(self.config.get("model_info", {}).get("horizon_minutes", 30))
        sub     = Path(symbol) / f"horizon_{horizon}min"

        # Build a list of candidate directories to search:
        #  1. Config-specified tier3_binary_root (absolute path)
        #  2. Repo-relative: <repo_root>/data/tier3_binary_v5/
        #  3. Sibling of SCRIPT_DIR: ../data/tier3_binary_v5/
        candidates: list[Path] = []
        cfg_root = self.config.get("data_paths", {}).get("tier3_binary_root", "")
        if cfg_root:
            candidates.append(Path(cfg_root) / sub)
        candidates.append(SCRIPT_DIR.parent / "data" / "tier3_binary_v5" / sub)
        candidates.append(SCRIPT_DIR / "data" / "tier3_binary_v5" / sub)

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
        for symbol in ALL_SYMBOLS:
            nm, ns = self._load_norm_stats(symbol)
            for agent in ALL_AGENTS:
                path = MODEL_DIR / f"stage1/{symbol}_agent{agent}.pt"
                if not path.exists():
                    continue
                try:
                    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
                    model = _build_model_from_ckpt(ckpt, agent_type=agent, device=self.device, symbol=symbol)
                    # Per-symbol Platt scaling — calibrates raw logits using
                    # coefficients fitted during training for each (symbol, agent).
                    platt_c = float(np.array(ckpt.get("platt_scaler_coef", [[1.0]])).flatten()[0])
                    platt_i = float(np.array(ckpt.get("platt_scaler_intercept", [0.0])).flatten()[0])
                    self.stage1[symbol][agent] = _Stage1Bundle(
                        model=model, norm_mean=nm, norm_std=ns,
                        platt_coef=platt_c, platt_intercept=platt_i,
                    )
                    loaded_s1 += 1
                except Exception as e:
                    logger.warning(f"  Failed to load stage1 {symbol}/{agent}: {e}")
        logger.info(f"  Stage1: {loaded_s1}/35 models loaded")

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
                loaded_s2 += 1
            except Exception as e:
                logger.warning(f"  Failed to load stage2 agent{agent}: {e}")
        logger.info(f"  Stage2: {loaded_s2}/7 models loaded")

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

    # ------------------------------------------------------------------
    # Stage-1 single prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _stage1_predict(
        self,
        seq: np.ndarray,
        chain: Optional[np.ndarray],
        bundle: _Stage1Bundle,
    ) -> Tuple[float, float]:
        """Returns (logit, prob) for one (symbol, agent) pair.

        Applies per-symbol Platt scaling to the raw model logit before
        converting to probability.  This uses the calibration coefficients
        fitted during training (stored in each checkpoint), which stretch
        the output range and make each symbol's agent more sensitive to
        changes in the underlying features.
        """
        x = torch.from_numpy(seq.astype(np.float32)).to(self.device)
        if bundle.norm_mean is not None and bundle.norm_std is not None:
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
        return calibrated_logit, float(1.0 / (1.0 + np.exp(-calibrated_logit)))

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
        snap_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        quality_scores      = self.bridge.update_history(snap_df)
        vix_features        = self.bridge.build_vix_features(agg_df, snap_df=snap_df)
        warmup_frac         = self.bridge.warmup_fraction
        # Persist warmup progress across service/dashboard restarts:
        # if today's prediction rows already reached sequence length, unlock immediately.
        persisted_rows_today = self._prediction_rows_today()
        persisted_warmup_frac = min(1.0, persisted_rows_today / float(SEQ_LEN))
        warmup_frac_effective = max(float(warmup_frac), float(persisted_warmup_frac))
        feature_completeness = float(np.mean(list(quality_scores.values()))) if quality_scores else 0.0
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
        elif warmup_frac_effective < 0.35:
            suppression_reason = f"warmup_{int(warmup_frac_effective * SEQ_LEN)}_of_{SEQ_LEN}"
        elif self.bridge.vix_level == 0.0 and warmup_frac_effective >= 1.0:
            suppression_reason = "vix_level_zero"

        if suppression_reason:
            return self._suppressed_result(
                suppression_reason, quality_score, feature_completeness,
                warmup_frac_effective, round((time.perf_counter() - t0) * 1000.0, 2),
                self.bridge.vix_level,
            )

        # ── STAGE 1 ───────────────────────────────────────────────────────
        stage1_logits: Dict[str, Dict[str, float]] = {s: {} for s in ALL_SYMBOLS}
        stage1_probs:  Dict[str, Dict[str, float]] = {s: {} for s in ALL_SYMBOLS}
        missing_s1 = 0

        for symbol in ALL_SYMBOLS:
            seq_t, chain_t = self.bridge.get_stage1_tensors(symbol)
            for agent in ALL_AGENTS:
                bundle = self.stage1.get(symbol, {}).get(agent)
                if bundle is None:
                    missing_s1 += 1
                    continue
                try:
                    lg, pb = self._stage1_predict(
                        seq_t, chain_t if agent == "2D" else None, bundle
                    )
                    stage1_logits[symbol][agent] = lg
                    stage1_probs[symbol][agent]  = pb
                except Exception as e:
                    logger.debug(f"Stage1 {symbol}/{agent} failed: {e}")
                    missing_s1 += 1

        # ── STAGE 2 ───────────────────────────────────────────────────────
        stage2_probs: Dict[str, float] = {}
        failed_s2:    List[str]        = []

        for agent in ALL_AGENTS:
            if agent not in self.stage2:
                logger.warning(f"Stage2 agent {agent} not loaded — using 0.5 fallback")
                stage2_probs[agent] = 0.5
                failed_s2.append(agent)
                continue
            fusion, ckpt = self.stage2[agent]
            try:
                X        = self._build_stage2_design_matrix(agent, ckpt, stage1_logits, stage1_probs)
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

        if failed_s2:
            logger.warning(f"Stage2 failed agents: {failed_s2}")

        # ── STAGE 3 ───────────────────────────────────────────────────────
        if self.stage3_model is None:
            return self._suppressed_result(
                "stage3_not_loaded", quality_score, feature_completeness,
                warmup_frac, round((time.perf_counter() - t0) * 1000.0, 2),
                self.bridge.vix_level,
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
                float(np.mean(agent_probs_ordered > 0.5)),
                float(agent_probs_ordered.max()),
                float(agent_probs_ordered.min()),
            ],
        ]).reshape(1, -1).astype(np.float32)

        prob  = float(self.stage3_model.predict_proba(meta_feat)[0, 1])
        pred  = int(prob > self.threshold)
        # Gates are uniformly 1.0 — RegimeGatedProbFusion is not deployed in this config.
        # conf_gate_conviction reflects flat-weighted conviction, not regime-adaptive gates.
        gates = {a: 1.0 for a in self.stage3_agent_order}

        latency_ms  = round((time.perf_counter() - t0) * 1000.0, 2)
        direction   = "BULL" if pred == 1 else "BEAR"

        agent_prob_arr = np.array([stage2_probs.get(a, 0.5) for a in ALL_AGENTS], dtype=np.float64)
        gate_arr       = np.array([gates.get(a, 1.0) for a in ALL_AGENTS], dtype=np.float64)
        confidence, signal_strength, conf_details = compute_confidence(
            agent_prob_arr, gate_arr, pred, feature_completeness, warmup_frac_effective
        )

        return {
            "prob":                 prob,
            "pred":                 pred,
            "threshold":            self.threshold,
            "confidence":           confidence,
            "signal_strength":      signal_strength,
            "direction":            direction,
            "stage2_probs":         {a: float(stage2_probs.get(a, 0.5)) for a in ALL_AGENTS},
            "gates":                gates,
            "quality_score":        quality_score,
            "feature_completeness": feature_completeness,
            "warmup_fraction":      warmup_frac_effective,
            "latency_ms":           latency_ms,
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
            "warmup_fraction":      warmup,
            "latency_ms":           latency,
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
            "warmup_fraction":      round(result["warmup_fraction"],      4),
            "latency_ms":           round(result["latency_ms"],           2),
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

        write_header = not self._pred_csv_path.exists()
        with open(self._pred_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PRED_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Main polling loop
    # ------------------------------------------------------------------

    def tick(self) -> bool:
        """
        Check for new data and run inference if a new batch is available.
        Returns True if a prediction row was written.
        """
        agg_path = self.data_dir / "theta_agg.csv"
        agg_df   = self._read_csv_safe(agg_path)
        snap_df  = self._read_csv_safe(self.data_dir / "theta_snapshot.csv")

        batch_id = self._get_latest_batch_id(agg_df)
        if batch_id is None:
            return False

        if batch_id == self._last_batch_id:
            try:
                current_mtime = agg_path.stat().st_mtime
            except OSError:
                return False
            if current_mtime != self._last_agg_mtime:
                self._stale_count += 1
                self._last_agg_mtime = current_mtime
                if self._stale_count < 3:
                    return False
                logger.info(
                    f"Batch {batch_id} unchanged but file rewritten "
                    f"{self._stale_count}x — forcing prediction"
                )
            else:
                return False

        self._stale_count    = 0
        self._last_batch_id  = batch_id
        try:
            self._last_agg_mtime = agg_path.stat().st_mtime
        except OSError:
            pass

        spot = self._get_spot_price(agg_df)
        try:
            result = self._run_inference(agg_df, snap_df)
            self._write_prediction_row(batch_id, result, spot)
            status   = "SUPPRESSED" if result["suppressed"] else result["direction"]
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
