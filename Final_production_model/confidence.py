"""
Evidence-based confidence computation for the Hybrid51 ensemble.

Replaces the old fake formula: confidence = abs(prob - threshold) * 2

Four components (all derived from actual model internals):
    1. Agent Agreement   (0.40) — std of 7 Stage-2 agent probabilities
    2. Consensus Ratio   (0.20) — fraction of agents on the same side as pred
    3. Gate Conviction   (0.20) — gate-weighted |agent_prob - 0.5|
    4. Data Quality      (0.20) — feature completeness + warmup fraction

Exports:
    compute_confidence
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_confidence(
    agent_probs: np.ndarray,
    gate_weights: np.ndarray,
    pred: int,
    feature_completeness: float,
    warmup_frac: float,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute evidence-based confidence from ensemble internals.

    Args:
        agent_probs         : shape (7,) Stage-2 output probabilities
        gate_weights        : shape (7,) Stage-3 gate values (all 1.0 with LogReg)
        pred                : final prediction (0 = BEAR, 1 = BULL)
        feature_completeness: fraction of 325 features that are non-zero
        warmup_frac         : fraction of SEQ_LEN history filled

    Returns:
        confidence     : float in [0, 1]
        signal_strength: float in [-1, 1]  (sign = direction, magnitude = confidence)
        details        : decomposition dict for CSV / dashboard transparency
    """
    n_agents = len(agent_probs)

    # ── 1. Agent Agreement (0.40) ──────────────────────────────────────────
    # Low std-dev means agents agree → high confidence.
    # Max practical std for 7 binary classifiers ≈ 0.20.
    agent_std = float(np.std(agent_probs))
    MAX_PRACTICAL_STD = 0.35   # raised from 0.20: real 4/3 agent split yields std≈0.14-0.20; allow headroom
    conf_agreement = float(max(0.0, 1.0 - agent_std / MAX_PRACTICAL_STD))

    # ── 2. Consensus Ratio (0.20) ──────────────────────────────────────────
    # How many agents agree with the final predicted direction.
    if pred == 1:
        agreeing = int(np.sum(agent_probs > 0.5))
    else:
        agreeing = int(np.sum(agent_probs <= 0.5))
    consensus_ratio = float(agreeing / n_agents)
    # 50% agreement → 0 confidence, 100% → 1.0
    conf_consensus = float(np.clip((consensus_ratio - 0.5) * 2.0, 0.0, 1.0))

    # ── 3. Gate-Weighted Conviction (0.20) ────────────────────────────────
    # How strongly the trusted agents feel about their calls.
    # |prob - 0.5| measures distance from uncertainty; max = 0.5.
    # Practical ceiling ≈ 0.25 → scale so 0.25 maps to 1.0.
    agent_convictions = np.abs(agent_probs - 0.5)
    gate_sum = float(np.sum(gate_weights))
    if gate_sum > 1e-8:
        gate_conv = float(np.dot(gate_weights / gate_sum, agent_convictions))
    else:
        gate_conv = float(np.mean(agent_convictions))
    conf_gate_conviction = float(np.clip(gate_conv / 0.25, 0.0, 1.0))

    # ── 4. Data Quality (0.20) ────────────────────────────────────────────
    conf_data_quality = float(np.clip(
        0.7 * feature_completeness + 0.3 * min(1.0, warmup_frac),
        0.0, 1.0,
    ))

    # ── Final composite ───────────────────────────────────────────────────
    confidence = float(np.clip(
        0.40 * conf_agreement
        + 0.20 * conf_consensus
        + 0.20 * conf_gate_conviction
        + 0.20 * conf_data_quality,
        0.0, 1.0,
    ))

    raw_direction = 1.0 if pred == 1 else -1.0
    signal_strength = float(np.clip(raw_direction * confidence, -1.0, 1.0))

    details: Dict[str, float] = {
        "agent_std":            round(agent_std, 6),
        "consensus_ratio":      round(consensus_ratio, 4),
        "conf_agreement":       round(conf_agreement, 4),
        "conf_consensus":       round(conf_consensus, 4),
        "conf_gate_conviction": round(conf_gate_conviction, 4),
        "conf_data_quality":    round(conf_data_quality, 4),
    }
    return confidence, signal_strength, details
