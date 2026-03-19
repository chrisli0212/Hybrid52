from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import spearmanr


@dataclass
class BinaryMetrics:
    accuracy: float
    f1: float
    auc: float
    ic: float


def compute_binary_metrics(raw_output: np.ndarray, labels: np.ndarray, returns: np.ndarray, threshold: float = 0.5) -> Tuple[BinaryMetrics, Dict]:
    """Compute standard binary metrics plus confidence buckets."""

    probs = 1 / (1 + np.exp(-raw_output))
    preds = (probs > threshold).astype(int)

    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average='binary'))

    try:
        auc = float(roc_auc_score(labels, raw_output))
    except Exception:
        auc = 0.5

    ic, _ = spearmanr(raw_output, returns)
    if np.isnan(ic):
        ic = 0.0

    confidence = np.abs(probs - 0.5) * 2
    buckets = {}
    for thr in [0.0, 0.2, 0.4, 0.6, 0.8]:
        mask = confidence >= thr
        if mask.sum() > 50:
            buckets[f'conf>={thr:.1f}'] = {
                'accuracy': round(float(accuracy_score(labels[mask], preds[mask])), 4),
                'f1': round(float(f1_score(labels[mask], preds[mask], average='binary')), 4),
                'coverage': round(float(mask.mean()), 4),
                'n': int(mask.sum()),
            }

    return BinaryMetrics(accuracy=acc, f1=f1, auc=auc, ic=float(ic)), buckets


def sweep_threshold_for_f1(probs: np.ndarray, labels: np.ndarray, lo: float = 0.30, hi: float = 0.65, step: float = 0.01) -> Tuple[float, float]:
    best_thr = 0.5
    best_f1 = -1.0

    thr = lo
    while thr <= hi + 1e-12:
        preds = (probs > thr).astype(int)
        f1 = float(f1_score(labels, preds, average='binary'))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
        thr += step

    return best_thr, best_f1
