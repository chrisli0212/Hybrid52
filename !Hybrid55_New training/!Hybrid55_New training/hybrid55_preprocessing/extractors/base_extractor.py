"""
BaseExtractor — abstract base class for all shared extractors.

All extractors (greek_features, gamma_exposure, iv_surface, etc.) inherit from this.
Provides:
  - safe_extract(): wraps extract() with try/except + zero fallback + alert logging
  - check_zeros(): warns when >50% of features are zero
  - alert_log: list of alert dicts for downstream inspection
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any

logger = logging.getLogger("hybrid55.extractor")


class BaseExtractor(ABC):
    """Abstract base for all shared feature extractors."""

    # Subclasses must declare expected output size
    N_FEATURES: int = 0
    GROUP_NAME: str = "unknown"

    def __init__(self):
        self.alert_log: List[Dict[str, Any]] = []

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from a DataFrame snapshot.
        Must return np.ndarray of shape (N_FEATURES,).
        """
        raise NotImplementedError

    def safe_extract(self, df: pd.DataFrame) -> tuple[np.ndarray, bool]:
        """
        Safe wrapper around extract().
        Returns (features, failed) where failed=True if extraction errored.
        On failure, returns zeros and logs an alert.
        """
        try:
            result = self.extract(df)
            if result.shape[0] != self.N_FEATURES:
                self._alert(
                    f"[SIZE MISMATCH] {self.GROUP_NAME}: "
                    f"expected {self.N_FEATURES}, got {result.shape[0]}"
                )
                return np.zeros(self.N_FEATURES, dtype=np.float32), True
            self._check_zeros(result)
            return result.astype(np.float32), False
        except Exception as e:
            self._alert(f"[EXTRACTOR FAIL] {self.GROUP_NAME}: {e}")
            return np.zeros(self.N_FEATURES, dtype=np.float32), True

    def _check_zeros(self, features: np.ndarray, threshold: float = 0.50) -> None:
        """Alert if >= threshold fraction of features are zero."""
        if self.N_FEATURES == 0:
            return
        zero_rate = float((features == 0).mean())
        if zero_rate >= threshold:
            self._alert(
                f"[ZERO ALERT] {self.GROUP_NAME}: "
                f"{zero_rate:.1%} zero ({int(zero_rate * self.N_FEATURES)}/{self.N_FEATURES})"
            )

    def _alert(self, msg: str) -> None:
        """Log an alert both to Python logging and internal alert_log."""
        logger.warning(msg)
        self.alert_log.append({"ts": pd.Timestamp.now().isoformat(), "msg": msg})

    def clear_alerts(self) -> None:
        self.alert_log.clear()
