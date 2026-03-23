from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


STAGE3_ROOT = Path("/workspace/Hybrid51/5. hybrid51_stage3")
if str(STAGE3_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE3_ROOT))
if str(STAGE3_ROOT / "hybrid51_preprocessing") not in sys.path:
    sys.path.insert(0, str(STAGE3_ROOT / "hybrid51_preprocessing"))

from hybrid51_preprocessing.master_extractor_v2 import MasterFeatureExtractor  # type: ignore  # noqa: E402
from hybrid51_preprocessing.chain_2d import Chain2DProcessor  # type: ignore  # noqa: E402


class LiveTrainingFeaturePort:
    """
    Port of the original training feature-construction logic for live usage.

    Uses the same stage3 preprocessing extractor to construct 325 flat features.
    """

    def __init__(self) -> None:
        self.extractor = MasterFeatureExtractor(
            include_chain_2d=False,  # 2D sequence is assembled from per-batch slices below.
            include_phase1=True,
            normalize=False,
        )
        self.chain_processor = Chain2DProcessor.from_config()

    def adapt_snapshot_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()

        rename_map = {
            "bid_quote": "bid",
            "ask_quote": "ask",
            "timestamp_trade": "trade_timestamp",
            "timestamp_quote": "quote_timestamp",
            "spot": "underlying_price",
        }
        for old, new in rename_map.items():
            if old in out.columns and new not in out.columns:
                out = out.rename(columns={old: new})

        # Training code expects C/P, while live fetcher stores CALL/PUT.
        if "right" in out.columns:
            out["right"] = (
                out["right"]
                .astype(str)
                .str.upper()
                .replace({"CALL": "C", "PUT": "P"})
            )

        # Fill missing trade/quote columns used by phase1 extractors.
        defaults: Dict[str, object] = {
            "condition": 18,
            "exchange": "C",
            "sequence": 0,
            "size": np.nan,
            "price": np.nan,
            "bid_size": np.nan,
            "ask_size": np.nan,
            "bid_exchange": "C",
            "ask_exchange": "C",
        }
        for c, default in defaults.items():
            if c not in out.columns:
                out[c] = default

        if out["size"].isna().all():
            if "volume" in out.columns:
                out["size"] = pd.to_numeric(out["volume"], errors="coerce")
            elif "count" in out.columns:
                out["size"] = pd.to_numeric(out["count"], errors="coerce")
            else:
                out["size"] = 0.0

        # Ensure numeric features are coercible.
        numeric_candidates = [
            "strike",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
            "epsilon",
            "lambda",
            "vanna",
            "charm",
            "vomma",
            "veta",
            "zomma",
            "color",
            "ultima",
            "dual_delta",
            "dual_gamma",
            "d1",
            "d2",
            "implied_vol",
            "iv_error",
            "underlying_price",
            "bid",
            "ask",
            "price",
            "size",
            "bid_size",
            "ask_size",
            "oi",
        ]
        for c in numeric_candidates:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        for tc in ("trade_timestamp", "quote_timestamp"):
            if tc in out.columns:
                out[tc] = pd.to_datetime(out[tc], errors="coerce")

        return out

    def extract_feature_vector(self, adapted_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        if adapted_df is None or adapted_df.empty:
            return np.zeros(325, dtype=np.float32), 0.0
        oi_val = None
        if "oi" in adapted_df.columns:
            oi_val = float(pd.to_numeric(adapted_df["oi"], errors="coerce").fillna(0.0).sum())
        result = self.extractor.extract(greek_df=adapted_df, trade_df=adapted_df, open_interest=oi_val)
        feat = np.asarray(result.features, dtype=np.float32)
        if feat.shape[0] != 325:
            if feat.shape[0] < 325:
                feat = np.pad(feat, (0, 325 - feat.shape[0]), constant_values=0.0)
            else:
                feat = feat[:325]
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat, float(result.quality_score)

    def extract_chain_slice30(self, adapted_df: pd.DataFrame) -> np.ndarray:
        if adapted_df is None or adapted_df.empty:
            return np.zeros((5, 30), dtype=np.float32)
        s = self.chain_processor.snapshot_to_slice(adapted_df)
        s = np.asarray(s, dtype=np.float32)
        if s.shape != (5, 30):
            fixed = np.zeros((5, 30), dtype=np.float32)
            rows = min(5, s.shape[0]) if s.ndim >= 2 else 0
            cols = min(30, s.shape[1]) if s.ndim >= 2 else 0
            if rows > 0 and cols > 0:
                fixed[:rows, :cols] = s[:rows, :cols]
            s = fixed
        return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def center_crop_30_to_20(slice_30: np.ndarray) -> np.ndarray:
        # Same center crop convention as build_tier3_binary.py (indices 5:25).
        start = 5
        return slice_30[:, start : start + 20]
