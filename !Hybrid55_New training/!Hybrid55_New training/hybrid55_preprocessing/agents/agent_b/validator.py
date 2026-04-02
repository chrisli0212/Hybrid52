"""
Agent B Validator — unit-test helpers and data sanity checks.

Usage:
    from hybrid55_preprocessing.agents.agent_b.validator import validate_agent_b
    report = validate_agent_b(greek_df, trade_df, ohlc_df)
    print(report)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from .extractor import AgentBExtractor
from .feature_config import AGENT_B_BASE_DIM, AGENT_B_LIVE_DIM, AGENT_B_REGISTRY


def validate_agent_b(
    greek_df: pd.DataFrame,
    trade_df: Optional[pd.DataFrame] = None,
    ohlc_df: Optional[pd.DataFrame] = None,
    historical_mode: bool = True,
) -> Dict[str, Any]:
    """
    Run Agent B extractor and return a per-block validation report.

    Returns dict with:
      - shape_ok, nan_count, zero_rate
      - per_block_zero_rates: {block_name: zero_rate}
      - alerts, features, feature_names
    """
    extractor = AgentBExtractor(historical_mode=historical_mode)
    features  = extractor.extract(greek_df, trade_df, ohlc_df)

    per_block = {}
    for block in AGENT_B_REGISTRY:
        block_feats = features[block.start:block.end]
        per_block[block.name] = {
            "zero_rate": float((block_feats == 0).mean()),
            "nan_count": int(np.isnan(block_feats).sum()),
            "mean": float(block_feats.mean()),
            "std":  float(block_feats.std()),
        }

    expected_dim = AGENT_B_BASE_DIM if historical_mode else AGENT_B_LIVE_DIM
    return {
        "shape_ok":         features.shape == (expected_dim,),
        "expected_dim":     expected_dim,
        "actual_dim":       features.shape[0],
        "nan_count":        int(np.isnan(features).sum()),
        "zero_rate":        float((features == 0).mean()),
        "per_block":        per_block,
        "alerts":           extractor.alert_log,
        "features":         features,
        "feature_names":    extractor.get_feature_names(),
    }


def print_validation_report(report: Dict[str, Any]) -> None:
    print("=" * 60)
    print("AGENT B VALIDATION REPORT")
    print("=" * 60)
    print(f"  Shape OK   : {report['shape_ok']} ({report['actual_dim']} features)")
    print(f"  NaN count  : {report['nan_count']}")
    print(f"  Zero rate  : {report['zero_rate']:.1%}")
    print("\n  Per-block zero rates:")
    for name, stats in report["per_block"].items():
        flag = " ⚠️  HIGH ZEROS" if stats["zero_rate"] >= 0.50 else ""
        print(f"    {name:<22}: {stats['zero_rate']:.1%}{flag}")
    if report["alerts"]:
        print(f"\n  Alerts ({len(report['alerts'])}):")
        for a in report["alerts"]:
            print(f"    [{a['ts']}] {a['msg']}")
    print("=" * 60)
