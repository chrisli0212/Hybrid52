"""
Agent A Validator — unit-test helpers and data sanity checks.

Usage:
    from hybrid55_preprocessing.agents.agent_a.validator import validate_agent_a
    report = validate_agent_a(hist_df, oi_df)
    print(report)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from .extractor import AgentAExtractor
from .feature_config import AGENT_A_DIM, AGENT_A_FEATURES, REQUIRED_GREEK_COLS


def validate_agent_a(
    hist_df: pd.DataFrame,
    oi_df: Optional[pd.DataFrame] = None,
    underlying_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run Agent A extractor and return a validation report dict.

    Returns:
        {
          'shape_ok': bool,
          'nan_count': int,
          'zero_rate': float,
          'missing_required_cols': list[str],
          'alerts': list[dict],
          'features': np.ndarray,
          'feature_names': list[str],
        }
    """
    extractor = AgentAExtractor()

    missing_cols = [c for c in REQUIRED_GREEK_COLS if c not in hist_df.columns]

    features = extractor.extract(hist_df, oi_df, underlying_price)

    report = {
        "shape_ok": features.shape == (AGENT_A_DIM,),
        "expected_dim": AGENT_A_DIM,
        "actual_dim": features.shape[0],
        "nan_count": int(np.isnan(features).sum()),
        "zero_rate": float((features == 0).mean()),
        "missing_required_cols": missing_cols,
        "alerts": extractor.alert_log,
        "features": features,
        "feature_names": extractor.get_feature_names(),
    }

    return report


def print_validation_report(report: Dict[str, Any]) -> None:
    print("=" * 60)
    print("AGENT A VALIDATION REPORT")
    print("=" * 60)
    print(f"  Shape OK        : {report['shape_ok']} ({report['actual_dim']} features)")
    print(f"  NaN count       : {report['nan_count']}")
    print(f"  Zero rate       : {report['zero_rate']:.1%}")
    if report["missing_required_cols"]:
        print(f"  Missing cols    : {report['missing_required_cols']}")
    else:
        print("  Missing cols    : none")
    if report["alerts"]:
        print(f"  Alerts ({len(report['alerts'])})     :")
        for a in report["alerts"]:
            print(f"    [{a['ts']}] {a['msg']}")
    else:
        print("  Alerts          : none")
    print("=" * 60)
