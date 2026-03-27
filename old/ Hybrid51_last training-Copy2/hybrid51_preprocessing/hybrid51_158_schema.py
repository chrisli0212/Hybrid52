"""
Hybrid51 158-d feature schema for preprocessing alignment.

Re-exports FEATURE_INDICES and FEATURE_SCHEMA from hybrid51_config so preprocessing
pipelines can target 158-d output for Hybrid51 training and inference.
"""

FEATURE_DIM = 158

try:
    from hybrid51_config import FEATURE_INDICES, FEATURE_SCHEMA
except ImportError:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from hybrid51_config import FEATURE_INDICES, FEATURE_SCHEMA


def get_expected_158_column_names() -> list:
    """Return ordered list of 158 feature names matching FEATURE_INDICES (orats 70, theta 70, sentiment 10, regime 8)."""
    names = []
    # orats: flatten in schema order, take first 70
    for group_name, feat_list in FEATURE_SCHEMA["orats"].items():
        names.extend(feat_list)
    names = names[: FEATURE_INDICES["orats"][1] - FEATURE_INDICES["orats"][0]]  # 70
    # theta_options: 70
    names.extend(FEATURE_SCHEMA["theta_options"]["derived"])
    # sentiment: 10
    names.extend(FEATURE_SCHEMA["sentiment"]["features"])
    # regime: 8
    names.extend(FEATURE_SCHEMA["regime"]["features"])
    assert len(names) == FEATURE_DIM, f"Expected {FEATURE_DIM} names, got {len(names)}"
    return names


__all__ = ["FEATURE_DIM", "FEATURE_INDICES", "FEATURE_SCHEMA", "get_expected_158_column_names"]
