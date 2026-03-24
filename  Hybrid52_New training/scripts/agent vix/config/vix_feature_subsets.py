"""
VIX Feature Subset Configuration for Agent VIX

Agent VIX operates on a SEPARATE feature space from the 325-dim option chain vector.
Its features are built from VIX/VIXW data resampled to 5-min bars.

This file:
  1. Defines the VIX feature schema (~10 features)
  2. Adds Agent V entry to AGENT_FEATURE_SUBSETS for compatibility
  3. Provides helper functions for VIX feature indexing

Integration:
  - Import and call `register_vix_agent()` at startup to inject V into the
    existing AGENT_FEATURE_SUBSETS dict from config/feature_subsets.py
"""

# ============================================================================
# VIX Feature Schema (5-min resolution)
# ============================================================================

VIX_FEAT_DIM = 10

VIX_FEATURE_NAMES = [
    'vix_level',         # 0: Current VIX spot price
    'vix_pct_5m',        # 1: 5-min percentage change
    'vix_pct_15m',       # 2: 15-min percentage change (3 bars back)
    'vix_pct_1h',        # 3: 1-hour percentage change (12 bars back)
    'vix_zscore_15m',    # 4: 15-min z-score vs 20-bar rolling
    'vix_percentile_1h', # 5: 1-hour percentile rank (12-bar window)
    'vix_term_slope',    # 6: VIX term structure slope (M2-M1)/M1
    'vvix_level',        # 7: Vol-of-vol (VVIX or proxy)
    'vix_vix1d_spread',  # 8: VIX - VIX1D spread
    'vix_hilo_range',    # 9: 5-min high-low range (normalized)
]

VIX_FEATURE_INDEX = {name: idx for idx, name in enumerate(VIX_FEATURE_NAMES)}


# ============================================================================
# Agent V Configuration (compatible with AGENT_FEATURE_SUBSETS format)
# ============================================================================

AGENT_V_CONFIG = {
    'name': 'VIX Regime Agent',
    'ranges': [],           # Does NOT use the 325-dim option vector
    'feat_dim': VIX_FEAT_DIM,
    'use_backbone': False,  # Static MLP, no temporal backbone
    'separate_input': True, # Flag: uses its own feature pipeline (5-min VIX)
    'update_cadence': '5min',
}


def register_vix_agent(agent_feature_subsets: dict) -> dict:
    """
    Register Agent V (VIX) in the existing AGENT_FEATURE_SUBSETS dict.

    Usage:
        from config.feature_subsets import AGENT_FEATURE_SUBSETS
        from config.vix_feature_subsets import register_vix_agent
        register_vix_agent(AGENT_FEATURE_SUBSETS)

    Args:
        agent_feature_subsets: The AGENT_FEATURE_SUBSETS dict from feature_subsets.py

    Returns:
        Updated dict with 'V' entry added
    """
    agent_feature_subsets['V'] = AGENT_V_CONFIG.copy()
    return agent_feature_subsets


def get_vix_feature_names() -> list:
    """Return ordered list of VIX feature names."""
    return VIX_FEATURE_NAMES.copy()


def get_vix_feature_dim() -> int:
    """Return VIX feature dimension."""
    return VIX_FEAT_DIM


# ============================================================================
# VIX Regime Label Definitions
# ============================================================================

REGIME_THRESHOLDS = [15.0, 20.0, 25.0, 35.0]
REGIME_NAMES = ['CALM', 'NORMAL', 'ELEVATED', 'HIGH', 'EXTREME']
NUM_REGIMES = len(REGIME_NAMES)


def vix_to_regime_index(vix_level: float) -> int:
    """Convert VIX level to regime class index."""
    for i, threshold in enumerate(REGIME_THRESHOLDS):
        if vix_level < threshold:
            return i
    return len(REGIME_THRESHOLDS)  # EXTREME


# ============================================================================
# Validation
# ============================================================================

if __name__ == '__main__':
    print("VIX Feature Subset Configuration")
    print("=" * 60)
    print(f"\nVIX Feature Dimension: {VIX_FEAT_DIM}")
    print(f"\nFeatures:")
    for idx, name in enumerate(VIX_FEATURE_NAMES):
        print(f"  [{idx:2d}] {name}")

    print(f"\nAgent V Config:")
    for k, v in AGENT_V_CONFIG.items():
        print(f"  {k}: {v}")

    print(f"\nRegime Thresholds:")
    for i, name in enumerate(REGIME_NAMES):
        low = REGIME_THRESHOLDS[i - 1] if i > 0 else 0.0
        high = REGIME_THRESHOLDS[i] if i < len(REGIME_THRESHOLDS) else 999.0
        print(f"  {name}: VIX {low:.0f} – {high:.0f}")

    # Test regime labeling
    print(f"\nSample regime labels:")
    for vix in [10.0, 12.5, 16.0, 18.5, 22.0, 28.0, 32.0, 40.0, 55.0]:
        idx = vix_to_regime_index(vix)
        print(f"  VIX={vix:5.1f} → {REGIME_NAMES[idx]} ({idx})")

    # Test registration
    mock_subsets = {'A': {}, 'B': {}}
    register_vix_agent(mock_subsets)
    assert 'V' in mock_subsets, "Registration failed"
    print(f"\n✓ Agent V registration: OK (keys: {list(mock_subsets.keys())})")
