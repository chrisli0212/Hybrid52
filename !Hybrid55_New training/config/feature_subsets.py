"""
Feature Subsetting Configuration for Agent Diversity — Hybrid55

Each agent sees a different subset of the 311-dim master feature tensor
(feature_config_v2.py layout) to force genuine specialisation.

311-dim historical layout (from feature_config_v2.py):
  GREEK_BY_STRIKE   :   0 –  74  (75 dims)  subgroups: bucket_greeks 0-65, atm_greeks 65-72, skew 72-75
  GAMMA_EXPOSURE    :  75 – 104  (30 dims)  subgroups: gamma_by_strike 75-95, net_gamma 95-99, dealer 99-102, zones 102-105
  VANNA_CHARM       : 105 – 124  (20 dims)  subgroups: vanna_by_bucket 105-110, charm_by_bucket 110-115, net 115-121, ratios 121-125
  IV_SURFACE        : 125 – 149  (25 dims)
  FLOW_VOLUME       : 150 – 179  (30 dims)  — mostly empty (volume/vwap near-zero in EOD snapshots)
  MICROSTRUCTURE    : 180 – 199  (20 dims)  — spread/bid_ask available; trade_velocity near-zero
  WALLS_POSITIONING : 200 – 219  (20 dims)  subgroups: max_gamma_strikes 200-202, max_oi_strikes 202-206, wall_distances 206-214, dealer 214-220
  CROSS_STRIKE      : 220 – 234  (15 dims)
  TIME_DECAY        : 235 – 249  (15 dims)
  SENTIMENT_REGIME  : 250 – 269  (20 dims)
  CSV_DERIVED       : 270 – 285  (16 dims)  — lambda/dist_atm/spread_pct + dte/cp_sign/mid/spread/iv_std + OI enrichments
  OHLC_DYNAMICS     : 286 – 310  (25 dims)  — sparse-safe OHLC chain block

Data reliability guide for range selection:
  HIGH   : Greeks (0-74), IV surface (125-149), OI/walls (200-219), cross-strike (220-234), time_decay (235-249), CSV_DERIVED (270-285)
  MEDIUM : Gamma/Vanna (75-124), microstructure spread/bid_ask (180-199), sentiment (250-269)
  LOW    : Flow/volume (150-179) — near-empty in EOD snapshots, AVOIDED

Agent expanded dim targets (match patched agent constructors):
  A  → input_dim = 130  (generalist: Greeks + IV + walls + CSV-derived)
  B  → input_dim = 75   (seq specialist: gamma + vanna/charm + cross-strike + time_decay; static_dim=130 from A)
  C  → input_dim = 69   (seq specialist: microstructure + sentiment + time_decay + walls; static_dim=130 from A)
  K  → input_dim = 75   (pure Greek MLP: GREEK_BY_STRIKE full)
  TQ → tq_feat_dim = 70     (medium-smoke robust channels: vanna/charm + flow + micro + csv_derived)
  H  → input_dim = 11       (medium-smoke live OHLC subset)
  2D → feat_dim = 0     (uses chain_2d tensor, not flat vector)

NOTE — Agent B and C static context:
  Agents B and C have a separate static_dim input (Agent A's feature set, 130 dims).
  The ranges below define ONLY the seq input dims for B and C.
  The static vector is extracted separately in independent_agent.py using Agent A's ranges.
"""

from hybrid55_preprocessing.feature_config_v2 import (
    TOTAL_FEATURES,
    FEATURE_SCHEMA_VERSION,
    FeatureGroup,
    FEATURE_GROUPS,
)

TOTAL_FEAT_DIM = TOTAL_FEATURES
SUPPORTED_SCHEMA_VERSION = "hybrid55_v1_live_raw_guarded_311"


def _full_group(group: FeatureGroup) -> tuple[int, int]:
    cfg = FEATURE_GROUPS[group]
    return (cfg.start_idx, cfg.end_idx)


def _sub_group(group: FeatureGroup, key: str) -> tuple[int, int]:
    return FEATURE_GROUPS[group].subgroups[key]

AGENT_FEATURE_SUBSETS = {
    'A': {
        'name': 'Alpha (Generalist Greek/IV/Positioning)',
        'ranges': [
            _full_group(FeatureGroup.GREEK_BY_STRIKE),
            _full_group(FeatureGroup.IV_SURFACE),
            _sub_group(FeatureGroup.WALLS_POSITIONING, "max_gamma_strikes"),
            _sub_group(FeatureGroup.WALLS_POSITIONING, "max_oi_strikes"),
            _sub_group(FeatureGroup.WALLS_POSITIONING, "wall_distances"),
            _full_group(FeatureGroup.CSV_DERIVED),
        ],
        'feat_dim': 130,
        'use_backbone': True,
    },

    'B': {
        'name': 'Beta (BiLSTM Temporal Sequence)',
        'ranges': [
            _full_group(FeatureGroup.GAMMA_EXPOSURE),
            _full_group(FeatureGroup.VANNA_CHARM),
            _full_group(FeatureGroup.CROSS_STRIKE),
            _sub_group(FeatureGroup.TIME_DECAY, "dte_buckets"),
            _sub_group(FeatureGroup.TIME_DECAY, "decay_accelerations"),
        ],
        'feat_dim': 75,
        'use_backbone': False,
    },

    'C': {
        'name': 'Chi (CNN+Attention Sequence)',
        'ranges': [
            _full_group(FeatureGroup.MICROSTRUCTURE),
            _full_group(FeatureGroup.SENTIMENT_REGIME),
            _full_group(FeatureGroup.TIME_DECAY),
            _sub_group(FeatureGroup.WALLS_POSITIONING, "max_gamma_strikes"),
            _sub_group(FeatureGroup.WALLS_POSITIONING, "max_oi_strikes"),
            _sub_group(FeatureGroup.WALLS_POSITIONING, "wall_distances"),
        ],
        'feat_dim': 69,
        'use_backbone': True,
    },

    'K': {
        'name': 'Kappa (Pure Greek MLP)',
        'ranges': [
            _full_group(FeatureGroup.GREEK_BY_STRIKE),
        ],
        'feat_dim': 75,
        'use_backbone': False,
    },

    'TQ': {
        'name': 'TauQuote (Unified Trade/Quote Specialist)',
        'ranges': [
            _full_group(FeatureGroup.VANNA_CHARM),
            _sub_group(FeatureGroup.FLOW_VOLUME, "call_put_ratios"),
            _sub_group(FeatureGroup.FLOW_VOLUME, "volume_by_aggression"),
            _sub_group(FeatureGroup.FLOW_VOLUME, "size_distribution"),
            _full_group(FeatureGroup.MICROSTRUCTURE),
            _full_group(FeatureGroup.CSV_DERIVED),
        ],
        'feat_dim': 70,
        'use_backbone': False,
    },

    'H': {
        'name': 'HorizonOHLC (OHLC Specialist)',
        'ranges': [
            (FEATURE_GROUPS[FeatureGroup.OHLC_DYNAMICS].start_idx, FEATURE_GROUPS[FeatureGroup.OHLC_DYNAMICS].start_idx + 1),
            (FEATURE_GROUPS[FeatureGroup.OHLC_DYNAMICS].start_idx + 3, FEATURE_GROUPS[FeatureGroup.OHLC_DYNAMICS].start_idx + 12),
            (FEATURE_GROUPS[FeatureGroup.OHLC_DYNAMICS].start_idx + 22, FEATURE_GROUPS[FeatureGroup.OHLC_DYNAMICS].start_idx + 23),
        ],
        'feat_dim': 11,
        'use_backbone': False,
    },

    'M': {
        'name': 'Mu (Mixer Temporal Specialist)',
        'ranges': [
            _full_group(FeatureGroup.GREEK_BY_STRIKE),
            _full_group(FeatureGroup.IV_SURFACE),
            _full_group(FeatureGroup.CROSS_STRIKE),
            _full_group(FeatureGroup.TIME_DECAY),
            _full_group(FeatureGroup.CSV_DERIVED),
        ],
        'feat_dim': 146,
        'use_backbone': True,
    },

    '2D': {
        'name': '2D (Chain Surface CNN)',
        'ranges': [],
        'feat_dim': 0,
        'use_backbone': False,
    },
}

for _agent_type, _cfg in AGENT_FEATURE_SUBSETS.items():
    _cfg['feat_dim'] = sum(end - start for start, end in _cfg.get('ranges', []))

_EXPECTED_DIMS = {'A': 130, 'B': 75, 'C': 69, 'K': 75, 'TQ': 70, 'H': 11, 'M': 146, '2D': 0}
for _agent_type, _expected in _EXPECTED_DIMS.items():
    _actual = AGENT_FEATURE_SUBSETS[_agent_type]['feat_dim']
    assert _actual == _expected, (
        f"Agent {_agent_type} feat_dim mismatch: ranges produce {_actual}, "
        f"patched constructor expects {_expected}. Fix ranges in feature_subsets.py."
    )


def get_feature_indices(agent_type: str):
    """Return flat list of feature indices for a given agent type."""
    config = AGENT_FEATURE_SUBSETS[agent_type.upper()]
    indices = []
    for start, end in config['ranges']:
        indices.extend(range(start, end))
    return indices


def get_feature_dim(agent_type: str) -> int:
    """Return the feature dimension for a given agent type."""
    return AGENT_FEATURE_SUBSETS[agent_type.upper()]['feat_dim']


def assert_supported_schema(schema_version: str) -> None:
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Incompatible feature schema: got '{schema_version}', "
            f"expected '{SUPPORTED_SCHEMA_VERSION}'. Update feature_subsets and checkpoints."
        )


def build_feature_mask(agent_type: str, total_dim: int = TOTAL_FEAT_DIM):
    """Return a boolean mask of shape (total_dim,) for selecting features."""
    import numpy as np
    mask = np.zeros(total_dim, dtype=bool)
    for start, end in AGENT_FEATURE_SUBSETS[agent_type.upper()]['ranges']:
        mask[start:end] = True
    return mask


def print_overlap_matrix():
    """Print pairwise feature overlap between agents (for verification)."""
    agents = ['A', 'B', 'K', 'C', 'TQ', 'H', 'M']
    print(f"{'':>6}", end='')
    for a in agents:
        print(f"{a:>6}", end='')
    print()

    for a1 in agents:
        idx1 = set(get_feature_indices(a1))
        print(f"{a1:>6}", end='')
        for a2 in agents:
            idx2 = set(get_feature_indices(a2))
            if len(idx1) == 0 or len(idx2) == 0:
                overlap = 0.0
            else:
                overlap = len(idx1 & idx2) / min(len(idx1), len(idx2))
            print(f"{overlap:>6.0%}", end='')
        print()


if __name__ == '__main__':
    print("Feature Subset Configuration — Hybrid55 (Expanded)")
    print("=" * 60)
    for agent_type, config in AGENT_FEATURE_SUBSETS.items():
        if agent_type == '2D':
            continue
        indices = get_feature_indices(agent_type)
        print(f"\nAgent {agent_type} ({config['name']})")
        print(f"  feat_dim : {len(indices)}")
        print(f"  ranges   : {config['ranges']}")
        print(f"  backbone : {config['use_backbone']}")

    print("\n" + "=" * 60)
    print("Pairwise Feature Overlap Matrix")
    print("=" * 60)
    print_overlap_matrix()
