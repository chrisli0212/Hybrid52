"""
Feature Subsetting Configuration for Agent Diversity — Hybrid52

Each agent sees a different subset of the 286-dim master feature tensor
(feature_config_v2.py layout) to force genuine specialisation.

286-dim historical layout (from feature_config_v2.py):
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

Data reliability guide for range selection:
  HIGH   : Greeks (0-74), IV surface (125-149), OI/walls (200-219), cross-strike (220-234), time_decay (235-249), CSV_DERIVED (270-285)
  MEDIUM : Gamma/Vanna (75-124), microstructure spread/bid_ask (180-199), sentiment (250-269)
  LOW    : Flow/volume (150-179) — near-empty in EOD snapshots, AVOIDED

Agent expanded dim targets (match patched agent constructors):
  A  → input_dim = 130  (generalist: Greeks + IV + walls + CSV-derived)
  B  → input_dim = 75   (seq specialist: gamma + vanna/charm + cross-strike + time_decay; static_dim=130 from A)
  C  → input_dim = 69   (seq specialist: microstructure + sentiment + time_decay + walls; static_dim=130 from A)
  K  → input_dim = 75   (pure Greek MLP: GREEK_BY_STRIKE full)
  T  → trade_feat_dim = 40  (microstructure + walls_positioning full)
  Q  → quote_feat_dim = 35  (IV surface + vanna/charm buckets)
  2D → feat_dim = 0     (uses chain_2d tensor, not flat vector)

NOTE — Agent B and C static context:
  Agents B and C have a separate static_dim input (Agent A's feature set, 130 dims).
  The ranges below define ONLY the seq input dims for B and C.
  The static vector is extracted separately in independent_agent.py using Agent A's ranges.
"""

TOTAL_FEAT_DIM = 286

AGENT_FEATURE_SUBSETS = {
    'A': {
        'name': 'Alpha (Generalist Greek/IV/Positioning)',
        'ranges': [
            (0, 75),     # GREEK_BY_STRIKE full (bucket+ATM+skew) = 75
            (125, 150),  # IV_SURFACE full = 25
            (200, 214),  # WALLS: max_gamma_strikes + max_oi_strikes + wall_distances = 14
            (270, 286),  # CSV_DERIVED full = 16
        ],
        'feat_dim': 130,
        'use_backbone': True,
    },

    'B': {
        'name': 'Beta (BiLSTM Temporal Sequence)',
        'ranges': [
            (75, 105),   # GAMMA_EXPOSURE full = 30
            (105, 125),  # VANNA_CHARM full = 20
            (220, 235),  # CROSS_STRIKE full = 15
            (235, 245),  # TIME_DECAY: dte_buckets + decay_accelerations = 10
        ],
        'feat_dim': 75,
        'use_backbone': False,
    },

    'C': {
        'name': 'Chi (CNN+Attention Sequence)',
        'ranges': [
            (180, 200),  # MICROSTRUCTURE full = 20
            (250, 270),  # SENTIMENT_REGIME full = 20
            (235, 250),  # TIME_DECAY full = 15
            (200, 214),  # WALLS: max_gamma + max_oi + wall_distances = 14
        ],
        'feat_dim': 69,
        'use_backbone': True,
    },

    'K': {
        'name': 'Kappa (Pure Greek MLP)',
        'ranges': [
            (0, 75),     # GREEK_BY_STRIKE full = 75
        ],
        'feat_dim': 75,
        'use_backbone': False,
    },

    'T': {
        'name': 'Tau (Microstructure + Positioning)',
        'ranges': [
            (180, 200),  # MICROSTRUCTURE full = 20
            (200, 220),  # WALLS_POSITIONING full = 20
        ],
        'feat_dim': 40,
        'use_backbone': False,
    },

    'Q': {
        'name': 'Quote (IV + Vanna/Charm Dynamics)',
        'ranges': [
            (125, 150),  # IV_SURFACE full = 25
            (105, 115),  # VANNA/CHARM: vanna_by_bucket + charm_by_bucket = 10
        ],
        'feat_dim': 35,
        'use_backbone': False,
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

_EXPECTED_DIMS = {'A': 130, 'B': 75, 'C': 69, 'K': 75, 'T': 40, 'Q': 35, '2D': 0}
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


def build_feature_mask(agent_type: str, total_dim: int = TOTAL_FEAT_DIM):
    """Return a boolean mask of shape (total_dim,) for selecting features."""
    import numpy as np
    mask = np.zeros(total_dim, dtype=bool)
    for start, end in AGENT_FEATURE_SUBSETS[agent_type.upper()]['ranges']:
        mask[start:end] = True
    return mask


def print_overlap_matrix():
    """Print pairwise feature overlap between agents (for verification)."""
    agents = ['A', 'B', 'K', 'C', 'T', 'Q']
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
    print("Feature Subset Configuration — Hybrid52 (Expanded)")
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
