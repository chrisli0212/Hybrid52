"""
Feature Subsetting Configuration for Agent Diversity

Each agent sees a different subset of the 325 features to force genuine specialization.
Feature groups from feature_config_v2.py:
  - Core Greeks:        dims 0–49
  - IV Surface:         dims 50–99
  - Term Structure:     dims 100–127
  - Flow & Volume:      dims 128–149
  - Microstructure:     dims 150–179
  - Sentiment/Regime:   dims 180–209
  - Cross-Strike-Time:  dims 210–239
  - Gamma Exposure:     dims 240–269
  - Smart Money:        dims 270–284
  - Volume Anomaly:     dims 285–296
  - Trade Conditions:   dims 297–306
  - Quote Pressure:     dims 307–324

Design Principle:
  - Each agent should see features most relevant to its architecture
  - Overlap is intentionally limited to ~30% between any two agents
  - Agent T and Q see ONLY their specialized features (no backbone needed)
  - Agent 2D uses chain_2d data (separate input), not the 325-dim vector
"""

# Full feature dimension
TOTAL_FEAT_DIM = 325

# Feature subset definitions per agent type
# Each maps to a list of (start, end) ranges or individual indices
AGENT_FEATURE_SUBSETS = {
    # Agent A (Alpha): Generalist — Greeks + IV Surface + Gamma + Sentiment
    # Broadest view but excludes trade/quote specifics
    'A': {
        'name': 'Alpha (Generalist)',
        'ranges': [(0, 50), (50, 100), (180, 210), (240, 270)],
        'feat_dim': 160,   # 50 + 50 + 30 + 30
        'use_backbone': True,
    },

    # Agent B (Beta): Temporal — Greeks + IV Surface + Term Structure
    # use_backbone=False: AgentB has its own stacked BiLSTM; outer TemporalBackbone is redundant
    'B': {
        'name': 'Beta (Temporal)',
        'ranges': [(50, 128), (210, 240)],
        'feat_dim': 108,
        'use_backbone': False,
    },

    # Agent K (Greeks): Static MLP — Core Greeks + Term Structure only
    # Pure Greek specialist, no temporal backbone needed
    'K': {
        'name': 'Kappa (Greek Specialist)',
        'ranges': [(0, 50), (100, 128)],
        'feat_dim': 78,
        'use_backbone': False,  # K uses its own static MLP
    },

    # Agent C (Attention): Flow + Microstructure + Sentiment
    # CNN+Attention good for detecting regime shifts and flow patterns
    'C': {
        'name': 'Chi (Market Activity)',
        'ranges': [(128, 180), (180, 210), (240, 270)],
        'feat_dim': 112,
        'use_backbone': True,
    },

    # Agent T (Trade Flow): Trade-specific features ONLY
    # Direct specialist — no shared backbone, uses its own input norm
    'T': {
        'name': 'Tau (Trade Flow + Context)',
        'ranges': [
            (0, 50),     # Core Greeks (delta, gamma context)
            (128, 150),  # Flow & Volume (complementary)
            (180, 210),  # Sentiment/Regime (regime awareness)
            (270, 307),  # Smart Money + Volume Anomaly + Trade Conditions
        ],
        'feat_dim': 139,
        'use_backbone': False,
    },

    # Agent Q (Quote Dynamics)
    'Q': {
        'name': 'Quote (Quote Dynamics + Context)',
        'ranges': [
            (50, 100),   # IV Surface (spread relates to IV)
            (150, 180),  # Microstructure (complementary)
            (180, 210),  # Sentiment/Regime
            (307, 325),  # Quote Pressure
        ],
        'feat_dim': 128,
        'use_backbone': False,
    },

    # Agent 2D (Chain Shape): Uses chain_2d data, not the flat vector
    '2D': {
        'name': '2D (Chain Shape)',
        'ranges': [],       # Uses chain_2d tensor, not flat features
        'feat_dim': 0,      # N/A — uses (n_greeks, n_strikes, n_timesteps)
        'use_backbone': False,
    },
}

# Ensure feat_dim entries match their ranges
for _agent_type, _cfg in AGENT_FEATURE_SUBSETS.items():
    _cfg['feat_dim'] = sum(end - start for start, end in _cfg.get('ranges', []))


def get_feature_indices(agent_type: str):
    """Return flat list of feature indices for a given agent type."""
    config = AGENT_FEATURE_SUBSETS[agent_type.upper()]
    indices = []
    for start, end in config['ranges']:
        indices.extend(range(start, end))
    return indices


def get_feature_dim(agent_type: str) -> int:
    """Return the feature dimension for a given agent type."""
    config = AGENT_FEATURE_SUBSETS[agent_type.upper()]
    return sum(end - start for start, end in config['ranges'])


def build_feature_mask(agent_type: str, total_dim: int = TOTAL_FEAT_DIM):
    """Return a boolean mask of shape (total_dim,) for selecting features."""
    import numpy as np
    mask = np.zeros(total_dim, dtype=bool)
    for start, end in AGENT_FEATURE_SUBSETS[agent_type.upper()]['ranges']:
        mask[start:end] = True
    return mask


# Overlap analysis (for debugging/verification)
def print_overlap_matrix():
    """Print pairwise feature overlap between agents."""
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
                overlap = 0
            else:
                overlap = len(idx1 & idx2) / min(len(idx1), len(idx2))
            print(f"{overlap:>6.0%}", end='')
        print()


if __name__ == '__main__':
    print("Feature Subset Configuration")
    print("=" * 60)
    for agent_type, config in AGENT_FEATURE_SUBSETS.items():
        if agent_type == '2D':
            continue
        indices = get_feature_indices(agent_type)
        print(f"\nAgent {agent_type} ({config['name']})")
        print(f"  Features: {len(indices)} dims")
        print(f"  Ranges: {config['ranges']}")
        print(f"  Use backbone: {config['use_backbone']}")
    
    print("\n" + "=" * 60)
    print("Pairwise Feature Overlap Matrix")
    print("=" * 60)
    print_overlap_matrix()
