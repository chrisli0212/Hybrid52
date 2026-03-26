"""
Feature Subsetting Configuration for Agent Diversity — Hybrid52

Each agent sees a different compact subset of the 286-dim master feature tensor
(feature_config_v2.py layout) to force genuine specialisation and match the
patched agent constructor input_dim values exactly.

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
  CSV_DERIVED       : 270 – 285  (16 dims)  — lambda/dist_atm/spread_pct + dual/d1d2/iv_error/ultima + OI enrichments

Data reliability guide for range selection:
  HIGH   : Greeks, IV surface, OI structure, walls/positioning, cross-strike, time_decay
  MEDIUM : Microstructure spread/bid_ask (180-189), sentiment (250-269), quote_pressure depth (317-319)
  LOW    : Flow/volume (150-179), smart_money (270-284), volume_anomaly (285-296), trade_conditions (297-306)

Agent compact dim targets (must match patched agent constructors):
  A  → input_dim = 53   (temporal + static Greek/IV/OI snapshot)
  B  → input_dim = 34   (seq feature dim for BiLSTM; static_dim=53 passed separately)
  C  → input_dim = 34   (seq feature dim for CNN+Attention; static_dim=53 passed separately)
  K  → input_dim = 53   (compact static MLP Greek specialist)
  T  → trade_feat_dim = 25  (remapped to available microstructure + quote dims)
  Q  → quote_feat_dim = 20  (quote pressure + microstructure extension)
  2D → feat_dim = 0     (uses chain_2d tensor, not flat 325-dim vector)

NOTE — Agent B and C static context:
  Agents B and C have a separate static_dim=53 input (Agent A's feature set).
  The ranges below define ONLY the seq input dims for B and C.
  The static vector is extracted separately in independent_agent.py using Agent A's ranges.

NOTE — Agent T data availability:
  Historical Theta Data EOD chain CSVs lack real OPRA trade-flow streams.
  Smart Money (270-284), Volume Anomaly (285-296), and Trade Conditions (297-306) are
  near-empty. Agent T is remapped to 25 dims from real microstructure fields:
    bid_ask_spread (180-184) + order_book_imbalance (184-189) + quote_intensity (189-192)
    + trade_velocity (192-194) + effective_spreads (194-196) + price_impact (196-199) = 19 dims
    + wall_distances (206-214) truncated to 6 dims = 25 dims total
  TODO: Re-enable OPRA-sourced ranges when real trade-flow data is available.

NOTE — Agent Q data availability:
  Quote Pressure dims 307-324 are partially real (depth_ratio 317-319 is reliable;
  cvd/tape-reading fields may be sparse). Extended with microstructure to reach 20 dims:
    quote_pressure depth_ratio (317-319) = 2 dims
    order_book_imbalance (184-189) = 5 dims
    bid_ask_spread (180-184) = 4 dims
    quote_intensity (189-192) = 3 dims
    iv_by_moneyness first 6 dims (125-131) = 6 dims
  Total = 20 dims
"""

TOTAL_FEAT_DIM = 286

AGENT_FEATURE_SUBSETS = {
    # Agent A: temporal + static Greek/IV/OI snapshot specialist
    # 53 dims drawn from highest-reliability subgroups:
    #   atm_greeks subgroup (65-72) = 7
    #   skew_metrics subgroup (72-75) = 3
    #   net_gamma + dealer + zones subgroups (95-105) = 10
    #   vanna_charm net_exposures + ratios subgroups (115-125) = 10
    #   iv_surface full (125-149) = 24  → trim to first 7 (iv_by_moneyness 125-132)
    #     + iv_term_structure (132-137) = 5
    #     + vol_skew_metrics (137-140) = 3  → total iv = 15
    #   walls_positioning max_oi_strikes (202-206) = 4
    #   walls_positioning wall_distances (206-210) = 4
    #   Total: 10 + 10 + 10 + 7 + 5 + 3 + 8 = 53
    'A': {
        'name': 'Alpha (Greek/IV/OI snapshot)',
        'ranges': [
            (65, 75),    # atm_greeks (7) + skew_metrics (3) = 10
            (95, 105),   # net_gamma (4) + dealer_positioning (3) + gamma_zones (3) = 10
            (115, 125),  # vanna_charm net_exposures (6) + cross_greek_ratios (4) = 10
            (125, 132),  # iv_by_moneyness = 7
            (132, 137),  # iv_term_structure = 5
            (137, 140),  # vol_skew_metrics first 3 = 3
            (202, 210),  # max_oi_strikes (4) + wall_distances first 4 = 8
            (270, 273),  # CSV-derived: lambda_mean, lambda_atm, lambda_skew = 3
            (278, 286),  # CSV-derived aux: dual/d1d2/iv_error/ultima + OI enrichments = 8
        ],
        'feat_dim': 64,
        'use_backbone': True,
    },

    # Agent B: BiLSTM sequence specialist (seq-only; static_dim=53 from Agent A passed separately)
    # 34 seq dims: per-bar temporal IV and cross-strike dynamics
    #   iv_surface vol_skew_metrics (137-142) = 5
    #   iv_surface iv_percentiles (142-145) = 3
    #   iv_surface put_call_iv_diff (145-150) = 5
    #   cross_strike oi_volume_dist (220-226) = 6
    #   cross_strike greek_concentrations (226-229) = 3
    #   cross_strike strike_clustering (229-232) = 3
    #   cross_strike liquidity_gradient (232-235) = 3
    #   time_decay dte_buckets (235-240) = 5
    #   time_decay decay_accelerations (240-241) = 1
    #   Total: 5+3+5+6+3+3+3+5+1 = 34
    'B': {
        'name': 'Beta (BiLSTM sequence)',
        'ranges': [
            (137, 150),  # vol_skew_metrics (5) + iv_percentiles (3) + put_call_iv_diff (5) = 13
            (220, 235),  # cross_strike full (15) = 15
            (235, 241),  # time_decay dte_buckets (5) + decay_accel first 1 = 6
            (273, 275),  # CSV-derived: dist_atm_mean/weighted = 2
            (241, 245),  # decay_accelerations [1-4] = 4
        ],
        'feat_dim': 40,
        'use_backbone': False,
    },

    # Agent C: CNN + Multi-head Attention sequence specialist (seq-only; static_dim=53 from Agent A)
    # 34 seq dims: microstructure + sentiment — different from B for diversity
    #   microstructure bid_ask_spread (180-184) = 4
    #   microstructure order_book_imbalance (184-189) = 5
    #   microstructure quote_intensity (189-192) = 3
    #   microstructure trade_velocity (192-194) = 2
    #   microstructure effective_spreads (194-196) = 2
    #   microstructure price_impact (196-200) = 4  → total micro = 20
    #   sentiment_regime sentiment_scores (250-253) = 3
    #   sentiment_regime volatility_regime (253-257) = 4
    #   sentiment_regime trend_stress (257-263) = 6
    #   time_decay time_concentrations (245-248) = 3 (avoids full overlap with B on dte_buckets)
    #   time_decay calendar_proximity (248-250) = 2
    #   Total: 20 + 3 + 4 + 6 + 3 + 2 = 38 → trim to 34
    #   Use: micro (180-200)=20, sentiment_scores+volatility (250-257)=7, trend_stress (257-264)=7 → 20+7+7=34
    'C': {
        'name': 'Chi (CNN+Attention sequence)',
        'ranges': [
            (180, 200),  # microstructure full (20) = 20
            (250, 264),  # sentiment_scores (3) + volatility_regime (4) + trend_stress first 7 = 14
            (275, 278),  # CSV-derived: spread_pct_mean/atm/skew = 3
            (264, 266),  # correlation_metrics first 2 = 2
        ],
        'feat_dim': 39,
        'use_backbone': True,
    },

    # Agent K: compact static MLP Greek specialist
    # 53 dims: same high-reliability indices as Agent A
    # Intentional overlap with A is acceptable — different architectures learn different representations
    'K': {
        'name': 'Kappa (static Greek MLP)',
        'ranges': [
            (65, 75),    # atm_greeks + skew_metrics = 10
            (95, 105),   # net_gamma + dealer + zones = 10
            (115, 125),  # vanna/charm net + ratios = 10
            (125, 132),  # iv_by_moneyness = 7
            (132, 137),  # iv_term_structure = 5
            (137, 143),  # vol_skew_metrics full 6 = 6
            (202, 210),  # max_oi_strikes + wall_distances = 8
            (270, 273),  # CSV-derived: lambda_mean/atm/skew = 3
            (284, 286),  # CSV-derived: oi_mean + oi_put_call_ratio = 2
            (143, 146),  # iv_percentiles first 3 = 3
        ],
        'feat_dim': 64,
        'use_backbone': False,
    },

    # Agent T: remapped to real microstructure + wall proximity dims
    # Original OPRA trade-flow dims (270-307) are near-empty in EOD historical snapshots.
    # Remapped to 25 real dims:
    #   bid_ask_spread (180-184) = 4
    #   order_book_imbalance (184-189) = 5
    #   quote_intensity (189-192) = 3
    #   trade_velocity (192-194) = 2
    #   effective_spreads (194-196) = 2
    #   price_impact (196-200) = 4  → micro = 20 (avoid overlap with K for diversity, accept micro overlap with C/Q)
    #   time_decay calendar_proximity (248-250) = 2
    #   walls wall_distances last 3 (211-214) = 3
    #   Total: 20 + 2 + 3 = 25
    # TODO: Replace with OPRA-sourced trade-flow dims when real TQ data is available.
    'T': {
        'name': 'Tau (microstructure proxy — remapped from OPRA)',
        'ranges': [
            (180, 200),  # microstructure full = 20
            (248, 250),  # calendar_proximity = 2
            (211, 214),  # wall_distances last 3 = 3
            (284, 286),  # CSV-derived aux: oi_mean + oi_put_call_ratio = 2
        ],
        'feat_dim': 27,
        'use_backbone': False,
    },

    # Agent Q: quote pressure + microstructure extension
    # Quote Pressure (307-324) is partially real; extended to reach 20 dims:
    #   quote_pressure depth_ratio (317-319) = 2
    #   quote_pressure order_book (317-319) already counted — use cvd_metrics (307-310) = 3
    #   quote_pressure quote_dynamics (310-313) = 3
    #   order_book_imbalance (184-189) = 5
    #   bid_ask_spread (180-184) = 4
    #   quote_intensity (189-192) = 3
    #   Total: 3 + 3 + 5 + 4 + 3 = 18 → add depth_ratio (317-319) = 2 → 20
    'Q': {
        'name': 'Quote (IV + microstructure remap)',
        'ranges': [
            (125, 131),  # iv_by_moneyness = 6
            (180, 189),  # bid_ask_spread + order_book_imbalance = 9
            (105, 107),  # vanna/charm net exposures = 2
            (189, 192),  # quote_intensity = 3
            (282, 284),  # CSV-derived aux: iv_error_mean + ultima_mean = 2
        ],
        'feat_dim': 22,
        'use_backbone': False,
    },

    # Agent 2D: uses chain_2d tensor — not the flat 325-dim vector
    '2D': {
        'name': '2D (chain surface CNN)',
        'ranges': [],
        'feat_dim': 0,
        'use_backbone': False,
    },
}

# Auto-recompute feat_dim from ranges (overrides any hardcoded value above)
for _agent_type, _cfg in AGENT_FEATURE_SUBSETS.items():
    _cfg['feat_dim'] = sum(end - start for start, end in _cfg.get('ranges', []))

# Sanity-check computed dims against patched agent constructor expectations
_EXPECTED_DIMS = {'A': 64, 'B': 40, 'C': 39, 'K': 64, 'T': 27, 'Q': 22, '2D': 0}
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
    print("Feature Subset Configuration — Hybrid52")
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
