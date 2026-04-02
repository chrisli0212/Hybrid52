"""
Agent B Feature Configuration.

Agent B: 1-minute intraday snapshot agent (Tier-2 real-time).
Data source: live 1-min Greek snapshots + trade/quote data.

Feature breakdown:
  Group                  Dims   Source
  ---------------------- -----  ------
  greek_by_strike          75   greek_df  (delta-bucketed + ATM + skew)
  gamma_exposure           30   greek_df  (GEX by strike, net, dealer, zones)
  vanna_charm              20   greek_df  (bucketed vanna/charm + net + ratios)
  iv_surface               25   greek_df  (moneyness, term, skew, percentiles)
  flow_volume              30   trade_df  (C/P ratios, aggression, premium)
  microstructure           20   trade_df  (spread, imbalance, velocity)
  walls_positioning        20   greek_df  (put/call walls, dealer position)
  cross_strike             15   greek_df  (OI/vol distribution)
  time_decay               15   greek_df  (DTE buckets, decay acceleration)
  sentiment_regime         20   greek_df  (sentiment scores, vol regime)
  csv_derived              16   greek_df  (lambda, dist_atm, spread_pct, OI)
  ohlc_dynamics            25   ohlc_df   (1-min bar features)
  ---------------------- -----
  TOTAL                   311

Note: Phase 1 features (SmartMoney + VolumeAnomaly + TradeCondition +
QuotePressure = 55 dims) are DISABLED for BigQuery historical training
(HISTORICAL_MODE=True) and ENABLED for live deployment only.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# ------------------------------------------------------------------ #
#  Dimension constants — single source of truth for Agent B           #
# ------------------------------------------------------------------ #

AGENT_B_BASE_DIM    = 311   # historical mode (BigQuery training)
AGENT_B_PHASE1_DIM  = 55    # Phase 1 live-only features
AGENT_B_LIVE_DIM    = AGENT_B_BASE_DIM + AGENT_B_PHASE1_DIM  # 366

AGENT_B_DIM = AGENT_B_BASE_DIM  # default: historical mode

# ------------------------------------------------------------------ #
#  Feature group registry                                              #
# ------------------------------------------------------------------ #

@dataclass
class FeatureBlock:
    name: str
    start: int
    size: int
    source: str  # 'greek_df' | 'trade_df' | 'ohlc_df'
    description: str = ""

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def indices(self):
        return range(self.start, self.end)


AGENT_B_REGISTRY: List[FeatureBlock] = [
    FeatureBlock("greek_by_strike",  0,    75,  "greek_df",  "Delta-bucketed greeks + ATM + skew"),
    FeatureBlock("gamma_exposure",   75,   30,  "greek_df",  "GEX by strike, net gamma, dealer, zones"),
    FeatureBlock("vanna_charm",      105,  20,  "greek_df",  "Bucketed vanna/charm, net, cross-greek ratios"),
    FeatureBlock("iv_surface",       125,  25,  "greek_df",  "IV by moneyness, term structure, skew"),
    FeatureBlock("flow_volume",      150,  30,  "trade_df",  "C/P ratios, aggression, premium, dark pool"),
    FeatureBlock("microstructure",   180,  20,  "trade_df",  "Spread, imbalance, velocity, impact"),
    FeatureBlock("walls_positioning",200,  20,  "greek_df",  "Put/call walls, dealer positioning"),
    FeatureBlock("cross_strike",     220,  15,  "greek_df",  "OI/vol distribution, clustering"),
    FeatureBlock("time_decay",       235,  15,  "greek_df",  "DTE buckets, decay acceleration"),
    FeatureBlock("sentiment_regime", 250,  20,  "greek_df",  "Sentiment, vol regime, stress"),
    FeatureBlock("csv_derived",      270,  16,  "greek_df",  "Lambda, dist_atm, spread_pct, OI enrichments"),
    FeatureBlock("ohlc_dynamics",    286,  25,  "ohlc_df",   "1-min OHLC chain dynamics"),
]

_total = sum(b.size for b in AGENT_B_REGISTRY)
assert _total == AGENT_B_BASE_DIM, f"Agent B registry total {_total} != {AGENT_B_BASE_DIM}"

# Flat feature name list (populated from config names, placeholder for now)
AGENT_B_FEATURES: List[str] = []
AGENT_B_GROUPS: Dict[str, List[str]] = {b.name: [] for b in AGENT_B_REGISTRY}

# Phase 1 block registry (live deployment only)
AGENT_B_PHASE1_REGISTRY: List[FeatureBlock] = [
    FeatureBlock("smart_money",       311, 15, "trade_df", "Smart money detection"),
    FeatureBlock("volume_anomaly",    326, 12, "trade_df", "Volume anomaly detection"),
    FeatureBlock("trade_conditions",  338, 10, "trade_df", "Trade condition analysis"),
    FeatureBlock("quote_pressure",    348, 18, "trade_df", "Quote pressure & exchange routing"),
]

_p1_total = sum(b.size for b in AGENT_B_PHASE1_REGISTRY)
assert _p1_total == AGENT_B_PHASE1_DIM, f"Phase 1 total {_p1_total} != {AGENT_B_PHASE1_DIM}"
