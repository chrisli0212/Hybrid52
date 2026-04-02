"""
Agent H Feature Configuration.

Agent H: Sequence / LSTM agent.
Input shape: (SEQ_LEN, AGENT_H_DIM) — a rolling window of flat feature vectors.

CRITICAL: SEQ_LEN must match the LSTM input_size in the model definition.
Change ONLY here; all training and deployment code reads from this file.

Feature dim is a subset of Agent B's base features:
  - greek_by_strike  (75)
  - gamma_exposure   (30)
  - vanna_charm      (20)
  - iv_surface       (25)
  - time_decay       (15)
  ————————————————
  TOTAL              165

Rationale: Flow/microstructure features are noisy at sequence level;
using greek surface + time decay only for temporal patterns.
"""

AGENT_H_SEQ_LEN: int = 20    # rolling window length (1-min bars)
                               # NOTE: must match LSTM hidden architecture

AGENT_H_DIM: int = 165        # features per timestep

# Source blocks (names must match AGENT_B_REGISTRY block names)
AGENT_H_SOURCE_BLOCKS = [
    "greek_by_strike",   # 75
    "gamma_exposure",    # 30
    "vanna_charm",       # 20
    "iv_surface",        # 25
    "time_decay",        # 15
]

_expected = 75 + 30 + 20 + 25 + 15
assert _expected == AGENT_H_DIM, f"Agent H: block sum {_expected} != {AGENT_H_DIM}"

# Required input columns for Agent H (same as Agent B greek blocks)
REQUIRED_COLS = [
    "delta", "gamma", "vega", "theta", "vanna", "charm",
    "implied_vol", "open_interest", "moneyness",
    "bid", "ask", "underlying_price",
    "dte_int",
]
