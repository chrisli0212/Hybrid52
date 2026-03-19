# Agent VIX — VIX Regime Gating Agent for Hybrid 5.1

## What This Is

Agent VIX (codename "V") is a **regime-conditioning agent** that reads volatility environment data and tells the meta-model *which of the 7 directional agents to trust right now*. It is NOT an 8th directional voter — it does NOT predict UP or DOWN. Instead, it produces a **regime embedding** that gates (scales) each directional agent's output before final fusion.

**Why this matters:** In calm markets, Greek/structural agents (K, 2D) are most reliable. In stress/crisis, flow agents (T, Q) become more reliable. A static ensemble averages them equally, causing "regime decay" — where agents that are excellent in one condition get diluted by the static weights. Agent VIX solves this by dynamically adjusting trust weights per agent based on the current volatility regime.

---

## Architecture Overview

```
                    ┌─────────────────────┐
                    │  VIX 5-min Features  │
                    │  (~10 features)      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   AgentVIX (MLP)     │
                    │   ~50K params        │
                    │                      │
                    │   Outputs:           │
                    │   - regime_emb (32d) │
                    │   - regime_class (5) │
                    │   - score (1)        │
                    │   - confidence (1)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │     Per-Agent Gate Networks      │
              │                                  │
              │  gate_A(regime_emb) → σ → w_A    │
              │  gate_B(regime_emb) → σ → w_B    │
              │  gate_K(regime_emb) → σ → w_K    │
              │  gate_C(regime_emb) → σ → w_C    │
              │  gate_T(regime_emb) → σ → w_T    │
              │  gate_Q(regime_emb) → σ → w_Q    │
              │  gate_2D(regime_emb) → σ → w_2D  │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │   Gated Agent Fusion             │
              │                                  │
              │   gated_i = agent_prob_i × w_i   │
              │   final = MLP([gated_0..6])      │
              └─────────────────────────────────┘
```

## Agent VIX Is NOT Like Other Agents

| Property | Agents A/B/C/K/T/Q/2D | Agent VIX |
|:---|:---|:---|
| **Goal** | Predict UP or DOWN | Classify volatility regime |
| **Input** | 325-dim option chain features | ~10 VIX-specific features |
| **Timeframe** | 1-min bars, 20-step lookback | 5-min bars, pre-computed multi-scale |
| **Update cadence** | Every 1 min | Every 5 min (cached) |
| **Output** | `(score, confidence, signal)` | `(score, confidence, signal)` + `regime_emb` |
| **Data source** | Per-symbol option chain (SPXW/SPY/QQQ...) | VIX/VIXW 1-min resampled to 5-min |
| **Wrapped by** | `IndependentAgent` | Standalone or `IndependentAgent` (V type) |
| **Training stage** | Stage 1 (binary classification) | Stage 1 warm-start → Stage 3 end-to-end |

---

## VIX Features (~10 dimensions)

Built from your existing 5-year 1-min VIXW data resampled to 5-min bars. **No new data fetch needed.**

| Feature | Description | Computation |
|:---|:---|:---|
| `vix_level` | Current VIX spot price | Last close of 5-min bar |
| `vix_pct_5m` | 5-min percentage change | `(close - open) / open` of current bar |
| `vix_pct_15m` | 15-min percentage change | Current close vs 3-bars-ago close |
| `vix_pct_1h` | 1-hour percentage change | Current close vs 12-bars-ago close |
| `vix_zscore_15m` | 15-min z-score vs 20-bar rolling | `(level - roll_mean) / roll_std` |
| `vix_percentile_1h` | 1-hour percentile rank | Rank of current level in 12-bar window |
| `vix_term_slope` | VIX term structure slope | `(VX_M2 - VX_M1) / VX_M1` or proxy |
| `vvix_level` | Vol-of-vol (VVIX) | Direct from data if available; else IV of VIX options |
| `vix_vix1d_spread` | VIX − VIX1D spread | Captures intraday vs realized vol gap |
| `vix_hilo_range` | 5-min high-low range | `(high - low) / open` of current bar |

### Why These Specific Features

- **vix_level**: Determines base regime (calm <15, elevated 20-25, stress >30)
- **Multi-timescale % changes** (5m, 15m, 1h): Captures momentum and regime transitions at different speeds
- **vix_zscore_15m**: Detects unusual VIX moves relative to recent history
- **vix_percentile_1h**: Regime context (is VIX high or low relative to the session?)
- **vix_term_slope**: Contango = calm expectations; backwardation = stress/fear
- **vvix_level**: Vol-of-vol signals regime instability
- **vix_vix1d_spread**: Intraday fear premium
- **vix_hilo_range**: Direct intraday VIX volatility

---

## Regime Classification (5 classes)

| Regime | VIX Range | Characteristics | Agent Trust Pattern |
|:---|:---|:---|:---|
| **CALM** | < 15 | Low vol, mean-reverting, Greeks dominate | Trust K, 2D, A highly |
| **NORMAL** | 15–20 | Balanced, all agents useful | Roughly equal trust |
| **ELEVATED** | 20–25 | Rising vol, flow signals matter more | Increase T, Q trust |
| **HIGH** | 25–35 | Stress, microstructure shifts | Strong T, Q, reduce K |
| **EXTREME** | > 35 | Crisis, structural breaks | Heavily trust T, Q; reduce structural agents |

---

## Data Pipeline — `build_tier3_vix.py`

```
Input: /workspace/data/tier2_minutes_v4/VIXW_minutes.parquet (1-min bars)
  → Resample to 5-min bars (OHLCV rules)
    → Compute 10 VIX features per 5-min bar
      → Build rolling sequences (shape: batch × lookback × 10)
        → Align timestamps with directional agent labels
          → Chronological 60/20/20 split
            → Z-score normalize using train-split stats
Output: /workspace/data/tier3_vix_v4/{SYMBOL}/
  ├── train_vix_features.npy, train_vix_labels.npy
  ├── val_vix_features.npy, val_vix_labels.npy
  ├── test_vix_features.npy, test_vix_labels.npy
  ├── vix_norm_mean.npy, vix_norm_std.npy
  └── vix_metadata.json
```

**Timestamp alignment**: Each 5-min VIX bar covers 5 directional-agent 1-min bars. The VIX regime is "cached" per 5-min window and applied to all 1-min predictions within that window.

---

## Integration with Meta Model — `RegimeGatedMetaModel`

The meta model (Stage 3) wraps the 7 frozen directional agents + Agent VIX:

```python
class RegimeGatedMetaModel(nn.Module):
    """
    Stage 3 meta model:
    - Loads 7 frozen Stage 1 agent checkpoints
    - Agent VIX produces regime embedding
    - Per-agent gate networks modulate agent outputs
    - Final MLP fuses gated outputs into direction prediction
    """
    def __init__(self, agent_checkpoints, vix_agent, n_agents=7):
        # 7 frozen directional agents (A, B, K, C, T, Q, 2D)
        # 1 trainable VIX agent
        # 7 gate networks (one per agent)
        # Final fusion MLP

    def forward(self, sequences_1min, vix_features_5min, chain_2d=None):
        # 1. Run each frozen agent → get 7 × (logit) outputs
        # 2. Run Agent VIX on 5-min features → regime_emb
        # 3. For each agent i: gate_i = sigmoid(gate_net_i(regime_emb))
        # 4. Gated outputs: gated_i = softmax(agent_logit_i) × gate_i
        # 5. Fuse: final_logit = fusion_mlp(concat(gated_0..6, regime_emb))
        # 6. Return final prediction + regime classification
```

### Training Strategy

1. **Stage 1 (warm-start)**: Train Agent VIX standalone on regime classification task using VIX 5-min features. Label = discretized VIX regime (5 classes). This gives the VIX encoder a good starting point.
2. **Stage 3 (end-to-end)**: Load frozen Stage 1 agents + warm-started Agent VIX. Train the gate networks + fusion MLP end-to-end on the directional prediction task. Gradients flow through gates → Agent VIX, allowing the optimizer to discover which regimes favor which agents.

---

## File Inventory

| File | Location | Purpose |
|:---|:---|:---|
| `AGENT_VIX.md` | `docs/` or project root | This document — Windsurf onboarding |
| `agent_vix.py` | `hybrid51_models/agents/` | PyTorch module for Agent VIX |
| `vix_feature_subsets.py` | `config/` | VIX feature definitions + subset config update |
| `build_tier3_vix.py` | `scripts/phase0/` | Data pipeline: 1-min VIXW → 5-min VIX features |
| `regime_gated_meta_model.py` | `hybrid51_models/` | Stage 3 meta model with gating |
| `train_vix_agent.py` | `scripts/stage1/` | Stage 1 training script for Agent VIX standalone |

---

## Implementation Order

1. **`build_tier3_vix.py`** — Build VIX 5-min features from existing VIXW data
2. **`agent_vix.py`** — Implement the VIX agent module
3. **`vix_feature_subsets.py`** — Register Agent V in the feature subset config
4. **`train_vix_agent.py`** — Train Agent VIX standalone (Stage 1 warm-start)
5. **`regime_gated_meta_model.py`** — Build the gated meta model (Stage 3)
6. Train Stage 3 end-to-end with frozen directional agents + trainable gates + VIX agent

---

## Key Constraints

- Agent VIX **MUST** output `(score, confidence, signal)` tuple — this interface is required by `IndependentAgent` and the ensemble pipeline
- Agent VIX additionally outputs `regime_emb` (32-d) for the gate networks
- VIX features are at 5-min resolution; directional agents stay at 1-min
- No new data fetch needed — use existing 1-min VIXW parquet resampled to 5-min
- Agent VIX is a **static MLP** (like Agent K) — no temporal backbone needed because the features are pre-computed at multi-timescale
- Target parameter budget: ~50K (similar to Agent S, much smaller than A/B/C/K)
- Stage 1 regime labels are derived from VIX level thresholds (supervised)
- Stage 3 gates are trained end-to-end with directional prediction loss

---

## Expected Impact

| Metric | Before VIX Agent | After VIX Agent (est.) |
|:---|:---|:---|
| Full accuracy | ~61% (meta-fused) | +0.5–1.8pp |
| High-confidence accuracy | ~65% | +2–5pp |
| AUC | 0.653 | 0.668–0.680 |
| IC | baseline | +0.02–0.04 |
| Regime stability | Static weights | Dynamic per-regime |

**Primary value**: Not raw accuracy boost, but **conditional accuracy improvement** — the model makes fewer bad trades in the wrong regime, which translates to better Sharpe ratio and drawdown control for 0DTE credit spread decisions.
