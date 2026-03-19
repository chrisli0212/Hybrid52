# Hybrid51 Production Model - h30 VIX-Gated

**Version:** 1.0  
**Date:** 2026-03-09  
**Performance:** Test AUC 0.7221 | F1 0.7162 | Accuracy 0.6100

## Overview

This is the production-ready deployment of the Hybrid51 3-stage ensemble model for 30-minute SPXW directional prediction. The model uses VIX regime-gated fusion at Stage 3 for optimal performance.

## Architecture

### Stage 1: Per-Symbol Per-Agent Predictions
- **35 models** (5 symbols × 7 agents)
- Symbols: SPXW, SPY, QQQ, IWM, TLT
- Agents: A, B, C, K, T, Q, 2D
- Each model: 325-dim input → 64-dim hidden → binary classifier

### Stage 2: Cross-Symbol Fusion
- **7 fusion models** (one per agent)
- Combines predictions across all 5 symbols
- Uses chain context (2D frozen features)
- Standard agents: 16-dim input (5 symbols × 2 + 4 diffs + 2 chain)
- Agent 2D: 14-dim input (SPXW unfrozen + 4 frozen peers + 4 diffs)

### Stage 3: VIX-Gated Meta-Ensemble
- **1 final model** (RegimeGatedProbFusion)
- Combines 7 agent probabilities with VIX regime features
- Learned per-agent gates conditioned on VIX volatility regime
- 10-dim VIX features → 32-dim regime embedding → gated fusion

## Directory Structure

```
Final_production_model/
├── models/
│   ├── stage1/          # 35 Stage 1 checkpoints
│   ├── stage2/          # 7 Stage 2 fusion models + chain_context.npz
│   └── stage3/          # VIX-gated final model + metrics.json
├── hybrid51_models/     # Model architecture definitions
├── hybrid51_utils/      # Utility functions and path helpers
├── scripts/             # Inference scripts
├── config/              # Configuration files
│   └── production_config.json
└── README.md            # This file
```

## Model Files

### Stage 1 (35 files)
- `models/stage1/{SYMBOL}_agent{AGENT}.pt`
- Example: `SPXW_agentA.pt`, `SPY_agent2D.pt`

### Stage 2 (8 files)
- `models/stage2/agent{AGENT}_fusion.pt` (7 files)
- `models/stage2/chain_context.npz` (precomputed 2D features)

### Stage 3 (2 files)
- `models/stage3/stage3_vix_gated.pt` (final model)
- `models/stage3/metrics.json` (performance metrics)

## Data Requirements

### Tier 3 Binary Features
- **Path:** `/workspace/data/tier3_binary_v5/{SYMBOL}/horizon_30min/`
- **Files per symbol:**
  - `{split}_sequences.npy` - (N, 325) feature sequences
  - `{split}_labels.npy` - (N,) binary labels
  - `{split}_chain_2d.npy` - (N, 2) chain context for 2D models
  - `train_sequences.npy` - for normalization stats

### VIX Features
- **Path:** `/workspace/data/tier3_vix_v4/VIXW/`
- **Files:**
  - `{split}_vix_features.npy` - (N, 4, 10) VIX regime features
  - `{split}_vix_labels.npy` - (N,) labels (for validation)

## Usage

### Quick Start

```python
import sys
sys.path.insert(0, '/path/to/Final_production_model')

from scripts.production_inference import ProductionPipeline

# Initialize pipeline
pipeline = ProductionPipeline(
    model_dir='./models',
    data_root='/workspace/data/tier3_binary_v5',
    vix_root='/workspace/data/tier3_vix_v4/VIXW',
    device='cuda'
)

# Run inference
predictions = pipeline.predict(split='test')

# predictions contains:
# - probs: (N,) final probabilities
# - preds: (N,) binary predictions (threshold=0.5)
# - labels: (N,) ground truth
# - stage1_preds: dict of per-symbol per-agent predictions
# - stage2_probs: dict of per-agent fusion probabilities
# - vix_gates: (N, 7) learned agent gates
```

### Command Line

```bash
cd Final_production_model

# Run inference on test split
python scripts/production_inference.py \
    --split test \
    --output predictions.npz

# Run on validation split
python scripts/production_inference.py \
    --split val \
    --output val_predictions.npz
```

## Performance Metrics

### Test Set Performance (h30)

| Method | AUC | F1 | Accuracy |
|--------|-----|----|----|
| **VIX-Gated (Production)** | **0.7221** | **0.7162** | 0.6100 |
| LogReg C=0.1 | 0.7212 | 0.7091 | 0.6622 |
| Simple Average | 0.7181 | 0.7189 | 0.6371 |

### Comparison to h15

| Horizon | AUC | Gain |
|---------|-----|------|
| h15 | 0.6814 | baseline |
| **h30** | **0.7221** | **+407 bps** |

The 30-minute horizon provides more stable regime signals and better predictive power.

## Model Characteristics

### Strengths
- **High AUC (0.72+)**: Strong rank-ordering of predictions
- **Stable across horizons**: Consistent performance from h15 to h30
- **Regime-aware**: VIX gating adapts to market volatility
- **Ensemble diversity**: 7 different agent architectures

### Use Case
- **Primary:** 0DTE credit spread entry/exit decisions
- **Horizon:** 30-minute directional prediction
- **Target:** SPXW (SPX weekly options)
- **Signal:** Binary up/down probability

### Limitations
- **SPXW-centric**: Peer symbols (SPY/QQQ/IWM/TLT) are weak individually
- **Intraday only**: Trained on 30-min bars, not for multi-day holds
- **Threshold-dependent F1**: Optimal threshold may vary by market regime

## Inference Pipeline Details

### Stage 1: Symbol-Agent Predictions
1. Load 5 symbols × 7 agents = 35 models
2. For each symbol-agent pair:
   - Load sequences and normalize (z-score from train stats)
   - Run model inference (batch_size=1024)
   - Output: logits and probabilities

### Stage 2: Cross-Symbol Fusion
1. For each agent:
   - Stack predictions from all 5 symbols
   - Compute cross-diffs (SPXW vs peers)
   - Append chain context (2D frozen features)
   - Run fusion MLP
   - Output: fused probability per agent

### Stage 3: VIX-Gated Meta
1. Stack 7 agent probabilities
2. Load VIX features (collapse time dimension)
3. Resample VIX to match agent probability length
4. Run RegimeGatedProbFusion:
   - VIX → regime embedding
   - Regime → per-agent gates
   - Gated fusion of agent probabilities
5. Output: final probability + gate values

## Threshold Selection

**Default threshold:** 0.5 (from validation sweep)

For custom thresholds:
```python
# Sweep threshold on validation set
from scripts.utils import sweep_threshold

val_preds = pipeline.predict(split='val')
optimal_thr, best_f1 = sweep_threshold(
    val_preds['probs'], 
    val_preds['labels']
)
print(f"Optimal threshold: {optimal_thr:.3f} (F1={best_f1:.4f})")
```

## Monitoring and Diagnostics

### Check Gate Behavior
```python
# Analyze which agents are trusted in different regimes
gates = predictions['vix_gates']  # (N, 7)
agent_names = ['A', 'B', 'C', 'K', 'T', 'Q', '2D']

for i, name in enumerate(agent_names):
    print(f"Agent {name}: mean gate = {gates[:, i].mean():.3f}")
```

### Stage-by-Stage Validation
```python
# Check Stage 1 predictions
for sym in ['SPXW', 'SPY', 'QQQ', 'IWM', 'TLT']:
    for ag in ['A', 'B', 'C', 'K', 'T', 'Q', '2D']:
        if ag in predictions['stage1_preds'].get(sym, {}):
            probs = predictions['stage1_preds'][sym][ag][1]
            print(f"{sym} Agent {ag}: mean prob = {probs.mean():.3f}")

# Check Stage 2 fusion
for ag in ['A', 'B', 'C', 'K', 'T', 'Q', '2D']:
    if ag in predictions['stage2_probs']:
        probs = predictions['stage2_probs'][ag]
        print(f"Stage 2 Agent {ag}: mean prob = {probs.mean():.3f}")
```

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
```

## Citation

If you use this model in your research or trading, please cite:

```
Hybrid51 h30 VIX-Gated Model
Version 1.0 (2026-03-09)
3-Stage Ensemble with Regime-Aware Fusion
Test AUC: 0.7221
```

## Support

For questions or issues:
1. Check configuration in `config/production_config.json`
2. Verify data paths and file existence
3. Review logs for model loading errors
4. Ensure CUDA is available if using GPU

## Version History

- **v1.0 (2026-03-09)**: Initial production release
  - h30 traditional VIX-gated model
  - Test AUC 0.7221, F1 0.7162
  - 35 Stage 1 + 7 Stage 2 + 1 Stage 3 models
