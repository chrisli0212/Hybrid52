# Production Model Inventory

**Model:** Hybrid51 h30 VIX-Gated  
**Version:** 1.0  
**Date:** 2026-03-09  
**Test Performance:** AUC 0.7221 | F1 0.7162 | Accuracy 0.6100

---

## Complete File Listing

### Model Checkpoints (43 files)

#### Stage 1: Per-Symbol Per-Agent (35 files)
```
models/stage1/SPXW_agentA.pt
models/stage1/SPXW_agentB.pt
models/stage1/SPXW_agentC.pt
models/stage1/SPXW_agentK.pt
models/stage1/SPXW_agentT.pt
models/stage1/SPXW_agentQ.pt
models/stage1/SPXW_agent2D.pt

models/stage1/SPY_agentA.pt
models/stage1/SPY_agentB.pt
models/stage1/SPY_agentC.pt
models/stage1/SPY_agentK.pt
models/stage1/SPY_agentT.pt
models/stage1/SPY_agentQ.pt
models/stage1/SPY_agent2D.pt

models/stage1/QQQ_agentA.pt
models/stage1/QQQ_agentB.pt
models/stage1/QQQ_agentC.pt
models/stage1/QQQ_agentK.pt
models/stage1/QQQ_agentT.pt
models/stage1/QQQ_agentQ.pt
models/stage1/QQQ_agent2D.pt

models/stage1/IWM_agentA.pt
models/stage1/IWM_agentB.pt
models/stage1/IWM_agentC.pt
models/stage1/IWM_agentK.pt
models/stage1/IWM_agentT.pt
models/stage1/IWM_agentQ.pt
models/stage1/IWM_agent2D.pt

models/stage1/TLT_agentA.pt
models/stage1/TLT_agentB.pt
models/stage1/TLT_agentC.pt
models/stage1/TLT_agentK.pt
models/stage1/TLT_agentT.pt
models/stage1/TLT_agentQ.pt
models/stage1/TLT_agent2D.pt
```

**Total Stage 1:** 35 checkpoints (~23.8 MB total)

#### Stage 2: Cross-Symbol Fusion (8 files)
```
models/stage2/agentA_fusion.pt
models/stage2/agentB_fusion.pt
models/stage2/agentC_fusion.pt
models/stage2/agentK_fusion.pt
models/stage2/agentT_fusion.pt
models/stage2/agentQ_fusion.pt
models/stage2/agent2D_fusion.pt
models/stage2/chain_context.npz
```

**Total Stage 2:** 7 fusion models + 1 chain context (~4.7 MB total)

#### Stage 3: VIX-Gated Meta (2 files)
```
models/stage3/stage3_vix_gated.pt
models/stage3/metrics.json
```

**Total Stage 3:** 1 model + 1 metrics file (~156 KB)

---

## Model Architecture Files

### Core Models (hybrid51_models/)
```
hybrid51_models/__init__.py
hybrid51_models/independent_agent.py          # Stage 1 agent architecture
hybrid51_models/cross_symbol_agent_fusion.py  # Stage 2 fusion MLP
hybrid51_models/regime_gated_meta_model.py    # Stage 3 VIX-gated fusion
hybrid51_models/tlt_gated_agent_fusion.py     # TLT-gated variant (not used in production)
```

### Utilities (hybrid51_utils/)
```
hybrid51_utils/__init__.py
hybrid51_utils/artifacts.py                   # Path helpers and data loading
```

---

## Scripts

### Inference
```
scripts/simple_inference.py                   # Main production inference script
```

---

## Configuration

```
config/production_config.json                 # Model parameters and paths
```

**Key Parameters:**
- Horizon: 30 minutes
- Target: SPXW
- Stage 1: 325-dim input, 64-dim hidden, dropout 0.3
- Stage 2: 32-dim hidden, dropout 0.2
- Stage 3: VIX 10-dim → 32-dim regime embedding, 64-dim fusion hidden, dropout 0.2
- Threshold: 0.5

---

## Documentation

```
README.md                                     # Complete usage guide
INVENTORY.md                                  # This file
```

---

## Data Dependencies

### Required Data Paths

**Tier 3 Binary Features:**
```
/workspace/data/tier3_binary_v5/{SYMBOL}/horizon_30min/
  ├── train_sequences.npy      # For normalization stats
  ├── val_sequences.npy
  ├── val_labels.npy
  ├── val_chain_2d.npy
  ├── test_sequences.npy
  ├── test_labels.npy
  └── test_chain_2d.npy
```

**Symbols:** SPXW, SPY, QQQ, IWM, TLT

**VIX Features:**
```
/workspace/data/tier3_vix_v4/VIXW/
  ├── val_vix_features.npy     # (37912, 4, 10)
  ├── val_vix_labels.npy
  ├── test_vix_features.npy    # (37913, 4, 10)
  └── test_vix_labels.npy
```

---

## Model Weights Summary

| Stage | Models | Total Params | Size |
|-------|--------|--------------|------|
| Stage 1 | 35 | ~23.8M | 23.8 MB |
| Stage 2 | 7 | ~7.6K | 30 KB |
| Stage 3 | 1 | ~39K | 156 KB |
| **Total** | **43** | **~23.8M** | **~24 MB** |

---

## Performance Breakdown

### Stage 1 (h30 SPXW only)
| Agent | AUC | F1 |
|-------|-----|-----|
| A | 0.685 | 0.710 |
| B | 0.689 | 0.715 |
| C | 0.687 | 0.712 |
| K | 0.691 | 0.718 |
| T | 0.688 | 0.714 |
| Q | 0.690 | 0.716 |
| 2D | 0.669 | 0.710 |

### Stage 2 (h30 Cross-Symbol Fusion)
| Agent | AUC | F1 |
|-------|-----|-----|
| A | 0.685 | 0.707 |
| B | 0.721 | 0.735 |
| C | 0.722 | 0.736 |
| K | 0.719 | 0.733 |
| T | 0.718 | 0.732 |
| Q | 0.720 | 0.734 |
| 2D | 0.669 | 0.710 |

### Stage 3 (h30 VIX-Gated Meta)
| Method | Test AUC | Test F1 | Test Acc |
|--------|----------|---------|----------|
| **VIX-Gated** | **0.7221** | **0.7162** | 0.6100 |
| LogReg C=0.1 | 0.7212 | 0.7091 | 0.6622 |
| Simple Avg | 0.7181 | 0.7189 | 0.6371 |

---

## Version Control

### v1.0 (2026-03-09)
- Initial production release
- h30 traditional pipeline (no TLT gating at Stage 2)
- VIX-gated Stage 3 meta-ensemble
- Test AUC: 0.7221
- 43 model files (35 Stage 1 + 7 Stage 2 + 1 Stage 3)

---

## Deployment Checklist

- [x] All 43 model checkpoints copied
- [x] Model architecture code included (hybrid51_models/)
- [x] Utility functions included (hybrid51_utils/)
- [x] Production inference script created
- [x] Configuration file with all parameters
- [x] Complete README with usage examples
- [x] Performance metrics documented
- [x] Data requirements specified
- [x] Dependencies listed

---

## Quick Verification

```bash
# Count model files
find models/ -name "*.pt" | wc -l        # Should be 43
find models/ -name "*.npz" | wc -l       # Should be 1
find models/ -name "*.json" | wc -l      # Should be 1

# Total: 45 files in models/

# Check directory structure
tree -L 2 Final_production_model/

# Test inference (requires data)
cd Final_production_model
python scripts/simple_inference.py --split test --output test_predictions.npz
```

---

## Contact & Support

For deployment assistance or questions about this production model, refer to:
- `README.md` for detailed usage instructions
- `config/production_config.json` for model parameters
- `models/stage3/metrics.json` for full performance breakdown

**Model trained by:** Hybrid51 Pipeline v5  
**Training completed:** 2026-03-09  
**Production ready:** Yes ✓
