# Forensic Study: Stage 2/3 Performance Analysis

**Date:** 2026-03-01  
**Objective:** Analyze prior Stage 2/3 training artifacts from `/workspace/Hybrid51/5. hybrid51_stage3/checkpoints` to inform improvements in `/workspace/Hybrid51/6. Hybrid51_new stage`.

---

## Executive Summary

**Best Prior Result:** Stage 3 v2 MLP meta-learner achieved **acc=0.5872, F1=0.622, AUC=0.6173** on test set.

**Key Success Factors:**
1. **29 raw cross-symbol feature diffs** (not engineered aggregates)
2. **Focal loss (γ=2.0)** with per-pair dynamic `alpha` from training UP fraction
3. **15-dim enriched meta features** combining pair probs + disagreement + SPXW core logits
4. **MLP meta-learner** trained on val split (80/20 train_meta/val_meta) with early stopping
5. **Threshold optimization** on validation set for F1 maximization
6. **Prefix alignment** (first N samples) for consistent cross-pair stacking

**Critical Failure Mode Avoided:**
- **Data leakage:** Prior v1 trained meta on train set → overfitting. v2 fixed by training meta on val only.

---

## Prior Checkpoint Analysis

### Stage 1 (Binary Agents)
- **Symbols:** SPXW, SPY, QQQ, IWM, TLT, VIXW
- **Agents:** A, B, K, C, T, Q, 2D (7 agents per symbol)
- **Horizon:** 15 min
- **Checkpoints:** 42 `.pt` files (6 symbols × 7 agents)
- **Status:** Available and functional

### Stage 2 v5 (Pair Fusion)
**Architecture:**
```
Input: 7 agent logits (target) + 7 agent logits (pair) + 29 cross-symbol diffs = 43 dims
Model: 2-layer MLP (128→64→1) with LayerNorm + GELU + Dropout
Loss: Focal (γ=2.0, α=UP_frac_train)
```

**Cross-Symbol Features (29 dims):**
```python
CROSS_SYMBOL_FEATURE_INDICES = [
    125, 126, 127, 128, 129, 130, 131,  # iv_by_moneyness (7)
    132, 133, 134,                        # iv_term_structure (3)
    137, 138,                             # vol_skew (2)
    150, 151, 152, 153,                   # call_put_ratios (4)
    167, 168, 169,                        # flow_direction (3)
    250, 251, 252,                        # sentiment_scores (3)
    257, 258, 259,                        # trend_stress (3)
    95, 96, 97, 98,                       # net_gamma (4)
]
# Computed as: target_static[:, indices] - pair_static[:, indices]
```

**Results (test set, threshold=0.5):**
| Pair       | Acc    | F1     | AUC    | Opt Thr |
|------------|--------|--------|--------|---------|
| SPXW-SPY   | 0.5849 | 0.5922 | 0.6076 | 0.30    |
| SPXW-QQQ   | 0.5850 | 0.5968 | 0.6074 | 0.30    |
| SPXW-IWM   | 0.5866 | 0.5962 | 0.6076 | 0.30    |
| SPXW-TLT   | 0.5867 | 0.5977 | 0.6109 | 0.31    |
| SPXW-VIXW  | 0.5907 | **0.6807** | 0.6280 | 0.48 |

**Key Observations:**
- VIXW shows anomalously high F1 (0.68 vs ~0.60 for others) but requires threshold=0.48 → likely overfitting to class imbalance
- All main pairs (SPY/QQQ/IWM/TLT) converge to similar performance
- Optimal thresholds cluster around 0.30 (not 0.50) → suggests model outputs are miscalibrated

### Stage 3 v2 (Meta-Learner)
**Architecture:**
```
Input: 15-dim enriched features
  - 5 pair probs (SPY, QQQ, IWM, TLT, VIXW)
  - 1 mean prob (consensus)
  - 1 std prob (uncertainty)
  - 1 max-min spread (polarization)
  - 4 pairwise diffs (SPY-QQQ, SPY-TLT, QQQ-IWM, IWM-TLT)
  - 1 agreement count (% predicting UP)
  - 1 SPXW core mean (avg of 7 agent logits)
  - 1 SPXW core std (agent disagreement)

Meta-Learner Options:
  1. LogisticRegression(C=0.01, class_weight='balanced')
  2. MLP (15→32→16→1) with GELU + Dropout(0.3) + Focal Loss
```

**Training Protocol:**
- **Train meta on VAL set** (not train) to avoid leakage
- Split val 80/20 → train_meta/val_meta for early stopping
- Evaluate all methods on TEST set
- Select best by val_meta performance

**Results (test set):**
| Method              | Acc    | F1     | AUC    |
|---------------------|--------|--------|--------|
| Simple Avg (4)      | 0.5879 | 0.5995 | 0.6118 |
| LogReg (C=1.0)      | 0.5891 | 0.5835 | 0.6168 |
| LogReg (C=0.1)      | 0.5883 | 0.5889 | 0.6180 |
| LogReg (C=0.01)     | 0.5885 | 0.5957 | 0.6184 |
| **MLP v2**          | **0.5872** | **0.6220** | **0.6173** |

**Winner:** MLP v2 (best F1 despite slightly lower accuracy)

---

## Design Comparison: Old vs New

### What Was Good (Kept)
✅ **29 raw cross-symbol diffs** (not engineered aggregates)  
✅ **Focal loss with dynamic alpha** (per-pair UP fraction)  
✅ **15-dim enriched meta features** (pair probs + core logits + disagreement)  
✅ **MLP meta-learner** with early stopping on val split  
✅ **Threshold optimization** on validation set  
✅ **Prefix alignment** ([:n]) for sample consistency  

### What Was Bad (Fixed)
❌ **Old:** Engineered cross features (cosine sim, L2 norm, etc.) → **New:** Raw diffs only  
❌ **Old:** Hardcoded focal alpha=0.52 → **New:** Dynamic alpha from train UP%  
❌ **Old:** Best epoch by val F1 with threshold sweep → **New:** Best epoch by val acc+F1 at fixed 0.5  
❌ **Old:** Missing core logits in Stage2 output → **New:** Save val/test_core_logits.npz  
❌ **Old:** Suffix alignment ([-n:]) → **New:** Prefix alignment ([:n])  
❌ **Old:** Test-based model selection → **New:** Val-based selection  

### What Was Missing (Added)
➕ **Multiple logistic C values** (0.01, 0.1, user-specified) for robust selection  
➕ **Auto-selection between LogReg and MLP** based on val performance  
➕ **Comprehensive val/test metrics** saved per method  
➕ **Runtime checks** for missing core logits in Stage2 outputs  
➕ **Robust pair coverage validation** in Stage3  

---

## Recommendations for Production

### Immediate Actions
1. **Drop VIXW from Stage2/3 pairs** (anomalous behavior, likely harmful)
2. **Run Stage1 for all symbols** (SPXW, SPY, QQQ, IWM, TLT) at horizon=15min
3. **Run Stage2 for 4 pairs** (SPXW-SPY, SPXW-QQQ, SPXW-IWM, SPXW-TLT)
4. **Run Stage3 with --meta=auto** to select best between LogReg and MLP

### Hyperparameter Recommendations
**Stage2:**
```bash
python scripts/stage2/train_stage2_pairs.py \
  --target SPXW \
  --all-pairs \
  --horizon 15 \
  --epochs 50 \
  --batch-size 1024 \
  --lr 5e-4 \
  --device cuda
```

**Stage3:**
```bash
python scripts/stage3/train_stage3_meta.py \
  --target SPXW \
  --horizon 15 \
  --all-pairs \
  --meta auto \
  --meta-epochs 100 \
  --meta-batch-size 2048 \
  --meta-lr 1e-3 \
  --meta-hidden-dim 32 \
  --meta-dropout 0.3 \
  --device cuda
```

### Expected Performance
Based on prior results, expect:
- **Stage2 pairs:** acc ~0.585, F1 ~0.596, AUC ~0.608
- **Stage3 meta:** acc ~0.587, F1 ~0.622, AUC ~0.617

### Risk Mitigation
1. **Verify Stage1 agent diversity** before Stage2 (check pairwise agreement <70%)
2. **Monitor focal alpha values** (should be ~0.52 for balanced data)
3. **Check threshold calibration** (optimal thr should be near 0.5, not 0.3)
4. **Validate sample alignment** (all pairs should have same N after alignment)

---

## Code Changes Summary

### `/workspace/Hybrid51/6. Hybrid51_new stage/scripts/stage2/train_stage2_pairs.py`
**Changes:**
- Added `CROSS_SYMBOL_FEATURE_INDICES` (29 dims) from proven v5
- Changed cross features from engineered to raw diffs: `target[:, idx] - pair[:, idx]`
- Changed focal alpha from hardcoded 0.52 to `train_up_frac = float(np.mean(y_train))`
- Changed best epoch selection from val F1 (with threshold sweep) to `val_acc + val_f1` at fixed 0.5
- Added `val_core_logits` and `test_core_logits` to saved .npz files
- Changed alignment from suffix `[-n:]` to prefix `[:n]`
- Changed `n_cross_features` from hardcoded 29 to `N_CROSS_FEATURES` constant

**Status:** ✅ Compiles cleanly

### `/workspace/Hybrid51/6. Hybrid51_new stage/scripts/stage3/train_stage3_meta.py`
**Changes:**
- Added `build_enriched_meta_features()` for 15-dim features (pair probs + core logits + disagreement)
- Added MLP meta-learner with focal loss + early stopping
- Added auto-selection between LogReg and MLP based on val performance
- Added multiple logistic C values (0.01, 0.1, user-specified)
- Changed model selection from test metrics to val metrics
- Added comprehensive val/test metrics per method
- Fixed focal loss alpha bug in MLP (`alpha_t` computation)
- Changed alignment from suffix `[-n:]` to prefix `[:n]`
- Added runtime checks for missing core logits in Stage2 outputs
- Added validation for full main-pair coverage (SPY/QQQ/IWM/TLT required)

**Status:** ✅ Compiles cleanly

---

## Conclusion

The new Stage2/3 design in `/workspace/Hybrid51/6. Hybrid51_new stage` now incorporates all proven best practices from prior successful iterations while fixing identified failure modes. The scripts are ready for execution once Stage1 checkpoints are available.

**Next Steps:**
1. Await user confirmation to proceed with data processing and training
2. Execute Stage1 → Stage2 → Stage3 pipeline
3. Monitor metrics against expected baselines
4. Iterate if performance deviates significantly from expectations
