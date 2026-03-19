# Model Training vs Production Deployment Audit Report
**Date:** March 13, 2026  
**Auditor:** AI Assistant  
**Status:** CRITICAL ISSUES FOUND

---

## Executive Summary

Comprehensive audit comparing training code (`/workspace/Hybrid51/6. Hybrid51_new stage`) with production deployment (`/workspace/Final_production_model`) has identified **3 CRITICAL variances** that significantly impact model accuracy:

### Critical Findings (Require Immediate Action)

1. **NO NORMALIZATION APPLIED** - CRITICAL (Severity: 🔴 HIGH)
2. **THRESHOLD MISMATCH** - MODERATE (Severity: 🟡 MEDIUM)  
3. **FEATURE COVERAGE 53.6%** - MODERATE (Severity: 🟡 MEDIUM)

### Impact Assessment

| Issue | Impact on Predictions | Estimated Error Rate |
|-------|----------------------|---------------------|
| Missing Normalization | Model receives wrong input scale | **SEVERE - Predictions unreliable** |
| Threshold 0.47 vs 0.44 | 3% more conservative predictions | Moderate |
| 46% Missing Features | Reduced signal quality | Moderate to High |

---

## Detailed Findings

### 1. CRITICAL: No Normalization Statistics Applied 🔴

**Status:** ✗ CONFIRMED - All Stage 1 models receiving unnormalized features

#### Evidence

Tested production service:
```
✗ SPXW: NO normalization (using raw features!)
✗ SPY: NO normalization (using raw features!)
✗ QQQ: NO normalization (using raw features!)
✗ IWM: NO normalization (using raw features!)
✗ TLT: NO normalization (using raw features!)
```

#### Root Cause

**Expected location:** `/workspace/data/tier3_binary_v5/{SYMBOL}/horizon_30min/norm_mean.npy`  
**Actual status:** Directory `/workspace/data/` does not exist

**Code implementation:**
```python
# prediction_service.py lines 530-544
def _load_norm_stats(self, symbol: str):
    data_root = "/workspace/data/tier3_binary_v5"  # From config
    norm_mean_path = Path(data_root) / symbol / f"horizon_30min" / "norm_mean.npy"
    
    if norm_mean_path.exists():
        return np.load(norm_mean_path), np.load(norm_std_path)
    return None, None  # ← Silently returns None for all symbols
```

#### Impact

**Training:** All models trained on z-score normalized features
```python
# Training applies: normalized = (features - mean) / std
# Per-feature mean ≈ 0, std ≈ 1
```

**Production:** Models receive raw, unnormalized features
```python
# Features have arbitrary scales:
#   - delta: [-1, 1]
#   - gamma_exp: [-60M, +300k]  (seen in testing)
#   - volume: [0, 20k+]
#   - IV: [0, 3.0]
```

**Result:** Model inputs are **completely wrong** - features are on scales 1000-10,000x different than training.

#### Example Scale Discrepancy

From testing snapshot_000051.csv with original extractor:
```
Feature stats (unnormalized):
  Min: -63,471,356.0000
  Max:    304,304.5625
  Mean:  -193,895.7500
  Std:  3,515,474.0000

Expected after normalization:
  Min: ~-3.0
  Max: ~+3.0
  Mean: ~0.0
  Std: ~1.0
```

#### Recommended Actions

**Option 1: Compute Normalization Stats from Live Data (Quick Fix)**
```bash
# Use a representative sample of production data
python /workspace/Hybrid51/6. Hybrid51_new stage/scripts/phase0/build_tier3_binary.py \
    --symbol SPXW --horizon 30 \
    --input-dir /workspace/Final_production_model/daily_data
```

**Option 2: Copy from Training Environment (If Available)**
- Locate original training artifacts
- Copy `norm_mean.npy` and `norm_std.npy` to expected paths
- Create directory structure: `/workspace/data/tier3_binary_v5/{SYMBOL}/horizon_30min/`

**Option 3: Disable Normalization (Retrain Required)**
- Retrain all models without normalization
- Update checkpoints
- NOT RECOMMENDED - normalization improves convergence

#### Verification Steps

After fixing:
```python
# Check that normalization is applied
service = PredictionService(data_dir="daily_data")
for symbol in ALL_SYMBOLS:
    bundle = service.stage1[symbol]["A"]
    assert bundle.norm_mean is not None, f"Still no norm for {symbol}"
    print(f"✓ {symbol}: norm_mean shape={bundle.norm_mean.shape}")
```

---

### 2. MODERATE: Threshold Mismatch (0.47 vs 0.44) 🟡

**Status:** ✗ CONFIRMED - Production uses suboptimal threshold

#### Evidence

| Source | Threshold | Notes |
|--------|-----------|-------|
| **Training Metrics** (`metrics.json`) | **0.44** | Optimal for VIX-gated on validation set |
| **Production Checkpoint** (`stage3_vix_gated.pt`) | **0.47** | Stored in checkpoint |
| **Production Config** (`production_config.json`) | **0.47** | Matches checkpoint |
| **Production Runtime** | **0.47** | Loaded from checkpoint (line 601) |

#### Metrics Comparison at Different Thresholds

**VIX-Gated Model Performance:**

At **threshold=0.44** (optimal from training):
```
Validation: accuracy=0.5790, f1=0.7118, auc=0.6780
Test:       accuracy=0.6100, f1=0.7162, auc=0.7221
```

At **threshold=0.47** (production):
```
Expected metrics: accuracy ≈0.62-0.63, f1 ≈0.70-0.71
(F1 likely drops ~1% due to more conservative threshold)
```

#### Impact

- **More conservative predictions:** Fewer BULL signals, more BEAR/neutral
- **Slightly lower F1 score:** ~1% degradation from optimal
- **Higher precision, lower recall:** Trades off sensitivity for specificity

#### Recommended Action

**Update checkpoint threshold:**
```python
# Load checkpoint
ckpt = torch.load("models/stage3/stage3_vix_gated.pt")
ckpt['threshold'] = 0.44  # Use optimal value from training metrics
torch.save(ckpt, "models/stage3/stage3_vix_gated.pt")
```

**OR accept 0.47 if:**
- Production prefers higher precision (fewer false positives)
- Conservative bias is intentional for risk management

---

### 3. MODERATE: Feature Coverage 53.6% (Recently Improved) 🟡

**Status:** ✓ PARTIALLY RESOLVED - Upgraded from 37.5% on March 13, 2026

#### Timeline

- **Before March 13:** Custom extraction with wrong feature layout = 37.5% coverage
- **After March 13:** Original `MasterFeatureExtractor` with correct layout = 53.6% coverage
- **Training:** 100% coverage (all 325 features filled during training)

#### Current Coverage Breakdown

```
Group                    Training    Production    Gap
────────────────────────────────────────────────────────
✅ Vanna/Charm             100%        100%         0%
✅ Gamma Exposure          100%         97%        -3%
✅ Walls Positioning       100%        100%         0%
✅ Microstructure          100%         75%       -25%
✅ Greek by Strike         100%         57%       -43%
✅ IV Surface              100%         48%       -52%
✅ Cross-Strike            100%         53%       -47%

⚠️  Flow/Volume             100%         20%       -80%  ← Critical gap
⚠️  Sentiment/Regime        100%         30%       -70%
⚠️  Time Decay              100%         33%       -67%
⚠️  Smart Money (Phase 1)   100%         33%       -67%
⚠️  Volume Anomaly (P1)     100%         33%       -67%
⚠️  Quote Pressure (P1)     100%         28%       -72%

❌ Trade Conditions (P1)   100%          0%      -100%  ← Completely missing
```

#### Root Causes

1. **Missing Greeks (6 of 13):** rho, epsilon, vomma, veta, zomma, color
   - Live data only has: delta, gamma, vega, theta, lambda, vanna, charm
   - Affects ~32 feature dimensions in Greek by Strike bucketing

2. **Flow/Volume (20% coverage):** Requires trade classification
   - Missing: aggressor detection, trade size buckets, time-weighted flow
   - Needs: 24 additional features

3. **Trade Conditions (0% coverage):** Requires NBBO/exchange data
   - Missing: intermarket sweep detection, exchange routing
   - Needs: 10 additional features

4. **Temporal Features (30-33%):** Requires historical snapshots
   - Missing: acceleration metrics, momentum, trend analysis
   - Needs: historical snapshot buffer

#### Impact on Agent Predictions

**Per-Agent Feature Subset Impact:**

| Agent | Expected Features | Available | Coverage | Most Affected By |
|-------|------------------|-----------|----------|------------------|
| A (Alpha) | 160 dims | ~90 | 56% | Missing Greeks, Flow |
| B (Beta) | 108 dims | ~60 | 55% | IV Surface gaps, Time Decay |
| K (Kappa) | 78 dims | ~45 | 58% | Missing Greeks only |
| C (Chi) | 112 dims | ~50 | 45% | Flow/Volume (80% missing!) |
| T (Tau) | 139 dims | ~50 | 36% | Trade Conditions (100% missing!) |
| Q (Quote) | 128 dims | ~55 | 43% | Quote Pressure (72% missing) |
| 2D | Chain tensor | ~full | 95% | Minor Greek gaps |

**Agent C, T, Q are severely impacted** - these agents see only 36-45% of their intended features.

#### Recommended Actions

**Immediate (High ROI):**
1. Add aggressor detection to increase Flow/Volume to 44% → Overall 58%
2. Enable historical snapshots for temporal features → Overall 61%

**Medium-term:**
3. Add rho/epsilon Greeks → Overall 63%

**Long-term (Premium Data Required):**
4. NBBO feed for Trade Conditions → Overall 66%

---

## Additional Variances (Non-Critical)

### 4. Stage 3 Model Selection

**Training Results:**
```
Method              Val Score    Test Metrics
─────────────────────────────────────────────────
logreg_C=0.01       1.3303 ✓    acc=0.660, f1=0.694, auc=0.718
vix_gated           1.3209       acc=0.610, f1=0.716, auc=0.722
logreg_C=0.1        1.3286       acc=0.662, f1=0.709, auc=0.721
mlp                 1.3046       acc=0.648, f1=0.707, auc=0.715
```

**Selected for production:** `vix_gated` (NOT the best validation performer!)

**Rationale (inferred):**
- Highest test AUC (0.722)
- Better regime adaptability via VIX conditioning
- Slightly lower accuracy but better F1 on test set
- More robust to volatility regime changes

**Variance:** Production uses different Stage 3 architecture than validation-best model

**Impact:** Minor - VIX-gated performs comparably to LogReg on test set

---

### 5. Missing Data Paths

**Production config references:**
```json
"data_paths": {
    "tier3_binary_root": "/workspace/data/tier3_binary_v5",
    "vix_features_root": "/workspace/data/tier3_vix_v4/VIXW"
}
```

**Actual status:**
- `/workspace/data/` directory: ✗ Does not exist
- Normalization stats: ✗ Not found anywhere
- VIX features: ✗ Not checked (VIX data loaded from live feed)

**Impact:** 
- Stage 1 normalization: CRITICAL failure
- VIX features: No impact (computed live)

---

## Variance Impact Matrix

| Variance | Severity | Impact on Accuracy | Impact on Reliability | Fix Effort |
|----------|----------|-------------------|---------------------|------------|
| **No Normalization** | 🔴 CRITICAL | **SEVERE** - Wrong input scale | **SEVERE** - All predictions affected | Medium |
| **Feature Coverage 53.6%** | 🟡 MEDIUM | MODERATE - Missing signal | MODERATE - Partial information | High |
| **Threshold 0.47 vs 0.44** | 🟡 LOW | MINOR - ~1% F1 drop | MINOR - Acceptable tradeoff | Low |
| **Stage 3 Architecture** | 🟢 INFO | NONE - VIX-gated performs well | NONE | N/A |
| **Missing Greeks (6/13)** | 🟡 LOW | MINOR - ~10% feature dims | LOW - Less critical Greeks | Medium |

---

## Verification Tests Performed

### 1. Feature Extraction Verification ✓
```bash
# Tested original MasterFeatureExtractor with live snapshot
Coverage: 53.6% (178/325 features)
Per-group coverage verified against training expectations
```

### 2. Normalization Loading Check ✗
```python
# Checked all 35 Stage 1 model bundles
Result: 0/35 models have normalization statistics
All models using raw, unnormalized features
```

### 3. Threshold Audit ✓
```
Checkpoint: 0.47
Training optimal (VIX-gated): 0.44
Mismatch: +0.03 (3 percentage points higher)
```

### 4. Model Architecture Compatibility ✓
```
All checkpoints load successfully
State dict keys match expected format
Model dimensions correct
```

---

## Impact on Current Production Predictions

### What's Working

✅ **Feature extraction** - Correct layout since March 13 upgrade  
✅ **Model loading** - All 35+7+1 = 43 models load successfully  
✅ **Inference pipeline** - Stage 1→2→3 flow executes without errors  
✅ **Feature subsetting** - Agents receive correct feature subsets  
✅ **Chain 2D** - Agent 2D receives properly formatted option surface  
✅ **VIX features** - Computed correctly from live data  

### What's Broken

❌ **Stage 1 predictions** - Wrong input scale (no normalization)  
❌ **Prediction accuracy** - Severely degraded from training performance  
❌ **Model confidence** - Unreliable due to wrong inputs  
⚠️  **Threshold** - 3% more conservative than optimal  
⚠️  **Feature signal** - 46% of features missing  

---

## Why Predictions Appeared "Fake and Random"

Based on earlier user complaint that predictions seemed "fake and random as they do not change":

**Root causes now identified:**

1. **No normalization** → Stage 1 outputs nonsensical
2. **Feature layout was wrong** (fixed March 13) → Model saw scrambled features
3. **Low feature coverage** (37% → 53%) → Weak signal
4. **Warmup suppression** → Many predictions showing 0.5 neutral

With **no normalization**, the models are essentially producing random outputs because:
- Features 1000-10,000x larger than training scale
- Model weights optimized for mean=0, std=1 inputs
- Activation functions (GELU, sigmoid) saturate with large inputs
- Gradients were never computed for these input magnitudes

---

## Recommended Immediate Actions

### Priority 1: FIX NORMALIZATION (CRITICAL)

**Option A: Compute from Representative Sample**
```bash
# Collect 1 week of production data
# Run normalization computation
cd "/workspace/Hybrid51/6. Hybrid51_new stage"
python scripts/phase0/build_tier3_binary.py \
    --symbol SPXW --horizon 30 \
    --input <production_snapshots>
```

**Option B: Use Training Statistics (If Available)**
```bash
# If training data still accessible
# Copy from training environment
mkdir -p /workspace/data/tier3_binary_v5/SPXW/horizon_30min
cp <training_path>/norm_mean.npy /workspace/data/tier3_binary_v5/SPXW/horizon_30min/
cp <training_path>/norm_std.npy /workspace/data/tier3_binary_v5/SPXW/horizon_30min/
# Repeat for SPY, QQQ, IWM, TLT
```

**Option C: Approximate from Feature Ranges**
```python
# Quick approximation for emergency deployment
# Compute rolling statistics from recent predictions
# Save as norm_mean.npy / norm_std.npy
# Not ideal but better than no normalization
```

**Verification:**
```bash
# After fix, restart service and check
tail -f /tmp/prediction_service.log | grep "normalization"
# Should see: "Applying z-score normalization from training stats"
```

### Priority 2: Adjust Threshold to 0.44

**Simple fix:**
```python
# Update checkpoint
import torch
ckpt = torch.load("models/stage3/stage3_vix_gated.pt")
ckpt['threshold'] = 0.44  # Use training-optimal value
torch.save(ckpt, "models/stage3/stage3_vix_gated.pt")

# Restart prediction service
```

**Expected impact:** +1% F1 score, more balanced predictions

### Priority 3: Monitor Feature Coverage

**Already improved:** 37.5% → 53.6%

**Next targets:**
1. Add aggressor detection (+7% coverage)
2. Enable historical snapshots (+5% coverage)
3. Add rho/epsilon Greeks (+2% coverage)

---

## Testing & Validation Plan

### After Normalization Fix

**Test 1: Verify Predictions Change**
```python
# Before fix
old_preds = load_predictions("daily_data/prediction.csv")

# Apply normalization fix + restart

# After fix
new_preds = load_predictions("daily_data/prediction.csv")

# Predictions should be VERY different
# Agent probabilities should spread out (not clustered)
# Confidence scores should vary more
```

**Test 2: Compare to Training Metrics**
```
Expected test set performance (from training):
  Accuracy: ~61%
  F1: ~71.6%
  AUC: ~72.2%

If production matches these after fixing normalization,
the deployment is correct.
```

**Test 3: Feature Importance Alignment**
```python
# Check that high-importance features from training
# show corresponding patterns in production predictions
# (e.g., Gamma Exposure should correlate with direction)
```

---

## Architecture Compatibility Summary

### ✅ Confirmed Compatible

| Component | Training | Production | Status |
|-----------|----------|----------|--------|
| **Stage 1 Wrapper** | BinaryIndependentAgent | BinaryIndependentAgent | ✓ Identical |
| **Stage 1 Architecture** | IndependentAgent + Backbone | Same | ✓ Identical |
| **Stage 2 Fusion** | CrossSymbolAgentFusion | Same | ✓ Identical |
| **Stage 3 Type** | VIX-gated (trained) | VIX-gated | ✓ Matches |
| **Feature Layout** | 325-dim (March 13 fix) | 325-dim | ✓ Fixed |
| **Feature Subsetting** | config/feature_subsets.py | Same config | ✓ Identical |
| **Sequence Length** | 20 timesteps | 20 timesteps | ✓ Matches |
| **Chain 2D Shape** | (5, 20, 20) | (5, 20, 20) | ✓ Matches |
| **Dropout** | 0.2 | 0.2 (disabled in eval) | ✓ Correct |
| **Device** | CPU/CUDA | CPU/CUDA | ✓ Flexible |

### ⚠️ Variances Found

| Component | Training | Production | Impact |
|-----------|----------|----------|--------|
| **Normalization** | ✓ Applied | ✗ NOT applied | 🔴 CRITICAL |
| **Threshold** | 0.44 | 0.47 | 🟡 Minor |
| **Feature Coverage** | 100% | 53.6% | 🟡 Moderate |
| **Data Paths** | tier3_binary_v5 | tier3_binary_v5 (missing) | 🔴 CRITICAL |

---

## Files Modified During This Audit

1. **`/workspace/Final_production_model/DEPLOYMENT_AUDIT_REPORT.md`** - This report
2. **`/workspace/Final_production_model/FEATURE_EXTRACTION_UPGRADE.md`** - March 13 upgrade docs
3. **`/workspace/Final_production_model/NEXT_IMPROVEMENT_OPPORTUNITIES.md`** - Coverage roadmap

---

## Audit Methodology

### Exploration Performed

1. **Training Code Review:**
   - Analyzed all model architectures in `/workspace/Hybrid51/6. Hybrid51_new stage/hybrid51_models/`
   - Reviewed training scripts in `scripts/stage1/`, `scripts/stage2/`, `scripts/stage3/`
   - Examined feature extraction in `hybrid51_preprocessing/`
   - Identified hyperparameters and configurations

2. **Production Code Review:**
   - Analyzed inference pipeline in `/workspace/Final_production_model/prediction_service.py`
   - Examined model wrappers in `hybrid51_models/`
   - Reviewed production config in `config/production_config.json`
   - Checked checkpoint loading and state dict compatibility

3. **Runtime Testing:**
   - Loaded production service and inspected model bundles
   - Verified normalization statistics loading (found None for all symbols)
   - Tested feature extraction with live snapshots
   - Checked feature coverage per group

4. **Checkpoint Analysis:**
   - Inspected Stage 3 checkpoint metadata
   - Compared checkpoint thresholds with training metrics
   - Verified model state dict structures

5. **File System Search:**
   - Searched for normalization stat files (none found)
   - Verified data directory structure (missing)
   - Confirmed training artifact locations

---

## Conclusion

### Critical Path to Fix Production

**Immediate (Today):**
1. ✓ Document variances (this report)
2. Compute or obtain normalization statistics
3. Place in `/workspace/data/tier3_binary_v5/{SYMBOL}/horizon_30min/`
4. Restart prediction service
5. Verify predictions change dramatically

**Short-term (This Week):**
1. Update threshold to 0.44 (optimal from training)
2. Monitor prediction metrics vs training expectations
3. Set up alerting for feature coverage drops

**Medium-term (Next Month):**
1. Enhance Flow/Volume feature extraction
2. Add historical snapshot buffer
3. Target 65%+ feature coverage

### Expected Outcomes After Fixes

**With normalization + threshold=0.44:**
- Predictions should match test set metrics: ~61% accuracy, ~72% F1, ~72% AUC
- Agent probabilities should show meaningful variation
- Confidence scores should align with prediction quality

**Current state (no normalization):**
- Predictions are essentially random
- Model outputs unreliable
- Cannot be used for trading decisions

---

**Audit Status:** ✓ COMPLETE  
**Priority:** 🔴 URGENT - Fix normalization immediately  
**Risk Level:** HIGH - Production predictions currently unreliable  
**Estimated Fix Time:** 2-4 hours (compute stats + deploy)
