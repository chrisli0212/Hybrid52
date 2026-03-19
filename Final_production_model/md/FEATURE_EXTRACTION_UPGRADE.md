# Feature Extraction Upgrade - March 13, 2026

## Summary

Replaced the custom feature extraction logic in `prediction_service.py` with the **original MasterFeatureExtractor** from the Hybrid51 training code. This fixes a critical feature layout mismatch and significantly improves feature coverage.

---

## Critical Issues Fixed

### 1. Feature Layout Mismatch (CRITICAL)
**Problem:** The original `extract_325_features` function used a completely different feature order than the trained model expected:
- Old: Greeks at dims 0-49 (50 features)
- Training: Greeks at dims 0-74 (75 features)
- Old: Gamma at dims 240-269
- Training: Gamma at dims 75-104

**Impact:** Model was receiving features in wrong positions, causing poor prediction accuracy.

**Fix:** Now uses original `MasterFeatureExtractor` with correct feature layout matching training.

### 2. Missing Advanced Greeks
**Problem:** Old extraction only used 5 basic Greeks (delta, gamma, theta, vega, lambda).

**Fix:** Now properly extracts and buckets 7 Greeks including:
- **vanna** - Sensitivity to volatility changes (dV/dσ)
- **charm** - Theta decay rate (dΘ/dt)

These 2nd-order Greeks are critical for options trading models.

### 3. Missing Feature Groups
**Problem:** Several feature groups were incomplete or missing:
- Vanna/Charm group: 0% → **100%** ✅
- Walls Positioning: ~30% → **100%** ✅
- Gamma Exposure: ~50% → **97%** ✅

---

## Results

### Coverage Improvement
```
Before: 37.5% average (122/325 features)
After:  53.6% average (178/325 features)
Gain:   +16 percentage points (+43% relative improvement)
```

### Per-Group Coverage (After)
```
✓ Vanna/Charm:           100.0%  (20/20)  ← Was 0%!
✓ Gamma Exposure:         96.7%  (29/30)
✓ Walls Positioning:     100.0%  (20/20)
✓ Microstructure:         75.0%  (15/20)
✓ Greek by Strike:        57.3%  (43/75)
✓ Cross-Strike:           53.3%  (8/15)
✓ IV Surface:             48.0%  (12/25)
○ Smart Money:            33.3%  (5/15)
○ Volume Anomaly:         33.3%  (4/12)
○ Time Decay:             33.3%  (5/15)
○ Sentiment/Regime:       30.0%  (6/20)
○ Quote Pressure:         27.8%  (5/18)
○ Flow/Volume:            20.0%  (6/30)
○ Trade Conditions:        0.0%  (0/10)
```

---

## Changes Made

### 1. Updated Imports (`prediction_service.py` lines 44-56)
```python
# Add Hybrid51 directory for preprocessing imports
HYBRID51_DIR = SCRIPT_DIR.parent / "Hybrid51" / "6. Hybrid51_new stage"
if str(HYBRID51_DIR) not in sys.path:
    sys.path.insert(1, str(HYBRID51_DIR))

# Import original feature extractors
from hybrid51_preprocessing.master_extractor_v2 import MasterFeatureExtractor, ExtractionResult
```

### 2. Updated FeatureBridge.__init__ (lines ~202-218)
```python
def __init__(self, seq_len: int = SEQ_LEN, strike_bins: int = STRIKE_BINS):
    # Initialize original master extractor
    self.extractor = MasterFeatureExtractor(
        include_chain_2d=False,  # We handle chain_2d separately for Agent-2D
        include_phase1=True,     # Enable all 325 features (270 base + 55 Phase 1)
        normalize=False          # We apply normalization at model level
    )
    # ... rest of initialization
```

### 3. Simplified extract_325_features (lines ~273-308)
Replaced 300+ lines of custom extraction logic with:
```python
def extract_325_features(self, snap_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Uses original MasterFeatureExtractor to ensure correct feature layout."""
    if snap_df is None or snap_df.empty:
        return np.zeros(FEAT_DIM, dtype=np.float32), 0.0
    
    df = self._adapt_columns(snap_df)
    result: ExtractionResult = self.extractor.extract(
        greek_df=df,
        trade_df=df,  # Snapshot contains both Greek and trade data
        historical_snapshots=None,
        open_interest=None
    )
    
    n_filled = int(np.count_nonzero(result.features))
    completeness = n_filled / FEAT_DIM
    return result.features, min(1.0, completeness)
```

### 4. Installed scipy
```bash
pip install scipy  # Required by original extractors
```

---

## Available vs Missing Features

### ✅ Available Greeks (7 of 13)
All present in live snapshots:
- **1st order:** delta, gamma, vega, theta, lambda
- **2nd order:** vanna, charm

### ❌ Missing Greeks (6 of 13)
Not calculated by theta_fetching_v5.py:
- **1st order:** rho, epsilon
- **2nd order:** vomma, veta, zomma, color

**Note:** These 6 Greeks are less commonly used and contribute only ~25 feature dimensions out of 325 (7.7%). The extractor handles missing Greeks gracefully by setting their features to 0.0.

### 📦 Phase 1 Trade/Quote Features
Partially available (depends on trade data quality):
- **Smart Money Detection:** 33% coverage (5/15)
- **Volume Anomaly:** 33% coverage (4/12)
- **Quote Pressure:** 28% coverage (5/18)
- **Trade Conditions:** 0% coverage (0/10) - requires NBBO/exchange routing data

---

## Next Steps for Further Improvement

### 1. Improve Flow/Volume Coverage (Currently 20%)
**Current bottleneck:** Requires:
- Aggressive/passive trade classification
- Trade size buckets (small/medium/large/block)
- Time-weighted flow metrics
- Dark pool vs lit venue data

**Options:**
- Add aggressor side detection in `theta_fetching_v5.py`
- Classify trades by size using volume percentiles
- Track flow over 1m/5m/15m/30m windows

### 2. Add Missing 1st-Order Greeks (rho, epsilon)
**Why it matters:**
- rho: Interest rate sensitivity (important for longer DTEs)
- epsilon: Dividend sensitivity (important for SPY/SPX)

**How to add:**
- Modify theta fetcher to calculate using Black-Scholes Greeks library
- Or import from existing Greeks calculation module

### 3. Improve Trade Conditions Coverage (Currently 0%)
**Requires:**
- NBBO comparison data
- Exchange routing information
- Trade condition flags (sweep, intermarket sweep, etc.)

These are harder to obtain from standard data feeds.

### 4. Historical Snapshots for Temporal Features
**Current limitation:** Time Decay and Sentiment/Regime groups are low (30-33%) because they benefit from historical comparison.

**Solution:**
- Store rolling window of snapshots in memory
- Pass `historical_snapshots` parameter to extractor
- Enable temporal acceleration and trend detection features

---

## Verification

To verify the upgrade is working:

```bash
# Check latest predictions
tail -3 daily_data/prediction.csv | cut -d',' -f24

# Expected output: 0.534, 0.537, 0.540 (53-54%)
# Old values were: 0.375, 0.375, 0.375 (37-38%)
```

To view feature breakdown:
```python
from hybrid51_preprocessing.feature_config_v2 import FEATURE_GROUPS
# ... then inspect result.features per group
```

---

## Impact on Model Predictions

### Accuracy Impact
- **Feature alignment fix is critical** - model now receives features in correct positions
- **43% more features** should improve prediction quality significantly
- **Advanced Greeks (vanna/charm)** enable better volatility regime detection

### Expected Improvements
- Better prediction stability (less random fluctuation)
- More accurate confidence scores
- Better regime detection (bull/bear/neutral)
- Improved dealer positioning signals

### Monitoring
Watch these metrics in the dashboard:
1. Feature Completeness should stay 53-54% (up from 37%)
2. Agent probabilities should show more variation (not stuck at similar values)
3. Confidence decomposition should show higher agreement scores

---

## Technical Details

### Original Extractor Architecture
```
MasterFeatureExtractor
├── GreekFeatureExtractor (75 features)
│   ├── 5 delta buckets × 13 Greeks = 65
│   ├── ATM Greeks (7) = 7
│   └── Skew metrics = 3
├── GammaExposureExtractor (30 features)
├── VannaCharmExtractor (20 features) ← NEW!
├── IVSurfaceExtractor (25 features)
├── FlowVolumeExtractor (30 features)
├── MicrostructureExtractor (20 features)
├── WallsPositioningExtractor (20 features)
├── CrossStrikeExtractor (15 features)
├── TimeDecayExtractor (15 features)
├── SentimentRegimeExtractor (20 features)
└── Phase 1 (55 features)
    ├── SmartMoneyDetector (15)
    ├── VolumeAnomalyDetector (12)
    ├── TradeConditionAnalyzer (10)
    └── QuotePressureAnalyzer (18)
```

### Files Modified
- `/workspace/Final_production_model/prediction_service.py`
  - Added import for `MasterFeatureExtractor`
  - Updated `FeatureBridge.__init__` to initialize extractor
  - Replaced `extract_325_features` method (300 lines → 30 lines)
  - Fixed sys.path order for correct module resolution

### Dependencies Added
- `scipy` - Required by VolumeAnomalyDetector and other Phase 1 extractors

---

**Date:** March 13, 2026  
**Status:** ✅ Complete and Verified  
**Prediction Service:** Running with upgraded extraction  
**Dashboard:** Ready to display improved metrics
