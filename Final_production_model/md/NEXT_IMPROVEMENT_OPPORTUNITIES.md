# Next Improvement Opportunities - Feature Coverage Roadmap

## Current State: 53.6% Coverage (178/325 features)

---

## Opportunity 1: Add Missing Greeks → 58% Coverage (+15 features)

### Missing 1st-Order Greeks (7 features)
**rho** and **epsilon** used in:
- Greek by Strike bucketing: 5 buckets × 2 Greeks = 10 dims
- ATM Greeks: 2 dims  
- Impact: Moderate for long-dated options and dividend plays

**How to add:**
```python
# In theta_fetching_v5.py, add to Greeks calculation
rho = S * T * stats.norm.cdf(d2)  # Interest rate sensitivity
epsilon = -S * T * stats.norm.cdf(d1)  # Dividend sensitivity
```

### Missing 2nd-Order Greeks (8 features)
**vomma, veta, zomma, color** used in:
- Greek by Strike bucketing: 5 buckets × 4 Greeks = 20 dims
- Minor impact: These are exotic Greeks rarely used in practice

**ROI:** Low - complex to calculate, marginal benefit

---

## Opportunity 2: Enhance Flow/Volume Features → 65% Coverage (+24 features)

### Currently Missing (24 of 30 features):
1. **Aggressor Side Classification**
   - Passive vs Aggressive volume (3 features)
   - Sweep volume detection (1 feature)
   - Call/Put aggression ratios (2 features)

2. **Trade Size Distribution**
   - Small/Medium/Large/Block volume (4 features)
   - Size-weighted metrics (2 features)

3. **Time-Weighted Flow**
   - 1min/5min/15min/30min rolling flow (4 features)
   - Flow acceleration/deceleration (2 features)

4. **Dark Pool vs Lit**
   - Venue routing (2 features)

5. **Premium Flow**
   - Premium by aggression side (2 features)
   - VWAP premium (1 feature)

**How to add:**
```python
# In theta_fetching_v5.py
# 1. Track bid/ask at trade time to detect aggressor
# 2. Classify trades: trade_price >= ask → aggressive buy
# 3. Calculate size percentiles for trade bucketing
# 4. Maintain rolling windows for temporal flow
```

**ROI:** High - Flow analysis is critical for options prediction

---

## Opportunity 3: Enable Historical Context → 70% Coverage (+16 features)

### Time Decay Features (Currently 33%, could reach 80%)
Requires historical snapshots to calculate:
- Theta acceleration (comparing t vs t-1)
- Gamma acceleration
- Charm acceleration
- Weighted decay metrics

### Sentiment/Regime Features (Currently 30%, could reach 60%)
Requires historical data for:
- IV expansion/contraction trends
- Momentum (1d/5d/20d)
- Trend strength
- Stress/fear indicators
- Correlation dynamics

**How to add:**
```python
# In FeatureBridge, maintain rolling history
self._snapshot_history = deque(maxlen=100)  # Last 100 snapshots

# Pass to extractor
result = self.extractor.extract(
    greek_df=df,
    trade_df=df,
    historical_snapshots=list(self._snapshot_history),  # Enable temporal features
    open_interest=current_oi
)
```

**ROI:** Medium - Helps with regime detection, but requires memory/storage

---

## Opportunity 4: Improve Trade Conditions → 72% Coverage (+10 features)

### Currently Missing (All 10 features):
- NBBO comparison metrics
- Intermarket sweep order (ISO) detection
- Exchange routing patterns
- Multi-leg trade detection
- Trade condition flags analysis

**Requirements:**
- Full NBBO data feed
- Exchange identifiers
- Trade condition codes

**ROI:** Low-Medium - Requires premium data feed, moderate benefit

---

## Priority Roadmap

### 🔥 High Priority (Should Do)
1. **Flow/Volume Enhancement** → +24 features (+7% coverage)
   - Effort: Medium (1-2 days)
   - Impact: High (critical for options trading)
   - Requires: Aggressor detection, size classification, rolling windows

### 🟡 Medium Priority (Nice to Have)
2. **Historical Context** → +16 features (+5% coverage)
   - Effort: Low (4-6 hours)
   - Impact: Medium (better regime detection)
   - Requires: Snapshot history buffer in memory

3. **Missing 1st-Order Greeks** → +7 features (+2% coverage)
   - Effort: Low (2-3 hours)
   - Impact: Medium (improves long-dated options)
   - Requires: Add rho/epsilon to Greeks calculation

### ⚪ Low Priority (Optional)
4. **Trade Conditions** → +10 features (+3% coverage)
   - Effort: High (requires premium data)
   - Impact: Low-Medium
   - Requires: NBBO feed, exchange routing data

5. **Exotic 2nd-Order Greeks** → +8 features (+2% coverage)
   - Effort: Medium
   - Impact: Low (rarely useful in practice)
   - Requires: Complex Greek calculations

---

## Estimated Maximum Coverage

With all realistic improvements:
```
Current:              53.6%  (178/325)
+ Flow/Volume:        60.9%  (+24)
+ Historical Context: 65.8%  (+16)
+ rho/epsilon:        67.9%  (+7)
────────────────────────────
Realistic Maximum:    67.9%  (225/325)
```

The remaining 100 features (32%) would require:
- Premium data feeds (NBBO, exchange routing)
- Exotic Greeks calculation (vomma, veta, zomma, color)
- Advanced trade classification metadata

**Conclusion:** 68% coverage is a realistic target with moderate effort. Beyond that requires significant infrastructure investment.

---

**Created:** March 13, 2026  
**Current Coverage:** 53.6%  
**Realistic Target:** 68%
