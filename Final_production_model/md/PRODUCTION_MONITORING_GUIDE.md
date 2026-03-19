# Production Monitoring & Alerting Guide
**Date:** March 13, 2026  
**Purpose:** Monitor Hybrid51 model deployment health and prediction quality

---

## Critical Metrics to Monitor

### 1. Normalization Status (CRITICAL)

**What to Monitor:**
- Whether normalization stats are loaded for each symbol
- Feature scale ranges before/after normalization

**How to Check:**
```python
# On service startup, log normalization status
for symbol in ALL_SYMBOLS:
    bundle = service.stage1[symbol]["A"]
    if bundle.norm_mean is None:
        logger.error(f"❌ {symbol}: NO NORMALIZATION - predictions will be wrong!")
    else:
        logger.info(f"✓ {symbol}: Normalization loaded (dim={len(bundle.norm_mean)})")
```

**Alert Triggers:**
- 🔴 CRITICAL: Any symbol missing normalization → Page on-call
- 🔴 CRITICAL: All symbols missing normalization → STOP PRODUCTION

**Expected Values:**
- norm_mean shape: (325,)
- norm_std shape: (325,)
- norm_std min: >1e-6 (no zero-variance features)

---

### 2. Feature Completeness

**What to Monitor:**
- `feature_completeness` metric in prediction.csv (column 24)
- Per-group feature coverage
- Trend over time

**Thresholds:**
```python
# Current baseline: 53.6%
# Target: 65%+

if feature_completeness < 0.50:
    alert("⚠️ Feature coverage dropped below 50%")
if feature_completeness < 0.40:
    alert("🔴 CRITICAL: Feature coverage <40%, investigate data pipeline")
```

**Dashboard Metrics:**
- **Gauge:** Current feature completeness (target: 53-65%)
- **Time series:** 24-hour rolling average
- **Histogram:** Distribution of coverage across predictions

**Alert Triggers:**
- 🟡 WARNING: Coverage < 50% for 10+ consecutive predictions
- 🔴 CRITICAL: Coverage < 40% or drops >10% suddenly
- 📊 INFO: Coverage increases >60% (new features available!)

---

### 3. Prediction Confidence

**What to Monitor:**
- `confidence` score in prediction.csv (column 6)
- Confidence components: agreement, consensus, gate conviction, data quality

**Expected Ranges:**
```python
# Healthy confidence distribution
conf_mean: 0.30 - 0.60
conf_std: 0.10 - 0.25

# Anomaly detection
if conf_mean < 0.15:
    alert("⚠️ Low confidence - model uncertainty high")
if conf_std < 0.05:
    alert("⚠️ Confidence not varying - check if suppressed")
```

**Dashboard Metrics:**
- **Gauge:** Current confidence
- **Time series:** Confidence trend with prediction direction overlay
- **Scatter:** Confidence vs |prob - 0.5| (should correlate)

**Alert Triggers:**
- 🟡 WARNING: Confidence <0.15 for >20 consecutive predictions
- 🟡 WARNING: Confidence std <0.05 over 1 hour (stuck)
- 📊 INFO: Confidence >0.70 (high-conviction signal)

---

### 4. Agent Probability Distribution

**What to Monitor:**
- Individual agent probabilities (columns 8-14)
- Agent agreement vs disagreement
- Agent gates (columns 15-21)

**Healthy Patterns:**
```python
# Agent probabilities should spread out
agent_probs = [prob_A, prob_B, prob_C, prob_K, prob_T, prob_Q, prob_2D]
agent_std = np.std(agent_probs)

if agent_std < 0.05:
    alert("⚠️ Agents not diversifying - check feature extraction")
if agent_std > 0.25:
    alert("📊 High agent disagreement - market regime change?")
```

**Expected Agent Gates (from training):**
```python
# These should match test set averages
gates_expected = {
    'A': 0.9996, 'B': 0.9997, 'K': 0.9993, 'Q': 0.9999,
    'C': 0.0029, 'T': 0.0031, '2D': 0.0026
}

# Monitor deviations
for agent, expected_gate in gates_expected.items():
    if abs(actual_gate - expected_gate) > 0.10:
        alert(f"⚠️ Agent {agent} gate deviation: {actual_gate:.3f} vs {expected_gate:.3f}")
```

**Dashboard Metrics:**
- **Bar chart:** Current agent probabilities
- **Heatmap:** Agent correlation matrix (24-hour window)
- **Line chart:** Agent gates over time

**Alert Triggers:**
- 🟡 WARNING: Agent std <0.05 (agents agreeing too much)
- 🟡 WARNING: Agent gates deviate >10% from training averages
- 📊 INFO: High agent disagreement (std >0.20) - regime uncertainty

---

### 5. Prediction Latency

**What to Monitor:**
- `latency_ms` in prediction.csv (column 25)
- Per-stage timing breakdown (if available)

**Thresholds:**
```python
# Expected latency
latency_p50: 300-500 ms
latency_p95: 500-800 ms
latency_p99: 800-1200 ms

# Alerts
if latency > 2000:
    alert("⚠️ Prediction latency >2s, check CPU/memory")
if latency > 5000:
    alert("🔴 CRITICAL: Latency >5s, prediction delay too high")
```

**Dashboard Metrics:**
- **Gauge:** Current latency
- **Time series:** Latency over time
- **Histogram:** Latency distribution (last 1000 predictions)

**Alert Triggers:**
- 🟡 WARNING: p95 latency >1000ms for 10 minutes
- 🔴 CRITICAL: p95 latency >2000ms
- 📊 INFO: Latency drops significantly (optimization opportunity)

---

### 6. Suppression Rate & Warmup

**What to Monitor:**
- `suppressed` flag (column 27)
- `reason` field (column 28)
- `warmup_fraction` (column 25)

**Healthy Patterns:**
```python
# After initial warmup (20 cycles = ~3 minutes)
suppression_rate_normal: <5%  # Only during data gaps/quality issues

# During startup
warmup_duration: ~3-4 minutes (20 cycles × 10s)

# Suppression reasons distribution
reasons = {
    'warmup_X_of_20': 'Normal during startup',
    'data_stale': 'Theta fetcher may be down',
    'quality_low': 'Data quality issue',
    'stage1_incomplete': 'Model loading issue',
}
```

**Dashboard Metrics:**
- **Badge:** Suppression status (LIVE / SUPPRESSED)
- **Counter:** Consecutive suppressed predictions
- **Pie chart:** Suppression reasons (last 100)

**Alert Triggers:**
- 🟡 WARNING: >10 consecutive suppressed predictions after warmup
- 🔴 CRITICAL: >50 consecutive suppressions (system failure)
- 📊 INFO: Exited warmup mode (predictions now live)

---

### 7. Data Freshness

**What to Monitor:**
- Batch ID changes
- Timestamp gaps between predictions
- File modification times (agg_snapshot.csv)

**Thresholds:**
```python
# Predictions should update every 10 seconds
if time_since_last_update > 30:
    alert("⚠️ No new predictions in 30s, check theta fetcher")
if time_since_last_update > 120:
    alert("🔴 CRITICAL: No predictions in 2 minutes, data pipeline down")
```

**Dashboard Metrics:**
- **Badge:** Data freshness (< 30s = green, 30-120s = yellow, >120s = red)
- **Time series:** Batch ID over time (should increment steadily)
- **Counter:** Seconds since last update

**Alert Triggers:**
- 🟡 WARNING: No new batch for >30 seconds
- 🔴 CRITICAL: No new batch for >2 minutes
- 📊 INFO: Batch update frequency changes

---

### 8. Model Performance Tracking

**What to Monitor:**
- Rolling accuracy (if labels available)
- Prediction distribution (% BULL vs BEAR)
- Probability distribution (should not cluster at 0.5)

**Expected Distributions:**
```python
# Probability histogram (over 1000 predictions)
# Should be bimodal with peaks around 0.35-0.40 and 0.60-0.65
# Avoid clustering at exactly 0.50 (indicates suppression or failure)

prob_bins = np.histogram(probs, bins=20)
if prob_bins[0][10] > 0.3 * len(probs):  # >30% at bin 0.50
    alert("⚠️ Too many predictions at 0.50 - check suppression logic")
```

**Dashboard Metrics:**
- **Histogram:** Prediction probability distribution
- **Pie chart:** BULL/BEAR/SUPPRESSED split
- **Time series:** Daily win rate (if labels available)

**Alert Triggers:**
- 🟡 WARNING: >40% predictions clustered at 0.45-0.55
- 🟡 WARNING: BULL/BEAR ratio >4:1 or <1:4 (extreme bias)
- 📊 INFO: Probability distribution changes significantly

---

## Automated Health Checks

### Service Startup Checks

```python
def perform_startup_health_check():
    """Run on service initialization."""
    
    checks = {}
    
    # 1. Normalization loaded
    checks['normalization'] = all(
        service.stage1[sym][agent].norm_mean is not None
        for sym in ALL_SYMBOLS for agent in ALL_AGENTS
        if agent in service.stage1[sym]
    )
    
    # 2. All models loaded
    checks['stage1_models'] = sum(
        len(service.stage1[sym]) for sym in ALL_SYMBOLS
    ) == 35
    checks['stage2_models'] = len(service.stage2) == 7
    checks['stage3_model'] = service.stage3_model is not None
    
    # 3. Feature extractor initialized
    checks['feature_extractor'] = service.bridge.extractor is not None
    
    # 4. Threshold configured
    checks['threshold'] = 0.44 <= service.threshold <= 0.50
    
    # Log results
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {check}")
    
    if not all(checks.values()):
        logger.error("❌ HEALTH CHECK FAILED - Review startup logs")
        return False
    
    logger.info("✅ All health checks passed")
    return True
```

### Runtime Health Checks (Every 100 Predictions)

```python
def perform_runtime_health_check(recent_predictions):
    """Run every 100 predictions."""
    
    df = pd.DataFrame(recent_predictions)
    
    checks = {}
    
    # 1. Feature completeness
    checks['feature_coverage'] = df['feature_completeness'].mean() > 0.50
    
    # 2. Confidence variation
    checks['confidence_varies'] = df['confidence'].std() > 0.05
    
    # 3. Agent diversity
    agent_cols = ['agent_A_prob', 'agent_B_prob', 'agent_C_prob', 
                  'agent_K_prob', 'agent_T_prob', 'agent_Q_prob', 'agent_2D_prob']
    agent_std = df[agent_cols].std(axis=1).mean()
    checks['agent_diversity'] = agent_std > 0.05
    
    # 4. Latency acceptable
    checks['latency_ok'] = df['latency_ms'].quantile(0.95) < 1000
    
    # 5. Not over-suppressed
    checks['live_predictions'] = (df['suppressed'] == False).mean() > 0.80
    
    # Log warnings
    for check, passed in checks.items():
        if not passed:
            logger.warning(f"⚠️  Runtime check failed: {check}")
    
    return all(checks.values())
```

---

## Dashboard Monitoring Panel

### Recommended Metrics Display

**Top Row (Critical):**
1. **Normalization Status** - Badge (GREEN/RED)
2. **Feature Coverage** - Gauge (53.6% current, target 65%)
3. **Prediction Status** - Badge (LIVE/SUPPRESSED/WARMUP)
4. **Confidence Score** - Gauge (0-100%)

**Middle Row (Performance):**
1. **Agent Probabilities** - 7 bars (A, B, C, K, T, Q, 2D)
2. **Agent Gates** - 7 mini-gauges (A~100%, B~100%, C~0%, etc.)
3. **Prediction Direction** - Indicator (BULL/BEAR + strength)
4. **Threshold Line** - Horizontal line at 0.44 on probability chart

**Bottom Row (Diagnostics):**
1. **Latency** - Line chart (last 100 predictions)
2. **Suppression Reasons** - Pie chart (last 100)
3. **Feature Coverage by Group** - Horizontal bars (11 groups)
4. **Data Freshness** - Time since last batch

---

## Alert Configuration

### Alert Severity Levels

**🔴 CRITICAL (Page Immediately):**
- No normalization loaded for any symbol
- Prediction service down >2 minutes
- Feature coverage <40%
- Latency >5 seconds

**🟡 WARNING (Review Within 1 Hour):**
- Feature coverage <50%
- Suppression rate >20% over 10 minutes
- Confidence <0.15 for >20 predictions
- Agent diversity <0.05 (not diversifying)
- Latency >1 second (p95)

**📊 INFO (Log for Review):**
- Feature coverage increases >60%
- Threshold automatically adjusted
- Model checkpoint updated
- Unusual agent gate patterns

### Alert Delivery

**Critical Alerts:**
- PagerDuty / phone notification
- Slack #trading-alerts channel
- Email to on-call engineer
- Automatic service status page update

**Warning Alerts:**
- Slack #trading-monitoring channel
- Email digest (batched)
- Dashboard warning banner

**Info Alerts:**
- Logged to monitoring system
- Daily summary email
- Dashboard info badge

---

## Log Monitoring

### Key Log Patterns to Watch

**Startup Logs:**
```bash
# Good startup
15:11:20 [INFO] Loading models...
15:11:21 [INFO]   Stage1: 35/35 models loaded
15:11:21 [INFO]   Stage2: 7/7 models loaded
15:11:21 [INFO]   Stage3: loaded
15:11:21 [INFO] Models loaded in 1.1s
15:11:21 [INFO] ✓ SPXW: Normalization loaded (dim=325)  ← MUST SEE THIS

# Bad startup (CURRENT STATE)
15:11:21 [INFO]   Stage1: 35/35 models loaded
15:11:21 [INFO]   Stage2: 7/7 models loaded
15:11:21 [INFO]   Stage3: loaded
# Missing normalization log = NO NORMALIZATION!
```

**Prediction Logs:**
```bash
# Healthy predictions
15:02:11 [INFO] Batch 72: LIVE prob=0.458 conf=0.414 latency=360ms quality=0.43

# Suppressed (acceptable during warmup)
14:58:29 [INFO] Batch 72: SUPPRESSED prob=--- conf=0.000 latency=326ms quality=0.39

# Data staleness (investigate)
15:05:00 [INFO] Batch 72 unchanged but file rewritten 3x — forcing prediction
```

**Error Patterns to Alert On:**
```bash
# Model loading failures
[WARNING] Failed to load stage1 SPXW/A: ...

# Feature extraction failures
[WARNING] Feature extraction failed: ..., returning zeros

# Inference failures
[ERROR] Stage 1 inference failed: ...
[ERROR] Stage 2 fusion failed: ...
[ERROR] Stage 3 ensemble failed: ...
```

---

## Performance Baseline (Post-Fix)

### Expected Metrics After Normalization Fix

**Test Set Performance (Training):**
```
Accuracy: 0.610
F1:       0.716
AUC:      0.722
Brier:    0.238
```

**Production Target (Accounting for 53.6% Coverage):**
```
Accuracy: 0.590-0.595  (within 2% of training)
F1:       0.695-0.705  (within 2% of training)
AUC:      0.700-0.710  (within 2% of training)
```

**Acceptance Criteria:**
- Accuracy within 5% of training test set
- F1 within 3% of training test set
- AUC within 3% of training test set

**If below threshold:**
1. Check normalization is applied
2. Verify feature coverage stable at 53.6%
3. Check threshold matches training (0.44)
4. Investigate data quality issues

---

## Rolling Performance Monitoring

### Daily Summary Report

Generate automatically at market close (4:00 PM ET):

```python
def generate_daily_summary():
    today_preds = load_predictions_for_date(date.today())
    
    return {
        'date': date.today(),
        'total_predictions': len(today_preds),
        'live_predictions': (today_preds['suppressed'] == False).sum(),
        'suppression_rate': today_preds['suppressed'].mean(),
        
        # Quality metrics
        'avg_feature_coverage': today_preds['feature_completeness'].mean(),
        'avg_quality_score': today_preds['quality_score'].mean(),
        'avg_confidence': today_preds['confidence'].mean(),
        'avg_latency_ms': today_preds['latency_ms'].mean(),
        
        # Prediction distribution
        'bull_count': (today_preds['direction'] == 'BULL').sum(),
        'bear_count': (today_preds['direction'] == 'BEAR').sum(),
        'avg_probability': today_preds['prob'].mean(),
        
        # Agent metrics
        'avg_agent_std': today_preds[agent_cols].std(axis=1).mean(),
        'avg_gates': {agent: today_preds[f'gate_{agent}'].mean() 
                      for agent in ALL_AGENTS},
        
        # Anomalies
        'normalization_status': check_normalization_loaded(),
        'alerts_triggered': count_todays_alerts(),
    }
```

**Email Format:**
```
Daily Hybrid51 Model Report - March 13, 2026
═══════════════════════════════════════════════════

📊 Predictions: 45 live, 2 suppressed (4.3% suppression)
✓ Feature Coverage: 53.7% avg (stable)
✓ Confidence: 41.2% avg (healthy variation)
⚠️  Normalization: NOT LOADED (CRITICAL - fix urgently)

Direction Distribution:
  BULL: 22 (48.9%)
  BEAR: 23 (51.1%)

Agent Performance:
  A: avg_prob=0.463, gate=99.96%
  B: avg_prob=0.394, gate=99.97%
  K: avg_prob=0.450, gate=99.93%
  Q: avg_prob=0.535, gate=99.99%
  C: avg_prob=0.509, gate=0.29% (suppressed)
  T: avg_prob=0.295, gate=0.31% (suppressed)
  2D: avg_prob=0.522, gate=0.26% (suppressed)

Latency: 354ms avg, 520ms p95
Alerts: 1 CRITICAL (normalization), 0 warnings

Action Required: Fix normalization immediately
```

---

## Comparison Monitoring

### Training vs Production Divergence

**Track these ratios weekly:**

| Metric | Training Test | Production Target | Alert If Diverges |
|--------|--------------|-------------------|-------------------|
| Accuracy | 0.610 | 0.590-0.595 | >5% below |
| F1 Score | 0.716 | 0.695-0.705 | >3% below |
| AUC | 0.722 | 0.700-0.710 | >3% below |
| Feature Coverage | 100% | 53.6% | <50% |
| Prediction Latency | N/A | 300-500ms | >1000ms |

**Review Schedule:**
- **Daily:** Check for any critical/warning alerts
- **Weekly:** Review performance vs training baseline
- **Monthly:** Full audit of feature coverage and model health

---

## Data Quality Indicators

### Feature Extraction Quality

**Monitor from `quality_score` field:**
```python
# quality_score = based on NaN count in original extractor
# Should be >0.95 for clean data

if quality_score < 0.80:
    alert("⚠️ Data quality degraded - check theta fetcher")
if quality_score < 0.50:
    alert("🔴 CRITICAL: Severe data quality issues")
```

**Typical Values:**
- Market hours (9:30-16:00 ET): quality 0.85-0.95
- Pre/post market: quality 0.40-0.60 (acceptable)
- Overnight: quality may drop further

---

## Rollover & Historical Tracking

### Prediction History CSV

**Monitor columns for patterns:**
```python
# Look for prediction rollover (prob changes >0.05)
df['prob_change'] = df['prob'].diff().abs()

if df['prob_change'].mean() < 0.01:
    alert("⚠️ Predictions not changing - check if stuck")
if df['prob_change'].max() > 0.30:
    alert("📊 Large prediction swing detected - regime change?")
```

**Rolling Statistics (1-hour window):**
- Mean probability: should vary, not stuck at 0.50
- Confidence range: should span 0.20-0.60
- Feature coverage: should be stable around 53.6%
- Agent std: should be >0.05

---

## System Dependencies Monitor

### External Services

**Theta Fetcher Status:**
```bash
# Check theta fetcher is running
ps aux | grep theta_fetching_v5.py
if not running:
    alert("🔴 CRITICAL: Theta fetcher down, no new data")
```

**Dashboard Application:**
```bash
# Check dashboard is running on port 8050
curl -s http://localhost:8050 > /dev/null
if exit_code != 0:
    alert("🔴 Dashboard down on port 8050")
```

**File System:**
```bash
# Monitor disk usage
disk_usage_pct=$(df /workspace | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $disk_usage_pct -gt 90 ]; then
    alert "⚠️ Disk usage >90%, cleanup may be needed"
fi
```

---

## Alert Escalation Path

### Level 1: Automated Recovery

**Triggers:**
- Service crashes
- Python exceptions
- Port conflicts

**Actions:**
- Auto-restart via systemd/supervisor
- Clear port (kill -9)
- Log error for review

### Level 2: Engineering Alert

**Triggers:**
- Health checks fail after restart
- Normalization missing
- Feature coverage <40%
- Suppression >50 consecutive

**Actions:**
- Slack alert
- Email notification
- Pause trading decisions
- Manual investigation required

### Level 3: Critical Escalation

**Triggers:**
- Models producing invalid outputs
- Data pipeline completely down
- Predictions don't match expected distributions
- Security/integrity issues

**Actions:**
- Page on-call
- Halt trading
- Emergency review
- Potential rollback

---

## Monitoring Tools Setup

### Recommended Stack

**Metrics Collection:**
- **Prometheus** - Scrape prediction.csv every 10s
- **Custom Python exporter** - Parse CSV and expose metrics
- **Grafana** - Dashboard visualization

**Alerting:**
- **Alertmanager** - Alert routing
- **PagerDuty** - Critical alerts
- **Slack** - Warning/info alerts

**Logging:**
- **ELK Stack** - Centralized log aggregation
- **Filebeat** - Ship logs from /tmp/prediction_service.log
- **Kibana** - Log search and analysis

### Quick Setup Script

```bash
#!/bin/bash
# setup_monitoring.sh

# 1. Create monitoring directory
mkdir -p /workspace/monitoring/{prometheus,grafana,alerts}

# 2. Install dependencies
pip install prometheus-client

# 3. Create metrics exporter
cat > /workspace/monitoring/metrics_exporter.py << 'PYTHON'
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter
import time

# Define metrics
feature_coverage = Gauge('hybrid51_feature_coverage', 'Feature completeness percentage')
prediction_confidence = Gauge('hybrid51_confidence', 'Prediction confidence score')
prediction_latency = Gauge('hybrid51_latency_ms', 'Prediction latency in milliseconds')
suppression_counter = Counter('hybrid51_suppressed_total', 'Total suppressed predictions')
prediction_counter = Counter('hybrid51_predictions_total', 'Total predictions')

def update_metrics():
    df = pd.read_csv('/workspace/Final_production_model/daily_data/prediction.csv')
    latest = df.iloc[-1]
    
    feature_coverage.set(latest['feature_completeness'] * 100)
    prediction_confidence.set(latest['confidence'] * 100)
    prediction_latency.set(latest['latency_ms'])
    
    if latest['suppressed']:
        suppression_counter.inc()
    prediction_counter.inc()

if __name__ == '__main__':
    start_http_server(9090)
    while True:
        update_metrics()
        time.sleep(10)
PYTHON

# 4. Start exporter
nohup python /workspace/monitoring/metrics_exporter.py > /tmp/metrics_exporter.log 2>&1 &

echo "Monitoring setup complete. Metrics available at :9090/metrics"
```

---

## Manual Checks (Weekly)

### Week Review Checklist

**Every Monday:**
- [ ] Review prediction accuracy (if labels available)
- [ ] Check feature coverage trend (stable at 53.6%?)
- [ ] Verify normalization still loaded
- [ ] Review suppression reasons distribution
- [ ] Check latency hasn't increased
- [ ] Compare agent probabilities to training averages
- [ ] Review any alerts triggered during week
- [ ] Update this checklist if monitoring gaps found

**Every Month:**
- [ ] Full deployment audit (re-run variance checks)
- [ ] Compare rolling metrics to training test set
- [ ] Review feature extraction improvements
- [ ] Check for model updates or retraining needs
- [ ] Validate threshold still optimal
- [ ] Review disk usage and cleanup old snapshots

---

## Monitoring Queries

### Useful Analysis Queries

**Check feature coverage stability:**
```sql
SELECT 
    DATE(ts) as date,
    AVG(feature_completeness) as avg_coverage,
    MIN(feature_completeness) as min_coverage,
    MAX(feature_completeness) as max_coverage
FROM predictions
WHERE suppressed = FALSE
GROUP BY date
ORDER BY date DESC
LIMIT 7;
```

**Check agent diversity:**
```sql
SELECT 
    DATE(ts) as date,
    AVG(agent_std) as avg_diversity
FROM (
    SELECT 
        ts,
        SQRT((
            POW(agent_A_prob - 0.5, 2) +
            POW(agent_B_prob - 0.5, 2) +
            -- ... other agents
        ) / 7) as agent_std
    FROM predictions
    WHERE suppressed = FALSE
)
GROUP BY date;
```

**Check suppression patterns:**
```sql
SELECT 
    reason,
    COUNT(*) as count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM predictions) as pct
FROM predictions
WHERE suppressed = TRUE
GROUP BY reason
ORDER BY count DESC;
```

---

## Normalization Validation (After Fix)

### Post-Fix Verification Steps

**1. Check service logs:**
```bash
tail -100 /tmp/prediction_service.log | grep -i "normalization"
# Should see: "Applying z-score normalization from training stats"
```

**2. Verify feature magnitudes:**
```python
# After normalization, features should be in range [-3, 3]
import numpy as np
from prediction_service import FeatureBridge, PredictionService

service = PredictionService("daily_data")
bundle = service.stage1["SPXW"]["A"]

# Load latest snapshot
snapshot = load_latest_snapshot()
vec, quality = service.bridge.extract_325_features(snapshot)

# Apply normalization manually to check
if bundle.norm_mean is not None:
    vec_norm = (vec - bundle.norm_mean) / np.maximum(bundle.norm_std, 1e-6)
    print(f"Normalized feature range: [{vec_norm.min():.2f}, {vec_norm.max():.2f}]")
    # Should be: [-3.0, 3.0] approximately
    
    if vec_norm.min() < -10 or vec_norm.max() > 10:
        print("❌ NORMALIZATION FAILED - features still on wrong scale")
    else:
        print("✓ Normalization working correctly")
```

**3. Compare predictions before/after:**
```python
# Load predictions from before fix
old_preds = pd.read_csv('prediction_archive_before_norm_fix.csv')

# Load predictions after fix
new_preds = pd.read_csv('daily_data/prediction.csv').tail(100)

# Compare agent probability distributions
for agent in ALL_AGENTS:
    old_std = old_preds[f'agent_{agent}_prob'].std()
    new_std = new_preds[f'agent_{agent}_prob'].std()
    print(f"Agent {agent}: std {old_std:.3f} → {new_std:.3f}")
    
# Should see significantly higher variation after normalization
```

---

## Continuous Improvement Tracking

### Feature Coverage Roadmap Progress

**Current:** 53.6% (March 13, 2026)

**Target Milestones:**
- **Q2 2026:** 58% (add aggressor detection)
- **Q3 2026:** 63% (add historical snapshots + rho/epsilon)
- **Q4 2026:** 68% (add NBBO if feasible)

**Track Monthly:**
```python
coverage_history = {
    '2026-03-13': 0.536,  # Feature extraction upgrade
    '2026-04-15': 0.580,  # Aggressor detection (planned)
    '2026-06-01': 0.630,  # Historical snapshots (planned)
    '2026-09-01': 0.680,  # Full enhancements (target)
}
```

---

## Monitoring Dashboard Example

```python
# Dash callback for real-time monitoring panel
@app.callback(
    Output('monitoring-panel', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_monitoring_panel(n):
    pred = load_latest_prediction()
    
    return html.Div([
        # Critical status badges
        html.Div([
            create_status_badge(
                "Normalization", 
                "OK" if check_normalization() else "MISSING",
                "success" if check_normalization() else "danger"
            ),
            create_status_badge(
                "Feature Coverage",
                f"{pred['feature_completeness']*100:.1f}%",
                "success" if pred['feature_completeness'] > 0.50 else "warning"
            ),
            create_status_badge(
                "Status",
                "SUPPRESSED" if pred['suppressed'] else "LIVE",
                "warning" if pred['suppressed'] else "success"
            ),
        ], style={'display': 'flex', 'gap': '10px'}),
        
        # Feature coverage gauge
        dcc.Graph(figure=create_gauge(
            value=pred['feature_completeness']*100,
            title="Feature Coverage",
            range=[0, 100],
            threshold=50
        )),
        
        # Agent probability bars
        dcc.Graph(figure=create_agent_bars(pred)),
        
        # Recent history
        dcc.Graph(figure=create_history_chart(load_recent_predictions(100))),
    ])
```

---

## Summary

### Must-Monitor Metrics

1. ✅ Normalization status (CRITICAL)
2. ✅ Feature completeness (>50%)
3. ✅ Prediction confidence (0.20-0.60 range)
4. ✅ Agent diversity (std >0.05)
5. ✅ Latency (p95 <1s)
6. ✅ Suppression rate (<10%)

### Alert Fatigue Prevention

**Avoid:**
- Alerting on known issues (e.g., low coverage in pre-market)
- Redundant alerts (group related issues)
- Info-level noise (use daily digest)

**Prioritize:**
- Critical production blockers
- Unexpected degradation
- Data pipeline failures

### Success Criteria

**Healthy production system:**
- ✓ No critical alerts
- ✓ Feature coverage stable
- ✓ Predictions varying (not stuck)
- ✓ Latency <500ms p95
- ✓ Performance within 5% of training

---

**Guide Status:** COMPLETE  
**Last Updated:** March 13, 2026  
**Next Review:** March 20, 2026
