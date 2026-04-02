# Hybrid51 Stage 1–3 Refactoring Plan

Comprehensive plan addressing DuckDB data integrity, agent architecture deficiencies, zero/constant feature contamination, and Stage 2/3 degradation — informed by a full audit of all expert suggestions, agent source code, MODEL_AUDIT.md, REMEDY_PLAN.md, and live DuckDB queries.

---

## Critical Findings from Audit

### 1. DuckDB Data Integrity

| DB | Table | Rows | Status |
|---|---|---|---|
| Part 1 | options_greek | 832M | ✅ Clean — zero NULLs in all critical columns, no duplicate source files |
| Part 1 | options_trade_quote | 1.4B | ✅ Clean |
| Part 2 | options_greek | 0 | ⚠️ Not yet ingested (PID 2410 still running, TQ first) |
| Part 2 | options_trade_quote | 959M | ✅ Clean |

**Critical data gap**: SPXW TQ in Part 1 covers only **331 dates** (2023-03 to 2024-06). Part 2 fills this with **1690 dates** (2019-01 to 2026-01). Combined, TQ coverage is now nearly complete. But Greek ingestion into Part 2 hasn't started yet — Part 1 already has full Greek coverage (1749 dates, 2019–2026).

**Verdict**: No corruption from OOM-interrupted ingestion. Data is safe. The remaining ingestion will add Greek data to Part 2 (lower priority since Part 1 already has it).

### 2. Zero/Constant Feature Problem

Per MODEL_AUDIT: **only 181 of 325 features are non-zero**. The 144 dead features span mostly the Phase 1 groups (Smart Money 270–284, Volume Anomaly 285–296, Trade Conditions 297–306, Quote Pressure 307–324) — because those features require trade/quote data, but:
- The old tier3 sequences were built from tier2 data that only had **331 TQ dates** (Part 1 TQ alone)
- For dates without TQ data, Phase 1 features (dims 270–324) were **zero-padded**
- Now that Part 2 has 1690 TQ dates, **rebuilding sequences should dramatically reduce dead features**

This is the **single biggest root cause** of Agent T/Q failure and the 144 dead features.

### 3. Agent T & Q Architecture Assessment

**Agent T** (`agent_t.py`):
- Uses `BatchNorm1d(256)` in `flow_encoder` — **confirmed broken** (running stats max = 16.3M)
- Has three good subcomponents (flow_encoder, flow_cnn, impact_net) but ALL feed from un-normalized raw trade features
- Fix: Replace BatchNorm with LayerNorm, add input LayerNorm, z-score normalize trade features in preprocessing
- With full TQ data (1690 dates), Agent T should become genuinely useful instead of a noise agent

**Agent Q** (`agent_q.py`):
- No BatchNorm (uses LayerNorm on LSTM output) — architecture is healthier
- Has BiLSTM for spread dynamics + imbalance detector — good design for quote analysis
- Main issue is the same: insufficient TQ data coverage made its features mostly zero
- Fix: Add input LayerNorm, z-score normalize quote features

**Assessment**: The agent architectures themselves are reasonable. The primary failure was **data starvation** (331 vs 1749 dates) plus BatchNorm instability in Agent T.

### 4. Agent A/B/C/K and 2D Assessment

| Agent | Architecture | Issues | Recommendation |
|---|---|---|---|
| **A** (Alpha) | MLP + Causal CNN, 620K params | `BatchNorm1d(256)` in static_path, high weight std=19.5 | Replace BatchNorm→LayerNorm, add residual connection |
| **B** (Beta) | Stacked BiLSTM, 986K params | Doesn't use backbone temporal embed — only raw seq | ✅ Already diverse by design (pure LSTM), keep as-is |
| **C** (Attention) | CNN+Attention+BiLSTM, 600K params | `BatchNorm1d(72)` in CNN path, uses only first 32 features of seq | Replace BatchNorm→LayerNorm, consider wider feature input |
| **K** (Greeks) | Static MLP only, 587K params | Only uses first 127 features (no temporal), has gamma_squeeze_detector that shifts score by ±0.1 | ✅ Good specialist design. Don't need backbone. |
| **2D** (Chain) | 2D CNN on strikes×time grid | Falls back to `_create_synthetic_chain()` (random noise!) when no chain_2d data is available | ⚠️ Effectively training on noise. Needs real chain data from DuckDB |

**Key insight about Agent 2D**: It currently falls back to synthetic random chains, making it essentially a random predictor with learned biases. With proper chain data extracted from DuckDB (strike×time×greeks grid), this could become one of the strongest agents since the "volatility surface movie" is genuinely predictive.

### 5. Why Stage 2 and Stage 3 Are Worse Than Stage 1

Based on MODEL_AUDIT Section 4 and expert analysis:

**Root causes (ranked by impact):**

1. **Agent homogeneity destroys ensemble value**: 85–92% pairwise agreement. All agents share the same backbone + see the same 325 features. The ensemble is worse than Agent A alone (58.52% vs 58.68%).

2. **Stage 2 pair fusion is predicting noise on noise**: Stage 2 takes 7 agent logits + 29 cross-symbol features. But if the 7 agents are highly correlated, Stage 2's input has very low effective dimensionality. It can't learn useful cross-symbol patterns when all base signals are the same.

3. **Stage 3 meta-learner has no new signal**: Stage 3 takes 5 pair probabilities + statistics. But all pairs produce similar outputs because they use the same correlated agents. The MLP meta-learner was selected for highest F1 (0.622) but has the lowest accuracy (58.72%) — a sign of overfitting to label noise.

4. **VIXW pair is harmful**: 51.78% high-confidence accuracy — WORSE than random. Should be dropped.

5. **Data leakage in Stage 3 v1 (fixed in v2)**: v2 correctly trains meta on val set only, but the fundamental diversity problem remains.

**Verdict**: Fixing Stage 2/3 requires fixing Stage 1 diversity FIRST. Without diverse base agents, no meta-learner can help.

---

## Execution Plan

### Phase 0: Data Pipeline (from DuckDB) — Must Do First

> All previous tier1/tier2/tier3 data is gone. Must rebuild from DuckDB.

**Step 0A**: Extract SPXW Greek + TQ from DuckDB Part 1 + Part 2
- Greek from Part 1 (269M rows, 1749 dates) → filtered tier1 parquets
- TQ from Part 1 (132M rows, 331 dates) + Part 2 (SPXW portion of 959M rows, 1690 dates) → tier1 TQ parquets
- Apply active contract filter during extraction (delta 0.2–0.9, DTE 0–3, bid≠0, ask≠0, vega≠0)
- Output: `/workspace/data/tier1_filtered/symbol=SPXW/` and `/workspace/data/tier1_filtered_tradequote/symbol=SPXW/`

**Step 0B**: Run MasterFeatureExtractor v2 (tier2 reprocess)
- Process filtered Greek + TQ into 325-dim minute bars
- Key: with full TQ coverage, Phase 1 features (dims 270–324) should now be populated instead of zero-padded
- Output: `/workspace/data/tier2_minutes_v2/symbol=SPXW/`

**Step 0C**: Build binary sequences (tier3)
- 20-timestep windows, binary UP/DOWN labels for 15-min horizon
- Chronological 60/20/20 split
- **New**: compute and save per-feature-group z-score normalization stats from training split
- **New**: identify and log zero-variance features from training split
- **New**: build chain_2d data (strikes×time×greeks grid) for Agent 2D
- Output: `/workspace/data/tier3_binary/SPXW/horizon_15min/`

### Phase 1: Fix Agent Architectures

**Step 1A**: Fix Agent T
- Replace `BatchNorm1d(hidden_dim)` → `LayerNorm(hidden_dim)` in flow_encoder
- Replace `BatchNorm1d(64)` → `LayerNorm(64)` in flow_cnn (after Conv1d, need to reshape)
- Add `self.input_norm = nn.LayerNorm(trade_feat_dim)` as first operation on static input

**Step 1B**: Fix Agent A
- Replace `BatchNorm1d(256)` → `LayerNorm(256)` in static_path
- Add residual connection: `self.residual_proj = nn.Linear(input_dim, 96)`

**Step 1C**: Fix Agent C
- Replace `BatchNorm1d(72)` → `LayerNorm(72)` after CNN concatenation (need to handle dimension ordering)

**Step 1D**: Fix Agent 2D
- Replace `BatchNorm2d` → `InstanceNorm2d` or `GroupNorm` (BatchNorm2d less problematic than 1d, but still remove)
- The real fix is providing actual chain_2d data (Phase 0C) instead of synthetic random noise

**Step 1E**: Add input LayerNorm to Agent Q
- `self.input_norm = nn.LayerNorm(quote_feat_dim)` before quote_encoder

### Phase 2: Improve Training Script

**Step 2A**: Preprocessing in training script
- Load per-feature-group z-score stats (computed in Phase 0C from train split only)
- Apply normalization to all splits using train stats
- Remove zero-variance features or mask them
- Save normalization params for inference

**Step 2B**: Loss function improvements
- Replace `BCEWithLogitsLoss` → `BinaryFocalLoss(gamma=2.0, alpha=0.52)` with label smoothing (0.05)
- Add optional soft-F1 loss component (weight 0.3)

**Step 2C**: Training schedule improvements
- Initial LR: 3e-4 (from 5e-4)
- Optimizer: AdamW with weight_decay=0.01 (from Adam with 1e-5)
- Scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
- Patience: 15 epochs (from 7)
- Max epochs: 80 (from 25)
- Gradient clipping: 1.0 (from 5.0)
- Batch size: 512 with gradient accumulation 4× = effective 2048

**Step 2D**: Threshold optimization
- After training, sweep thresholds [0.30, 0.65] on val set to maximize F1
- Save per-agent optimal threshold

**Step 2E**: Feature subsetting for agent diversity
- Agent A: All non-zero features (general)
- Agent B: Greeks + IV Surface (dims 0–149) — BiLSTM on temporal structure
- Agent K: Core Greeks only (dims 0–127) — static MLP, no backbone needed
- Agent C: Flow + Microstructure + Sentiment (dims 150–269) — market activity signals
- Agent T: Trade features (dims 270–306) — direct specialist, no shared backbone
- Agent Q: Quote features (dims 307–324) — direct specialist, no shared backbone
- Agent 2D: Chain2D data only — 2D CNN on strike×time grid

### Phase 3: Stage 2 & 3 Fixes (after Phase 1–2 validated)

**Step 3A**: Re-evaluate Stage 1 diversity
- Retrain all agents with Phase 2 changes
- Measure pairwise agreement — target <75% (from 85–92%)
- If agreement still too high, add negative correlation learning (NCL) penalty

**Step 3B**: Fix Stage 2 pair fusion
- Drop VIXW pair entirely (51.78% high-conf accuracy)
- Use only SPY, QQQ, IWM, TLT pairs
- Retrain pair fusion with new Stage 1 agent outputs

**Step 3C**: Fix Stage 3 meta
- Replace MLP meta-learner with Logistic Regression (C=0.01)
- Fewer params = less overfitting
- Train on val set only (no leakage)

---

## Dependency Order

```
Phase 0 (Data) → Phase 1 (Architecture) → Phase 2 (Training) → Phase 3 (Stage 2/3)
                                                    ↓
                                          Train & evaluate single-symbol SPXW
                                          Validate acc/F1 improvements before Stage 2/3
```

**Blocker**: Part 2 ingestion (PID 2410) must complete TQ ingestion before Phase 0A can extract full TQ data. Greek ingestion into Part 2 is not needed (Part 1 already has full coverage).

---

## Expected Outcomes

| Metric | Current | Target | Source of Gain |
|---|---|---|---|
| Best single agent acc | 58.7% | 61–63% | Full TQ data, z-score norm, focal loss, better schedule |
| Best single agent F1 | 0.60 | 0.64–0.66 | Threshold optimization, label smoothing, focal loss |
| Agent T acc | 57.4% | 59–60% | Full TQ data (1690 dates vs 331), LayerNorm fix |
| Agent pairwise agreement | 85–92% | 70–75% | Feature subsetting, diverse backbones |
| Ensemble acc | 58.5% (worse than solo) | 62–64% | Diversity fixes enable real ensemble value |
| Stage 3 meta acc | 58.7% | 63–65% | Better base agents + drop VIXW + LogReg meta |
