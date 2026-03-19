# Hybrid51 Model Architecture, Agents, and Stage 3 Meaning

This document explains the live model used in `Final_production_model` in plain language.

## 1) High-level architecture (3 stages)

The production predictor runs in `prediction_service.py` and uses a **3-stage ensemble**:

1. **Stage 1: Per-symbol, per-agent binary models**
   - Symbols: `SPXW`, `SPY`, `QQQ`, `IWM`, `TLT`
   - Agents: `A`, `B`, `C`, `K`, `T`, `Q`, `2D`
   - Total models: `5 symbols x 7 agents = 35` models
   - Output from each model: probability that target direction is up (`P(up)` for the horizon).

2. **Stage 2: Cross-symbol fusion per agent**
   - One fusion model per agent (`7` total).
   - Each Stage 2 agent combines Stage 1 outputs across symbols.
   - Standard agents (`A/B/C/K/T/Q`) use 13 input features.
   - Agent `2D` uses 14 input features (includes TLT peer context).
   - Output: one refined probability per agent (7 probabilities total).

3. **Stage 3: Meta-ensemble (final decision)**
   - Current production method: **LogisticRegression** (`stage3_logreg.joblib`)
   - Inputs: 13 meta features:
     - 7 Stage 2 agent probabilities
     - 6 summary stats: mean, std, spread, majority ratio (>0.5), max, min
   - Output:
     - `prob`: final probability from the meta model
     - `pred`: final class based on threshold
   - Current threshold: **0.36** (from `config/production_config.json`)

---

## 2) What each agent means

Your system has seven specialist agent streams:

- `A`, `B`, `C`, `K`, `T`, `Q`: independent directional agents with different learned feature emphases.
- `2D`: chain-context-aware agent that also uses chain-derived context features and broader peer structure.

How to think about them:

- Each agent is a different "opinion generator."
- Stage 1 gives each opinion at each symbol level.
- Stage 2 upgrades each agent opinion by checking agreement/divergence across symbols.
- Stage 3 combines all agent opinions into one final market direction call.

---

## 3) Stage 3 prediction meaning (important)

In `prediction.csv`, these columns are the key interpretation fields:

- `prob`: Stage 3 meta-model probability for class 1 (bull/up class).
- `threshold`: decision cutoff (currently `0.36`).
- `pred`: binary decision:
  - `1` if `prob > threshold`
  - `0` otherwise
- `direction`:
  - `BULL` when `pred = 1`
  - `BEAR` when `pred = 0`
  - `SUPPRESSED` when warmup/data-quality checks block prediction

### Why can `prob` be around 0.44 and still be `BULL`?

Because your threshold is **0.36**, not 0.50.  
So any `prob > 0.36` becomes `pred = 1` (`BULL`).

Example:

- `prob = 0.45`
- `threshold = 0.36`
- Result: `pred = 1`, `direction = BULL`

This is expected behavior and matches your configured operating point.

---

## 4) Confidence vs probability (different concepts)

`prob` and `confidence` are not the same:

- `prob` = output of Stage 3 logistic model.
- `confidence` = separate evidence score from `confidence.py`, computed from:
  1. agent agreement (low dispersion among 7 Stage 2 probs),
  2. consensus ratio (fraction of agents on predicted side),
  3. gate-weighted conviction (`|p-0.5|` strength),
  4. data quality (feature completeness + warmup progress).

Current production note:

- Stage 3 gates in live service are fixed at `1.0` (regime-gated fusion not active in this deployed path).
- So gate columns are present for transparency, but not dynamically changing yet.

---

## 5) Suppressed predictions

Before releasing a normal prediction, the service can suppress output when data is not ready.

Common suppression reasons:

- warmup not complete (for sequence history build-up),
- invalid VIX feature vector,
- missing/insufficient snapshot quality,
- Stage 3 model not loaded.

When suppressed:

- `direction = SUPPRESSED`
- `prob = 0.5`, `confidence = 0.0`
- agent probabilities default to neutral placeholders

---

## 6) Practical reading guide for your `prediction.csv`

Use this quick read flow:

1. Check `suppressed` and `reason` first.
2. If not suppressed, read `prob` vs `threshold` to understand the class decision.
3. Confirm `direction` (`BULL`/`BEAR`) from `pred`.
4. Use `confidence` + component fields (`conf_*`, `agent_std`, `consensus_ratio`) to assess reliability.
5. Review per-agent Stage 2 probabilities (`agent_*_prob`) to see ensemble agreement.

This gives both the **decision** (direction) and the **decision quality** (confidence and consensus).
