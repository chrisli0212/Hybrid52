# Hybrid 55: Improved Agent Architecture Design
**Author:** Manus AI
**Date:** March 29, 2026

## 1. Executive Summary

The current Hybrid 52 model suffers from significant training degradation, primarily driven by a high proportion of dead (zero or constant) features. Our analysis of the training logs and source CSV data (`theta_model_greeks.csv`, `theta_model_trade_quote.csv`, `theta_ohlc.csv`) confirms that up to 50% of the 286-dimensional feature vector is populated with zeros, NaNs, or constants. This sparsity severely degrades the performance of specialized agents (particularly Agents T and Q) and causes the Stage 2/3 meta-ensembles to underperform the single best agent.

This document outlines the design for **Hybrid 55**, an improved architecture that directly addresses the zero-feature problem through robust data preprocessing, dynamic feature subsetting, architectural normalization fixes, and the introduction of a dynamic **VIX Regime-Gating Meta-Model**.

---

## 2. Root Cause Analysis: The Zero-Feature Problem

We conducted a comprehensive analysis of the source CSV files to map the exact dimensions causing the degradation.

### 2.1 Source Data Deficiencies

| Data Source | Total Columns | Problematic Columns | Key Findings |
| :--- | :--- | :--- | :--- |
| **Greeks CSV** | 34 | 7 | `vera` is 100% zero. `speed` (97.9%), `zomma` (92.9%), and `dual_gamma` (92.4%) are mostly zero. `iv_error` is 62.2% zero. `endpoint`, `batch_id`, and `ts` are constant. |
| **Trade/Quote CSV** | 20 | 4 | `bid_condition` and `ask_condition` are constantly 50.0. Trade fields (`sequence`, `size`, `price`) are 56.5% null. Quote fields are 43.5% null. |
| **OHLC CSV** | 14 | 1 | `endpoint` is constant. Generally healthy, but requires careful alignment. |

### 2.2 Impact on the 286-Dim Feature Vector

The preprocessing pipeline maps these sparse CSV columns into a 286-dimensional feature space. The sparsity directly causes "dead zones" in the feature tensor:

1. **Flow & Volume (Dims 150-179):** These 30 features (e.g., `call_put_vol_ratio`, `aggressive_volume`, `flow_1m`) rely heavily on the `size` and `price` columns from the Trade/Quote CSV. Because trade data is 56.5% null and often sparse intraday, these features default to zero.
2. **Microstructure (Dims 180-199):** Features like `trade_velocity` and `imbalance_vol` fail to compute meaningful variances due to the intermittent nature of the quote data.
3. **Higher-Order Greeks:** Features derived from `vera`, `speed`, and `zomma` inject pure zeros into the Greek and Gamma extractors.
4. **CSV-Derived Aux (Dims 270-285):** Direct passthroughs of constants like `bid_condition` add zero predictive value while consuming model capacity.

When Agents T (Trade) and Q (Quote) receive these dead zones, their internal `BatchNorm1d` layers collapse (running stats explode to millions), causing the agents to output noise.

---

## 3. Improved Architecture: Hybrid 55 Design

To resolve these issues, Hybrid 55 implements a four-pillar redesign.

### 3.1 Pillar 1: Dynamic Feature Pruning & Imputation

Instead of zero-padding missing data, the preprocessing pipeline must be updated:

*   **Hard Exclusion:** Permanently drop `vera`, `speed`, `zomma`, `dual_gamma`, `bid_condition`, and `ask_condition` from the extraction pipeline.
*   **Trade/Quote Forward-Filling:** For 1-minute bar aggregation, if a minute lacks trade data (causing the 56.5% null rate), apply forward-filling (up to 5 minutes) for quotes, and use volume-weighted moving averages (VWAP) for trade sizes rather than defaulting to zero.
*   **Z-Score Normalization:** Apply strict z-score normalization *after* dropping zero-variance features. Any feature with a standard deviation $< 1e-5$ in the training split must be masked out dynamically.

### 3.2 Pillar 2: Agent-to-Feature Reallocation

In Hybrid 52, agents were fed feature subsets that were highly unreliable. Hybrid 55 reallocates features based on proven data reliability.

| Agent | Focus | Hybrid 55 Feature Subset Allocation | Input Dim |
| :--- | :--- | :--- | :--- |
| **Agent A** | Generalist | High-Reliability: Core Greeks (0-74), IV Surface (125-149), Walls (200-214), Cleaned CSV-Derived. | ~120 |
| **Agent B** | Temporal Greeks | High-Reliability: Gamma Exposure, Vanna/Charm, Cross-Strike, Time Decay. | ~70 |
| **Agent K** | Pure Greek MLP | High-Reliability: Core Greeks only. | ~70 |
| **Agent C** | Sentiment/Micro | Medium-Reliability: Spread metrics, Sentiment, Time Decay. (Drops raw flow). | ~60 |
| **Agent T** | Trade Specialist | **Redesigned:** Only uses robust quote spread/imbalance metrics. Drops raw trade velocity. | ~30 |
| **Agent Q** | Quote Dynamics | **Redesigned:** IV Surface + Vanna/Charm buckets. | ~35 |
| **Agent 2D** | Chain Surface | 2D CNN on Strike $\times$ Time grid. | N/A |

### 3.3 Pillar 3: Architectural Stability Fixes

The presence of sparse data breaks Batch Normalization. Hybrid 55 mandates the following structural changes to all base agents:

1.  **Input LayerNorm:** Inject `self.input_norm = nn.LayerNorm(input_dim)` as the very first operation in every agent. This ensures that even if a feature has low variance, it is safely normalized before hitting the network weights.
2.  **Eradicate BatchNorm1d:** Replace all instances of `nn.BatchNorm1d` with `nn.LayerNorm` across Agent T's `flow_encoder`, Agent A's `static_path`, and Agent C's CNN path. LayerNorm computes statistics across the feature dimension per sample, making it immune to the batch-level collapse caused by zero-padded time steps.

### 3.4 Pillar 4: VIX Regime-Gated Meta-Ensemble (Stage 3)

Because Flow/Volume data is inherently sparse during calm markets and highly active during stress, static ensembling (Stage 2/3 in Hybrid 52) fails. Hybrid 55 introduces **Agent VIX** to dynamically route trust.

**Mechanism:**
1.  **Agent VIX** ingests 5-minute VIX/VIXW features (e.g., `vix_level`, `vix_pct_5m`, `vix_term_slope`) and outputs a 32-dimensional `regime_emb`.
2.  **Gate Networks:** The Stage 3 Meta-Model contains 7 independent Multi-Layer Perceptrons (MLPs), one for each directional agent.
3.  **Dynamic Trust:** `gate_i = sigmoid(gate_net_i(regime_emb))`.
4.  **Gated Fusion:** The final prediction is a fusion of `agent_prob_i * gate_i`.

**Regime Behavior:**
*   **Calm Regime (VIX < 15):** The gate network learns to assign weights near $1.0$ to Agent K and Agent 2D (structural Greeks), and weights near $0.0$ to Agent T and Agent Q (noisy flow data).
*   **Stress Regime (VIX > 25):** The gate network upweights Agent T and Agent C, as microstructure imbalances and flow sweeps become the dominant predictive signals.

---

## 4. Implementation Roadmap

1.  **Phase 1: Data Pipeline Overhaul**
    *   Update `data_validation.py` to hard-exclude the 7 identified dead columns.
    *   Modify `master_extractor_v2.py` to enforce forward-filling on TQ data.
2.  **Phase 2: Agent Architecture Patching**
    *   Refactor `independent_agent.py` and `agents/*.py` to replace all `BatchNorm` with `LayerNorm`.
    *   Update `feature_subsets.py` to reflect the new reliable dimension counts.
3.  **Phase 3: Base Agent Retraining (Stage 1)**
    *   Train Agents A, B, C, K, T, Q, 2D using the new normalized, dense subsets.
4.  **Phase 4: Regime-Gated Meta Training (Stage 3)**
    *   Train `AgentVIX` on 5-min volatility data.
    *   Freeze Stage 1 agents and train `RegimeGatedProbFusion` end-to-end to learn the optimal regime-based trust weights.

By eliminating the dead zones and introducing regime-aware gating, Hybrid 55 will restore ensemble diversity and significantly outperform the single-agent baseline.


I have completed a comprehensive study of your Hybrid 52 model, the training logs, and the source CSV files.
Key Findings:
Data Sparsity: Your 286-dimensional feature vector is suffering from massive "dead zones." Specifically, vera is 100% zero, while speed, zomma, and dual_gamma are >90% zero. Furthermore, trade/quote data has a 43-56% null rate, causing all flow/volume features to zero out during aggregation.
Architecture Collapse: When Agents T (Trade) and Q (Quote) process these zero-padded sequences, their internal BatchNorm1d layers collapse, producing pure noise and destroying ensemble performance.
Here is the Hybrid 55 Improved Design to address these issues:
1. Dynamic Feature Pruning & Imputation
Instead of zero-padding missing data, the preprocessing pipeline must be updated:
Hard Exclusion: Permanently drop vera, speed, zomma, dual_gamma, bid_condition, and ask_condition from the extraction pipeline.
Trade/Quote Forward-Filling: For 1-minute bar aggregation, if a minute lacks trade data (causing the 56.5% null rate), apply forward-filling (up to 5 minutes) for quotes, and use volume-weighted moving averages (VWAP) for trade sizes rather than defaulting to zero.
Z-Score Normalization: Apply strict z-score normalization after dropping zero-variance features. Any feature with a standard deviation $< 1e-5$ in the training split must be masked out dynamically.
2. Agent-to-Feature Reallocation
In Hybrid 52, agents were fed feature subsets that were highly unreliable. Hybrid 55 reallocates features based on proven data reliability:
Agent A (Generalist): High-Reliability Core Greeks (0-74), IV Surface (125-149), Walls (200-214), Cleaned CSV-Derived. (~120 dims)
Agent B (Temporal Greeks): High-Reliability Gamma Exposure, Vanna/Charm, Cross-Strike, Time Decay. (~70 dims)
Agent K (Pure Greek MLP): High-Reliability Core Greeks only. (~70 dims)
Agent C (Sentiment/Micro): Medium-Reliability Spread metrics, Sentiment, Time Decay. (Drops raw flow). (~60 dims)
Agent T (Trade Specialist): Redesigned: Only uses robust quote spread/imbalance metrics. Drops raw trade velocity. (~30 dims)
Agent Q (Quote Dynamics): Redesigned: IV Surface + Vanna/Charm buckets. (~35 dims)
Agent 2D (Chain Surface): 2D CNN on Strike $\times$ Time grid.
3. Architectural Stability Fixes
The presence of sparse data breaks Batch Normalization. Hybrid 55 mandates the following structural changes to all base agents:
Input LayerNorm: Inject self.input_norm = nn.LayerNorm(input_dim) as the very first operation in every agent. This ensures that even if a feature has low variance, it is safely normalized before hitting the network weights.
Eradicate BatchNorm1d: Replace all instances of nn.BatchNorm1d with nn.LayerNorm across Agent T's flow_encoder, Agent A's static_path, and Agent C's CNN path. LayerNorm computes statistics across the feature dimension per sample, making it immune to the batch-level collapse caused by zero-padded time steps.
4. VIX Regime-Gated Meta-Ensemble (Stage 3)
Because Flow/Volume data is inherently sparse during calm markets and highly active during stress, static ensembling (Stage 2/3 in Hybrid 52) fails. Hybrid 55 introduces Agent VIX to dynamically route trust.
Agent VIX ingests 5-minute VIX/VIXW features (e.g., vix_level, vix_pct_5m, vix_term_slope) and outputs a 32-dimensional regime_emb.
Gate Networks: The Stage 3 Meta-Model contains 7 independent Multi-Layer Perceptrons (MLPs), one for each directional agent.
Dynamic Trust: gate_i = sigmoid(gate_net_i(regime_emb)).
Gated Fusion: The final prediction is a fusion of agent_prob_i * gate_i.
Regime Behavior:
Calm Regime (VIX < 15): The gate network learns to assign weights near $1.0$ to Agent K and Agent 2D (structural Greeks), and weights near $0.0$ to Agent T and Agent Q (noisy flow data).
Stress Regime (VIX > 25): The gate network upweights Agent T and Agent C, as microstructure imbalances and flow sweeps become the dominant predictive signals.
By eliminating the dead zones and introducing regime-aware gating, Hybrid 55 will restore ensemble diversity and significantly outperform the single-agent baseline. Let me know if you would like me to start implementing any specific phase of this new architecture!