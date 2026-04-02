# Hybrid 55 Architecture Review & Cursor AI Action Plan

This document outlines the findings from a deep audit of the amended Hybrid 55 codebase. It is designed specifically for Cursor AI to systematically resolve architectural gaps, dimensional mismatches, and legacy debt remaining from the Hybrid 52 transition.

---

## 1. Critical Missing File: `config/feature_subsets.py`

The most critical issue is that the amended codebase heavily relies on `config/feature_subsets.py`, but this file was **not updated** to reflect the new 311-dimensional layout or the new agents. 

**Current State (Broken):**
* `TOTAL_FEAT_DIM` is still hardcoded to `286`.
* Agent T and Agent Q are still defined separately.
* Agent TQ and Agent H are **missing entirely**.
* `independent_agent.py` and `train_binary_agents_v2.py` import this file and will crash or silently fail when trying to load subsets for TQ and H.

**Cursor AI Action Required:**
Rewrite `config/feature_subsets.py` completely.
1. Update `TOTAL_FEAT_DIM = 311`.
2. Add `AGENT_FEATURE_SUBSETS['TQ']` mapping to ranges `[(150, 200), (270, 286)]` (or wherever the new Trade/Quote and Phase 1 features are strictly located based on `feature_config_v2.py`).
3. Add `AGENT_FEATURE_SUBSETS['H']` mapping strictly to the OHLC block: `[(286, 311)]`.
4. Update Agent A, B, C, K subsets to safely ignore the new TQ/OHLC blocks if they shouldn't see them, or include them if they should.

---

## 2. Inconsistent Modality Definitions in Tier 3 Builder

The script `scripts/phase0/build_tier3_binary.py` has conflicting definitions for the Trade/Quote (TQ) block, which corrupts the sparse-aware zero-variance rescue policy.

**Current State (Bugged):**
* `MODALITY_RANGES` defines TQ as `(150, 286)`.
* However, lines 57-58 define `TQ_FEAT_START = 270` and `TQ_FEAT_END = 286`.
* The diagnostic coverage metric on line 380 only checks `all_features[:, TQ_FEAT_START:TQ_FEAT_END]`, which is actually the `CSV_DERIVED` block, completely ignoring the real Trade/Quote features (150-200).

**Cursor AI Action Required:**
1. In `build_tier3_binary.py`, align `TQ_FEAT_START` and `TQ_FEAT_END` with the actual Trade/Quote block defined in `feature_config_v2.py` (likely 150 to 200).
2. Ensure the `SPARSE_KEEP_MIN_NONZERO_RATIO` logic correctly applies to the true TQ block and the new OHLC block (286-311).
3. Remove all legacy comments referencing the old 325-dim or 286-dim layouts.

---

## 3. Stale Stage 3 Inference & Meta-Model Namespaces

The Stage 3 meta-model and inference scripts were only partially updated, leaving dangerous references to the old `hybrid51` namespace and the old Agent T/Q lineup.

**Current State (Bugged):**
* `scripts/stage3/infer_stage3_meta.py` imports from `hybrid51_utils` and `hybrid51_models` instead of `hybrid55_*`.
* It defines `ALL_AGENTS = ['A','B','C','K','T','Q','2D']` instead of the new `['A','B','C','K','TQ','H','2D']`.
* `train_stage3_cross_agent_meta.py` has updated the agent list, but its docstrings and internal comments still refer to `prob_T` and `prob_Q`.

**Cursor AI Action Required:**
1. In `infer_stage3_meta.py`, change all `hybrid51` imports to `hybrid55`.
2. Update `ALL_AGENTS` to match the Stage 1 trainer: `['A', 'B', 'K', 'C', 'TQ', 'H', '2D']`.
3. Ensure the inference loop expects 7 probabilities corresponding to the new lineup.
4. Clean up stale docstrings in `train_stage3_cross_agent_meta.py`.

---

## 4. `IndependentAgent` Forward Pass Mismatch

The `independent_agent.py` orchestrator was updated to support TQ and H, but it still contains legacy routing for T and Q that bypasses the feature subsetting logic in confusing ways.

**Current State (Technical Debt):**
* Lines 186-193 contain hardcoded `trade_feat_start/end` slicing for Agent T and Q.
* Since the pipeline now trains `TQ` instead of `T` and `Q` separately, this code is dead but risky.
* Agent H and TQ are routed correctly (lines 201-202), but the existence of the old T/Q logic clutters the forward pass.

**Cursor AI Action Required:**
1. Remove `trade_feat_start`, `trade_feat_end`, `quote_feat_start`, and `quote_feat_end` from the `IndependentAgent` constructor.
2. Delete the specific `if self.agent_type == 'T':` and `elif self.agent_type == 'Q':` blocks in the `forward` method.
3. Rely entirely on the `_select_features` subsetting mechanism for all agents.

---

## 5. Normalization Double-Dip Risk

The amended codebase introduced `LitAgent` in `train_binary_agents_v2.py`. While the author noted removing double-normalization, there is still a structural risk.

**Current State (Risk):**
* `SequenceWithOptionalChainDataset` applies normalization in `__getitem__`.
* `LitAgent.validation_step` *also* applies normalization because the validation dataloader is assumed to pass raw sequences.
* If `SequenceWithOptionalChainDataset` is ever instantiated with `norm_mean` for the validation set, the data will be normalized twice.

**Cursor AI Action Required:**
1. In `train_binary_agents_v2.py`, enforce a strict contract: The Dataset *never* normalizes. 
2. Move the normalization logic entirely into the `LitAgent`'s `forward`, `training_step`, and `validation_step` using registered buffers. This guarantees normalization is applied exactly once, on the GPU, and is saved perfectly with the checkpoint.

---

## Summary of Successes (Do Not Change)
* **LayerNorm Migration:** `backbone.py` successfully replaced `BatchNorm1d` with `LayerNorm` and implemented proper Causal Convolutions (`DilatedCausalTCN`).
* **OHLC Integration:** `ohlc_features.py` and `AgentH` are well-implemented.
* **Stage 3 Gating:** `AgentVIX` and `RegimeGatedProbFusion` correctly implement the dynamic regime-gating concept.

Here is the detailed Cursor AI improvement report. It identifies 5 critical issues ranked by severity, each with exact file paths, line numbers, and specific instructions for Cursor AI to fix.
Summary of the 5 Issues:
CRITICAL — Missing config/feature_subsets.py update: TOTAL_FEAT_DIM is still 286 (should be 311), and Agents TQ and H are completely missing from the subset definitions. This will crash at import time.
BUG — Inconsistent TQ modality ranges in build_tier3_binary.py: MODALITY_RANGES says TQ is dims 150–286, but the diagnostic constants TQ_FEAT_START/END only check 270–286 (the CSV_DERIVED block). The sparse-aware rescue policy and coverage metrics are therefore wrong.
BUG — Stage 3 inference uses old hybrid51 namespace: infer_stage3_meta.py imports from hybrid51_utils/hybrid51_models and uses the old agent lineup [A,B,C,K,T,Q,2D] instead of the new [A,B,C,K,TQ,H,2D].
DEBT — Dead T/Q routing in independent_agent.py: The forward pass still has hardcoded trade_feat_start/end slicing for the old separate T and Q agents, which is dead code that adds confusion.
RISK — Normalization double-dip: The Dataset normalizes in __getitem__ for training, but LitAgent.validation_step also normalizes. If the contract is ever violated, data gets normalized twice.
The report also confirms what was done correctly and should not be changed: LayerNorm migration, OHLC feature extractor, Agent H architecture, and the VIX regime-gated meta-model.
