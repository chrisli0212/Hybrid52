# Hybrid52 Revised Agents and Related Files Report for Cursor AI

## Overview
This report captures what was amended in the revised Hybrid52 agent files and closely related preprocessing and planning files, based on the conversation history and attached planning documents. The purpose is to give Cursor AI a precise implementation record: what changed, why it changed, what technical risks those changes were meant to fix, and what follow-up wiring work is still required before retraining.[cite:111][cite:108]

The key architectural theme across the amendments was a shift away from blindly trusting the legacy Hybrid51-style feature contract and toward agents grounded in real, populated Theta Data fields. Several agents were redesigned to use compact feature domains such as 53, 34, 25, and 20 dimensions, while the legacy training pipeline still assumed a shared 325-dim sequence tensor sliced by old index ranges. That mismatch is now recognized as the central blocker before retraining.[cite:111][cite:108]

## Main intent of the revisions
The agent revisions were not cosmetic. They were driven by a reliability problem in the SPX directional model: many legacy features were dead, near-constant, structurally absent, or poorly matched to the actual historical chain data. The revised work aimed to improve directional prediction reliability by reducing dead-input contamination, removing silent failure modes, widening useful feature paths, and making each agent more specialized around signal that plausibly exists in the available data.[cite:111][cite:108]

The user’s stated objective is a directional SPX model for decision support rather than auto-trading, with accuracy and reliability as the highest priorities. That objective favors conservative architecture choices, explicit failure behavior, and agent designs that abstain or degrade gracefully when their input domain is weak, instead of silently proceeding on fake or structurally empty data.[cite:108]

## Agent-by-agent amendments

## Agent A
Agent A was revised from a legacy broad-input baseline into a compact, cleaned primary options-structure model. Its constructor now declares `input_dim=53`, explicitly documenting that it is grounded in actual Theta Data columns and no longer uses the broader legacy input that contained many zero or constant fields.[cite:111]

The architecture was widened and gated in a targeted way. The old causal-CNN summary path was replaced by a conv path followed by both adaptive max pooling and adaptive average pooling, concatenated into a 96-dim representation. A learned scalar gate was added on top of this 96-dim CNN output so the model can modulate how much recent local temporal structure should influence the fused decision state.[cite:111]

The reason for this amendment was to reduce dependence on dead columns and improve the model’s ability to separate trend-like information from spike-like information. Max pooling alone emphasizes the strongest local burst, while avg pooling preserves the broader temporal level; using both makes the temporal summary more stable for directional prediction, especially in a regime where reliability matters more than hyper-reactivity.[cite:111]

The practical amendment summary for Cursor AI is:
- `input_dim` changed to 53
- residual projection kept, but now aligned to 53 real features
- causal CNN changed from max-pool-only to max+avg pooled summary
- `self.gate = nn.Sequential(nn.Linear(96, 1), nn.Sigmoid())` added
- fusion path changed to `96 + temporal_dim + 96`
- comments now explicitly document why 53 was chosen.[cite:111]

## Agent B
Agent B was described in the conversation history as a major rewrite rather than a small patch. The intent was to convert it from an overly broad, weakly specified BiLSTM sequence learner into a narrower sequence specialist with separate sequence and static semantics, plus better temporal modeling for directional reliability.[cite:105][cite:108]

The revised specification from the session states that Agent B moved from `input_dim=158` to a compact design with approximately `34` sequence features and `53` static features, with explicit separation between `seq` and `static`. It also added a stronger temporal stack: a two-layer BiLSTM with normalization, attention pooling with recency bias, momentum deltas, time-of-day embeddings, and a parallel TCN branch before final fusion.[cite:105]

The reason for this amendment was twofold. First, the earlier design mixed static and sequential semantics too loosely and included too many dead or weak features. Second, the old design was vulnerable to shape confusion and underpowered temporal fusion, which undermines confidence calibration and regime robustness in a directional SPX setting.[cite:105][cite:108]

For Cursor AI, the important implementation memory is not just “Agent B was improved,” but exactly what that implies for wiring:
- Agent B no longer fits the legacy assumption that a single flat subset can be passed as one undifferentiated input.
- Agent B requires correct handling of sequence vs static dimensions.
- `feature_subsets.py` and `independent_agent.py` must produce the right slices and route them correctly.
- Any Stage 1 training path that still assumes one old broad tensor contract will mis-wire Agent B or crash.[cite:108]

## Agent C
Agent C was revised to fix both information loss and redundant feature extraction. In the session notes, the main bugs identified were that its embedding layer truncated the input to only the first 32 dimensions and that its backbone gating logic used only part of the pooled representation, effectively making later attention work redundant or badly conditioned.[cite:105]

The corrective amendment was to make Agent C consume the full intended input rather than truncating it, and to widen the gate input so the gating decision uses the full BiLSTM pooled representation rather than a sliced prefix. The conversation record describes this as changing the effective gate input from 96 to 192 dimensions and removing hidden truncation so Agent C can actually use the information it was supposed to model.[cite:105]

The reason for this change was reliability, not merely capacity. Hidden truncation means the model silently discards part of the feature space without any explicit design justification, which is exactly the kind of silent mismatch that can make a directional model appear functional while underperforming. Cursor should treat Agent C as an example of why all agent input contracts must be verified numerically rather than assumed from comments.[cite:105][cite:108]

For follow-up wiring, Cursor should remember:
- Agent C conceptually follows the same compact-contract problem as Agent B
- static-vs-sequence semantics must be explicit
- any legacy subset slicing must be recalculated to match the revised expected dims
- comments and docstrings should be updated to describe the actual input path rather than the older Hybrid51 behavior.[cite:108]

## Agent K
The session notes identify Agent K as one of the compact agents expected to consume a `53`-dimensional input domain after redesign. Although fewer architecture-level specifics were captured for K than for A, B, or C, the planning documents explicitly list it among the agents whose revised compact dimension contract conflicts with the old `feature_subsets.py` slicing logic.[cite:108]

The reason this matters is that K cannot be left on autopilot within a 325-dim legacy pipeline. Even if its code is structurally simple, a wrong subset slice means the model receives the wrong features in the wrong order or wrong dimension, which is functionally equivalent to training on mislabeled inputs. For reliability-first SPX direction prediction, that is unacceptable.[cite:108]

Cursor should therefore treat Agent K as a wiring-sensitive compact specialist that needs exact subset alignment, even if its internal architecture required fewer conceptual changes than the others.[cite:108]

## Agent Q
Agent Q was revised because its prior architecture produced too narrow a useful encoded representation and because its feature domain was conceptually tied to quote-flow dynamics that are not fully present in the available historical chain snapshots. The session notes describe adding a gated residual encoder and removing a meaningless single-layer LSTM dropout setting that only produced warnings without adding real regularization.[cite:105]

The residual-encoder amendment changed Q from a brittle bottleneck into a blended encoder that can combine transformed information with a residual view of the original quote-domain input. This is important because quote-pressure signals are often subtle and noisy; overcompressing them can make the agent degenerate into a weak directional prior rather than a specialized contributor.[cite:105]

However, the revised planning document adds a second layer of correction: even a better Q architecture is not enough if the underlying historical source does not actually contain rich OPRA-style quote update data. The revised plan says Agent Q must either be remapped to the fields that truly exist in chain history, such as `bid_size`, `ask_size`, `spread`, and `spread_pct`, or be disabled until a real TQ source exists.[cite:108]

For Cursor AI, the amendment memory is:
- architecture improvement already done: gated residual encoder, dropout warning fix
- data-domain alignment still unresolved: Q’s feature domain must be remapped or explicitly constrained to what the source data actually has
- `feature_subsets.py` and potentially a new feature-config file are needed to lock this domain correctly.[cite:105][cite:108]

## Agent T
Agent T was revised because its previous max-pool temporal summarization led to an unstable directional bias, especially in persistent trend regimes. The session notes state that `AdaptiveMaxPool1d(1)` was replaced with a recency-weighted pooling scheme computed dynamically per forward pass, giving greater weight to later sequence elements while still aggregating across the full time dimension.[cite:105]

The reason for this change was very specific: a max-pooled temporal path can collapse into a spike detector, which in a low-volatility or persistently bullish regime may cause the model to over-fire in one direction and lose discriminative power. The revised weighted pooling was designed to preserve recency sensitivity without turning the agent into a one-spike or one-regime classifier.[cite:105]

But, like Agent Q, Agent T also has a data-domain problem that architecture alone cannot solve. The revised plan says its original intended trade-flow feature space is not reliably available in `/workspace/historical_data_1yr`, because the source files are chain snapshots rather than true OPRA trade/quote streams. That means T must either be remapped to available quote-pressure proxies or be excluded from the retrain until a real TQ source is available.[cite:108]

For Cursor AI, the amendment record is therefore dual:
- internal architecture change: recency-weighted temporal pooling replaced max pooling
- external wiring change still needed: redefine Agent T’s domain to use real available fields or disable the agent for the retrain.[cite:105][cite:108]

## Agent 2D
Agent 2D was amended to remove a dangerous silent failure mode. The conversation history states that when `chain_2d=None`, the agent previously proceeded using synthetic Gaussian noise. That behavior was patched so bad chain data would trigger a warning rather than quietly contaminating training.[cite:105]

The reason for this amendment is fundamental to model reliability. A directional SPX model may survive missing a modality by skipping a weak specialist, but it should never silently fabricate that modality and continue as though the data were real. Silent synthetic fallback destroys experimental validity, because the training run looks successful while the model is learning from noise.[cite:105][cite:108]

The revised planning document adds an important correction for Cursor: do not re-apply this patch blindly. First verify whether the patch is already present in `agent_2d.py`, then lock the rest of the chain_2d contract around file naming, tensor shape, split generation, and model input shape. The plan defines the intended downstream contract as monolithic builder output feeding split files that Agent 2D consumes only when real chain tensors are available.[cite:108]

## Related non-agent files revised or introduced

## feature_config_agent_a.py
A new preprocessing support file was created to formalize the 53 real Agent A features. The session notes describe it as grouping those features into categories such as ATM Greeks, GEX/vanna/charm, OI structure, IV surface, liquidity, quote imbalance, bucketed OI, and DTE structure. It also explicitly documents dead historical fields that should be dropped, including `open`, `high`, `low`, `close`, `count`, `volume`, `vwap`, `bid_exchange`, and `ask_exchange`.[cite:105]

The reason this file matters is not only for Agent A itself. It serves as a proof-of-concept showing how to ground the feature space in real historical data instead of inherited legacy assumptions. The revised wiring plan recommends using this file as a validation reference for deciding which parts of the 325-feature shared stack are genuinely informative and which are dead or replaceable.[cite:105][cite:108]

## extract_agent_a_features.py
Another new preprocessing file was created to actually construct the Agent A snapshot feature vector from historical chain and OI data. The session notes describe it as joining OI on strike and right, clipping lambda, and computing the full 53-feature set from real fields in the raw data.[cite:105]

The reason for this addition was to give the project at least one concrete, data-grounded extraction path that reflects real schema availability. Even if Hybrid52 ultimately keeps the shared 325-feature Tier3 path as canonical, this extractor is valuable as a mapping reference and as evidence that the compact redesign is anchored in actual source columns rather than arbitrary reduction.[cite:105][cite:108]

## feature_subsets.py
This file was not described as already amended in code during the session, but it is the most important related file that still needs amendment. The revised planning document identifies it as the main contract file that still slices the legacy 325-dim tensor using old range assumptions such as broad blocks like `(0, 50)`, `(50, 100)`, and so on. That is incompatible with the revised compact-agent contracts.[cite:108]

The reason this file is central is simple: even perfectly redesigned agents will fail if their upstream subset slices produce the wrong dimensions or wrong semantic groups. Cursor should treat `feature_subsets.py` as the first wiring file to modify before any retraining or path cleanup starts.[cite:108]

## independent_agent.py
This is the second crucial related file. The revised plan says it must be audited because it is the integration point that constructs agents and passes inputs into them. For agents such as B and C, where sequence and static semantics were revised, `independent_agent.py` can no longer assume that “subset dim equals one uniform input_dim” is a valid rule.[cite:108]

The reason this matters is that the internal agent improvements only help if the caller routes the right tensors to the right arguments. Cursor should inspect this file for assumptions about constructor signatures, static vs seq handling, and whether temporal features are optional, synthetic, or required.[cite:108]

## chain_2d.py, build_chain_2d.py, split_chain_2d.py
The revised plan calls these out as a family that still needs contract cleanup even after the `agent_2d.py` fallback fix. The issues are not merely file paths; they include monolithic-vs-split naming conventions, crop and transpose behavior, and the final stored tensor shape consumed by the model.[cite:108]

The reason these files are grouped together is that Agent 2D reliability depends on the entire chain pathway being explicit end to end. A good model class cannot compensate for inconsistent builder output or ambiguous split logic. Cursor should treat the chain_2d stack as one contract, not as isolated scripts.[cite:108]

## Reasoning behind the whole revision set
The broader reason behind all these amendments is the same: the original Hybrid51-derived training tree mixed old namespaces, old path defaults, old normalization artifacts, old subset assumptions, and newer compact agent redesigns. That hybrid state creates subtle false-success scenarios where code runs but the model is mis-specified, mis-normalized, or trained on the wrong semantic inputs.[cite:108]

The revised planning document makes clear that the highest-priority fix is not cosmetic refactoring but contract alignment. The current accepted ordering is: resolve feature-contract alignment first, decide how T/Q will be supported from actual data, then clean namespaces and artifact roots, then regenerate Tier3 and normalization files, then finalize the chain_2d contract, and only then retrain.[cite:108]

## What was amended vs what is still only planned
It is important for Cursor AI to distinguish between amendments already made in session and amendments that are only specified in planning documents.

The following were described in the session history as already amended at the code level:
- Agent A compact rewrite and gating update
- Agent B major compact rewrite
- Agent C truncation and gate fixes
- Agent Q residual encoder and dropout fix
- Agent T recency-weighted pooling fix
- Agent 2D silent synthetic fallback patch
- `feature_config_agent_a.py` creation
- `extract_agent_a_features.py` creation.[cite:105][cite:111]

The following are described as still needing code work before retraining is safe:
- recalculation of `feature_subsets.py`
- auditing and likely changes in `independent_agent.py`
- namespace cleanup from Hybrid51 imports to Hybrid52 imports
- artifact root standardization
- regeneration of norm files in a new Tier3 root
- explicit T/Q data-domain remapping or disablement
- final chain_2d path/shape contract enforcement.[cite:108]

## Cursor AI action guidance
Cursor AI should use this report as an implementation memory document, not as a command to retrain immediately. The most important operational lesson from the revision history is that model reliability now depends more on contract correctness and data realism than on adding more complexity.[cite:108]

The recommended next engineering order is:
1. audit `feature_subsets.py` against the revised agent constructor dims
2. audit `independent_agent.py` for routing of static, seq, and temporal tensors
3. verify the actual code state of each revised agent file against the intended amendments recorded here
4. decide the short-term strategy for Agents T and Q based on real data availability
5. standardize output roots and regenerate Tier3 + norm files
6. lock the chain_2d builder/split/model contract
7. only then begin Stage 1 retraining.[cite:108]

## File-by-file amendment summary

| File | What changed or must change | Why |
|---|---|---|
| `hybrid52_models/agents/agent_a.py` | `input_dim=53`, max+avg pooled CNN, learned gate, new fusion contract | Remove dead fields, improve temporal stability, align to real Theta data [cite:111] |
| `hybrid52_models/agents/agent_b.py` | Major compact rewrite with seq/static separation, stronger temporal stack | Improve sequence modeling and remove broad dead-input contamination [cite:105] |
| `hybrid52_models/agents/agent_c.py` | Removed hidden truncation, widened gate input, better backbone interplay | Prevent silent information loss and redundant extraction [cite:105] |
| `hybrid52_models/agents/agent_k.py` | Compact-contract alignment required | K must receive exact intended subset dims, not old generic slices [cite:108] |
| `hybrid52_models/agents/agent_q.py` | Gated residual encoder, dropout warning fix, domain remap still needed | Improve quote specialist while adapting to real available data [cite:105][cite:108] |
| `hybrid52_models/agents/agent_t.py` | Recency-weighted pooling replaced max pooling, domain remap still needed | Reduce regime collapse and align to actual data availability [cite:105][cite:108] |
| `hybrid52_models/agents/agent_2d.py` | Silent synthetic fallback removed or intended to be removed; verify patch state | Never train on fake chain tensors [cite:105][cite:108] |
| `hybrid52_preprocessing/feature_config_agent_a.py` | New compact feature spec | Document real 53-field domain and dead-column exclusions [cite:105] |
| `hybrid52_preprocessing/extract_agent_a_features.py` | New real-data extractor | Ground compact design in actual chain + OI schema [cite:105] |
| `config/feature_subsets.py` | Still needs recalculation | Legacy slice contract conflicts with compact revised agents [cite:108] |
| `hybrid52_models/independent_agent.py` | Still needs audit/amendment | Routing layer must respect new static/seq contracts [cite:108] |
| `hybrid52_preprocessing/chain_2d.py` and related chain files | Still need contract cleanup | Lock builder, split, storage, and model input behavior [cite:108] |

## Final interpretation for Cursor AI
The revised agent work should be understood as a partial but meaningful architectural cleanup focused on model reliability. It successfully identified and corrected several agent-level weaknesses: dead-feature contamination, hidden truncation, unstable temporal summarization, narrow encoder bottlenecks, and silent synthetic fallback. But those improvements are only safe and useful if the surrounding pipeline is rewritten to honor the new contracts.[cite:105][cite:108]

In short, the agent files were made smarter and more realistic, but the training tree is not yet fully coherent. Cursor AI should preserve the revised agent logic, avoid undoing the compact redesigns, and focus next on the upstream contract and data-path wiring that allows those redesigns to work as intended in a real Hybrid52 retrain.[cite:108]
