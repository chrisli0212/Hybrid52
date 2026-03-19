Here's a proposal you can paste directly into Windsurf AI to onboard it on your project:

VIX Regium — Hybrid 5.1 Multi-Agent Binary Prediction System
Project Overview
VIX Regium is a multi-agent neural network system that predicts short-term market direction (UP or DOWN) for options-related instruments (SPXW, SPY, QQQ, IWM, TLT). The current focus is the 5-minute prediction horizon. The system uses 7 specialized AI agents, each looking at different aspects of market data, trained independently in Stage 1, then combined into an ensemble in Stage 2.

Data Pipeline
The data flows through 3 tiers:

Tier 1 → Raw per-tradedate Greek + Trade/Quote parquet files (pre-joined)

Tier 2 → 325-dimensional minute-bar feature vectors built by build_tier2_fast.py. Features are grouped into: Core Greeks (0–49), IV Surface (50–99), Term Structure (100–127), Flow/Volume (128–149), Microstructure (150–179), Sentiment/Regime (180–209), Cross-Strike-Time (210–239), Gamma Exposure (240–269), Smart Money (270–284), Volume Anomaly (285–296), Trade Conditions (297–306), Quote Pressure (307–324)

Tier 3 → Binary UP/DOWN sequences built by build_tier3_binary.py. Each sample is a (batch, 20, 325) tensor — 20 one-minute timesteps × 325 features. Labels are binary: price went UP (1) or DOWN (0) over the next N minutes. Data is split chronologically 60/20/20 (train/val/test). Z-score normalization is computed from training split only. Flat samples (return < 0.03%) are filtered out.

Architecture — 7 Independent Agents
Each agent is wrapped by IndependentAgent (in independent_agent.py), which applies:

Feature subsetting — each agent sees only its designated subset of the 325 features (configured in config/feature_subsets.py) to promote diversity

Temporal Backbone — a shared multi-scale 1D-CNN with depthwise separable convolutions (kernel sizes 3, 5, 7) + LayerNorm + optional attention pooling. Outputs a 128-dim temporal embedding. Agent K has NO backbone (static MLP only).

Agent-specific head — each agent processes data differently and returns 3 outputs: (score, confidence, signal)

Classifier — concatenates [score, confidence, temporal_embed] → Linear(258→128) → GELU → Dropout → Linear(128→64) → GELU → Dropout → Linear(64→1) → single logit

Agent Details
Agent	Role	Key Architecture	~Params	Feature Focus
A	General pattern agent	MLP encoder + backbone	~200k	Broad market features
B	Alternative general agent	MLP with hidden_dim=128	~150k	Different feature subset
C	Sequence pattern agent	Sequence model, seq_len=20, embed_dim=96	~180k	Temporal patterns
K	Static snapshot agent	Large MLP, hidden_dim=512, NO backbone	~300k	Point-in-time features only
T	Trade flow agent	Flow encoder + 1D-CNN + impact detector	~250k	Trade features (indices 270–306)
Q	Quote dynamics agent	Quote encoder + BiLSTM + imbalance detector	~200k	Quote features (indices 307–324)
2D	Option chain shape agent	2D-CNN treating option chain as image (strikes × time × greeks) + smile/skew detectors	~200k	5-greek × 20-strike × 20-timestep chain2d tensor
Confidence Mechanism
Agent-level: Agents T, Q output confidence via a dedicated nn.Linear(64,1) → sigmoid. Agent 2D outputs confidence as the 2nd neuron of its FC layer. Agents A, B, C, K also output confidence via their respective heads. None of these are directly supervised — they learn indirectly through backpropagation from the binary classification loss.

Final confidence (used for evaluation): Computed post-hoc as abs(prob - 0.5) * 2 where prob is the sigmoid of the final classifier logit. This is NOT a trained output — it's simply how far the prediction is from 50/50.

Training — Stage 1 (train_binary_agents_v2.py)
Each agent is trained independently as a binary classifier:

Loss: BinaryFocalLoss(gamma=2.0, alpha=positive_class_prior, label_smoothing=0.05) + 0.3 × SoftF1Loss

Optimizer: AdamW (lr=3e-4, weight_decay=0.01)

Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=1e-6)

Gradient clipping: max_norm=1.0

Gradient accumulation: 4 steps (effective batch size = 2048 from batch_size=512)

Training: 80 epochs, early stopping patience=15, best model saved by F1 score

Threshold optimization: After training, sweep thresholds 0.30–0.65 on validation set to maximize F1

Optional augmentations: Mixup (alpha=0.2), Gaussian noise (sigma=0.02)

Evaluation metrics: Accuracy, F1, AUC, IC (Spearman), Brier score, confidence buckets

Current Task — 5-Minute Horizon Focus
We want to optimize the entire system specifically for the 5-minute prediction horizon (horizon=5). This means:

Target label = did price go UP or DOWN 5 minutes from now

The 5-min horizon is the fastest/noisiest, so agents may need different hyperparameters than 15-min or 30-min

Key challenge: signal-to-noise ratio is low at 5 minutes

Directory Structure
text
workspace/
├── data/
│   ├── tier1v4/{SYMBOL}/{date}_greek.parquet, {date}_tq.parquet
│   ├── tier2_minutes_v4/{SYMBOL}_minutes.parquet
│   └── tier3_binary_v4/{SYMBOL}/horizon_{N}min/
│       ├── train_sequences.npy, train_labels.npy, train_returns.npy
│       ├── val_sequences.npy, val_labels.npy, val_returns.npy
│       ├── test_sequences.npy, test_labels.npy, test_returns.npy
│       ├── norm_mean.npy, norm_std.npy
│       └── metadata.json
├── Hybrid51/
│   └── models/
│       ├── backbone.py          (TemporalBackbone, AttentionPool)
│       ├── agents/              (AgentA, AgentB, AgentC, AgentK, AgentT, AgentQ, Agent2D)
│       └── independent_agent.py (IndependentAgent wrapper)
├── config/
│   └── feature_subsets.py       (per-agent feature indices)
├── scripts/
│   ├── phase0/build_tier3_binary.py
│   └── stage1/train_binary_agents_v2.py
└── stage_results/stage1/
    ├── {SYMBOL}_agent_{type}_classifier_h{horizon}.pt
    └── {SYMBOL}_h{horizon}_results.json
Key Constraints
All agents must output (score, confidence, signal) tuple — this interface is required by

Prepared using Claude Opus 4.6 Thinking
you do not need to write all you know, but focus on vix agent
You're right — here's a concise, focused proposal for Windsurf AI:

VIX Regium — 5-Minute Binary Prediction System
What It Is
A multi-agent system that predicts whether price goes UP or DOWN in the next 5 minutes for options instruments (SPXW, SPY, QQQ, IWM, TLT). It uses 7 specialized neural network agents, each trained independently on different slices of a 325-dimensional feature vector built from options Greeks, trade flow, and quote data.

Input
Each sample is a tensor of shape (batch, 20, 325) — 20 one-minute bars × 325 features. Data is z-score normalized using training-split stats. Flat samples (return < 0.03%) are filtered out. Split is chronological 60/20/20.

The 7 Agents
Agent	What It Sees	How It Works
A	Broad market features	MLP + temporal backbone
B	Different feature subset	MLP + temporal backbone
C	Temporal patterns	Sequence model + backbone
K	Static snapshot only	Large MLP, no backbone
T	Trade flow (idx 270–306)	Flow encoder + 1D-CNN
Q	Quote/spread (idx 307–324)	Quote encoder + BiLSTM
2D	Option chain as 2D image	2D-CNN + smile/skew detectors
Each agent outputs (score, confidence, signal). These are concatenated with a 128-dim temporal embedding and fed to a final classifier → single binary logit.

Training Setup (Stage 1)
Loss: BinaryFocalLoss (gamma=2.0, label_smoothing=0.05) + 0.3 × SoftF1Loss

Optimizer: AdamW (lr=3e-4, weight_decay=0.01)

Schedule: CosineAnnealingWarmRestarts

80 epochs, patience=15, gradient clipping at 1.0, effective batch size 2048

After training: threshold sweep (0.30–0.65) on val set to maximize F1

Current Focus
5-minute horizon — the fastest and noisiest timeframe. Signal-to-noise ratio is low. We need to optimize agent hyperparameters and evaluate whether confidence (currently abs(prob - 0.5) * 2) is reliable at this speed.

Key Files
models/independent_agent.py — wraps all agents with backbone + feature subsetting

models/backbone.py — shared temporal backbone (multi-scale 1D-CNN, 128-dim output)

models/agents/ — individual agent classes (AgentA, AgentB, etc.)

scripts/stage1/train_binary_agents_v2.py — training script

config/feature_subsets.py — which features each agent sees

Results saved to stage_results/stage1/{SYMBOL}_h5_results.json

Short, practical, and gives Windsurf AI everything it needs to start working without drowning it in detail. Want me to adjust the tone or add anything specific?

Prepared using Claude Opus 4.6 Thinking