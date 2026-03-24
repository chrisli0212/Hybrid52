# Hybrid51 v2: Refactored Stage 1–3

Comprehensive refactoring of the Hybrid51 model pipeline based on audit findings.
See `PLAN.md` for the full refactoring plan and rationale.

## Key Changes from v1

### Architecture Fixes (Phase 1)
- **Agent T**: `BatchNorm1d` → `LayerNorm` + input `LayerNorm` (fixes exploding running stats)
- **Agent A**: `BatchNorm1d` → `LayerNorm` + residual connection
- **Agent C**: `BatchNorm1d` → `LayerNorm` in CNN path
- **Agent Q**: Added input `LayerNorm` for quote features
- **Agent 2D**: `BatchNorm2d` → `GroupNorm`
- **Backbone**: `BatchNorm1d` → `LayerNorm`

### Training Improvements (Phase 2)
- **Loss**: `BCEWithLogitsLoss` → `BinaryFocalLoss(gamma=2, alpha=0.52)` + soft-F1 (weight=0.3)
- **Optimizer**: `Adam(lr=5e-4, wd=1e-5)` → `AdamW(lr=3e-4, wd=0.01)`
- **Scheduler**: `ReduceLROnPlateau` → `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`
- **Grad clip**: 5.0 → 1.0
- **Epochs/Patience**: 25/7 → 80/15
- **Threshold**: Fixed 0.5 → optimized per-agent (sweep [0.30, 0.65])
- **Feature subsetting**: Each agent sees different features for diversity

### Data Pipeline (Phase 0)
- Full TQ coverage: 1690 dates (was 331) → eliminates 144 dead features
- Per-feature z-score normalization from training split
- Zero-variance feature detection and logging

### Stage 2/3 Fixes (Phase 3)
- **Drop VIXW pair** (51.78% high-conf accuracy, worse than random)
- **LogReg meta** replaces MLP (fewer params, less overfitting)
- **Diversity check** before Stage 2 (target: <75% pairwise agreement)

## Directory Structure

```
6. Hybrid51_new stage/
├── PLAN.md                          # Full refactoring plan
├── README.md                        # This file
├── config/
│   └── feature_subsets.py           # Per-agent feature subset definitions
├── hybrid51_models/
│   ├── agents/                      # Fixed agent architectures
│   │   ├── agent_a.py               # LayerNorm + residual
│   │   ├── agent_b.py               # Unchanged (already diverse)
│   │   ├── agent_c.py               # LayerNorm in CNN
│   │   ├── agent_k.py               # Unchanged (static specialist)
│   │   ├── agent_q.py               # Input LayerNorm
│   │   ├── agent_t.py               # LayerNorm + input norm (critical fix)
│   │   └── agent_2d.py              # GroupNorm
│   ├── backbone.py                  # LayerNorm (was BatchNorm)
│   └── independent_agent.py         # Feature subsetting support
├── hybrid51_preprocessing/          # Copied from stage3 (feature extractors)
├── scripts/
│   ├── phase0/                      # Data pipeline
│   │   ├── convert_csv_to_parquet.py
│   │   ├── ingest_to_duckdb.py
│   │   ├── extract_tier1.py
│   │   ├── build_tier2.py
│   │   └── build_tier3_binary.py
│   ├── stage1/
│   │   └── train_binary_agents_v2.py  # Focal loss, cosine, diversity
│   ├── stage2/
│   │   └── train_stage2_pairs.py      # VIXW dropped, diversity check
│   └── stage3/
│       └── train_stage3_meta.py       # LogReg meta-learner
├── checkpoints/                     # Model checkpoints
├── results/                         # Training results
└── logs/                            # Training logs
```

## Execution Order

```
Phase 0: Data Pipeline (must complete first)
  0.1  convert_csv_to_parquet.py   # Convert 6,876 CSVs → parquet
  0.2  ingest_to_duckdb.py         # Ingest new parquets to DuckDB
  0.3  extract_tier1.py            # Extract filtered Greek + TQ
  0.4  build_tier2.py              # MasterFeatureExtractor → 325-dim minute bars
  0.5  build_tier3_binary.py       # Build sequences with normalization stats

Phase 1: Already done (architecture fixes in agents/ and backbone.py)

Phase 2: Train Stage 1
  # Production default horizon is 30 min (Tier3 `horizon_30min/`). Override with --horizon if needed.
  python scripts/stage1/train_binary_agents_v2.py --symbol SPXW --horizon 30

Phase 3: Train Stage 2 + 3 (after Stage 1 diversity verified)
  python scripts/stage2/train_stage2_pairs.py --target SPXW --horizon 30
  python scripts/stage3/train_stage3_meta.py --target SPXW --horizon 30
```

## Expected Outcomes

| Metric | v1 (Current) | v2 (Target) | Source of Gain |
|---|---|---|---|
| Best agent acc | 58.7% | 61–63% | Full TQ data, z-score, focal loss |
| Best agent F1 | 0.60 | 0.64–0.66 | Threshold optimization, focal loss |
| Agent T acc | 57.4% | 59–60% | Full TQ (1690 dates), LayerNorm fix |
| Agent agreement | 85–92% | 70–75% | Feature subsetting |
| Ensemble acc | 58.5% | 62–64% | Diversity fixes |
| Stage 3 meta | 58.7% | 63–65% | Drop VIXW + LogReg meta |
