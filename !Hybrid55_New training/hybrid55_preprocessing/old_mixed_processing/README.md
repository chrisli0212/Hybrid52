# Old Mixed Data Processing Files - BACKUP

## Purpose
This directory contains backup copies of the old "mixed" data processing architecture that processed features for all agents together in a monolithic fashion. These files were moved here to prevent accidental use and to clearly separate the new per-agent architecture from the legacy code.

## Problem with Old Architecture
The old architecture had a single `MasterFeatureExtractor` that processed all 311 features for all 7 agents in one pass. This created several critical issues:

1. **Cross-contamination**: One agent's feature failure affected all agents
2. **No isolation**: A bug in one extractor block shifted all subsequent features
3. **Debugging nightmare**: Impossible to tell which agent's feature caused failures
4. **Training failures**: Zero and constant field data at stage1 training
5. **Dimension mismatches**: TOTAL_FEATURES disagreed across 4 different files

## New Architecture
The new per-agent architecture (`hybrid55_preprocessing/agents/`) provides:

1. **Complete isolation**: Each agent has dedicated `extractor.py`, `feature_config.py`, `validator.py`
2. **Shared computation**: Common extractors in `extractors/` subfolder (reusable, not duplicated)
3. **Import-time assertions**: Dimension checks catch configuration drift immediately
4. **Safe extraction**: `_safe_extract()` wrapper prevents one failure from breaking others
5. **Zero-field alerts**: Per-agent validation catches dead features early

## Files in This Backup

### Master Extractors (Monolithic Pattern)
- `master_extractor.py` - Original 270-feature master extractor
- `master_extractor_v2.py` - Extended with Phase 1 (311/366 features)
- `training_pipeline.py` - Training pipeline that used master extractor

### Root-Level Monolithic Feature Extractors
Located in `extractors_root_level/`:
- `greek_features.py` - 75 Greek features
- `gamma_exposure.py` - 50 Gamma/Vanna/Charm features
- `iv_surface.py` - IV Surface features (25)
- `flow_volume.py` - Flow/Volume features (30)
- `microstructure.py` - Microstructure features (20)
- `walls_positioning.py` - Walls/Positioning features (20)
- `cross_strike_time.py` - Cross-strike and time decay (30)
- `sentiment_regime.py` - Sentiment/Regime features (20)
- `smart_money.py` - Smart Money Detector (15, Phase 1)
- `volume_anomaly.py` - Volume Anomaly Detector (12, Phase 1)
- `trade_conditions.py` - Trade Condition Analyzer (10, Phase 1)
- `quote_pressure.py` - Quote Pressure Analyzer (18, Phase 1)
- `csv_derived.py` - CSV-derived enrichments (16)
- `ohlc_features.py` - OHLC dynamics (25)

### Legacy Agent Files
Located in `legacy_agent_files/`:
- `extract_agent_a_features.py` - Old Agent A extractor (superseded by `agents/agent_a/extractor.py`)
- `feature_config_agent_a.py` - Old Agent A config (superseded by `agents/agent_a/feature_config.py`)
- `__init___v2.py` - Legacy init file with master extractor exports

### Legacy Configs
- `feature_config.py` - Hybrid51 legacy configuration (superseded by `feature_config_v2.py`)

## Migration Date
April 2, 2026

## DO NOT USE THESE FILES
These files are kept for reference only. All new development should use the per-agent architecture in `hybrid55_preprocessing/agents/`.

## Replacement Mapping

| Old File | New Replacement |
|----------|----------------|
| `master_extractor_v2.py` | `agents/agent_*/extractor.py` (7 separate extractors) |
| `training_pipeline.py` | Use agent-specific training scripts in `scripts/stage1/` |
| Root-level extractors | `extractors/` (shared) + `agents/agent_*/extractor.py` (assembly) |
| `extract_agent_a_features.py` | `agents/agent_a/extractor.py` |
| `feature_config_agent_a.py` | `agents/agent_a/feature_config.py` |
| `__init___v2.py` | `__init__.py` (Version 0.3.0) |
| `feature_config.py` | `feature_config_v2.py` |

## Scripts That Need Updating
Scripts that referenced the old master extractor pattern:
- `scripts/phase0/build_tier2.py` - References `hybrid51_preprocessing.master_extractor_v2`

These will be updated to use per-agent extractors or the shared extractors from `extractors/`.
