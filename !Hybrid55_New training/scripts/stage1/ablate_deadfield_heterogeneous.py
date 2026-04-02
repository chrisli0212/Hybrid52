#!/usr/bin/env python3
"""
Quick ablation/report utility for:
1) dead-field exclusion coverage
2) heterogeneous agent mechanism diversity

This is a lightweight structural ablation (no full retraining required).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from hybrid55_preprocessing.feature_config_v2 import (  # noqa: E402
    DEAD_RAW_TIER1_FIELDS,
    FEATURE_SCHEMA_VERSION,
    TOTAL_FEATURES,
    get_feature_names,
)
from config.feature_subsets import AGENT_FEATURE_SUBSETS, get_feature_indices  # noqa: E402
from hybrid55_models.independent_agent import IndependentAgent  # noqa: E402


def _dead_feature_indices_from_names(names: list[str]) -> list[int]:
    tokens = tuple(sorted(DEAD_RAW_TIER1_FIELDS))
    idx = []
    for i, n in enumerate(names):
        low = n.lower()
        if any(t in low for t in tokens):
            idx.append(i)
    return idx


def main() -> None:
    names = get_feature_names()
    if len(names) != TOTAL_FEATURES:
        raise RuntimeError(f"Feature-name length mismatch: {len(names)} vs {TOTAL_FEATURES}")

    dead_named_idx = _dead_feature_indices_from_names(names)
    all_agents = ["A", "B", "C", "K", "TQ", "H", "M", "2D"]

    rows = []
    for ag in all_agents:
        model = IndependentAgent(agent_type=ag, feat_dim=TOTAL_FEATURES, use_feature_subset=True)
        params = model.count_parameters()
        subset = set(get_feature_indices(ag)) if ag in AGENT_FEATURE_SUBSETS else set()
        overlap_dead = len(subset.intersection(dead_named_idx))
        rows.append(
            {
                "agent": ag,
                "subset_dim": int(params["subset_feat_dim"]),
                "use_backbone": bool(params["use_backbone"]),
                "n_params_total": int(params["total"]),
                "n_params_backbone": int(params["backbone"]),
                "dead_named_overlap": int(overlap_dead),
            }
        )

    # quick heterogeneous signal check on synthetic data
    torch.manual_seed(7)
    x = torch.randn(16, 20, TOTAL_FEATURES)
    with torch.no_grad():
        logits_a = IndependentAgent("A", feat_dim=TOTAL_FEATURES, use_feature_subset=True)(x).squeeze(-1).cpu().numpy()
        logits_m = IndependentAgent("M", feat_dim=TOTAL_FEATURES, use_feature_subset=True)(x).squeeze(-1).cpu().numpy()
    corr = float(np.corrcoef(logits_a.reshape(-1), logits_m.reshape(-1))[0, 1])

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "deadfield_heterogeneous_ablation.json"
    md_path = out_dir / "deadfield_heterogeneous_ablation.md"

    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "total_features": TOTAL_FEATURES,
        "dead_raw_tier1_fields": sorted(DEAD_RAW_TIER1_FIELDS),
        "dead_named_feature_indices": dead_named_idx,
        "agent_rows": rows,
        "synthetic_logit_corr_A_vs_M": corr,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    lines = []
    lines.append("# Dead-Field + Heterogeneous Ablation (Structural)")
    lines.append("")
    lines.append(f"- Feature schema: `{FEATURE_SCHEMA_VERSION}`")
    lines.append(f"- Total flat features: `{TOTAL_FEATURES}`")
    lines.append(f"- Dead raw tier1 fields locked: `{sorted(DEAD_RAW_TIER1_FIELDS)}`")
    lines.append(f"- Derived feature names matching dead tokens: `{len(dead_named_idx)}`")
    lines.append("")
    lines.append("## Agent Summary")
    lines.append("")
    lines.append("| Agent | Subset Dim | Backbone | Total Params | Backbone Params | Dead-name overlap |")
    lines.append("|---|---:|:---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['agent']} | {r['subset_dim']} | {str(r['use_backbone'])} | "
            f"{r['n_params_total']} | {r['n_params_backbone']} | {r['dead_named_overlap']} |"
        )
    lines.append("")
    lines.append("## Heterogeneous Mechanism Check")
    lines.append("")
    lines.append(
        f"- Synthetic logit correlation between `Agent A` and `Agent M`: `{corr:.4f}` "
        "(lower usually means more architectural diversity)."
    )
    lines.append("")
    lines.append(f"- JSON details: `{json_path}`")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()

