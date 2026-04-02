#!/usr/bin/env python3
"""
Analyze agent flatline behavior from prediction.csv.

Outputs:
- suppression ratio and reasons
- per-agent std/span (all + unsuppressed)
- winner frequency per row
- surpass matrix: how often row-agent > col-agent
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


AGENT_COLS = [
    "agent_A_prob",
    "agent_B_prob",
    "agent_C_prob",
    "agent_K_prob",
    "agent_T_prob",
    "agent_Q_prob",
    "agent_2D_prob",
]


def _to_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[col].astype(str).str.lower().isin(["1", "true", "t", "yes"])


def _fmt_dict(d: Dict[str, float], order: List[str]) -> str:
    lines = []
    for k in order:
        v = d.get(k, 0.0)
        lines.append(f"  {k}: {v:.6f}")
    return "\n".join(lines)


def analyze(pred_path: Path, window: int) -> str:
    if not pred_path.exists():
        return f"prediction file not found: {pred_path}"

    df = pd.read_csv(pred_path)
    missing = [c for c in AGENT_COLS if c not in df.columns]
    if missing:
        return f"missing required columns: {missing}"

    if window > 0:
        df = df.tail(window).copy()

    suppressed = _to_bool_series(df, "suppressed")
    unsup = df.loc[~suppressed].copy()

    out: List[str] = []
    out.append(f"rows={len(df)} unsuppressed={len(unsup)} suppressed={int(suppressed.sum())}")

    if "reason" in df.columns:
        rs = df.loc[suppressed, "reason"].fillna("").value_counts()
        out.append("suppression_reasons_top:")
        if rs.empty:
            out.append("  (none)")
        else:
            for k, v in rs.head(8).items():
                out.append(f"  {k}: {v}")

    all_std = {c: float(df[c].std()) for c in AGENT_COLS}
    all_span = {c: float(df[c].max() - df[c].min()) for c in AGENT_COLS}
    out.append("std_all:")
    out.append(_fmt_dict(all_std, AGENT_COLS))
    out.append("span_all:")
    out.append(_fmt_dict(all_span, AGENT_COLS))

    if len(unsup) > 0:
        unsup_std = {c: float(unsup[c].std()) for c in AGENT_COLS}
        unsup_span = {c: float(unsup[c].max() - unsup[c].min()) for c in AGENT_COLS}
        out.append("std_unsuppressed:")
        out.append(_fmt_dict(unsup_std, AGENT_COLS))
        out.append("span_unsuppressed:")
        out.append(_fmt_dict(unsup_span, AGENT_COLS))

        winners = unsup[AGENT_COLS].idxmax(axis=1).value_counts()
        out.append("winner_frequency_unsuppressed:")
        for k, v in winners.items():
            out.append(f"  {k}: {int(v)}")

        out.append("surpass_matrix_unsuppressed (row > col counts):")
        for a in AGENT_COLS:
            parts = []
            for b in AGENT_COLS:
                if a == b:
                    continue
                parts.append(f"{b}:{int((unsup[a] > unsup[b]).sum())}")
            out.append(f"  {a} | " + ", ".join(parts))
    else:
        out.append("no unsuppressed rows in selected window")

    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze agent flatline behavior from prediction.csv")
    p.add_argument(
        "--prediction-csv",
        default="/workspace/Final_production_model/daily_data/prediction.csv",
        help="Path to prediction.csv",
    )
    p.add_argument("--window", type=int, default=300, help="Use last N rows (0=all rows)")
    args = p.parse_args()

    result = analyze(Path(args.prediction_csv), args.window)
    print(result)


if __name__ == "__main__":
    main()
