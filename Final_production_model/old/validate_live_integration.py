#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from live_inference_service import LiveHybrid51InferenceService


def load_agg(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "batch_id" in df.columns:
        df["batch_id"] = pd.to_numeric(df["batch_id"], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay validation for live Hybrid51 integration.")
    parser.add_argument("--agg", default="/workspace/daily_data/theta_agg.csv")
    parser.add_argument("--snapshots-dir", default="/workspace/daily_data/snapshots")
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--step", type=int, default=1, help="Process every Nth snapshot.")
    args = parser.parse_args()

    agg_df = load_agg(Path(args.agg))
    snap_dir = Path(args.snapshots_dir)
    snaps = sorted(snap_dir.glob("snapshot_*.csv"))
    if args.step > 1:
        snaps = snaps[:: args.step]
    if args.max_batches > 0:
        snaps = snaps[-args.max_batches :]

    service = LiveHybrid51InferenceService()

    probs = []
    suppressed = 0
    low_quality = 0
    failures = 0
    quality_scores = []

    for s in snaps:
        try:
            snap_df = pd.read_csv(s)
            bid = None
            if "batch_id" in snap_df.columns:
                vv = pd.to_numeric(snap_df["batch_id"], errors="coerce").dropna()
                if not vv.empty:
                    bid = int(vv.max())
            if bid is not None and not agg_df.empty and "batch_id" in agg_df.columns:
                agg_cut = agg_df[agg_df["batch_id"] <= bid].copy()
            else:
                agg_cut = agg_df
            out = service.predict_latest(agg_cut, snap_df)
            q = float((out.get("diagnostics", {}) or {}).get("quality_score", 0.0) or 0.0)
            quality_scores.append(q)
            if out.get("suppressed", False):
                suppressed += 1
            if q < 0.6:
                low_quality += 1
            probs.append(float(out.get("prob", 0.5) or 0.5))
        except Exception:
            failures += 1

    n = max(1, len(probs))
    probs_np = np.asarray(probs, dtype=np.float32) if probs else np.asarray([0.5], dtype=np.float32)
    stability = float(np.mean(np.abs(np.diff(probs_np)))) if probs_np.size > 1 else 0.0
    drift_proxy = float(np.std(probs_np))
    quality_mean = float(np.mean(quality_scores)) if quality_scores else 0.0

    print("Replay Validation Summary")
    print("-" * 32)
    print(f"snapshots_processed: {len(snaps)}")
    print(f"inference_failures: {failures}")
    print(f"suppressed_count: {suppressed} ({suppressed / n:.2%})")
    print(f"low_quality_count: {low_quality} ({low_quality / n:.2%})")
    print(f"prob_mean: {float(np.mean(probs_np)):.4f}")
    print(f"prob_std (drift proxy): {drift_proxy:.4f}")
    print(f"adjacent_prob_delta_mean (stability): {stability:.4f}")
    print(f"quality_score_mean: {quality_mean:.4f}")

    print("\nRecommended gates")
    print("-" * 32)
    print("feature_completeness >= 0.60")
    print("warmup_fraction >= 0.35")
    print("adjacent_prob_delta_mean <= 0.20")
    print("suppressed_rate <= 0.35")


if __name__ == "__main__":
    main()
