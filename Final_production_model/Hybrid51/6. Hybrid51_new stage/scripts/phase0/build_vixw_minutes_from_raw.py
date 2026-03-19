#!/usr/bin/env python3
"""
Build /workspace/data/tier2_minutes_v4/VIXW_minutes.parquet from raw VIXW parquet files.

Sources:
  - /workspace/data/data_in_2026/Options_greek/VIXW/*.parquet
  - /workspace/data/data_in_2026/Options_trade_quote/VIXW/*.parquet

This is a pragmatic bootstrap for Agent VIX feature building.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_GREEK_DIR = Path("/workspace/data/data_in_2026/Options_greek/VIXW")
DEFAULT_TQ_DIR = Path("/workspace/data/data_in_2026/Options_trade_quote/VIXW")
DEFAULT_OUT = Path("/workspace/data/tier2_minutes_v4/VIXW_minutes.parquet")


def _aggregate_greek_file(path: Path) -> pd.DataFrame:
    cols = ["underlying_timestamp", "underlying_price"]
    df = pd.read_parquet(path, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "greek_count"])

    df = df.dropna(subset=["underlying_timestamp", "underlying_price"])
    if df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "greek_count"])

    df["datetime"] = pd.to_datetime(df["underlying_timestamp"], errors="coerce").dt.floor("min")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "greek_count"])

    agg = (
        df.groupby("datetime", sort=False)["underlying_price"]
        .agg(open="first", high="max", low="min", close="last", greek_count="size")
        .reset_index()
    )
    return agg


def _aggregate_tq_file(path: Path) -> pd.DataFrame:
    cols = ["trade_timestamp", "size"]
    df = pd.read_parquet(path, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=["datetime", "trade_count", "volume"])

    df = df.dropna(subset=["trade_timestamp"])
    if df.empty:
        return pd.DataFrame(columns=["datetime", "trade_count", "volume"])

    df["datetime"] = pd.to_datetime(df["trade_timestamp"], errors="coerce").dt.floor("min")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return pd.DataFrame(columns=["datetime", "trade_count", "volume"])

    if "size" not in df.columns:
        df["size"] = 0.0
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)

    agg = (
        df.groupby("datetime", sort=False)
        .agg(trade_count=("size", "size"), volume=("size", "sum"))
        .reset_index()
    )
    return agg


def build_vixw_minutes(greek_dir: Path, tq_dir: Path, output_path: Path) -> Path:
    greek_files = sorted(greek_dir.glob("*.parquet"))
    tq_files = sorted(tq_dir.glob("*.parquet"))

    if not greek_files:
        raise FileNotFoundError(f"No Greek parquet files found in {greek_dir}")

    logger.info("Greek files: %d", len(greek_files))
    logger.info("Trade/quote files: %d", len(tq_files))

    greek_parts = []
    for i, fp in enumerate(greek_files, 1):
        part = _aggregate_greek_file(fp)
        if not part.empty:
            greek_parts.append(part)
        if i % 10 == 0 or i == len(greek_files):
            logger.info("Processed Greek %d/%d", i, len(greek_files))

    greek_df = pd.concat(greek_parts, ignore_index=True)
    greek_df = (
        greek_df.groupby("datetime", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            greek_count=("greek_count", "sum"),
        )
        .sort_values("datetime")
    )

    if tq_files:
        tq_parts = []
        for i, fp in enumerate(tq_files, 1):
            part = _aggregate_tq_file(fp)
            if not part.empty:
                tq_parts.append(part)
            if i % 50 == 0 or i == len(tq_files):
                logger.info("Processed TQ %d/%d", i, len(tq_files))

        if tq_parts:
            tq_df = pd.concat(tq_parts, ignore_index=True)
            tq_df = (
                tq_df.groupby("datetime", as_index=False)
                .agg(trade_count=("trade_count", "sum"), volume=("volume", "sum"))
            )
            minutes_df = greek_df.merge(tq_df, how="left", on="datetime")
        else:
            minutes_df = greek_df.copy()
            minutes_df["trade_count"] = 0
            minutes_df["volume"] = 0.0
    else:
        minutes_df = greek_df.copy()
        minutes_df["trade_count"] = 0
        minutes_df["volume"] = 0.0

    minutes_df["trade_count"] = minutes_df["trade_count"].fillna(0).astype(np.int64)
    minutes_df["greek_count"] = minutes_df["greek_count"].fillna(0).astype(np.int64)
    minutes_df["volume"] = minutes_df["volume"].fillna(0.0).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    minutes_df.to_parquet(output_path, index=False)

    logger.info("Saved %d minute bars -> %s", len(minutes_df), output_path)
    logger.info(
        "Date range: %s to %s",
        minutes_df["datetime"].min(),
        minutes_df["datetime"].max(),
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VIXW minute parquet from raw VIXW files")
    parser.add_argument("--greek-dir", default=str(DEFAULT_GREEK_DIR))
    parser.add_argument("--tq-dir", default=str(DEFAULT_TQ_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    build_vixw_minutes(Path(args.greek_dir), Path(args.tq_dir), Path(args.output))
