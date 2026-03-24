#!/usr/bin/env python3
"""
Build /workspace/data/tier2_minutes_v4/VIXW_minutes.parquet from raw VIXW files.

Supported inputs:
  1) Parquet directories (legacy):
     - greek_dir/*.parquet (expects underlying_timestamp + underlying_price)
     - tq_dir/*.parquet    (expects trade_timestamp + size)
  2) Weekly CSV archive (2026 1-year flow):
     - greek_dir/VIXW_historical_*.csv (expects timestamp + underlying_price)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_GREEK_DIR = Path("/workspace/historical_data_1yr/VIXW")
DEFAULT_TQ_DIR = Path("/workspace/historical_data_1yr/VIXW/OI")
DEFAULT_OUT = Path("/workspace/data/tier2_minutes_v4/VIXW_minutes.parquet")


def _empty_greek_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "greek_count"])


def _empty_tq_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["datetime", "trade_count", "volume"])


def _aggregate_greek_parquet(path: Path) -> pd.DataFrame:
    cols = ["underlying_timestamp", "underlying_price"]
    df = pd.read_parquet(path, columns=cols)
    if df.empty:
        return _empty_greek_frame()

    df = df.dropna(subset=["underlying_timestamp", "underlying_price"])
    if df.empty:
        return _empty_greek_frame()

    df["datetime"] = pd.to_datetime(df["underlying_timestamp"], errors="coerce").dt.floor("min")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return _empty_greek_frame()

    return (
        df.groupby("datetime", sort=False)["underlying_price"]
        .agg(open="first", high="max", low="min", close="last", greek_count="size")
        .reset_index()
    )


def _aggregate_tq_parquet(path: Path) -> pd.DataFrame:
    cols = ["trade_timestamp", "size"]
    df = pd.read_parquet(path, columns=cols)
    if df.empty:
        return _empty_tq_frame()

    df = df.dropna(subset=["trade_timestamp"])
    if df.empty:
        return _empty_tq_frame()

    df["datetime"] = pd.to_datetime(df["trade_timestamp"], errors="coerce").dt.floor("min")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return _empty_tq_frame()

    df["size"] = pd.to_numeric(df.get("size", 0.0), errors="coerce").fillna(0.0)
    return (
        df.groupby("datetime", sort=False)
        .agg(trade_count=("size", "size"), volume=("size", "sum"))
        .reset_index()
    )


def _aggregate_historical_csv(path: Path) -> pd.DataFrame:
    cols = ["timestamp", "underlying_price", "size"]
    usecols = lambda c: c in cols
    df = pd.read_csv(path, usecols=usecols)
    if df.empty:
        return _empty_greek_frame().assign(trade_count=[], volume=[])

    if "timestamp" not in df.columns:
        raise ValueError(f"CSV missing timestamp column: {path}")
    if "underlying_price" not in df.columns:
        raise ValueError(f"CSV missing underlying_price column: {path}")

    df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.floor("min")
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
    df = df.dropna(subset=["datetime", "underlying_price"])
    if df.empty:
        return _empty_greek_frame().assign(trade_count=[], volume=[])

    if "size" in df.columns:
        size = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)
    else:
        # CSV snapshots are mostly quote/greek snapshots; use row count as weak activity proxy.
        size = pd.Series(np.ones(len(df), dtype=np.float32), index=df.index)
    df["size_proxy"] = size

    return (
        df.groupby("datetime", sort=False)
        .agg(
            open=("underlying_price", "first"),
            high=("underlying_price", "max"),
            low=("underlying_price", "min"),
            close=("underlying_price", "last"),
            greek_count=("underlying_price", "size"),
            trade_count=("size_proxy", "size"),
            volume=("size_proxy", "sum"),
        )
        .reset_index()
    )


def _collapse_greek(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        raise RuntimeError("No usable VIXW source rows found.")
    greek_df = pd.concat(parts, ignore_index=True)
    return (
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


def build_vixw_minutes(greek_dir: Path, tq_dir: Path, output_path: Path) -> Path:
    parquet_greek_files = sorted(greek_dir.glob("*.parquet"))
    csv_greek_files = sorted(greek_dir.glob("VIXW_historical_*.csv"))

    if csv_greek_files:
        source_mode = "historical_csv"
        logger.info("Detected CSV mode with %d weekly files", len(csv_greek_files))
        greek_parts = []
        tq_parts = []
        for i, fp in enumerate(csv_greek_files, 1):
            part = _aggregate_historical_csv(fp)
            if not part.empty:
                greek_parts.append(part[["datetime", "open", "high", "low", "close", "greek_count"]])
                tq_parts.append(part[["datetime", "trade_count", "volume"]])
            if i % 10 == 0 or i == len(csv_greek_files):
                logger.info("Processed CSV %d/%d", i, len(csv_greek_files))
        greek_df = _collapse_greek(greek_parts)

        if tq_parts:
            tq_df = (
                pd.concat(tq_parts, ignore_index=True)
                .groupby("datetime", as_index=False)
                .agg(trade_count=("trade_count", "sum"), volume=("volume", "sum"))
            )
            minutes_df = greek_df.merge(tq_df, how="left", on="datetime")
        else:
            minutes_df = greek_df.copy()
            minutes_df["trade_count"] = 0
            minutes_df["volume"] = 0.0
    elif parquet_greek_files:
        source_mode = "legacy_parquet"
        tq_files = sorted(tq_dir.glob("*.parquet"))
        logger.info("Detected parquet mode with Greek=%d, TQ=%d files", len(parquet_greek_files), len(tq_files))

        greek_parts = []
        for i, fp in enumerate(parquet_greek_files, 1):
            part = _aggregate_greek_parquet(fp)
            if not part.empty:
                greek_parts.append(part)
            if i % 10 == 0 or i == len(parquet_greek_files):
                logger.info("Processed Greek %d/%d", i, len(parquet_greek_files))
        greek_df = _collapse_greek(greek_parts)

        if tq_files:
            tq_parts = []
            for i, fp in enumerate(tq_files, 1):
                part = _aggregate_tq_parquet(fp)
                if not part.empty:
                    tq_parts.append(part)
                if i % 50 == 0 or i == len(tq_files):
                    logger.info("Processed TQ %d/%d", i, len(tq_files))
            if tq_parts:
                tq_df = (
                    pd.concat(tq_parts, ignore_index=True)
                    .groupby("datetime", as_index=False)
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
    else:
        raise FileNotFoundError(
            f"No supported input files found in {greek_dir} "
            "(expected VIXW_historical_*.csv or *.parquet)"
        )

    minutes_df["trade_count"] = minutes_df["trade_count"].fillna(0).astype(np.int64)
    minutes_df["greek_count"] = minutes_df["greek_count"].fillna(0).astype(np.int64)
    minutes_df["volume"] = minutes_df["volume"].fillna(0.0).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    minutes_df.to_parquet(output_path, index=False)

    logger.info("Source mode: %s", source_mode)
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
