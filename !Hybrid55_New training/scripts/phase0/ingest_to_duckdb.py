#!/usr/bin/env python3
"""
Phase 0 Step 1: Ingest newly-converted parquet files into DuckDB Part 2.

Re-runs the existing ingest_parquet_optimized.py logic — it already deduplicates
by checking source_file in both Part 1 and Part 2 before inserting.

Usage:
    python scripts/phase0/ingest_to_duckdb.py
"""

import subprocess
import sys


def main():
    """Just delegates to the existing optimized ingestion script."""
    script = '/workspace/ingest_parquet_optimized.py'
    print(f"Running {script} ...")
    result = subprocess.run(
        [sys.executable, script],
        cwd='/workspace',
    )
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
