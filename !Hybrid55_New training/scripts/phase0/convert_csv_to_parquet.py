#!/usr/bin/env python3
"""
Phase 0 Step 0: Convert CSV files to Parquet format for DuckDB ingestion.

Scans Options_greek/ and Options_trade_quote/ for .csv files,
converts them to .parquet in-place (same directory), then optionally
deletes the original CSV.

Usage:
    python scripts/phase0/convert_csv_to_parquet.py
    python scripts/phase0/convert_csv_to_parquet.py --delete-csv   # delete CSVs after conversion
    python scripts/phase0/convert_csv_to_parquet.py --dry-run      # just count, don't convert
"""

import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb


DATA_ROOT = Path('/workspace/data/data_in_2026')
GREEK_DIR = DATA_ROOT / 'Options_greek'
TQ_DIR = DATA_ROOT / 'Options_trade_quote'

# Skip non-data CSVs
SKIP_FILES = {'completed_dates.csv', 'theta_progress.csv'}


def find_csv_files(base_dir: Path):
    """Find all CSV files recursively, excluding metadata files."""
    csv_files = []
    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if f.endswith('.csv') and f not in SKIP_FILES:
                csv_files.append(Path(root) / f)
    return csv_files


def convert_single_csv(csv_path: str) -> dict:
    """Convert a single CSV to parquet using DuckDB (fast columnar conversion)."""
    csv_path = Path(csv_path)
    parquet_path = csv_path.with_suffix('.parquet')

    if parquet_path.exists():
        return {'path': str(csv_path), 'status': 'skipped', 'reason': 'parquet exists'}

    try:
        t0 = time.time()
        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv_auto('{csv_path}', header=true, sample_size=10000)
            ) TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        con.close()
        elapsed = time.time() - t0
        size_mb = parquet_path.stat().st_size / (1024 * 1024)
        return {
            'path': str(csv_path),
            'status': 'converted',
            'elapsed': round(elapsed, 1),
            'size_mb': round(size_mb, 1),
        }
    except Exception as e:
        return {'path': str(csv_path), 'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Convert CSV files to Parquet')
    parser.add_argument('--delete-csv', action='store_true', help='Delete CSV after successful conversion')
    parser.add_argument('--dry-run', action='store_true', help='Count files only, do not convert')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    # Find all CSV files
    greek_csvs = find_csv_files(GREEK_DIR)
    tq_csvs = find_csv_files(TQ_DIR)
    all_csvs = greek_csvs + tq_csvs

    print(f"Found CSV files:")
    print(f"  Greek: {len(greek_csvs)}")
    print(f"  TQ:    {len(tq_csvs)}")
    print(f"  Total: {len(all_csvs)}")

    if args.dry_run:
        # Show per-symbol breakdown
        from collections import Counter
        symbols = Counter()
        for p in all_csvs:
            sym = p.parent.name
            symbols[sym] += 1
        print("\nPer-symbol breakdown:")
        for sym, count in symbols.most_common():
            print(f"  {sym}: {count}")
        return

    if not all_csvs:
        print("No CSV files to convert.")
        return

    print(f"\nConverting {len(all_csvs)} CSVs with {args.workers} workers...")
    t_start = time.time()

    converted = 0
    skipped = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_single_csv, str(p)): p
            for p in all_csvs
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result['status'] == 'converted':
                converted += 1
                if args.delete_csv:
                    try:
                        Path(result['path']).unlink()
                    except Exception:
                        pass
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                errors += 1
                print(f"  ERROR: {result['path']}: {result.get('error', 'unknown')}")

            if i % 100 == 0 or i == len(all_csvs):
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  Progress: {i}/{len(all_csvs)} ({rate:.0f}/s) "
                      f"converted={converted} skipped={skipped} errors={errors}")

    total_elapsed = time.time() - t_start
    print(f"\nDone in {total_elapsed:.1f}s")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")


if __name__ == '__main__':
    main()
