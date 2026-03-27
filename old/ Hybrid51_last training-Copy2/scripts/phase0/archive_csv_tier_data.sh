#!/usr/bin/env bash
# Move Tier1–Tier3 outputs from the CSV (non-DuckDB) pipeline into a dated folder
# so new DuckDB-derived data can use clean names without mix-ups.
#
# Does NOT move:
#   - data_in_2026/          (DuckDB files)
#   - tier3_binary_v5/       (5-year baseline — keep separate)
#
# Usage:
#   bash archive_csv_tier_data.sh
#   ARCHIVE_ROOT=/other/path bash archive_csv_tier_data.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${DATA_ROOT}/archive_csv_pipeline_${STAMP}}"

DIRS_TO_ARCHIVE=(
  "tier1_2026_v1"
  "tier2_minutes_2026_v1"
  "tier2_minutes_v4"
  "tier3_binary_2026_v1"
  "tier3_binary_2026_v2_aligned"
  "tier3_vix_v4"
)

mkdir -p "${ARCHIVE_ROOT}"

echo "Archive destination: ${ARCHIVE_ROOT}"
echo

for d in "${DIRS_TO_ARCHIVE[@]}"; do
  src="${DATA_ROOT}/${d}"
  if [[ -d "${src}" ]]; then
    dest="${ARCHIVE_ROOT}/${d}"
    if [[ -e "${dest}" ]]; then
      echo "SKIP (already exists): ${dest}"
      continue
    fi
    echo "mv ${src} -> ${dest}"
    mv "${src}" "${dest}"
  else
    echo "SKIP (not found): ${src}"
  fi
done

echo
echo "Done. DuckDB stays at: ${DATA_ROOT}/data_in_2026/"
echo "5-year Tier3 baseline unchanged: ${DATA_ROOT}/tier3_binary_v5/"
echo
echo "After rebuilding from DuckDB, point HYBRID51_DATA_ROOT (or config) at the NEW tier3 folder name you create."
