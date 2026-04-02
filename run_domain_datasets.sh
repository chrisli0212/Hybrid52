#!/usr/bin/env bash
# ============================================================
# run_domain_datasets.sh  (v2 — bug-fixed)
# Launcher for build_direct_domain_datasets.py
#
# BUG-9 fix: OUT_ROOT changed from /workspace/data/direct_domain
#            to /workspace/data (as originally requested).
#
# Usage:
#   bash run_domain_datasets.sh smoke      # smoke test SPXW only
#   bash run_domain_datasets.sh spxw       # full SPXW all horizons
#   bash run_domain_datasets.sh spy
#   bash run_domain_datasets.sh qqq
#   bash run_domain_datasets.sh iwm
#   bash run_domain_datasets.sh tlt
#   bash run_domain_datasets.sh vix        # VIXW only
#   bash run_domain_datasets.sh all        # all symbols (no VIXW)
#   bash run_domain_datasets.sh all-vix    # all symbols + VIXW
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/build_direct_domain_datasets.py"
CSV_ROOT="/workspace/historical_data_1yr"
OUT_ROOT="/workspace/data"          # BUG-9 fix: was /workspace/data/direct_domain
HORIZONS="5 15 30"

MODE="${1:-smoke}"

# Guard: ensure the Python script exists
if [[ ! -f "${PY}" ]]; then
    echo "❌ Cannot find ${PY}"
    echo "   Place build_direct_domain_datasets.py in the same directory as this script."
    exit 1
fi

case "${MODE}" in

smoke)
    echo "▶ SMOKE TEST — SPXW only, first 2 CSV files, horizon 15min"
    python3 "${PY}" \
        --symbol SPXW \
        --horizons 15 \
        --smoke \
        --csv-root  "${CSV_ROOT}" \
        --output-root "${OUT_ROOT}"
    ;;

spxw)
    echo "▶ SPXW full run — all horizons"
    python3 "${PY}" \
        --symbol SPXW \
        --horizons ${HORIZONS} \
        --csv-root  "${CSV_ROOT}" \
        --output-root "${OUT_ROOT}"
    ;;

spy)
    echo "▶ SPY full run — all horizons"
    python3 "${PY}" \
        --symbol SPY \
        --horizons ${HORIZONS} \
        --csv-root  "${CSV_ROOT}" \
        --output-root "${OUT_ROOT}"
    ;;

qqq)
    echo "▶ QQQ full run — all horizons"
    python3 "${PY}" \
        --symbol QQQ \
        --horizons ${HORIZONS} \
        --csv-root  "${CSV_ROOT}" \
        --output-root "${OUT_ROOT}"
    ;;

iwm)
    echo "▶ IWM full run — all horizons"
    python3 "${PY}" \
        --symbol IWM \
        --horizons ${HORIZONS} \
        --csv-root  "${CSV_ROOT}" \
        --output-root "${OUT_ROOT}"
    ;;

tlt)
    echo "▶ TLT full run — all horizons"
    python3 "${PY}" \
        --symbol TLT \
        --horizons ${HORIZONS} \
        --csv-root  "${CSV_ROOT}" \
        --output-root "${OUT_ROOT}"
    ;;

vix)
    echo "▶ VIXW only — vix_sequences.npy"
    python3 "${PY}" \
        --symbol VIXW \
        --vixw \
        --horizons ${HORIZONS} \
        --csv-root  "${CSV_ROOT}" \
        --output-root "${OUT_ROOT}"
    ;;

all)
    echo "▶ ALL symbols (SPXW SPY QQQ IWM TLT) — no VIXW"
    python3 "${PY}" \
        --all-symbols \
        --horizons ${HORIZONS} \
        --csv-root  "${CSV_ROOT}" \
        --output-root "${OUT_ROOT}"
    ;;

all-vix)
    echo "▶ ALL symbols + VIXW"
    python3 "${PY}" \
        --all-symbols \
        --vixw \
        --horizons ${HORIZONS} \
        --csv-root  "${CSV_ROOT}" \
        --output-root "${OUT_ROOT}"
    ;;

*)
    echo "Usage: $0 [smoke|spxw|spy|qqq|iwm|tlt|vix|all|all-vix]"
    exit 1
    ;;

esac
