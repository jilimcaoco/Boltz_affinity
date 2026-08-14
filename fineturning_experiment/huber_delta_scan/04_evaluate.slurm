#!/bin/bash
#SBATCH --job-name=delta_evaluate
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --output=logs/04_evaluate_%j.log
#
# Step 4 (delta scan): compute per-delta ranking / regression metrics using
# the shared evaluate_adapters.py from the main analysis pipeline, then call
# analyse_delta_scan.py to aggregate and plot metric vs delta.
#
# Inputs (in SCORES_DIR):
#   {TARGET}_vanilla.csv               — no-LoRA baseline (written by task 0 of step 3)
#   {TARGET}_delta_{DELTA_TAG}.csv     — one per delta × target (5 × 2 = 10 files)
# Outputs:
#   metrics/{delta_tag}/               — per-delta metrics from evaluate_adapters.py
#   metrics/delta_comparison.csv       — aggregated metric vs delta table
#   plots/delta_scan_*.png             — metric vs delta plots

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

ANALYSIS_EVALUATE="${EXP_DIR}/analysis/evaluate_adapters.py"
ANALYSE_SCRIPT="${SCAN_DIR}/analyse_delta_scan.py"

mkdir -p "${METRICS_DIR}" "${PLOTS_DIR}" "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

DELTAS=(1.00 0.75 0.50 0.25 0.00)

# ── Per-delta evaluation via the shared evaluate_adapters.py ──────────────────
# For each delta we build a temporary scores directory containing:
#   {TARGET}_vanilla.csv   (symlink to the shared baseline)
#   {TARGET}_<adapter>.csv (the delta-specific scores)
# evaluate_adapters.py auto-discovers every {TARGET}_*.csv so this is the
# cleanest way to pass exactly vanilla + one delta per invocation.
for DELTA in "${DELTAS[@]}"; do
    DELTA_TAG=$(echo "${DELTA}" | tr '.' 'p')
    TMP_SCORES=$(mktemp -d)
    trap "rm -rf ${TMP_SCORES}" EXIT

    for TGT in DRD4 5HT2A; do
        VANILLA="${SCORES_DIR}/${TGT}_vanilla.csv"
        DELTA_CSV="${SCORES_DIR}/${TGT}_delta_${DELTA_TAG}.csv"
        if [[ ! -f "${VANILLA}" ]]; then
            echo "[$(date)] WARNING: ${VANILLA} missing — skipping delta=${DELTA}."
            continue 2
        fi
        if [[ ! -f "${DELTA_CSV}" ]]; then
            echo "[$(date)] WARNING: ${DELTA_CSV} missing — skipping delta=${DELTA}."
            continue 2
        fi
        ln -sf "${VANILLA}"   "${TMP_SCORES}/${TGT}_vanilla.csv"
        ln -sf "${DELTA_CSV}" "${TMP_SCORES}/${TGT}_delta_${DELTA_TAG}.csv"
    done

    echo "[$(date)] Running evaluate_adapters.py for delta=${DELTA}"
    python "${ANALYSIS_EVALUATE}" \
        --scores-dir  "${TMP_SCORES}" \
        --labels-dir  "${VAL_LABELS_DIR}" \
        --metrics-dir "${METRICS_DIR}" \
        --plots-dir   "${PLOTS_DIR}/delta_${DELTA_TAG}" \
        --run-id      "delta_${DELTA_TAG}" \
        --targets     DRD4 5HT2A \
        --n-boot      500

    rm -rf "${TMP_SCORES}"
    trap - EXIT
done

# ── Aggregate and plot metric vs delta ────────────────────────────────────────
echo "[$(date)] Running analyse_delta_scan.py"
python "${ANALYSE_SCRIPT}" \
    --metrics-dir "${METRICS_DIR}" \
    --plots-dir   "${PLOTS_DIR}" \
    --output-csv  "${METRICS_DIR}/delta_comparison.csv"

echo ""
echo "[$(date)] Delta comparison table:"
column -s, -t < "${METRICS_DIR}/delta_comparison.csv" | head -40 || true
echo ""
echo "[$(date)] Plots written to ${PLOTS_DIR}/"
ls -lh "${PLOTS_DIR}"/*.png 2>/dev/null || true
