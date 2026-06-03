#!/bin/bash
#SBATCH --job-name=val_evaluate
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --output=../logs/analysis_04_evaluate_%j.log
#
# Step 4 (analysis): compute ranking / retrieval / regression metrics and
# produce plots comparing vanilla Boltz-2 vs LoRA adapter on validation.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ../config.env

ANALYSIS_DIR="${EXP_DIR}/analysis"
SCORES_DIR="${ANALYSIS_DIR}/scores"
LABELS_DIR="${ANALYSIS_DIR}/labels"
METRICS_DIR="${ANALYSIS_DIR}/metrics"
PLOTS_DIR="${ANALYSIS_DIR}/plots"

mkdir -p "${METRICS_DIR}" "${PLOTS_DIR}" "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Force conda env libstdc++ to win over /lib64 (cuda module pollutes LD_LIBRARY_PATH).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python "${ANALYSIS_DIR}/evaluate_adapters.py" \
    --scores-dir  "${SCORES_DIR}" \
    --labels-dir  "${LABELS_DIR}" \
    --metrics-dir "${METRICS_DIR}" \
    --plots-dir   "${PLOTS_DIR}" \
    --targets     DRD4 5HT2A \
    --n-boot      1000

echo "[$(date)] Metrics:"
column -s, -t < "${METRICS_DIR}/metrics_summary.csv" | head -40 || true
echo "[$(date)] Plots:"
ls -lh "${PLOTS_DIR}"
