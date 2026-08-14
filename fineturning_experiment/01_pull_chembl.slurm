#!/bin/bash
#SBATCH --job-name=chembl_pull
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200          # H200 node — also fine for CPU-only work
#SBATCH --output=logs/01_pull_chembl_%j.log
#
# Step 1: pull and curate ChEMBL affinity data for DRD4 (CHEMBL219) and
# 5HT2A (CHEMBL224). CPU-only.  Output: $CHEMBL_CSV.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

mkdir -p "${DATA_DIR}" "${LOG_DIR}"
cd "${DATA_DIR}"

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Force conda env libstdc++ to win over /lib64 (cuda module pollutes LD_LIBRARY_PATH).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date)] Pulling ChEMBL data → ${CHEMBL_CSV}"

python "${PROJECT_ROOT}/scripts/pull_chembl_affinity_data.py" \
    --cache \
    --output "${CHEMBL_CSV}"

echo "[$(date)] Done. Rows: $(wc -l < "${CHEMBL_CSV}")"
