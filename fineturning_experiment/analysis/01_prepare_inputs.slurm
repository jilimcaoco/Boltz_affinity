#!/bin/bash
#SBATCH --job-name=val_prepare
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --output=../logs/analysis_01_prepare_%j.log
#
# Step 1 (analysis): build per-ligand Boltz prediction YAMLs for the
# held-out in-vitro validation set, plus tidy labels CSVs.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ../config.env

ANALYSIS_DIR="${EXP_DIR}/analysis"
VAL_DIR="${EXP_DIR}/validation_data"
VAL_INPUT_DIR="${ANALYSIS_DIR}/poses_inputs"
LABELS_DIR="${ANALYSIS_DIR}/labels"

mkdir -p "${VAL_INPUT_DIR}/DRD4" "${VAL_INPUT_DIR}/5HT2A" "${LABELS_DIR}" "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Force conda env libstdc++ to win over /lib64 (cuda module pollutes LD_LIBRARY_PATH).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python "${ANALYSIS_DIR}/prepare_validation_inputs.py" \
    --drd4-csv      "${VAL_DIR}/D4_in_vitro_data.csv" \
    --ht2a-csv      "${VAL_DIR}/5ht2a_in_vitro_data.csv" \
    --drd4-receptor "${DRD4_RECEPTOR_YAML}" \
    --ht2a-receptor "${HT2A_RECEPTOR_YAML}" \
    --out-dir       "${VAL_INPUT_DIR}" \
    --labels-dir    "${LABELS_DIR}" \
    --msa-dir       "${MSA_DIR}"

echo "[$(date)] DRD4  YAMLs : $(ls -1 "${VAL_INPUT_DIR}/DRD4"  | wc -l)"
echo "[$(date)] 5HT2A YAMLs : $(ls -1 "${VAL_INPUT_DIR}/5HT2A" | wc -l)"
