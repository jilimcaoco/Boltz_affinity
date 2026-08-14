#!/bin/bash
#SBATCH --job-name=manifest_drd4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=98G
#SBATCH --time=24:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --output=logs/04a_manifest_drd4_%j.log
#
# Step 4a: build the DRD4-only LoRA training manifest.
# Depends on step 03 (poses for DRD4 must exist in $DRD4_POSES_DIR).
# Runs in parallel with 04b (5HT2A).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

mkdir -p "${MANIFEST_DIR}" "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Force conda env libstdc++ to win over /lib64 (cuda module pollutes LD_LIBRARY_PATH).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date)] Building DRD4 manifest → ${DRD4_LORA_MANIFEST}"

python "${EXP_DIR}/build_lora_manifest.py" \
    --curated-csv      "${CHEMBL_CSV}" \
    --drd4-receptor    "${DRD4_RECEPTOR_YAML}" \
    --ht2a-receptor    "${HT2A_RECEPTOR_YAML}" \
    --drd4-poses       "${DRD4_POSES_DIR}" \
    --ht2a-poses       "${HT2A_POSES_DIR}" \
    --target           DRD4 \
    --binder-threshold "${LORA_BINDER_THRESHOLD:-1.0}" \
    --out              "${DRD4_LORA_MANIFEST}"

echo "[$(date)] DRD4 manifest rows: $(tail -n +2 "${DRD4_LORA_MANIFEST}" | wc -l)"
