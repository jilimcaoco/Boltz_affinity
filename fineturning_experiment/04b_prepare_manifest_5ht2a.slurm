#!/bin/bash
#SBATCH --job-name=manifest_5ht2a
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=98G
#SBATCH --time=24:30:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --output=logs/04b_manifest_5ht2a_%j.log
#
# Step 4b: build the 5HT2A-only LoRA training manifest.
# Depends on step 03 (poses for 5HT2A must exist in $HT2A_POSES_DIR).
# Runs in parallel with 04a (DRD4).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

mkdir -p "${MANIFEST_DIR}" "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Force conda env libstdc++ to win over /lib64 (cuda module pollutes LD_LIBRARY_PATH).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date)] Building 5HT2A manifest → ${HT2A_LORA_MANIFEST}"

python "${EXP_DIR}/build_lora_manifest.py" \
    --curated-csv      "${CHEMBL_CSV}" \
    --drd4-receptor    "${DRD4_RECEPTOR_YAML}" \
    --ht2a-receptor    "${HT2A_RECEPTOR_YAML}" \
    --drd4-poses       "${DRD4_POSES_DIR}" \
    --ht2a-poses       "${HT2A_POSES_DIR}" \
    --target           5HT2A \
    --binder-threshold "${LORA_BINDER_THRESHOLD:-1.0}" \
    --out              "${HT2A_LORA_MANIFEST}"

echo "[$(date)] 5HT2A manifest rows: $(tail -n +2 "${HT2A_LORA_MANIFEST}" | wc -l)"
