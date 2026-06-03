#!/bin/bash
#SBATCH --job-name=train_drd4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --output=logs/05a_train_drd4_%j.log
#
# Step 5a: train a DRD4-specific LoRA adapter.
# Depends on 04a (DRD4 manifest).  Runs in parallel with 05b (5HT2A).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

mkdir -p "${ADAPTERS_DIR}" "${LOG_DIR}"

module load cuda/12.8.1 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Force conda env libstdc++ to win over /lib64 (cuda module pollutes LD_LIBRARY_PATH).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export BOLTZ_CACHE BOLTZ_LORA_DIR BOLTZ_MSA_CACHE_DIR

nvidia-smi || true
echo "[$(date)] Starting DRD4 LoRA training: ${DRD4_LORA_NAME}"
echo "  manifest   : ${DRD4_LORA_MANIFEST}"
echo "  rank/alpha : ${LORA_RANK}/${LORA_ALPHA}"
echo "  epochs     : ${LORA_EPOCHS}"
echo "  batch_size : ${LORA_BATCH_SIZE}"

boltz lora train \
    --name          "${DRD4_LORA_NAME}" \
    --csv           "${DRD4_LORA_MANIFEST}" \
    --loss          "${LORA_LOSS}" \
    --rank          "${LORA_RANK}" \
    --alpha         "${LORA_ALPHA}" \
    --epochs        "${LORA_EPOCHS}" \
    --learning-rate "${LORA_LR}" \
    --batch-size    "${LORA_BATCH_SIZE}" \
    --target-spec   "${LORA_TARGET_SPEC}" \
    --early-stopping-patience "${LORA_EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-delta "${LORA_EARLY_STOPPING_MIN_DELTA}" \
    --device        auto \
    --notes         "DRD4 ChEMBL fine-tuning, intra-assay Huber, $(date +%Y-%m-%d)"

echo "[$(date)] Done."
boltz lora show "${DRD4_LORA_NAME}" || true
