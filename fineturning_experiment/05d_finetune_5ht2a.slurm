#!/bin/bash
#SBATCH --job-name=ft_5ht2a
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
#SBATCH --output=logs/05d_finetune_5ht2a_%j.log
#
# Step 5d: train a 5HT2A-specific FULL fine-tune of the affinity module.
# Acts as the non-LoRA control for the LoRA adapter from 05b.
# Depends on 04b (5HT2A manifest). Runs in parallel with 05a/05b/05c.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

mkdir -p "${FINETUNES_DIR}" "${LOG_DIR}"

# ── Skip guard ──────────────────────────────────────────────────────────────
_SENTINEL="${BOLTZ_FINETUNE_DIR}/${HT2A_FINETUNE_NAME}/meta.json"
if [[ -f "${_SENTINEL}" && "${FORCE_RETRAIN:-false}" != "true" ]]; then
    echo "[$(date)] Full finetune '${HT2A_FINETUNE_NAME}' already exists; skipping."
    echo "  sentinel  : ${_SENTINEL}"
    echo "  To retrain: pass --force to run_pipeline.sh (or set FORCE_RETRAIN=true before sbatch)."
    boltz finetune show "${HT2A_FINETUNE_NAME}" || true
    exit 0
fi
_OVR=(); [[ "${FORCE_RETRAIN:-false}" == "true" ]] && _OVR=("--overwrite")

module load cuda/12.8.1 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export BOLTZ_CACHE BOLTZ_FINETUNE_DIR BOLTZ_MSA_CACHE_DIR

nvidia-smi || true
echo "[$(date)] Starting 5HT2A full fine-tune: ${HT2A_FINETUNE_NAME}"
echo "  manifest    : ${HT2A_LORA_MANIFEST}"
echo "  target_spec : ${FINETUNE_TARGET_SPEC}"
echo "  epochs      : ${FINETUNE_EPOCHS}"
echo "  batch_size  : ${FINETUNE_BATCH_SIZE}"
echo "  lr          : ${FINETUNE_LR}"

boltz finetune train \
    "${_OVR[@]}" \
    --name          "${HT2A_FINETUNE_NAME}" \
    --csv           "${HT2A_LORA_MANIFEST}" \
    --loss          "${FINETUNE_LOSS}" \
    --target-spec   "${FINETUNE_TARGET_SPEC}" \
    --epochs        "${FINETUNE_EPOCHS}" \
    --learning-rate "${FINETUNE_LR}" \
    --batch-size    "${FINETUNE_BATCH_SIZE}" \
    --weight-decay  "${FINETUNE_WEIGHT_DECAY}" \
    --gradient-clip "${FINETUNE_GRADIENT_CLIP}" \
    --early-stopping-patience  "${FINETUNE_EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-delta "${FINETUNE_EARLY_STOPPING_MIN_DELTA}" \
    --device        auto \
    --notes         "5HT2A ChEMBL full FT control vs LoRA, $(date +%Y-%m-%d)"

echo "[$(date)] Done."
boltz finetune show "${HT2A_FINETUNE_NAME}" || true
