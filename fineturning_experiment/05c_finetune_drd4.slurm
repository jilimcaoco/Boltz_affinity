#!/bin/bash
#SBATCH --job-name=ft_drd4
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
#SBATCH --output=logs/05c_finetune_drd4_%j.log
#
# Step 5c: train a DRD4-specific FULL fine-tune of the affinity module.
# Acts as the non-LoRA control for the LoRA adapter from 05a.
# Same manifest, same loss, same target-spec — only the parameterisation
# (full weights vs low-rank residual) differs.
# Depends on 04a (DRD4 manifest). Runs in parallel with 05a/05b/05d.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

mkdir -p "${FINETUNES_DIR}" "${LOG_DIR}"

# ── Skip guard ──────────────────────────────────────────────────────────────
_SENTINEL="${BOLTZ_FINETUNE_DIR}/${DRD4_FINETUNE_NAME}/meta.json"
if [[ -f "${_SENTINEL}" && "${FORCE_RETRAIN:-false}" != "true" ]]; then
    echo "[$(date)] Full finetune '${DRD4_FINETUNE_NAME}' already exists; skipping."
    echo "  sentinel  : ${_SENTINEL}"
    echo "  To retrain: pass --force to run_pipeline.sh (or set FORCE_RETRAIN=true before sbatch)."
    boltz finetune show "${DRD4_FINETUNE_NAME}" || true
    exit 0
fi
_OVR=(); [[ "${FORCE_RETRAIN:-false}" == "true" ]] && _OVR=("--overwrite")

module load cuda/12.8.1 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Force conda env libstdc++ to win over /lib64 (cuda module pollutes LD_LIBRARY_PATH).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export BOLTZ_CACHE BOLTZ_FINETUNE_DIR BOLTZ_MSA_CACHE_DIR

nvidia-smi || true
echo "[$(date)] Starting DRD4 full fine-tune: ${DRD4_FINETUNE_NAME}"
echo "  manifest    : ${DRD4_LORA_MANIFEST}"
echo "  target_spec : ${FINETUNE_TARGET_SPEC}"
echo "  epochs      : ${FINETUNE_EPOCHS}"
echo "  batch_size  : ${FINETUNE_BATCH_SIZE}"
echo "  lr          : ${FINETUNE_LR}"

boltz finetune train \
    "${_OVR[@]}" \
    --name          "${DRD4_FINETUNE_NAME}" \
    --csv           "${DRD4_LORA_MANIFEST}" \
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
    --notes         "DRD4 ChEMBL full FT control vs LoRA, $(date +%Y-%m-%d)"

echo "[$(date)] Done."
boltz finetune show "${DRD4_FINETUNE_NAME}" || true
