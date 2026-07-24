#!/bin/bash
#SBATCH --job-name=delta_train_drd4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --array=0-4
#SBATCH --output=logs/01_train_drd4_%A_%a.log
#
# Step 1 (delta scan): train one DRD4 LoRA adapter per delta value.
# Array tasks 0-4 map to delta = 1.00, 0.75, 0.50, 0.25, 0.00.
# All five run in parallel (no inter-task dependency).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

# ── Resolve delta for this array task ─────────────────────────────────────────
DELTAS=(1.00 0.75 0.50 0.25 0.00)
DELTA="${DELTAS[$SLURM_ARRAY_TASK_ID]}"
# Replace '.' with 'p' for use in names/paths: 0.75 → 0p75
DELTA_TAG=$(echo "${DELTA}" | tr '.' 'p')
# Corresponding loss function in the parametric losses file
LOSS_SPEC="${SCAN_LOSSES_FILE}:boltz2_affinity_delta_${DELTA_TAG}"

export BOLTZ_LORA_DELTA="${DELTA}"  # for provenance in adapter notes

ADAPTER_NAME="drd4_delta_scan_${DELTA_TAG}"

mkdir -p "${ADAPTERS_DIR}" "${LOG_DIR}"

# ── Skip guard ────────────────────────────────────────────────────────────────
_SENTINEL="${BOLTZ_LORA_DIR}/${ADAPTER_NAME}/meta.json"
if [[ -f "${_SENTINEL}" && "${FORCE_RETRAIN:-false}" != "true" ]]; then
    echo "[$(date)] Adapter '${ADAPTER_NAME}' already exists; skipping."
    echo "  sentinel  : ${_SENTINEL}"
    echo "  To retrain: pass --force to run_scan.sh (or set FORCE_RETRAIN=true)."
    boltz lora show "${ADAPTER_NAME}" || true
    exit 0
fi
_OVR=(); [[ "${FORCE_RETRAIN:-false}" == "true" ]] && _OVR=("--overwrite")

module load cuda/12.8.1 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export BOLTZ_CACHE BOLTZ_LORA_DIR BOLTZ_MSA_CACHE_DIR BOLTZ_LORA_BINDER_THRESHOLD

nvidia-smi || true
echo "[$(date)] DRD4 delta scan: delta=${DELTA}  adapter=${ADAPTER_NAME}"
echo "  loss_spec  : ${LOSS_SPEC}"
echo "  manifest   : ${DRD4_LORA_MANIFEST}"
echo "  rank/alpha : ${LORA_RANK}/${LORA_ALPHA}"
echo "  epochs     : ${LORA_EPOCHS}"
echo "  batch_size : ${LORA_BATCH_SIZE}"

boltz lora train \
    "${_OVR[@]}" \
    --name          "${ADAPTER_NAME}" \
    --csv           "${DRD4_LORA_MANIFEST}" \
    --loss          "${LOSS_SPEC}" \
    --rank          "${LORA_RANK}" \
    --alpha         "${LORA_ALPHA}" \
    --epochs        "${LORA_EPOCHS}" \
    --learning-rate "${LORA_LR}" \
    --batch-size    "${LORA_BATCH_SIZE}" \
    --target-spec   "${LORA_TARGET_SPEC}" \
    --early-stopping-patience "${LORA_EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-delta "${LORA_EARLY_STOPPING_MIN_DELTA}" \
    --device        auto \
    --notes         "DRD4 delta scan delta=${DELTA}, boltz2_affinity_loss, $(date +%Y-%m-%d)"

echo "[$(date)] Done."
boltz lora show "${ADAPTER_NAME}" || true
