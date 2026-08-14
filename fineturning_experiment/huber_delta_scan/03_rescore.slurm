#!/bin/bash
#SBATCH --job-name=delta_rescore
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --array=0-4
#SBATCH --output=logs/03_rescore_%A_%a.log
#
# Step 3 (delta scan): rescore validation poses with the delta-specific adapters.
# Each array task handles one delta value across both DRD4 and 5HT2A.
# Validation poses are reused from the main analysis pipeline:
#   fineturning_experiment/analysis/poses/{DRD4,5HT2A}/
#
# Also produces vanilla (no-LoRA) scores once per target (task 0 only;
# subsequent tasks skip if the file already exists) so that 04_evaluate.slurm
# has a common baseline to compare all delta adapters against.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

# ── Resolve delta for this array task ─────────────────────────────────────────
DELTAS=(1.00 0.75 0.50 0.25 0.00)
DELTA="${DELTAS[$SLURM_ARRAY_TASK_ID]}"
DELTA_TAG=$(echo "${DELTA}" | tr '.' 'p')

DRD4_ADAPTER="drd4_delta_scan_${DELTA_TAG}"
HT2A_ADAPTER="5ht2a_delta_scan_${DELTA_TAG}"

mkdir -p "${SCORES_DIR}" "${LOG_DIR}"

module load cuda/12.8.1 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export BOLTZ_CACHE BOLTZ_LORA_DIR BOLTZ_MSA_CACHE_DIR

nvidia-smi || true

# Returns 0 (proceed) when output is absent or FORCE_RESCORE=true.
_should_rescore() {
    local out="$1" tag="$2"
    if [[ -f "${out}" && "${FORCE_RESCORE:-false}" != "true" ]]; then
        echo "[$(date)] SKIP ${tag}: output already exists: ${out}"
        return 1
    fi
    return 0
}

# ── Vanilla baseline: run once, shared by all delta adapters ──────────────────
# Task 0 is responsible for producing the vanilla scores; all other tasks skip.
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    for TGT in DRD4 5HT2A; do
        IN_DIR="${VAL_POSES_DIR}/${TGT}"
        OUT="${SCORES_DIR}/${TGT}_vanilla.csv"
        if [[ ! -d "${IN_DIR}" ]] || [[ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]]; then
            echo "[$(date)] WARNING: no validation poses for ${TGT} at ${IN_DIR}; skipping vanilla."
            continue
        fi
        if _should_rescore "${OUT}" "${TGT} VANILLA"; then
            echo "[$(date)] rescore ${TGT} VANILLA (no LoRA)"
            boltz rescore batch \
                --input-dir     "${IN_DIR}" \
                --output        "${OUT}" \
                --output-format csv \
                --device        cuda \
                --msa-directory "${MSA_DIR}" \
                --log-level     INFO
        fi
    done
fi

# ── Delta-specific adapter rescoring ─────────────────────────────────────────
for TGT_PAIR in "DRD4 ${DRD4_ADAPTER}" "5HT2A ${HT2A_ADAPTER}"; do
    TGT="${TGT_PAIR%% *}"
    ADAPTER="${TGT_PAIR##* }"
    IN_DIR="${VAL_POSES_DIR}/${TGT}"
    OUT="${SCORES_DIR}/${TGT}_delta_${DELTA_TAG}.csv"

    if [[ ! -d "${IN_DIR}" ]] || [[ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]]; then
        echo "[$(date)] WARNING: no validation poses for ${TGT} at ${IN_DIR}; skipping."
        continue
    fi

    _SENTINEL="${BOLTZ_LORA_DIR}/${ADAPTER}/meta.json"
    if [[ ! -f "${_SENTINEL}" ]]; then
        echo "[$(date)] ERROR: adapter '${ADAPTER}' not found at ${_SENTINEL}."
        echo "  Run 01_train_drd4.slurm / 02_train_5ht2a.slurm first."
        exit 1
    fi

    if _should_rescore "${OUT}" "${TGT} delta=${DELTA} adapter=${ADAPTER}"; then
        echo "[$(date)] rescore ${TGT}  delta=${DELTA}  adapter=${ADAPTER}"
        boltz rescore batch \
            --input-dir     "${IN_DIR}" \
            --output        "${OUT}" \
            --output-format csv \
            --device        cuda \
            --msa-directory "${MSA_DIR}" \
            --use-lora      "${ADAPTER}" \
            --log-level     INFO
        echo "[$(date)] Wrote ${OUT}"
    fi
done

echo "[$(date)] Rescore task ${SLURM_ARRAY_TASK_ID} (delta=${DELTA}) complete."
