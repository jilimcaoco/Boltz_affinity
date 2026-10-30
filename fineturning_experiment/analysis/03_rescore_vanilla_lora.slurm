#!/bin/bash
#SBATCH --job-name=val_rescore
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
#SBATCH --output=../logs/analysis_03_rescore_%j.log
#
# Step 3 (analysis): rescore each validation pose with the vanilla affinity
# head, the per-target LoRA adapter, AND the per-target full fine-tune.
# Same poses, three scorers → ranking deltas vs the vanilla baseline are
# attributable purely to the head modification (LoRA residual vs full FT).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ../config.env

ANALYSIS_DIR="${EXP_DIR}/analysis"
VAL_POSES_DIR="${ANALYSIS_DIR}/poses"
SCORES_DIR="${ANALYSIS_DIR}/scores"
LABELS_DIR="${ANALYSIS_DIR}/labels"

mkdir -p "${SCORES_DIR}" "${LOG_DIR}"

module load cuda/12.8.1 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# `module load cuda/...` puts /lib64 ahead of $CONDA_PREFIX/lib on
# LD_LIBRARY_PATH, which makes the system libstdc++ (missing
# GLIBCXX_3.4.29) shadow the newer one shipped with the env and breaks
# `import numpy`.  Force the env's libs to win.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export BOLTZ_CACHE BOLTZ_LORA_DIR BOLTZ_FINETUNE_DIR BOLTZ_MSA_CACHE_DIR

nvidia-smi || true

# Ligand SMILES from the labels (columns name,smiles; name == pose stem).
# Without them rescoring infers each ligand from pose coordinates, which fails
# outright for many poses and feeds an inferred molecular weight into the MW
# correction. Populates the SMILES_ARGS array for the given target.
_smiles_args() {
    local tgt="$1" labels="${LABELS_DIR}/labels_$1.csv"
    SMILES_ARGS=()
    if [[ -s "${labels}" ]]; then
        SMILES_ARGS=(--smiles-csv "${labels}")
    else
        echo "[$(date)] WARNING: ${labels} missing; ${tgt} ligand SMILES will be inferred from coordinates."
    fi
}

# Returns 0 (proceed) when the output does not exist or FORCE_RESCORE=true.
_should_rescore() {
    local out="$1" tag="$2"
    if [[ -f "${out}" && "${FORCE_RESCORE:-false}" != "true" ]]; then
        echo "[$(date)] SKIP ${tag}: output already exists: ${out}"
        echo "  (set FORCE_RESCORE=true or pass --force to run_pipeline.sh to re-run)"
        return 1
    fi
    return 0
}

rescore_vanilla() {
    local tgt="$1"
    local in_dir="${VAL_POSES_DIR}/${tgt}"
    if [[ ! -d "${in_dir}" ]] || [[ -z "$(ls -A "${in_dir}" 2>/dev/null)" ]]; then
        echo "[$(date)] WARNING: no poses for ${tgt} at ${in_dir}; skipping."
        return
    fi
    if _should_rescore "${SCORES_DIR}/${tgt}_vanilla.csv" "${tgt} VANILLA"; then
        echo "[$(date)] rescore ${tgt} VANILLA"
        _smiles_args "${tgt}"
        boltz rescore batch \
            --input-dir   "${in_dir}" \
            --output      "${SCORES_DIR}/${tgt}_vanilla.csv" \
            --output-format csv \
            --device      cuda \
            --msa-directory "${MSA_DIR}" \
            "${SMILES_ARGS[@]}" \
            --log-level   INFO
    fi
}

rescore_lora() {
    local tgt="$1" adapter="$2"
    local in_dir="${VAL_POSES_DIR}/${tgt}"
    if [[ ! -d "${in_dir}" ]] || [[ -z "$(ls -A "${in_dir}" 2>/dev/null)" ]]; then
        return
    fi
    if _should_rescore "${SCORES_DIR}/${tgt}_lora_${adapter}.csv" "${tgt} LoRA=${adapter}"; then
        echo "[$(date)] rescore ${tgt} LoRA=${adapter}"
        _smiles_args "${tgt}"
        boltz rescore batch \
            --input-dir   "${in_dir}" \
            --output      "${SCORES_DIR}/${tgt}_lora_${adapter}.csv" \
            --output-format csv \
            --device      cuda \
            --msa-directory "${MSA_DIR}" \
            "${SMILES_ARGS[@]}" \
            --use-lora    "${adapter}" \
            --log-level   INFO
    fi
}

rescore_finetune() {
    local tgt="$1" adapter="$2"
    local in_dir="${VAL_POSES_DIR}/${tgt}"
    if [[ ! -d "${in_dir}" ]] || [[ -z "$(ls -A "${in_dir}" 2>/dev/null)" ]]; then
        return
    fi
    if _should_rescore "${SCORES_DIR}/${tgt}_finetune_${adapter}.csv" "${tgt} FINETUNE=${adapter}"; then
        echo "[$(date)] rescore ${tgt} FINETUNE=${adapter}"
        _smiles_args "${tgt}"
        boltz rescore batch \
            --input-dir   "${in_dir}" \
            --output      "${SCORES_DIR}/${tgt}_finetune_${adapter}.csv" \
            --output-format csv \
            --device      cuda \
            --msa-directory "${MSA_DIR}" \
            "${SMILES_ARGS[@]}" \
            --use-finetune "${adapter}" \
            --log-level   INFO
    fi
}

# Score vanilla baseline once per target
rescore_vanilla DRD4
rescore_vanilla 5HT2A

# Score all LoRA adapters for each target
for lora in ${DRD4_LORA_NAMES}; do
    rescore_lora DRD4 "${lora}"
done
for lora in ${HT2A_LORA_NAMES}; do
    rescore_lora 5HT2A "${lora}"
done

# Score all finetune adapters (if defined)
if [[ -n "${DRD4_FINETUNE_NAMES:-}" ]]; then
    for ft in ${DRD4_FINETUNE_NAMES}; do
        rescore_finetune DRD4 "${ft}"
    done
fi
if [[ -n "${HT2A_FINETUNE_NAMES:-}" ]]; then
    for ft in ${HT2A_FINETUNE_NAMES}; do
        rescore_finetune 5HT2A "${ft}"
    done
fi

echo "[$(date)] Done.  Score files:"
ls -lh "${SCORES_DIR}"
