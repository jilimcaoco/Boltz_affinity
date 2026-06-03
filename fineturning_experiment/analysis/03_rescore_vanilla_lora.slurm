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
# Step 3 (analysis): rescore each validation pose with BOTH the vanilla
# affinity head and the per-target LoRA adapter. Same poses, two scorers
# → any ranking delta is attributable purely to the LoRA-modified head.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ../config.env

ANALYSIS_DIR="${EXP_DIR}/analysis"
VAL_POSES_DIR="${ANALYSIS_DIR}/poses"
SCORES_DIR="${ANALYSIS_DIR}/scores"

mkdir -p "${SCORES_DIR}" "${LOG_DIR}"

module load cuda/12.8.1 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# `module load cuda/...` puts /lib64 ahead of $CONDA_PREFIX/lib on
# LD_LIBRARY_PATH, which makes the system libstdc++ (missing
# GLIBCXX_3.4.29) shadow the newer one shipped with the env and breaks
# `import numpy`.  Force the env's libs to win.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export BOLTZ_CACHE BOLTZ_LORA_DIR BOLTZ_MSA_CACHE_DIR

nvidia-smi || true

run_pair() {
    local tgt="$1" lora="$2"
    local in_dir="${VAL_POSES_DIR}/${tgt}"
    if [[ ! -d "${in_dir}" ]] || [[ -z "$(ls -A "${in_dir}" 2>/dev/null)" ]]; then
        echo "[$(date)] WARNING: no poses for ${tgt} at ${in_dir}; skipping."
        return
    fi

    echo "[$(date)] rescore ${tgt} VANILLA"
    boltz rescore batch \
        --input-dir   "${in_dir}" \
        --output      "${SCORES_DIR}/${tgt}_vanilla.csv" \
        --output-format csv \
        --device      cuda \
        --msa-directory "${MSA_DIR}" \
        --log-level   INFO

    echo "[$(date)] rescore ${tgt} LoRA=${lora}"
    boltz rescore batch \
        --input-dir   "${in_dir}" \
        --output      "${SCORES_DIR}/${tgt}_lora.csv" \
        --output-format csv \
        --device      cuda \
        --msa-directory "${MSA_DIR}" \
        --use-lora    "${lora}" \
        --log-level   INFO
}

run_pair DRD4   "${DRD4_LORA_NAME}"
run_pair 5HT2A  "${HT2A_LORA_NAME}"

echo "[$(date)] Done.  Score files:"
ls -lh "${SCORES_DIR}"
