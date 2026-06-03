#!/bin/bash
#SBATCH --job-name=val_predict
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --output=../logs/analysis_02_predict_%j.log
#
# Step 2 (analysis): run vanilla `boltz predict` on the validation YAMLs to
# generate per-ligand complex PDBs that BOTH the vanilla and the LoRA
# rescoring will operate on (apples-to-apples comparison).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ../config.env

ANALYSIS_DIR="${EXP_DIR}/analysis"
VAL_INPUT_DIR="${ANALYSIS_DIR}/poses_inputs"
VAL_OUTPUT_DIR="/scratch/maom_root/maom/limcaoco/boltz_val_outputs"
VAL_POSES_DIR="${ANALYSIS_DIR}/poses"

mkdir -p "${VAL_OUTPUT_DIR}/DRD4"  "${VAL_OUTPUT_DIR}/5HT2A" \
         "${VAL_POSES_DIR}/DRD4"   "${VAL_POSES_DIR}/5HT2A" \
         "${LOG_DIR}"

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

for TGT in DRD4 5HT2A; do
    RECORDS_DIR="${VAL_OUTPUT_DIR}/${TGT}/boltz_results_${TGT}/processed/records"
    if [[ -d "${RECORDS_DIR}" ]]; then
        find "${RECORDS_DIR}" -name '*.json' -size 0 -delete
    fi
    echo "[$(date)] boltz predict ${TGT} (vanilla, validation set)"
    boltz predict "${VAL_INPUT_DIR}/${TGT}" \
        --out_dir            "${VAL_OUTPUT_DIR}/${TGT}" \
        --cache              "${BOLTZ_CACHE}" \
        --accelerator        gpu \
        --devices            1 \
        --recycling_steps    "${PREDICT_RECYCLING_STEPS}" \
        --sampling_steps     "${PREDICT_SAMPLING_STEPS}" \
        --diffusion_samples  "${PREDICT_DIFFUSION_SAMPLES}" \
        --output_format      "${PREDICT_OUTPUT_FORMAT}"
done

# Flatten predictions → analysis/poses/{TARGET}/{name}.pdb
echo "[$(date)] Collecting validation poses"
python "${EXP_DIR}/collect_predicted_poses.py" \
    --boltz-out "${VAL_OUTPUT_DIR}/DRD4"  --poses-dir "${VAL_POSES_DIR}/DRD4"
python "${EXP_DIR}/collect_predicted_poses.py" \
    --boltz-out "${VAL_OUTPUT_DIR}/5HT2A" --poses-dir "${VAL_POSES_DIR}/5HT2A"

echo "[$(date)] DRD4  poses: $(ls -1 "${VAL_POSES_DIR}/DRD4"  | wc -l)"
echo "[$(date)] 5HT2A poses: $(ls -1 "${VAL_POSES_DIR}/5HT2A" | wc -l)"
