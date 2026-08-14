#!/bin/bash
#SBATCH --job-name=lora_predict
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=6-00:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --output=logs/03_predict_poses_%j.log
#
# Step 3: build per-ligand Boltz input YAMLs (with precomputed MSA paths
# embedded), run `boltz predict` for both targets, then flatten the top-rank
# model for each ligand into poses/{target}/{molecule}.pdb.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

mkdir -p "${PREDICT_INPUT_DIR}/DRD4" "${PREDICT_INPUT_DIR}/5HT2A" \
         "${PREDICT_OUTPUT_DIR}/DRD4" "${PREDICT_OUTPUT_DIR}/5HT2A" \
         "${DRD4_POSES_DIR}" "${HT2A_POSES_DIR}" \
         "${LOG_DIR}"

module load cuda/12.8.1 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Force conda env libstdc++ to win over /lib64 (cuda module pollutes LD_LIBRARY_PATH).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export BOLTZ_CACHE BOLTZ_LORA_DIR BOLTZ_MSA_CACHE_DIR

nvidia-smi || true

# ─── 3a. Build per-ligand input YAMLs (MSA paths baked in) ─────────────────
echo "[$(date)] Building Boltz prediction inputs"
python "${EXP_DIR}/prepare_predict_inputs.py" \
    --curated-csv   "${CHEMBL_CSV}" \
    --drd4-receptor "${DRD4_RECEPTOR_YAML}" \
    --ht2a-receptor "${HT2A_RECEPTOR_YAML}" \
    --msa-dir       "${MSA_DIR}" \
    --out-dir       "${PREDICT_INPUT_DIR}"

# ─── 3b. Run Boltz prediction for each target (no MSA server needed) ────────
for TGT in DRD4 5HT2A; do
    # Remove any zero-byte processed-record JSONs left behind by a previous
    # crashed run; boltz's resumption logic json.load()s every *.json under
    # processed/records/ before honoring --override, and an empty file raises
    # JSONDecodeError. Safe no-op when the dir doesn't exist yet.
    RECORDS_DIR="${PREDICT_OUTPUT_DIR}/${TGT}/boltz_results_${TGT}/processed/records"
    if [[ -d "${RECORDS_DIR}" ]]; then
        find "${RECORDS_DIR}" -name '*.json' -size 0 -delete
    fi
    echo "[$(date)] boltz predict ${TGT}"
    boltz predict "${PREDICT_INPUT_DIR}/${TGT}" \
        --out_dir            "${PREDICT_OUTPUT_DIR}/${TGT}" \
        --cache              "${BOLTZ_CACHE}" \
        --accelerator        gpu \
        --devices            1 \
        --recycling_steps    "${PREDICT_RECYCLING_STEPS}" \
        --sampling_steps     "${PREDICT_SAMPLING_STEPS}" \
        --diffusion_samples  "${PREDICT_DIFFUSION_SAMPLES}" \
        --output_format      "${PREDICT_OUTPUT_FORMAT}"
        # NOTE: --override intentionally omitted so Boltz resumes from any
        # completed predictions/<id>/ + affinity_<id>.json on disk. The
        # zero-byte-records scrub above handles crash-mid-write cleanup.
done

# ─── 3c. Flatten predictions into poses/{target}/{molecule}.pdb ─────────────
echo "[$(date)] Collecting predicted poses"
python "${EXP_DIR}/collect_predicted_poses.py" \
    --boltz-out "${PREDICT_OUTPUT_DIR}/DRD4"  \
    --poses-dir "${DRD4_POSES_DIR}"

python "${EXP_DIR}/collect_predicted_poses.py" \
    --boltz-out "${PREDICT_OUTPUT_DIR}/5HT2A" \
    --poses-dir "${HT2A_POSES_DIR}"

echo "[$(date)] DRD4 poses : $(ls -1 "${DRD4_POSES_DIR}" | wc -l)"
echo "[$(date)] 5HT2A poses: $(ls -1 "${HT2A_POSES_DIR}" | wc -l)"
echo "[$(date)] Done."
