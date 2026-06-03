#!/bin/bash
#SBATCH --job-name=lora_msa
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --output=logs/02_precompute_msas_%j.log
#
# Step 2: precompute protein MSAs via ColabFold MMseqs2 API.
#
# There are exactly two protein targets (DRD4, 5HT2A), so this step makes
# only two API calls regardless of how many ligands are in the dataset.
# The resulting .a3m files are embedded into every per-ligand Boltz input
# YAML in step 03, so `boltz predict` never contacts the MSA server again.
#
# If the .a3m files already exist this step is a no-op (precompute_msas.py
# skips chains whose file is present and non-empty).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source ./config.env

mkdir -p "${MSA_DIR}" "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Force conda env libstdc++ to win over /lib64 (cuda module pollutes LD_LIBRARY_PATH).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date)] Precomputing MSAs → ${MSA_DIR}"
echo "  DRD4  receptor: ${DRD4_RECEPTOR_YAML}"
echo "  5HT2A receptor: ${HT2A_RECEPTOR_YAML}"

python "${EXP_DIR}/precompute_msas.py" \
    --drd4-receptor "${DRD4_RECEPTOR_YAML}" \
    --ht2a-receptor "${HT2A_RECEPTOR_YAML}" \
    --msa-dir       "${MSA_DIR}" \
    --host-url      "${MSA_HOST_URL:-https://api.colabfold.com}"

echo "[$(date)] MSA files:"
ls -lh "${MSA_DIR}"/*.a3m 2>/dev/null || echo "  (none written — check receptor YAMLs)"
echo "[$(date)] Done."
