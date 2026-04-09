#!/bin/bash
# scripts/interpretability/slurm/01_cache.slurm
# -----------------------------------------------
# SLURM array job: run cache_affinity_inputs.py for one complex per task.
#
# Uses the affinity-rescoring pipeline (trunk + affinity head, NO
# diffusion) with receptor PDB + docked MOL2 coordinate injection.
#
# Submitted by submit_pipeline.sh — do not run directly.
# All paths are passed via --export from the submitter.
#
# Required env vars (set by submit_pipeline.sh):
#   MANIFEST        path to the manifest TSV
#   CACHED_DIR      directory to write .pt files into
#   CHECKPOINT      path to boltz2_aff.ckpt
#   PROJECT_ROOT    absolute path to the repository root
#
# Optional env vars:
#   CONDA_ENV       conda environment name  (default: boltz_env)
#   DEVICE          pytorch device          (default: cuda)
#   MSA_DIRECTORY   pre-computed MSA dir    (default: unset)

#SBATCH --job-name=boltz_cache
#SBATCH --output=logs/cache_%A_%a.out
#SBATCH --error=logs/cache_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# --partition and --account are set by submit_pipeline.sh via the CLI.

set -euo pipefail

# --- Validate required env vars -------------------------------------------
: "${MANIFEST:?MANIFEST env var not set — run via submit_pipeline.sh}"
: "${CACHED_DIR:?CACHED_DIR env var not set}"
: "${CHECKPOINT:?CHECKPOINT env var not set}"
: "${PROJECT_ROOT:?PROJECT_ROOT env var not set}"
CONDA_ENV="${CONDA_ENV:-boltz_env}"
DEVICE="${DEVICE:-cuda}"

# --- Parse manifest row for this array task --------------------------------
# Manifest columns (TSV, header on line 1):
#   complex_name  receptor_pdb  mol2_file  ligand_name  dataset_type  pIC50
# Task 0 → line 2, task 1 → line 3, etc.
ROW=$(awk -v n="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==n' "$MANIFEST")
COMPLEX_NAME=$(printf '%s' "$ROW" | cut -f1)
RECEPTOR_PDB=$(printf '%s' "$ROW"  | cut -f2)
MOL2_FILE=$(printf '%s' "$ROW"     | cut -f3)
LIGAND_NAME=$(printf '%s' "$ROW"   | cut -f4)

if [[ -z "$COMPLEX_NAME" ]]; then
    echo "ERROR: no row for task $SLURM_ARRAY_TASK_ID in $MANIFEST"
    exit 1
fi

echo "======================================================="
echo "Task       : $SLURM_ARRAY_TASK_ID"
echo "Complex    : $COMPLEX_NAME"
echo "Receptor   : $RECEPTOR_PDB"
echo "MOL2       : $MOL2_FILE"
echo "Ligand     : $LIGAND_NAME"
echo "Output     : $CACHED_DIR/${COMPLEX_NAME}.pt"
echo "======================================================="

mkdir -p "$CACHED_DIR" logs

# --- Activate conda ---------------------------------------------------------
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# --- Run cache script -------------------------------------------------------
python "$PROJECT_ROOT/interpretability/cache_affinity_inputs.py" \
    --receptor    "$RECEPTOR_PDB" \
    --mol2        "$MOL2_FILE" \
    --ligand_name "$LIGAND_NAME" \
    --out         "$CACHED_DIR/${COMPLEX_NAME}.pt" \
    --checkpoint  "$CHECKPOINT" \
    --device      "$DEVICE" \
    ${MSA_DIRECTORY:+--msa_directory "$MSA_DIRECTORY"}

echo "Done → $CACHED_DIR/${COMPLEX_NAME}.pt"
