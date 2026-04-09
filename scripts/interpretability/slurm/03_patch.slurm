#!/bin/bash
# scripts/interpretability/slurm/03_patch.slurm
# -----------------------------------------------
# SLURM array job: run causal patching experiments for one complex per task.
#
# Submitted by submit_pipeline.sh (optionally, after 02_analyze).
# CPU-only.
#
# Required env vars:
#   MANIFEST           path to the manifest TSV
#   CACHED_DIR         directory containing .pt files
#   CHECKPOINT         path to boltz2_aff.ckpt
#   RESULTS_DIR        directory for output CSVs
#   PROJECT_ROOT       absolute path to repo root
#
# Optional env vars:
#   CONDA_ENV          conda env name             (default: boltz_env)
#   PATCH_EXPERIMENT   experiment type            (default: head_ablation)
#                        head_ablation | alanine_scan |
#                        target_shuffle | distance_perturb
#   REFERENCE_COMPLEX  name of reference complex  (target_shuffle only)
#   DISPLACEMENTS      comma-separated Å values   (distance_perturb only,
#                                                   default: 0.5,1.0,1.5,2.0)
#   N_DIRECTIONS       number of random directions (distance_perturb only,
#                                                   default: 8)

#SBATCH --job-name=boltz_patch
#SBATCH --output=logs/patch_%A_%a.out
#SBATCH --error=logs/patch_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# --partition and --account set by submit_pipeline.sh.

set -euo pipefail

: "${MANIFEST:?}"
: "${CACHED_DIR:?}"
: "${CHECKPOINT:?}"
: "${RESULTS_DIR:?}"
: "${PROJECT_ROOT:?}"
CONDA_ENV="${CONDA_ENV:-boltz_env}"
PATCH_EXPERIMENT="${PATCH_EXPERIMENT:-head_ablation}"
REFERENCE_COMPLEX="${REFERENCE_COMPLEX:-}"
DISPLACEMENTS="${DISPLACEMENTS:-0.5,1.0,1.5,2.0}"
N_DIRECTIONS="${N_DIRECTIONS:-8}"

# --- Parse manifest --------------------------------------------------------
ROW=$(awk -v n="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==n' "$MANIFEST")
COMPLEX_NAME=$(printf '%s' "$ROW" | cut -f1)

if [[ -z "$COMPLEX_NAME" ]]; then
    echo "ERROR: no row for task $SLURM_ARRAY_TASK_ID in $MANIFEST"
    exit 1
fi

CACHED_PT="$CACHED_DIR/${COMPLEX_NAME}.pt"
if [[ ! -f "$CACHED_PT" ]]; then
    echo "ERROR: $CACHED_PT not found"
    exit 1
fi

echo "======================================================="
echo "Task       : $SLURM_ARRAY_TASK_ID"
echo "Complex    : $COMPLEX_NAME"
echo "Experiment : $PATCH_EXPERIMENT"
echo "======================================================="

mkdir -p "$RESULTS_DIR/patching" logs

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# --- Build experiment-specific extra args ----------------------------------
EXTRA_ARGS=()

if [[ "$PATCH_EXPERIMENT" == "target_shuffle" ]]; then
    if [[ -z "$REFERENCE_COMPLEX" ]]; then
        echo "ERROR: REFERENCE_COMPLEX must be set for target_shuffle"
        exit 1
    fi
    EXTRA_ARGS+=(
        "--reference_cached" "$CACHED_DIR/${REFERENCE_COMPLEX}.pt"
    )
fi

if [[ "$PATCH_EXPERIMENT" == "distance_perturb" ]]; then
    EXTRA_ARGS+=(
        "--displacements" "$DISPLACEMENTS"
        "--n_directions"  "$N_DIRECTIONS"
    )
fi

python "$PROJECT_ROOT/scripts/interpretability/run_patching_experiment.py" \
    --complex_name  "$COMPLEX_NAME" \
    --cached_inputs "$CACHED_PT" \
    --checkpoint    "$CHECKPOINT" \
    --output_dir    "$RESULTS_DIR/patching" \
    --experiment    "$PATCH_EXPERIMENT" \
    "${EXTRA_ARGS[@]}"

echo "Done → $RESULTS_DIR/patching/${COMPLEX_NAME}_${PATCH_EXPERIMENT}.csv"
