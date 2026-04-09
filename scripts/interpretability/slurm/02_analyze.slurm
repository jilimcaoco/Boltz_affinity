#!/bin/bash
# scripts/interpretability/slurm/02_analyze.slurm
# -------------------------------------------------
# SLURM array job: run logit-lens + SVD analysis for one complex per task.
#
# Submitted by submit_pipeline.sh after 01_cache.slurm completes.
# CPU-only; no GPU needed (no gradient computation).
#
# Required env vars (set by submit_pipeline.sh):
#   MANIFEST        path to the manifest TSV
#   CACHED_DIR      directory containing .pt files from 01_cache
#   CHECKPOINT      path to boltz2_aff.ckpt (needed to build AffinityModule)
#   RESULTS_DIR     directory to write result CSVs into
#   PROJECT_ROOT    absolute path to the repository root
#
# Optional env vars:
#   CONDA_ENV       conda environment name  (default: boltz_env)

#SBATCH --job-name=boltz_analyze
#SBATCH --output=logs/analyze_%A_%a.out
#SBATCH --error=logs/analyze_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# --partition and --account set by submit_pipeline.sh.

set -euo pipefail

: "${MANIFEST:?MANIFEST env var not set}"
: "${CACHED_DIR:?CACHED_DIR env var not set}"
: "${CHECKPOINT:?CHECKPOINT env var not set}"
: "${RESULTS_DIR:?RESULTS_DIR env var not set}"
: "${PROJECT_ROOT:?PROJECT_ROOT env var not set}"
CONDA_ENV="${CONDA_ENV:-boltz_env}"

# --- Parse manifest --------------------------------------------------------
ROW=$(awk -v n="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==n' "$MANIFEST")
COMPLEX_NAME=$(printf '%s' "$ROW" | cut -f1)

if [[ -z "$COMPLEX_NAME" ]]; then
    echo "ERROR: no row for task $SLURM_ARRAY_TASK_ID in $MANIFEST"
    exit 1
fi

CACHED_PT="$CACHED_DIR/${COMPLEX_NAME}.pt"
if [[ ! -f "$CACHED_PT" ]]; then
    echo "ERROR: cached inputs not found: $CACHED_PT"
    exit 1
fi

echo "======================================================="
echo "Task   : $SLURM_ARRAY_TASK_ID"
echo "Complex: $COMPLEX_NAME"
echo "Inputs : $CACHED_PT"
echo "Results: $RESULTS_DIR"
echo "======================================================="

mkdir -p "$RESULTS_DIR" logs

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

python "$PROJECT_ROOT/interpretability/run_analysis.py" \
    --complex_name  "$COMPLEX_NAME" \
    --cached_inputs "$CACHED_PT" \
    --checkpoint    "$CHECKPOINT" \
    --output_dir    "$RESULTS_DIR" \
    --run_logit_lens \
    --run_svd \
    --device cpu

echo "Done → $RESULTS_DIR/${COMPLEX_NAME}_logit_lens.csv"
echo "Done → $RESULTS_DIR/${COMPLEX_NAME}_svd.csv"
