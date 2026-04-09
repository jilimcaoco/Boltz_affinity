#!/bin/bash
# scripts/interpretability/slurm/04_aggregate.slurm
# ---------------------------------------------------
# Single SLURM job: aggregate all result CSVs and produce summary files.
#
# Submitted by submit_pipeline.sh after 02_analyze (and optionally 03_patch).
# CPU-only.
#
# Required env vars:
#   MANIFEST        path to the manifest TSV
#   RESULTS_DIR     directory containing per-complex result CSVs
#   PROJECT_ROOT    absolute path to repo root
#
# Optional env vars:
#   CONDA_ENV       conda env name  (default: boltz_env)

#SBATCH --job-name=boltz_aggregate
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# --partition and --account set by submit_pipeline.sh.

set -euo pipefail

: "${MANIFEST:?}"
: "${RESULTS_DIR:?}"
: "${PROJECT_ROOT:?}"
CONDA_ENV="${CONDA_ENV:-boltz_env}"

echo "======================================================="
echo "Aggregating results in: $RESULTS_DIR"
echo "Manifest: $MANIFEST"
echo "======================================================="

mkdir -p "$RESULTS_DIR/summary" logs

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

python "$PROJECT_ROOT/scripts/interpretability/aggregate_results.py" \
    --results_dir "$RESULTS_DIR" \
    --manifest    "$MANIFEST" \
    --output_dir  "$RESULTS_DIR/summary"

echo "Summary files written to $RESULTS_DIR/summary/"
ls -lh "$RESULTS_DIR/summary/"
