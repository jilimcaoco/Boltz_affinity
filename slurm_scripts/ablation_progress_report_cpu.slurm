#!/bin/bash
#SBATCH --job-name=ablation_report
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96000m
#SBATCH --time=08:00:00
#SBATCH --account=maom0
#SBATCH --partition=standard
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --output=/home/limcaoco/turbo/limcaoco/Boltz_affinity/ablation_experiment/logs/ablation_report_%j.log

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz_affinity

REPO=/home/limcaoco/turbo/limcaoco/Boltz_affinity
ABL=${REPO}/ablation_experiment
PY=python

cd "${ABL}"
mkdir -p logs

echo "[$(date)] Starting ablation progress-report pipeline"
echo "[$(date)] Host=$(hostname) CPUs=${SLURM_CPUS_PER_TASK:-unset}"

# Stage 0a: merge per-receptor DUDEZ result CSVs.
${PY} analysis_scripts/combine_dudez_results.py \
  --input-dir results/dudez_ablation \
  --output results/ablation/feature_ablation_results.csv

# Stage 0: tie density audit (informative; does not block this pipeline).
${PY} analysis_scripts/audit_tie_density.py

# Stage 1: n=250 logAUC bootstrap, parallelized across CPUs.
${PY} analysis_scripts/compute_ablation_bootstrap.py \
  --n-bootstraps 250 \
  --allow-low-bootstraps \
  --jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --baseline-experiment baseline

# Stages 2-5 are independent once stage 1 exists; run in parallel.
set +e
${PY} analysis_scripts/compute_shapley_interactions.py &
PID_SHAPLEY=$!
${PY} analysis_scripts/compute_property_control.py &
PID_PROP=$!
${PY} analysis_scripts/compute_effect_normalization.py &
PID_NORM=$!
${PY} analysis_scripts/compute_distribution_stats.py \
  --input results/ablation/feature_ablation_results.csv \
  --output-dir analysis_data/distribution_stats \
  --ridgeline &
PID_DIST=$!

FAIL=0
for PID in ${PID_SHAPLEY} ${PID_PROP} ${PID_NORM} ${PID_DIST}; do
  wait ${PID} || FAIL=1
done
set -e

if [[ ${FAIL} -ne 0 ]]; then
  echo "[$(date)] WARNING: one or more optional analysis stages failed"
fi

# Plot summary panels over produced outputs.
${PY} analysis_scripts/plot_ablation_summary.py --analysis-dir analysis_data || true

# Build a separate report directory with writeup + raw bootstrap + graphs.
REPORT_TAG="$(date +%F)_dudez_ablation_progress"
${PY} analysis_scripts/generate_progress_report.py --tag "${REPORT_TAG}"

echo "[$(date)] Pipeline complete"
echo "Report root: ${ABL}/progress_reports/${REPORT_TAG}"
