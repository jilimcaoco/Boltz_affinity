#!/bin/bash
#SBATCH --job-name=dist_stats
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --account=maom
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000m
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --output=logs/distribution_stats_%j.log

source ~/opt/anaconda3/etc/profile.d/conda.sh
conda activate boltz_affinity

cd /home/limcaoco/turbo/limcaoco/Boltz_affinity

INPUT=${INPUT:-ablation_experiment/results/ablation/feature_ablation_results.csv}
OUTDIR=${OUTDIR:-ablation_experiment/analysis_data/distribution_stats}

echo "Running distribution-stats analysis on ${INPUT}"
python ablation_experiment/analysis_scripts/compute_distribution_stats.py \
    --input "${INPUT}" \
    --output-dir "${OUTDIR}" \
    --ridgeline

echo "Done at $(date)."
