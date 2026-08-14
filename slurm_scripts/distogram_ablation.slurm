#!/bin/bash
#SBATCH --job-name=distogram_ablation
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --account=maom0
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=48000m
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --output=slurm_scripts/distogram_ablation_%j.log

# Load conda environment
module load cuda/12.8.1

# Set working directory
cd /home/limcaoco/turbo/limcaoco/Boltz_affinity/experiment_files/experiment_scripts

# Use turbo filesystem for .boltz cache (avoids home quota pressure)
export BOLTZ_CACHE=/home/limcaoco/turbo/limcaoco/.boltz

# Run distogram ablation
echo "Starting distogram ablation at $(date)"
echo "Receptor: AA2AR"
echo "Max ligands: 10"

python run_distogram_ablation.py \
  --receptor ../receptors/AA2AR_receptor.pdb \
  --ligands ../DOCK3.8_poses/AA2AR_poses.mol2 \
  --output ../results/ablation/AA2AR_ablation.csv \
  --max-ligands 10 \
  --use-msa-server \
  --log-level INFO

echo "Finished at $(date)"
