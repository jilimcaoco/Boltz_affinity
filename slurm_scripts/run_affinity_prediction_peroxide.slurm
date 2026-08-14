#!/bin/bash
#SBATCH --job-name=boltz_screen_test
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:60:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=48000m
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --output=boltz_screen_test.log

# Create logs directory if it doesn't exist
#mkdir -p logs

# Load conda environment
source ~/opt/anaconda3/etc/profile.d/conda.sh
conda activate boltz_affinity
module load cuda/12.8.1
# Set working directory
cd /home/limcaoco/turbo/limcaoco/Boltz_affinity


# Run affinity prediction
echo "Starting affinity prediction at $(date)"


boltz rescore receptor \
  --receptor AA2AR_receptor.pdb \
  --ligands AA2AR_poses.mol2 \
  -o AA2AR_scores.csv \
  --use-msa-server \
  --log-level DEBUG