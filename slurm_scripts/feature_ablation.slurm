#!/bin/bash
#SBATCH --job-name=feat_ablation
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=48000m
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --output=logs/feature_ablation_%j.log

# ── Environment ──────────────────────────────────────────────────────
source ~/opt/anaconda3/etc/profile.d/conda.sh
conda activate boltz_affinity
module load cuda/12.8.1

cd /home/limcaoco/turbo/limcaoco/Boltz_affinity
export BOLTZ_CACHE=/home/limcaoco/turbo/limcaoco/.boltz

mkdir -p experiment_files/results/ablation logs

# ── Configuration ────────────────────────────────────────────────────
# Uncomment ONE of the run modes below, or customise as needed.

# --- Mode 1: All receptors, all experiments (full matrix) ---
 MAX_LIGANDS=200
 EXPERIMENTS=""
 INPUT_FLAG="--all-receptors"

# --- Mode 2: All receptors, channel-level only (faster) ---
# MAX_LIGANDS=20
# EXPERIMENTS="--experiments baseline no_distogram no_z_trunk no_s_inputs distogram_only z_trunk_only s_inputs_only bias_only"
# INPUT_FLAG="--all-receptors"

# --- Mode 3: All receptors, sub-component only ---
# MAX_LIGANDS=20
# EXPERIMENTS="--experiments baseline no_atom_encoder no_msa_profile no_res_type only_atom_encoder only_msa_profile only_res_type"
# INPUT_FLAG="--all-receptors"

# --- Mode 4: Single receptor quick test (default) ---
#MAX_LIGANDS=10
#EXPERIMENTS=""
#INPUT_FLAG="--receptor experiment_files/receptors/AA2AR_receptor.pdb --ligands experiment_files/DOCK3.8_poses/AA2AR_poses.mol2"

# ── Run ──────────────────────────────────────────────────────────────
echo "Starting feature ablation at $(date)"
echo "Input: ${INPUT_FLAG}"
echo "Max ligands: ${MAX_LIGANDS}"
echo "Experiments: ${EXPERIMENTS:-all}"

python experiment_files/experiment_scripts/run_feature_ablation.py \
    ${INPUT_FLAG} \
    --max-ligands ${MAX_LIGANDS} \
    ${EXPERIMENTS} \
    --use-msa-server \
    --output experiment_files/results/ablation/feature_ablation_results.csv \
    --log-level INFO

echo "Feature ablation finished at $(date) with exit code $?"
