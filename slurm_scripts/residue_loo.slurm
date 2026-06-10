#!/bin/bash
#SBATCH --job-name=residue_loo
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=48000m
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --output=logs/residue_loo_%j.log

# ── Environment ──────────────────────────────────────────────────────
source ~/opt/anaconda3/etc/profile.d/conda.sh
conda activate boltz_affinity
module load cuda/12.8.1

cd /home/limcaoco/turbo/limcaoco/Boltz_affinity
export BOLTZ_CACHE=/home/limcaoco/turbo/limcaoco/.boltz

mkdir -p ablation_experiment/results/loo logs

# ── Configuration ────────────────────────────────────────────────────
# Mode A: single-receptor quick test (default)
MODE="single"
RECEPTOR_ID="AA2AR"
MAX_ACTIVES=20
MAX_DECOYS=20
RADIUS=10

# Mode B: all receptors — uncomment to enable
# MODE="all"
# MAX_ACTIVES=10
# MAX_DECOYS=10
# RADIUS=8

# ── Run ──────────────────────────────────────────────────────────────
echo "Starting residue LOO at $(date) (mode=${MODE})"

if [[ "${MODE}" == "all" ]]; then
    python ablation_experiment/experiment_scripts/run_residue_loo.py \
        --all-receptors \
        --max-actives ${MAX_ACTIVES} \
        --max-decoys ${MAX_DECOYS} \
        --radius ${RADIUS} \
        --use-msa-server \
        --output ablation_experiment/results/loo/all_residue_loo.csv \
        --log-level INFO
else
    python ablation_experiment/experiment_scripts/run_residue_loo.py \
        --receptor ablation_experiment/receptors/${RECEPTOR_ID}_receptor.pdb \
        --ligands  ablation_experiment/DOCK3.8_poses/${RECEPTOR_ID}_poses.mol2 \
        --max-actives ${MAX_ACTIVES} \
        --max-decoys ${MAX_DECOYS} \
        --radius ${RADIUS} \
        --use-msa-server \
        --output ablation_experiment/results/loo/${RECEPTOR_ID}_residue_loo.csv \
        --log-level INFO
fi

echo "Residue LOO finished at $(date) with exit code $?"

# ── Post-processing (analysis + heatmaps) ─────────────────────────────
python ablation_experiment/analysis_scripts/plot_residue_saliency.py \
    --input  ablation_experiment/results/loo/${RECEPTOR_ID}_residue_loo.csv \
    --output-dir ablation_experiment/analysis_data/residue_loo
