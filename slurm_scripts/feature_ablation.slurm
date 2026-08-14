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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz_affinity
module load cuda/12.8.1

cd /home/limcaoco/turbo/limcaoco/Boltz_affinity
export BOLTZ_CACHE=/home/limcaoco/turbo/limcaoco/.boltz

mkdir -p ablation_experiment/results/ablation logs

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
#INPUT_FLAG="--receptor ablation_experiment/receptors/AA2AR_receptor.pdb --ligands ablation_experiment/DOCK3.8_poses/AA2AR_poses.mol2"

# ── Run ──────────────────────────────────────────────────────────────
echo "Starting feature ablation at $(date)"
echo "Input: ${INPUT_FLAG}"
echo "Max ligands: ${MAX_LIGANDS}"
echo "Experiments: ${EXPERIMENTS:-all}"

python ablation_experiment/experiment_scripts/run_feature_ablation.py \
    ${INPUT_FLAG} \
    --max-ligands ${MAX_LIGANDS} \
    ${EXPERIMENTS} \
    --use-msa-server \
    --trunk-cache-dir ablation_experiment/results/trunk_cache \
    --output ablation_experiment/results/ablation/feature_ablation_results.csv \
    --log-level INFO
ABLATION_RC=$?

echo "Feature ablation finished at $(date) with exit code ${ABLATION_RC}"
if [[ ${ABLATION_RC} -ne 0 ]]; then
    echo "ERROR: ablation run failed; not proceeding to the fidelity gate." >&2
    exit ${ABLATION_RC}
fi

# ── Trunk-cache fidelity gate (Task 4) ────────────────────────────────
# Every ablation number above came from replaying the affinity head against
# a CACHED trunk rather than recomputing it. This re-runs a stratified
# sample end-to-end, uncached, and asserts the two agree to |dpIC50| < 1e-4.
# A silent cache or featurization mismatch would corrupt every downstream
# number while leaving the analysis scripts running happily -- so this is a
# hard gate: if it fails, the results must not be analyzed.
echo "Running trunk-cache fidelity gate at $(date)"
python ablation_experiment/experiment_scripts/verify_trunk_cache.py \
    --all-receptors \
    --n-samples 200 \
    --trunk-cache-dir ablation_experiment/results/trunk_cache \
    --use-msa-server \
    --output ablation_experiment/analysis_data/cache_fidelity.csv \
    --log-level INFO
GATE_RC=$?

if [[ ${GATE_RC} -ne 0 ]]; then
    echo "ERROR: trunk-cache fidelity gate FAILED (exit ${GATE_RC})." >&2
    echo "Do NOT analyze these results. See ablation_experiment/analysis_data/cache_fidelity.csv" >&2
    exit ${GATE_RC}
fi
echo "Fidelity gate passed at $(date)"

# ── Tie-density audit (Task 0b) ───────────────────────────────────────
# Cheap, CPU-only, and the gate that everything else waits on. Run here so
# the audit CSV is ready for review as soon as the GPU job lands.
python ablation_experiment/analysis_scripts/audit_tie_density.py \
    || echo "audit_tie_density.py failed (review manually)"

cat <<'EOF'

────────────────────────────────────────────────────────────────────────
GPU stage complete. The remaining analysis is CPU-only and gated on human
review of the tie-density audit -- it is deliberately NOT run here.

  1. Read  ablation_experiment/analysis_data/tie_density_audit.csv
     Confirm bias_only (the v(0) anchor for every Shapley value and NAE
     denominator) is sound, and that no condition you intend to interpret
     is marked 'unusable'.

  2. Then run the analysis chain (login node or a small CPU allocation):
       cd ablation_experiment && ./run_analysis.sh --tie-audit-reviewed

     Add --n-bootstraps 2000 for a faster first pass (default is 10000,
     roughly 1.5-2 h over ~14 experiments x ~18 receptors).
────────────────────────────────────────────────────────────────────────
EOF

echo "Job finished at $(date)"
