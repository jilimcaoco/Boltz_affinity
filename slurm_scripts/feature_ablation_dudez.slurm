#!/bin/bash
#SBATCH --job-name=dudez_ablation
#SBATCH --account=maom0
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=48000m
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --chdir=/home/limcaoco/turbo/limcaoco/Boltz_affinity
#SBATCH --output=/home/limcaoco/turbo/limcaoco/Boltz_affinity/ablation_experiment/logs/dudez_ablation_%A_%a.log
#
# DUDEZ feature-ablation pipeline — one SLURM array task per receptor.
# Each task:
#   1. Untars combined_structures.tar.gz if the directory isn't already
#      populated (idempotent — skipped on retry).
#   2. Runs ablation_experiment/experiment_scripts/run_feature_ablation_dudez.py
#      against that single receptor.
#   3. Writes per-receptor CSV under
#      ablation_experiment/results/dudez_ablation/<RECEPTOR>_dudez_ablation.csv
#
# Submission helper:
#   ablation_experiment/experiment_scripts/submit_feature_ablation_dudez.sh
# resolves the receptor list, writes it to the manifest and submits this
# script as an array indexed into the manifest.

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────
#module load cuda/12.8.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz_affinity

# Ensure conda's C/C++ runtime is preferred over system libstdc++ on compute nodes.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

cd /home/limcaoco/turbo/limcaoco/Boltz_affinity
export BOLTZ_CACHE=/home/limcaoco/turbo/limcaoco/.boltz

echo "[$(date)] env sanity: CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "[$(date)] env sanity: python=$(which python)"
echo "[$(date)] env sanity: LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

mkdir -p ablation_experiment/logs ablation_experiment/results/dudez_ablation

# ── Receptor selection ──────────────────────────────────────────────
# Manifest is 1 line per receptor; SLURM_ARRAY_TASK_ID picks the row.
MANIFEST="${MANIFEST:-ablation_experiment/results/dudez_ablation/_receptor_manifest.txt}"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: receptor manifest not found at ${MANIFEST}." >&2
    echo "Submit via submit_feature_ablation_dudez.sh, which builds it." >&2
    exit 2
fi

RECEPTOR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")
if [[ -z "${RECEPTOR}" ]]; then
    echo "ERROR: empty receptor at line ${SLURM_ARRAY_TASK_ID} of ${MANIFEST}" >&2
    exit 3
fi

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} receptor=${RECEPTOR}"

# ── Paths ───────────────────────────────────────────────────────────
BOLTZ_OUTPUTS_ROOT="${BOLTZ_OUTPUTS_ROOT:-/scratch/maom_root/maom/limcaoco/Boltz_outputs}"
DUDEZ_INPUTS_ROOT="${DUDEZ_INPUTS_ROOT:-/home/limcaoco/turbo/limcaoco/boltz_benchmark/input_files/DUDEZ_benchmark}"
MSA_DIR="${MSA_DIR:-/home/limcaoco/turbo/limcaoco/boltz_benchmark/input_files/msa}"

RECEPTOR_DIR="${BOLTZ_OUTPUTS_ROOT}/${RECEPTOR}"
STRUCTURES_DIR="${RECEPTOR_DIR}/combined_structures"
TARBALL="${RECEPTOR_DIR}/combined_structures.tar.gz"
SMILES_CSV="${DUDEZ_INPUTS_ROOT}/${RECEPTOR}_combined_ids.csv"
MSA_PATH="${MSA_DIR}/${RECEPTOR}_mmseqs2.a3m"
OUTPUT="ablation_experiment/results/dudez_ablation/${RECEPTOR}_dudez_ablation.csv"
# Keep the DUDEZ trunk cache SEPARATE from the MOL2 runner's: entries are
# keyed on (receptor, ligand) with no record of which pose source produced
# them, so a shared directory would serve the wrong z.
# Donor cache lives on SCRATCH, not in the repo tree: entries are ~16 MB
# each (z is N*N*token_z fp16) and the cache is a throwaway artefact fully
# reproducible from the inputs. It is bounded to --donor-pool-size ligands
# and deleted by the runner unless --keep-trunk-cache is passed.
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/maom_root/maom/limcaoco}"
TRUNK_CACHE_DIR="${TRUNK_CACHE_DIR:-${SCRATCH_BASE}/ablation_trunk_cache_dudez/${SLURM_JOB_ID:-manual}}"
mkdir -p "${TRUNK_CACHE_DIR}"

# Untarred structures are the other big consumer. Keep them only if they
# were already there when we arrived; if this task extracted them, remove
# them on exit so a 43-receptor sweep doesn't leave every receptor's CIFs
# expanded on scratch at once. Set KEEP_STRUCTURES=1 to retain (useful when
# re-running the same receptor repeatedly).
KEEP_STRUCTURES="${KEEP_STRUCTURES:-0}"
# Set KEEP_TRUNK_CACHE=1 (and pass --keep-trunk-cache) to retain the donor
# cache for debugging; both are needed or one will undo the other.
KEEP_TRUNK_CACHE="${KEEP_TRUNK_CACHE:-0}"
WE_EXTRACTED=0

cleanup() {
    rc=$?
    if [[ "${WE_EXTRACTED}" == "1" && "${KEEP_STRUCTURES}" != "1" ]]; then
        echo "[$(date)] cleaning up extracted structures at ${STRUCTURES_DIR}"
        rm -rf "${STRUCTURES_DIR}"
    fi
    # The runner removes its own donor cache; this catches the case where it
    # died before doing so. Respect an explicit keep request, otherwise this
    # trap would silently undo --keep-trunk-cache.
    if [[ -d "${TRUNK_CACHE_DIR}" && "${KEEP_TRUNK_CACHE:-0}" != "1" ]]; then
        echo "[$(date)] cleaning up trunk cache at ${TRUNK_CACHE_DIR}"
        rm -rf "${TRUNK_CACHE_DIR}"
    fi
    exit ${rc}
}
trap cleanup EXIT

# ── Untar combined_structures if missing ────────────────────────────
# Treat "directory exists with at least one .cif" as already-extracted.
needs_untar=1
if [[ -d "${STRUCTURES_DIR}" ]]; then
    if compgen -G "${STRUCTURES_DIR}/*.cif" > /dev/null; then
        needs_untar=0
    fi
fi
if [[ "${needs_untar}" == "1" ]]; then
    if [[ ! -f "${TARBALL}" ]]; then
        echo "ERROR: tarball not found: ${TARBALL}" >&2
        exit 4
    fi
    echo "[$(date)] extracting ${TARBALL} ..."
    mkdir -p "${RECEPTOR_DIR}"
    tar -xf "${TARBALL}" -C "${RECEPTOR_DIR}"
    WE_EXTRACTED=1
    if ! compgen -G "${STRUCTURES_DIR}/*.cif" > /dev/null; then
        echo "ERROR: extraction did not produce CIFs in ${STRUCTURES_DIR}" >&2
        exit 5
    fi
    echo "[$(date)] extraction OK."
else
    echo "[$(date)] reusing existing structures dir."
fi

# ── Sanity-check inputs ─────────────────────────────────────────────
for p in "${SMILES_CSV}" "${MSA_PATH}"; do
    if [[ ! -f "${p}" ]]; then
        echo "ERROR: required input missing: ${p}" >&2
        exit 6
    fi
done

# ── Run ─────────────────────────────────────────────────────────────
EXTRA_ARGS=("${@}")
echo "[$(date)] launching run_feature_ablation_dudez.py"
python ablation_experiment/experiment_scripts/run_feature_ablation_dudez.py \
    --receptor "${RECEPTOR}" \
    --boltz-outputs-root "${BOLTZ_OUTPUTS_ROOT}" \
    --dudez-inputs-root "${DUDEZ_INPUTS_ROOT}" \
    --msa-dir "${MSA_DIR}" \
    --trunk-cache-dir "${TRUNK_CACHE_DIR}" \
    --donor-pool-size "${DONOR_POOL_SIZE:-64}" \
    --output "${OUTPUT}" \
    --log-level INFO \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
ABLATION_RC=$?

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} (${RECEPTOR}) ablation finished with ${ABLATION_RC}"
if [[ ${ABLATION_RC} -ne 0 ]]; then
    echo "ERROR: ablation failed for ${RECEPTOR}; not running the fidelity gate." >&2
    exit ${ABLATION_RC}
fi

# ── Trunk-cache fidelity gate ───────────────────────────────────────
# Every number above came from replaying the affinity head against a CACHED
# trunk. Re-run a stratified sample end-to-end, uncached, and assert the two
# agree. A silent cache/featurisation mismatch would corrupt every
# downstream number while the analysis scripts ran happily.
echo "[$(date)] running trunk-cache fidelity gate for ${RECEPTOR}"
python ablation_experiment/experiment_scripts/verify_trunk_cache.py \
    --receptor "${RECEPTOR}" \
    --trunk-cache-dir "${TRUNK_CACHE_DIR}" \
    --n-samples 50 \
    --output "ablation_experiment/analysis_data/cache_fidelity_${RECEPTOR}.csv" \
    --log-level INFO
GATE_RC=$?
if [[ ${GATE_RC} -ne 0 ]]; then
    echo "ERROR: fidelity gate FAILED for ${RECEPTOR} (exit ${GATE_RC})." >&2
    echo "Do NOT analyse ${OUTPUT}." >&2
    exit ${GATE_RC}
fi
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} (${RECEPTOR}) DONE — fidelity gate passed"
echo "[$(date)] scratch footprint before cleanup:"
du -sh "${STRUCTURES_DIR}" "${TRUNK_CACHE_DIR}" 2>/dev/null || true
