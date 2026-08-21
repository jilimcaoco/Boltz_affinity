#!/bin/bash
#SBATCH --job-name=dudez_mean_ablation
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --gpus-per-node=1
# Featurisation (process_input -> tokenize -> crop -> featurize) is CPU-bound
# and runs between GPU forwards, so starving it stalls the GPU. 4 is a
# reasonable floor; raise if the log shows long gaps between ligands.
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=48000m
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --chdir=/home/limcaoco/turbo/limcaoco/Boltz_affinity
#SBATCH --output=/home/limcaoco/turbo/limcaoco/Boltz_affinity/ablation_experiment/logs/dudez_mean_ablation_%A_%a.log
#
# DUDEZ MEAN ablation — one SLURM array task per receptor.
#
# Sibling of feature_ablation_dudez.slurm (zero ablation) and
# replacement_ablation_dudez.slurm (resample ablation). Writes to its OWN
# output directory, manifest, and trunk-cache base path so none of the
# three runs can clobber each other:
#
#   Zero ablation         -> ablation_experiment/results/dudez_ablation/<R>_dudez_ablation.csv
#   Replacement (resample)-> ablation_experiment/results/dudez_ablation_resample/<R>_dudez_ablation_resample.csv
#   Mean (this script)    -> ablation_experiment/results/dudez_ablation_mean/<R>_dudez_ablation_mean.csv
#
# NOTE on redundancy with zero ablation: "mean" substitutes the per-position
# AVERAGE of the donor pool's channel value instead of zero. Whether this is
# informative beyond the zero-ablation result depends on whether the mean
# donor representation is itself close to the zero vector (in which case
# "mean" and "no_z_trunk"/etc. would tell you nearly the same thing) versus
# a genuinely different in-distribution value (in which case it isolates the
# "network reacts badly to OOD input" confound the same way resample does,
# just with one averaged donor instead of 5 individual ones). This is an
# empirical question -- run this after the replacement (resample) ablation
# and compare `*__mean` against both `*` (zero) and `*__resample__d*` before
# deciding whether mean runs are worth repeating for the remaining
# receptors.
#
# Submission helper:
#   ablation_experiment/experiment_scripts/submit_mean_ablation_dudez.sh

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────
#module load cuda/12.8.1
CONDA_BASE="${CONDA_BASE:-/home/limcaoco/opt/anaconda3}"
CONDA_INIT="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_INIT}" ]]; then
    echo "ERROR: Conda initialization script not found: ${CONDA_INIT}" >&2
    echo "Set CONDA_BASE to the Conda installation root when submitting." >&2
    exit 1
fi
source "${CONDA_INIT}"
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

mkdir -p ablation_experiment/logs ablation_experiment/results/dudez_ablation_mean

# ── Receptor selection ──────────────────────────────────────────────
# Own manifest file -- deliberately NOT _receptor_manifest.txt or
# _receptor_manifest_resample.txt.
MANIFEST="${MANIFEST:-ablation_experiment/results/dudez_ablation_mean/_receptor_manifest_mean.txt}"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: receptor manifest not found at ${MANIFEST}." >&2
    echo "Submit via submit_mean_ablation_dudez.sh, which builds it." >&2
    exit 2
fi

RECEPTOR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")
if [[ -z "${RECEPTOR}" ]]; then
    echo "ERROR: empty receptor at line ${SLURM_ARRAY_TASK_ID} of ${MANIFEST}" >&2
    exit 3
fi

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} receptor=${RECEPTOR}"

# ── Paths ───────────────────────────────────────────────────────────
BOLTZ_OUTPUTS_ROOT="${BOLTZ_OUTPUTS_ROOT:-/scratch/maom_root/maom/limcaoco/Boltz_benchmark_outputs/Boltz_outputs}"
DUDEZ_INPUTS_ROOT="${DUDEZ_INPUTS_ROOT:-/home/limcaoco/turbo/limcaoco/boltz_benchmark/input_files/DUDEZ_benchmark}"
MSA_DIR="${MSA_DIR:-/home/limcaoco/turbo/limcaoco/boltz_benchmark/input_files/msa}"

RECEPTOR_DIR="${BOLTZ_OUTPUTS_ROOT}/${RECEPTOR}"
STRUCTURES_DIR="${RECEPTOR_DIR}/combined_structures"
TARBALL="${RECEPTOR_DIR}/combined_structures.tar.gz"
SMILES_CSV="${DUDEZ_INPUTS_ROOT}/${RECEPTOR}_combined_ids.csv"
MSA_PATH="${MSA_DIR}/${RECEPTOR}_mmseqs2.a3m"
OUTPUT="ablation_experiment/results/dudez_ablation_mean/${RECEPTOR}_dudez_ablation_mean.csv"
# Separate scratch base from the zero-ablation AND replacement-ablation
# runs -- see replacement_ablation_dudez.slurm for why sharing is unsafe.
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/maom_root/maom/limcaoco}"
TRUNK_CACHE_DIR="${TRUNK_CACHE_DIR:-${SCRATCH_BASE}/ablation_trunk_cache_dudez_mean/${SLURM_JOB_ID:-manual}}"
mkdir -p "${TRUNK_CACHE_DIR}"

KEEP_STRUCTURES="${KEEP_STRUCTURES:-0}"
KEEP_TRUNK_CACHE="${KEEP_TRUNK_CACHE:-0}"
WE_EXTRACTED=0

cleanup() {
    rc=$?
    if [[ "${WE_EXTRACTED}" == "1" && "${KEEP_STRUCTURES}" != "1" ]]; then
        echo "[$(date)] cleaning up extracted structures at ${STRUCTURES_DIR}"
        rm -rf "${STRUCTURES_DIR}"
    fi
    if [[ -d "${TRUNK_CACHE_DIR}" && "${KEEP_TRUNK_CACHE:-0}" != "1" ]]; then
        echo "[$(date)] cleaning up trunk cache at ${TRUNK_CACHE_DIR}"
        rm -rf "${TRUNK_CACHE_DIR}"
    fi
    exit ${rc}
}
trap cleanup EXIT

# ── Untar combined_structures if missing ────────────────────────────
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

# ── Experiment selection ────────────────────────────────────────────
# Default: the `__mean` variant of all 7 non-baseline factorial cells.
# Override with EXPERIMENTS="a b c" to run a subset.
if [[ -n "${EXPERIMENTS:-}" ]]; then
    read -r -a EXPERIMENT_LIST <<< "${EXPERIMENTS}"
else
    CELLS=(no_distogram no_z_trunk no_s_inputs distogram_only z_trunk_only s_inputs_only bias_only)
    EXPERIMENT_LIST=()
    for c in "${CELLS[@]}"; do
        EXPERIMENT_LIST+=("${c}__mean")
    done
fi
echo "[$(date)] experiments (${#EXPERIMENT_LIST[@]}): ${EXPERIMENT_LIST[*]}"

# MW correction is disabled by default -- it's applied post-hoc to the raw
# net score during analysis instead. Set INCLUDE_MW_CORRECTION=1 to have the
# runner apply it inline instead.
MW_FLAG=(--no-mw-correction)
if [[ "${INCLUDE_MW_CORRECTION:-0}" == "1" ]]; then
    MW_FLAG=()
fi

# ── Run ─────────────────────────────────────────────────────────────
EXTRA_ARGS=("${@}")
echo "[$(date)] launching run_feature_ablation_dudez.py (mean)"
python ablation_experiment/experiment_scripts/run_feature_ablation_dudez.py \
    --receptor "${RECEPTOR}" \
    --boltz-outputs-root "${BOLTZ_OUTPUTS_ROOT}" \
    --dudez-inputs-root "${DUDEZ_INPUTS_ROOT}" \
    --msa-dir "${MSA_DIR}" \
    --trunk-cache-dir "${TRUNK_CACHE_DIR}" \
    --donor-pool-size "${DONOR_POOL_SIZE:-64}" \
    --keep-trunk-cache \
    --experiments "${EXPERIMENT_LIST[@]}" \
    "${MW_FLAG[@]}" \
    --output "${OUTPUT}" \
    --log-level INFO \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
ABLATION_RC=$?

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} (${RECEPTOR}) mean ablation finished with ${ABLATION_RC}"
if [[ ${ABLATION_RC} -ne 0 ]]; then
    echo "ERROR: ablation failed for ${RECEPTOR}; not running the fidelity gate." >&2
    exit ${ABLATION_RC}
fi

# ── Trunk-cache fidelity gate ───────────────────────────────────────
# Own output filename (cache_fidelity_mean_*).
echo "[$(date)] running trunk-cache fidelity gate for ${RECEPTOR}"
python ablation_experiment/experiment_scripts/verify_trunk_cache.py \
    --dudez-receptor "${RECEPTOR}" \
    --boltz-outputs-root "${BOLTZ_OUTPUTS_ROOT}" \
    --dudez-inputs-root "${DUDEZ_INPUTS_ROOT}" \
    --msa-dir "${MSA_DIR}" \
    --trunk-cache-dir "${TRUNK_CACHE_DIR}" \
    --n-samples 50 \
    --output "ablation_experiment/analysis_data/cache_fidelity_mean_${RECEPTOR}.csv" \
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
