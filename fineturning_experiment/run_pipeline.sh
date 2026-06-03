#!/bin/bash
#
# Submit the full pipeline as a dependency chain that forks after step 03:
#
#   01_pull_chembl
#     → 02_precompute_msas
#       → 03_predict_poses
#           ├── 04a_manifest_drd4  → 05a_train_drd4   (DRD4 chain)
#           └── 04b_manifest_5ht2a → 05b_train_5ht2a  (5HT2A chain)
#
# The two receptor chains always run in parallel with each other.
#
# Run from inside fineturning_experiment/:
#
#     ./run_pipeline.sh                  # full pipeline
#     ./run_pipeline.sh --from 02        # skip ChEMBL pull
#     ./run_pipeline.sh --from 03        # skip pull + MSA precompute
#     ./run_pipeline.sh --from 04        # manifest build + train only
#     ./run_pipeline.sh --from 05        # retrain both adapters only

set -euo pipefail

cd "$(dirname "$0")"
source ./config.env
mkdir -p "${LOG_DIR}" logs

FROM_STEP="01"
if [[ "${1:-}" == "--from" && -n "${2:-}" ]]; then
    FROM_STEP="$2"
fi

# submit_dep <dep_jid|""> <script> <label>
# Submits a job, optionally with afterok dependency.  Echoes the new job ID.
submit_dep() {
    local dep="$1" script="$2" label="$3" flag="" jid
    [[ -n "${dep}" ]] && flag="--dependency=afterok:${dep}"
    jid=$(sbatch --parsable ${flag} "${script}")
    printf 'Submitted %-32s (job %s)\n' "${label}" "${jid}" >&2
    echo "${jid}"
}

JID_PULL="" JID_MSA="" JID_PRED=""

# ─── Shared upstream steps (sequential) ──────────────────────────────────────
[[ "${FROM_STEP}" < "02" || "${FROM_STEP}" == "01" ]] && \
    JID_PULL=$(submit_dep ""          01_pull_chembl.slurm     "01_pull_chembl")

[[ "${FROM_STEP}" < "03" || "${FROM_STEP}" == "02" ]] && \
    JID_MSA=$(submit_dep  "${JID_PULL}" 02_precompute_msas.slurm "02_precompute_msas")

[[ "${FROM_STEP}" < "04" || "${FROM_STEP}" == "03" ]] && \
    JID_PRED=$(submit_dep "${JID_MSA}"  03_predict_poses.slurm   "03_predict_poses")

# ─── Fork: DRD4 chain ─────────────────────────────────────────────────────────
JID_MAN_DRD4=""
[[ "${FROM_STEP}" < "05" || "${FROM_STEP}" == "04" ]] && \
    JID_MAN_DRD4=$(submit_dep "${JID_PRED}" 04a_prepare_manifest_drd4.slurm "04a_manifest_drd4")

JID_TRAIN_DRD4=$(submit_dep "${JID_MAN_DRD4}" 05a_train_drd4.slurm "05a_train_drd4")

# ─── Fork: 5HT2A chain ────────────────────────────────────────────────────────
JID_MAN_HT2A=""
[[ "${FROM_STEP}" < "05" || "${FROM_STEP}" == "04" ]] && \
    JID_MAN_HT2A=$(submit_dep "${JID_PRED}" 04b_prepare_manifest_5ht2a.slurm "04b_manifest_5ht2a")

JID_TRAIN_HT2A=$(submit_dep "${JID_MAN_HT2A}" 05b_train_5ht2a.slurm "05b_train_5ht2a")

echo ""
echo "Track progress with:  squeue -u $USER"
echo "Logs:                 ${LOG_DIR}/"
echo "Adapters (when done): ${ADAPTERS_DIR}/${DRD4_LORA_NAME}/  ${HT2A_LORA_NAME}/"
