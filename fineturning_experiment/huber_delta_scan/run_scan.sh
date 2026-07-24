#!/usr/bin/env bash
# Submit the full Huber-delta scan pipeline as a SLURM dependency chain.
#
# Dependency graph:
#
#   01_train_drd4   (array 0-4, parallel per delta)
#   02_train_5ht2a  (array 0-4, parallel per delta, runs alongside 01)
#       ↓  (both arrays must complete)
#   03_rescore      (array 0-4, one task per delta, reuses existing val poses)
#       ↓
#   04_evaluate     (single job: per-delta metrics + delta vs metric plots)
#
# Usage:
#   ./run_scan.sh                  # full pipeline
#   ./run_scan.sh --from 03        # reuse trained adapters, redo rescore + eval
#   ./run_scan.sh --from 04        # reuse rescore CSVs, redo evaluation only
#   ./run_scan.sh --force          # retrain + re-rescore even if outputs exist

set -euo pipefail

cd "$(dirname "$0")"
source ./config.env
mkdir -p "${LOG_DIR}"

FROM_STEP="01"
FORCE="false"
while [[ $# -gt 0 ]]; do
    case "${1}" in
        --from)  FROM_STEP="${2}"; shift 2 ;;
        --force) FORCE="true";     shift   ;;
        *) echo "Unknown flag: ${1}" >&2; exit 1 ;;
    esac
done
export FORCE_RETRAIN="${FORCE}"
export FORCE_RESCORE="${FORCE}"

# submit_dep <dep_jids_colon_sep|""> <script> <label>
# Returns new job/array ID on stdout; prints label to stderr.
submit_dep() {
    local dep="$1" script="$2" label="$3"
    local flag="" jid
    [[ -n "${dep}" ]] && flag="--dependency=afterok:${dep}"
    jid=$(sbatch --parsable ${flag} "${script}")
    printf 'Submitted %-36s (job %s)\n' "${label}" "${jid}" >&2
    echo "${jid}"
}

JID_DRD4="" JID_HT2A="" JID_RESCORE="" JID_EVAL=""

# ── Train (DRD4 and 5HT2A in parallel, no upstream dependency) ───────────────
if [[ "${FROM_STEP}" < "02" || "${FROM_STEP}" == "01" ]]; then
    JID_DRD4=$(submit_dep "" 01_train_drd4.slurm  "01_train_drd4 [array 0-4]")
    JID_HT2A=$(submit_dep "" 02_train_5ht2a.slurm "02_train_5ht2a [array 0-4]")
fi

if [[ "${FROM_STEP}" < "02" || "${FROM_STEP}" == "02" ]]; then
    # Only submitted if we skipped step 01 (start from 02 doesn't make sense
    # for this pipeline since 01 and 02 are the train steps).
    # Handle gracefully: if FROM_STEP=02, treat same as 01.
    if [[ -z "${JID_DRD4}" ]]; then
        JID_DRD4=$(submit_dep "" 01_train_drd4.slurm  "01_train_drd4 [array 0-4]")
    fi
    if [[ -z "${JID_HT2A}" ]]; then
        JID_HT2A=$(submit_dep "" 02_train_5ht2a.slurm "02_train_5ht2a [array 0-4]")
    fi
fi

# ── Rescore (depends on BOTH train arrays finishing) ─────────────────────────
if [[ "${FROM_STEP}" < "04" || "${FROM_STEP}" == "03" ]]; then
    # Build combined dependency: afterok:JID_DRD4:JID_HT2A (if both are set).
    TRAIN_DEP=""
    if [[ -n "${JID_DRD4}" && -n "${JID_HT2A}" ]]; then
        TRAIN_DEP="${JID_DRD4}:${JID_HT2A}"
    elif [[ -n "${JID_DRD4}" ]]; then
        TRAIN_DEP="${JID_DRD4}"
    elif [[ -n "${JID_HT2A}" ]]; then
        TRAIN_DEP="${JID_HT2A}"
    fi
    JID_RESCORE=$(submit_dep "${TRAIN_DEP}" 03_rescore.slurm "03_rescore [array 0-4]")
fi

# ── Evaluate ─────────────────────────────────────────────────────────────────
JID_EVAL=$(submit_dep "${JID_RESCORE}" 04_evaluate.slurm "04_evaluate")

echo ""
echo "Track progress:   squeue -u ${USER}"
echo "Logs:             ${LOG_DIR}/"
echo "Adapters:         ${ADAPTERS_DIR}/"
echo "Scores:           ${SCORES_DIR}/"
echo "Metrics:          ${METRICS_DIR}/"
echo "Plots:            ${PLOTS_DIR}/"
echo ""
echo "Delta values:     1.00  0.75  0.50  0.25  0.00"
echo "Adapter prefix:   drd4_delta_scan_<tag>  /  5ht2a_delta_scan_<tag>"
