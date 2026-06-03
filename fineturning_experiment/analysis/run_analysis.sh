#!/usr/bin/env bash
# Submit the four analysis SLURM steps as an --afterok dependency chain.
#
# Usage:
#   ./run_analysis.sh            # full chain (01 → 02 → 03 → 04)
#   ./run_analysis.sh --from 02  # skip 01 (reuse YAMLs)
#   ./run_analysis.sh --from 03  # skip 01+02 (reuse poses)
#   ./run_analysis.sh --from 04  # only re-run evaluation

set -euo pipefail

FROM="01"
if [[ "${1:-}" == "--from" ]]; then
    FROM="${2:?--from needs a step number, e.g. 02}"
fi

cd "$(dirname "$0")"
mkdir -p ../logs

prev=""
submit() {
    local step="$1" script="$2"
    local dep=""
    if [[ -n "${prev}" ]]; then
        dep="--dependency=afterok:${prev}"
    fi
    local jid
    jid=$(sbatch --parsable ${dep} "${script}")
    echo "  step ${step} → job ${jid}  (${script})"
    prev="${jid}"
}

echo "Submitting analysis chain (from step ${FROM}):"
[[ "${FROM}" < "02" || "${FROM}" == "01" ]] && submit 01 01_prepare_inputs.slurm
[[ "${FROM}" < "03" || "${FROM}" == "02" ]] && submit 02 02_predict_poses.slurm
[[ "${FROM}" < "04" || "${FROM}" == "03" ]] && submit 03 03_rescore_vanilla_lora.slurm
                                              submit 04 04_evaluate.slurm
echo "Done."
