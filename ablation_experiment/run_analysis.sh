#!/usr/bin/env bash
#
# Ablation analysis chain (CPU-only) with the handoff's mandatory gates.
#
# The experiment itself (run_feature_ablation.py) and the trunk-cache
# fidelity gate (verify_trunk_cache.py) need a GPU and run separately --
# see slurm_scripts/feature_ablation.slurm. Everything here operates on the
# already-written results CSV and is cheap enough to run on a login node or
# a small CPU allocation.
#
# Stages:
#   0  audit_tie_density.py           Task 0b -- HARD GATE, see below
#   1  compute_ablation_bootstrap.py  Task 0/3: tie-aware, cluster+BCa, paired
#   2  compute_shapley_interactions.py Task 2: Shapley + Mobius
#   3  compute_property_control.py    Task 5: 2D ceiling, AVE bias, residuals
#   4  compute_effect_normalization.py Task 6: NAE, receptor class, rank pres.
#   5  compute_distribution_stats.py  pre-existing supporting analysis
#   6  plot_ablation_summary.py       plots over the new columns
#
# THE TIE-AUDIT GATE
# ------------------
# Stage 0 writes analysis_data/tie_density_audit.csv and then STOPS. A
# tie-handling bug previously inflated near-constant score distributions
# (most importantly bias_only, the v(0) anchor of every normalized effect
# size) toward the theoretical maximum logAUC. Which existing conditions
# survive that bug is a judgement call a human has to make by reading the
# audit -- it is not something this script can decide, because a tie-dense
# bias_only is the expected *finding*, not a failure. Re-run with
# --tie-audit-reviewed once you have read the CSV.
#
# Usage:
#   ./run_analysis.sh                          # stage 0 only, then stop
#   ./run_analysis.sh --tie-audit-reviewed     # full chain
#   ./run_analysis.sh --tie-audit-reviewed --from 2
#   ./run_analysis.sh --tie-audit-reviewed --n-bootstraps 2000   # faster
#
# Runtime note: stage 1 at the default B=10000 over ~14 experiments x ~18
# receptors is roughly 1.5-2 h wall (measured ~0.7 ms per per-receptor
# replicate and ~11 ms per cluster replicate at 400 compounds/receptor).
# Pass --n-bootstraps 2000 (the documented minimum) for a ~5x faster pass
# while iterating.

set -euo pipefail

cd "$(dirname "$0")"

SCRIPTS="analysis_scripts"
RESULTS_CSV="results/ablation/feature_ablation_results.csv"
ANALYSIS_DIR="analysis_data"

FROM_STAGE=0
TIE_AUDIT_REVIEWED="false"
N_BOOTSTRAPS=""

while [[ $# -gt 0 ]]; do
    case "${1}" in
        --from)                FROM_STAGE="${2}"; shift 2 ;;
        --tie-audit-reviewed)  TIE_AUDIT_REVIEWED="true"; shift ;;
        --n-bootstraps)        N_BOOTSTRAPS="${2}"; shift 2 ;;
        -h|--help)             sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "Unknown flag: ${1}" >&2; exit 1 ;;
    esac
done

if [[ ! -f "${RESULTS_CSV}" ]]; then
    echo "ERROR: ${RESULTS_CSV} not found." >&2
    echo "Run the ablation first (slurm_scripts/feature_ablation.slurm)." >&2
    exit 1
fi

mkdir -p "${ANALYSIS_DIR}"

FAILED_STAGES=()
# Plain counter alongside the array: under `set -u`, bash 3.2 (macOS system
# bash) errors on ${#arr[@]} for an empty array, so we never ask for it.
N_FAILED=0

# run_stage <n> <blocking|optional> <label> <cmd...>
#
# "blocking" stages abort the chain on failure -- everything downstream
# reads their output, so continuing would just produce confusing secondary
# errors. "optional" stages are independent of one another (the property
# control, normalization and distribution stats each read stage 1's output,
# not each other's), so one failing must not silently cost you the others.
# Failures are collected and re-reported at the end, and the script exits
# non-zero, so an optional failure is never mistaken for a clean run.
run_stage() {
    local n="$1" mode="$2" label="$3"; shift 3
    if (( FROM_STAGE > n )); then
        printf '\n=== stage %s: %s -- SKIPPED (--from %s) ===\n' "${n}" "${label}" "${FROM_STAGE}"
        return 0
    fi
    printf '\n=== stage %s: %s ===\n' "${n}" "${label}"
    # Capture the command's own status directly. Using `if "$@"; then ...; fi`
    # and reading $? afterwards reports 0 for a *failed* command, because an
    # if-statement whose branch didn't run exits 0 -- which silently turned
    # every failure into "failed with exit 0" here.
    local rc=0
    "$@" || rc=$?
    if (( rc == 0 )); then
        return 0
    fi
    if [[ "${mode}" == "blocking" ]]; then
        printf '\nERROR: stage %s (%s) failed with exit %s -- aborting; downstream stages read its output.\n' \
            "${n}" "${label}" "${rc}" >&2
        exit "${rc}"
    fi
    printf '\nWARNING: stage %s (%s) failed with exit %s -- continuing with the remaining independent stages.\n' \
        "${n}" "${label}" "${rc}" >&2
    FAILED_STAGES+=("${n}:${label}")
    N_FAILED=$((N_FAILED + 1))
    return 0
}

# ── stage 0: tie-density audit (HARD GATE) ──────────────────────────────
run_stage 0 blocking "tie-density audit (Task 0b)" \
    python "${SCRIPTS}/audit_tie_density.py"

if [[ "${TIE_AUDIT_REVIEWED}" != "true" ]]; then
    cat <<EOF

────────────────────────────────────────────────────────────────────────
STOPPING: tie-density audit written, awaiting review.

  Read: ${ANALYSIS_DIR}/tie_density_audit.csv

Confirm which conditions survive -- in particular that bias_only (the v(0)
anchor for every Shapley value and NAE denominator downstream) is sound.
Conditions marked 'unusable' must be recomputed, not interpreted.

Then re-run:
  ./run_analysis.sh --tie-audit-reviewed
────────────────────────────────────────────────────────────────────────
EOF
    exit 0
fi

# ── stage 1: bootstrap ──────────────────────────────────────────────────
# Note the ${arr[@]+"${arr[@]}"} guard: under `set -u`, expanding an empty
# array as "${arr[@]}" is an "unbound variable" error on bash 3.2 (still the
# system bash on macOS), so the plain form breaks whenever --n-bootstraps
# is omitted.
BOOT_ARGS=()
[[ -n "${N_BOOTSTRAPS}" ]] && BOOT_ARGS+=(--n-bootstraps "${N_BOOTSTRAPS}")
run_stage 1 blocking "bootstrapped logAUC (Tasks 0/3)" \
    python "${SCRIPTS}/compute_ablation_bootstrap.py" ${BOOT_ARGS[@]+"${BOOT_ARGS[@]}"}

# ── stage 2: factorial attribution ──────────────────────────────────────
# Depends on stage 1's summary_per_receptor.csv and on bias_only being sound.
run_stage 2 optional "Shapley + Mobius interactions (Task 2)" \
    python "${SCRIPTS}/compute_shapley_interactions.py"

# ── stage 3: ligand-property control ────────────────────────────────────
run_stage 3 optional "ligand-property control (Task 5)" \
    python "${SCRIPTS}/compute_property_control.py"

# ── stage 4: effect-size normalization / reporting ──────────────────────
run_stage 4 optional "effect normalization + reporting (Task 6)" \
    python "${SCRIPTS}/compute_effect_normalization.py"

# ── stage 5: pre-existing distribution stats ────────────────────────────
run_stage 5 optional "distribution stats" \
    python "${SCRIPTS}/compute_distribution_stats.py" \
        --input "${RESULTS_CSV}" \
        --output-dir "${ANALYSIS_DIR}/distribution_stats" \
        --ridgeline

# ── stage 6: plots over the new columns ─────────────────────────────────
# Runs last: each panel reads a different upstream stage's output and is
# skipped (not failed) when that stage didn't produce anything.
run_stage 6 optional "ablation summary plots" \
    python "${SCRIPTS}/plot_ablation_summary.py" \
        --analysis-dir "${ANALYSIS_DIR}"

if (( N_FAILED > 0 )); then
    printf '\n=== analysis chain finished WITH FAILURES ===\n' >&2
    printf 'The following stage(s) failed; their outputs are missing or stale:\n' >&2
    for s in ${FAILED_STAGES[@]+"${FAILED_STAGES[@]}"}; do
        printf '  - stage %s\n' "${s}" >&2
    done
    printf 'Everything else above completed. Re-run individual stages with --from N.\n' >&2
    exit 1
fi

printf '\n=== analysis chain complete ===\n'
echo "Outputs under ${ANALYSIS_DIR}/:"
echo "  tie_density_audit.csv                  (diagnostic -- read first)"
echo "  ablation_bootstrap_results/            (primary logAUC + CIs)"
echo "  shapley_values.csv, mobius_interactions.csv"
echo "  receptor_bias_profile.csv, ablation_with_property_control.csv"
echo "  normalized_ablation_effect.csv, rank_preservation.csv"
echo "  graphs/                                (plots)"
