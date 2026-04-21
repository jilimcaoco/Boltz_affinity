#!/bin/bash
# scripts/interpretability/run_pipeline_local.sh
# ================================================
# Master script to run the full interpretability pipeline locally
# (no SLURM required). Processes complexes sequentially.
#
# End-to-end workflow:
#   1. datasets/prepare_interpretability_set.py
#        → Filters FEP benchmark, extracts PDB+MOL2, writes manifest
#   2. cache_affinity_inputs.py (per complex)
#        → Runs affinity-rescoring trunk (no diffusion), saves .pt files
#   3. run_analysis.py (per complex)
#        → Logit-lens + SVD analysis on cached tensors
#   4. run_patching_experiment.py (per complex, optional)
#        → Causal patching (head ablation by default)
#   5. aggregate_results.py
#        → Aggregates all per-complex CSVs into summary files
#
# Usage
# -----
#   # Full pipeline from raw CIF files:
#   bash scripts/interpretability/run_pipeline_local.sh \
#       --checkpoint ~/.boltz/boltz2_aff.ckpt
#
#   # With pre-existing manifest (skips dataset preparation):
#   bash scripts/interpretability/run_pipeline_local.sh \
#       --manifest datasets/interpretability/manifest.tsv \
#       --checkpoint ~/.boltz/boltz2_aff.ckpt
#
#   # Skip caching (re-run analysis only):
#   bash scripts/interpretability/run_pipeline_local.sh \
#       --manifest datasets/interpretability/manifest.tsv \
#       --checkpoint ~/.boltz/boltz2_aff.ckpt \
#       --skip_cache
#
#   # With patching experiments:
#   bash scripts/interpretability/run_pipeline_local.sh \
#       --checkpoint ~/.boltz/boltz2_aff.ckpt \
#       --run_patch head_ablation
#
#   # Limit to first N complexes (quick test):
#   bash scripts/interpretability/run_pipeline_local.sh \
#       --checkpoint ~/.boltz/boltz2_aff.ckpt \
#       --max_complexes 5

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MANIFEST=""
CHECKPOINT=""
CACHED_DIR=""
RESULTS_DIR=""
INPUT_CSV=""
OUTPUT_BASE="datasets/interpretability"
DEVICE="cpu"
SKIP_CACHE=0
SKIP_ANALYSIS=0
RUN_PATCH=""
MAX_COMPLEXES=0
MSA_DIRECTORY=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --manifest)        MANIFEST="$2";      shift ;;
        --checkpoint)      CHECKPOINT="$2";    shift ;;
        --cached_dir)      CACHED_DIR="$2";    shift ;;
        --results_dir)     RESULTS_DIR="$2";   shift ;;
        --input_csv)       INPUT_CSV="$2";     shift ;;
        --output_base)     OUTPUT_BASE="$2";   shift ;;
        --device)          DEVICE="$2";        shift ;;
        --skip_cache)      SKIP_CACHE=1        ;;
        --skip_analysis)   SKIP_ANALYSIS=1     ;;
        --run_patch)       RUN_PATCH="$2";     shift ;;
        --max_complexes)   MAX_COMPLEXES="$2"; shift ;;
        --msa_directory)   MSA_DIRECTORY="$2"; shift ;;
        --help|-h)
            head -46 "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
[[ -z "$CHECKPOINT" ]] && { echo "ERROR: --checkpoint is required"; exit 1; }

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
[[ -f "$CHECKPOINT" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }

# Default output directories
[[ -z "$CACHED_DIR" ]]  && CACHED_DIR="$OUTPUT_BASE/cached"
[[ -z "$RESULTS_DIR" ]] && RESULTS_DIR="$OUTPUT_BASE/results"

mkdir -p "$CACHED_DIR" "$RESULTS_DIR"

echo "======================================================="
echo "Boltz-2 Interpretability Pipeline (local)"
echo "======================================================="
echo "Project root : $PROJECT_ROOT"
echo "Checkpoint   : $CHECKPOINT"
echo "Device       : $DEVICE"
echo "Cached dir   : $CACHED_DIR"
echo "Results dir  : $RESULTS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Stage 0: Dataset preparation (if no manifest provided)
# ---------------------------------------------------------------------------
if [[ -z "$MANIFEST" ]]; then
    echo "=== Stage 0: Prepare interpretability dataset ==="

    PREP_ARGS=(
        --output_dir "$OUTPUT_BASE"
    )
    [[ -n "$INPUT_CSV" ]] && PREP_ARGS+=(--input "$INPUT_CSV")

    python "$PROJECT_ROOT/datasets/prepare_interpretability_set.py" "${PREP_ARGS[@]}"

    MANIFEST="$OUTPUT_BASE/manifest.tsv"
    echo ""
fi

[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest not found: $MANIFEST"; exit 1; }

# Count complexes
N_TOTAL=$(tail -n +2 "$MANIFEST" | grep -c . || true)
if [[ "$N_TOTAL" -eq 0 ]]; then
    echo "ERROR: manifest is empty"
    exit 1
fi

N_COMPLEXES="$N_TOTAL"
if [[ "$MAX_COMPLEXES" -gt 0 ]] && [[ "$MAX_COMPLEXES" -lt "$N_TOTAL" ]]; then
    N_COMPLEXES="$MAX_COMPLEXES"
    echo "Limiting to first $N_COMPLEXES of $N_TOTAL complexes (--max_complexes)"
fi

echo "Manifest     : $MANIFEST ($N_TOTAL total, processing $N_COMPLEXES)"
echo ""

# ---------------------------------------------------------------------------
# Stage 1: Cache AffinityModule inputs
# ---------------------------------------------------------------------------
if [[ "$SKIP_CACHE" -eq 0 ]]; then
    echo "=== Stage 1: Cache AffinityModule inputs ($N_COMPLEXES complexes) ==="
    CACHED_OK=0
    CACHED_FAIL=0
    CACHED_SKIP=0

    for IDX in $(seq 0 $((N_COMPLEXES - 1))); do
        ROW=$(awk -v n="$((IDX + 2))" 'NR==n' "$MANIFEST")
        COMPLEX_NAME=$(printf '%s' "$ROW" | cut -f1)
        RECEPTOR_PDB=$(printf '%s' "$ROW" | cut -f2)
        MOL2_FILE=$(printf '%s' "$ROW"    | cut -f3)
        LIGAND_NAME=$(printf '%s' "$ROW"  | cut -f4)

        OUT_PT="$CACHED_DIR/${COMPLEX_NAME}.pt"

        if [[ -f "$OUT_PT" ]]; then
            CACHED_SKIP=$((CACHED_SKIP + 1))
            continue
        fi

        echo "  [$((IDX + 1))/$N_COMPLEXES] $COMPLEX_NAME"

        CACHE_ARGS=(
            --receptor    "$RECEPTOR_PDB"
            --mol2        "$MOL2_FILE"
            --ligand_name "$LIGAND_NAME"
            --out         "$OUT_PT"
            --checkpoint  "$CHECKPOINT"
            --device      "$DEVICE"
        )
        [[ -n "$MSA_DIRECTORY" ]] && CACHE_ARGS+=(--msa_directory "$MSA_DIRECTORY")

        if python "$PROJECT_ROOT/interpretability/cache_affinity_inputs.py" \
                "${CACHE_ARGS[@]}" 2>&1; then
            CACHED_OK=$((CACHED_OK + 1))
        else
            echo "    FAILED: $COMPLEX_NAME"
            CACHED_FAIL=$((CACHED_FAIL + 1))
        fi
    done

    echo ""
    echo "Cache summary: $CACHED_OK ok, $CACHED_FAIL failed, $CACHED_SKIP skipped (already cached)"
    echo ""
else
    echo "=== Stage 1: SKIPPED (--skip_cache) ==="
    echo ""
fi

# ---------------------------------------------------------------------------
# Stage 2: Logit-lens + SVD analysis
# ---------------------------------------------------------------------------
if [[ "$SKIP_ANALYSIS" -eq 0 ]]; then
    echo "=== Stage 2: Logit-lens + SVD analysis ($N_COMPLEXES complexes) ==="
    ANALYZED_OK=0
    ANALYZED_FAIL=0
    ANALYZED_SKIP=0

    for IDX in $(seq 0 $((N_COMPLEXES - 1))); do
        ROW=$(awk -v n="$((IDX + 2))" 'NR==n' "$MANIFEST")
        COMPLEX_NAME=$(printf '%s' "$ROW" | cut -f1)

        CACHED_PT="$CACHED_DIR/${COMPLEX_NAME}.pt"
        LOGIT_CSV="$RESULTS_DIR/${COMPLEX_NAME}_logit_lens.csv"

        if [[ ! -f "$CACHED_PT" ]]; then
            continue
        fi

        if [[ -f "$LOGIT_CSV" ]]; then
            ANALYZED_SKIP=$((ANALYZED_SKIP + 1))
            continue
        fi

        echo "  [$((IDX + 1))/$N_COMPLEXES] $COMPLEX_NAME"

        if python "$PROJECT_ROOT/interpretability/run_analysis.py" \
                --complex_name  "$COMPLEX_NAME" \
                --cached_inputs "$CACHED_PT" \
                --checkpoint    "$CHECKPOINT" \
                --output_dir    "$RESULTS_DIR" \
                --run_logit_lens \
                --run_svd \
                --run_activation_svd \
                --device "$DEVICE" 2>&1; then
            ANALYZED_OK=$((ANALYZED_OK + 1))
        else
            echo "    FAILED: $COMPLEX_NAME"
            ANALYZED_FAIL=$((ANALYZED_FAIL + 1))
        fi
    done

    echo ""
    echo "Analysis summary: $ANALYZED_OK ok, $ANALYZED_FAIL failed, $ANALYZED_SKIP skipped"
    echo ""
else
    echo "=== Stage 2: SKIPPED (--skip_analysis) ==="
    echo ""
fi

# ---------------------------------------------------------------------------
# Stage 3: Causal patching (optional)
# ---------------------------------------------------------------------------
if [[ -n "$RUN_PATCH" ]]; then
    echo "=== Stage 3: Causal patching — $RUN_PATCH ($N_COMPLEXES complexes) ==="
    PATCHED_OK=0
    PATCHED_FAIL=0

    PATCH_DIR="$RESULTS_DIR/patching"
    mkdir -p "$PATCH_DIR"

    for IDX in $(seq 0 $((N_COMPLEXES - 1))); do
        ROW=$(awk -v n="$((IDX + 2))" 'NR==n' "$MANIFEST")
        COMPLEX_NAME=$(printf '%s' "$ROW" | cut -f1)
        CACHED_PT="$CACHED_DIR/${COMPLEX_NAME}.pt"

        if [[ ! -f "$CACHED_PT" ]]; then
            continue
        fi

        PATCH_CSV="$PATCH_DIR/${COMPLEX_NAME}_${RUN_PATCH}.csv"
        if [[ -f "$PATCH_CSV" ]]; then
            continue
        fi

        echo "  [$((IDX + 1))/$N_COMPLEXES] $COMPLEX_NAME"

        if python "$PROJECT_ROOT/scripts/interpretability/run_patching_experiment.py" \
                --complex_name  "$COMPLEX_NAME" \
                --cached_inputs "$CACHED_PT" \
                --checkpoint    "$CHECKPOINT" \
                --output_dir    "$PATCH_DIR" \
                --experiment    "$RUN_PATCH" \
                --device cpu 2>&1; then
            PATCHED_OK=$((PATCHED_OK + 1))
        else
            echo "    FAILED: $COMPLEX_NAME"
            PATCHED_FAIL=$((PATCHED_FAIL + 1))
        fi
    done

    echo ""
    echo "Patching summary: $PATCHED_OK ok, $PATCHED_FAIL failed"
    echo ""
fi

# ---------------------------------------------------------------------------
# Stage 4: Aggregate results
# ---------------------------------------------------------------------------
echo "=== Stage 4: Aggregate results ==="

SUMMARY_DIR="$RESULTS_DIR/summary"
mkdir -p "$SUMMARY_DIR"

python "$PROJECT_ROOT/scripts/interpretability/aggregate_results.py" \
    --results_dir "$RESULTS_DIR" \
    --manifest    "$MANIFEST" \
    --output_dir  "$SUMMARY_DIR"

echo ""
echo "======================================================="
echo "Pipeline complete."
echo ""
echo "Outputs:"
echo "  Cached tensors : $CACHED_DIR/"
echo "  Per-complex    : $RESULTS_DIR/"
echo "  Summaries      : $SUMMARY_DIR/"
echo ""
echo "Key summary files:"
[[ -f "$SUMMARY_DIR/layer_importance.csv" ]]      && echo "  $SUMMARY_DIR/layer_importance.csv"
[[ -f "$SUMMARY_DIR/head_spectrum_ratios.csv" ]]   && echo "  $SUMMARY_DIR/head_spectrum_ratios.csv"
[[ -f "$SUMMARY_DIR/correlation_by_layer.csv" ]]   && echo "  $SUMMARY_DIR/correlation_by_layer.csv"
[[ -f "$SUMMARY_DIR/summary.csv" ]]                && echo "  $SUMMARY_DIR/summary.csv"
[[ -f "$SUMMARY_DIR/head_ablation_summary.csv" ]]  && echo "  $SUMMARY_DIR/head_ablation_summary.csv"
echo "======================================================="
