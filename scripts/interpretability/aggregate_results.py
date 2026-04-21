"""
scripts/interpretability/aggregate_results.py
==============================================
Aggregate per-complex result CSVs from the interpretability pipeline
and produce dataset-level summary files.

Outputs (all written to --output_dir)
--------------------------------------
layer_importance.csv
    Mean absolute IC50 delta per layer transition across all complexes.
    Identifies which pairformer layers are most active in shaping the
    affinity prediction.
    Columns: layer_transition, mean_abs_delta, std_abs_delta, n_complexes

head_spectrum_ratios.csv
    Mean OV spectrum ratio per (layer, head) across all complexes.
    High values indicate heads performing focused, low-rank computation.
    Columns: layer, head, mean_spectrum_ratio, std_spectrum_ratio, n_complexes

correlation_by_layer.csv  [only if pIC50 data available in manifest]
    Pearson r between pseudo_IC50 and experimental pIC50 at each layer.
    The layer where r first approaches its maximum is where the model has
    encoded experimental affinity rank.
    Columns: layer, pearson_r, n_complexes

summary.csv
    Per-complex single-row summary.
    Columns: complex_name, dataset_type, pIC50, baseline_ic50,
             convergence_layer, max_spectrum_ratio_layer,
             max_spectrum_ratio_head

head_ablation_summary.csv  [if patching/results present]
    Mean |causal_effect| per (layer, head) across all complexes.
    Columns: layer, head, mean_abs_effect, std_abs_effect, n_complexes

Usage
-----
    python scripts/interpretability/aggregate_results.py \\
        --results_dir results/ \\
        --manifest    manifest.tsv \\
        --output_dir  results/summary/
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Pure-Python statistics helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient using numpy (always available in boltz_env)."""
    import numpy as np
    if len(xs) < 3:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  (no data for {path.name})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Return {complex_name: {dataset_type, pIC50}} from the manifest TSV."""
    info: dict[str, dict] = {}
    with open(manifest_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            info[row["complex_name"]] = {
                "dataset_type": int(row.get("dataset_type", 0)),
                "pIC50": None if row.get("pIC50", "NA") == "NA"
                         else float(row["pIC50"]),
            }
    return info


# ---------------------------------------------------------------------------
# Aggregate logit-lens CSVs
# ---------------------------------------------------------------------------

def aggregate_logit_lens(
    results_dir: Path,
    manifest_info: dict[str, dict],
    output_dir: Path,
) -> tuple[dict, dict]:
    """
    Returns {complex_name: baseline_ic50} for use in summary.csv.
    Key layer here is -1 (pre-layer-0 baseline), 0, 1, ..., L-1.
    """
    csv_files = sorted(results_dir.glob("*_logit_lens.csv"))
    if not csv_files:
        print("  No *_logit_lens.csv files found.")
        return {}, {}

    # {layer_transition: [delta, ...]}
    delta_by_transition: dict[str, list[float]] = defaultdict(list)
    # {layer_idx: [(pseudo_ic50, experimental_pIC50), ...]}
    correlation_data: dict[int, list[tuple[float, float]]] = defaultdict(list)

    baseline_ic50s: dict[str, float] = {}
    convergence_layers: dict[str, int] = {}

    for csv_file in csv_files:
        complex_name = csv_file.stem.replace("_logit_lens", "")

        rows: list[tuple[int, float]] = []
        with open(csv_file) as f:
            for r in csv.DictReader(f):
                rows.append((int(r["layer"]), float(r["pseudo_ic50"])))

        if not rows:
            continue

        rows.sort(key=lambda x: x[0])  # -1, 0, 1, 2, ...

        # Baseline IC50 = the last layer's value (most processed).
        baseline_ic50s[complex_name] = rows[-1][1]

        # Experimental pIC50 for correlation.
        exp = manifest_info.get(complex_name, {}).get("pIC50")

        for i in range(1, len(rows)):
            prev_layer, prev_val = rows[i - 1]
            curr_layer, curr_val = rows[i]
            key = f"{prev_layer}→{curr_layer}"
            delta_by_transition[key].append(abs(curr_val - prev_val))

            if exp is not None:
                correlation_data[curr_layer].append((curr_val, exp))

        # Convergence layer: layer after which absolute delta < 1% of total range.
        values = [v for _, v in rows]
        total_range = max(values) - min(values)
        thresh = 0.01 * total_range if total_range > 0 else 1e-9
        convergence_layer = rows[-1][0]  # default: last
        for i in range(1, len(rows)):
            remaining = max(abs(rows[j][1] - rows[-1][1]) for j in range(i, len(rows)))
            if remaining < thresh:
                convergence_layer = rows[i - 1][0]
                break
        convergence_layers[complex_name] = convergence_layer

    # Write layer importance.
    importance_rows = [
        {
            "layer_transition": k,
            "mean_abs_delta": f"{_mean(v):.6f}",
            "std_abs_delta": f"{_std(v):.6f}",
            "n_complexes": len(v),
        }
        for k, v in sorted(delta_by_transition.items())
    ]
    _write_csv(importance_rows, output_dir / "layer_importance.csv")

    # Write correlation table if any pIC50 data is available.
    if correlation_data:
        corr_rows = []
        for layer_idx in sorted(correlation_data.keys()):
            pairs = correlation_data[layer_idx]
            pseudo_vals = [p for p, _ in pairs]
            exp_vals = [e for _, e in pairs]
            r = _pearson_r(pseudo_vals, exp_vals)
            corr_rows.append({
                "layer": layer_idx,
                "pearson_r": f"{r:.4f}",
                "n_complexes": len(pairs),
            })
        _write_csv(corr_rows, output_dir / "correlation_by_layer.csv")

    return baseline_ic50s, convergence_layers


# ---------------------------------------------------------------------------
# Aggregate SVD CSVs
# ---------------------------------------------------------------------------

def aggregate_svd(
    results_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, dict], dict[tuple[int, int], list[float]]]:
    """Returns (per_complex_max, ratios).

    per_complex_max: {complex_name: {max_spectrum_layer, max_spectrum_head}}
    ratios: {(layer, head): [spectrum_ratio, ...]} for cross-analysis
    """
    csv_files = sorted(results_dir.glob("*_svd.csv"))
    if not csv_files:
        print("  No *_svd.csv files found.")
        return {}, {}

    # {(layer, head): [spectrum_ratio, ...]}
    ratios: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_complex_max: dict[str, dict] = {}

    for csv_file in csv_files:
        complex_name = csv_file.stem.replace("_svd", "")
        local_max = -1.0
        local_max_layer = -1
        local_max_head = -1

        # The SVD CSV writes one row per direction (4 total per layer×head).
        # Filter to a single canonical direction to avoid duplicating spectrum_ratio.
        with open(csv_file) as f:
            for r in csv.DictReader(f):
                if r.get("direction_name", "") != "ov_input":
                    continue
                try:
                    sr = float(r["spectrum_ratio"])
                except (ValueError, KeyError):
                    continue
                if math.isnan(sr) or math.isinf(sr):
                    continue
                layer = int(r["layer"])
                head = int(r["head"])
                ratios[(layer, head)].append(sr)
                if sr > local_max:
                    local_max = sr
                    local_max_layer = layer
                    local_max_head = head

        per_complex_max[complex_name] = {
            "max_spectrum_ratio_layer": local_max_layer,
            "max_spectrum_ratio_head": local_max_head,
        }

    spectrum_rows = [
        {
            "layer": layer,
            "head": head,
            "mean_spectrum_ratio": f"{_mean(vals):.4f}",
            "std_spectrum_ratio":  f"{_std(vals):.4f}",
            "n_complexes": len(vals),
        }
        for (layer, head), vals in sorted(ratios.items())
    ]
    _write_csv(spectrum_rows, output_dir / "head_spectrum_ratios.csv")

    return per_complex_max, ratios


# ---------------------------------------------------------------------------
# Aggregate head-ablation patching CSVs
# ---------------------------------------------------------------------------

def aggregate_head_ablation(results_dir: Path, output_dir: Path) -> dict:
    """Returns {(layer, head): mean_abs_effect} for cross-analysis."""
    patch_dir = results_dir / "patching"
    if not patch_dir.exists():
        return {}

    csv_files = sorted(patch_dir.glob("*_head_ablation.csv"))
    if not csv_files:
        return {}

    # {(layer, head): [|effect|, ...]}
    effects: dict[tuple[int, int], list[float]] = defaultdict(list)

    for csv_file in csv_files:
        with open(csv_file) as f:
            for r in csv.DictReader(f):
                layer = int(r["layer"])
                head = int(r["head"])
                effects[(layer, head)].append(abs(float(r["causal_effect"])))

    ablation_rows = [
        {
            "layer": layer,
            "head": head,
            "mean_abs_effect": f"{_mean(vals):.6f}",
            "std_abs_effect":  f"{_std(vals):.6f}",
            "n_complexes": len(vals),
        }
        for (layer, head), vals in sorted(effects.items())
    ]
    _write_csv(ablation_rows, output_dir / "head_ablation_summary.csv")

    return {k: _mean(v) for k, v in effects.items()}


# ---------------------------------------------------------------------------
# Aggregate activation SVD CSVs
# ---------------------------------------------------------------------------

def aggregate_activation_svd(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Aggregate per-complex z-delta and attention SVD results."""

    # --- Z-delta SVD ---
    z_files = sorted(results_dir.glob("*_z_delta_svd.csv"))
    if z_files:
        # Per-layer effective rank (averaged across complexes)
        eff_rank_by_layer: dict[int, list[float]] = defaultdict(list)
        # Per-layer top singular value (how much update occurs)
        top_sv_by_layer: dict[int, list[float]] = defaultdict(list)

        for csv_file in z_files:
            with open(csv_file) as f:
                for r in csv.DictReader(f):
                    layer = int(r["layer"])
                    rank = int(r["rank"])
                    if rank == 0:
                        eff_rank_by_layer[layer].append(float(r["effective_rank"]))
                        top_sv_by_layer[layer].append(float(r["singular_value"]))

        z_rows = [
            {
                "layer": layer,
                "mean_effective_rank": f"{_mean(eff_rank_by_layer[layer]):.4f}",
                "std_effective_rank": f"{_std(eff_rank_by_layer[layer]):.4f}",
                "mean_top_sv": f"{_mean(top_sv_by_layer[layer]):.6f}",
                "std_top_sv": f"{_std(top_sv_by_layer[layer]):.6f}",
                "n_complexes": len(eff_rank_by_layer[layer]),
            }
            for layer in sorted(eff_rank_by_layer.keys())
        ]
        _write_csv(z_rows, output_dir / "z_delta_svd_summary.csv")

    # --- Attention SVD ---
    attn_files = sorted(results_dir.glob("*_attn_svd.csv"))
    if attn_files:
        eff_rank_by_head: dict[tuple[int, str, int], list[float]] = defaultdict(list)

        for csv_file in attn_files:
            with open(csv_file) as f:
                for r in csv.DictReader(f):
                    if int(r["rank"]) != 0:
                        continue
                    key = (int(r["layer"]), r["direction"], int(r["head"]))
                    eff_rank_by_head[key].append(float(r["effective_rank"]))

        attn_rows = [
            {
                "layer": layer,
                "direction": direction,
                "head": head,
                "mean_effective_rank": f"{_mean(vals):.4f}",
                "std_effective_rank": f"{_std(vals):.4f}",
                "n_complexes": len(vals),
            }
            for (layer, direction, head), vals in sorted(eff_rank_by_head.items())
        ]
        _write_csv(attn_rows, output_dir / "attn_svd_summary.csv")


# ---------------------------------------------------------------------------
# Cross-analysis: triangulate logit lens, SVD spectrum, and ablation
# ---------------------------------------------------------------------------

def aggregate_cross_analysis(
    results_dir: Path,
    output_dir: Path,
    svd_spectrum: dict[tuple[int, int], list[float]] | None = None,
    ablation_effects: dict[tuple[int, int], float] | None = None,
) -> None:
    """Produce a unified per-(layer, head) table merging all three signals.

    Columns:
      layer, head, mean_spectrum_ratio, mean_ablation_effect,
      mean_logit_lens_delta_at_layer, convergent_signal

    ``convergent_signal`` is True when all three metrics agree the head is
    important: spectrum_ratio > median, ablation_effect > median, and the
    logit lens delta at this layer is above its median.  This is the
    triangulation indicator.
    """
    # Collect logit lens layer deltas (layer → mean_abs_delta).
    layer_deltas: dict[int, float] = {}
    ll_summary = output_dir / "layer_importance.csv"
    if ll_summary.exists():
        with open(ll_summary) as f:
            for r in csv.DictReader(f):
                # Parse transition "X→Y" to get the target layer Y.
                transition = r["layer_transition"]
                try:
                    target_layer = int(transition.split("→")[1])
                except (IndexError, ValueError):
                    continue
                layer_deltas[target_layer] = float(r["mean_abs_delta"])

    # Collect SVD spectrum ratios if not already provided.
    if svd_spectrum is None:
        svd_spectrum = {}
        svd_summary = output_dir / "head_spectrum_ratios.csv"
        if svd_summary.exists():
            with open(svd_summary) as f:
                for r in csv.DictReader(f):
                    key = (int(r["layer"]), int(r["head"]))
                    svd_spectrum[key] = [float(r["mean_spectrum_ratio"])]

    # Build the cross table.
    all_keys: set[tuple[int, int]] = set()
    if svd_spectrum:
        all_keys.update(svd_spectrum.keys())
    if ablation_effects:
        all_keys.update(ablation_effects.keys())

    if not all_keys:
        print("  Cross-analysis: insufficient data (need SVD + ablation results).")
        return

    rows: list[dict] = []
    spectrum_vals: list[float] = []
    ablation_vals: list[float] = []
    delta_vals: list[float] = []

    for layer, head in sorted(all_keys):
        sr = _mean(svd_spectrum.get((layer, head), []))
        ae = ablation_effects.get((layer, head), float("nan"))
        ld = layer_deltas.get(layer, float("nan"))
        spectrum_vals.append(sr)
        ablation_vals.append(ae)
        delta_vals.append(ld)
        rows.append({
            "layer": layer,
            "head": head,
            "spectrum_ratio": sr,
            "ablation_effect": ae,
            "logit_lens_delta": ld,
        })

    # Compute medians for convergence test.
    def _median(xs):
        clean = sorted(x for x in xs if not math.isnan(x))
        if not clean:
            return 0.0
        mid = len(clean) // 2
        return clean[mid] if len(clean) % 2 else (clean[mid - 1] + clean[mid]) / 2

    sr_med = _median(spectrum_vals)
    ae_med = _median(ablation_vals)
    ld_med = _median(delta_vals)

    for row in rows:
        sr_above = row["spectrum_ratio"] > sr_med if not math.isnan(row["spectrum_ratio"]) else False
        ae_above = row["ablation_effect"] > ae_med if not math.isnan(row["ablation_effect"]) else False
        ld_above = row["logit_lens_delta"] > ld_med if not math.isnan(row["logit_lens_delta"]) else False

        # All three signals must agree for convergent_signal.
        has_all = not (math.isnan(row["spectrum_ratio"]) or
                       math.isnan(row["ablation_effect"]) or
                       math.isnan(row["logit_lens_delta"]))
        row["convergent_signal"] = has_all and sr_above and ae_above and ld_above

        # Format floats for CSV.
        row["spectrum_ratio"] = f"{row['spectrum_ratio']:.4f}"
        row["ablation_effect"] = f"{row['ablation_effect']:.6f}"
        row["logit_lens_delta"] = f"{row['logit_lens_delta']:.6f}"

    _write_csv(rows, output_dir / "cross_analysis.csv")
    convergent = sum(1 for r in rows if r["convergent_signal"])
    print(f"  {convergent}/{len(rows)} (layer, head) pairs show convergent signal.")


# ---------------------------------------------------------------------------
# Per-complex summary
# ---------------------------------------------------------------------------

def write_summary(
    manifest_info: dict[str, dict],
    baseline_ic50s: dict[str, float],
    convergence_layers: dict[str, int],
    per_complex_max: dict[str, dict],
    output_dir: Path,
) -> None:
    rows = []
    for complex_name, info in sorted(manifest_info.items()):
        row = {
            "complex_name": complex_name,
            "dataset_type": info["dataset_type"],
            "pIC50": info["pIC50"] if info["pIC50"] is not None else "NA",
            "baseline_ic50": f"{baseline_ic50s.get(complex_name, float('nan')):.6f}",
            "convergence_layer": convergence_layers.get(complex_name, "NA"),
            "max_spectrum_ratio_layer": per_complex_max.get(
                complex_name, {}
            ).get("max_spectrum_ratio_layer", "NA"),
            "max_spectrum_ratio_head": per_complex_max.get(
                complex_name, {}
            ).get("max_spectrum_ratio_head", "NA"),
        }
        rows.append(row)
    _write_csv(rows, output_dir / "summary.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate interpretability pipeline result CSVs."
    )
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing per-complex *_logit_lens.csv and *_svd.csv.")
    parser.add_argument("--manifest",    required=True,
                        help="Manifest TSV (complex_name, yaml_path, dataset_type, pIC50).")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to write summary CSVs.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    output_dir  = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results dir : {results_dir}")
    print(f"Manifest    : {args.manifest}")
    print(f"Output dir  : {output_dir}")
    print()

    manifest_info = _load_manifest(Path(args.manifest).expanduser())
    print(f"Loaded {len(manifest_info)} complexes from manifest.")

    print("\n--- Logit lens aggregation ---")
    baseline_ic50s, convergence_layers = aggregate_logit_lens(
        results_dir, manifest_info, output_dir
    )

    print("\n--- SVD aggregation ---")
    per_complex_max, svd_spectrum = aggregate_svd(results_dir, output_dir)

    print("\n--- Activation SVD aggregation ---")
    aggregate_activation_svd(results_dir, output_dir)

    print("\n--- Head ablation aggregation (if present) ---")
    ablation_effects = aggregate_head_ablation(results_dir, output_dir)

    print("\n--- Cross-analysis (triangulation) ---")
    aggregate_cross_analysis(
        results_dir, output_dir,
        svd_spectrum=svd_spectrum,
        ablation_effects=ablation_effects,
    )

    print("\n--- Per-complex summary ---")
    write_summary(
        manifest_info, baseline_ic50s, convergence_layers,
        per_complex_max, output_dir,
    )

    print(f"\nAll summary files written to {output_dir}/")


if __name__ == "__main__":
    main()
