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
        return {}

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
) -> dict[str, dict]:
    """Returns {complex_name: {max_spectrum_layer, max_spectrum_head}}."""
    csv_files = sorted(results_dir.glob("*_svd.csv"))
    if not csv_files:
        print("  No *_svd.csv files found.")
        return {}

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

    return per_complex_max


# ---------------------------------------------------------------------------
# Aggregate head-ablation patching CSVs
# ---------------------------------------------------------------------------

def aggregate_head_ablation(results_dir: Path, output_dir: Path) -> None:
    patch_dir = results_dir / "patching"
    if not patch_dir.exists():
        return

    csv_files = sorted(patch_dir.glob("*_head_ablation.csv"))
    if not csv_files:
        return

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
    per_complex_max = aggregate_svd(results_dir, output_dir)

    print("\n--- Head ablation aggregation (if present) ---")
    aggregate_head_ablation(results_dir, output_dir)

    print("\n--- Per-complex summary ---")
    write_summary(
        manifest_info, baseline_ic50s, convergence_layers,
        per_complex_max, output_dir,
    )

    print(f"\nAll summary files written to {output_dir}/")


if __name__ == "__main__":
    main()
