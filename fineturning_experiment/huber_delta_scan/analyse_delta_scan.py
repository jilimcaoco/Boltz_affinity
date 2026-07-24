#!/usr/bin/env python
"""Aggregate per-delta metrics and produce delta vs metric comparison plots.

Called by 04_evaluate.slurm after evaluate_adapters.py has written per-delta
``metrics_summary.csv`` files into:

    <metrics_dir>/delta_<tag>/metrics_summary.csv

Outputs:
    <metrics_dir>/delta_comparison.csv        — wide table: delta × metric × target
    <plots_dir>/delta_scan_<target>_<metric>.png — one plot per target × metric
    <plots_dir>/delta_scan_overview.png         — 2×N grid, DRD4 vs 5HT2A

Usage (from 04_evaluate.slurm):

    python analyse_delta_scan.py \\
        --metrics-dir path/to/metrics \\
        --plots-dir   path/to/plots \\
        --output-csv  path/to/metrics/delta_comparison.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Metrics of interest ──────────────────────────────────────────────────────
# Tuples of (column_name_in_metrics_summary, display_label, higher_is_better)
METRICS_OF_INTEREST = [
    ("spearman_rho",    "Spearman ρ",    True),
    ("roc_auc",         "ROC AUC",       True),
    ("bedroc_20",       "BEDROC α=20",   True),
    ("ef_0.01",         "EF 1%",         True),
    ("ap",              "Avg Precision", True),
    ("rmse",            "RMSE",          False),
]

DELTA_ORDER = [1.00, 0.75, 0.50, 0.25, 0.00]
TARGETS = ["DRD4", "5HT2A"]
TARGET_COLORS = {"DRD4": "#1565C0", "5HT2A": "#C62828"}


def _tag(delta: float) -> str:
    return str(delta).replace(".", "p")


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    """Load per-delta metrics_summary.csv files and stack into one DataFrame."""
    rows = []
    for delta in DELTA_ORDER:
        tag = _tag(delta)
        path = metrics_dir / f"delta_{tag}" / "metrics_summary.csv"
        if not path.exists():
            print(f"WARNING: metrics file not found, skipping delta={delta}: {path}")
            continue
        df = pd.read_csv(path)
        df["delta"] = delta
        df["delta_tag"] = tag
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No per-delta metrics_summary.csv files found under {metrics_dir}. "
            "Run 04_evaluate.slurm first."
        )
    return pd.concat(rows, ignore_index=True)


def build_comparison_table(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long metrics frame into a wide comparison table."""
    # Normalise column names that evaluate_adapters.py may write differently
    col_map = {}
    for col_actual in long.columns:
        col_lower = col_actual.lower().replace(" ", "_")
        col_map[col_actual] = col_lower
    long = long.rename(columns=col_map)

    # Determine the model column name (evaluate_adapters uses 'model' or 'adapter')
    model_col = "model" if "model" in long.columns else "adapter"
    if model_col not in long.columns:
        raise KeyError(
            f"Expected 'model' or 'adapter' column in metrics_summary; "
            f"found: {list(long.columns)}"
        )

    # Keep only the delta adapter rows (not vanilla) for the main comparison,
    # but also expose the vanilla row so callers can check the baseline.
    target_col = "target" if "target" in long.columns else None
    if target_col is None:
        raise KeyError("Expected 'target' column in metrics_summary.")

    wanted_cols = ["delta", "delta_tag", target_col, model_col]
    metric_cols = [c for (c, _, _) in METRICS_OF_INTEREST if c in long.columns]
    ci_cols = [f"{c}_ci_lo" for c in metric_cols if f"{c}_ci_lo" in long.columns]
    ci_hi_cols = [f"{c}_ci_hi" for c in metric_cols if f"{c}_ci_hi" in long.columns]
    keep = wanted_cols + metric_cols + ci_cols + ci_hi_cols
    keep = [c for c in keep if c in long.columns]
    return long[keep].sort_values(["delta", target_col]).reset_index(drop=True)


def plot_metric_vs_delta(
    table: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    higher_is_better: bool,
    plots_dir: Path,
    targets: list[str],
) -> Path:
    """Plot one metric vs delta for both targets on a single axes."""
    fig, ax = plt.subplots(figsize=(6, 4))

    target_col = "target" if "target" in table.columns else list(
        set(table.columns) - {"delta", "delta_tag", "model", "adapter"}
    )[0]

    for tgt in targets:
        sub = table[table[target_col] == tgt].copy()
        if sub.empty or metric_col not in sub.columns:
            continue
        sub = sub.sort_values("delta")
        x = sub["delta"].values
        y = sub[metric_col].values
        color = TARGET_COLORS.get(tgt, None)

        ci_lo_col = f"{metric_col}_ci_lo"
        ci_hi_col = f"{metric_col}_ci_hi"
        if ci_lo_col in sub.columns and ci_hi_col in sub.columns:
            lo = sub[ci_lo_col].values
            hi = sub[ci_hi_col].values
            ax.fill_between(x, lo, hi, alpha=0.15, color=color)

        ax.plot(x, y, marker="o", linewidth=2, markersize=6, label=tgt, color=color)

        # Mark the best delta
        best_idx = int(np.nanargmax(y)) if higher_is_better else int(np.nanargmin(y))
        ax.axvline(x=x[best_idx], color=color, linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Huber delta (log₁₀ IC₅₀ µM)")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs Huber delta")
    ax.set_xticks(DELTA_ORDER)
    ax.set_xticklabels([str(d) for d in DELTA_ORDER])
    ax.invert_xaxis()  # show 1.0 → 0.0 left to right
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    arrow = "↑ better" if higher_is_better else "↓ better"
    ax.text(
        0.99, 0.02, arrow,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="grey",
    )

    fig.tight_layout()
    out = plots_dir / f"delta_scan_{metric_col}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_overview(
    table: pd.DataFrame,
    metrics: list[tuple[str, str, bool]],
    plots_dir: Path,
    targets: list[str],
) -> Path:
    """Grid of all metrics: rows = targets, cols = metrics."""
    valid_metrics = [(c, l, h) for (c, l, h) in metrics if c in table.columns]
    n_metrics = len(valid_metrics)
    n_targets = len(targets)
    if n_metrics == 0 or n_targets == 0:
        return plots_dir / "delta_scan_overview.png"

    fig, axes = plt.subplots(
        n_targets, n_metrics,
        figsize=(3.5 * n_metrics, 3.2 * n_targets),
        squeeze=False,
    )

    target_col = "target" if "target" in table.columns else list(
        set(table.columns) - {"delta", "delta_tag", "model", "adapter"}
    )[0]

    for ti, tgt in enumerate(targets):
        sub = table[table[target_col] == tgt].copy()
        color = TARGET_COLORS.get(tgt, "steelblue")
        for mi, (col, label, higher) in enumerate(valid_metrics):
            ax = axes[ti][mi]
            if sub.empty or col not in sub.columns:
                ax.set_visible(False)
                continue
            sub_sorted = sub.sort_values("delta")
            x = sub_sorted["delta"].values
            y = sub_sorted[col].values

            ci_lo_col = f"{col}_ci_lo"
            ci_hi_col = f"{col}_ci_hi"
            if ci_lo_col in sub_sorted.columns and ci_hi_col in sub_sorted.columns:
                lo = sub_sorted[ci_lo_col].values
                hi = sub_sorted[ci_hi_col].values
                ax.fill_between(x, lo, hi, alpha=0.15, color=color)

            ax.plot(x, y, marker="o", linewidth=1.8, markersize=5, color=color)
            ax.set_xticks(DELTA_ORDER)
            ax.set_xticklabels([str(d) for d in DELTA_ORDER], fontsize=7)
            ax.invert_xaxis()
            ax.set_title(f"{tgt} — {label}", fontsize=9)
            ax.set_xlabel("delta", fontsize=8)
            ax.grid(True, alpha=0.2)

            best_idx = int(np.nanargmax(y)) if higher else int(np.nanargmin(y))
            ax.axvline(x=x[best_idx], color=color, linestyle="--", linewidth=0.8, alpha=0.7)
            arrow = "↑" if higher else "↓"
            ax.text(
                0.97, 0.03, arrow,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="grey",
            )

    fig.suptitle("boltz2_affinity_loss — Huber delta scan", fontsize=12, y=1.01)
    fig.tight_layout()
    out = plots_dir / "delta_scan_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", required=True, type=Path,
                        help="Directory containing delta_<tag>/metrics_summary.csv subdirs.")
    parser.add_argument("--plots-dir",   required=True, type=Path,
                        help="Directory to write PNG plots into.")
    parser.add_argument("--output-csv",  required=True, type=Path,
                        help="Path to write the aggregated delta_comparison.csv.")
    parser.add_argument("--targets", nargs="+", default=TARGETS,
                        help="Target names to include (default: DRD4 5HT2A).")
    args = parser.parse_args()

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading per-delta metrics from {args.metrics_dir} …")
    long = load_metrics(args.metrics_dir)

    comparison = build_comparison_table(long)
    comparison.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}  ({len(comparison)} rows)")

    # Filter to delta adapter rows only (skip vanilla rows in the per-delta files)
    target_col = "target" if "target" in comparison.columns else None
    delta_rows = comparison[comparison["delta"].notna()]

    print("Generating per-metric plots …")
    for metric_col, metric_label, higher in METRICS_OF_INTEREST:
        if metric_col not in delta_rows.columns:
            print(f"  SKIP {metric_col}: not present in metrics files.")
            continue
        out = plot_metric_vs_delta(
            delta_rows, metric_col, metric_label, higher,
            args.plots_dir, args.targets,
        )
        print(f"  {out.name}")

    print("Generating overview grid …")
    out = plot_overview(delta_rows, METRICS_OF_INTEREST, args.plots_dir, args.targets)
    print(f"  {out.name}")

    print("\nDelta comparison summary (first 40 rows):")
    print(delta_rows.to_string(index=False, max_rows=40))


if __name__ == "__main__":
    main()
