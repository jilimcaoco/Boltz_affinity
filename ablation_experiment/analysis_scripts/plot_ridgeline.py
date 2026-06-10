#!/usr/bin/env python3
"""
Ridgeline plot of bootstrapped average logAUC distributions.
Reads pre-computed bootstrap CSVs from bootstrap_results/ — no recomputation needed.

Usage:
    python plot_ridgeline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent           # experiment_files/
BOOT_DIR = BASE_DIR / "analysis_data" / "bootstrap_results"

METHODS = ["DiffDock", "DOCK3.8", "OG_Affinity", "Boltz_Rescore"]


def load_avg_bootstraps():
    """Load the saved average bootstrap CSVs for each method."""
    data = {}
    for method in METHODS:
        fpath = BOOT_DIR / f"{method}_avg_bootstraps.csv"
        if not fpath.exists():
            print(f"Warning: {fpath} not found, skipping {method}")
            continue
        df = pd.read_csv(fpath)
        vals = df["avg_logAUC"].dropna().values
        if len(vals) > 0:
            data[method] = vals
    return data


def ridgeline_plot(method_avg_bootstraps, out_path):
    """Ridgeline plot — each scoring method on its own y-axis row."""
    methods = list(method_avg_bootstraps.keys())
    n_methods = len(methods)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    if n_methods > len(colors):
        colors = list(plt.cm.Set2(np.linspace(0, 1, n_methods)))

    fig, axes = plt.subplots(n_methods, 1, figsize=(8, 1.6 * n_methods),
                             sharex=True)
    if n_methods == 1:
        axes = [axes]

    # Shared x range
    all_vals = [method_avg_bootstraps[m] for m in methods]
    x_min = min(v.min() for v in all_vals) - 3
    x_max = max(v.max() for v in all_vals) + 3
    x_grid = np.linspace(x_min, x_max, 500)

    for i, (method, ax) in enumerate(zip(reversed(methods), axes)):
        idx = n_methods - 1 - i
        vals = method_avg_bootstraps[method]

        if len(vals) < 2:
            ax.set_ylabel(method, fontsize=11, rotation=0, labelpad=80, va="center")
            continue

        kde = gaussian_kde(vals)
        y = kde(x_grid)

        ax.fill_between(x_grid, y, alpha=0.6, color=colors[idx % len(colors)])
        ax.plot(x_grid, y, color=colors[idx % len(colors)], lw=1.5)

        # Mean line + annotation
        mean_val = np.mean(vals)
        ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
        ax.axvline(mean_val, color="k", ls="--", lw=0.8, alpha=0.7)
        y_at_mean = kde(mean_val)[0]
        ax.text(mean_val, y_at_mean * 1.05,
                f"  {mean_val:.1f} [{ci_lo:.1f}, {ci_hi:.1f}]",
                fontsize=8, va="bottom")

        ax.set_ylabel(method, fontsize=11, rotation=0, labelpad=80, va="center")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if i < n_methods - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)

    axes[-1].set_xlabel("Average logAUC", fontsize=12)
    fig.suptitle("Bootstrapped Average logAUC Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Ridgeline plot saved to {out_path}")
    plt.close(fig)


def main():
    data = load_avg_bootstraps()
    if not data:
        print("No bootstrap data found. Run compute_logauc_bootstrap.py first.")
        return
    out_path = BOOT_DIR / "ridgeline_logAUC.png"
    ridgeline_plot(data, out_path)


if __name__ == "__main__":
    main()
