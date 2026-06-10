#!/usr/bin/env python3
"""Generate scatter plots comparing Boltz affinity rescoring against
DOCK3.8 docking scores and original (OG) affinity predictions.

Reads the combined CSV produced by combine_scores.py (or generates it on
the fly if missing).

Outputs
-------
graphs/scatter_rescore_vs_dock.png
graphs/scatter_rescore_vs_og_affinity.png
graphs/scatter_og_affinity_vs_dock.png
graphs/scatter_rescore_vs_dock_per_receptor.png
graphs/scatter_rescore_vs_og_per_receptor.png
graphs/scatter_rescore_vs_og_vs_dock_per_receptor.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # experiment_files/
DATA_DIR = BASE_DIR / "analysis_data"
COMBINED_CSV = DATA_DIR / "combined_scores.csv"
GRAPHS_DIR = BASE_DIR / "graphs"


def ensure_combined_csv() -> pd.DataFrame:
    """Return the combined dataframe, creating it if the CSV is missing."""
    if COMBINED_CSV.exists():
        return pd.read_csv(COMBINED_CSV)
    print("Combined CSV not found – running combine_scores.py first …")
    import combine_scores  # noqa: F401 – runs main when imported as script
    combine_scores.main()
    return pd.read_csv(COMBINED_CSV)


# ── plotting helpers ─────────────────────────────────────────────────────────

def _add_regression(ax, x, y):
    """Overlay OLS line + Pearson / Spearman annotations."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return
    slope, intercept = np.polyfit(x, y, 1)
    xrange = np.linspace(x.min(), x.max(), 100)
    ax.plot(xrange, slope * xrange + intercept, "r-", lw=1.2, alpha=0.8)
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)
    text = (
        f"Pearson r = {r_pearson:.3f}  (p={p_pearson:.2e})\n"
        f"Spearman ρ = {r_spearman:.3f}  (p={p_spearman:.2e})\n"
        f"n = {len(x):,}"
    )
    ax.annotate(text, xy=(0.03, 0.97), xycoords="axes fraction",
                fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))


def scatter_global(df: pd.DataFrame) -> None:
    """One scatter per comparison across ALL receptors combined."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Rescore vs DOCK3.8 ───────────────────────────────────────────────
    sub = df.dropna(subset=["rescore_affinity", "dock_score_best"])
    # Transform rescore_affinity to match dock_score_best scale
    sub["kcal_rescore_affinity"] = sub["rescore_affinity"].apply(lambda x: (6 - x) * 1.364)  
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub["dock_score_best"], sub["kcal_rescore_affinity"],
                   alpha=0.25, s=8, edgecolors="none")
        _add_regression(ax, sub["dock_score_best"].values,
                        sub["kcal_rescore_affinity"].values)
        ax.set_xlabel("DOCK3.8 Score (best pose)")
        ax.set_ylabel("Boltz Affinity Rescore (kcal/mol)")
        ax.set_title("Boltz Affinity Rescore (kcal/mol) vs DOCK3.8 Score (all receptors)")
        fig.tight_layout()
        out = GRAPHS_DIR / "scatter_rescore_vs_dock.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")

    # ── Rescore vs OG affinity ───────────────────────────────────────────
    sub = df.dropna(subset=["rescore_affinity", "og_affinity_pred"])
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub["og_affinity_pred"], sub["rescore_affinity"],
                   alpha=0.25, s=8, edgecolors="none")
        _add_regression(ax, sub["og_affinity_pred"].values,
                        sub["rescore_affinity"].values)
        ax.set_xlabel("Original Affinity Prediction")
        ax.set_ylabel("Boltz Affinity Rescore")
        ax.set_title("Boltz Affinity Rescore vs Original Affinity (all receptors)")
        fig.tight_layout()
        out = GRAPHS_DIR / "scatter_rescore_vs_og_affinity.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")

    # ── OG affinity vs DOCK3.8 ───────────────────────────────────────────
    sub = df.dropna(subset=["og_affinity_pred", "dock_score_best"])
    sub["kcal_og_affinity"] = (6 - sub["og_affinity_pred"]) * 1.364
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub["dock_score_best"], sub["kcal_og_affinity"],
                   alpha=0.25, s=8, edgecolors="none")
        _add_regression(ax, sub["dock_score_best"].values,
                        sub["kcal_og_affinity"].values)
        ax.set_xlabel("DOCK3.8 Score (kcal/mol)")
        ax.set_ylabel("Original Affinity Prediction (kcal/mol)")
        ax.set_title("OG Affinity (kcal/mol) vs DOCK3.8 Score (all receptors)")
        fig.tight_layout()
        out = GRAPHS_DIR / "scatter_og_affinity_vs_dock.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")


def scatter_per_receptor(df: pd.DataFrame) -> None:
    """Faceted grid: one subplot per receptor for each comparison."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    receptors = sorted(df["receptor_id"].dropna().unique())

    # Convert to kcal/mol for dock comparisons
    df["kcal_rescore_affinity"] = (6 - df["rescore_affinity"]) * 1.364
    df["kcal_og_affinity"] = (6 - df["og_affinity_pred"]) * 1.364

    for tag, xcol, xlabel, ycol, ylabel in [
        ("dock", "dock_score_best", "DOCK3.8 Score (best pose)",
         "kcal_rescore_affinity", "Boltz Affinity Rescore (kcal/mol)"),
        ("og", "og_affinity_pred", "Original Affinity Prediction",
         "rescore_affinity", "Boltz Affinity Rescore"),
        ("og_vs_dock", "dock_score_best", "DOCK3.8 Score (kcal/mol)",
         "kcal_og_affinity", "Original Affinity Prediction (kcal/mol)"),
    ]:
        # Only use receptors with enough data
        valid_recs = []
        for rec in receptors:
            rsub = df[df["receptor_id"] == rec].dropna(subset=[xcol, ycol])
            if len(rsub) >= 5:
                valid_recs.append(rec)
        if not valid_recs:
            continue

        ncols = 6
        nrows = int(np.ceil(len(valid_recs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                                 squeeze=False)

        for idx, rec in enumerate(valid_recs):
            ax = axes[idx // ncols][idx % ncols]
            rsub = df[df["receptor_id"] == rec].dropna(subset=[xcol, ycol])
            ax.scatter(rsub[xcol], rsub[ycol], alpha=0.3, s=6, edgecolors="none")
            _add_regression(ax, rsub[xcol].values, rsub[ycol].values)
            ax.set_title(rec, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=7)

        # Hide unused axes
        for idx in range(len(valid_recs), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.supxlabel(xlabel, fontsize=12)
        fig.supylabel(ylabel, fontsize=12)
        fig.suptitle(f"Rescore vs {tag.upper()} per Receptor", fontsize=14,
                     fontweight="bold")
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])
        out = GRAPHS_DIR / f"scatter_rescore_vs_{tag}_per_receptor.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


def main() -> None:
    df = ensure_combined_csv()
    print(f"Loaded {len(df):,} rows from combined CSV\n")

    # Exclude docking failures (steric clashes with scores in the 10000+ range)
    clash_mask = df["dock_score_best"].notna() & (df["dock_score_best"] >= 1000)
    n_clash = clash_mask.sum()
    if n_clash:
        print(f"Excluding {n_clash:,} rows with dock_score_best >= 1000 (steric clashes)")
        df = df[~clash_mask]

    scatter_global(df)
    scatter_per_receptor(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
