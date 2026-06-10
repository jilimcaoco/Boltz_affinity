#!/usr/bin/env python3
"""Compute per-ablation pIC50 distribution statistics (proposal #3).

For each (experiment, receptor) cell, this script summarises the per-ligand
``affinity_pred_value`` distribution so that interpretations of the bootstrap
logAUC results can distinguish *collapse* (model predicts a constant),
*inversion* (decoys ranked above actives), and *noise inflation*.

Inputs
------
- ``feature_ablation_results.csv`` (or a custom path via ``--input``) with
  the columns produced by ``run_feature_ablation.py``:
  ``receptor_id, ligand_name, experiment, affinity_pred_value,
   affinity_probability_binary, error, time_ms`` and (in the extended
  schema) ``pose_noise_sigma, pose_noise_target, pose_noise_seed``.
- Active/decoy labels follow the convention used elsewhere in the repo:
  ``ZINC*`` IDs are decoys, everything else is an active ligand.

Outputs
-------
1. ``distribution_stats_per_receptor.csv`` — per (experiment, receptor)
   summary including mean/std/min/max of pIC50, split actives vs decoys,
   mean difference, Cohen's d, AUROC, and counts.
2. ``distribution_stats_summary.csv`` — per experiment, averaged across
   receptors with 95% percentile intervals and standard errors of the
   per-receptor means.
3. Two PNG bar charts (use ``--no-plots`` to skip):
   - ``distribution_std_bar.png``: ``std(pIC50)`` per experiment.
   - ``distribution_mean_diff_bar.png``: actives−decoys mean Δ pIC50
     per experiment, with 95% CI error bars.
4. ``distribution_ridges_active_decoy.png`` (when ``--ridgeline``): per
   experiment, KDE ridges split by active/decoy class.

Usage
-----
python compute_distribution_stats.py \
    --input ../results/ablation/feature_ablation_results.csv \
    --output-dir ../analysis_data/distribution_stats
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


# ── label helpers ───────────────────────────────────────────────────────────

def is_decoy(name: str) -> bool:
    return str(name).startswith("ZINC")


def clean_ligand_name(raw: str) -> str:
    return re.sub(r"\s+none$", "", str(raw)).strip()


# ── statistics ──────────────────────────────────────────────────────────────

def auroc(active_scores: np.ndarray, decoy_scores: np.ndarray) -> float:
    """ROC-AUC where *lower* pred values indicate stronger predicted binding.

    Returns NaN if either class is empty.
    """
    if active_scores.size == 0 or decoy_scores.size == 0:
        return float("nan")
    # Rank-based formulation. lower=better => active should have lower scores
    # than decoys for a "good" model; we compute P(active < decoy).
    all_scores = np.concatenate([active_scores, decoy_scores])
    labels = np.concatenate([
        np.ones(active_scores.size, dtype=int),
        np.zeros(decoy_scores.size, dtype=int),
    ])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    # Average ranks for ties
    sorted_scores = all_scores[order]
    i = 0
    n = sorted_scores.size
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    # AUC where positive = active and "score" should rank low for positives;
    # for a metric in [0,1] where higher = better-separating-actives-low,
    # invert by negating the scores when computing the Mann-Whitney U.
    # Equivalent: AUC = (R_neg - n_neg*(n_neg+1)/2) / (n_pos * n_neg)
    neg_ranks = ranks[labels == 0]
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (neg_ranks.sum() - n_neg * (n_neg + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for means(a - b) with pooled std. NaN on degenerate inputs."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) /
                     (a.size + b.size - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def summarise_cell(df_cell: pd.DataFrame) -> dict:
    """Compute distribution statistics for one (experiment, receptor) cell."""
    pred = df_cell["affinity_pred_value"].to_numpy(dtype=float)
    is_dec = df_cell["ligand_id"].apply(is_decoy).to_numpy()
    act = pred[~is_dec]
    dec = pred[is_dec]

    row = {
        "n_total": int(pred.size),
        "n_actives": int(act.size),
        "n_decoys": int(dec.size),
        "mean_pic50_all": float(np.mean(pred)) if pred.size else float("nan"),
        "std_pic50_all": float(np.std(pred, ddof=1)) if pred.size > 1 else float("nan"),
        "min_pic50_all": float(np.min(pred)) if pred.size else float("nan"),
        "max_pic50_all": float(np.max(pred)) if pred.size else float("nan"),
        "mean_pic50_actives": float(np.mean(act)) if act.size else float("nan"),
        "std_pic50_actives": float(np.std(act, ddof=1)) if act.size > 1 else float("nan"),
        "mean_pic50_decoys": float(np.mean(dec)) if dec.size else float("nan"),
        "std_pic50_decoys": float(np.std(dec, ddof=1)) if dec.size > 1 else float("nan"),
        # actives - decoys: positive ⇒ actives predicted *higher* pIC50 (correct
        # sign for binding affinity).
        "mean_diff_act_minus_dec": (
            float(np.mean(act) - np.mean(dec))
            if act.size and dec.size else float("nan")
        ),
        "cohens_d_act_vs_dec": cohens_d(act, dec),
        # AUROC where *lower* pIC50 predicts active. The training target is
        # higher = stronger binder, but we keep this metric so users can spot
        # cases where rank order is inverted (auroc < 0.5).
        "auroc_active_low": auroc(act, dec),
        # Same idea but where higher pIC50 predicts active (correct direction
        # for a well-behaved affinity head).
        "auroc_active_high": auroc(-act, -dec),
    }
    return row


# ── plotting ────────────────────────────────────────────────────────────────

def _experiment_order(experiments: list[str]) -> list[str]:
    """Stable display order: baseline first, then channel, sub-component, noise."""
    priority = {
        "baseline": 0,
        "no_distogram": 1, "no_z_trunk": 2, "no_s_inputs": 3,
        "distogram_only": 4, "z_trunk_only": 5, "s_inputs_only": 6,
        "bias_only": 7,
        "no_atom_encoder": 10, "no_msa_profile": 11, "no_res_type": 12,
        "only_atom_encoder": 13, "only_msa_profile": 14, "only_res_type": 15,
    }
    def key(name: str) -> tuple:
        if name in priority:
            return (0, priority[name], name)
        if name.startswith("lig_noise_"):
            return (1, 0, name)
        if name.startswith("rec_noise_"):
            return (2, 0, name)
        if name.startswith("all_noise_"):
            return (3, 0, name)
        return (4, 0, name)
    return sorted(experiments, key=key)


def bar_plot(summary: pd.DataFrame, col: str, errlow: str, errhigh: str,
             title: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(summary)), 4.5))
    x = np.arange(len(summary))
    vals = summary[col].to_numpy(dtype=float)
    lo = summary[errlow].to_numpy(dtype=float)
    hi = summary[errhigh].to_numpy(dtype=float)
    yerr = np.vstack([vals - lo, hi - vals])
    ax.bar(x, vals, yerr=yerr, color="#3a7fb0", alpha=0.85, capsize=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["experiment"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ridgeline_active_decoy(df: pd.DataFrame, out_path: Path,
                            experiments: list[str]):
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(
        len(experiments), 1,
        figsize=(8, max(2, 1.0 * len(experiments))),
        sharex=True,
    )
    if len(experiments) == 1:
        axes = [axes]

    all_pred = df["affinity_pred_value"].to_numpy(dtype=float)
    finite = all_pred[np.isfinite(all_pred)]
    if finite.size == 0:
        plt.close(fig)
        return
    x_min, x_max = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    pad = 0.1 * (x_max - x_min + 1e-6)
    x_grid = np.linspace(x_min - pad, x_max + pad, 400)

    for ax, exp in zip(axes, experiments):
        sub = df[df["experiment"] == exp]
        act = sub.loc[~sub["ligand_id"].apply(is_decoy), "affinity_pred_value"].to_numpy(dtype=float)
        dec = sub.loc[sub["ligand_id"].apply(is_decoy), "affinity_pred_value"].to_numpy(dtype=float)
        act = act[np.isfinite(act)]
        dec = dec[np.isfinite(dec)]
        for arr, color, label in [(act, "#1b7837", "actives"),
                                   (dec, "#762a83", "decoys")]:
            if arr.size < 2 or np.allclose(arr, arr[0]):
                continue
            try:
                kde = gaussian_kde(arr)
                y = kde(x_grid)
                ax.fill_between(x_grid, y, alpha=0.45, color=color, label=label)
                ax.plot(x_grid, y, color=color, lw=1.0)
            except Exception:
                continue
        ax.set_yticks([])
        ax.set_ylabel(exp, rotation=0, ha="right", va="center",
                      fontsize=7, labelpad=70)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Predicted pIC50")
    fig.suptitle("Per-ablation predicted pIC50 distributions (actives vs decoys)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-ablation pIC50 distribution statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = Path(__file__).resolve().parent
    default_input = here.parent / "results" / "ablation" / "feature_ablation_results.csv"
    default_out = here.parent / "analysis_data" / "distribution_stats"
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--ridgeline", action="store_true",
                        help="Also produce a ridgeline KDE plot (slow for many exps).")
    parser.add_argument("--receptors", nargs="+", default=None,
                        help="Restrict to these receptor IDs.")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Restrict to these experiment names.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input)
    df["ligand_id"] = df["ligand_name"].apply(clean_ligand_name)
    df = df[df["error"].fillna("") == ""]
    df = df[df["affinity_pred_value"].notna()]
    df["affinity_pred_value"] = pd.to_numeric(
        df["affinity_pred_value"], errors="coerce"
    )
    df = df.dropna(subset=["affinity_pred_value"])

    if args.receptors:
        df = df[df["receptor_id"].isin(args.receptors)]
    if args.experiments:
        df = df[df["experiment"].isin(args.experiments)]

    if df.empty:
        raise SystemExit("No data left after filtering; check --input/--receptors/--experiments.")

    # Per-cell stats
    cell_rows = []
    for (exp, rec), df_cell in df.groupby(["experiment", "receptor_id"]):
        row = {"experiment": exp, "receptor_id": rec}
        row.update(summarise_cell(df_cell))
        cell_rows.append(row)
    per_receptor = pd.DataFrame(cell_rows)
    per_receptor_path = args.output_dir / "distribution_stats_per_receptor.csv"
    per_receptor.to_csv(per_receptor_path, index=False)
    print(f"Wrote {per_receptor_path} ({len(per_receptor)} rows)")

    # Per-experiment summary across receptors
    numeric_cols = [c for c in per_receptor.columns
                    if c not in ("experiment", "receptor_id")
                    and pd.api.types.is_numeric_dtype(per_receptor[c])]
    rows = []
    for exp, grp in per_receptor.groupby("experiment"):
        row = {"experiment": exp, "n_receptors": len(grp)}
        for col in numeric_cols:
            vals = grp[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row[f"{col}__mean"] = float("nan")
                row[f"{col}__sem"] = float("nan")
                row[f"{col}__ci_low"] = float("nan")
                row[f"{col}__ci_high"] = float("nan")
                continue
            row[f"{col}__mean"] = float(vals.mean())
            row[f"{col}__sem"] = (
                float(vals.std(ddof=1) / np.sqrt(vals.size))
                if vals.size > 1 else float("nan")
            )
            row[f"{col}__ci_low"] = float(np.percentile(vals, 2.5))
            row[f"{col}__ci_high"] = float(np.percentile(vals, 97.5))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["experiment"] = pd.Categorical(
        summary["experiment"],
        categories=_experiment_order(summary["experiment"].tolist()),
        ordered=True,
    )
    summary = summary.sort_values("experiment").reset_index(drop=True)
    summary_path = args.output_dir / "distribution_stats_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary)} rows)")

    if not args.no_plots:
        bar_plot(
            summary, "std_pic50_all__mean",
            "std_pic50_all__ci_low", "std_pic50_all__ci_high",
            title="pIC50 dynamic range per ablation (≈0 ⇒ collapsed model)",
            ylabel="std(pIC50) across ligands",
            out_path=args.output_dir / "distribution_std_bar.png",
        )
        bar_plot(
            summary, "mean_diff_act_minus_dec__mean",
            "mean_diff_act_minus_dec__ci_low",
            "mean_diff_act_minus_dec__ci_high",
            title="Mean pIC50 separation (actives − decoys) per ablation",
            ylabel="Δ pIC50 (positive ⇒ correct direction)",
            out_path=args.output_dir / "distribution_mean_diff_bar.png",
        )
        bar_plot(
            summary, "auroc_active_high__mean",
            "auroc_active_high__ci_low", "auroc_active_high__ci_high",
            title="AUROC (actives = higher pIC50) per ablation",
            ylabel="AUROC",
            out_path=args.output_dir / "distribution_auroc_bar.png",
        )
        print(f"Wrote bar plots in {args.output_dir}")

        if args.ridgeline:
            ridgeline_active_decoy(
                df,
                args.output_dir / "distribution_ridges_active_decoy.png",
                experiments=_experiment_order(df["experiment"].unique().tolist()),
            )
            print(f"Wrote ridgeline plot in {args.output_dir}")


if __name__ == "__main__":
    main()
