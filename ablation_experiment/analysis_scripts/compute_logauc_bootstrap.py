#!/usr/bin/env python3
"""
Compute per-receptor logAUC with bootstrapping (100 replicates) for 4 scoring
methods, then compute the average logAUC across receptors (also bootstrapped),
and display as a ridgeline plot.

Scoring methods (all normalized to this codebase's lower-is-better convention,
i.e. ``affinity_pred_value`` = log10(IC50_uM), confirmed in
``src/boltz/lora/losses.py`` — see per-loader docstrings for polarity):
  1. diffdock_scores  – .tsv files, space-separated, col6 (0-indexed: 5) as score.
                        DiffDock confidence is higher-is-better, so this is negated.
  2. dock3.8_scores   – CSV with ligand_id,dock_score (take min per ligand).
                        Lower-is-better already; not negated.
  3. OG_affinity_scores – master_scores.csv, "Affinity Pred Value" column. Same
                        field name/convention as affinity_pred_value elsewhere
                        in this codebase (lower-is-better); not negated. NOTE:
                        an earlier version of this docstring claimed this value
                        was negated when it never was — see Task 0c writeup.
  4. rescoring (Boltz) – *_rescored.csv, affinity_score column. This is
                        affinity_pred_value under the hood (see
                        rescorer.py::score.affinity_score assignment),
                        lower-is-better; not negated.

Convention: ZINC-prefixed IDs = decoys; everything else = ligands.
"""

import os
import sys
import glob
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logauc_utils import (  # noqa: E402
    DEFAULT_BOOTSTRAPS,
    MIN_BOOTSTRAPS,
    bootstrap_logauc,
    compute_avg_logauc_bootstrap,
    point_estimate_logauc,
    polarity_warning,
    tie_stats,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent           # experiment_files/
DIFFDOCK_DIR = BASE_DIR / "analysis_data" / "diffdock_scores"
DOCK38_DIR   = BASE_DIR / "analysis_data" / "dock3.8_scores"
OG_DIR       = BASE_DIR / "analysis_data" / "OG_affinity_scores"
RESCORE_DIR  = BASE_DIR / "results" / "rescoring"
OUT_DIR      = BASE_DIR / "analysis_data" / "bootstrap_results"

# Task 3b: 100 replicates is too few for a stable 2.5/97.5 percentile
# interval (it's set by ~2.5 order statistics). Raised to the same default
# as compute_ablation_bootstrap.py; override with --n-bootstraps.
N_BOOTSTRAPS = DEFAULT_BOOTSTRAPS
SEED = 42

# logAUC / ROC / bootstrap core now lives in logauc_utils.py (shared with
# compute_ablation_bootstrap.py) so the tie-handling fix only has one place
# to live. See that module's docstring for the bug this replaced.


# ── Data loaders ─────────────────────────────────────────────────────────────

def is_decoy(name):
    return name.startswith("ZINC")


def split_lig_dec(names):
    ligs = {n for n in names if not is_decoy(n)}
    decs = {n for n in names if is_decoy(n)}
    return ligs, decs


def load_diffdock(receptor):
    """Load diffdock scores: .tsv, space-separated, col2=name, col6=confidence.
    DiffDock confidence is higher-is-better; negated here so it matches this
    module's lower-is-better ranking convention (Task 0c polarity fix — this
    loader previously used the raw confidence unnegated, which silently
    inverted every DiffDock ROC curve)."""
    fpath = DIFFDOCK_DIR / f"{receptor}_urusai.tsv"
    if not fpath.exists():
        return None
    records = []
    with open(fpath) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            name = parts[1]
            try:
                confidence = float(parts[5])
            except ValueError:
                continue
            records.append((name, -confidence))
    return records


def load_dock38(receptor):
    """Load dock3.8 scores: CSV, take min (best) score per ligand."""
    fpath = DOCK38_DIR / f"{receptor}_dock_score.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    best = df.groupby("ligand_id")["dock_score"].min().reset_index()
    records = list(zip(best["ligand_id"], best["dock_score"]))
    return records


def load_og_affinity(receptor):
    """Load OG affinity scores for a receptor.

    Task 0c polarity note: this docstring previously claimed the "Affinity
    Pred Value" column was negated before ranking, but the code never did
    so — a mismatch between documented intent and actual behavior. The
    column name matches ``affinity_pred_value`` used everywhere else in this
    codebase (log10(IC50_uM), lower-is-better, confirmed in
    src/boltz/lora/losses.py), which is already this module's ranking
    convention, so no negation is applied. This assumes master_scores.csv's
    "Affinity Pred Value" is the same quantity; if OG_Affinity's mean logAUC
    comes out strongly negative once real data is audited, treat that as a
    signal this assumption is wrong rather than as "OG_Affinity is bad" —
    see polarity_warning() in logauc_utils.py, which flags this automatically.
    """
    fpath = OG_DIR / "master_scores.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    df_rec = df[df["receptor"] == receptor].copy()
    if df_rec.empty:
        return None
    records = list(zip(df_rec["compound_ID"], df_rec["Affinity Pred Value"]))
    return records


def load_rescoring(receptor):
    """Load Boltz affinity rescoring results."""
    fpath = RESCORE_DIR / f"{receptor}_rescored.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    # Clean ligand name: strip trailing whitespace + 'none' suffix
    df["ligand_id"] = df["ligand_name"].str.replace(r"\s+none$", "", regex=True).str.strip()
    df = df.dropna(subset=["affinity_score"])
    records = list(zip(df["ligand_id"], df["affinity_score"]))
    return records


# ── Main pipeline ────────────────────────────────────────────────────────────

def get_receptor_list():
    """Get union of receptors across all sources, excluding duplicates like ADA_."""
    receptors = set()
    for f in DOCK38_DIR.glob("*_dock_score.csv"):
        receptors.add(f.stem.replace("_dock_score", ""))
    for f in DIFFDOCK_DIR.glob("*_urusai.tsv"):
        name = f.stem.replace("_urusai", "")
        if not name.endswith("_"):  # skip ADA_ duplicate
            receptors.add(name)
    for f in RESCORE_DIR.glob("*_rescored.csv"):
        receptors.add(f.stem.replace("_rescored", ""))
    return sorted(receptors)


def process_method(method_name, loader_fn, receptors, rng):
    """For a given method, compute per-receptor bootstrapped logAUC, plus
    tie-density diagnostics and a polarity sanity check (Task 0c)."""
    all_bootstraps = {}
    all_tie_stats = {}
    all_point_est = {}
    per_receptor_means = []
    for receptor in receptors:
        records = loader_fn(receptor)
        if records is None or len(records) == 0:
            print(f"  [{method_name}] {receptor}: no data, skipping")
            continue

        # Deduplicate: keep best (lowest) score per compound
        best = {}
        for name, score in records:
            if name not in best or score < best[name]:
                best[name] = score

        names = set(best.keys())
        lig_set, dec_set = split_lig_dec(names)
        if len(lig_set) == 0 or len(dec_set) == 0:
            print(f"  [{method_name}] {receptor}: no ligs ({len(lig_set)}) or decs ({len(dec_set)}), skipping")
            continue

        lig_scores = [(n, best[n]) for n in lig_set]
        dec_scores = [(n, best[n]) for n in dec_set]

        boot_vals = bootstrap_logauc(lig_scores, dec_scores, lig_set, dec_set, N_BOOTSTRAPS, rng)
        all_bootstraps[receptor] = boot_vals
        mean_val = np.nanmean(boot_vals)
        per_receptor_means.append(mean_val)

        ties = tie_stats(lig_scores + dec_scores)
        all_tie_stats[receptor] = ties
        all_point_est[receptor] = point_estimate_logauc(lig_scores, dec_scores, lig_set, dec_set, rng)

        tie_flag = "  [TIED TOP-1%]" if ties["tied_top1pct_flag"] else ""
        print(f"  [{method_name}] {receptor}: mean logAUC = {mean_val:.2f} ({len(lig_set)} ligs, {len(dec_set)} decs, "
              f"{ties['distinct_values']} distinct scores, largest tie block {ties['largest_tie_block']}){tie_flag}")
        if ties["tied_top1pct_flag"]:
            print(f"    WARNING: top-1% of the ranking is >=99% inside a single tied score block — "
                  f"logAUC for [{method_name}] {receptor} is determined by tie-break order, not by the model.")

    warning = polarity_warning(method_name, float(np.nanmean(per_receptor_means)) if per_receptor_means else np.nan)
    if warning:
        print(f"  {warning}")

    return all_bootstraps, all_tie_stats, all_point_est


def ridgeline_plot(method_bootstraps, method_avg_bootstraps, out_path):
    """Ridgeline plot of bootstrapped average logAUC distributions.
    Each scoring method gets its own row (y-axis) with overlapping KDE fills."""
    methods = list(method_avg_bootstraps.keys())
    n_methods = len(methods)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    if n_methods > len(colors):
        colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    fig, axes = plt.subplots(n_methods, 1, figsize=(8, 1.6 * n_methods),
                             sharex=True)
    if n_methods == 1:
        axes = [axes]

    # Compute shared x range across all methods
    all_vals = []
    for method in methods:
        v = np.array(method_avg_bootstraps[method])
        v = v[~np.isnan(v)]
        all_vals.append(v)
    x_min = min(v.min() for v in all_vals) - 3
    x_max = max(v.max() for v in all_vals) + 3
    x_grid = np.linspace(x_min, x_max, 500)

    for i, (method, ax) in enumerate(zip(reversed(methods), axes)):
        idx = n_methods - 1 - i  # reversed so top row = first method
        vals = np.array(method_avg_bootstraps[method])
        vals = vals[~np.isnan(vals)]
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
        y_at_mean = kde(mean_val)[0]
        ax.axvline(mean_val, color="k", ls="--", lw=0.8, alpha=0.7)
        ax.text(mean_val, y_at_mean * 1.05,
                f"  {mean_val:.1f} [{ci_lo:.1f}, {ci_hi:.1f}]",
                fontsize=8, va="bottom")

        # Style: remove clutter, label on y-axis
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
    print(f"\nRidgeline plot saved to {out_path}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    receptors = get_receptor_list()
    print(f"Found {len(receptors)} receptors: {', '.join(receptors)}\n")

    methods = {
        "DiffDock":       load_diffdock,
        "DOCK3.8":        load_dock38,
        "OG_Affinity":    load_og_affinity,
        "Boltz_Rescore":  load_rescoring,
    }

    method_bootstraps = {}       # {method: {receptor: [logAUC_boot_values]}}
    method_avg_bootstraps = {}   # {method: [avg_logAUC_boot_values]}
    method_tie_stats = {}        # {method: {receptor: tie_stats dict}}
    method_point_est = {}        # {method: {receptor: point-estimate logAUC}}

    for method_name, loader_fn in methods.items():
        print(f"Processing {method_name}...")
        per_rec, per_rec_ties, per_rec_point = process_method(method_name, loader_fn, receptors, rng)
        method_bootstraps[method_name] = per_rec
        method_tie_stats[method_name] = per_rec_ties
        method_point_est[method_name] = per_rec_point

        # Save per-receptor bootstraps
        for rec, vals in per_rec.items():
            out_file = OUT_DIR / f"{method_name}_{rec}_bootstraps.csv"
            pd.DataFrame({"logAUC": vals}).to_csv(out_file, index=False)

        # Compute average logAUC bootstrap
        avg_boots = compute_avg_logauc_bootstrap(per_rec, N_BOOTSTRAPS, rng)
        method_avg_bootstraps[method_name] = avg_boots

        # Save average bootstraps
        avg_file = OUT_DIR / f"{method_name}_avg_bootstraps.csv"
        pd.DataFrame({"avg_logAUC": avg_boots}).to_csv(avg_file, index=False)

        mean_avg = np.nanmean(avg_boots)
        ci_low, ci_high = np.nanpercentile(avg_boots, [2.5, 97.5])
        print(f"  → {method_name} avg logAUC: {mean_avg:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])\n")

    # Save summary table
    summary_rows = []
    for method_name in methods:
        for rec, vals in method_bootstraps[method_name].items():
            ties = method_tie_stats[method_name][rec]
            summary_rows.append({
                "method": method_name,
                "receptor": rec,
                "mean_logAUC": np.nanmean(vals),
                "std_logAUC": np.nanstd(vals),
                "ci_low": np.nanpercentile(vals, 2.5),
                "ci_high": np.nanpercentile(vals, 97.5),
                "point_estimate_logAUC": method_point_est[method_name][rec],
                "n_scores": ties["n"],
                "distinct_score_values": ties["distinct_values"],
                "largest_tie_block": ties["largest_tie_block"],
                "top1pct_tie_fraction": ties["top1pct_tie_fraction"],
                "tied_top1pct_flag": ties["tied_top1pct_flag"],
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary_per_receptor.csv", index=False)

    n_flagged = int(summary_df["tied_top1pct_flag"].sum()) if not summary_df.empty else 0
    if n_flagged:
        print(f"\nWARNING: {n_flagged} (method, receptor) condition(s) have a tied top-1% ranking. "
              f"See tied_top1pct_flag in summary_per_receptor.csv.")

    avg_summary_rows = []
    for method_name in methods:
        vals = method_avg_bootstraps[method_name]
        avg_summary_rows.append({
            "method": method_name,
            "mean_avg_logAUC": np.nanmean(vals),
            "std_avg_logAUC": np.nanstd(vals),
            "ci_low": np.nanpercentile(vals, 2.5),
            "ci_high": np.nanpercentile(vals, 97.5),
        })
    avg_summary_df = pd.DataFrame(avg_summary_rows)
    avg_summary_df.to_csv(OUT_DIR / "summary_avg_logAUC.csv", index=False)
    print("\nSummary tables saved.")

    # ── Ridgeline plot (best-effort; CSVs above are already written) ──
    try:
        ridgeline_plot(
            method_bootstraps,
            method_avg_bootstraps,
            OUT_DIR / "ridgeline_logAUC.png",
        )
    except Exception as e:
        print(f"WARNING: ridgeline plot failed ({e}); CSV outputs above are unaffected.")


if __name__ == "__main__":
    main()
