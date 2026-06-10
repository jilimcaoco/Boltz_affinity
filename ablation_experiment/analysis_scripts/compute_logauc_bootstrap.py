#!/usr/bin/env python3
"""
Compute per-receptor logAUC with bootstrapping (100 replicates) for 4 scoring
methods, then compute the average logAUC across receptors (also bootstrapped),
and display as a ridgeline plot.

Scoring methods:
  1. diffdock_scores  – .tsv files, space-separated, col6 (0-indexed: 5) as score
  2. dock3.8_scores   – CSV with ligand_id,dock_score (take min per ligand)
  3. OG_affinity_scores – master_scores.csv, negate Affinity Pred Value (higher=better → lower=better for ranking)
  4. rescoring (Boltz) – *_rescored.csv, affinity_score column

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

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent           # experiment_files/
DIFFDOCK_DIR = BASE_DIR / "analysis_data" / "diffdock_scores"
DOCK38_DIR   = BASE_DIR / "analysis_data" / "dock3.8_scores"
OG_DIR       = BASE_DIR / "analysis_data" / "OG_affinity_scores"
RESCORE_DIR  = BASE_DIR / "results" / "rescoring"
OUT_DIR      = BASE_DIR / "analysis_data" / "bootstrap_results"

N_BOOTSTRAPS = 100
SEED = 42

# ── logAUC calculation (from bootstrap_tldr.py) ─────────────────────────────

def do_roc(scores, lig_set, decoy_set, nbins=10000):
    """Compute ROC curve points from a ranked list of (name, score) tuples.
    scores must be sorted ascending by score (best first = most negative)."""
    num_data = len(scores)
    binsize = max(int(num_data / nbins), 1)
    num_lig = len(lig_set)
    num_dec = len(decoy_set)
    if num_lig == 0 or num_dec == 0:
        return None

    found_ligand = 0
    results = []
    for i in range(num_data):
        if i % binsize == 0:
            results.append([i - found_ligand, found_ligand])
        if scores[i][0] in lig_set:
            found_ligand += 1
    results.append([num_data - found_ligand, found_ligand])
    results.append([num_dec, num_lig])

    points = []
    for x in results:
        fpr = x[0] * 100.0 / num_dec
        tpr = x[1] * 100.0 / num_lig
        points.append([fpr, tpr])
    return points


def interpolate_curve(points):
    i = 0
    while i < len(points) and points[i][0] < 0.1:
        i += 1
    if i == 0:
        return points
    slope = (points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0] + 1e-12)
    intercept = points[i][1] - slope * points[i][0]
    point_one = [0.100001, slope * 0.100001 + intercept]
    npoints = list(points)
    npoints.insert(i, point_one)
    return npoints


def logAUC(points):
    LOGAUC_MAX = 1.0
    LOGAUC_MIN = 0.001
    RANDOM_LOGAUC = (LOGAUC_MAX - LOGAUC_MIN) / np.log(10) / np.log10(LOGAUC_MAX / LOGAUC_MIN)

    npoints = []
    for x in points:
        if (x[0] >= LOGAUC_MIN * 100) and (x[0] <= LOGAUC_MAX * 100):
            npoints.append([x[0] / 100, x[1] / 100])

    area = 0.0
    for point2, point1 in zip(npoints[1:], npoints[:-1]):
        if point2[0] - point1[0] < 0.000001:
            continue
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        intercept = point2[1] - (dy) / (dx) * point2[0]
        area += dy / np.log(10) + intercept * (np.log10(point2[0]) - np.log10(point1[0]))

    return area / np.log10(LOGAUC_MAX / LOGAUC_MIN) - RANDOM_LOGAUC


def compute_logauc(ranked_list, lig_set, decoy_set):
    """Given a ranked list of (name, score), compute logAUC * 100."""
    points = do_roc(ranked_list, lig_set, decoy_set)
    if points is None:
        return np.nan
    points = interpolate_curve(points)
    return logAUC(points) * 100


def bootstrap_logauc(lig_scores, decoy_scores, lig_set, decoy_set, n_boot, rng):
    """Bootstrap logAUC by resampling ligands and decoys separately."""
    results = []
    for _ in range(n_boot):
        boot_lig_idx = rng.integers(0, len(lig_scores), size=len(lig_scores))
        boot_dec_idx = rng.integers(0, len(decoy_scores), size=len(decoy_scores))
        boot_scores = [lig_scores[i] for i in boot_lig_idx] + [decoy_scores[i] for i in boot_dec_idx]
        boot_scores.sort(key=lambda x: x[1])
        ranked = [(s[0], s[1]) for s in boot_scores]
        val = compute_logauc(ranked, lig_set, decoy_set)
        results.append(val)
    return results


# ── Data loaders ─────────────────────────────────────────────────────────────

def is_decoy(name):
    return name.startswith("ZINC")


def split_lig_dec(names):
    ligs = {n for n in names if not is_decoy(n)}
    decs = {n for n in names if is_decoy(n)}
    return ligs, decs


def load_diffdock(receptor):
    """Load diffdock scores: .tsv, space-separated, col2=name, col6=score."""
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
                score = float(parts[5])
            except ValueError:
                continue
            records.append((name, score))
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
    """Load OG affinity scores for a receptor. Negate Affinity Pred Value
    so that higher affinity → lower (better) ranking score."""
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
    """For a given method, compute per-receptor bootstrapped logAUC."""
    all_bootstraps = {}
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
        print(f"  [{method_name}] {receptor}: mean logAUC = {mean_val:.2f} ({len(lig_set)} ligs, {len(dec_set)} decs)")

    return all_bootstraps


def compute_avg_logauc_bootstrap(per_receptor_boots, n_boot, rng):
    """Bootstrap the average logAUC across receptors.
    For each bootstrap replicate, sample one logAUC from each receptor's
    per-receptor bootstrap distribution, then average."""
    receptors = list(per_receptor_boots.keys())
    if not receptors:
        return []
    avg_boots = []
    for _ in range(n_boot):
        sampled = []
        for rec in receptors:
            vals = per_receptor_boots[rec]
            idx = rng.integers(0, len(vals))
            sampled.append(vals[idx])
        avg_boots.append(np.nanmean(sampled))
    return avg_boots


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

    for method_name, loader_fn in methods.items():
        print(f"Processing {method_name}...")
        per_rec = process_method(method_name, loader_fn, receptors, rng)
        method_bootstraps[method_name] = per_rec

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
            summary_rows.append({
                "method": method_name,
                "receptor": rec,
                "mean_logAUC": np.nanmean(vals),
                "std_logAUC": np.nanstd(vals),
                "ci_low": np.nanpercentile(vals, 2.5),
                "ci_high": np.nanpercentile(vals, 97.5),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary_per_receptor.csv", index=False)

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

    # ── Ridgeline plot ──
    ridgeline_plot(
        method_bootstraps,
        method_avg_bootstraps,
        OUT_DIR / "ridgeline_logAUC.png",
    )


if __name__ == "__main__":
    main()
