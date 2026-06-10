#!/usr/bin/env python3
"""
Compute per-receptor logAUC with bootstrapping for each ablation experiment,
then bootstrap the average logAUC across receptors, save results, and produce
a ridgeline plot.

Reads: experiment_files/results/ablation/feature_ablation_results.csv
Score column: affinity_pred_value (lower = better binding)
Convention: ZINC-prefixed compound IDs = decoys; everything else = ligands.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent           # experiment_files/
ABLATION_CSV = BASE_DIR / "results" / "ablation" / "feature_ablation_results.csv"
OUT_DIR = BASE_DIR / "analysis_data" / "ablation_bootstrap_results"

N_BOOTSTRAPS = 100
SEED = 42

# ── logAUC calculation (same as compute_logauc_bootstrap.py) ─────────────────

def do_roc(scores, lig_set, decoy_set, nbins=10000):
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
    points = do_roc(ranked_list, lig_set, decoy_set)
    if points is None:
        return np.nan
    points = interpolate_curve(points)
    return logAUC(points) * 100


def bootstrap_logauc(lig_scores, decoy_scores, lig_set, decoy_set, n_boot, rng):
    results = []
    for _ in range(n_boot):
        boot_lig_idx = rng.integers(0, len(lig_scores), size=len(lig_scores))
        boot_dec_idx = rng.integers(0, len(decoy_scores), size=len(decoy_scores))
        boot_scores = [lig_scores[i] for i in boot_lig_idx] + [decoy_scores[i] for i in boot_dec_idx]
        boot_scores.sort(key=lambda x: x[1])
        val = compute_logauc(boot_scores, lig_set, decoy_set)
        results.append(val)
    return results


# ── helpers ──────────────────────────────────────────────────────────────────

def is_decoy(name):
    return name.startswith("ZINC")


def clean_ligand_name(raw):
    """Strip trailing whitespace + 'none' suffix from ligand_name."""
    import re
    return re.sub(r"\s+none$", "", str(raw)).strip()


def compute_avg_logauc_bootstrap(per_receptor_boots, n_boot, rng):
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


# ── ridgeline plot ───────────────────────────────────────────────────────────

def ridgeline_plot(experiment_avg_bootstraps, out_path):
    """Ridgeline plot — each ablation experiment on its own y-axis row."""
    experiments = list(experiment_avg_bootstraps.keys())
    n_exp = len(experiments)

    cmap = plt.cm.tab20(np.linspace(0, 1, max(n_exp, 1)))

    fig, axes = plt.subplots(n_exp, 1, figsize=(10, 1.4 * n_exp), sharex=True)
    if n_exp == 1:
        axes = [axes]

    # Shared x range
    all_vals = [experiment_avg_bootstraps[e] for e in experiments]
    x_min = min(v.min() for v in all_vals) - 3
    x_max = max(v.max() for v in all_vals) + 3
    x_grid = np.linspace(x_min, x_max, 500)

    for i, (exp, ax) in enumerate(zip(reversed(experiments), axes)):
        idx = n_exp - 1 - i
        vals = experiment_avg_bootstraps[exp]

        if len(vals) < 2:
            ax.set_ylabel(exp, fontsize=9, rotation=0, labelpad=110, va="center")
            continue

        kde = gaussian_kde(vals)
        y = kde(x_grid)

        ax.fill_between(x_grid, y, alpha=0.6, color=cmap[idx % len(cmap)])
        ax.plot(x_grid, y, color=cmap[idx % len(cmap)], lw=1.5)

        mean_val = np.mean(vals)
        ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
        ax.axvline(mean_val, color="k", ls="--", lw=0.8, alpha=0.7)
        y_at_mean = kde(mean_val)[0]
        ax.text(mean_val, y_at_mean * 1.05,
                f"  {mean_val:.1f} [{ci_lo:.1f}, {ci_hi:.1f}]",
                fontsize=7, va="bottom")

        ax.set_ylabel(exp, fontsize=9, rotation=0, labelpad=110, va="center")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if i < n_exp - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)

    axes[-1].set_xlabel("Average logAUC", fontsize=12)
    fig.suptitle("Feature Ablation — Bootstrapped Average logAUC", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nRidgeline plot saved to {out_path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading ablation data...")
    df = pd.read_csv(ABLATION_CSV)
    df["ligand_id"] = df["ligand_name"].apply(clean_ligand_name)
    # Drop rows with errors or missing scores
    df = df[df["error"].isna() & df["affinity_pred_value"].notna()]

    experiments = sorted(df["experiment"].unique())
    receptors = sorted(df["receptor_id"].unique())
    print(f"Experiments: {', '.join(experiments)}")
    print(f"Receptors ({len(receptors)}): {', '.join(receptors)}\n")

    experiment_bootstraps = {}      # {experiment: {receptor: [logAUC vals]}}
    experiment_avg_bootstraps = {}  # {experiment: [avg_logAUC vals]}

    for exp in experiments:
        print(f"Processing {exp}...")
        df_exp = df[df["experiment"] == exp]
        per_rec = {}

        for receptor in receptors:
            df_rec = df_exp[df_exp["receptor_id"] == receptor]
            if df_rec.empty:
                print(f"  [{exp}] {receptor}: no data, skipping")
                continue

            # Deduplicate: keep best (lowest) score per compound
            best = df_rec.groupby("ligand_id")["affinity_pred_value"].min().reset_index()
            names = set(best["ligand_id"])
            lig_set = {n for n in names if not is_decoy(n)}
            dec_set = {n for n in names if is_decoy(n)}

            if len(lig_set) == 0 or len(dec_set) == 0:
                print(f"  [{exp}] {receptor}: no ligs ({len(lig_set)}) or decs ({len(dec_set)}), skipping")
                continue

            lig_scores = [(r.ligand_id, r.affinity_pred_value)
                          for r in best[best["ligand_id"].isin(lig_set)].itertuples()]
            dec_scores = [(r.ligand_id, r.affinity_pred_value)
                          for r in best[best["ligand_id"].isin(dec_set)].itertuples()]

            boot_vals = bootstrap_logauc(lig_scores, dec_scores, lig_set, dec_set, N_BOOTSTRAPS, rng)
            per_rec[receptor] = boot_vals
            mean_val = np.nanmean(boot_vals)
            print(f"  [{exp}] {receptor}: mean logAUC = {mean_val:.2f} ({len(lig_set)} ligs, {len(dec_set)} decs)")

        experiment_bootstraps[exp] = per_rec

        # Save per-receptor bootstraps
        for rec, vals in per_rec.items():
            out_file = OUT_DIR / f"{exp}_{rec}_bootstraps.csv"
            pd.DataFrame({"logAUC": vals}).to_csv(out_file, index=False)

        # Bootstrap average across receptors
        avg_boots = compute_avg_logauc_bootstrap(per_rec, N_BOOTSTRAPS, rng)
        experiment_avg_bootstraps[exp] = np.array(avg_boots)

        avg_file = OUT_DIR / f"{exp}_avg_bootstraps.csv"
        pd.DataFrame({"avg_logAUC": avg_boots}).to_csv(avg_file, index=False)

        mean_avg = np.nanmean(avg_boots)
        ci_low, ci_high = np.nanpercentile(avg_boots, [2.5, 97.5])
        print(f"  → {exp} avg logAUC: {mean_avg:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])\n")

    # Save summary tables
    summary_rows = []
    for exp in experiments:
        for rec, vals in experiment_bootstraps[exp].items():
            summary_rows.append({
                "experiment": exp,
                "receptor": rec,
                "mean_logAUC": np.nanmean(vals),
                "std_logAUC": np.nanstd(vals),
                "ci_low": np.nanpercentile(vals, 2.5),
                "ci_high": np.nanpercentile(vals, 97.5),
            })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary_per_receptor.csv", index=False)

    avg_rows = []
    for exp in experiments:
        vals = experiment_avg_bootstraps[exp]
        avg_rows.append({
            "experiment": exp,
            "mean_avg_logAUC": np.nanmean(vals),
            "std_avg_logAUC": np.nanstd(vals),
            "ci_low": np.nanpercentile(vals, 2.5),
            "ci_high": np.nanpercentile(vals, 97.5),
        })
    pd.DataFrame(avg_rows).to_csv(OUT_DIR / "summary_avg_logAUC.csv", index=False)
    print("Summary tables saved.")

    # Ridgeline plot
    ridgeline_plot(experiment_avg_bootstraps, OUT_DIR / "ridgeline_ablation_logAUC.png")


if __name__ == "__main__":
    main()
