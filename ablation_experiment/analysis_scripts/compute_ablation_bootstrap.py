#!/usr/bin/env python3
"""
Compute per-receptor logAUC with bootstrapping for each ablation experiment,
then bootstrap the average logAUC across receptors, save results, and produce
a ridgeline plot.

Reads: experiment_files/results/ablation/feature_ablation_results.csv
Score column: affinity_pred_value (lower = better binding)
Active/decoy labels come from the ``is_binder`` column when present (the
DUDEZ runner writes it from the benchmark manifest); otherwise they fall
back to the "ZINC" name-prefix convention. See
``logauc_utils.label_actives_decoys``.
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logauc_utils import (  # noqa: E402
    DEFAULT_BOOTSTRAPS,
    label_actives_decoys,
    MIN_BOOTSTRAPS,
    bca_interval,
    bootstrap_logauc,
    cluster_bootstrap_avg_logauc,
    cluster_bootstrap_paired_difference,
    jackknife_leave_one_out_means,
    point_estimate_logauc,
    tie_stats,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent           # experiment_files/
ABLATION_CSV = BASE_DIR / "results" / "ablation" / "feature_ablation_results.csv"
OUT_DIR = BASE_DIR / "analysis_data" / "ablation_bootstrap_results"

SEED = 42
BASELINE_EXPERIMENT = "baseline"

# logAUC / ROC / bootstrap core now lives in logauc_utils.py (shared with
# compute_logauc_bootstrap.py) so the tie-handling fix only has one place to
# live. See that module's docstring for the bug this replaced.


# ── helpers ──────────────────────────────────────────────────────────────────

def is_decoy(name):
    return name.startswith("ZINC")


def clean_ligand_name(raw):
    """Strip trailing whitespace + 'none' suffix from ligand_name."""
    import re
    return re.sub(r"\s+none$", "", str(raw)).strip()


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


def _paired_receptor_scores(df, exp_a, exp_b, receptors):
    """Build per-receptor (lig_scores, dec_scores, lig_set, dec_set) for two
    experiments, restricted to compounds present in *both* on that receptor
    and in a shared, deterministic order -- required for
    cluster_bootstrap_paired_difference's index-level pairing to line up
    the same compound between arms."""
    out_a, out_b = {}, {}
    df_a_all = df[df["experiment"] == exp_a]
    df_b_all = df[df["experiment"] == exp_b]

    for receptor in receptors:
        best_a = (df_a_all[df_a_all["receptor_id"] == receptor]
                  .groupby("ligand_id")["affinity_pred_value"].min())
        best_b = (df_b_all[df_b_all["receptor_id"] == receptor]
                  .groupby("ligand_id")["affinity_pred_value"].min())
        common = sorted(set(best_a.index) & set(best_b.index))
        if not common:
            continue
        common_df = pd.DataFrame({"ligand_id": common})
        if "is_binder" in df.columns:
            common_df = common_df.merge(
                df[["ligand_id", "is_binder"]].drop_duplicates("ligand_id"),
                on="ligand_id", how="left",
            )
        act, dec, _ = label_actives_decoys(common_df)
        # Preserve `common`'s deterministic order: the paired bootstrap
        # pairs arms by list index, so both arms must see the same compound
        # at the same position.
        lig_ids = [n for n in common if n in act]
        dec_ids = [n for n in common if n in dec]
        if not lig_ids or not dec_ids:
            continue

        out_a[receptor] = (
            [(n, float(best_a[n])) for n in lig_ids],
            [(n, float(best_a[n])) for n in dec_ids],
            set(lig_ids), set(dec_ids),
        )
        out_b[receptor] = (
            [(n, float(best_b[n])) for n in lig_ids],
            [(n, float(best_b[n])) for n in dec_ids],
            set(lig_ids), set(dec_ids),
        )
    return out_a, out_b


# ── parallel workers ─────────────────────────────────────────────────────────
# Each task carries its own SeedSequence child, so the run is reproducible for a
# given --seed no matter how many workers run or what order they finish in.

def _bootstrap_condition(task):
    exp, rec, lig_scores, dec_scores, lig_set, dec_set, n_boot, seed = task
    rng = np.random.default_rng(seed)
    boot_vals = bootstrap_logauc(lig_scores, dec_scores, lig_set, dec_set, n_boot, rng)
    point = point_estimate_logauc(lig_scores, dec_scores, lig_set, dec_set, rng)
    ties = tie_stats(list(lig_scores) + list(dec_scores))
    return exp, rec, boot_vals, point, ties


def _cluster_condition(task):
    exp, receptor_scores, n_boot, seed = task
    rng = np.random.default_rng(seed)
    return exp, cluster_bootstrap_avg_logauc(receptor_scores, n_boot, rng)


def _paired_condition(task):
    exp, paired_a, paired_b, n_boot, seed = task
    rng = np.random.default_rng(seed)
    return exp, cluster_bootstrap_paired_difference(paired_a, paired_b, n_boot, rng)


def _run_pool(fn, work, jobs):
    """Map fn over work in a process pool when jobs > 1, else serially.
    Yields results in completion order; callers key them by name."""
    if jobs > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            yield from ex.map(fn, work, chunksize=1)
    else:
        for w in work:
            yield fn(w)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstraps", type=int, default=DEFAULT_BOOTSTRAPS,
                         help=f"Bootstrap replicates (min {MIN_BOOTSTRAPS}).")
    parser.add_argument(
        "--allow-low-bootstraps",
        action="store_true",
        help=(
            "Allow exploratory runs below the standard minimum. "
            "When set, the minimum is relaxed to 250 replicates."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--baseline-experiment", default=BASELINE_EXPERIMENT,
                         help="Experiment used as the comparison arm for paired differences.")
    parser.add_argument("--jobs", "-j", type=int, default=min(8, os.cpu_count() or 1),
                         help="Worker processes for the bootstrap (default: min(8, ncpu)).")
    args = parser.parse_args()
    min_bootstraps = 250 if args.allow_low_bootstraps else MIN_BOOTSTRAPS
    if args.n_bootstraps < min_bootstraps:
        parser.error(f"--n-bootstraps must be >= {min_bootstraps} (got {args.n_bootstraps})")
    if args.jobs < 1:
        parser.error(f"--jobs must be >= 1 (got {args.jobs})")
    return args


def main():
    args = parse_args()
    n_bootstraps = args.n_bootstraps
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Bootstrap replicates: {n_bootstraps}")
    print("Loading ablation data...")
    df = pd.read_csv(ABLATION_CSV)
    df["ligand_id"] = df["ligand_name"].apply(clean_ligand_name)
    # Drop rows with errors or missing scores
    df = df[df["error"].isna() & df["affinity_pred_value"].notna()]

    experiments = sorted(df["experiment"].unique())
    receptors = sorted(df["receptor_id"].unique())
    print(f"Experiments: {', '.join(experiments)}")
    print(f"Receptors ({len(receptors)}): {', '.join(receptors)}\n")

    experiment_avg_bootstraps = {}   # {experiment: [naive avg_logAUC vals]} (deprecated, kept for the ridgeline plot)

    # Phase A (serial, cheap): dedup to best-per-compound and label actives /
    # decoys for every (experiment, receptor). Phase B does the heavy lifting.
    experiment_receptor_scores = {exp: {} for exp in experiments}  # {exp: {rec: (lig_scores, dec_scores, lig_set, dec_set)}}
    label_sources = {}
    for exp in experiments:
        df_exp = df[df["experiment"] == exp]
        for receptor in receptors:
            df_rec = df_exp[df_exp["receptor_id"] == receptor]
            if df_rec.empty:
                print(f"  [{exp}] {receptor}: no data, skipping")
                continue

            # Deduplicate: keep best (lowest) score per compound
            best = df_rec.groupby("ligand_id")["affinity_pred_value"].min().reset_index()
            # Carry is_binder through the dedup so the authoritative label
            # survives (DUDEZ output); falls back to the ZINC-prefix
            # heuristic for MOL2-runner output that has no such column.
            if "is_binder" in df_rec.columns:
                best = best.merge(
                    df_rec[["ligand_id", "is_binder"]].drop_duplicates("ligand_id"),
                    on="ligand_id", how="left",
                )
            lig_set, dec_set, label_source = label_actives_decoys(best)

            if len(lig_set) == 0 or len(dec_set) == 0:
                print(f"  [{exp}] {receptor}: no ligs ({len(lig_set)}) or decs ({len(dec_set)}), skipping")
                continue

            lig_scores = [(r.ligand_id, r.affinity_pred_value)
                          for r in best[best["ligand_id"].isin(lig_set)].itertuples()]
            dec_scores = [(r.ligand_id, r.affinity_pred_value)
                          for r in best[best["ligand_id"].isin(dec_set)].itertuples()]
            experiment_receptor_scores[exp][receptor] = (lig_scores, dec_scores, lig_set, dec_set)
            label_sources[(exp, receptor)] = label_source

    # Phase B (parallel): per-(experiment, receptor) bootstrap + point estimate
    # + tie stats. A SeedSequence child per condition keeps it reproducible.
    conditions = [(exp, rec) for exp in experiments
                  for rec in sorted(experiment_receptor_scores[exp])]
    child_seeds = np.random.SeedSequence(args.seed).spawn(len(conditions))
    boot_work = [
        (exp, rec, *experiment_receptor_scores[exp][rec], n_bootstraps, seed)
        for (exp, rec), seed in zip(conditions, child_seeds)
    ]

    print(f"\nBootstrapping {len(boot_work)} (experiment, receptor) conditions "
          f"on {args.jobs} process(es)...")
    experiment_bootstraps = {exp: {} for exp in experiments}
    experiment_tie_stats = {exp: {} for exp in experiments}
    experiment_point_est = {exp: {} for exp in experiments}

    for exp, rec, boot_vals, point, ties in _run_pool(_bootstrap_condition, boot_work, args.jobs):
        experiment_bootstraps[exp][rec] = boot_vals
        experiment_tie_stats[exp][rec] = ties
        experiment_point_est[exp][rec] = point
        lig_scores, dec_scores, lig_set, dec_set = experiment_receptor_scores[exp][rec]
        mean_val = np.nanmean(boot_vals)
        tie_flag = "  [TIED TOP-1%]" if ties["tied_top1pct_flag"] else ""
        print(f"  [{exp}] {rec}: mean logAUC = {mean_val:.2f} ({len(lig_set)} ligs, {len(dec_set)} decs, "
              f"labels={label_sources[(exp, rec)]}, "
              f"{ties['distinct_values']} distinct scores, largest tie block {ties['largest_tie_block']}){tie_flag}")
        if ties["tied_top1pct_flag"]:
            print(f"    WARNING: top-1% of the ranking is >=99% inside a single tied score block — "
                  f"logAUC for [{exp}] {rec} is determined by tie-break order, not by the model.")

    # Per-receptor bootstrap CSVs + the deprecated naive avg-bootstrap (kept
    # only to drive the ridgeline plot's KDE). The naive avg is cheap and rides
    # one shared RNG stream, so it stays serial.
    from logauc_utils import compute_avg_logauc_bootstrap as _naive_avg_bootstrap
    naive_rng = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(1)[0])
    for exp in experiments:
        per_rec = experiment_bootstraps[exp]
        for rec, vals in per_rec.items():
            pd.DataFrame({"logAUC": vals}).to_csv(OUT_DIR / f"{exp}_{rec}_bootstraps.csv", index=False)

        naive_avg_boots = _naive_avg_bootstrap(per_rec, n_bootstraps, naive_rng)
        experiment_avg_bootstraps[exp] = np.array(naive_avg_boots)
        pd.DataFrame({"avg_logAUC": naive_avg_boots}).to_csv(
            OUT_DIR / f"{exp}_avg_bootstraps.csv", index=False)
        if len(naive_avg_boots):
            mean_avg = np.nanmean(naive_avg_boots)
            ci_low, ci_high = np.nanpercentile(naive_avg_boots, [2.5, 97.5])
            print(f"  → {exp} avg logAUC: {mean_avg:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")


    # Save summary tables
    summary_rows = []
    for exp in experiments:
        for rec, vals in experiment_bootstraps[exp].items():
            ties = experiment_tie_stats[exp][rec]
            summary_rows.append({
                "experiment": exp,
                "receptor": rec,
                "mean_logAUC": np.nanmean(vals),
                "std_logAUC": np.nanstd(vals),
                "ci_low": np.nanpercentile(vals, 2.5),
                "ci_high": np.nanpercentile(vals, 97.5),
                "point_estimate_logAUC": experiment_point_est[exp][rec],
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
        print(f"\nWARNING: {n_flagged} (experiment, receptor) condition(s) have a tied top-1% ranking. "
              f"See tied_top1pct_flag in summary_per_receptor.csv.")

    # Task 3a/3b: population-level (cluster) CI, properly resampling
    # receptors as well as ligands, with a BCa correction. This is the
    # number that supports "receptors in general" claims; the per-receptor
    # rows in summary_per_receptor.csv are the within-receptor CIs that
    # support "this specific receptor" claims. See logauc_utils.py.
    print("\nComputing population-level cluster bootstrap (Task 3a/3b)...")
    cluster_seeds = np.random.SeedSequence(args.seed + 1).spawn(len(experiments))
    cluster_work = [(exp, experiment_receptor_scores[exp], n_bootstraps, seed)
                    for exp, seed in zip(experiments, cluster_seeds)]
    cluster_boots_by_exp = {exp: boots
                            for exp, boots in _run_pool(_cluster_condition, cluster_work, args.jobs)}

    avg_rows = []
    for exp in experiments:
        cluster_boots = cluster_boots_by_exp[exp]
        point_ests = list(experiment_point_est[exp].values())
        point_estimate = float(np.nanmean(point_ests)) if point_ests else float("nan")
        jackknife_vals = jackknife_leave_one_out_means(point_ests)
        bca_lo, bca_hi = bca_interval(cluster_boots, point_estimate, jackknife_vals)

        cluster_file = OUT_DIR / f"{exp}_cluster_avg_bootstraps.csv"
        pd.DataFrame({"cluster_avg_logAUC": cluster_boots}).to_csv(cluster_file, index=False)

        avg_rows.append({
            "experiment": exp,
            "n_receptors": len(experiment_receptor_scores[exp]),
            "point_estimate_avg_logAUC": point_estimate,
            "mean_cluster_avg_logAUC": float(np.nanmean(cluster_boots)) if cluster_boots else float("nan"),
            "cluster_ci_percentile_low": float(np.nanpercentile(cluster_boots, 2.5)) if cluster_boots else float("nan"),
            "cluster_ci_percentile_high": float(np.nanpercentile(cluster_boots, 97.5)) if cluster_boots else float("nan"),
            "cluster_ci_bca_low": bca_lo,
            "cluster_ci_bca_high": bca_hi,
            # Deprecated naive avg-bootstrap, kept only for continuity with
            # old output; do not use for population-level claims (Task 3a).
            "deprecated_naive_mean_avg_logAUC": float(np.nanmean(experiment_avg_bootstraps[exp])),
            "deprecated_naive_ci_low": float(np.nanpercentile(experiment_avg_bootstraps[exp], 2.5)),
            "deprecated_naive_ci_high": float(np.nanpercentile(experiment_avg_bootstraps[exp], 97.5)),
        })
        print(f"  {exp}: cluster avg logAUC = {avg_rows[-1]['mean_cluster_avg_logAUC']:.2f} "
              f"(BCa 95% CI: [{bca_lo:.2f}, {bca_hi:.2f}], n={len(experiment_receptor_scores[exp])} receptors)")

    pd.DataFrame(avg_rows).to_csv(OUT_DIR / "summary_avg_logAUC.csv", index=False)

    # Task 3c: paired difference vs. baseline for every other experiment,
    # resampling the same receptors/ligands in both arms so receptor
    # difficulty cancels instead of adding noise. This is the correct way to
    # test arm-vs-arm claims -- overlapping independently-bootstrapped CIs do
    # not imply "not significantly different."
    if args.baseline_experiment in experiment_receptor_scores:
        print(f"\nComputing paired differences vs. {args.baseline_experiment!r} (Task 3c)...")
        paired_inputs = []
        for exp in experiments:
            if exp == args.baseline_experiment:
                continue
            paired_a, paired_b = _paired_receptor_scores(df, exp, args.baseline_experiment, receptors)
            if not paired_a:
                continue
            paired_inputs.append((exp, paired_a, paired_b))
        paired_seeds = np.random.SeedSequence(args.seed + 2).spawn(len(paired_inputs))
        paired_work = [(exp, pa, pb, n_bootstraps, seed)
                       for (exp, pa, pb), seed in zip(paired_inputs, paired_seeds)]
        diffs_by_exp = {exp: diffs
                        for exp, diffs in _run_pool(_paired_condition, paired_work, args.jobs)}

        paired_rows = []
        for exp, pa, _pb in paired_inputs:
            diffs = diffs_by_exp.get(exp)
            if not diffs:
                continue
            mean_diff = float(np.nanmean(diffs))
            ci_lo, ci_hi = float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))
            significant = (ci_lo > 0) or (ci_hi < 0)
            paired_rows.append({
                "experiment": exp,
                "baseline_experiment": args.baseline_experiment,
                "n_receptors": len(pa),
                "mean_diff_logAUC": mean_diff,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "significant_95pct": significant,
            })
            print(f"  {exp} - {args.baseline_experiment}: Δ = {mean_diff:.2f} "
                  f"(95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]){' *' if significant else ''}")
        if paired_rows:
            pd.DataFrame(paired_rows).to_csv(OUT_DIR / f"pairwise_vs_{args.baseline_experiment}.csv", index=False)
    else:
        print(f"\nBaseline experiment {args.baseline_experiment!r} not found in data -- "
              f"skipping paired-difference report.")

    print("\nSummary tables saved.")

    # Ridgeline plot -- best-effort. All CSVs above are already written by
    # this point, so a plotting-backend failure shouldn't take down a run
    # that otherwise succeeded.
    try:
        ridgeline_plot(experiment_avg_bootstraps, OUT_DIR / "ridgeline_ablation_logAUC.png")
    except Exception as e:
        print(f"WARNING: ridgeline plot failed ({e}); CSV outputs above are unaffected.")


if __name__ == "__main__":
    main()
