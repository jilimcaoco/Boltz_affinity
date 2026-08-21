#!/usr/bin/env python3
"""Generate a DUDEZ ablation progress-report bundle.

Creates a timestamped report directory with:
- markdown write-up of completed vs pending work
- copied raw bootstrap CSV outputs
- additional summary plots focused on progress + interpretation
"""

from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_CSV = BASE_DIR / "results" / "ablation" / "feature_ablation_results.csv"
DEFAULT_BOOT_DIR = BASE_DIR / "analysis_data" / "ablation_bootstrap_results"
DEFAULT_ANALYSIS_DIR = BASE_DIR / "analysis_data"
DEFAULT_REPORT_ROOT = BASE_DIR / "progress_reports"
DEFAULT_MANIFEST = BASE_DIR / "results" / "dudez_ablation" / "_receptor_manifest.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    p.add_argument("--boot-dir", type=Path, default=DEFAULT_BOOT_DIR)
    p.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--tag", default=f"{date.today().isoformat()}_dudez_ablation_progress")
    return p.parse_args()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _load_all_experiments() -> tuple[list[str], list[str], list[str]]:
    # Keep this explicit and import-free so report generation does not depend
    # on heavyweight model modules during CPU analysis runs.
    base = [
        "baseline", "no_distogram", "no_z_trunk", "no_s_inputs",
        "distogram_only", "z_trunk_only", "s_inputs_only", "bias_only",
        "no_atom_encoder", "no_msa_profile", "no_res_type",
        "only_atom_encoder", "only_msa_profile", "only_res_type",
    ]

    top_level_channels = ("distogram", "z_trunk", "s_inputs")
    factorial_kept = {
        "baseline": ("distogram", "z_trunk", "s_inputs"),
        "no_distogram": ("z_trunk", "s_inputs"),
        "no_z_trunk": ("distogram", "s_inputs"),
        "no_s_inputs": ("distogram", "z_trunk"),
        "distogram_only": ("distogram",),
        "z_trunk_only": ("z_trunk",),
        "s_inputs_only": ("s_inputs",),
        "bias_only": (),
    }
    donor_seeds = (11, 22, 33, 44, 55)

    resample_and_mean = []
    for cell_name, kept in factorial_kept.items():
        ablated = [c for c in top_level_channels if c not in kept]
        if not ablated:
            continue
        for seed in donor_seeds:
            resample_and_mean.append(f"{cell_name}__resample__d{seed}")
        resample_and_mean.append(f"{cell_name}__mean")

    noise = []
    for sigma in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        sigma_tag = f"{sigma:g}".replace(".", "p")
        for seed in (101, 202, 303):
            noise.append(f"lig_noise_{sigma_tag}_s{seed}")
    for sigma in (1.0, 5.0):
        sigma_tag = f"{sigma:g}".replace(".", "p")
        for seed in (101, 202, 303):
            noise.append(f"rec_noise_{sigma_tag}_s{seed}")
            noise.append(f"all_noise_{sigma_tag}_s{seed}")

    all_exps = base + resample_and_mean + noise
    default_exps = base + noise
    dudez_defaults = base
    return all_exps, default_exps, dudez_defaults


def _copy_bootstrap_csvs(boot_dir: Path, out_raw: Path) -> int:
    out_raw.mkdir(parents=True, exist_ok=True)
    n = 0
    for csv in sorted(boot_dir.glob("*.csv")):
        shutil.copy2(csv, out_raw / csv.name)
        n += 1
    return n


def _plot_experiment_mean_with_ci(summary_avg: pd.DataFrame, out_path: Path) -> bool:
    if summary_avg.empty:
        return False
    if {"experiment", "point_estimate_avg_logAUC", "cluster_ci_bca_low", "cluster_ci_bca_high"}.issubset(summary_avg.columns):
        df = summary_avg[["experiment", "point_estimate_avg_logAUC", "cluster_ci_bca_low", "cluster_ci_bca_high"]].copy()
    elif {"experiment", "mean_avg_logAUC", "ci_low", "ci_high"}.issubset(summary_avg.columns):
        df = summary_avg[["experiment", "mean_avg_logAUC", "ci_low", "ci_high"]].copy()
        df = df.rename(columns={
            "mean_avg_logAUC": "point_estimate_avg_logAUC",
            "ci_low": "cluster_ci_bca_low",
            "ci_high": "cluster_ci_bca_high",
        })
    else:
        return False
    df = df.sort_values("point_estimate_avg_logAUC", ascending=True)

    y = np.arange(len(df))
    vals = df["point_estimate_avg_logAUC"].to_numpy(dtype=float)
    lo = vals - df["cluster_ci_bca_low"].to_numpy(dtype=float)
    hi = df["cluster_ci_bca_high"].to_numpy(dtype=float) - vals

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df) + 1.2)))
    ax.barh(y, vals, color="#4C78A8", alpha=0.85)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor="#2f2f2f", capsize=3, lw=1.2)
    ax.axvline(0, color="#666666", lw=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["experiment"].tolist(), fontsize=8)
    ax.set_xlabel("Average logAUC (point estimate) with BCa 95% CI")
    ax.set_title("Ablation Conditions Ranked by Average logAUC")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_delta_vs_baseline(summary_per_receptor: pd.DataFrame, out_path: Path) -> bool:
    if summary_per_receptor.empty:
        return False
    if {"receptor", "experiment", "point_estimate_logAUC"}.issubset(summary_per_receptor.columns):
        value_col = "point_estimate_logAUC"
    elif {"receptor", "experiment", "mean_logAUC"}.issubset(summary_per_receptor.columns):
        value_col = "mean_logAUC"
    else:
        return False

    pivot = summary_per_receptor.pivot_table(index="receptor", columns="experiment", values=value_col, aggfunc="mean")
    if "baseline" not in pivot.columns:
        return False

    delta = pivot.sub(pivot["baseline"], axis=0)
    if delta.shape[1] <= 1:
        return False

    delta = delta.drop(columns=["baseline"], errors="ignore")
    order = delta.mean(axis=0).sort_values(ascending=False).index.tolist()
    delta = delta[order]

    v = np.nanpercentile(np.abs(delta.to_numpy(dtype=float)), 95)
    if not np.isfinite(v) or v <= 0:
        v = 1.0

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * delta.shape[1] + 3), max(6, 0.25 * delta.shape[0] + 2)))
    im = ax.imshow(delta.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(np.arange(delta.shape[1]))
    ax.set_xticklabels(delta.columns.tolist(), rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(delta.shape[0]))
    ax.set_yticklabels(delta.index.tolist(), fontsize=7)
    ax.set_title("Per-Receptor Delta logAUC vs Baseline")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Delta logAUC")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_receptor_completeness(results_df: pd.DataFrame, manifest_receptors: list[str], out_path: Path) -> bool:
    if results_df.empty or not manifest_receptors:
        return False
    need = {"receptor_id", "experiment", "affinity_pred_value", "error"}
    if not need.issubset(results_df.columns):
        return False

    valid = results_df[(results_df["error"].isna()) & (results_df["affinity_pred_value"].notna())]
    counts = valid.groupby("receptor_id")["experiment"].nunique().to_dict()
    vals = [counts.get(r, 0) for r in manifest_receptors]

    fig, ax = plt.subplots(figsize=(max(10, 0.22 * len(manifest_receptors) + 3), 4.5))
    ax.bar(np.arange(len(manifest_receptors)), vals, color="#59A14F", alpha=0.9)
    ax.set_xticks(np.arange(len(manifest_receptors)))
    ax.set_xticklabels(manifest_receptors, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Completed experiments per receptor")
    ax.set_title("Experiment Completeness Across Receptors")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()

    report_dir = args.report_root / args.tag
    raw_dir = report_dir / "raw_bootstrap_data"
    graph_dir = report_dir / "graphs"
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    results_df = _safe_read_csv(args.results_csv)
    summary_per_receptor = _safe_read_csv(args.boot_dir / "summary_per_receptor.csv")
    summary_avg = _safe_read_csv(args.boot_dir / "summary_avg_logAUC.csv")
    pairwise = _safe_read_csv(args.boot_dir / "pairwise_vs_baseline.csv")
    tie_audit = _safe_read_csv(args.analysis_dir / "tie_density_audit.csv")

    manifest_receptors = []
    if args.manifest.exists():
        manifest_receptors = [x.strip() for x in args.manifest.read_text().splitlines() if x.strip()]

    all_exps, default_exps, dudez_defaults = _load_all_experiments()

    copied_csvs = _copy_bootstrap_csvs(args.boot_dir, raw_dir)

    upstream_graph_dir = args.analysis_dir / "graphs"
    if upstream_graph_dir.exists():
        dst = graph_dir / "upstream_ablation_summary"
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(upstream_graph_dir.glob("*.png")):
            shutil.copy2(p, dst / p.name)

    made_graphs = []
    if _plot_experiment_mean_with_ci(summary_avg, graph_dir / "avg_logauc_bca95_ranked.png"):
        made_graphs.append("avg_logauc_bca95_ranked.png")
    if _plot_delta_vs_baseline(summary_per_receptor, graph_dir / "delta_vs_baseline_heatmap.png"):
        made_graphs.append("delta_vs_baseline_heatmap.png")
    if _plot_receptor_completeness(results_df, manifest_receptors, graph_dir / "receptor_experiment_completeness.png"):
        made_graphs.append("receptor_experiment_completeness.png")

    done_receptors = sorted(results_df["receptor_id"].dropna().unique().tolist()) if not results_df.empty else []
    done_experiments = sorted(results_df["experiment"].dropna().unique().tolist()) if not results_df.empty else []

    pending_receptors = [r for r in manifest_receptors if r not in set(done_receptors)]
    pending_dudez_defaults = [e for e in dudez_defaults if e not in set(done_experiments)]
    pending_all_defined = [e for e in all_exps if e not in set(done_experiments)]

    top_block = pd.DataFrame()
    if not summary_avg.empty:
        if "point_estimate_avg_logAUC" in summary_avg.columns:
            top_block = summary_avg.sort_values("point_estimate_avg_logAUC", ascending=False).head(8)
        elif "mean_avg_logAUC" in summary_avg.columns:
            top_block = summary_avg.rename(columns={"mean_avg_logAUC": "point_estimate_avg_logAUC"})
            top_block = top_block.sort_values("point_estimate_avg_logAUC", ascending=False).head(8)

    sig_vs_baseline = pd.DataFrame()
    if not pairwise.empty and "significant_95pct" in pairwise.columns:
        sig_vs_baseline = pairwise[pairwise["significant_95pct"] == True]  # noqa: E712

    tie_flags = 0
    if not summary_per_receptor.empty and "tied_top1pct_flag" in summary_per_receptor.columns:
        tie_flags = int(summary_per_receptor["tied_top1pct_flag"].sum())

    tie_unusable = 0
    if not tie_audit.empty and "status" in tie_audit.columns:
        tie_unusable = int((tie_audit["status"] == "unusable").sum())

    md = []
    md.append(f"# Ablation Progress Report ({date.today().isoformat()})")
    md.append("")
    md.append("## Scope")
    md.append("This report summarizes the current DUDEZ ablation status, newly recomputed bootstrap logAUC outputs (n=250), and additional progress-focused visual diagnostics.")
    md.append("")

    md.append("## Completion Status")
    md.append(f"- Receptors in manifest: {len(manifest_receptors)}")
    md.append(f"- Receptors present in combined results: {len(done_receptors)}")
    md.append(f"- Missing receptors: {len(pending_receptors)}")
    if pending_receptors:
        md.append(f"- Missing receptor IDs: {', '.join(pending_receptors)}")
    md.append(f"- Experiments observed in combined results: {len(done_experiments)}")
    md.append(f"- DUDEZ default experiments defined in code: {len(dudez_defaults)}")
    md.append(f"- Missing DUDEZ-default experiments: {len(pending_dudez_defaults)}")
    if pending_dudez_defaults:
        md.append(f"- Pending DUDEZ-default experiments: {', '.join(pending_dudez_defaults[:30])}")
    md.append(f"- All experiments defined in registry (includes pose-noise/resample/mean): {len(all_exps)}")
    md.append(f"- Defined-but-not-run experiments: {len(pending_all_defined)}")
    md.append("")

    md.append("## What Is Done vs Not Done")
    md.append("Done:")
    md.append("- Per-receptor DUDEZ ablation outputs have been merged into one analysis table.")
    md.append("- logAUC bootstrapping has been recomputed with n=250 and parallel CPU workers.")
    md.append("- Per-receptor and cluster-level summaries were generated, including BCa confidence intervals.")
    md.append("- Pairwise experiment-vs-baseline bootstrap differences were generated.")
    md.append("- Additional progress plots were generated for interpretability and planning.")
    md.append("Not done / pending:")
    md.append("- Any experiment conditions that are registry-defined but absent from the combined table remain pending (typically pose-noise + resample/mean variants unless explicitly run).")
    md.append("- If tie-audit marks conditions as unusable, those conditions need recomputation before interpretation.")
    md.append("")

    md.append("## Result Overview")
    md.append(f"- Conditions with tied top-1% ranking flag: {tie_flags}")
    md.append(f"- Tie-audit rows marked unusable: {tie_unusable}")
    md.append(f"- Significant pairwise differences vs baseline (95% bootstrap CI excludes 0): {len(sig_vs_baseline)}")
    if not top_block.empty:
        md.append("- Top experiments by point-estimate average logAUC:")
        for row in top_block.itertuples(index=False):
            if hasattr(row, "cluster_ci_bca_low") and hasattr(row, "cluster_ci_bca_high"):
                md.append(
                    f"  - {row.experiment}: {row.point_estimate_avg_logAUC:.2f} "
                    f"(BCa 95% CI [{row.cluster_ci_bca_low:.2f}, {row.cluster_ci_bca_high:.2f}])"
                )
            else:
                lo = getattr(row, "ci_low", float("nan"))
                hi = getattr(row, "ci_high", float("nan"))
                md.append(
                    f"  - {row.experiment}: {row.point_estimate_avg_logAUC:.2f} "
                    f"(95% CI [{lo:.2f}, {hi:.2f}])"
                )
    md.append("")

    md.append("## Additional Informative Metrics/Graphs Added")
    md.append("- Ranked average logAUC with BCa 95% CIs to compare experimental conditions directly.")
    md.append("- Receptor-by-experiment heatmap of delta logAUC vs baseline to expose target-specific heterogeneity.")
    md.append("- Receptor completeness chart (number of completed experiments per receptor) to track execution gaps.")
    md.append("")

    md.append("## Deliverables in This Directory")
    md.append(f"- Raw bootstrap CSV files copied: {copied_csvs}")
    md.append(f"- Graphs generated: {', '.join(made_graphs) if made_graphs else 'none'}")
    md.append("- Upstream plot directory: analysis_data/graphs (if generated by plot_ablation_summary.py)")
    md.append("")

    md.append("## Notes")
    md.append("Interpretation should prioritize cluster-bootstrap BCa intervals and paired-difference intervals over overlap/non-overlap of independent per-condition CIs.")
    md.append("")

    (report_dir / "progress_report.md").write_text("\n".join(md))

    # Keep a compact machine-readable summary for downstream automation.
    summary = {
        "report_dir": str(report_dir),
        "n_manifest_receptors": len(manifest_receptors),
        "n_done_receptors": len(done_receptors),
        "n_missing_receptors": len(pending_receptors),
        "n_done_experiments": len(done_experiments),
        "n_dudez_default_defined": len(dudez_defaults),
        "n_pending_dudez_default": len(pending_dudez_defaults),
        "n_all_defined": len(all_exps),
        "n_pending_all_defined": len(pending_all_defined),
        "n_copied_bootstrap_csv": copied_csvs,
        "graphs": made_graphs,
    }
    pd.DataFrame([summary]).to_csv(report_dir / "report_summary.csv", index=False)

    print(f"Report written: {report_dir}")
    print(f"Raw bootstrap CSVs copied: {copied_csvs}")
    print(f"Graphs: {', '.join(made_graphs) if made_graphs else 'none'}")


if __name__ == "__main__":
    main()
