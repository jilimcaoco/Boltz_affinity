#!/usr/bin/env python3
"""Task 0b/0c — tie-density audit.

Before any ablation number is trusted or any Shapley/factorial analysis is
run, this script reports, for every (experiment, receptor) pair in the
feature-ablation results, how many distinct score values are present, how
large the biggest tie block is, and what fraction of the top 1% of the
ranking sits inside a single tie block. Zero-heavy conditions (``bias_only``,
the ``*_only`` variants) are the ones expected to be tie-dense — this script
verifies that rather than assuming it, and flags anything that fails the
thresholds below so it can be re-run under the fixed tie-aware bootstrap
(see ``logauc_utils.py``) before being interpreted.

Thresholds (Task 0b):
  - fewer than 10 distinct affinity_pred_value   -> "unusable"
  - fewer than 100 distinct affinity_pred_value, *and* fewer distinct values
    than compounds (i.e. real duplication present) -> "suspect". The second
    condition matters: on a receptor with only 80 compounds, 80 distinct
    scores means zero ties and is perfectly fine, but would trip a bare
    "<100 distinct" test purely for being a small receptor. See classify().
  - top-1% of the ranking >=99% inside one tie block -> flagged regardless
    of distinct-value count (this is the direct failure mode of the bug).

Task 0c extension: if the four method-comparison data sources used by
compute_logauc_bootstrap.py (DiffDock, DOCK3.8, OG_Affinity, Boltz_Rescore)
are present on disk, they are audited the same way and appended to the same
output table with source="method:<name>", since Boltz-2 binder-probability-
style scores can saturate and produce the same tie artifact.

Usage
-----
python audit_tie_density.py
python audit_tie_density.py --ablation-csv /path/to/feature_ablation_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logauc_utils import tie_stats  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent  # ablation_experiment/
DEFAULT_ABLATION_CSV = BASE_DIR / "results" / "ablation" / "feature_ablation_results.csv"
OUT_DIR = BASE_DIR / "analysis_data"
OUT_CSV = OUT_DIR / "tie_density_audit.csv"

SUSPECT_THRESHOLD = 100
UNUSABLE_THRESHOLD = 10


def is_decoy(name: str) -> bool:
    return str(name).startswith("ZINC")


def clean_ligand_name(raw) -> str:
    import re
    return re.sub(r"\s+none$", "", str(raw)).strip()


def classify(distinct_values: int, tied_top1pct_flag: bool, n_scores: int) -> str:
    """Classify one condition's tie density.

    The spec's thresholds are absolute (<100 distinct = suspect, <10 =
    unusable), which is the right proxy for "the ranking is too coarse to
    trust" when a receptor has many hundreds of compounds. Applied blindly
    they misfire on small receptors: a condition with 80 distinct scores
    across 80 compounds has *zero* ties and is perfectly well-behaved, yet
    trips a bare `distinct < 100` test purely because the receptor is small.

    So the suspect rule additionally requires real duplication
    (``distinct_values < n_scores``). ``unusable`` stays absolute -- a
    ranking with under 10 distinct levels is uninterpretable no matter how
    few compounds produced it -- and a tied top-1% is always at least
    suspect, since that is the direct failure mode of the tie bug.
    """
    if distinct_values < UNUSABLE_THRESHOLD:
        return "unusable"
    if tied_top1pct_flag:
        return "suspect"
    has_ties = distinct_values < n_scores
    if has_ties and distinct_values < SUSPECT_THRESHOLD:
        return "suspect"
    return "ok"


def audit_ablation_csv(ablation_csv: Path) -> list[dict]:
    if not ablation_csv.exists():
        raise FileNotFoundError(
            f"Ablation results CSV not found at {ablation_csv}. This audit needs the "
            f"real feature_ablation_results.csv produced by run_feature_ablation.py "
            f"(generated on the training cluster, not checked into this repo) — copy "
            f"it here or pass --ablation-csv before interpreting any ablation logAUC."
        )

    df = pd.read_csv(ablation_csv)
    df["ligand_id"] = df["ligand_name"].apply(clean_ligand_name)
    df = df[df["error"].isna() & df["affinity_pred_value"].notna()]

    rows = []
    for (exp, receptor), df_group in df.groupby(["experiment", "receptor_id"]):
        # Same dedup rule as the bootstrap scripts: best (lowest) score per compound.
        best = df_group.groupby("ligand_id")["affinity_pred_value"].min().reset_index()
        names = set(best["ligand_id"])
        lig_set = {n for n in names if not is_decoy(n)}
        dec_set = {n for n in names if is_decoy(n)}

        scores = list(zip(best["ligand_id"], best["affinity_pred_value"]))
        stats = tie_stats(scores)
        status = classify(stats["distinct_values"], stats["tied_top1pct_flag"], stats["n"])

        rows.append({
            "source": "ablation",
            "condition": exp,
            "receptor": receptor,
            "n_ligands": len(lig_set),
            "n_decoys": len(dec_set),
            **stats,
            "status": status,
        })
    return rows


def audit_method_comparison_sources() -> list[dict]:
    """Task 0c extension: audit the 4 method-comparison data sources used by
    compute_logauc_bootstrap.py, if present on disk."""
    try:
        import compute_logauc_bootstrap as clb
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"  [method-comparison audit] could not import compute_logauc_bootstrap.py: {exc}")
        return []

    have_any = any(d.exists() for d in (clb.DIFFDOCK_DIR, clb.DOCK38_DIR, clb.OG_DIR, clb.RESCORE_DIR))
    if not have_any:
        print("  [method-comparison audit] none of the diffdock/dock3.8/OG_affinity/rescoring "
              "data directories exist locally — skipping (Task 0c extension not run).")
        return []

    receptors = clb.get_receptor_list()
    methods = {
        "DiffDock": clb.load_diffdock,
        "DOCK3.8": clb.load_dock38,
        "OG_Affinity": clb.load_og_affinity,
        "Boltz_Rescore": clb.load_rescoring,
    }

    rows = []
    for method_name, loader_fn in methods.items():
        for receptor in receptors:
            records = loader_fn(receptor)
            if not records:
                continue
            best = {}
            for name, score in records:
                if name not in best or score < best[name]:
                    best[name] = score
            names = set(best.keys())
            lig_set, dec_set = clb.split_lig_dec(names)
            if not lig_set or not dec_set:
                continue

            scores = [(n, best[n]) for n in names]
            stats = tie_stats(scores)
            status = classify(stats["distinct_values"], stats["tied_top1pct_flag"], stats["n"])
            rows.append({
                "source": f"method:{method_name}",
                "condition": method_name,
                "receptor": receptor,
                "n_ligands": len(lig_set),
                "n_decoys": len(dec_set),
                **stats,
                "status": status,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation-csv", type=Path, default=DEFAULT_ABLATION_CSV)
    parser.add_argument("--skip-method-comparison", action="store_true",
                         help="Skip the Task 0c method-comparison audit even if data is present.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Auditing {args.ablation_csv} ...")
    rows = audit_ablation_csv(args.ablation_csv)

    if not args.skip_method_comparison:
        print("Auditing method-comparison data sources (Task 0c extension)...")
        rows.extend(audit_method_comparison_sources())

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(audit_df)} rows to {OUT_CSV}")

    # ── summary ──
    print("\n" + "=" * 70)
    print("TIE-DENSITY AUDIT SUMMARY")
    print("=" * 70)

    for source in sorted(audit_df["source"].unique()):
        sub = audit_df[audit_df["source"] == source]
        n_ok = int((sub["status"] == "ok").sum())
        n_suspect = int((sub["status"] == "suspect").sum())
        n_unusable = int((sub["status"] == "unusable").sum())
        print(f"\n[{source}] {len(sub)} conditions: {n_ok} ok, {n_suspect} suspect, {n_unusable} unusable")

        flagged = sub[sub["status"] != "ok"].sort_values("distinct_values")
        for _, r in flagged.iterrows():
            tie_note = " TIED-TOP-1%" if r["tied_top1pct_flag"] else ""
            print(f"    {r['status'].upper():9s} {r['condition']:20s} {r['receptor']:15s} "
                  f"distinct={r['distinct_values']:<6d} largest_tie_block={r['largest_tie_block']:<6d}{tie_note}")

    n_unusable_total = int((audit_df["status"] == "unusable").sum())
    n_suspect_total = int((audit_df["status"] == "suspect").sum())
    print("\n" + "-" * 70)
    if n_unusable_total or n_suspect_total:
        print(f"ACTION REQUIRED: {n_unusable_total} unusable + {n_suspect_total} suspect condition(s). "
              f"These must be recomputed under the tie-aware bootstrap (logauc_utils.bootstrap_logauc) "
              f"before use in any downstream analysis (Shapley, NAE, factorial, etc.).")
    else:
        print("No suspect or unusable conditions found.")
    print("=" * 70)


if __name__ == "__main__":
    main()
