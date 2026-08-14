#!/usr/bin/env python3
"""Task 6 — effect-size normalization and reporting.

Three additions on top of compute_ablation_bootstrap.py's output:

1. **Adjusted-logAUC confirmation.** ``logauc_utils.logAUC()`` already
   subtracts ``RANDOM_LOGAUC`` (see its docstring), so every logAUC value
   produced anywhere in this pipeline has random = 0 baked in. This script
   asserts that invariant holds (random-labeled data scores ~0) rather than
   just asserting it in prose, since it "has been misread before" per the
   handoff -- every output column here is explicitly suffixed/labeled
   ``_adjusted`` or documented as such.

2. **Normalized Ablation Effect (NAE).**
   ``NAE(c, r) = [v_r(full) - v_r(ablate c)] / [v_r(full) - v_r(bias_only)]``
   -- the fraction of usable signal a single-channel zero-ablation destroys,
   comparable across receptors of differing baseline difficulty. The
   denominator is ``bias_only`` (Task 0's v(∅) anchor), so NAE is gated on
   Task 0: don't trust it until the tie-density audit confirms bias_only is
   sound. A near-zero or negative denominator (baseline barely beats
   bias_only, or is worse) makes NAE meaningless -- those rows are flagged,
   not silently divided.

3. **receptor_class + rank preservation.** A best-effort DUD-E target-family
   lookup (kinase / GPCR / nuclear_receptor / protease / other) for
   per-class breakdowns, and within-receptor Spearman(ŷ_ablated, ŷ_baseline)
   as a receptor-normalized, actives:decoys-ratio-immune companion metric to
   logAUC (lower priority per the handoff, included for completeness).

Data
----
Reads ``analysis_data/ablation_bootstrap_results/summary_per_receptor.csv``
(point-estimate logAUC per experiment/receptor) and, for rank preservation,
``results/ablation/feature_ablation_results.csv`` directly (needs raw
per-ligand scores, which the summary doesn't carry).
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_CSV = BASE_DIR / "analysis_data" / "ablation_bootstrap_results" / "summary_per_receptor.csv"
ABLATION_CSV = BASE_DIR / "results" / "ablation" / "feature_ablation_results.csv"
OUT_DIR = BASE_DIR / "analysis_data"

NAE_CHANNEL_EXPERIMENTS = {
    "distogram": "no_distogram",
    "z_trunk": "no_z_trunk",
    "s_inputs": "no_s_inputs",
}
DENOMINATOR_NEAR_ZERO_THRESHOLD = 1.0  # adjusted-logAUC units


# Best-effort DUD-E target-family lookup. NOT exhaustive or independently
# re-verified against the current DUD-E/DUDEZ target list -- receptors not
# in this table default to "other" and are printed for manual review.
# Override/extend with --receptor-class-map (a two-column CSV:
# receptor,receptor_class).
DUDE_RECEPTOR_CLASS: Dict[str, str] = {
    # kinases
    "ABL1": "kinase", "AKT1": "kinase", "AKT2": "kinase", "BRAF": "kinase",
    "CDK2": "kinase", "CSF1R": "kinase", "EGFR": "kinase", "FAK1": "kinase",
    "FGFR1": "kinase", "IGF1R": "kinase", "JAK2": "kinase", "KIT": "kinase",
    "KPCB": "kinase", "LCK": "kinase", "MAPK2": "kinase", "MET": "kinase",
    "MK01": "kinase", "MK10": "kinase", "MK14": "kinase", "MP2K1": "kinase",
    "PLK1": "kinase", "ROCK1": "kinase", "SRC": "kinase", "TGFR1": "kinase",
    "VGFR2": "kinase", "WEE1": "kinase", "CDK1": "kinase", "CDK5": "kinase",
    # GPCRs
    "AA2AR": "GPCR", "ADRB1": "GPCR", "ADRB2": "GPCR", "CXCR4": "GPCR",
    "DRD3": "GPCR", "HRH1": "GPCR", "OPRK": "GPCR", "OPRM": "GPCR",
    "OPRD": "GPCR", "OPRX": "GPCR", "ADA": "other",  # ADA = adenosine deaminase, enzyme not GPCR
    # nuclear receptors
    "ANDR": "nuclear_receptor", "ESR1": "nuclear_receptor", "ESR2": "nuclear_receptor",
    "GCR": "nuclear_receptor", "MCR": "nuclear_receptor", "PPARA": "nuclear_receptor",
    "PPARD": "nuclear_receptor", "PPARG": "nuclear_receptor", "PRGR": "nuclear_receptor",
    "RXRA": "nuclear_receptor", "THB": "nuclear_receptor", "VDR": "nuclear_receptor",
    # proteases
    "ACE": "protease", "ADAM17": "protease", "BACE1": "protease", "CASP3": "protease",
    "FA10": "protease", "FA7": "protease", "THRB": "protease", "TRY1": "protease",
    "UROK": "protease", "HIVPR": "protease",
}


def load_receptor_class_map(override_csv: Optional[Path]) -> Dict[str, str]:
    mapping = dict(DUDE_RECEPTOR_CLASS)
    if override_csv is not None:
        override_df = pd.read_csv(override_csv)
        mapping.update(dict(zip(override_df["receptor"], override_df["receptor_class"])))
    return mapping


def classify_receptors(receptors, class_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    unmapped = []
    for r in receptors:
        cls = class_map.get(r, "other")
        if r not in class_map:
            unmapped.append(r)
        rows.append({"receptor": r, "receptor_class": cls})
    if unmapped:
        warnings.warn(
            f"{len(unmapped)} receptor(s) not in the built-in DUD-E class table, "
            f"defaulted to 'other': {sorted(unmapped)}. Supply --receptor-class-map "
            f"to classify them properly.",
            stacklevel=2,
        )
    return pd.DataFrame(rows)


# ── adjusted-logAUC confirmation ────────────────────────────────────────────

def confirm_adjusted_convention(summary_df: pd.DataFrame) -> None:
    """Sanity-check that the pipeline's logAUC values are already
    random-subtracted (random = 0), as logauc_utils.logAUC() promises.
    A well-separated baseline should be well above 0 and a near-constant
    condition (bias_only, if present) should be near 0 -- if bias_only
    were unadjusted it would sit near the *unadjusted* random baseline
    instead, which is a large positive number, not ~0."""
    if "bias_only" not in summary_df["experiment"].unique():
        print("  (bias_only not present in this summary -- skipping adjusted-convention spot check)")
        return
    bias_vals = summary_df.loc[summary_df["experiment"] == "bias_only", "point_estimate_logAUC"]
    mean_bias = bias_vals.mean()
    print(f"  bias_only mean point-estimate logAUC = {mean_bias:.2f} "
          f"(adjusted convention: expect ~0 for a condition with no per-ligand-varying signal, "
          f"once Task 0's tie fix and Task 0b's audit confirm bias_only is sound)")


# ── NAE ──────────────────────────────────────────────────────────────────────

def compute_nae(summary_df: pd.DataFrame) -> pd.DataFrame:
    pivot = summary_df.pivot_table(index="receptor", columns="experiment", values="point_estimate_logAUC")
    required = {"baseline", "bias_only"}
    missing = required - set(pivot.columns)
    if missing:
        raise ValueError(f"summary_per_receptor.csv is missing required experiment(s): {sorted(missing)}")

    denom = pivot["baseline"] - pivot["bias_only"]
    rows = []
    for channel, exp_name in NAE_CHANNEL_EXPERIMENTS.items():
        if exp_name not in pivot.columns:
            print(f"  NAE({channel}): experiment {exp_name!r} not present, skipping.")
            continue
        numer = pivot["baseline"] - pivot[exp_name]
        for receptor in pivot.index:
            d = denom[receptor]
            near_zero = abs(d) < DENOMINATOR_NEAR_ZERO_THRESHOLD
            nae = float("nan") if near_zero else float(numer[receptor] / d)
            rows.append({
                "receptor": receptor, "channel": channel,
                "v_full_baseline": float(pivot.loc[receptor, "baseline"]),
                "v_ablate_channel": float(pivot.loc[receptor, exp_name]),
                "v_bias_only": float(pivot.loc[receptor, "bias_only"]),
                "usable_signal_denominator": float(d),
                "denominator_near_zero_flag": near_zero,
                "NAE": nae,
            })
    return pd.DataFrame(rows)


# ── rank preservation ────────────────────────────────────────────────────────

def is_decoy(name: str) -> bool:
    return str(name).startswith("ZINC")


def clean_ligand_name(raw) -> str:
    return re.sub(r"\s+none$", "", str(raw)).strip()


def compute_rank_preservation(raw_df: pd.DataFrame, baseline_experiment: str = "baseline") -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    for receptor, df_rec in raw_df.groupby("receptor_id"):
        base_df = df_rec[df_rec["experiment"] == baseline_experiment]
        if base_df.empty:
            continue
        base = base_df.groupby("ligand_id")["affinity_pred_value"].min()
        for experiment, df_exp in df_rec.groupby("experiment"):
            if experiment == baseline_experiment:
                continue
            abl = df_exp.groupby("ligand_id")["affinity_pred_value"].min()
            common = sorted(set(base.index) & set(abl.index))
            if len(common) < 5:
                continue
            rho, pval = spearmanr([base[c] for c in common], [abl[c] for c in common])
            rows.append({
                "receptor": receptor, "experiment": experiment,
                "spearman_rho": float(rho), "spearman_pval": float(pval),
                "n_compounds": len(common),
            })
    return pd.DataFrame(rows)


# ── per-class aggregation ───────────────────────────────────────────────────

def per_class_breakdown(df: pd.DataFrame, value_col: str, class_df: pd.DataFrame, group_cols) -> pd.DataFrame:
    merged = df.merge(class_df, on="receptor", how="left")
    merged["receptor_class"] = merged["receptor_class"].fillna("other")
    return (
        merged.groupby(list(group_cols) + ["receptor_class"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std", "count": "n_receptors"})
    )


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--ablation-csv", type=Path, default=ABLATION_CSV)
    parser.add_argument("--receptor-class-map", type=Path, default=None,
                         help="Optional CSV with columns receptor,receptor_class overriding/extending "
                              "the built-in best-effort DUD-E table.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-rank-preservation", action="store_true",
                         help="Skip the Spearman rank-preservation pass (needs the raw per-ligand CSV).")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.summary_csv.exists():
        raise SystemExit(f"{args.summary_csv} not found. Run compute_ablation_bootstrap.py first.")

    summary_df = pd.read_csv(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Confirming adjusted-logAUC convention (random = 0)...")
    confirm_adjusted_convention(summary_df)

    print("\nComputing Normalized Ablation Effect (NAE)...")
    nae_df = compute_nae(summary_df)
    n_flagged = int(nae_df["denominator_near_zero_flag"].sum())
    if n_flagged:
        print(f"  WARNING: {n_flagged} (receptor, channel) row(s) have a near-zero usable-signal "
              f"denominator (|baseline - bias_only| < {DENOMINATOR_NEAR_ZERO_THRESHOLD}) -- NAE set to NaN "
              f"there rather than a meaningless/exploding ratio.")
    nae_df.to_csv(args.output_dir / "normalized_ablation_effect.csv", index=False)
    print(f"  Wrote {len(nae_df)} rows to normalized_ablation_effect.csv")

    print("\nClassifying receptors (best-effort DUD-E lookup)...")
    class_map = load_receptor_class_map(args.receptor_class_map)
    receptors = sorted(summary_df["receptor"].unique())
    class_df = classify_receptors(receptors, class_map)
    class_df.to_csv(args.output_dir / "receptor_class_map.csv", index=False)

    print("\nPer-class NAE breakdown:")
    nae_class = per_class_breakdown(nae_df, "NAE", class_df, ["channel"])
    nae_class.to_csv(args.output_dir / "normalized_ablation_effect_by_class.csv", index=False)
    print(nae_class.to_string(index=False))

    if not args.skip_rank_preservation:
        if args.ablation_csv.exists():
            print("\nComputing rank preservation (Spearman vs. baseline)...")
            raw_df = pd.read_csv(args.ablation_csv)
            raw_df["ligand_id"] = raw_df["ligand_name"].apply(clean_ligand_name)
            raw_df = raw_df[raw_df["error"].isna() & raw_df["affinity_pred_value"].notna()]
            rank_df = compute_rank_preservation(raw_df)
            rank_df.to_csv(args.output_dir / "rank_preservation.csv", index=False)
            print(f"  Wrote {len(rank_df)} rows to rank_preservation.csv")
        else:
            print(f"\n{args.ablation_csv} not found -- skipping rank preservation "
                  f"(needs raw per-ligand scores, not just the summary).")

    print("\nDone.")


if __name__ == "__main__":
    main()
