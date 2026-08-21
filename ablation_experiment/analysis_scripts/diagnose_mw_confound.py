#!/usr/bin/env python3
"""Diagnose the molecular-weight confound in ablation scores.

Why this exists
---------------
``affinity_head_forward`` applies Boltz's post-hoc molecular-weight
correction *outside* the network::

    pred = 1.03525938 * net_output - 0.59992683 * MW**0.3 + 2.83288489

That term is added after the affinity head, so **no ablation can remove
it**. In ``bias_only`` -- where z_trunk, s_inputs and the distogram are all
zeroed -- the network output is (near) constant across ligands, which
leaves ``-0.5999 * MW**0.3`` as essentially the *only* thing that varies.
``bias_only`` is therefore not a random baseline at all: it is a molecular
weight ranker, and since the correction is negative, it ranks *heavier*
compounds as better binders.

This script measures that directly from an existing results CSV, without a
GPU or a model. For each experiment it reports:

  spearman_vs_mw    rank correlation between the score and ligand MW.
                    |rho| near 1 for bias_only == the confound is confirmed.
  n_distinct_net    distinct values of the implied network output after
                    algebraically removing the MW term. If bias_only's
                    network output really is constant, this collapses to a
                    handful of values.

Ligand MW is taken from RDKit on the SMILES manifest (DUDEZ flow) so no
model or structure files are needed.

Usage
-----
python diagnose_mw_confound.py --dudez-inputs-root /path/to/DUDEZ_benchmark
python diagnose_mw_confound.py --experiments bias_only baseline no_z_trunk
"""

from __future__ import annotations

import argparse
import csv as _csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ABLATION_CSV = BASE_DIR / "results" / "ablation" / "feature_ablation_results.csv"
DEFAULT_DUDEZ_INPUTS = Path(
    "/home/limcaoco/turbo/limcaoco/boltz_benchmark/input_files/DUDEZ_benchmark"
)
OUT_DIR = BASE_DIR / "analysis_data"

# The exact constants from affinity_head_forward.
MODEL_COEF = 1.03525938
MW_COEF = -0.59992683
MW_BIAS = 2.83288489


def clean_ligand_name(raw) -> str:
    return re.sub(r"\s+none$", "", str(raw)).strip()


def load_mw_table(manifest_dir: Path, receptors) -> dict:
    """{(receptor, compound_id): molecular weight} from the SMILES manifests."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    out = {}
    for receptor in receptors:
        path = manifest_dir / f"{receptor}_combined_ids.csv"
        if not path.exists():
            print(f"  [{receptor}] no manifest at {path}, skipping")
            continue
        n_ok = n_bad = 0
        with path.open() as fh:
            reader = _csv.DictReader(fh)
            fields = reader.fieldnames or []
            skey = next((k for k in fields if k.strip().upper() == "SMILES"), None)
            ckey = next((k for k in fields if k.strip().lower() == "compound_id"), None)
            if skey is None or ckey is None:
                print(f"  [{receptor}] cannot find SMILES/compound_ID columns, skipping")
                continue
            for row in reader:
                cid = (row.get(ckey) or "").strip()
                smi = (row.get(skey) or "").strip()
                if not cid or not smi:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    n_bad += 1
                    continue
                out[(receptor, cid)] = float(Descriptors.MolWt(mol))
                n_ok += 1
        print(f"  [{receptor}] {n_ok} MW values ({n_bad} unparsable SMILES)")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation-csv", type=Path, default=ABLATION_CSV)
    p.add_argument("--dudez-inputs-root", type=Path, default=DEFAULT_DUDEZ_INPUTS)
    p.add_argument("--experiments", nargs="+", default=None,
                    help="Restrict to these experiments (default: all present).")
    p.add_argument("--output", type=Path, default=OUT_DIR / "mw_confound_diagnosis.csv")
    return p.parse_args()


def main():
    from scipy.stats import spearmanr

    args = parse_args()
    if not args.ablation_csv.exists():
        raise SystemExit(f"{args.ablation_csv} not found. Run the ablation first.")

    df = pd.read_csv(args.ablation_csv)
    df["ligand_id"] = df["ligand_name"].apply(clean_ligand_name)
    df = df[df["error"].isna() & df["affinity_pred_value"].notna()]
    if args.experiments:
        df = df[df["experiment"].isin(args.experiments)]
    if df.empty:
        raise SystemExit("No usable rows after filtering.")

    receptors = sorted(df["receptor_id"].unique())
    print(f"Loading molecular weights for {len(receptors)} receptor(s)...")
    mw_table = load_mw_table(args.dudez_inputs_root, receptors)
    if not mw_table:
        raise SystemExit(
            f"No molecular weights could be loaded from {args.dudez_inputs_root}. "
            f"Point --dudez-inputs-root at the directory holding "
            f"<RECEPTOR>_combined_ids.csv."
        )

    df["mw"] = [mw_table.get((r, l)) for r, l in zip(df["receptor_id"], df["ligand_id"])]
    n_missing = int(df["mw"].isna().sum())
    df = df[df["mw"].notna()]
    if df.empty:
        raise SystemExit("No rows matched a molecular weight; check compound-ID naming.")
    if n_missing:
        print(f"  ({n_missing} rows had no MW match and were dropped)")

    # Algebraically strip the MW term to recover the implied network output.
    df["implied_net"] = (df["affinity_pred_value"] - MW_COEF * df["mw"] ** 0.3 - MW_BIAS) / MODEL_COEF

    rows = []
    for exp, g in df.groupby("experiment"):
        rho_all, p_all = spearmanr(g["affinity_pred_value"], g["mw"])
        # Per-receptor rho matters more: logAUC is computed within a receptor,
        # so a pooled correlation can be diluted by between-receptor offsets.
        per_rec = []
        for _, gr in g.groupby("receptor_id"):
            if len(gr) >= 8 and gr["mw"].nunique() > 1:
                r, _ = spearmanr(gr["affinity_pred_value"], gr["mw"])
                if not np.isnan(r):
                    per_rec.append(r)
        net = g["implied_net"].to_numpy()
        net_rounded = np.round(net, 6)
        rows.append({
            "experiment": exp,
            "n_rows": len(g),
            "spearman_vs_mw_pooled": float(rho_all),
            "spearman_vs_mw_within_receptor_mean": float(np.mean(per_rec)) if per_rec else float("nan"),
            "n_receptors": g["receptor_id"].nunique(),
            "score_std": float(g["affinity_pred_value"].std()),
            "implied_net_std": float(net.std()),
            "implied_net_distinct": int(len(np.unique(net_rounded))),
            "implied_net_distinct_frac": float(len(np.unique(net_rounded)) / len(net)),
        })

    out = pd.DataFrame(rows).sort_values("spearman_vs_mw_within_receptor_mean",
                                          ascending=False, na_position="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("MOLECULAR-WEIGHT CONFOUND DIAGNOSIS")
    print("=" * 78)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("=" * 78)

    bias = out[out["experiment"] == "bias_only"]
    if not bias.empty:
        r = float(bias.iloc[0]["spearman_vs_mw_within_receptor_mean"])
        frac = float(bias.iloc[0]["implied_net_distinct_frac"])
        print(f"\nbias_only: within-receptor Spearman vs MW = {r:+.3f}")
        if abs(r) > 0.9:
            print("  => CONFIRMED. bias_only is essentially a molecular-weight ranker.")
            print("     It is NOT a random baseline and must not be used as v(0) for")
            print("     Shapley values or as the NAE denominator. Re-run with")
            print("     --no-mw-correction.")
        elif abs(r) > 0.5:
            print("  => Strong MW dependence, though not total: some signal survives")
            print("     zeroing through another path (e.g. token-count effects in the")
            print("     pairformer). Re-run with --no-mw-correction to separate them.")
        else:
            print("  => Weak MW dependence; the elevated bias_only score has another")
            print("     cause. Check the tie audit and the token-count leak.")
        print(f"  implied network output takes {frac:.1%} distinct values "
              f"(near 0 => the network really is constant, as expected when every "
              f"input channel is zeroed)")

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
