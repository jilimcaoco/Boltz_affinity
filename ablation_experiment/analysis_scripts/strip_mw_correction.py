#!/usr/bin/env python3
"""Recover un-MW-corrected affinity scores from an existing results CSV.

Why this can work without a GPU re-run
--------------------------------------
``affinity_head_forward`` applies Boltz's molecular-weight correction as the
*final* operation on the score::

    pred = MODEL_COEF * net - MW_COEF_ABS * MW**0.3 + MW_BIAS

Nothing downstream of that consumes ``net``, so the transform is a pure,
invertible function of (net, MW). Given the recorded ``pred`` and the
ligand's molecular weight we can recover ``net`` exactly::

    net = (pred - MW_COEF * MW**0.3 - MW_BIAS) / MODEL_COEF

which is bit-for-bit what the runner would have written under
``--no-mw-correction``, up to float round-trip through the CSV. That turns a
full GPU re-run into a CPU post-processing step.

Self-validation (important)
---------------------------
The recovery is only trustworthy if the MW we recompute here matches the MW
the model actually used (``record.affinity.mw``, i.e.
``rdkit Descriptors.MolWt`` on the ligand). We do not assume that -- we
*test* it, using ``bias_only`` as an oracle:

  With z_trunk, s_inputs and the distogram all zeroed, the network output is
  constant across ligands (every input to the affinity head is a constant
  tensor, and both the triangle-multiplication LayerNorm and the masked-mean
  readout are scale-invariant, so no token-count dependence survives). So
  the recovered ``net`` for bias_only must collapse to essentially ONE
  distinct value. If it does not, either our MW is wrong or there is a
  residual leak -- and in both cases the recovery is not safe to use.

The script refuses to write output when that check fails, unless --force.

Usage
-----
python strip_mw_correction.py --dudez-inputs-root /path/to/DUDEZ_benchmark
python strip_mw_correction.py --force   # override a failed validation
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_mw_confound import (  # noqa: E402
    MODEL_COEF,
    MW_BIAS,
    MW_COEF,
    DEFAULT_DUDEZ_INPUTS,
    load_mw_table,
)

BASE_DIR = Path(__file__).resolve().parent.parent
ABLATION_CSV = BASE_DIR / "results" / "ablation" / "feature_ablation_results.csv"

# bias_only's recovered network output should be a single value. Allow a
# little slack for fp16 z storage and LayerNorm eps.
ORACLE_EXPERIMENT = "bias_only"
ORACLE_MAX_DISTINCT_FRAC = 0.05
ORACLE_ROUND_DECIMALS = 4


def clean_ligand_name(raw) -> str:
    return re.sub(r"\s+none$", "", str(raw)).strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation-csv", type=Path, default=ABLATION_CSV)
    p.add_argument("--dudez-inputs-root", type=Path, default=DEFAULT_DUDEZ_INPUTS)
    p.add_argument("--output", type=Path, default=None,
                    help="Default: <ablation-csv stem>_nomw.csv next to the input.")
    p.add_argument("--oracle-experiment", default=ORACLE_EXPERIMENT,
                    help="Experiment used to validate the recovery (must be the "
                         "all-channels-zeroed condition).")
    p.add_argument("--force", action="store_true",
                    help="Write output even if the validation check fails.")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.output or args.ablation_csv.with_name(
        args.ablation_csv.stem + "_nomw.csv"
    )

    if not args.ablation_csv.exists():
        raise SystemExit(f"{args.ablation_csv} not found.")

    df = pd.read_csv(args.ablation_csv)
    df["ligand_id"] = df["ligand_name"].apply(clean_ligand_name)

    scored = df["affinity_pred_value"].notna()
    receptors = sorted(df.loc[scored, "receptor_id"].unique())
    print(f"Loading molecular weights for {len(receptors)} receptor(s)...")
    mw_table = load_mw_table(args.dudez_inputs_root, receptors)
    if not mw_table:
        raise SystemExit(
            f"No molecular weights loaded from {args.dudez_inputs_root}. "
            f"Point --dudez-inputs-root at the <RECEPTOR>_combined_ids.csv files."
        )

    df["_mw"] = [mw_table.get((r, l)) for r, l in zip(df["receptor_id"], df["ligand_id"])]
    recoverable = scored & df["_mw"].notna()
    n_unrecoverable = int((scored & df["_mw"].isna()).sum())

    df["affinity_pred_value_mwcorrected"] = df["affinity_pred_value"]
    df.loc[recoverable, "affinity_pred_value"] = (
        (df.loc[recoverable, "affinity_pred_value"]
         - MW_COEF * df.loc[recoverable, "_mw"] ** 0.3
         - MW_BIAS) / MODEL_COEF
    )
    # A row we cannot recover must not silently keep its MW-contaminated
    # score alongside corrected ones -- that would mix two scales within a
    # receptor and corrupt its ranking.
    if n_unrecoverable:
        df.loc[scored & df["_mw"].isna(), "affinity_pred_value"] = np.nan
        df.loc[scored & df["_mw"].isna(), "error"] = "mw_unavailable_for_correction"

    # ── Validation against the all-zeroed oracle ────────────────────────
    oracle = df[(df["experiment"] == args.oracle_experiment) & recoverable]
    ok = True
    print("\n" + "=" * 74)
    print("VALIDATION — recovered network output for "
          f"{args.oracle_experiment!r} should be constant")
    print("=" * 74)
    if oracle.empty:
        print(f"  {args.oracle_experiment!r} not present; CANNOT validate the recovery.")
        ok = False
    else:
        for receptor, g in oracle.groupby("receptor_id"):
            vals = np.round(g["affinity_pred_value"].to_numpy(), ORACLE_ROUND_DECIMALS)
            frac = len(np.unique(vals)) / len(vals)
            spread = float(g["affinity_pred_value"].max() - g["affinity_pred_value"].min())
            verdict = "OK" if frac <= ORACLE_MAX_DISTINCT_FRAC else "FAIL"
            if verdict == "FAIL":
                ok = False
            print(f"  [{receptor}] {len(vals):>6} rows, "
                  f"{len(np.unique(vals)):>5} distinct ({frac:.1%}), "
                  f"range {spread:.6f}  {verdict}")

    print("=" * 74)
    if ok:
        print("PASSED. The recovered scores are equivalent to a --no-mw-correction run:")
        print("  * the MW used here matches the MW the model used, and")
        print("  * no residual per-ligand signal survives zeroing every channel,")
        print("    so the MW term was the ONLY leak.")
    else:
        print("FAILED. bias_only did not collapse to a constant. Either the")
        print("recomputed MW differs from record.affinity.mw, or something other")
        print("than the MW term varies per ligand when all channels are zeroed.")
        print("Do NOT use this output -- re-run the ablation with")
        print("--no-mw-correction instead, which is correct by construction.")

    if not ok and not args.force:
        raise SystemExit(2)

    df = df.drop(columns=["_mw"])
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"  affinity_pred_value            -> MW correction removed")
    print(f"  affinity_pred_value_mwcorrected -> original value, kept for reference")
    if n_unrecoverable:
        print(f"  {n_unrecoverable} row(s) had no MW and were voided rather than "
              f"left on the old scale")
    print("\nPoint the analysis chain at this file, e.g.:")
    print(f"  cp {out_path} {args.ablation_csv}   # or --ablation-csv where supported")


if __name__ == "__main__":
    main()
