#!/usr/bin/env python3
"""Concatenate per-receptor DUDEZ ablation CSVs into the single results file
the analysis chain reads.

``run_feature_ablation_dudez.py`` runs one receptor per SLURM array task and
writes ``results/dudez_ablation/<RECEPTOR>_dudez_ablation.csv``. Everything
downstream (``audit_tie_density.py``, ``compute_ablation_bootstrap.py``, ...)
expects one combined table at ``results/ablation/feature_ablation_results.csv``.
This is the join.

Both runners share a column schema, so the combined file is readable by the
same chain either way. The DUDEZ rows additionally carry ``is_binder``,
which the analysis prefers over the "ZINC" name-prefix heuristic (see
``logauc_utils.label_actives_decoys``).

Usage
-----
python combine_dudez_results.py
python combine_dudez_results.py --input-dir ... --output ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "results" / "dudez_ablation"
DEFAULT_OUTPUT = BASE_DIR / "results" / "ablation" / "feature_ablation_results.csv"

REQUIRED_COLUMNS = {"receptor_id", "ligand_name", "experiment", "affinity_pred_value"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--pattern", default="*_dudez_ablation.csv")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        raise SystemExit(
            f"{args.input_dir} does not exist. Run the DUDEZ ablation first "
            f"(slurm_scripts/feature_ablation_dudez.slurm)."
        )

    csvs = sorted(args.input_dir.glob(args.pattern))
    if not csvs:
        raise SystemExit(
            f"No files matching {args.pattern!r} under {args.input_dir}. "
            f"Run the DUDEZ ablation first."
        )

    frames = []
    n_rows = 0
    for path in csvs:
        df = pd.read_csv(path)
        if df.empty:
            print(f"  skip {path.name}: empty")
            continue
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            # Fail loudly rather than silently dropping a receptor: a schema
            # mismatch here usually means the file came from an older runner.
            raise SystemExit(
                f"{path} is missing required column(s) {sorted(missing)}. "
                f"It was probably produced by an older version of "
                f"run_feature_ablation_dudez.py -- re-run that receptor."
            )
        frames.append(df)
        n_rows += len(df)
        n_rec = df["receptor_id"].nunique()
        print(f"  + {path.name}: {len(df)} rows, {n_rec} receptor(s)")

    if not frames:
        raise SystemExit(f"Every file under {args.input_dir} was empty.")

    combined = pd.concat(frames, ignore_index=True)

    dup = combined.duplicated(subset=["receptor_id", "ligand_name", "experiment"]).sum()
    if dup:
        print(f"\nWARNING: {dup} duplicate (receptor, ligand, experiment) row(s). "
              f"Re-running a receptor overwrites its own CSV, so duplicates usually "
              f"mean two files cover the same receptor. Keeping all rows -- the "
              f"analysis dedups by taking the best score per compound.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    n_err = combined["error"].notna().sum() if "error" in combined.columns else 0
    print(f"\nWrote {len(combined)} rows from {len(frames)} file(s) to {args.output}")
    print(f"  receptors  : {combined['receptor_id'].nunique()}")
    print(f"  experiments: {combined['experiment'].nunique()}")
    if "is_binder" in combined.columns:
        n_act = combined.drop_duplicates(["receptor_id", "ligand_name"])["is_binder"].sum()
        print(f"  labels     : is_binder present ({int(n_act)} actives across receptors)")
    else:
        print(f"  labels     : no is_binder column -- analysis will fall back to "
              f"the ZINC name-prefix heuristic")
    if n_err:
        print(f"  rows with a recorded error: {n_err} (kept; the analysis filters them)")


if __name__ == "__main__":
    main()
