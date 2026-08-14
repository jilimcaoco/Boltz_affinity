#!/usr/bin/env python
"""Build a LoRA training manifest from the curated ChEMBL CSV + docked poses.

Takes the output of `pull_chembl_affinity_data.py` and the per-target pose
directories, and emits the manifest CSV expected by `boltz lora train`:

    name, ligand, receptor, target, structure, group_id, is_binder, is_censored

Mapping:
  - `target`      ← `log10_aff_uM`   (Boltz-2 affinity-head scale)
  - `group_id`    ← `assay_chembl_id` (for intra-assay pairwise Huber loss)
  - `is_binder`   ← 1 if `log10_aff_uM <= --binder-threshold` else 0
                   (consumed by the Boltz-2-style ``boltz2_affinity`` loss)
  - `is_censored` ← passed through from the curated CSV (0 or 1)
  - `receptor`    ← per-target receptor YAML
  - `structure`   ← {poses_dir}/{molecule_chembl_id}.pdb
  - `ligand`      ← `canonical_smiles_std`
  - `name`        ← `{source_target}_{molecule_chembl_id}_{assay_chembl_id}`

Censored (`is_censored=True`) rows are dropped by default since none of the
built-in losses handle right-censoring correctly.  Use ``--keep-censored``
together with ``--loss censored_intra_assay_huber`` to include them.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd


TARGET_KEYS = ("DRD4", "5HT2A")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curated-csv", required=True,
                   help="Output of pull_chembl_affinity_data.py.")
    p.add_argument("--drd4-receptor", required=True,
                   help="Path to the DRD4 Boltz YAML.")
    p.add_argument("--ht2a-receptor", required=True,
                   help="Path to the 5HT2A Boltz YAML.")
    p.add_argument("--drd4-poses", required=True,
                   help="Directory containing DRD4 docked complex PDBs.")
    p.add_argument("--ht2a-poses", required=True,
                   help="Directory containing 5HT2A docked complex PDBs.")
    p.add_argument("--pose-pattern", default="{molecule_chembl_id}.pdb",
                   help="Filename template inside the poses directory.")
    p.add_argument("--keep-censored", action="store_true",
                   help="Retain rows with is_censored=True (default: drop).")
    p.add_argument("--target", choices=["DRD4", "5HT2A", "all"], default="all",
                   help="Restrict manifest to a single receptor (default: all).")
    p.add_argument("--binder-threshold", type=float, default=1.0,
                   help="log10(IC50_uM) cutoff for the is_binder column "
                        "(default 1.0 = ≤10 µM is a binder, matching the "
                        "convention used by the Boltz-2 paper's BCE branch).")
    p.add_argument("--out", required=True, help="Output manifest CSV.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    df = pd.read_csv(args.curated_csv)
    logging.info("Loaded curated CSV: %d rows", len(df))

    # Censored handling
    if not args.keep_censored:
        before = len(df)
        df = df[~df["is_censored"].astype(bool)].copy()
        logging.info("Dropped censored rows: %d → %d", before, len(df))

    # Per-receptor filter
    if args.target != "all":
        before = len(df)
        df = df[df["source_target"] == args.target].copy()
        logging.info("Filtered to target=%s: %d → %d rows", args.target, before, len(df))
        if df.empty:
            logging.error("No rows remain after filtering to target=%s", args.target)
            return 1

    target_to_receptor = {
        "DRD4": Path(args.drd4_receptor).expanduser().resolve(),
        "5HT2A": Path(args.ht2a_receptor).expanduser().resolve(),
    }
    target_to_poses_dir = {
        "DRD4": Path(args.drd4_poses).expanduser().resolve(),
        "5HT2A": Path(args.ht2a_poses).expanduser().resolve(),
    }
    for k, p in {**target_to_receptor, **target_to_poses_dir}.items():
        if not p.exists():
            logging.error("Missing path for %s: %s", k, p)
            return 1

    out_rows: list[dict[str, str]] = []
    missing_poses: list[str] = []
    unknown_target_rows = 0

    for _, row in df.iterrows():
        src = str(row["source_target"])
        if src not in target_to_receptor:
            unknown_target_rows += 1
            continue

        pose_name = args.pose_pattern.format(
            molecule_chembl_id=row["molecule_chembl_id"],
            inchikey=row["inchikey"],
        )
        pose_path = target_to_poses_dir[src] / pose_name
        if not pose_path.exists():
            missing_poses.append(f"{src}/{pose_name}")
            continue

        name = f"{src}_{row['molecule_chembl_id']}_{row['assay_chembl_id']}"
        log10_uM = float(row['log10_aff_uM'])
        is_binder = 1 if log10_uM <= args.binder_threshold else 0
        is_censored = int(bool(row.get("is_censored", False)))
        out_rows.append({
            "name": name,
            "ligand": str(row["canonical_smiles_std"]),
            "receptor": str(target_to_receptor[src]),
            "target": f"{log10_uM:.6f}",
            "structure": str(pose_path),
            "group_id": str(row["assay_chembl_id"]),
            "is_binder": is_binder,
            "is_censored": is_censored,
        })

    if unknown_target_rows:
        logging.warning("Skipped %d rows with unknown source_target",
                        unknown_target_rows)
    if missing_poses:
        miss_log = Path(args.out).with_suffix(".missing.txt")
        miss_log.write_text("\n".join(missing_poses) + "\n")
        logging.warning("Missing pose files for %d compounds; list → %s",
                        len(missing_poses), miss_log)

    if not out_rows:
        logging.error("No usable rows — manifest is empty. Aborting.")
        return 1

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["name", "ligand", "receptor", "target",
                        "structure", "group_id", "is_binder", "is_censored"],
        )
        w.writeheader()
        w.writerows(out_rows)

    logging.info("Wrote %d manifest rows → %s", len(out_rows), out_path)

    # Quick per-target / per-assay summary so the user can sanity-check that
    # the assay-grouped sampler will have multi-row groups.
    manifest_df = pd.DataFrame(out_rows)
    print("\n--- Manifest summary ---")
    for src, sub in manifest_df.assign(
        src=manifest_df["name"].str.split("_").str[0]
    ).groupby("src"):
        print(f"  {src}: rows={len(sub)} assays={sub['group_id'].nunique()}")
    g = manifest_df.groupby("group_id").size()
    print(f"  multi-row assays (n>=2): {(g >= 2).sum()} / {len(g)}")
    print(f"  median rows per assay  : {int(g.median())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
