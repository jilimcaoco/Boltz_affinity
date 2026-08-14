#!/usr/bin/env python3
"""Combine DOCK3.8 scores, Boltz affinity rescoring results, and original
affinity predictions into a single CSV.

Data sources
------------
1. analysis_data/dock3.8_scores/{RECEPTOR}_dock_score.csv
   Columns: ligand_id, dock_score  (multiple poses per ligand)
2. analysis_data/OG_affinity_scores/master_scores.csv
   Columns: compound_ID, receptor, Affinity Pred Value, Affinity Probability Binary
3. results/rescoring/{RECEPTOR}_rescored.csv
   Columns: ligand_name (with '      none' suffix), affinity_score, ...

Output
------
analysis_data/combined_scores.csv
"""

from pathlib import Path

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # experiment_files/
DOCK_DIR = BASE_DIR / "analysis_data" / "dock3.8_scores"
OG_DIR   = BASE_DIR / "analysis_data" / "OG_affinity_scores"
RESCORE_DIR = BASE_DIR / "results" / "rescoring"
OUT_PATH = BASE_DIR / "analysis_data" / "combined_scores.csv"


def load_dock_scores(dock_dir: Path) -> pd.DataFrame:
    """Load per-receptor DOCK3.8 CSVs and take the *best* (most negative)
    dock_score per ligand."""
    frames = []
    for csv_path in sorted(dock_dir.glob("*_dock_score.csv")):
        receptor_id = csv_path.stem.replace("_dock_score", "")
        df = pd.read_csv(csv_path)
        df["receptor_id"] = receptor_id
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No dock score CSVs found in {dock_dir}")
    combined = pd.concat(frames, ignore_index=True)
    # Aggregate: best (min) dock score per (receptor, ligand)
    agg = (
        combined
        .groupby(["receptor_id", "ligand_id"], as_index=False)
        .agg(
            dock_score_best=("dock_score", "min"),
            dock_score_mean=("dock_score", "mean"),
            dock_n_poses=("dock_score", "count"),
        )
    )
    return agg


def load_og_affinity(og_dir: Path) -> pd.DataFrame:
    """Load the master OG affinity CSV."""
    csv_path = og_dir / "master_scores.csv"
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "compound_ID": "ligand_id",
        "receptor": "receptor_id",
        "Affinity Pred Value": "og_affinity_pred",
        "Affinity Probability Binary": "og_affinity_prob",
    })
    return df[["receptor_id", "ligand_id", "og_affinity_pred", "og_affinity_prob"]]


def load_rescoring(rescore_dir: Path) -> pd.DataFrame:
    """Load per-receptor Boltz affinity rescoring CSVs."""
    frames = []
    for csv_path in sorted(rescore_dir.glob("*_rescored.csv")):
        receptor_id = csv_path.stem.replace("_rescored", "")
        df = pd.read_csv(csv_path)
        df["receptor_id"] = receptor_id
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No rescored CSVs found in {rescore_dir}")
    combined = pd.concat(frames, ignore_index=True)
    # Clean ligand name: strip trailing whitespace + 'none' suffix
    combined["ligand_id"] = (
        combined["ligand_name"]
        .str.replace(r"\s+none$", "", regex=True)
        .str.strip()
    )
    # Keep useful columns
    combined = combined.rename(columns={
        "affinity_score": "rescore_affinity",
        "affinity_uncertainty": "rescore_uncertainty",
        "confidence": "rescore_confidence",
        "n_atoms": "rescore_n_atoms",
        "validation_status": "rescore_status",
    })
    keep_cols = [
        "receptor_id", "ligand_id",
        "rescore_affinity", "rescore_uncertainty",
        "rescore_confidence", "rescore_n_atoms", "rescore_status",
    ]
    return combined[keep_cols]


def main() -> None:
    print("Loading DOCK3.8 scores …")
    dock_df = load_dock_scores(DOCK_DIR)
    print(f"  {len(dock_df):,} ligand-receptor pairs from DOCK3.8")

    print("Loading OG affinity scores …")
    og_df = load_og_affinity(OG_DIR)
    print(f"  {len(og_df):,} ligand-receptor pairs from OG affinity")

    print("Loading Boltz rescoring results …")
    rescore_df = load_rescoring(RESCORE_DIR)
    print(f"  {len(rescore_df):,} ligand-receptor pairs from rescoring")

    # ── merge ────────────────────────────────────────────────────────────
    # Outer-join everything so no data is lost; NaN where a source is missing
    merged = dock_df.merge(rescore_df, on=["receptor_id", "ligand_id"], how="outer")
    merged = merged.merge(og_df, on=["receptor_id", "ligand_id"], how="outer")

    # Sort for readability
    merged = merged.sort_values(["receptor_id", "ligand_id"]).reset_index(drop=True)

    # ── save ─────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"\nCombined CSV written to {OUT_PATH}")
    print(f"  Total rows: {len(merged):,}")

    # Quick overlap summary
    has_dock = merged["dock_score_best"].notna()
    has_rescore = merged["rescore_affinity"].notna()
    has_og = merged["og_affinity_pred"].notna()
    print(f"  Rows with DOCK3.8 score:   {has_dock.sum():,}")
    print(f"  Rows with rescore:         {has_rescore.sum():,}")
    print(f"  Rows with OG affinity:     {has_og.sum():,}")
    print(f"  Rows with DOCK + rescore:  {(has_dock & has_rescore).sum():,}")
    print(f"  Rows with rescore + OG:    {(has_rescore & has_og).sum():,}")
    print(f"  Rows with all three:       {(has_dock & has_rescore & has_og).sum():,}")


if __name__ == "__main__":
    main()
