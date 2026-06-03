#!/usr/bin/env python
"""Check for data leakage between LoRA fine-tuning manifests and held-out validation sets.

For each validation compound (from labels_{TARGET}.csv) we compute the maximum
Tanimoto similarity to any compound in the LoRA training manifests using
Morgan fingerprints (radius=2, 2048 bits).  Both within-target and
cross-target comparisons are reported.

Outputs
-------
  leakage/leakage_summary.csv   — one row per (val_compound, train_target) pair
  leakage/leakage_report.txt    — human-readable summary

Usage
-----
  conda run -n boltz_affinity python check_leakage.py [OPTIONS]

  --labels-dir   DIR   directory containing labels_{TARGET}.csv  [./labels]
  --manifests-dir DIR  directory containing lora_manifest_{TARGET}.csv
                       [../../manifests]
  --output-dir   DIR   where to write results  [./leakage]
  --targets      TARGETS ...  [DRD4 5HT2A]
  --threshold    FLOAT  Tanimoto threshold to flag as potential leakage [0.85]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")  # suppress RDKit deprecation/info messages


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint helpers
# ─────────────────────────────────────────────────────────────────────────────

def canonical_smiles(smi: str) -> str | None:
    """Return RDKit canonical SMILES, or None if unparseable."""
    # Strip salts: keep only the largest fragment.
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    """Return Morgan fingerprint BitVect, or None if unparseable."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def bulk_tanimoto(query_fp, ref_fps: list) -> np.ndarray:
    """Return array of Tanimoto similarities from query_fp to each fp in ref_fps."""
    return np.array(DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_validation(labels_dir: Path, target: str) -> pd.DataFrame:
    """Load validation labels; return df with columns [name, smiles, canon_smi]."""
    path = labels_dir / f"labels_{target}.csv"
    df = pd.read_csv(path, usecols=lambda c: c in {"name", "smiles"})
    df["canon_smi"] = df["smiles"].apply(canonical_smiles)
    n_bad = df["canon_smi"].isna().sum()
    if n_bad:
        print(f"  [warn] {target} validation: {n_bad} SMILES could not be parsed, skipped")
    df = df[df["canon_smi"].notna()].copy()
    return df


def load_manifest(manifests_dir: Path, target: str) -> pd.DataFrame:
    """Load LoRA manifest; deduplicate by canonical SMILES.

    Returns df with columns [manifest_name, ligand_smiles, canon_smi, target_affinity].
    """
    path = manifests_dir / f"lora_manifest_{target}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"name": "manifest_name", "ligand": "ligand_smiles",
                             "target": "target_affinity"})
    df["canon_smi"] = df["ligand_smiles"].apply(canonical_smiles)
    n_bad = df["canon_smi"].isna().sum()
    if n_bad:
        print(f"  [warn] {target} manifest: {n_bad} SMILES could not be parsed, skipped")
    df = df[df["canon_smi"].notna()].copy()
    # Deduplicate: keep one row per unique canonical SMILES.
    before = len(df)
    df = df.drop_duplicates(subset="canon_smi").reset_index(drop=True)
    after = len(df)
    print(f"  {target} manifest: {before} rows → {after} unique compounds after dedup")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Core comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare(val_df: pd.DataFrame, train_df: pd.DataFrame,
            val_target: str, train_target: str) -> pd.DataFrame:
    """For every validation compound find the nearest training compound (Tanimoto).

    Returns a DataFrame with one row per validation compound.
    """
    # Build fingerprints.
    val_fps = []
    val_valid_idx = []
    for i, row in val_df.iterrows():
        fp = morgan_fp(row["canon_smi"])
        if fp is not None:
            val_fps.append(fp)
            val_valid_idx.append(i)

    train_fps = []
    train_valid_idx = []
    for i, row in train_df.iterrows():
        fp = morgan_fp(row["canon_smi"])
        if fp is not None:
            train_fps.append(fp)
            train_valid_idx.append(i)

    if not val_fps or not train_fps:
        print(f"  [skip] {val_target} vs {train_target}: no valid fingerprints")
        return pd.DataFrame()

    train_sub = train_df.iloc[train_valid_idx].reset_index(drop=True)
    val_sub   = val_df.iloc[val_valid_idx].reset_index(drop=True)

    rows = []
    for j, (fp, (_, vrow)) in enumerate(zip(val_fps, val_sub.iterrows())):
        tani = bulk_tanimoto(fp, train_fps)
        best_idx = int(np.argmax(tani))
        best_tani = float(tani[best_idx])
        best_train = train_sub.iloc[best_idx]

        # Exact canonical SMILES match (after desalting).
        exact = vrow["canon_smi"] == best_train["canon_smi"]

        rows.append({
            "val_target":       val_target,
            "train_target":     train_target,
            "val_name":         vrow["name"],
            "val_smiles":       vrow["smiles"],
            "val_canon_smi":    vrow["canon_smi"],
            "best_train_name":  best_train["manifest_name"],
            "best_train_smiles": best_train["ligand_smiles"],
            "best_train_canon": best_train["canon_smi"],
            "max_tanimoto":     best_tani,
            "exact_match":      exact,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _report_block(df: pd.DataFrame, threshold: float, label: str, buf: list[str]) -> None:
    buf.append(f"\n{'='*70}")
    buf.append(f"  {label}  (n_val={len(df)})")
    buf.append(f"{'='*70}")
    buf.append(f"  Max Tanimoto:  {df['max_tanimoto'].max():.4f}")
    buf.append(f"  Mean Tanimoto: {df['max_tanimoto'].mean():.4f}")
    buf.append(f"  Median:        {df['max_tanimoto'].median():.4f}")

    exact = df[df["exact_match"]]
    buf.append(f"  Exact matches (canonical SMILES): {len(exact)}")
    if len(exact):
        for _, r in exact.iterrows():
            buf.append(f"    * val={r['val_name']}  train={r['best_train_name']}")

    flagged = df[df["max_tanimoto"] >= threshold]
    buf.append(f"  Compounds >= {threshold:.2f} Tanimoto: {len(flagged)}")
    if len(flagged):
        flagged_sorted = flagged.sort_values("max_tanimoto", ascending=False)
        for _, r in flagged_sorted.iterrows():
            marker = " [EXACT]" if r["exact_match"] else ""
            buf.append(
                f"    Tanimoto={r['max_tanimoto']:.4f}{marker}  "
                f"val={r['val_name']}  best_train={r['best_train_name']}"
            )

    # Distribution buckets.
    buckets = [(0.9, 1.01, "0.90–1.00"), (0.85, 0.9, "0.85–0.90"),
               (0.7, 0.85, "0.70–0.85"), (0.5, 0.7, "0.50–0.70"),
               (0.0, 0.5, "< 0.50")]
    buf.append("  Similarity distribution:")
    for lo, hi, label_b in buckets:
        n = ((df["max_tanimoto"] >= lo) & (df["max_tanimoto"] < hi)).sum()
        buf.append(f"    {label_b}: {n}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-dir",    type=Path, default=here / "labels")
    p.add_argument("--manifests-dir", type=Path,
                   default=here.parent / "manifests")
    p.add_argument("--output-dir",    type=Path, default=here / "leakage")
    p.add_argument("--targets",       nargs="+", default=["DRD4", "5HT2A"])
    p.add_argument("--threshold",     type=float, default=0.85,
                   help="Tanimoto threshold to flag as potential leakage")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all datasets.
    print("\n[load] Validation sets")
    val = {}
    for t in args.targets:
        try:
            val[t] = load_validation(args.labels_dir, t)
            print(f"  {t}: {len(val[t])} compounds")
        except FileNotFoundError as e:
            print(f"  [skip] {e}")

    print("\n[load] LoRA manifests")
    train = {}
    for t in args.targets:
        try:
            train[t] = load_manifest(args.manifests_dir, t)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")

    # All (val_target, train_target) combinations.
    all_results: list[pd.DataFrame] = []
    report_buf: list[str] = ["Leakage Report", "=" * 70]
    report_buf.append(f"Targets:   {args.targets}")
    report_buf.append(f"Threshold: {args.threshold}")

    for vt in args.targets:
        if vt not in val:
            continue
        for tt in args.targets:
            if tt not in train:
                continue
            label = f"val={vt}  vs  train={tt}"
            print(f"\n[compare] {label} …")
            df = compare(val[vt], train[tt], vt, tt)
            if df.empty:
                continue
            all_results.append(df)
            kind = "WITHIN-TARGET" if vt == tt else "CROSS-TARGET"
            _report_block(df, args.threshold, f"{kind}: {label}", report_buf)

    # Save combined CSV.
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_csv = args.output_dir / "leakage_summary.csv"
        combined.to_csv(out_csv, index=False)
        print(f"\n[output] {out_csv}")
    else:
        print("[output] No results produced.")
        return 1

    # Save report.
    report_text = "\n".join(report_buf)
    out_txt = args.output_dir / "leakage_report.txt"
    out_txt.write_text(report_text)
    print(f"[output] {out_txt}")
    print("\n" + report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
