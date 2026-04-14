#!/usr/bin/env python3
"""
identify_congeneric_series.py

Identifies congeneric series within the FEP+ benchmark structures by
grouping ligands by Murcko scaffold, then computing pairwise similarity
and substitution counts from maximum common substructure (MCS).

Input:  datasets/fep_benchmark_annotated.csv
Output: datasets/congeneric_series.csv
        datasets/congeneric_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("datasets")
INPUT_CSV = DATASETS_DIR / "fep_benchmark_annotated.csv"
SERIES_CSV = DATASETS_DIR / "congeneric_series.csv"
PAIRS_CSV = DATASETS_DIR / "congeneric_pairs.csv"

# ---------------------------------------------------------------------------
# Series criteria
# ---------------------------------------------------------------------------

MIN_SERIES_SIZE = 3
MIN_PICO50_RANGE = 1.5
MIN_MAX_TANIMOTO = 0.6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_and_filter(path: Path) -> list[dict[str, Any]]:
    """Load annotated CSV and filter to usable entries."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    filtered = []
    for r in rows:
        # Skip entries flagged as preparation errors or weak binders
        # (columns may not exist yet — treat absent as False)
        if r.get("preparation_error", "False").strip().lower() == "true":
            continue
        if r.get("weak_binder", "False").strip().lower() == "true":
            continue
        # Must have affinity data
        if not r.get("pIC50") or r["pIC50"] == "":
            continue
        # Must have passed size filter
        if r.get("passed_size_filter", "True").strip().lower() == "false":
            continue
        filtered.append(r)

    return filtered


def _get_mol_from_cif(cif_path: str, ligand_id: str) -> Chem.Mol | None:
    """
    Extract a ligand from a CIF file as an RDKit Mol.

    Try gemmi first to pull the ligand component, fall back to reading
    HETATM records from the CIF and parsing via RDKit PDB block.
    """
    path = Path(cif_path)
    if not path.exists():
        return None

    # Strategy: read the CIF, extract HETATM lines for the target ligand,
    # build a minimal PDB block, and parse with RDKit.
    try:
        import gemmi  # type: ignore[import]
        doc = gemmi.cif.read(str(path))
        block = doc.sole_block()

        atom_site = block.find(
            "_atom_site.",
            ["group_PDB", "label_comp_id", "label_asym_id",
             "Cartn_x", "Cartn_y", "Cartn_z",
             "type_symbol", "label_atom_id", "label_seq_id",
             "auth_asym_id"],
        )

        pdb_lines = []
        serial = 1
        for row in atom_site:
            if row[0].strip().upper() != "HETATM":
                continue
            if row[1].strip() != ligand_id:
                continue
            x, y, z = float(row[3]), float(row[4]), float(row[5])
            elem = row[6].strip()
            atom_name = row[7].strip()
            chain = row[9].strip() if row[9].strip() else row[2].strip()
            resseq = row[8].strip() if row[8].strip() not in (".", "?") else "1"
            line = (
                f"HETATM{serial:5d} {atom_name:<4s} {ligand_id:>3s} "
                f"{chain:1s}{int(resseq):4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {elem:>2s}  "
            )
            pdb_lines.append(line)
            serial += 1

        if not pdb_lines:
            return None

        pdb_block = "\n".join(pdb_lines) + "\nEND\n"
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
        return mol

    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: try to read the whole CIF as a PDB (unlikely to work well)
    try:
        mol = Chem.MolFromPDBFile(str(path), removeHs=True, sanitize=False)
        return mol
    except Exception:
        return None


def compute_fingerprint(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)


def compute_scaffold(mol: Chem.Mol) -> str:
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core)


def compute_descriptors(mol: Chem.Mol) -> dict[str, float]:
    return {
        "MW": Descriptors.MolWt(mol),
        "clogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
    }


def tanimoto(fp1: DataStructs.ExplicitBitVect,
             fp2: DataStructs.ExplicitBitVect) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def build_ligand_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Augment rows with RDKit mol, fingerprint, scaffold, descriptors."""
    records = []
    seen: set[tuple[str, str]] = set()

    for r in rows:
        pdb_id = r["pdb_id"]
        ligand_id = r["ligand_id"]
        key = (pdb_id, ligand_id)
        if key in seen:
            continue
        seen.add(key)

        cif_path = r.get("cif_path", "")
        mol = _get_mol_from_cif(cif_path, ligand_id)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            continue

        try:
            fp = compute_fingerprint(mol)
            scaffold = compute_scaffold(mol)
            descs = compute_descriptors(mol)
        except Exception:
            continue

        pic50 = float(r["pIC50"])

        records.append({
            "pdb_id": pdb_id,
            "ligand_id": ligand_id,
            "target": r["target"],
            "mol": mol,
            "fp": fp,
            "scaffold": scaffold,
            "pIC50": pic50,
            **descs,
        })

    return records


def find_series(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group by scaffold and filter to valid congeneric series."""
    by_scaffold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_scaffold[rec["scaffold"]].append(rec)

    valid_series: list[list[dict[str, Any]]] = []
    for scaffold, members in by_scaffold.items():
        if len(members) < MIN_SERIES_SIZE:
            continue

        pics = [m["pIC50"] for m in members]
        pic_range = max(pics) - min(pics)
        if pic_range < MIN_PICO50_RANGE:
            continue

        # Check maximum pairwise Tanimoto
        max_tan = 0.0
        for a, b in itertools.combinations(members, 2):
            t = tanimoto(a["fp"], b["fp"])
            if t > max_tan:
                max_tan = t
        if max_tan < MIN_MAX_TANIMOTO:
            continue

        valid_series.append(members)

    return valid_series


def compute_mcs(mols: list[Chem.Mol]) -> Chem.Mol | None:
    """Find MCS across a list of molecules."""
    if len(mols) < 2:
        return None
    result = rdFMCS.FindMCS(
        mols,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        timeout=10,
    )
    if result.canceled or result.numAtoms == 0:
        return None
    mcs_mol = Chem.MolFromSmarts(result.smartsString)
    return mcs_mol


def substitution_count(mol: Chem.Mol, mcs_mol: Chem.Mol) -> int:
    """Number of heavy atoms in mol not part of the MCS."""
    match = mol.GetSubstructMatch(mcs_mol)
    return mol.GetNumHeavyAtoms() - len(match)


def build_pairs(
    series_id: int,
    members: list[dict[str, Any]],
    mcs_mol: Chem.Mol | None,
) -> list[dict[str, Any]]:
    """Build all pairwise records for a series."""
    pairs = []
    for a, b in itertools.combinations(members, 2):
        tan = tanimoto(a["fp"], b["fp"])
        delta = abs(a["pIC50"] - b["pIC50"])

        sub_a, sub_b = -1, -1
        total_subs = -1
        is_single = False
        sub_smarts = ""

        if mcs_mol is not None:
            sub_a = substitution_count(a["mol"], mcs_mol)
            sub_b = substitution_count(b["mol"], mcs_mol)
            total_subs = sub_a + sub_b
            is_single = total_subs == 1
            try:
                sub_smarts = Chem.MolToSmarts(mcs_mol)
            except Exception:
                sub_smarts = ""

        pairs.append({
            "series_id": series_id,
            "pdb_id_a": a["pdb_id"],
            "pdb_id_b": b["pdb_id"],
            "ligand_id_a": a["ligand_id"],
            "ligand_id_b": b["ligand_id"],
            "tanimoto": round(tan, 4),
            "delta_pIC50": round(delta, 3),
            "single_point_substitution": is_single,
            "substitution_smarts": sub_smarts,
        })

    return pairs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SERIES_COLUMNS = [
    "series_id", "target", "pdb_id", "ligand_id", "scaffold_smiles",
    "pIC50", "series_size", "pIC50_range",
]

PAIRS_COLUMNS = [
    "series_id", "pdb_id_a", "pdb_id_b", "ligand_id_a", "ligand_id_b",
    "tanimoto", "delta_pIC50", "single_point_substitution",
    "substitution_smarts",
]


def save_csvs(
    all_series_rows: list[dict[str, Any]],
    all_pairs_rows: list[dict[str, Any]],
    series_path: Path,
    pairs_path: Path,
) -> None:
    series_path.parent.mkdir(parents=True, exist_ok=True)

    with series_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SERIES_COLUMNS)
        w.writeheader()
        w.writerows(all_series_rows)
    print(f"Saved {len(all_series_rows)} rows → {series_path}")

    with pairs_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIRS_COLUMNS)
        w.writeheader()
        w.writerows(all_pairs_rows)
    print(f"Saved {len(all_pairs_rows)} rows → {pairs_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(
    series_list: list[list[dict[str, Any]]],
    all_pairs: list[dict[str, Any]],
) -> None:
    n_series = len(series_list)
    print(f"\n{'=' * 60}")
    print(f"Congeneric series found: {n_series}")

    # Size distribution
    sizes = [len(s) for s in series_list]
    size_counts: dict[str, int] = {"3": 0, "4": 0, "5": 0, "6+": 0}
    for sz in sizes:
        if sz >= 6:
            size_counts["6+"] += 1
        else:
            size_counts[str(sz)] = size_counts.get(str(sz), 0) + 1
    print("\nSeries size distribution:")
    for label, count in size_counts.items():
        print(f"  {label} members: {count}")

    # Single-point substitution pairs
    sps = [p for p in all_pairs if p["single_point_substitution"]]
    print(f"\nSingle-point substitution pairs: {len(sps)}")

    # Top 5 by pIC50 range
    series_with_range = []
    for i, members in enumerate(series_list, 1):
        pics = [m["pIC50"] for m in members]
        r = max(pics) - min(pics)
        target = members[0]["target"]
        scaffold = members[0]["scaffold"]
        series_with_range.append((i, target, scaffold, r, len(members)))

    series_with_range.sort(key=lambda x: x[3], reverse=True)
    print("\nTop 5 series by pIC50 range (most SAR signal):")
    print(f"  {'ID':>4}  {'Target':<12} {'Size':>5}  {'Range':>6}  Scaffold")
    for sid, target, scaffold, rng, sz in series_with_range[:5]:
        sc_short = scaffold[:50] + ("…" if len(scaffold) > 50 else "")
        print(f"  {sid:>4}  {target:<12} {sz:>5}  {rng:>6.2f}  {sc_short}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Identify congeneric series in FEP+ benchmark structures."
    )
    p.add_argument("--input", default=str(INPUT_CSV), metavar="PATH",
                    help=f"Annotated CSV (default: {INPUT_CSV})")
    p.add_argument("--series-out", default=str(SERIES_CSV), metavar="PATH",
                    help=f"Series CSV output (default: {SERIES_CSV})")
    p.add_argument("--pairs-out", default=str(PAIRS_CSV), metavar="PATH",
                    help=f"Pairs CSV output (default: {PAIRS_CSV})")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input)
    series_path = Path(args.series_out)
    pairs_path = Path(args.pairs_out)

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run annotate_affinities.py first.",
              file=sys.stderr)
        sys.exit(1)

    # Step 1: load & filter
    print(f"Loading {input_path} …")
    rows = _load_and_filter(input_path)
    print(f"  {len(rows)} entries after filtering.")

    # Step 2: compute RDKit features
    print("Computing RDKit features (fingerprints, scaffolds, descriptors) …")
    records = build_ligand_records(rows)
    print(f"  {len(records)} ligands with valid RDKit mols.")

    if not records:
        print("No valid ligands found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 3: find series
    print("Grouping by Murcko scaffold and filtering series …")
    series_list = find_series(records)
    print(f"  {len(series_list)} congeneric series pass criteria.")

    # Steps 4-5: MCS, pairs
    print("Computing MCS and pairwise comparisons …")
    all_series_rows: list[dict[str, Any]] = []
    all_pairs_rows: list[dict[str, Any]] = []

    for sid, members in enumerate(series_list, 1):
        pics = [m["pIC50"] for m in members]
        pic_range = round(max(pics) - min(pics), 3)
        series_size = len(members)

        # MCS
        mols = [m["mol"] for m in members]
        mcs_mol = compute_mcs(mols)

        # Series rows
        for m in members:
            all_series_rows.append({
                "series_id": sid,
                "target": m["target"],
                "pdb_id": m["pdb_id"],
                "ligand_id": m["ligand_id"],
                "scaffold_smiles": m["scaffold"],
                "pIC50": m["pIC50"],
                "series_size": series_size,
                "pIC50_range": pic_range,
            })

        # Pair rows
        pairs = build_pairs(sid, members, mcs_mol)
        all_pairs_rows.extend(pairs)

    # Step 6: save
    save_csvs(all_series_rows, all_pairs_rows, series_path, pairs_path)

    # Step 7: summary
    print_summary(series_list, all_pairs_rows)


if __name__ == "__main__":
    main()
