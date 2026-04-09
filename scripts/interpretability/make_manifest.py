"""
scripts/interpretability/make_manifest.py
==========================================
Scan a multi-mol2 file (docked ligand poses) and produce a manifest
TSV for the interpretability pipeline.

The manifest pairs a fixed receptor PDB with each ligand in the MOL2.

Usage
-----
    python scripts/interpretability/make_manifest.py \\
        --receptor protein.pdb \\
        --mol2 docked_ligands.mol2 \\
        --output manifest.tsv \\
        --dataset_type 1 \\
        [--pIC50_csv experimental_affinities.csv]

Output columns (tab-separated)
-------------------------------
    complex_name  receptor_pdb  mol2_file  ligand_name  dataset_type  pIC50

Dataset type meanings
---------------------
    1 = Congeneric series (same scaffold, varying substituents — SAR)
    2 = Alanine scanning (fixed ligand, single residue mutations)
    3 = Physicochemical decoupling (similar geometry, different MW/clogP)
    4 = Target shuffling (selective ligands vs. wrong protein targets)
    5 = Distance perturbation (same complex, ligand translated by Å increments)

    Edit the output TSV to set per-row dataset_type before submission.

pIC50 CSV format
----------------
    complex_name,pIC50
    LIGAND_42,8.52
    LIGAND_17,7.31
    ...
    If not provided, the pIC50 column is filled with "NA".
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _extract_mol2_names(mol2_path: Path) -> list[str]:
    """Return the molecule name from each @<TRIPOS>MOLECULE block.

    Multi-mol2 files use ``@<TRIPOS>MOLECULE`` as the block separator.
    The first non-blank line after the marker is the molecule name.
    """
    text = mol2_path.read_text()
    blocks = text.split("@<TRIPOS>MOLECULE")
    names: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if lines:
            names.append(lines[0].strip())
    return names


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a manifest TSV for the interpretability pipeline."
    )
    p.add_argument(
        "--receptor",
        required=True,
        help="Path to the receptor PDB (or CIF) file.",
    )
    p.add_argument(
        "--mol2",
        required=True,
        help="Path to the (multi-)MOL2 file with docked ligand poses.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output TSV path (e.g. manifest.tsv).",
    )
    p.add_argument(
        "--dataset_type",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Default dataset type assigned to every complex (1–5). "
             "Edit the output TSV to override per row.",
    )
    p.add_argument(
        "--pIC50_csv",
        default=None,
        help="Optional CSV with columns 'complex_name,pIC50'. "
             "Matched on complex_name (= ligand name). Unmatched complexes get 'NA'.",
    )
    args = p.parse_args()

    receptor_pdb = Path(args.receptor).resolve()
    mol2_file = Path(args.mol2).resolve()

    if not receptor_pdb.exists():
        raise FileNotFoundError(f"Receptor not found: {receptor_pdb}")
    if not mol2_file.exists():
        raise FileNotFoundError(f"MOL2 file not found: {mol2_file}")

    ligand_names = _extract_mol2_names(mol2_file)
    if not ligand_names:
        raise ValueError(f"No molecules found in {mol2_file}")

    # Load experimental affinities if provided.
    pic50: dict[str, str] = {}
    if args.pIC50_csv:
        with open(args.pIC50_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pic50[row["complex_name"]] = row["pIC50"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "complex_name", "receptor_pdb", "mol2_file",
            "ligand_name", "dataset_type", "pIC50",
        ])
        for name in ligand_names:
            writer.writerow([
                name,
                str(receptor_pdb),
                str(mol2_file),
                name,
                args.dataset_type,
                pic50.get(name, "NA"),
            ])

    print(f"Wrote {len(ligand_names)} entries → {out}")
    print("Edit the TSV to set per-row dataset_type before running submit_pipeline.sh")


if __name__ == "__main__":
    main()
