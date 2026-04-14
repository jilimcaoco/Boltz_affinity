#!/usr/bin/env python3
"""
prepare_interpretability_set.py

Reads the FEP benchmark structures CSV, filters to complexes suitable for
mechanistic interpretability analysis, extracts receptor PDB + ligand MOL2
from each CIF, and writes a manifest TSV for the interpretability pipeline.

Filtering criteria (discussed in project design):
  - passed_size_filter == True  (drug-like heavy atom count [10,45])
  - possible_cofactor  == False (removes ATP, ions, etc.)
  - Deduplicate by (pdb_id, ligand_id) — one complex per unique placement
  - pIC50 is NOT a filter — mechanistic analysis is model-internal

Output structure:
  datasets/interpretability/
    manifest.tsv                          — pipeline manifest
    structures/{target}/{pdb_id}/
      receptor.pdb                        — protein chain(s)
      {ligand_id}.mol2                    — ligand with bonds

Usage:
    python datasets/prepare_interpretability_set.py
    python datasets/prepare_interpretability_set.py --input datasets/fep_benchmark_annotated.csv
    python datasets/prepare_interpretability_set.py --input datasets/fep_benchmark_structures.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

DATASETS_DIR = Path("datasets")
DEFAULT_INPUTS = [
    DATASETS_DIR / "fep_benchmark_annotated.csv",
    DATASETS_DIR / "fep_benchmark_structures.csv",
]
OUTPUT_DIR = DATASETS_DIR / "interpretability"
MANIFEST_PATH = OUTPUT_DIR / "manifest.tsv"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def load_and_filter(path: Path) -> list[dict[str, Any]]:
    """Load the structures CSV and filter to interpretability-appropriate entries."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    total = len(rows)
    filtered = []
    seen: set[tuple[str, str]] = set()

    for r in rows:
        # Must have passed the drug-like size filter
        if r.get("passed_size_filter", "True").strip().lower() != "true":
            continue
        # Must NOT be a cofactor/buffer/ion
        if r.get("possible_cofactor", "False").strip().lower() == "true":
            continue
        # Deduplicate by (pdb_id, ligand_id)
        key = (r["pdb_id"].lower(), r["ligand_id"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(r)

    print(f"Loaded {total} rows from {path}")
    print(f"After filtering: {len(filtered)} unique complexes")
    print(f"  passed_size_filter=True, possible_cofactor=False, deduplicated")
    return filtered


# ---------------------------------------------------------------------------
# CIF → PDB + MOL2 extraction
# ---------------------------------------------------------------------------

def _extract_protein_pdb(cif_path: Path, out_pdb: Path, exclude_ligand: str) -> bool:
    """Extract protein (ATOM) records from CIF to a PDB file using gemmi.

    Writes all polymer chains, excluding the target ligand's chain if it
    contains only that ligand.
    """
    try:
        import gemmi
    except ImportError:
        raise ImportError("gemmi is required: pip install gemmi")

    structure = gemmi.read_structure(str(cif_path))
    if not structure:
        return False

    # Remove waters
    structure.remove_waters()

    # Remove all HETATM residues that are the ligand, solvents, or ions
    remove_names = {
        "HOH", "WAT", "H2O", "SO4", "GOL", "EDO", "PEG",
        "MPD", "PO4", "EPE", "MES", "ACT", "EOH", "FMT",
        "TRS", "BME", "DTT", "NH2", "NO3", "CL", "NA",
        "MG", "ZN", "CA", "K", "MN", "FE", "CU",
        exclude_ligand,
    }

    for model in structure:
        for chain in model:
            residues_to_remove = []
            for i, residue in enumerate(chain):
                if residue.het_flag == "H" or residue.name in remove_names:
                    residues_to_remove.append(i)
            for i in reversed(residues_to_remove):
                del chain[i]

    # Remove empty chains
    for model in structure:
        chains_to_remove = [
            i for i, chain in enumerate(model) if len(chain) == 0
        ]
        for i in reversed(chains_to_remove):
            del model[i]

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(out_pdb))
    return True


def _extract_ligand_mol2(
    cif_path: Path,
    ligand_id: str,
    out_mol2: Path,
) -> bool:
    """Extract ligand HETATM records from CIF, infer bonds via RDKit, write MOL2.

    Uses gemmi to get atom positions and RDKit to infer bond connectivity
    from 3D coordinates.
    """
    try:
        import gemmi
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDetermineBonds
    except ImportError as e:
        raise ImportError(f"gemmi and rdkit are required: {e}")

    structure = gemmi.read_structure(str(cif_path))
    if not structure:
        return False

    # Collect ligand atoms
    atoms: list[dict] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name.strip() != ligand_id:
                    continue
                if residue.het_flag != "H":
                    continue
                for atom in residue:
                    elem = atom.element.name if atom.element else ""
                    if elem == "H":
                        continue  # skip hydrogens
                    atoms.append({
                        "name": atom.name.strip(),
                        "element": elem,
                        "x": atom.pos.x,
                        "y": atom.pos.y,
                        "z": atom.pos.z,
                        "charge": atom.charge if hasattr(atom, "charge") else 0.0,
                    })
        if atoms:
            break  # Use first model only

    if not atoms:
        logger.warning(f"No HETATM atoms for {ligand_id} in {cif_path}")
        return False

    # Build RDKit mol from 3D coordinates to infer bonds
    rwmol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, a in enumerate(atoms):
        rdatom = Chem.Atom(a["element"])
        idx = rwmol.AddAtom(rdatom)
        conf.SetAtomPosition(idx, (a["x"], a["y"], a["z"]))
    rwmol.AddConformer(conf, assignId=True)

    # Use RDKit's bond determination from 3D coordinates
    try:
        rdDetermineBonds.DetermineBonds(rwmol, charge=0)
    except Exception:
        try:
            rdDetermineBonds.DetermineConnectivity(rwmol)
        except Exception:
            logger.warning(
                f"Bond determination failed for {ligand_id} in {cif_path}; "
                "writing MOL2 without bonds."
            )

    mol = rwmol.GetMol()

    # Collect bonds
    bonds: list[tuple[int, int, str]] = []
    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        bt_str = {
            Chem.BondType.SINGLE: "1",
            Chem.BondType.DOUBLE: "2",
            Chem.BondType.TRIPLE: "3",
            Chem.BondType.AROMATIC: "ar",
        }.get(bt, "1")
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bt_str))

    # Sybyl atom type mapping
    sybyl_map = {
        "C": "C.3", "N": "N.3", "O": "O.3", "S": "S.3", "P": "P.3",
        "F": "F", "Cl": "Cl", "Br": "Br", "I": "I", "Se": "Se",
        "B": "B", "Si": "Si",
    }

    # Write MOL2
    out_mol2.parent.mkdir(parents=True, exist_ok=True)
    with open(out_mol2, "w") as f:
        # MOLECULE section
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{ligand_id}\n")
        f.write(f" {len(atoms)} {len(bonds)} 0 0 0\n")
        f.write("SMALL\n")
        f.write("GASTEIGER\n")
        f.write("\n")

        # ATOM section
        f.write("@<TRIPOS>ATOM\n")
        for i, a in enumerate(atoms):
            sybyl = sybyl_map.get(a["element"], a["element"])
            # Try to refine sybyl type using RDKit hybridization
            if mol.GetNumBonds() > 0:
                rdatom = mol.GetAtomWithIdx(i)
                hyb = rdatom.GetHybridization()
                if a["element"] == "C":
                    if rdatom.GetIsAromatic():
                        sybyl = "C.ar"
                    elif hyb == Chem.HybridizationType.SP2:
                        sybyl = "C.2"
                    elif hyb == Chem.HybridizationType.SP:
                        sybyl = "C.1"
                elif a["element"] == "N":
                    if rdatom.GetIsAromatic():
                        sybyl = "N.ar"
                    elif hyb == Chem.HybridizationType.SP2:
                        sybyl = "N.2"
                elif a["element"] == "O":
                    if hyb == Chem.HybridizationType.SP2:
                        sybyl = "O.2"
            charge = a.get("charge", 0.0)
            f.write(
                f"{i + 1:>7d} {a['name']:<8s} "
                f"{a['x']:>10.4f}{a['y']:>10.4f}{a['z']:>10.4f} "
                f"{sybyl:<8s} 1 {ligand_id:<8s} {charge:>8.4f}\n"
            )

        # BOND section
        f.write("@<TRIPOS>BOND\n")
        for i, (a1, a2, bt) in enumerate(bonds):
            f.write(f"{i + 1:>6d}{a1 + 1:>5d}{a2 + 1:>5d} {bt}\n")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an interpretability dataset from FEP benchmark structures. "
            "Extracts receptor PDB + ligand MOL2 from each CIF and writes "
            "a manifest TSV for the interpretability pipeline."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input CSV (fep_benchmark_annotated.csv or fep_benchmark_structures.csv). "
            "Auto-detected if not specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=f"Manifest output path (default: {{output_dir}}/manifest.tsv)",
    )
    args = parser.parse_args()

    # Find input CSV
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = None
        for p in DEFAULT_INPUTS:
            if p.exists():
                input_path = p
                break
    if input_path is None or not input_path.exists():
        print("ERROR: No input CSV found. Run annotate_affinities.py first,", file=sys.stderr)
        print("       or specify --input.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.tsv"

    # Load and filter
    rows = load_and_filter(input_path)
    if not rows:
        print("ERROR: No complexes pass the filters.", file=sys.stderr)
        sys.exit(1)

    # Per-target summary
    targets: dict[str, int] = {}
    for r in rows:
        targets[r["target"]] = targets.get(r["target"], 0) + 1
    print("\nPer-target counts:")
    for t, n in sorted(targets.items()):
        print(f"  {t:<12s}: {n}")

    # Extract structures and build manifest
    structures_dir = output_dir / "structures"
    manifest_rows: list[dict[str, str]] = []
    success = 0
    fail = 0

    print(f"\nExtracting structures to {structures_dir}/ ...")
    for i, r in enumerate(rows):
        target = r["target"]
        pdb_id = r["pdb_id"]
        ligand_id = r["ligand_id"]
        cif_path = Path(r["cif_path"])

        if not cif_path.exists():
            logger.warning(f"CIF not found: {cif_path}")
            fail += 1
            continue

        complex_dir = structures_dir / target / pdb_id
        receptor_pdb = complex_dir / "receptor.pdb"
        ligand_mol2 = complex_dir / f"{ligand_id}.mol2"

        # Extract protein PDB (skip if already exists)
        if not receptor_pdb.exists():
            try:
                ok = _extract_protein_pdb(cif_path, receptor_pdb, ligand_id)
                if not ok:
                    logger.warning(f"No protein atoms in {cif_path}")
                    fail += 1
                    continue
            except Exception as e:
                logger.warning(f"PDB extraction failed for {pdb_id}: {e}")
                fail += 1
                continue

        # Extract ligand MOL2
        if not ligand_mol2.exists():
            try:
                ok = _extract_ligand_mol2(cif_path, ligand_id, ligand_mol2)
                if not ok:
                    logger.warning(f"MOL2 extraction failed for {pdb_id}/{ligand_id}")
                    fail += 1
                    continue
            except Exception as e:
                logger.warning(f"MOL2 extraction failed for {pdb_id}/{ligand_id}: {e}")
                fail += 1
                continue

        complex_name = f"{target}_{pdb_id}_{ligand_id}"

        # Get pIC50 if available
        pic50 = r.get("pIC50", "")
        if not pic50 or pic50.strip() == "":
            pic50 = "NA"

        manifest_rows.append({
            "complex_name": complex_name,
            "receptor_pdb": str(receptor_pdb.resolve()),
            "mol2_file": str(ligand_mol2.resolve()),
            "ligand_name": ligand_id,
            "dataset_type": "1",
            "pIC50": pic50,
        })
        success += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"  Processed {i + 1}/{len(rows)} ({success} ok, {fail} failed)")

    # Write manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "complex_name", "receptor_pdb", "mol2_file",
            "ligand_name", "dataset_type", "pIC50",
        ])
        for row in manifest_rows:
            writer.writerow([
                row["complex_name"], row["receptor_pdb"], row["mol2_file"],
                row["ligand_name"], row["dataset_type"], row["pIC50"],
            ])

    print(f"\n{'=' * 60}")
    print(f"Extracted: {success} complexes ({fail} failed)")
    print(f"Manifest:  {manifest_path} ({len(manifest_rows)} entries)")
    print(f"Structures: {structures_dir}/")
    print(f"{'=' * 60}")

    if manifest_rows:
        print(f"\nNext step:")
        print(f"  bash scripts/interpretability/run_pipeline_local.sh \\")
        print(f"      --manifest {manifest_path} \\")
        print(f"      --checkpoint ~/.boltz/boltz2_aff.ckpt \\")
        print(f"      --cached_dir {output_dir}/cached \\")
        print(f"      --results_dir {output_dir}/results")


if __name__ == "__main__":
    main()
