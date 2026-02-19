"""
Automatic SMILES inference from PDB/CIF ligand coordinates.

Uses RDKit's rdDetermineBonds to perceive bond orders from 3D
atomic coordinates in HETATM records. Falls back gracefully
when inference fails, allowing the user to provide SMILES manually.

Strategy (in priority order):
1. User-provided SMILES (always wins)
2. RDKit bond perception from 3D coordinates (rdDetermineBonds)
3. Failure with actionable error message
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from boltz.affinity_rescoring.models import AtomInfo

logger = logging.getLogger(__name__)


# ─── Element → Atomic Number ─────────────────────────────────────────────────

_ELEMENT_TO_ATOMIC_NUM: Dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
}


def _build_rwmol(
    atoms: List[AtomInfo],
) -> "tuple[Any, Any, List[AtomInfo]] | None":
    """Build an RWMol + Conformer from a list of AtomInfo objects.

    Returns (mol, conformer, used_atoms) or None on failure.
    """
    from rdkit import Chem

    valid_atoms: List[AtomInfo] = []
    mol = Chem.RWMol()

    for atom in atoms:
        elem = atom.element.strip().capitalize()
        atomic_num = _ELEMENT_TO_ATOMIC_NUM.get(elem)
        if atomic_num is None:
            try:
                atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(elem)
            except Exception:
                logger.debug(f"Unknown element '{elem}', skipping atom {atom.name}")
                continue
        rd_atom = Chem.Atom(atomic_num)
        mol.AddAtom(rd_atom)
        valid_atoms.append(atom)

    if mol.GetNumAtoms() == 0:
        return None

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, atom in enumerate(valid_atoms):
        conf.SetAtomPosition(i, (atom.x, atom.y, atom.z))
    mol.AddConformer(conf, assignId=True)
    return mol, conf, valid_atoms


def _validate_smiles(smiles: str) -> Optional[str]:
    """Re-parse a SMILES string and return canonical form, or None."""
    from rdkit import Chem

    if not smiles:
        return None
    check = Chem.MolFromSmiles(smiles)
    if check is None:
        logger.debug(f"Generated SMILES '{smiles}' fails re-parse validation")
        return None
    return Chem.MolToSmiles(check)


def infer_smiles_from_atoms(
    atoms: List[AtomInfo],
    charge: int = 0,
    embed_chiral: bool = True,
) -> Optional[str]:
    """
    Infer a SMILES string from 3D atom coordinates using RDKit bond perception.

    Strategy:
    1. Try ``DetermineBonds`` with **all** atoms (including H when present).
       This gives the best results because the xyz2mol algorithm can balance
       valences correctly with explicit hydrogens.
    2. If that fails (common when only heavy atoms are available), fall back
       to ``DetermineConnectivity`` which infers single-bond topology from
       inter-atomic distances.  Then attempt ``DetermineBondOrders``; if
       bond-order perception also fails the connectivity-only SMILES is
       still returned (RDKit fills implicit H based on standard valence).

    Parameters
    ----------
    atoms : list of AtomInfo
        Atoms with 3D coordinates (may include hydrogens).
    charge : int
        Net molecular charge (0 for neutral).
    embed_chiral : bool
        Embed chirality information in the SMILES.

    Returns
    -------
    str or None
        Canonical SMILES string, or None if inference fails entirely.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        logger.warning(
            "RDKit is required for SMILES inference from coordinates. "
            "Install via: pip install rdkit"
        )
        return None

    if not atoms:
        return None

    # Check whether hydrogens are present in the atom list
    has_hydrogens = any(a.element.upper() in ("H", "D") for a in atoms)
    heavy_atoms = [a for a in atoms if a.element.upper() not in ("H", "D")]
    if not heavy_atoms:
        return None

    # ── Attempt 1: Full bond perception with ALL atoms (H included) ─────
    if has_hydrogens:
        try:
            result = _build_rwmol(atoms)
            if result is not None:
                mol, _, _ = result
                rdDetermineBonds.DetermineBonds(mol, charge=charge)
                Chem.SanitizeMol(mol)
                # Remove explicit H from output to get cleaner SMILES
                mol_no_h = Chem.RemoveHs(mol)
                smiles = Chem.MolToSmiles(mol_no_h, isomericSmiles=embed_chiral)
                valid = _validate_smiles(smiles)
                if valid is not None:
                    logger.info(f"Inferred SMILES (with H) from coordinates: {valid}")
                    return valid
        except Exception as e:
            logger.debug(f"DetermineBonds with H failed: {e}")

    # ── Attempt 2: Full bond perception on heavy atoms only ─────────────
    try:
        result = _build_rwmol(heavy_atoms)
        if result is not None:
            mol, _, _ = result
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=embed_chiral)
            valid = _validate_smiles(smiles)
            if valid is not None:
                logger.info(f"Inferred SMILES (heavy-only) from coordinates: {valid}")
                return valid
    except Exception as e:
        logger.debug(f"DetermineBonds heavy-only failed: {e}")

    # ── Attempt 3: Connectivity-only + optional bond orders ─────────────
    try:
        result = _build_rwmol(heavy_atoms)
        if result is None:
            return None
        mol, _, _ = result

        rdDetermineBonds.DetermineConnectivity(mol)

        # Try to add bond orders (may fail without H)
        try:
            rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=embed_chiral)
            valid = _validate_smiles(smiles)
            if valid is not None:
                logger.info(
                    f"Inferred SMILES (connectivity+orders) from coordinates: {valid}"
                )
                return valid
        except Exception:
            pass

        # Bond orders failed — use connectivity-only SMILES.
        # MolToSmiles on an unsanitized mol with only single bonds produces
        # valid SMILES like "CCO" because RDKit infers implicit H from
        # standard valence rules when the SMILES is re-parsed.
        smiles = Chem.MolToSmiles(mol, isomericSmiles=embed_chiral)
        valid = _validate_smiles(smiles)
        if valid is not None:
            logger.info(
                f"Inferred SMILES (connectivity-only) from coordinates: {valid}. "
                f"Bond orders may be approximate — provide explicit SMILES for "
                f"best accuracy."
            )
            return valid

    except Exception as e:
        logger.debug(f"Connectivity fallback failed: {e}")

    return None


def infer_ligand_smiles_from_structure(
    atoms: List[AtomInfo],
    ligand_chains: List[str],
    user_smiles: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Resolve SMILES for all ligand chains, combining user overrides
    with automatic inference from coordinates.

    Priority:
    1. User-provided SMILES (from --ligand-smiles or parameter)
    2. Automatic inference via rdDetermineBonds

    Parameters
    ----------
    atoms : list of AtomInfo
        All atoms from the parsed structure.
    ligand_chains : list of str
        Chain IDs identified as ligands.
    user_smiles : dict, optional
        User-provided chain_id → SMILES overrides.

    Returns
    -------
    resolved_smiles : dict
        Map of chain_id → SMILES for each ligand chain that was resolved.
    warnings : list of str
        Warnings and info messages about the resolution process.
    """
    resolved: Dict[str, str] = {}
    warnings: List[str] = []
    user_smiles = user_smiles or {}

    # Group ligand atoms by chain and residue
    ligand_groups: Dict[str, Dict[str, List[AtomInfo]]] = {}
    for a in atoms:
        if a.chain_id in ligand_chains:
            ligand_groups.setdefault(a.chain_id, {}).setdefault(
                a.residue_name, []
            ).append(a)

    for chain_id in ligand_chains:
        # Priority 1: User-provided SMILES
        if chain_id in user_smiles:
            smiles = user_smiles[chain_id]
            resolved[chain_id] = smiles
            logger.info(
                f"Chain {chain_id}: using user-provided SMILES: {smiles}"
            )
            continue

        # Priority 2: Infer from coordinates
        chain_atoms = []
        for res_atoms in ligand_groups.get(chain_id, {}).values():
            chain_atoms.extend(res_atoms)

        if not chain_atoms:
            warnings.append(
                f"Chain {chain_id}: no atoms found for SMILES inference."
            )
            continue

        # Get the residue name(s) for logging
        res_names = sorted(set(a.residue_name for a in chain_atoms))
        n_heavy = len([a for a in chain_atoms if a.element.upper() not in ("H", "D")])

        logger.info(
            f"Chain {chain_id}: inferring SMILES from {n_heavy} heavy atoms "
            f"(residue(s): {', '.join(res_names)})"
        )

        smiles = infer_smiles_from_atoms(chain_atoms)

        if smiles is not None:
            resolved[chain_id] = smiles
            warnings.append(
                f"Chain {chain_id}: auto-inferred SMILES from coordinates: "
                f"{smiles} (from {n_heavy} heavy atoms, residue(s): "
                f"{', '.join(res_names)}). Verify this is correct or "
                f"provide explicit --ligand-smiles to override."
            )
        else:
            warnings.append(
                f"Chain {chain_id}: SMILES inference failed for "
                f"residue(s) {', '.join(res_names)} ({n_heavy} heavy atoms). "
                f"Provide SMILES manually via --ligand-smiles "
                f"'{{\"{chain_id}\": \"<SMILES>\"}}'"
            )

    return resolved, warnings
