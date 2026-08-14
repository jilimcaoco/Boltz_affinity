"""Canonical SMILES standardization shared across Boltz parsing and the
data-preparation scripts.

This is the single source of truth for turning a raw SMILES string into the
canonical form that Boltz's ligand parser (and the affinity head) expects.
Both the package parser (:mod:`boltz.data.parse.schema`) and the
finetuning/validation input builders import from here so their behaviour can
never drift apart.

The module is intentionally dependency-light (only RDKit and
``chembl_structure_pipeline``) so preprocessing scripts can import it without
pulling in the heavyweight structure-parsing stack.
"""

from __future__ import annotations

from typing import Optional

from chembl_structure_pipeline.exclude_flag import exclude_flag
from chembl_structure_pipeline.standardizer import standardize_mol
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize(smiles: str) -> Optional[str]:
    """Standardize a molecule and return its canonical SMILES.

    This version has exception handling, which the original in mol-finder/data doesn't have. I didn't change the mol-finder/data
    since there are a lot of other functions that depend on it and I didn't want to break them.

    Raises ``ValueError`` if the molecule is excluded or cannot be
    standardized; callers that prefer a non-raising API should use
    :func:`try_standardize`.
    """
    LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()

    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    exclude = exclude_flag(mol, includeRDKitSanitization=False)

    if exclude:
        raise ValueError("Molecule is excluded")

    # sanitize=False skips implicit-valence calculation, which causes
    # LargestFragmentChooser to raise "getNumImplicitHs() called without
    # preceding call to calcImplicitValence()".  UpdatePropertyCache with
    # strict=False fills in valences without hard-failing on exotic valence
    # states (those will be caught by the Chem.MolFromSmiles round-trip below).
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    # Standardize with ChEMBL data curation pipeline. During standardization, the molecule may be broken
    # Choose molecule with largest component
    mol = LARGEST_FRAGMENT_CHOOSER.choose(mol)
    # Standardize with ChEMBL data curation pipeline. During standardization, the molecule may be broken
    mol = standardize_mol(mol)
    smiles = Chem.MolToSmiles(mol)

    # Check if molecule can be parsed by RDKit (in rare cases, the molecule may be broken during standardization)
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError("Molecule is broken")

    return smiles


def try_standardize(smiles: str) -> Optional[str]:
    """Non-raising wrapper around :func:`standardize`.

    Returns the canonical, fully-standardized SMILES, or ``None`` if the input
    is empty/invalid or the molecule is rejected by the standardization
    pipeline (e.g. excluded or broken).  This mirrors exactly what Boltz does
    internally when parsing an affinity ligand, so a ``None`` result means
    Boltz would also reject the molecule.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        result = standardize(smiles)
    except Exception:
        return None
    return result or None
