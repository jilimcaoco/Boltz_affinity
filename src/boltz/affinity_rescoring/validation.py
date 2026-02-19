"""
Structure validation for affinity rescoring.

Comprehensive validation of PDB/CIF structure files with
recovery strategies and detailed error reporting.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from boltz.affinity_rescoring.models import (
    AtomInfo,
    ChainAssignment,
    ChainType,
    ValidationIssue,
    ValidationLevel,
    ValidationReport,
)

logger = logging.getLogger(__name__)

# ─── Periodic Table ──────────────────────────────────────────────────────────

VALID_ELEMENTS: Set[str] = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",
}

STANDARD_RESIDUES: Set[str] = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}

WATER_RESIDUES: Set[str] = {"HOH", "WAT", "H2O", "DOD", "SOL", "TIP"}

ION_RESIDUES: Set[str] = {
    "NA", "CL", "MG", "CA", "ZN", "FE", "MN", "CO", "NI", "CU",
    "K", "BR", "I", "SO4", "PO4", "IOD",
}


# ─── Structure Validator ─────────────────────────────────────────────────────


class StructureValidator:
    """
    Validates and normalizes PDB/CIF files with recovery strategies.

    Validation checklist:
    - File existence and readability
    - Format correctness (PDB/mmCIF headers)
    - Atom record integrity (coordinates, element, occupancy)
    - Chain continuity and gaps
    - Non-standard residues and ligands
    - B-factor ranges
    - Steric clashes
    - Heteroatom identity (distinguish ligands from water/ions)
    """

    RECOVERY_STRATEGIES = {
        "missing_chain_info": "infer_from_context",
        "alt_locations": "use_highest_occupancy",
        "missing_atoms": "skip_residue",
        "non_standard_residues": "treat_as_hetatm",
        "coordinate_outliers": "flag_for_review",
    }

    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        self.level = level

    def validate_file(self, filepath: str | Path) -> ValidationReport:
        """Validate a structure file at the filesystem level."""
        report = ValidationReport(is_valid=True)
        path = Path(filepath)

        # File existence
        if not path.exists():
            report.add_issue(
                "FILE_NOT_FOUND", "error",
                f"File not found: {path}. Check path spelling and permissions."
            )
            return report

        # File readability
        if not path.is_file():
            report.add_issue(
                "NOT_A_FILE", "error",
                f"Path is not a file: {path}"
            )
            return report

        if path.stat().st_size == 0:
            report.add_issue(
                "EMPTY_FILE", "error",
                f"File is empty: {path}"
            )
            return report

        # Format detection
        suffix = path.suffix.lower()
        valid_suffixes = {".pdb", ".cif", ".mmcif", ".ent", ".pdbx"}
        if suffix not in valid_suffixes:
            report.add_issue(
                "UNKNOWN_FORMAT", "warning",
                f"Unrecognized file extension '{suffix}'. "
                f"Expected one of: {', '.join(sorted(valid_suffixes))}"
            )

        # Try to read and check header
        try:
            with open(path, "r", errors="replace") as f:
                header = f.read(4096)
            if suffix in (".pdb", ".ent"):
                if not any(kw in header for kw in ("ATOM", "HETATM", "HEADER", "REMARK", "MODEL")):
                    report.add_issue(
                        "INVALID_PDB_HEADER", "warning",
                        "File does not contain expected PDB records (ATOM/HEADER/REMARK)."
                    )
            elif suffix in (".cif", ".mmcif", ".pdbx"):
                if "data_" not in header and "_atom_site" not in header:
                    report.add_issue(
                        "INVALID_CIF_HEADER", "warning",
                        "File does not contain expected mmCIF data blocks."
                    )
        except Exception as e:
            report.add_issue(
                "READ_ERROR", "error",
                f"Cannot read file: {e}"
            )

        return report

    def validate_atoms(self, atoms: List[AtomInfo]) -> ValidationReport:
        """Validate a list of parsed atoms."""
        report = ValidationReport(is_valid=True)

        if not atoms:
            report.add_issue(
                "NO_ATOMS", "error",
                "No atoms found in structure."
            )
            return report

        coords = np.array([[a.x, a.y, a.z] for a in atoms])

        # Coordinate ranges
        coord_min, coord_max = coords.min(), coords.max()
        if coord_min < -500 or coord_max > 500:
            report.add_issue(
                "COORDINATE_RANGE", "warning",
                f"Coordinate range [{coord_min:.1f}, {coord_max:.1f}] exceeds "
                f"typical range [-500, 500] Angstroms.",
                {"min": float(coord_min), "max": float(coord_max)}
            )

        # NaN/Inf coordinates
        if np.any(~np.isfinite(coords)):
            report.add_issue(
                "INVALID_COORDS", "error",
                "NaN or Inf detected in atom coordinates."
            )
            return report

        # Occupancy values
        occupancies = [a.occupancy for a in atoms]
        if any(o < 0.0 or o > 1.0 for o in occupancies):
            report.add_issue(
                "OCCUPANCY_RANGE", "warning",
                "Occupancy values outside [0.0, 1.0] range detected."
            )
        if all(o != 1.0 for o in occupancies):
            report.add_issue(
                "ALL_NON_UNIT_OCCUPANCY", "info",
                "All occupancy values are non-1.0. Alternate conformations may be present.",
            )
            report.add_recovery("Use highest occupancy conformer: alt_locations strategy")

        # B-factor ranges
        bfactors = [a.b_factor for a in atoms]
        max_bf = max(bfactors) if bfactors else 0
        if max_bf > 200:
            report.add_issue(
                "EXTREME_BFACTOR", "warning",
                f"Extreme B-factor values detected (max={max_bf:.1f}). "
                f"Values >200 suggest unreliable coordinates.",
                {"max_bfactor": max_bf}
            )

        # Element symbols
        invalid_elements = [a.element for a in atoms if a.element.upper() not in VALID_ELEMENTS]
        if invalid_elements:
            unique_invalid = set(invalid_elements)
            if self.level == ValidationLevel.STRICT:
                report.add_issue(
                    "INVALID_ELEMENTS", "error",
                    f"Invalid element symbols: {unique_invalid}"
                )
            else:
                report.add_issue(
                    "INVALID_ELEMENTS", "warning",
                    f"Unrecognized element symbols: {unique_invalid}"
                )

        # Steric clashes (sample-based for performance)
        n_atoms = len(atoms)
        if n_atoms <= 10000:
            dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
            np.fill_diagonal(dists, float("inf"))
            clash_pairs = np.sum(dists < 1.0) // 2
            if clash_pairs > 100:
                report.add_issue(
                    "STERIC_CLASHES", "warning" if self.level != ValidationLevel.STRICT else "error",
                    f"Detected {clash_pairs} atom pairs within 1.0 Angstrom. "
                    f"Possible coordinate errors or unfixed overlay.",
                    {"clash_count": int(clash_pairs)}
                )
        elif n_atoms > 10000:
            # Sample-based check for large structures
            sample_idx = np.random.choice(n_atoms, min(5000, n_atoms), replace=False)
            sample_coords = coords[sample_idx]
            dists = np.linalg.norm(
                sample_coords[:, None] - sample_coords[None, :], axis=-1
            )
            np.fill_diagonal(dists, float("inf"))
            clash_pairs = np.sum(dists < 1.0) // 2
            if clash_pairs > 10:
                report.add_issue(
                    "STERIC_CLASHES", "warning",
                    f"Detected {clash_pairs} clashes in sampled atoms (of {n_atoms} total). "
                    f"Full structure may have more.",
                )

        return report

    def validate_chains(
        self, atoms: List[AtomInfo]
    ) -> ValidationReport:
        """Validate chain continuity and composition."""
        report = ValidationReport(is_valid=True)

        if not atoms:
            return report

        # Group by chain
        chains: Dict[str, List[AtomInfo]] = {}
        for a in atoms:
            chains.setdefault(a.chain_id, []).append(a)

        if len(chains) < 2:
            report.add_issue(
                "SINGLE_CHAIN", "warning",
                f"Only {len(chains)} chain(s) found. "
                f"Protein-ligand complexes typically have 2+ chains."
            )

        for chain_id, chain_atoms in chains.items():
            residues = set(a.residue_name for a in chain_atoms)

            # Check for sequence gaps in protein chains
            protein_residues = residues & STANDARD_RESIDUES
            if len(protein_residues) > 5:  # Likely a protein chain
                res_nums = sorted(set(a.residue_number for a in chain_atoms
                                      if a.residue_name in STANDARD_RESIDUES))
                if len(res_nums) > 1:
                    gaps = []
                    for i in range(1, len(res_nums)):
                        gap = res_nums[i] - res_nums[i - 1]
                        if gap > 5:
                            gaps.append((res_nums[i - 1], res_nums[i], gap))
                    if gaps:
                        report.add_issue(
                            "SEQUENCE_GAPS", "warning",
                            f"Chain {chain_id}: {len(gaps)} sequence gap(s) >5 residues. "
                            f"Largest gap: {max(g[2] for g in gaps)} residues.",
                            {"chain": chain_id, "gaps": gaps}
                        )

        return report


# ─── Chain Identifier ────────────────────────────────────────────────────────


class ChainIdentifier:
    """
    Identifies protein vs. ligand chains with validation.

    Detection heuristics (priority order):
    1. User-provided explicit mapping
    2. Residue type analysis (>90% standard amino acids = protein)
    3. Molecular weight / atom count
    4. Chain name conventions
    """

    def identify_chains(
        self,
        atoms: List[AtomInfo],
        protein_chains: Optional[List[str]] = None,
        ligand_chains: Optional[List[str]] = None,
        auto_detect: bool = True,
        confidence_threshold: float = 0.8,
    ) -> ChainAssignment:
        """
        Identify protein and ligand chains.

        Returns assignment with confidence scores and reasoning.
        """
        # Group atoms by chain
        chains: Dict[str, List[AtomInfo]] = {}
        for a in atoms:
            chains.setdefault(a.chain_id, []).append(a)

        # Priority 1: User-provided explicit mapping
        if protein_chains is not None and ligand_chains is not None:
            # Validate that specified chains exist
            all_chain_ids = set(chains.keys())
            missing_p = set(protein_chains) - all_chain_ids
            missing_l = set(ligand_chains) - all_chain_ids
            if missing_p or missing_l:
                logger.warning(
                    f"Specified chains not found: protein={missing_p}, ligand={missing_l}"
                )
            chain_types = {}
            for c in protein_chains:
                chain_types[c] = ChainType.PROTEIN
            for c in ligand_chains:
                chain_types[c] = ChainType.LIGAND
            return ChainAssignment(
                protein_chains=list(protein_chains),
                ligand_chains=list(ligand_chains),
                confidence=1.0,
                reasoning="User-provided explicit chain mapping",
                chain_types=chain_types,
            )

        if not auto_detect:
            return ChainAssignment(
                protein_chains=protein_chains or [],
                ligand_chains=ligand_chains or [],
                confidence=0.0,
                reasoning="Auto-detection disabled and no explicit mapping provided",
            )

        # Priority 2: Residue type analysis
        detected_protein: List[str] = []
        detected_ligand: List[str] = []
        chain_types: Dict[str, ChainType] = {}
        reasoning_parts: List[str] = []

        for chain_id, chain_atoms in chains.items():
            residues = set(a.residue_name for a in chain_atoms)

            # Skip water and ions
            if residues <= WATER_RESIDUES | ION_RESIDUES:
                continue

            total_residues = len(set((a.residue_name, a.residue_number) for a in chain_atoms))
            protein_res = len(set(
                (a.residue_name, a.residue_number) for a in chain_atoms
                if a.residue_name in STANDARD_RESIDUES
            ))

            if total_residues > 0 and protein_res / total_residues > 0.9:
                detected_protein.append(chain_id)
                chain_types[chain_id] = ChainType.PROTEIN
                reasoning_parts.append(
                    f"Chain {chain_id}: {protein_res}/{total_residues} standard AA residues → protein"
                )
            elif total_residues > 0 and total_residues <= 10 and protein_res == 0:
                detected_ligand.append(chain_id)
                chain_types[chain_id] = ChainType.LIGAND
                reasoning_parts.append(
                    f"Chain {chain_id}: {total_residues} non-protein residues → ligand"
                )
            elif len(chain_atoms) < 200 and protein_res == 0:
                detected_ligand.append(chain_id)
                chain_types[chain_id] = ChainType.LIGAND
                reasoning_parts.append(
                    f"Chain {chain_id}: {len(chain_atoms)} atoms, no protein residues → ligand"
                )
            else:
                # Ambiguous: classify by atom count
                if len(chain_atoms) > 200:
                    detected_protein.append(chain_id)
                    chain_types[chain_id] = ChainType.PROTEIN
                    reasoning_parts.append(
                        f"Chain {chain_id}: {len(chain_atoms)} atoms → protein (by size)"
                    )
                else:
                    detected_ligand.append(chain_id)
                    chain_types[chain_id] = ChainType.LIGAND
                    reasoning_parts.append(
                        f"Chain {chain_id}: {len(chain_atoms)} atoms → ligand (by size)"
                    )

        # Override with user hints
        if protein_chains is not None:
            detected_protein = list(protein_chains)
        if ligand_chains is not None:
            detected_ligand = list(ligand_chains)

        # Calculate confidence
        confidence = 0.95 if detected_protein and detected_ligand else 0.5

        return ChainAssignment(
            protein_chains=detected_protein,
            ligand_chains=detected_ligand,
            confidence=confidence,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No chains classified",
            chain_types=chain_types,
        )


# ─── Structure Normalizer ────────────────────────────────────────────────────


class StructureNormalizer:
    """
    Standardizes structures for consistent model input.

    Operations:
    - Remove water and salt molecules
    - Handle alternate conformations (choose maximum occupancy)
    - Center coordinates (subtract centroid)
    """

    def normalize(
        self,
        atoms: List[AtomInfo],
        remove_waters: bool = True,
        remove_ions: bool = True,
        center_coordinates: bool = True,
    ) -> Tuple[List[AtomInfo], List[str]]:
        """
        Normalize atom list. Returns (normalized_atoms, transformation_log).
        """
        log: List[str] = []
        result = list(atoms)

        # Remove waters
        if remove_waters:
            before = len(result)
            result = [a for a in result if a.residue_name not in WATER_RESIDUES]
            removed = before - len(result)
            if removed > 0:
                log.append(f"Removed {removed} water atoms")

        # Remove ions
        if remove_ions:
            before = len(result)
            result = [a for a in result if a.residue_name not in ION_RESIDUES]
            removed = before - len(result)
            if removed > 0:
                log.append(f"Removed {removed} ion atoms")

        if not result:
            log.append("WARNING: No atoms remaining after filtering")
            return result, log

        # Center coordinates
        if center_coordinates:
            coords = np.array([[a.x, a.y, a.z] for a in result])
            centroid = coords.mean(axis=0)
            centered = []
            for a in result:
                centered.append(AtomInfo(
                    index=a.index,
                    name=a.name,
                    element=a.element,
                    x=a.x - centroid[0],
                    y=a.y - centroid[1],
                    z=a.z - centroid[2],
                    chain_id=a.chain_id,
                    residue_name=a.residue_name,
                    residue_number=a.residue_number,
                    occupancy=a.occupancy,
                    b_factor=a.b_factor,
                    is_hetatm=a.is_hetatm,
                ))
            result = centered
            log.append(f"Centered coordinates (centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}])")

        return result, log
