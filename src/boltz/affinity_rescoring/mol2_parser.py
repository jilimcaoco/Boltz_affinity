"""
MOL2 file parser for affinity rescoring.

Extracts individual ligand structures from multi-molecule MOL2 files,
preserving ligand names for output identification.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from boltz.affinity_rescoring.models import AtomInfo, LigandStructure

logger = logging.getLogger(__name__)

# MOL2 atom type → element mapping
MOL2_TYPE_TO_ELEMENT = {
    "C.3": "C", "C.2": "C", "C.1": "C", "C.ar": "C", "C.cat": "C",
    "N.3": "N", "N.2": "N", "N.1": "N", "N.ar": "N", "N.am": "N",
    "N.pl3": "N", "N.4": "N",
    "O.3": "O", "O.2": "O", "O.co2": "O", "O.spc": "O", "O.t3p": "O",
    "S.3": "S", "S.2": "S", "S.O": "S", "S.O2": "S",
    "P.3": "P",
    "H": "H", "H.spc": "H", "H.t3p": "H",
    "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "Li": "Li", "Na": "Na", "Mg": "Mg", "Al": "Al", "Si": "Si",
    "K": "K", "Ca": "Ca", "Cr.th": "Cr", "Cr.oh": "Cr",
    "Mn": "Mn", "Fe": "Fe", "Co.oh": "Co", "Cu": "Cu", "Zn": "Zn",
    "Se": "Se", "Mo": "Mo", "Sn": "Sn",
    "LP": "",  # Lone pair — skip
    "Du": "",  # Dummy atom — skip
}

# Bond type mapping
MOL2_BOND_TYPES = {"1": 1, "2": 2, "3": 3, "ar": 4, "am": 5, "du": 0, "un": 0, "nc": 0}

# Water/solvent molecule names to skip
SOLVENT_NAMES = {"HOH", "WAT", "H2O", "SOL", "TIP3P", "TIP4P", "SPC"}
ION_NAMES = {"NA+", "CL-", "K+", "MG2+", "CA2+", "ZN2+", "FE2+", "FE3+"}


class MOL2Parser:
    """
    Extracts individual ligand structures from multi-molecule MOL2 files.

    Handles:
    - Multiple molecules in single file
    - Preserves ligand names (crucial for output)
    - Validates atom coordinates and connectivity
    - Detects partial charges (used for better scoring)
    - Handles common artifact molecules (solvent, salt removed)
    """

    def __init__(
        self,
        remove_water: bool = True,
        remove_ions: bool = True,
        remove_hydrogens: bool = False,
    ):
        self.remove_water = remove_water
        self.remove_ions = remove_ions
        self.remove_hydrogens = remove_hydrogens

    def extract_ligands_with_names(
        self,
        mol2_path: str | Path,
    ) -> List[LigandStructure]:
        """
        Extract all ligands preserving their names.

        Parameters
        ----------
        mol2_path : str or Path
            Path to MOL2 file (single or multi-molecule)

        Returns
        -------
        list of LigandStructure
            Extracted ligands with names, atoms, bonds, and charges
        """
        path = Path(mol2_path)
        if not path.exists():
            raise FileNotFoundError(f"MOL2 file not found: {path}")

        with open(path, "r") as f:
            content = f.read()

        # Split into molecules
        molecule_blocks = self._split_molecules(content)
        logger.info(f"Found {len(molecule_blocks)} molecule(s) in {path.name}")

        ligands: List[LigandStructure] = []
        skipped = 0

        for i, block in enumerate(molecule_blocks):
            try:
                ligand = self._parse_molecule_block(block, i)
                if ligand is None:
                    skipped += 1
                    continue

                # Filter by type
                if self.remove_water and ligand.name.upper() in SOLVENT_NAMES:
                    skipped += 1
                    continue
                if self.remove_ions and ligand.name.upper() in ION_NAMES:
                    skipped += 1
                    continue

                ligands.append(ligand)

            except Exception as e:
                logger.warning(f"Failed to parse molecule {i + 1}: {e}")
                skipped += 1

        if skipped > 0:
            logger.info(f"Skipped {skipped} molecules (water/ions/errors)")

        return ligands

    def _split_molecules(self, content: str) -> List[str]:
        """Split MOL2 content into individual molecule blocks."""
        blocks: List[str] = []
        current_block: List[str] = []

        for line in content.splitlines(keepends=True):
            if line.strip().startswith("@<TRIPOS>MOLECULE"):
                if current_block:
                    blocks.append("".join(current_block))
                current_block = [line]
            else:
                current_block.append(line)

        if current_block:
            blocks.append("".join(current_block))

        return blocks

    def _parse_molecule_block(
        self, block: str, index: int
    ) -> Optional[LigandStructure]:
        """Parse a single @<TRIPOS>MOLECULE block."""
        sections = self._extract_sections(block)

        # Parse molecule header
        mol_section = sections.get("MOLECULE", "")
        mol_lines = [l for l in mol_section.strip().splitlines() if l.strip()]
        if not mol_lines:
            return None

        name = mol_lines[0].strip() if mol_lines else f"molecule_{index + 1}"
        if not name:
            name = f"molecule_{index + 1}"

        # Parse counts
        n_atoms = 0
        n_bonds = 0
        if len(mol_lines) > 1:
            counts = mol_lines[1].strip().split()
            if counts:
                try:
                    n_atoms = int(counts[0])
                except ValueError:
                    pass
                if len(counts) > 1:
                    try:
                        n_bonds = int(counts[1])
                    except ValueError:
                        pass

        # Parse atoms
        atom_section = sections.get("ATOM", "")
        atoms, charges = self._parse_atoms(atom_section, name)

        if not atoms:
            logger.debug(f"No atoms parsed for molecule '{name}'")
            return None

        # Remove hydrogens if requested
        if self.remove_hydrogens:
            non_h_indices = set()
            filtered_atoms = []
            for a in atoms:
                if a.element != "H":
                    non_h_indices.add(a.index)
                    filtered_atoms.append(a)
            atoms = filtered_atoms
            if charges:
                charges = [c for i, c in enumerate(charges)
                           if i in non_h_indices]

        # Parse bonds
        bond_section = sections.get("BOND", "")
        bonds = self._parse_bonds(bond_section)

        # Metadata
        metadata: Dict[str, Any] = {"source_index": index}
        mol_type_line = mol_lines[2].strip() if len(mol_lines) > 2 else ""
        if mol_type_line:
            metadata["mol_type"] = mol_type_line
        charge_type_line = mol_lines[3].strip() if len(mol_lines) > 3 else ""
        if charge_type_line:
            metadata["charge_type"] = charge_type_line

        # Parse comment section if present
        comment_section = sections.get("COMMENT", "")
        if comment_section.strip():
            metadata["comment"] = comment_section.strip()

        return LigandStructure(
            name=name,
            atoms=atoms,
            bonds=bonds,
            partial_charges=charges if any(c != 0.0 for c in charges) else None,
            metadata=metadata,
        )

    def _extract_sections(self, block: str) -> Dict[str, str]:
        """Extract named sections from a MOL2 block."""
        sections: Dict[str, str] = {}
        current_section = None
        current_lines: List[str] = []

        for line in block.splitlines():
            if line.strip().startswith("@<TRIPOS>"):
                if current_section is not None:
                    sections[current_section] = "\n".join(current_lines)
                section_name = line.strip().replace("@<TRIPOS>", "")
                current_section = section_name
                current_lines = []
            else:
                current_lines.append(line)

        if current_section is not None:
            sections[current_section] = "\n".join(current_lines)

        return sections

    def _parse_atoms(
        self, atom_section: str, mol_name: str
    ) -> Tuple[List[AtomInfo], List[float]]:
        """Parse @<TRIPOS>ATOM section."""
        atoms: List[AtomInfo] = []
        charges: List[float] = []

        for line in atom_section.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            try:
                atom_id = int(parts[0])
                atom_name = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                atom_type = parts[5]  # e.g., C.ar, N.3, etc.

                # Extract element from atom type
                element = self._atom_type_to_element(atom_type)
                if not element:
                    continue  # Skip lone pairs and dummy atoms

                # Parse charge (typically column 8)
                charge = 0.0
                if len(parts) >= 9:
                    try:
                        charge = float(parts[8])
                    except ValueError:
                        pass

                # Parse substructure info
                residue_name = mol_name
                residue_number = 1
                if len(parts) >= 7:
                    try:
                        residue_number = int(parts[6])
                    except ValueError:
                        pass
                if len(parts) >= 8:
                    residue_name = parts[7]

                atoms.append(AtomInfo(
                    index=atom_id,
                    name=atom_name,
                    element=element,
                    x=x, y=y, z=z,
                    chain_id="L",  # Ligand default chain
                    residue_name=residue_name,
                    residue_number=residue_number,
                    occupancy=1.0,
                    b_factor=0.0,
                    is_hetatm=True,
                ))
                charges.append(charge)

            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping malformed atom line: {line} ({e})")
                continue

        return atoms, charges

    def _parse_bonds(self, bond_section: str) -> List[tuple]:
        """Parse @<TRIPOS>BOND section."""
        bonds: List[tuple] = []

        for line in bond_section.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                atom1 = int(parts[1])
                atom2 = int(parts[2])
                bond_type_str = parts[3].lower()
                bond_type = MOL2_BOND_TYPES.get(bond_type_str, 1)
                bonds.append((atom1, atom2, bond_type))
            except (ValueError, IndexError):
                continue

        return bonds

    def _atom_type_to_element(self, atom_type: str) -> str:
        """Convert MOL2 atom type to element symbol."""
        # Direct lookup
        if atom_type in MOL2_TYPE_TO_ELEMENT:
            return MOL2_TYPE_TO_ELEMENT[atom_type]

        # Try extracting element from type (e.g., "C.3" → "C")
        base = atom_type.split(".")[0]
        if base in MOL2_TYPE_TO_ELEMENT:
            return MOL2_TYPE_TO_ELEMENT[base]

        # Try as raw element symbol
        if len(base) <= 2 and base[0].isupper():
            return base

        logger.debug(f"Unknown atom type: {atom_type}")
        return base if len(base) <= 2 else ""
