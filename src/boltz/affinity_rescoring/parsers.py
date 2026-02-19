"""
Structure parsing for affinity rescoring.

Wraps Boltz's existing gemmi-based PDB/CIF parsers and adds
support for extracting atom-level information suitable for
the affinity rescoring pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from boltz.affinity_rescoring.models import AtomInfo, ValidationReport

logger = logging.getLogger(__name__)


def parse_structure_file(
    filepath: str | Path,
    remove_waters: bool = True,
    remove_hydrogens: bool = False,
) -> Tuple[List[AtomInfo], Dict[str, str]]:
    """
    Parse a PDB or CIF file into a list of AtomInfo objects.

    Uses gemmi for robust structure parsing (same library as Boltz core).

    Parameters
    ----------
    filepath : str or Path
        Path to structure file (.pdb, .cif, .mmcif, .ent, .pdbx)
    remove_waters : bool
        Remove water molecules
    remove_hydrogens : bool
        Remove hydrogen atoms

    Returns
    -------
    atoms : list of AtomInfo
        Parsed atom information
    metadata : dict
        File metadata (format, resolution, etc.)
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    try:
        import gemmi
    except ImportError:
        raise ImportError(
            "gemmi is required for structure parsing. Install via: pip install gemmi"
        )

    metadata: Dict[str, str] = {
        "source_file": str(path),
        "format": suffix,
    }

    # Parse structure
    try:
        if suffix in (".pdb", ".ent"):
            structure = gemmi.read_structure(str(path))
            metadata["format"] = "PDB"
        elif suffix in (".cif", ".mmcif", ".pdbx"):
            structure = gemmi.read_structure(str(path))
            metadata["format"] = "mmCIF"
        else:
            # Try PDB first, then CIF
            try:
                structure = gemmi.read_structure(str(path))
                metadata["format"] = "auto-detected"
            except Exception:
                raise ValueError(
                    f"Cannot parse file with extension '{suffix}'. "
                    f"Supported: .pdb, .cif, .mmcif, .ent, .pdbx"
                )
    except Exception as e:
        raise ValueError(f"Failed to parse {path}: {e}") from e

    # Extract metadata
    if structure.cell and structure.cell.a > 0:
        metadata["cell"] = (
            f"{structure.cell.a:.2f} {structure.cell.b:.2f} {structure.cell.c:.2f}"
        )

    # Extract atoms
    atoms: List[AtomInfo] = []
    atom_idx = 0

    water_names = {"HOH", "WAT", "H2O", "DOD", "SOL", "TIP"}

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.name.strip()

                if remove_waters and res_name in water_names:
                    continue

                for atom in residue:
                    element = atom.element.name if atom.element else ""

                    if remove_hydrogens and element == "H":
                        continue

                    atoms.append(AtomInfo(
                        index=atom_idx,
                        name=atom.name.strip(),
                        element=element,
                        x=atom.pos.x,
                        y=atom.pos.y,
                        z=atom.pos.z,
                        chain_id=chain.name,
                        residue_name=res_name,
                        residue_number=residue.seqid.num if residue.seqid else 0,
                        occupancy=atom.occ,
                        b_factor=atom.b_iso,
                        is_hetatm=residue.het_flag == "H",
                    ))
                    atom_idx += 1
        break  # Only process first model

    metadata["num_atoms"] = str(len(atoms))
    metadata["num_chains"] = str(len(set(a.chain_id for a in atoms)))

    # Extract SEQRES / entity-based sequences
    seqres = _extract_seqres(structure)
    if seqres:
        metadata["has_seqres"] = "true"
        for chain_id, seq in seqres.items():
            metadata[f"seqres_{chain_id}"] = str(len(seq))

    return atoms, metadata


# ─── SEQRES / Entity Sequence Extraction ─────────────────────────────────────

# Non-standard 3-letter → standard 3-letter mapping (same as get_chain_sequences
# but used here for SEQRES conversion).  Uppercase one-letter-code conversion
# from gemmi already handles standard residues; this catches force-field names
# that appear in some PDB SEQRES records.
_SEQRES_NONSTANDARD_TO_STANDARD: Dict[str, str] = {
    # Histidine protonation
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS",
    "HSE": "HIS", "HSP": "HIS",
    # Cysteine
    "CYX": "CYS", "CYM": "CYS",
    # Protonation states
    "ASH": "ASP", "GLH": "GLU",
    # Modified residues → parent
    "MSE": "MET", "SEC": "CYS", "TPO": "THR", "SEP": "SER", "PTR": "TYR",
    "PYL": "LYS",
}


def _seqres_three_to_one(three_letter: str) -> Optional[str]:
    """Convert a 3-letter SEQRES residue code to 1-letter amino acid code.

    Uses gemmi's tabulated residue info as the primary source, with
    fallback to our non-standard mapping table.

    Returns None for non-amino-acid entries (e.g. ligand components).
    """
    try:
        import gemmi
    except ImportError:
        return None

    name = three_letter.strip().upper()

    # Check our non-standard mapping first (force-field names)
    if name in _SEQRES_NONSTANDARD_TO_STANDARD:
        name = _SEQRES_NONSTANDARD_TO_STANDARD[name]

    info = gemmi.find_tabulated_residue(name)
    if not info.is_amino_acid():
        return None

    olc = info.one_letter_code
    if olc and olc != '?':
        # gemmi uses lowercase for modified residues (e.g. 'm' for MSE)
        # but Boltz-2 needs standard uppercase
        return olc.upper()

    return None


def _extract_seqres(structure: Any) -> Dict[str, str]:
    """Extract full deposited sequences from SEQRES / entity records.

    For PDB files, this reads the SEQRES records.  For mmCIF files,
    this reads ``_entity_poly_seq``.  Both are exposed through gemmi's
    ``Entity.full_sequence`` attribute.

    Parameters
    ----------
    structure : gemmi.Structure
        Parsed gemmi Structure object.

    Returns
    -------
    dict
        Map of chain_id → full one-letter amino acid sequence.
        Only protein chains with non-empty sequences are included.
    """
    try:
        import gemmi
    except ImportError:
        return {}

    # Ensure entity information is populated
    try:
        structure.setup_entities()
    except Exception:
        pass

    seqres: Dict[str, str] = {}

    for entity in structure.entities:
        # Only process polymer entities (proteins)
        if entity.entity_type != gemmi.EntityType.Polymer:
            continue
        if entity.polymer_type not in (
            gemmi.PolymerType.PeptideL,
            gemmi.PolymerType.PeptideD,
        ):
            continue

        # Convert 3-letter code list → 1-letter sequence
        full_seq = entity.full_sequence
        if not full_seq:
            continue

        one_letter_seq = []
        for code in full_seq:
            olc = _seqres_three_to_one(code)
            if olc is not None:
                one_letter_seq.append(olc)
            else:
                # Non-amino-acid in SEQRES — skip (ligand component, etc.)
                logger.debug(
                    f"Entity {entity.name}: skipping non-AA SEQRES entry '{code}'"
                )

        if not one_letter_seq:
            continue

        seq_str = "".join(one_letter_seq)

        # Map to chain IDs via entity.subchains
        for subchain in entity.subchains:
            # Find the auth chain name for this subchain
            for model in structure:
                for chain in model:
                    if chain.name in seqres:
                        continue
                    # Check if this chain uses this entity
                    try:
                        sub = chain.get_subchain(subchain)
                        if sub and len(sub) > 0:
                            seqres[chain.name] = seq_str
                            logger.debug(
                                f"SEQRES chain {chain.name}: "
                                f"{len(seq_str)} residues from entity {entity.name}"
                            )
                    except Exception:
                        pass
                break  # Only first model

        # Fallback: if subchains didn't map, try get_entity_of
        if not any(cid in seqres for cid in entity.subchains):
            for model in structure:
                for chain in model:
                    if chain.name in seqres:
                        continue
                    try:
                        ent = structure.get_entity_of(chain)
                        if ent and ent.name == entity.name:
                            seqres[chain.name] = seq_str
                    except Exception:
                        pass
                break

    return seqres


def get_seqres_sequences(filepath: str | Path) -> Dict[str, str]:
    """Extract SEQRES sequences from a PDB or CIF file.

    This is a convenience function that parses the file and returns
    only the SEQRES-derived sequences, without extracting atoms.

    Parameters
    ----------
    filepath : str or Path
        Path to PDB or CIF file.

    Returns
    -------
    dict
        Map of chain_id → full deposited amino acid sequence.
    """
    try:
        import gemmi
    except ImportError:
        logger.warning("gemmi is required for SEQRES extraction")
        return {}

    path = Path(filepath)
    try:
        structure = gemmi.read_structure(str(path))
        return _extract_seqres(structure)
    except Exception as e:
        logger.warning(f"Failed to extract SEQRES from {path}: {e}")
        return {}


def get_chain_sequences(
    atoms: List[AtomInfo],
    reference_sequences: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Extract amino acid sequences from parsed atoms.

    Handles non-standard residue names from AMBER, CHARMM, and other
    force-field conventions by mapping them to standard amino acids.
    Missing loop regions can optionally be filled from reference sequences.

    Parameters
    ----------
    atoms : list of AtomInfo
        Parsed atoms from a PDB/CIF file.
    reference_sequences : dict, optional
        Map of chain_id → full amino acid sequence (e.g. from UniProt).
        If provided and the extracted sequence has gaps, the reference
        sequence is used instead, with a warning.

    Returns
    -------
    dict
        Map of chain_id → amino acid sequence (1-letter codes).
    """
    # Standard 3-letter → 1-letter mapping
    THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    # Non-standard / force-field residue name → standard name mapping.
    # Covers AMBER, CHARMM, GROMACS protonation states and variants.
    NONSTANDARD_TO_STANDARD = {
        # Histidine protonation states
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS",
        "HSE": "HIS", "HSP": "HIS", "HSC": "HIS",
        # Cysteine variants
        "CYX": "CYS", "CYM": "CYS", "CYS2": "CYS", "CYF": "CYS",
        # Aspartate / glutamate protonation
        "ASH": "ASP", "GLH": "GLU", "GLUP": "GLU", "ASPP": "ASP",
        # Lysine
        "LYN": "LYS", "LYP": "LYS", "LYSH": "LYS",
        # Terminal variants (AMBER)
        "NALA": "ALA", "NARG": "ARG", "NASN": "ASN", "NASP": "ASP",
        "NCYS": "CYS", "NGLN": "GLN", "NGLU": "GLU", "NGLY": "GLY",
        "NHIS": "HIS", "NILE": "ILE", "NLEU": "LEU", "NLYS": "LYS",
        "NMET": "MET", "NPHE": "PHE", "NPRO": "PRO", "NSER": "SER",
        "NTHR": "THR", "NTRP": "TRP", "NTYR": "TYR", "NVAL": "VAL",
        "CALA": "ALA", "CARG": "ARG", "CASN": "ASN", "CASP": "ASP",
        "CCYS": "CYS", "CGLN": "GLN", "CGLU": "GLU", "CGLY": "GLY",
        "CHIS": "HIS", "CILE": "ILE", "CLEU": "LEU", "CLYS": "LYS",
        "CMET": "MET", "CPHE": "PHE", "CPRO": "PRO", "CSER": "SER",
        "CTHR": "THR", "CTRP": "TRP", "CTYR": "TYR", "CVAL": "VAL",
        # Other common non-standard names
        "MSE": "MET",  # selenomethionine
        "SEC": "CYS",  # selenocysteine (approximate)
        "PYL": "LYS",  # pyrrolysine (approximate)
        "TPO": "THR",  # phosphothreonine
        "SEP": "SER",  # phosphoserine
        "PTR": "TYR",  # phosphotyrosine
    }

    chains: Dict[str, Dict[int, str]] = {}
    mapped_nonstandard: Dict[str, set] = {}  # Track for logging

    for a in atoms:
        if a.name != "CA":
            continue

        res_name = a.residue_name.strip()

        # Try standard mapping first
        one_letter = THREE_TO_ONE.get(res_name)

        # Try non-standard mapping
        if one_letter is None:
            standard_name = NONSTANDARD_TO_STANDARD.get(res_name)
            if standard_name:
                one_letter = THREE_TO_ONE[standard_name]
                mapped_nonstandard.setdefault(a.chain_id, set()).add(
                    f"{res_name}→{standard_name}"
                )

        if one_letter is not None:
            if a.chain_id not in chains:
                chains[a.chain_id] = {}
            chains[a.chain_id][a.residue_number] = one_letter

    # Log non-standard mappings
    for chain_id, mappings in mapped_nonstandard.items():
        logger.info(
            f"Chain {chain_id}: mapped non-standard residues: {', '.join(sorted(mappings))}"
        )

    # Build sequences and detect gaps
    sequences: Dict[str, str] = {}
    for chain_id, residues in chains.items():
        sorted_nums = sorted(residues.keys())

        # Detect gaps
        gaps = []
        for i in range(1, len(sorted_nums)):
            gap_size = sorted_nums[i] - sorted_nums[i - 1]
            if gap_size > 1:
                gaps.append((sorted_nums[i - 1], sorted_nums[i], gap_size - 1))

        if gaps:
            total_missing = sum(g[2] for g in gaps)
            logger.warning(
                f"Chain {chain_id}: {len(gaps)} gap(s) totaling {total_missing} "
                f"missing residues detected in extracted sequence. "
                f"Largest gap: residues {gaps[-1][0]}–{gaps[-1][1]} "
                f"({gaps[-1][2]} missing)."
            )

        # Use reference sequence if available and gaps exist
        if reference_sequences and chain_id in reference_sequences and gaps:
            ref_seq = reference_sequences[chain_id]
            extracted_seq = "".join(residues[n] for n in sorted_nums)
            if len(ref_seq) >= len(extracted_seq):
                logger.info(
                    f"Chain {chain_id}: using reference sequence "
                    f"({len(ref_seq)} residues) instead of extracted "
                    f"({len(extracted_seq)} residues with gaps)."
                )
                sequences[chain_id] = ref_seq
                continue

        sequences[chain_id] = "".join(residues[n] for n in sorted_nums)

    return sequences


def count_receptor_contacts(
    protein_atoms: List[AtomInfo],
    ligand_atoms: List[AtomInfo],
    distance_cutoff: float = 5.0,
) -> int:
    """
    Count number of protein residues within distance_cutoff of any ligand atom.
    """
    import numpy as np

    if not protein_atoms or not ligand_atoms:
        return 0

    prot_coords = np.array([[a.x, a.y, a.z] for a in protein_atoms])
    lig_coords = np.array([[a.x, a.y, a.z] for a in ligand_atoms])

    # Compute minimum distance from each protein atom to any ligand atom
    # Use chunked computation for memory efficiency
    contact_residues = set()
    chunk_size = 1000
    for i in range(0, len(prot_coords), chunk_size):
        chunk = prot_coords[i:i + chunk_size]
        dists = np.min(
            np.linalg.norm(chunk[:, None] - lig_coords[None, :], axis=-1),
            axis=1,
        )
        for j, d in enumerate(dists):
            if d < distance_cutoff:
                atom = protein_atoms[i + j]
                contact_residues.add((atom.chain_id, atom.residue_number))

    return len(contact_residues)
