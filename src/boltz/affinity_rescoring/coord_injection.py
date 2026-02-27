"""
Coordinate injection utilities for direct PDB-coordinate affinity rescoring.

Takes parsed PDB atoms and injects their 3D coordinates into a processed
StructureV2, creating the pre_affinity npz that the affinity data pipeline
expects. This replaces the normal flow of running diffusion to produce
predicted coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from boltz.data.types import (
    Coords,
    Ensemble,
    StructureV2,
)

logger = logging.getLogger(__name__)


def inject_pdb_coords_into_structure(
    processed_structure: StructureV2,
    pdb_atoms: list,
    chain_id_map: Dict[str, int],
) -> StructureV2:
    """Inject PDB atom coordinates into a processed StructureV2.

    Matches atoms by chain_id → residue_index → atom_name. For any atom
    in the processed structure that has a matching PDB atom, the PDB
    coordinate is used. Missing atoms keep the CCD reference coordinate.

    Parameters
    ----------
    processed_structure : StructureV2
        The structure created by ``process_input`` with CCD reference coords.
    pdb_atoms : list
        Parsed atoms (AtomInfo namedtuples from ``parsers.parse_structure_file``).
        Each has: chain_id, residue_name, residue_number, atom_name, x, y, z,
        element, bfactor, occupancy, is_hetatm.
    chain_id_map : dict
        Mapping from PDB chain_id (str, e.g. "A") to the asym_id (int) used
        in the processed StructureV2.

    Returns
    -------
    StructureV2
        A new StructureV2 with PDB coordinates injected.
    """
    # Build a lookup: (asym_id, residue_idx_0based, atom_name) → (x, y, z)
    # Group PDB atoms by chain, then sort by residue number to get 0-based index
    pdb_by_chain: Dict[str, list] = {}
    for atom in pdb_atoms:
        cid = atom.chain_id
        if cid not in pdb_by_chain:
            pdb_by_chain[cid] = []
        pdb_by_chain[cid].append(atom)

    # Build coordinate lookup keyed by (asym_id, res_idx_in_chain, atom_name)
    coord_lookup: Dict[Tuple[int, int, str], np.ndarray] = {}
    for pdb_chain_id, atoms in pdb_by_chain.items():
        if pdb_chain_id not in chain_id_map:
            continue
        asym_id = chain_id_map[pdb_chain_id]

        # Get unique residue numbers in order
        res_numbers = []
        seen = set()
        for a in atoms:
            rn = a.residue_number
            if rn not in seen:
                res_numbers.append(rn)
                seen.add(rn)

        res_num_to_idx = {rn: i for i, rn in enumerate(res_numbers)}

        for a in atoms:
            res_idx = res_num_to_idx[a.residue_number]
            key = (asym_id, res_idx, a.atom_name.strip())
            coord_lookup[key] = np.array([a.x, a.y, a.z], dtype=np.float32)

    # Now inject into structure
    struct = processed_structure
    atoms = struct.atoms.copy()
    coords_array = struct.coords.copy()

    n_injected = 0
    n_missing = 0

    for chain in struct.chains[struct.mask]:
        asym_id = int(chain["asym_id"])
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]

        for local_res_idx, res in enumerate(struct.residues[res_start:res_end]):
            atom_start = res["atom_idx"]
            atom_end = atom_start + res["atom_num"]

            for atom_offset, atom in enumerate(atoms[atom_start:atom_end]):
                atom_name = atom["name"].strip() if isinstance(atom["name"], str) else atom["name"]
                # Handle bytes vs str
                if isinstance(atom_name, bytes):
                    atom_name = atom_name.decode("utf-8").strip()
                else:
                    atom_name = str(atom_name).strip()

                key = (asym_id, local_res_idx, atom_name)
                if key in coord_lookup:
                    new_coord = coord_lookup[key]
                    abs_atom_idx = atom_start + atom_offset

                    # Update atoms["coords"] (the deprecated per-atom field)
                    atoms[abs_atom_idx]["coords"] = new_coord

                    # Update coords table (ensemble coords)
                    # For the 0th ensemble (and all ensembles pointing to this atom)
                    for ens in struct.ensemble:
                        offset = ens["atom_coord_idx"]
                        coords_array[offset + abs_atom_idx]["coords"] = new_coord

                    # Mark atom as present
                    atoms[abs_atom_idx]["is_present"] = True
                    n_injected += 1
                else:
                    n_missing += 1

    logger.info(
        f"Coordinate injection: {n_injected} atoms matched, "
        f"{n_missing} atoms without PDB match (kept reference coords)."
    )

    return StructureV2(
        atoms=atoms,
        bonds=struct.bonds,
        residues=struct.residues,
        chains=struct.chains,
        interfaces=struct.interfaces,
        mask=struct.mask,
        coords=coords_array,
        ensemble=struct.ensemble,
        pocket=struct.pocket,
    )


def build_chain_id_map(
    processed_structure: StructureV2,
    yaml_chain_ids: List[str],
) -> Dict[str, int]:
    """Build mapping from YAML chain IDs to StructureV2 asym_ids.

    The YAML input assigns chain IDs like "A", "B", etc.  The processed
    StructureV2 has integer ``asym_id`` values.  This function builds
    the correspondence by matching the order of chains in both.

    Parameters
    ----------
    processed_structure : StructureV2
        The processed structure with integer asym_ids.
    yaml_chain_ids : list of str
        The chain IDs from the YAML in order (protein chains first,
        then ligand chains, matching the order they appear in the
        YAML ``sequences`` list).

    Returns
    -------
    dict
        Mapping from YAML chain_id (str) → asym_id (int).
    """
    chains = processed_structure.chains[processed_structure.mask]
    mapping = {}
    for i, chain in enumerate(chains):
        if i < len(yaml_chain_ids):
            mapping[yaml_chain_ids[i]] = int(chain["asym_id"])
    return mapping


def save_pre_affinity_structure(
    structure: StructureV2,
    output_dir: Path,
    record_id: str,
) -> Path:
    """Save a StructureV2 as a pre_affinity npz file.

    Parameters
    ----------
    structure : StructureV2
        The structure with injected PDB coordinates.
    output_dir : Path
        The predictions directory (will create {record_id}/ subdirectory).
    record_id : str
        The record ID for naming.

    Returns
    -------
    Path
        Path to the saved npz file.
    """
    pred_dir = output_dir / record_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"pre_affinity_{record_id}.npz"
    structure.dump(out_path)
    logger.info(f"Saved pre_affinity structure to {out_path}")
    return out_path
