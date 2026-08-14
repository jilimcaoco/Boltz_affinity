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
) -> Tuple[StructureV2, List[int]]:
    """Inject PDB atom coordinates into a processed StructureV2.

    Matches atoms by chain_id → residue_index → atom_name. For any atom
    in the processed structure that has a matching PDB atom, the PDB
    coordinate is used. Unmatched atoms are zeroed out (coords set to
    ``[0, 0, 0]``) and their ``is_present`` flag remains ``False``.

    Parameters
    ----------
    processed_structure : StructureV2
        The structure created by ``process_input`` with CCD reference coords.
    pdb_atoms : list
        Parsed atoms (AtomInfo namedtuples from ``parsers.parse_structure_file``).
        Each has: chain_id, residue_name, residue_number, name, x, y, z,
        element, b_factor, occupancy, is_hetatm.
    chain_id_map : dict
        Mapping from PDB chain_id (str, e.g. "A") to the asym_id (int) used
        in the processed StructureV2.

    Returns
    -------
    tuple of (StructureV2, list of int)
        A new StructureV2 with PDB coordinates injected, and a list of
        atom indices that had no PDB match (coords zeroed).
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
            key = (asym_id, res_idx, a.name.strip())
            coord_lookup[key] = np.array([a.x, a.y, a.z], dtype=np.float32)

    # Now inject into structure
    struct = processed_structure
    atoms = struct.atoms.copy()
    coords_array = struct.coords.copy()

    n_injected = 0
    n_missing = 0
    unmatched_atom_indices: List[int] = []

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
                    abs_atom_idx = atom_start + atom_offset
                    unmatched_atom_indices.append(abs_atom_idx)

                    # Zero out unmatched atom coords so CCD-frame reference
                    # coordinates don't pollute pairwise distance computation.
                    # The is_present flag remains False, so the featurizer's
                    # resolved_mask will exclude them from centering.
                    atoms[abs_atom_idx]["coords"] = np.zeros(3, dtype=np.float32)
                    for ens in struct.ensemble:
                        offset = ens["atom_coord_idx"]
                        coords_array[offset + abs_atom_idx]["coords"] = (
                            np.zeros(3, dtype=np.float32)
                        )

                    n_missing += 1

    if n_missing > 0:
        logger.warning(
            f"Coordinate injection: {n_missing} atoms had no PDB match and "
            f"were zeroed out. If these include representative atoms for any "
            f"token, pairwise distances in the affinity module will be affected."
        )
    logger.info(
        f"Coordinate injection: {n_injected} atoms matched, "
        f"{n_missing} atoms without PDB match (coords zeroed)."
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
    ), unmatched_atom_indices


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


# ---------------------------------------------------------------------------
# Binding-pocket proximity check for unresolved residues
# ---------------------------------------------------------------------------

class UnresolvedResidueInfo(NamedTuple):
    """Summary of one unresolved residue's proximity to the binding pocket."""

    chain_asym_id: int
    chain_name: str
    residue_index: int  # 0-based within chain
    n_unresolved_atoms: int
    n_total_atoms: int
    min_distance_to_ligand: float  # Å, from flanking resolved atoms
    nearest_ligand_chain: str


class ProximityReport(NamedTuple):
    """Result of checking unresolved residue proximity to ligand."""

    n_unresolved_residues: int
    n_near_pocket: int  # within threshold
    threshold_angstrom: float
    near_pocket_residues: List[UnresolvedResidueInfo]
    all_unresolved_residues: List[UnresolvedResidueInfo]


def check_unresolved_near_pocket(
    structure: StructureV2,
    unmatched_atom_indices: List[int],
    threshold: float = 10.0,
) -> ProximityReport:
    """Check if any unresolved residues are near the ligand binding pocket.

    For each residue that contains unmatched (zeroed-out) atoms, this
    function computes the minimum distance from its **flanking resolved
    residues** to any ligand atom. Because the unresolved residue's own
    coordinates are zeroed, we use the closest resolved neighbours in
    the same chain as a proxy.  If the flanking residues are within
    ``threshold`` Å of a ligand atom, the unresolved region likely
    overlaps with or is adjacent to the binding pocket.

    Parameters
    ----------
    structure : StructureV2
        The structure **after** coordinate injection (resolved atoms
        have PDB coords, unresolved atoms are zeroed).
    unmatched_atom_indices : list of int
        Atom indices that had no PDB match (returned by
        ``inject_pdb_coords_into_structure``).
    threshold : float
        Distance cutoff in Ångströms (default 10.0).

    Returns
    -------
    ProximityReport
        Contains counts and per-residue details.
    """
    if not unmatched_atom_indices:
        return ProximityReport(
            n_unresolved_residues=0,
            n_near_pocket=0,
            threshold_angstrom=threshold,
            near_pocket_residues=[],
            all_unresolved_residues=[],
        )

    unmatched_set = set(unmatched_atom_indices)
    atoms = structure.atoms
    residues = structure.residues
    chains = structure.chains[structure.mask]

    # ── 1. Collect all resolved ligand atom coordinates ──────────────
    # NONPOLYMER = 3 in const.chain_type_ids
    NONPOLYMER = 3
    ligand_coords_by_chain: Dict[str, np.ndarray] = {}
    all_ligand_coords: List[np.ndarray] = []

    for chain in chains:
        if int(chain["mol_type"]) != NONPOLYMER:
            continue
        chain_name = str(chain["name"]).strip()
        a_start = int(chain["atom_idx"])
        a_end = a_start + int(chain["atom_num"])
        coords = []
        for ai in range(a_start, a_end):
            if atoms[ai]["is_present"] and ai not in unmatched_set:
                coords.append(atoms[ai]["coords"].astype(np.float64))
        if coords:
            arr = np.array(coords)
            ligand_coords_by_chain[chain_name] = arr
            all_ligand_coords.append(arr)

    if not all_ligand_coords:
        logger.warning(
            "No resolved ligand atoms found — cannot check pocket proximity."
        )
        return ProximityReport(
            n_unresolved_residues=0,
            n_near_pocket=0,
            threshold_angstrom=threshold,
            near_pocket_residues=[],
            all_unresolved_residues=[],
        )

    all_lig = np.concatenate(all_ligand_coords, axis=0)  # (N_lig, 3)

    # ── 2. Identify unresolved residues and their flanking resolved
    #       neighbours per chain ──────────────────────────────────────
    unresolved_info: List[UnresolvedResidueInfo] = []

    for chain in chains:
        mol_type = int(chain["mol_type"])
        if mol_type == NONPOLYMER:
            continue  # only checking receptor chains

        chain_name = str(chain["name"]).strip()
        asym_id = int(chain["asym_id"])
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        chain_residues = residues[res_start:res_end]

        # Classify each residue as resolved / unresolved
        # A residue is "unresolved" if any of its atoms are in unmatched_set
        res_resolved_coords: Dict[int, List[np.ndarray]] = {}
        res_unresolved: Dict[int, Tuple[int, int]] = {}  # local_idx → (n_unmatched, n_total)

        for local_idx, res in enumerate(chain_residues):
            a_start = int(res["atom_idx"])
            a_num = int(res["atom_num"])
            n_unmatched = 0
            resolved_coords = []
            for ai in range(a_start, a_start + a_num):
                if ai in unmatched_set:
                    n_unmatched += 1
                elif atoms[ai]["is_present"]:
                    resolved_coords.append(atoms[ai]["coords"].astype(np.float64))
            if n_unmatched > 0:
                res_unresolved[local_idx] = (n_unmatched, a_num)
            if resolved_coords:
                res_resolved_coords[local_idx] = resolved_coords

        if not res_unresolved:
            continue

        # For each unresolved residue, find closest flanking resolved
        # residues and compute their min distance to any ligand atom
        sorted_resolved = sorted(res_resolved_coords.keys())

        for local_idx, (n_un, n_tot) in res_unresolved.items():
            # Gather coordinates from flanking resolved residues.
            # Walk outward in both directions to find the nearest
            # resolved neighbor(s).
            flanking_coords: List[np.ndarray] = []

            # Search backward
            for ri in range(local_idx - 1, -1, -1):
                if ri in res_resolved_coords:
                    flanking_coords.extend(res_resolved_coords[ri])
                    break

            # Search forward
            for ri in range(local_idx + 1, len(chain_residues)):
                if ri in res_resolved_coords:
                    flanking_coords.extend(res_resolved_coords[ri])
                    break

            if not flanking_coords:
                # Entire chain is unresolved — use infinity
                min_dist = float("inf")
                nearest_chain = "N/A"
            else:
                flank = np.array(flanking_coords)  # (N_flank, 3)
                # Compute pairwise distances to all ligand atoms
                # flank[:, None, :] - all_lig[None, :, :] → (N_flank, N_lig, 3)
                diffs = flank[:, None, :] - all_lig[None, :, :]
                dists = np.sqrt((diffs ** 2).sum(axis=-1))  # (N_flank, N_lig)
                min_dist = float(dists.min())

                # Which ligand chain is closest?
                flat_idx = int(dists.argmin())
                lig_atom_idx = flat_idx % dists.shape[1]
                nearest_chain = "?"
                offset = 0
                for cname, lcoords in ligand_coords_by_chain.items():
                    if offset + len(lcoords) > lig_atom_idx:
                        nearest_chain = cname
                        break
                    offset += len(lcoords)

            unresolved_info.append(UnresolvedResidueInfo(
                chain_asym_id=asym_id,
                chain_name=chain_name,
                residue_index=local_idx,
                n_unresolved_atoms=n_un,
                n_total_atoms=n_tot,
                min_distance_to_ligand=round(min_dist, 2),
                nearest_ligand_chain=nearest_chain,
            ))

    near_pocket = [r for r in unresolved_info if r.min_distance_to_ligand <= threshold]

    # Log results
    if near_pocket:
        logger.warning(
            f"⚠ {len(near_pocket)} unresolved residue(s) are within "
            f"{threshold} Å of the ligand binding pocket:"
        )
        for r in near_pocket:
            logger.warning(
                f"  Chain {r.chain_name} res {r.residue_index}: "
                f"{r.n_unresolved_atoms}/{r.n_total_atoms} atoms unresolved, "
                f"flanking distance to ligand = {r.min_distance_to_ligand:.1f} Å "
                f"(nearest ligand chain: {r.nearest_ligand_chain})"
            )
    elif unresolved_info:
        logger.info(
            f"✓ {len(unresolved_info)} unresolved residue(s) found, "
            f"none within {threshold} Å of ligand."
        )

    return ProximityReport(
        n_unresolved_residues=len(unresolved_info),
        n_near_pocket=len(near_pocket),
        threshold_angstrom=threshold,
        near_pocket_residues=near_pocket,
        all_unresolved_residues=unresolved_info,
    )
