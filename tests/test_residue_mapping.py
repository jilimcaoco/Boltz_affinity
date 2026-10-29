"""PDB residue → processed residue mapping for coordinate injection.

Uses a synthetic StructureV2 (backbone atoms only, CA spaced 3.8 Å along x),
so no CCD / process_input run is needed.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from boltz.affinity_rescoring.coord_injection import (
    ResidueMappingError,
    _check_ca_continuity,
    inject_pdb_coords_into_structure,
    map_chain_residues,
    map_chains_by_name,
)
from boltz.affinity_rescoring.models import AtomInfo
from boltz.affinity_rescoring.parsers import THREE_TO_ONE
from boltz.data import const
from boltz.data.types import AtomV2, Chain, Coords, Ensemble, Interface, Residue, StructureV2

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}
SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV"  # 40 residues, non-repetitive
BACKBONE = ("N", "CA", "C", "O")
OFFSETS = {"N": (-1.0, 0.5, 0.0), "CA": (0.0, 0.0, 0.0), "C": (1.0, 0.5, 0.0), "O": (1.0, 1.5, 0.0)}


def _ca(true_idx: int) -> np.ndarray:
    return np.array([3.8 * true_idx, 0.0, 0.0], dtype=np.float32)


def _structure(seq: str = SEQ, chain_name: str = "A") -> StructureV2:
    n_res = len(seq)
    atoms = np.zeros(n_res * len(BACKBONE), dtype=AtomV2)
    residues = np.zeros(n_res, dtype=Residue)
    for i, aa in enumerate(seq):
        residues[i]["name"] = ONE_TO_THREE[aa]
        residues[i]["res_idx"] = i
        residues[i]["atom_idx"] = i * len(BACKBONE)
        residues[i]["atom_num"] = len(BACKBONE)
        for j, name in enumerate(BACKBONE):
            atoms[i * len(BACKBONE) + j]["name"] = name
    chains = np.zeros(1, dtype=Chain)
    chains[0]["name"] = chain_name
    chains[0]["mol_type"] = const.chain_type_ids["PROTEIN"]
    chains[0]["asym_id"] = 0
    chains[0]["atom_num"] = len(atoms)
    chains[0]["res_num"] = n_res
    coords = np.zeros(len(atoms), dtype=Coords)
    ensemble = np.zeros(1, dtype=Ensemble)
    ensemble[0]["atom_num"] = len(atoms)
    return StructureV2(
        atoms=atoms, bonds=np.zeros(0, dtype=[("x", "i4")]), residues=residues,
        chains=chains, interfaces=np.zeros(0, dtype=Interface),
        mask=np.ones(1, dtype=bool), coords=coords, ensemble=ensemble,
    )


def _pdb_atoms(true_indices, *, seq=SEQ, numbering=None, chain="A", names=None):
    """Backbone AtomInfo for the given true residue indices of ``seq``."""
    out = []
    for k, ti in enumerate(true_indices):
        num, icode = numbering[k] if numbering else (ti + 1, "")
        resname = names[k] if names else ONE_TO_THREE[seq[ti]]
        for name in BACKBONE:
            x, y, z = _ca(ti) + np.array(OFFSETS[name], dtype=np.float32)
            out.append(AtomInfo(
                index=len(out), name=name, element=name[0], x=float(x), y=float(y),
                z=float(z), chain_id=chain, residue_name=resname,
                residue_number=num, insertion_code=icode,
            ))
    return out


def _residues(atoms):
    seen, out = set(), []
    for a in atoms:
        key = (a.residue_number, a.insertion_code)
        if key not in seen:
            seen.add(key)
            out.append((key, a.residue_name))
    return out


def _proc_names(seq=SEQ):
    return [ONE_TO_THREE[a] for a in seq]


def _injected_ca(struct: StructureV2) -> list:
    out = []
    for i in range(len(struct.residues)):
        a = struct.atoms[i * len(BACKBONE) + 1]
        out.append(np.array(a["coords"]) if a["is_present"] else None)
    return out


# ─── map_chain_residues ──────────────────────────────────────────────────────


def test_identity_fast_path():
    res = _residues(_pdb_atoms(range(len(SEQ))))
    mapping, realigned = map_chain_residues(res, _proc_names())
    assert not realigned
    assert mapping == {(i + 1, ""): i for i in range(len(SEQ))}


def test_forcefield_names_still_take_fast_path():
    names = [ONE_TO_THREE[a] for a in SEQ]
    names = ["HID" if n == "HIS" else "CYX" if n == "CYS" else n for n in names]
    res = _residues(_pdb_atoms(range(len(SEQ)), names=names))
    _, realigned = map_chain_residues(res, _proc_names())
    assert not realigned


def test_n_terminal_truncation_is_realigned():
    res = _residues(_pdb_atoms(range(10, len(SEQ))))
    mapping, realigned = map_chain_residues(res, _proc_names())
    assert realigned
    assert mapping == {(i + 1, ""): i for i in range(10, len(SEQ))}


def test_internal_gap_is_realigned():
    present = [i for i in range(len(SEQ)) if not 15 <= i < 20]
    mapping, realigned = map_chain_residues(_residues(_pdb_atoms(present)), _proc_names())
    assert realigned
    assert mapping == {(i + 1, ""): i for i in present}


def test_insertion_codes_are_distinct_residues():
    numbering = [(i + 1, "") for i in range(len(SEQ))]
    numbering[21] = (21, "A")  # 1..21, 21A, 23.. : 21A is the 22nd residue
    res = _residues(_pdb_atoms(range(len(SEQ)), numbering=numbering))
    mapping, realigned = map_chain_residues(res, _proc_names())
    assert not realigned
    assert mapping[(21, "A")] == 21 and mapping[(21, "")] == 20


def test_in_chain_hetatm_is_ignored():
    res = _residues(_pdb_atoms(range(len(SEQ))))
    res.insert(5, ((900, ""), "GOL"))
    mapping, realigned = map_chain_residues(res, _proc_names())
    assert realigned
    assert (900, "") not in mapping
    assert mapping == {(i + 1, ""): i for i in range(len(SEQ))}


def test_wrong_chain_raises():
    other = "GGSWDEPNHGGSWDEPNHGGSWDEPNHGGSWDEPNHGGSW"
    res = _residues(_pdb_atoms(range(len(other)), seq=other))
    with pytest.raises(ResidueMappingError, match="identity"):
        map_chain_residues(res, _proc_names(), chain_label="A")


def test_no_amino_acids_raises():
    with pytest.raises(ResidueMappingError):
        map_chain_residues([((1, ""), "LIG")], _proc_names())


# ─── inject_pdb_coords_into_structure ────────────────────────────────────────


def test_injection_lands_on_true_residues_after_truncation():
    """Old behaviour put PDB residue 11 on processed residue 0."""
    struct = _structure()
    injected, unmatched = inject_pdb_coords_into_structure(
        struct, _pdb_atoms(range(10, len(SEQ))), {"A": 0},
    )
    cas = _injected_ca(injected)
    assert all(c is None for c in cas[:10])
    for i in range(10, len(SEQ)):
        np.testing.assert_allclose(cas[i], _ca(i))
    assert len(unmatched) == 10 * len(BACKBONE)


def test_injection_full_chain_unchanged():
    struct = _structure()
    injected, unmatched = inject_pdb_coords_into_structure(
        struct, _pdb_atoms(range(len(SEQ))), {"A": 0},
    )
    assert not unmatched
    for i, ca in enumerate(_injected_ca(injected)):
        np.testing.assert_allclose(ca, _ca(i))


def test_injection_wrong_sequence_raises():
    other = "GGSWDEPNHGGSWDEPNHGGSWDEPNHGGSWDEPNHGGSW"
    with pytest.raises(ResidueMappingError):
        inject_pdb_coords_into_structure(
            _structure(), _pdb_atoms(range(len(other)), seq=other), {"A": 0},
        )


# ─── CA continuity check ─────────────────────────────────────────────────────


def _shifted_injection(n_jumps: int) -> dict:
    """CA coords for every residue, with ``n_jumps`` 10 Å discontinuities."""
    coords, x = {}, 0.0
    for i in range(len(SEQ)):
        x += 10.0 if 0 < i <= n_jumps else 3.8
        coords[i * len(BACKBONE) + 1] = np.array([x, 0, 0], dtype=np.float32)
    return coords


def test_ca_continuity_raises_when_strict():
    struct = _structure()
    with pytest.raises(ResidueMappingError, match="CA-CA"):
        _check_ca_continuity(
            struct, struct.chains[0], _shifted_injection(5), chain_label="A", strict=True,
        )


def test_ca_continuity_only_warns_when_not_strict(caplog):
    struct = _structure()
    with caplog.at_level(logging.WARNING):
        _check_ca_continuity(
            struct, struct.chains[0], _shifted_injection(5), chain_label="A", strict=False,
        )
    assert "CA-CA" in caplog.text


def test_ca_continuity_tolerates_a_couple_of_breaks():
    struct = _structure()
    _check_ca_continuity(
        struct, struct.chains[0], _shifted_injection(2), chain_label="A", strict=True,
    )


# ─── map_chains_by_name ──────────────────────────────────────────────────────


def test_map_chains_by_name_prefers_names(caplog):
    struct = _structure(chain_name="B")
    assert map_chains_by_name(struct, ["A", "B"]) == {"B": 0}
    struct = _structure(chain_name="A")
    with caplog.at_level(logging.WARNING):
        assert map_chains_by_name(struct, ["X"]) == {"X": 0}
    assert "paired by order" in caplog.text
