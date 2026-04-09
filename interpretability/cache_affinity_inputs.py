"""
interpretability/cache_affinity_inputs.py
==========================================
Run the **affinity-rescoring** pipeline (trunk + affinity head ONLY,
NO diffusion) and capture the tensors that are fed to
AffinityModule.forward(), saving them to a .pt file for offline
circuit analysis.

This uses the deterministic code path in
``boltz.affinity_rescoring.inference.affinity_forward`` which:
  1.  Parses a receptor PDB and a docked-pose MOL2
  2.  Injects PDB/MOL2 3-D coordinates into the processed structure
  3.  Runs the trunk (recycling only) + affinity head
  ⇒  NO diffusion, NO confidence, fully deterministic

The captured tensors are:

  - s_inputs : re-embedded single representation (affinity=True)
               shape (B, N_tokens, token_s)
  - z         : masked pair representation (cross-pair mask applied)
                shape (B, N_tokens, N_tokens, token_z)
  - x_pred   : injected PDB/MOL2 coordinates
                shape (B, 1, N_atoms, 3)

Token-type metadata is recovered from the feature dict that flows
through the model (feats["mol_type"], feats["res_type"],
feats["affinity_token_mask"]).

Usage
-----
    python interpretability/cache_affinity_inputs.py \\
        --receptor protein.pdb \\
        --mol2 ligands.mol2 \\
        --ligand_name LIGAND_42 \\
        --out cached_inputs.pt \\
        [--checkpoint ~/.boltz/boltz2_aff.ckpt] \\
        [--device cuda] \\
        [--msa_directory /precomputed/msa]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Ensure the src/ directory is on the path so boltz imports work when
# running this script directly from the interpretability/ folder.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from boltz.data import const  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token label helpers
# ---------------------------------------------------------------------------

# const.tokens: ["<pad>", "-", "ALA", ..., "DN"]  (length 33 = const.num_tokens)
# const.chain_types: ["PROTEIN", "DNA", "RNA", "NONPOLYMER"]
# const.chain_type_ids: {"PROTEIN": 0, "DNA": 1, "RNA": 2, "NONPOLYMER": 3}

def _build_token_type_labels(feats: dict[str, torch.Tensor]) -> list[str]:
    """Build a human-readable label for each token position.

    For polymer tokens (protein / DNA / RNA) the label is the residue
    three-letter code from const.tokens (e.g. "ALA", "DA").
    For non-polymer (ligand) tokens the residue type is always UNK in the
    tokenizer, so we label them as "LIG_<idx>" where <idx> is the
    within-ligand position.

    Source of truth:
      - feats["res_type"]  : one-hot (1, N, 33) over const.tokens
                             (from featurizerv2.py lines 653-654, originally
                              integer token_data["res_type"] then one_hot)
      - feats["mol_type"]  : (1, N) integer  0=PROTEIN 1=DNA 2=RNA 3=NONPOLYMER
                             (from featurizerv2.py line 652)
    """
    # res_type is one-hot: (B, N, 33)  →  argmax to get integer index
    res_type_idx = feats["res_type"][0].argmax(dim=-1)  # (N,)
    mol_type = feats["mol_type"][0]  # (N,)

    labels: list[str] = []
    lig_counter = 0
    for i in range(res_type_idx.shape[0]):
        mt = mol_type[i].item()
        rt = res_type_idx[i].item()
        if mt == const.chain_type_ids["NONPOLYMER"]:
            labels.append(f"LIG_{lig_counter}")
            lig_counter += 1
        elif 0 <= rt < len(const.tokens):
            labels.append(const.tokens[rt])
        else:
            labels.append(f"UNK_{rt}")
    return labels


def _build_is_protein_mask(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Boolean tensor: True where mol_type == PROTEIN (0).

    Source: feats["mol_type"] from featurizerv2.py line 652.
    """
    return (feats["mol_type"][0] == const.chain_type_ids["PROTEIN"]).bool()


def _ligand_token_range(feats: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Return (start, end) indices of the contiguous ligand token span.

    Ligand tokens are identified by feats["affinity_token_mask"] == 1.
    Source: featurizerv2.py, from token_data["affinity_mask"] set in the
    Boltz2 tokenizer (src/boltz/data/tokenize/boltz2.py line 172-174).

    Returns (start, end) such that tokens[start:end] are the ligand.
    If no ligand tokens exist, returns (0, 0).
    """
    mask = feats["affinity_token_mask"][0].bool()
    indices = mask.nonzero(as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        return (0, 0)
    return (indices[0].item(), indices[-1].item() + 1)


# ---------------------------------------------------------------------------
# Extract a single ligand from a multi-mol2 file
# ---------------------------------------------------------------------------

def _extract_ligand_to_tmpfile(
    mol2_path: Path,
    ligand_name: str,
) -> Path:
    """Write a single-ligand MOL2 file for *ligand_name* to a temp file.

    Multi-mol2 files use ``@<TRIPOS>MOLECULE`` as the block separator.
    Returns the path to a temporary MOL2 containing only the target
    ligand, or the original path if it has exactly one molecule.

    Raises ``ValueError`` if the ligand name is not found.
    """
    text = mol2_path.read_text()
    blocks = text.split("@<TRIPOS>MOLECULE")
    # First element is anything before the first block (usually empty).
    blocks = [b for b in blocks if b.strip()]

    if len(blocks) == 1:
        # Single-molecule file — return as-is.
        return mol2_path

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        # First non-blank line after @<TRIPOS>MOLECULE is the molecule name.
        name = lines[0].strip()
        if name == ligand_name:
            fd, tmp = tempfile.mkstemp(suffix=".mol2", prefix=f"{ligand_name}_")
            os.close(fd)
            Path(tmp).write_text("@<TRIPOS>MOLECULE\n" + block)
            return Path(tmp)

    available = []
    for block in blocks:
        lines = block.strip().splitlines()
        if lines:
            available.append(lines[0].strip())
    raise ValueError(
        f"Ligand '{ligand_name}' not found in {mol2_path}.  "
        f"Available names: {available}"
    )


# ---------------------------------------------------------------------------
# Main capture logic
# ---------------------------------------------------------------------------

def cache_affinity_inputs(
    receptor_path: str | Path,
    mol2_path: str | Path,
    ligand_name: str,
    output_path: str | Path,
    checkpoint: str | Path | None = None,
    cache_dir: str | Path = "~/.boltz",
    device: str = "cpu",
    msa_directory: str | Path | None = None,
) -> None:
    """Run the affinity-rescoring pipeline and cache AffinityModule inputs.

    This uses the deterministic rescoring path (trunk + affinity head,
    no diffusion) with coordinate injection from receptor PDB + docked
    MOL2 poses.

    Parameters
    ----------
    receptor_path : path
        Path to the receptor PDB or CIF file.
    mol2_path : path
        Path to a (multi-)MOL2 file with docked ligand poses.
    ligand_name : str
        Name of the target ligand inside the MOL2 file.
    output_path : path
        Where to write the .pt file.
    checkpoint : path, optional
        Affinity model checkpoint.  Defaults to ~/.boltz/boltz2_aff.ckpt.
    cache_dir : path
        Boltz cache directory for CCD data and default checkpoints.
    device : str
        "cpu" or "cuda".
    msa_directory : path, optional
        Directory with pre-computed MSA files (.a3m).  Avoids MSA server.
    """
    from boltz.affinity_rescoring.inference import AffinityModelManager
    from boltz.affinity_rescoring.mol2_parser import MOL2Parser
    from boltz.affinity_rescoring.parsers import (
        get_chain_sequences,
        parse_structure_file,
    )
    from boltz.affinity_rescoring.models import DeviceOption
    from boltz.affinity_rescoring.validation import ChainIdentifier

    warnings.filterwarnings("ignore", ".*that has Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    receptor_path = Path(receptor_path).expanduser().resolve()
    mol2_path = Path(mol2_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir_p = Path(cache_dir).expanduser()
    cache_dir_p.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Extract the target ligand from the MOL2 to a temp file
    # ---------------------------------------------------------------
    single_mol2 = _extract_ligand_to_tmpfile(mol2_path, ligand_name)
    print(f"Ligand '{ligand_name}' → {single_mol2}")

    # ---------------------------------------------------------------
    # 2. Parse receptor + ligand atoms
    # ---------------------------------------------------------------
    receptor_atoms, _meta = parse_structure_file(receptor_path)

    chain_identifier = ChainIdentifier()
    chain_assignment = chain_identifier.identify_chains(
        receptor_atoms, protein_chains=None, ligand_chains=[],
    )
    if not chain_assignment.protein_chains:
        raise ValueError("No protein chain detected in receptor.")

    protein_chain_id = chain_assignment.protein_chains[0]
    protein_atoms = [a for a in receptor_atoms if a.chain_id == protein_chain_id]
    sequences = get_chain_sequences(receptor_atoms)
    protein_seq = sequences.get(protein_chain_id, "")
    print(
        f"Receptor: chain {protein_chain_id}, "
        f"{len(protein_atoms)} atoms, {len(protein_seq)} residues"
    )

    # ---------------------------------------------------------------
    # 3. Convert MOL2 → SMILES, remap atom names, build combined atoms
    # ---------------------------------------------------------------
    mol2_parser = MOL2Parser()
    ligand_structures = mol2_parser.extract_ligands_with_names(single_mol2)
    if not ligand_structures:
        raise ValueError(f"No ligands extracted from {single_mol2}")

    lig_struct = ligand_structures[0]
    print(f"Ligand: {lig_struct.name}, {lig_struct.num_atoms} atoms")

    # SMILES conversion (mirrors AffinityRescorer._mol2_to_smiles)
    smiles = _mol2_to_smiles(lig_struct)
    if smiles is None:
        from boltz.affinity_rescoring.smiles_inference import infer_smiles_from_atoms
        smiles = infer_smiles_from_atoms(lig_struct.atoms)
    if smiles is None:
        raise ValueError(
            f"Cannot convert MOL2 to SMILES for {lig_struct.name}. "
            "Provide a ligand with valid bonding information."
        )
    print(f"SMILES: {smiles}")

    # Remap atom names to canonical (matches Boltz processed structure).
    ligand_chain_id = "L"
    remapped_atoms = _remap_ligand_atoms(lig_struct, smiles, ligand_chain_id)
    if remapped_atoms is None:
        from boltz.affinity_rescoring.models import AtomInfo
        logger.warning(
            f"Canonical atom remapping failed for {lig_struct.name}; "
            "using original mol2 atom names (coords may not inject)."
        )
        remapped_atoms = [
            AtomInfo(
                index=a.index, name=a.name, element=a.element,
                x=a.x, y=a.y, z=a.z,
                chain_id=ligand_chain_id,
                residue_name=a.residue_name,
                residue_number=a.residue_number,
                occupancy=a.occupancy, b_factor=a.b_factor,
                is_hetatm=True,
            )
            for a in lig_struct.atoms
        ]

    combined_atoms = list(protein_atoms) + remapped_atoms

    # ---------------------------------------------------------------
    # 4. Load the Boltz-2 model (affinity-only, no diffusion)
    # ---------------------------------------------------------------
    device_option = DeviceOption.CUDA if device == "cuda" else DeviceOption.CPU
    manager = AffinityModelManager(
        device=device_option, cache_dir=str(cache_dir_p),
    )
    ckpt = str(Path(checkpoint).expanduser()) if checkpoint else None
    model = manager.load_model(checkpoint_path=ckpt)
    print(f"Model loaded on {manager.device}")

    # ---------------------------------------------------------------
    # 5. Register hooks to capture AffinityModule inputs
    # ---------------------------------------------------------------
    # affinity_forward() calls _get_module(model, "affinity_module{,1,2}")
    # which unwraps torch.compile, then calls forward() on the
    # unwrapped module.  We hook that same unwrapped module.
    from boltz.affinity_rescoring.inference import _get_module

    captured: dict[str, object] = {}

    def _make_capture_hook(label: str):
        """Create a pre-hook that captures (args, kwargs) by name."""

        def _hook(module: nn.Module, args: tuple, kwargs: dict):
            # AffinityModule.forward signature:
            #   forward(self, s_inputs, z, x_pred, feats, multiplicity, use_kernels)
            # Called with keyword args from affinity_forward().
            captured[f"{label}_s_inputs"] = (
                kwargs.get("s_inputs", args[0] if len(args) > 0 else None)
            )
            captured[f"{label}_z"] = (
                kwargs.get("z", args[1] if len(args) > 1 else None)
            )
            captured[f"{label}_x_pred"] = (
                kwargs.get("x_pred", args[2] if len(args) > 2 else None)
            )
            captured[f"{label}_feats"] = (
                kwargs.get("feats", args[3] if len(args) > 3 else None)
            )

        return _hook

    hooks = []
    if model.affinity_ensemble:
        for attr, lbl in [("affinity_module1", "m1"), ("affinity_module2", "m2")]:
            mod = _get_module(model, attr)
            hooks.append(
                mod.register_forward_pre_hook(
                    _make_capture_hook(lbl), with_kwargs=True,
                )
            )
    else:
        mod = _get_module(model, "affinity_module")
        hooks.append(
            mod.register_forward_pre_hook(
                _make_capture_hook("m"), with_kwargs=True,
            )
        )

    # ---------------------------------------------------------------
    # 6. Build YAML and run affinity-only inference
    #    (trunk + affinity head, coord injection, no diffusion)
    # ---------------------------------------------------------------
    from boltz.affinity_rescoring.inference import (
        create_affinity_yaml,
        run_direct_affinity_inference,
    )

    # Resolve MSA if provided.
    msa_paths = None
    if msa_directory is not None:
        msa_dir_p = Path(msa_directory).expanduser().resolve()
        # Look for .a3m files for the protein chain.
        for ext in ("*.a3m", "*.csv"):
            candidates = list(msa_dir_p.glob(ext))
            if candidates:
                msa_paths = {protein_chain_id: str(candidates[0])}
                break
        if msa_paths:
            print(f"MSA: {msa_paths}")
        else:
            print(f"WARNING: --msa_directory given but no .a3m/.csv found in {msa_dir_p}")

    yaml_path = create_affinity_yaml(
        protein_sequence=protein_seq,
        ligand_smiles=smiles,
        protein_chain_id=protein_chain_id,
        ligand_chain_id=ligand_chain_id,
    )

    # If we have MSA paths, inject them into the YAML so process_input
    # does not need to contact the MSA server.
    if msa_paths:
        import yaml as _yaml
        with open(yaml_path) as f:
            yaml_data = _yaml.safe_load(f)
        for entry in yaml_data.get("sequences", []):
            if "protein" in entry:
                pid = entry["protein"].get("id", "")
                if pid in msa_paths:
                    entry["protein"]["msa"] = msa_paths[pid]
        with open(yaml_path, "w") as f:
            _yaml.dump(yaml_data, f, default_flow_style=False)

    chain_id_map = {
        protein_chain_id: protein_chain_id,
        ligand_chain_id: ligand_chain_id,
    }

    try:
        results = run_direct_affinity_inference(
            model=model,
            yaml_path=yaml_path,
            pdb_atoms=combined_atoms,
            chain_id_map=chain_id_map,
            cache_dir=cache_dir_p,
            use_msa_server=False,
            device=device,
        )
        print(
            f"Affinity prediction: "
            f"pred={results.get('affinity_pred_value', 'N/A'):.3f}, "
            f"prob={results.get('affinity_probability_binary', 'N/A'):.3f}"
        )
    finally:
        # Clean up hooks regardless of success/failure.
        for h in hooks:
            h.remove()

    # ---------------------------------------------------------------
    # 7. Extract and save captured tensors
    # ---------------------------------------------------------------
    # Pick the first captured set (ensemble module 1 or the single module).
    prefix = "m1" if model.affinity_ensemble else "m"

    s_inputs = captured[f"{prefix}_s_inputs"]
    z = captured[f"{prefix}_z"]
    x_pred = captured[f"{prefix}_x_pred"]
    feats = captured[f"{prefix}_feats"]

    # Build token-type metadata
    token_type_labels = _build_token_type_labels(feats)
    is_protein_mask = _build_is_protein_mask(feats)
    ligand_start, ligand_end = _ligand_token_range(feats)

    # Build a feats dict containing every key that AffinityModule.forward()
    # accesses: token_to_rep_atom, token_pad_mask, mol_type,
    # affinity_token_mask.  We save the full batch-dim-0 slice so the
    # analysis script can reconstruct the feats argument directly.
    feats_to_save = {}
    for key in (
        "token_to_rep_atom",
        "token_pad_mask",
        "mol_type",
        "affinity_token_mask",
        "res_type",
    ):
        if key in feats:
            feats_to_save[key] = feats[key].detach().cpu()

    save_dict = {
        # Core tensors captured at the AffinityModule boundary
        "s_inputs": s_inputs.detach().cpu(),
        "z": z.detach().cpu(),
        "x_pred": x_pred.detach().cpu(),
        # Cross-pair mask (re-derived from feats for convenience)
        "cross_pair_mask": _rebuild_cross_pair_mask(feats).cpu(),
        # Token-type metadata
        "token_type_labels": token_type_labels,
        "is_protein_mask": is_protein_mask.cpu(),
        "ligand_token_range": (ligand_start, ligand_end),
        # Raw feature keys useful for downstream analysis
        "mol_type": feats["mol_type"][0].detach().cpu(),
        "res_type_onehot": feats["res_type"][0].detach().cpu(),
        "affinity_token_mask": feats["affinity_token_mask"][0].detach().cpu(),
        # Full feats subset needed to re-run AffinityModule.forward()
        "feats": feats_to_save,
    }

    torch.save(save_dict, output_path)

    # ---------------------------------------------------------------
    # 8. Print summary
    # ---------------------------------------------------------------
    print("\n=== Cached AffinityModule inputs ===")
    for key, val in save_dict.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:30s}  shape={str(list(val.shape)):20s}  dtype={val.dtype}")
        elif isinstance(val, list):
            print(f"  {key:30s}  list[{len(val)}]  e.g. {val[:5]}")
        elif isinstance(val, tuple):
            print(f"  {key:30s}  {val}")
        else:
            print(f"  {key:30s}  {type(val).__name__}")
    print(f"\nSaved to: {output_path}")

    # Clean up temp mol2 (if we created one)
    if single_mol2 != mol2_path:
        try:
            single_mol2.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Helpers: SMILES conversion + canonical atom remapping
# (replicated from AffinityRescorer private methods to avoid
#  instantiating the full rescorer class)
# ---------------------------------------------------------------------------

def _mol2_to_smiles(ligand) -> str | None:
    """Convert a MOL2 LigandStructure to SMILES via RDKit."""
    try:
        from rdkit import Chem

        mol = Chem.RWMol()
        atom_map = {}
        for atom in ligand.atoms:
            idx = mol.AddAtom(Chem.Atom(atom.element))
            atom_map[atom.index] = idx

        for a1, a2, btype in ligand.bonds:
            if a1 in atom_map and a2 in atom_map:
                bt = {
                    1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC,
                    5: Chem.BondType.SINGLE,
                }.get(btype, Chem.BondType.SINGLE)
                try:
                    mol.AddBond(atom_map[a1], atom_map[a2], bt)
                except Exception:
                    pass

        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol) or None
        except Exception:
            return Chem.MolToSmiles(mol) or None
    except ImportError:
        return None
    except Exception:
        return None


def _remap_ligand_atoms(ligand, smiles: str, ligand_chain_id: str) -> list | None:
    """Remap MOL2 atoms to Boltz canonical atom names via substructure match.

    Returns a list of AtomInfo with canonical names so that coordinate
    injection can match atoms by name, or None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.data.parse.schema import standardize

        std_smiles = standardize(smiles) if smiles else None
        if not std_smiles:
            return None

        smiles_mol_h = Chem.AddHs(Chem.MolFromSmiles(std_smiles))
        canonical_order = AllChem.CanonicalRankAtoms(smiles_mol_h)
        Chem.AssignStereochemistry(smiles_mol_h, force=True, cleanIt=True)
        for atom, can_idx in zip(smiles_mol_h.GetAtoms(), canonical_order):
            atom.SetProp("name", atom.GetSymbol().upper() + str(can_idx + 1))
        smiles_mol_noh = Chem.RemoveHs(smiles_mol_h, sanitize=False)

        mol2_rdmol = Chem.RWMol()
        mol2_atom_map = {}
        for a in ligand.atoms:
            ridx = mol2_rdmol.AddAtom(Chem.Atom(a.element))
            mol2_atom_map[a.index] = ridx
        _bt = {
            1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC,
            5: Chem.BondType.SINGLE,
        }
        for a1, a2, btype in ligand.bonds:
            if a1 in mol2_atom_map and a2 in mol2_atom_map:
                try:
                    mol2_rdmol.AddBond(
                        mol2_atom_map[a1], mol2_atom_map[a2],
                        _bt.get(btype, Chem.BondType.SINGLE),
                    )
                except Exception:
                    pass
        try:
            Chem.SanitizeMol(mol2_rdmol)
        except Exception:
            pass

        match = mol2_rdmol.GetSubstructMatch(smiles_mol_noh)
        if not match or len(match) != smiles_mol_noh.GetNumAtoms():
            match = mol2_rdmol.GetSubstructMatch(smiles_mol_noh, useChirality=False)
        if not match or len(match) != smiles_mol_noh.GetNumAtoms():
            return None

        rdkit_idx_to_info = {mol2_atom_map[a.index]: a for a in ligand.atoms}
        residue_number = ligand.atoms[0].residue_number if ligand.atoms else 1

        remapped = []
        for smiles_idx in range(smiles_mol_noh.GetNumAtoms()):
            mol2_rdkit_idx = match[smiles_idx]
            src = rdkit_idx_to_info.get(mol2_rdkit_idx)
            if src is None:
                return None
            can_name = smiles_mol_noh.GetAtomWithIdx(smiles_idx).GetProp("name")
            remapped.append(
                AtomInfo(
                    index=src.index, name=can_name, element=src.element,
                    x=src.x, y=src.y, z=src.z,
                    chain_id=ligand_chain_id,
                    residue_name=src.residue_name,
                    residue_number=residue_number,
                    occupancy=src.occupancy, b_factor=src.b_factor,
                    is_hetatm=True,
                )
            )
        return remapped

    except Exception:
        return None


def _rebuild_cross_pair_mask(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Re-derive the cross-pair mask used by affinity_forward().

    This replicates the mask construction at inference.py / boltz2.py:
        pad_token_mask = feats["token_pad_mask"][0]
        rec_mask = feats["mol_type"][0] == 0
        rec_mask = rec_mask * pad_token_mask
        lig_mask = feats["affinity_token_mask"][0].to(torch.bool)
        lig_mask = lig_mask * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, None] * rec_mask[None, :]
            + rec_mask[:, None] * lig_mask[None, :]
            + lig_mask[:, None] * lig_mask[None, :]
        )
    """
    pad_token_mask = feats["token_pad_mask"][0]
    rec_mask = (feats["mol_type"][0] == 0).float() * pad_token_mask
    lig_mask = feats["affinity_token_mask"][0].to(torch.bool).float() * pad_token_mask
    cross_pair_mask = (
        lig_mask[:, None] * rec_mask[None, :]
        + rec_mask[:, None] * lig_mask[None, :]
        + lig_mask[:, None] * lig_mask[None, :]
    )

    # Exclude diagonal (self-pairs).  The real AffinityHeadsTransformer.forward()
    # multiplies by (1 - eye) before mean-pooling so that ligand self-pairs
    # z[i,i,:] do not contaminate the pooled representation.
    N = cross_pair_mask.shape[0]
    cross_pair_mask = cross_pair_mask * (
        1 - torch.eye(N, device=cross_pair_mask.device)
    )
    assert cross_pair_mask.diagonal().sum() == 0

    return cross_pair_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cache AffinityModule inputs via the affinity-rescoring pipeline "
            "(trunk + affinity head, NO diffusion).  Input: receptor PDB + "
            "docked MOL2 pose.  Output: .pt file for offline interpretability."
        ),
    )
    parser.add_argument(
        "--receptor", required=True,
        help="Path to receptor PDB or CIF file.",
    )
    parser.add_argument(
        "--mol2", required=True,
        help="Path to (multi-)MOL2 file with docked ligand poses.",
    )
    parser.add_argument(
        "--ligand_name", required=True,
        help="Name of the target ligand inside the MOL2 file.",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output .pt file path (e.g. cached_inputs_tyk2.pt).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Affinity model checkpoint.  Defaults to auto-download.",
    )
    parser.add_argument(
        "--cache", default="~/.boltz",
        help="Boltz cache directory.",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device to run on.",
    )
    parser.add_argument(
        "--msa_directory", default=None,
        help="Directory with pre-computed MSA files (.a3m).  "
             "Avoids contacting the MSA server.",
    )
    args = parser.parse_args()

    cache_affinity_inputs(
        receptor_path=args.receptor,
        mol2_path=args.mol2,
        ligand_name=args.ligand_name,
        output_path=args.out,
        checkpoint=args.checkpoint,
        cache_dir=args.cache,
        device=args.device,
        msa_directory=args.msa_directory,
    )


if __name__ == "__main__":
    main()
