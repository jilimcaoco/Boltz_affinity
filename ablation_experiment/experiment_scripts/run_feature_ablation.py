#!/usr/bin/env python3
"""Feature ablation study for Boltz affinity rescoring.

Systematically ablates the three main information channels that feed the
affinity head, plus sub-components of s_inputs, to identify which
features drive the model's predictions.

Information channels
--------------------
1. **z_trunk**      : Pair representation from the trunk pairformer
                      (carries MSA, templates, relative position, bonds)
2. **s_inputs**     : Single representation from InputEmbedder
                      (atom encoder + residue type + MSA profile)
3. **distogram**    : 3D coordinate distances (embedded distance matrix)

Sub-components of s_inputs
--------------------------
- atom_encoder  : 3D geometry, charges, element types, atom names
- res_type      : One-hot residue / token type encoding
- msa_profile   : MSA sequence profile + deletion mean

Ablation operators (Task 1)
----------------------------
Each of the 8 factorial cells over {distogram, z_trunk, s_inputs} (baseline,
the 3 no_*, the 3 *_only, bias_only) can ablate its excluded channel(s)
under three operators:

  - ``zero``     : replace with zeros (original behavior, still the default
                   and still what the bare experiment names mean).
  - ``resample`` : replace with the same channel computed for a *different*
                   ligand on the *same* receptor, token-count matched. Five
                   independent donor draws (seeds 11/22/33/44/55).
  - ``mean``     : replace with the per-position mean over all other cached
                   donor complexes on the same receptor.

Zeroing substitutes an input the network never saw in training, which
confounds "this channel carries signal" with "the network reacts badly to
an out-of-distribution input." Resample/mean substitute a channel value the
network *has* plausibly seen, isolating the former. See
``trunk_cache.py`` and the ``no_distogram__resample__d11`` naming
convention below.

Trunk caching (Task 4)
-----------------------
The trunk (recycled MSA + Pairformer, ~10s) is computed once per
(receptor, ligand) and cached to disk (see ``trunk_cache.py``); every
ablation experiment for that ligand — including all operators — replays
only the cheap affinity head (~10ms) against the cached trunk. This is also
what makes the resample/mean operators possible: they need another
ligand's trunk output as a donor, which requires it to already be cached.
Each receptor is therefore processed in two passes: pass 1 populates the
cache for every ligand, pass 2 runs every experiment against it. Run
``verify_trunk_cache.py`` to confirm cached-replay matches an uncached
end-to-end recomputation before trusting any result.

Experiment matrix
-----------------
Each experiment is a named combination of ablation flags:

  Name                    distogram  z_trunk  s_inputs  atom_enc  msa_prof  res_type
  ───────────────────────────────────────────────────────────────────────────────────
  baseline                keep       keep     keep      keep      keep      keep
  no_distogram            ZERO       keep     keep      keep      keep      keep
  no_z_trunk              keep       ZERO     keep      keep      keep      keep
  no_s_inputs             keep       keep     ZERO      -         -         -
  distogram_only          keep       ZERO     ZERO      -         -         -
  z_trunk_only            ZERO       keep     ZERO      -         -         -
  s_inputs_only           ZERO       ZERO     keep      keep      keep      keep
  bias_only               ZERO       ZERO     ZERO      -         -         -
  no_atom_encoder         keep       keep     keep      ZERO      keep      keep
  no_msa_profile          keep       keep     keep      keep      ZERO      keep
  no_res_type             keep       keep     keep      keep      keep      ZERO
  only_atom_encoder       ZERO       ZERO     keep      keep      ZERO      ZERO
  only_msa_profile        ZERO       ZERO     keep      ZERO      keep      ZERO
  only_res_type           ZERO       ZERO     keep      ZERO      ZERO      keep

Each of the 7 non-baseline rows above also has 5 ``__resample__d<seed>``
variants and 1 ``__mean`` variant (e.g. ``no_distogram__resample__d11``,
``bias_only__mean``) that substitute donor content instead of zeros for
whichever channel(s) that row ablates. Bare names remain aliases for the
``zero`` operator — nothing about existing output changes.

Usage
-----
# Quick test on one receptor:
python run_feature_ablation.py \\
    --receptor ../receptors/AA2AR_receptor.pdb \\
    --ligands ../DOCK3.8_poses/AA2AR_poses.mol2 \\
    --max-ligands 5 \\
    --experiments baseline no_distogram no_z_trunk no_s_inputs \\
    --output ../results/ablation/AA2AR_feature_ablation.csv

# Full matrix on all receptors:
python run_feature_ablation.py \\
    --all-receptors \\
    --max-ligands 20 \\
    --output ../results/ablation/all_feature_ablation.csv

# Channel-only experiments:
python run_feature_ablation.py \\
    --all-receptors --max-ligands 20 \\
    --experiments baseline no_distogram no_z_trunk no_s_inputs \\
        distogram_only z_trunk_only s_inputs_only bias_only \\
    --output ../results/ablation/channel_ablation.csv

# Sub-component experiments:
python run_feature_ablation.py \\
    --all-receptors --max-ligands 20 \\
    --experiments baseline no_atom_encoder no_msa_profile no_res_type \\
        only_atom_encoder only_msa_profile only_res_type \\
    --output ../results/ablation/subcomponent_ablation.csv

# Resample/mean operator variants (needs enough ligands per receptor for
# donors to exist -- see trunk_cache.py):
python run_feature_ablation.py \\
    --all-receptors --max-ligands 40 \\
    --experiments baseline no_z_trunk no_z_trunk__mean \\
        no_z_trunk__resample__d11 no_z_trunk__resample__d22 \\
        no_z_trunk__resample__d33 no_z_trunk__resample__d44 no_z_trunk__resample__d55 \\
    --output ../results/ablation/resample_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the repo's src/ is on sys.path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import trunk_cache  # noqa: E402

logger = logging.getLogger(__name__)


# ── Experiment definitions ────────────────────────────────────────────

@dataclass(frozen=True)
class AblationExperiment:
    """Defines one ablation configuration."""
    name: str
    distogram_mask_mode: str = "none"
    zero_z_trunk: bool = False
    zero_s_inputs: bool = False
    zero_atom_encoder: bool = False
    zero_msa_profile: bool = False
    zero_res_type: bool = False
    # Pose-noise perturbation (proposal #1). ``pose_noise_sigma`` is the
    # Gaussian σ in Å added to atom coordinates before the affinity head.
    # ``pose_noise_target`` selects which atoms are perturbed: "ligand"
    # (default — only ligand atoms), "receptor", or "all".
    # ``pose_noise_seed`` is the per-experiment RNG seed for reproducibility.
    pose_noise_sigma: float = 0.0
    pose_noise_target: str = "ligand"
    pose_noise_seed: Optional[int] = None
    # Task 1: ablation operator. "zero" (default) is the original behavior,
    # governed entirely by the flags above -- unchanged. "resample"/"mean"
    # ignore distogram_mask_mode/zero_z_trunk/zero_s_inputs and instead
    # substitute donor content (see trunk_cache.py) for every channel named
    # in resample_channels, drawn from the same receptor's other ligands.
    operator: str = "zero"
    resample_channels: Tuple[str, ...] = ()
    donor_seed: Optional[int] = None


# Registry of all named experiments
EXPERIMENTS: Dict[str, AblationExperiment] = {}

def _register(*args, **kwargs):
    exp = AblationExperiment(*args, **kwargs)
    EXPERIMENTS[exp.name] = exp

# Channel-level experiments
_register("baseline")
_register("no_distogram",    distogram_mask_mode="zero_all")
_register("no_z_trunk",      zero_z_trunk=True)
_register("no_s_inputs",     zero_s_inputs=True)
_register("distogram_only",  zero_z_trunk=True,  zero_s_inputs=True)
_register("z_trunk_only",    distogram_mask_mode="zero_all", zero_s_inputs=True)
_register("s_inputs_only",   distogram_mask_mode="zero_all", zero_z_trunk=True)
_register("bias_only",       distogram_mask_mode="zero_all", zero_z_trunk=True, zero_s_inputs=True)

# Sub-component experiments (ablate one component of s_inputs)
_register("no_atom_encoder", zero_atom_encoder=True)
_register("no_msa_profile",  zero_msa_profile=True)
_register("no_res_type",     zero_res_type=True)

# Sub-component isolation (keep only one component of s_inputs, zero everything else)
_register("only_atom_encoder",
          distogram_mask_mode="zero_all", zero_z_trunk=True,
          zero_msa_profile=True, zero_res_type=True)
_register("only_msa_profile",
          distogram_mask_mode="zero_all", zero_z_trunk=True,
          zero_atom_encoder=True, zero_res_type=True)
_register("only_res_type",
          distogram_mask_mode="zero_all", zero_z_trunk=True,
          zero_atom_encoder=True, zero_msa_profile=True)

# ── Resample/mean operator variants of the 2^3 factorial (Task 1) ─────
# The 8-cell factorial over {distogram, z_trunk, s_inputs} is exactly the
# 8 experiments above (7 non-baseline + baseline itself, which ablates
# nothing). For each non-baseline cell, generate a resample variant (5
# donor seeds) and a mean variant that substitute donor content instead of
# zero for whichever channel(s) that cell excludes. Scoped to the three
# top-level channels only -- no sub-component resample in this phase.
_TOP_LEVEL_CHANNELS = ("distogram", "z_trunk", "s_inputs")
_FACTORIAL_CELLS_KEPT_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "baseline":       ("distogram", "z_trunk", "s_inputs"),
    "no_distogram":   ("z_trunk", "s_inputs"),
    "no_z_trunk":     ("distogram", "s_inputs"),
    "no_s_inputs":    ("distogram", "z_trunk"),
    "distogram_only": ("distogram",),
    "z_trunk_only":   ("z_trunk",),
    "s_inputs_only":  ("s_inputs",),
    "bias_only":      (),
}
_RESAMPLE_DONOR_SEEDS = (11, 22, 33, 44, 55)

for _cell_name, _kept in _FACTORIAL_CELLS_KEPT_CHANNELS.items():
    _ablated = tuple(c for c in _TOP_LEVEL_CHANNELS if c not in _kept)
    if not _ablated:
        continue  # baseline: nothing to resample
    for _seed in _RESAMPLE_DONOR_SEEDS:
        _register(
            f"{_cell_name}__resample__d{_seed}",
            operator="resample",
            resample_channels=_ablated,
            donor_seed=_seed,
        )
    _register(
        f"{_cell_name}__mean",
        operator="mean",
        resample_channels=_ablated,
    )

# ── Pose-noise sweep (proposal #1) ────────────────────────────────────
# Perturb only ligand atoms by Gaussian noise of σ ∈ {0.25, 0.5, 1, 2, 5, 10} Å.
# Each σ is repeated under 3 seeds so the bootstrap analysis can average
# out the noise draw.  Receptor-side and "all-atom" controls are included.
_NOISE_SIGMAS_LIG = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
_NOISE_SEEDS = (101, 202, 303)
for _sigma in _NOISE_SIGMAS_LIG:
    sigma_tag = f"{_sigma:g}".replace(".", "p")
    for _seed in _NOISE_SEEDS:
        _register(
            f"lig_noise_{sigma_tag}_s{_seed}",
            pose_noise_sigma=_sigma,
            pose_noise_target="ligand",
            pose_noise_seed=_seed,
        )
# Receptor-only and all-atom controls at moderate σ for comparison.
for _sigma in (1.0, 5.0):
    sigma_tag = f"{_sigma:g}".replace(".", "p")
    for _seed in _NOISE_SEEDS:
        _register(
            f"rec_noise_{sigma_tag}_s{_seed}",
            pose_noise_sigma=_sigma,
            pose_noise_target="receptor",
            pose_noise_seed=_seed,
        )
        _register(
            f"all_noise_{sigma_tag}_s{_seed}",
            pose_noise_sigma=_sigma,
            pose_noise_target="all",
            pose_noise_seed=_seed,
        )

ALL_EXPERIMENT_NAMES = list(EXPERIMENTS.keys())


# Default subset for the main DUD-Z ablation run.  Keeps the full matrix
# tractable while still spanning channels, sub-components, and the pose-noise
# sweep. Resample/mean variants are NOT included by default (42 extra
# heavy experiments) -- request them explicitly via --experiments.
DEFAULT_EXPERIMENT_NAMES = (
    [
        "baseline", "no_distogram", "no_z_trunk", "no_s_inputs",
        "distogram_only", "z_trunk_only", "s_inputs_only", "bias_only",
        "no_atom_encoder", "no_msa_profile", "no_res_type",
        "only_atom_encoder", "only_msa_profile", "only_res_type",
    ]
    + [n for n in ALL_EXPERIMENT_NAMES if n.startswith("lig_noise_")]
    + [n for n in ALL_EXPERIMENT_NAMES if n.startswith("rec_noise_")]
    + [n for n in ALL_EXPERIMENT_NAMES if n.startswith("all_noise_")]
)


# ── Helpers ───────────────────────────────────────────────────────────

def default_paths():
    experiment_root = _script_dir.parent
    receptors_dir = experiment_root / "receptors"
    poses_dir = experiment_root / "DOCK3.8_poses"
    output_dir = experiment_root / "results" / "ablation"
    trunk_cache_dir = experiment_root / "results" / "trunk_cache"
    return receptors_dir, poses_dir, output_dir, trunk_cache_dir


def parse_args() -> argparse.Namespace:
    receptors_dir, poses_dir, output_dir, trunk_cache_dir = default_paths()

    parser = argparse.ArgumentParser(
        description="Feature ablation study for Boltz affinity rescoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--receptor", type=Path, help="Single receptor PDB file.")
    group.add_argument("--all-receptors", action="store_true",
                       help="Run on all receptor/pose pairs.")

    parser.add_argument("--ligands", type=Path, default=None,
                        help="MOL2 ligand poses (required with --receptor).")
    parser.add_argument("--receptors-dir", type=Path, default=receptors_dir)
    parser.add_argument("--poses-dir", type=Path, default=poses_dir)

    # Experiment selection
    parser.add_argument(
        "--experiments", nargs="+", default=list(DEFAULT_EXPERIMENT_NAMES),
        choices=ALL_EXPERIMENT_NAMES,
        help="Which ablation experiments to run.",
    )

    # Processing
    parser.add_argument("--max-ligands", type=int, default=None)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--recycling-steps", type=int, default=5)
    parser.add_argument("--use-msa-server", action="store_true", default=False)
    parser.add_argument("--trunk-cache-dir", type=Path, default=trunk_cache_dir,
                        help="Disk cache for per-(receptor,ligand) trunk output "
                             "(Task 4). Reused across runs; also the donor pool "
                             "for resample/mean operators.")
    parser.add_argument("--rebuild-trunk-cache", action="store_true", default=False,
                        help="Recompute and overwrite cached trunk entries even if present.")

    # Output
    parser.add_argument("--output", "-o", type=Path,
                        default=output_dir / "feature_ablation_results.csv")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    if args.receptor and not args.ligands:
        parser.error("--ligands is required when using --receptor.")

    return args


def setup_logging(level: str):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def discover_pairs(receptors_dir: Path, poses_dir: Path) -> List[tuple]:
    pairs = []
    for receptor_path in sorted(receptors_dir.glob("*_receptor.pdb")):
        receptor_id = receptor_path.stem.removesuffix("_receptor")
        poses_path = poses_dir / f"{receptor_id}_poses.mol2"
        if poses_path.exists():
            pairs.append((receptor_id, receptor_path, poses_path))
        else:
            logger.warning(f"No poses found for {receptor_id}, skipping.")
    return pairs


def _remap_ligand_atoms(ligand, smiles, ligand_chain_id):
    """Remap mol2 ligand atoms to canonical Boltz naming via RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol_smiles = Chem.MolFromSmiles(smiles)
        if mol_smiles is None:
            return None
        mol_smiles = Chem.AddHs(mol_smiles)
        AllChem.EmbedMolecule(mol_smiles, randomSeed=42)
        ranks = list(AllChem.CanonicalRankAtoms(mol_smiles))

        canon_names = {}
        for i, atom in enumerate(mol_smiles.GetAtoms()):
            elem = atom.GetSymbol().upper()
            rank = ranks[i]
            canon_names[i] = f"{elem}{rank + 1}"

        from boltz.affinity_rescoring.models import AtomInfo as _AI

        elem_canon = {}
        for i, atom in enumerate(mol_smiles.GetAtoms()):
            if atom.GetAtomicNum() == 1:
                continue
            elem = atom.GetSymbol().upper()
            if elem not in elem_canon:
                elem_canon[elem] = []
            elem_canon[elem].append((i, canon_names[i]))

        elem_mol2 = {}
        for a in ligand.atoms:
            elem = a.element.upper() if a.element else ""
            if elem == "H":
                continue
            if elem not in elem_mol2:
                elem_mol2[elem] = []
            elem_mol2[elem].append(a)

        remapped = []
        for elem in elem_mol2:
            if elem not in elem_canon:
                continue
            canon_list = elem_canon[elem]
            mol2_list = elem_mol2[elem]
            for j, a in enumerate(mol2_list):
                if j < len(canon_list):
                    _, cname = canon_list[j]
                    remapped.append(_AI(
                        index=a.index, name=cname, element=a.element,
                        x=a.x, y=a.y, z=a.z,
                        chain_id=ligand_chain_id,
                        residue_name=a.residue_name,
                        residue_number=a.residue_number,
                        occupancy=a.occupancy,
                        b_factor=a.b_factor,
                        is_hetatm=True,
                    ))

        return remapped if remapped else None
    except Exception:
        return None


# ── Per-ligand featurization (shared by both cache and replay passes) ──

def _featurize_ligand(
    ligand,
    receptor_id: str,
    protein_chain_id: str,
    protein_seq: str,
    protein_atoms,
    msa_path: Optional[str],
    ccd,
    mol_dir: Path,
    tokenizer,
    cropper,
    featurizer,
    device,
) -> Tuple[Optional[dict], Optional[str]]:
    """Run the full (cheap, CPU-side) featurization pipeline for one ligand
    pose and return ``(batch, error)``. ``batch`` is None iff ``error`` is
    set. This is intentionally re-run once per pass (cache pass + replay
    pass) rather than cached itself -- unlike the trunk forward, this is not
    the expensive step (see module docstring)."""
    import numpy as np
    import torch
    from torch import Tensor

    from boltz.affinity_rescoring.coord_injection import (
        build_chain_id_map,
        inject_pdb_coords_into_structure,
        save_pre_affinity_structure,
    )
    from boltz.affinity_rescoring.mol2_parser import MOL2Parser  # noqa: F401 (import parity)
    from boltz.data import const
    from boltz.data.crop.affinity import AffinityCropper  # noqa: F401 (import parity)
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer  # noqa: F401
    from boltz.data.mol import load_molecules
    from boltz.data.module.inferencev2 import load_input
    from boltz.data.types import Record, StructureV2
    from boltz.main import process_input

    ligand_name = ligand.name

    smiles = None
    try:
        from boltz.affinity_rescoring.smiles_inference import infer_smiles_from_atoms
        smiles = infer_smiles_from_atoms(ligand.atoms)
    except Exception:
        pass

    if smiles is None:
        return None, "SMILES inference failed"

    ligand_chain_id = "L"
    ligand_atoms_remapped = _remap_ligand_atoms(ligand, smiles, ligand_chain_id)
    if ligand_atoms_remapped is None:
        from boltz.affinity_rescoring.models import AtomInfo as _AI
        ligand_atoms_remapped = [
            _AI(
                index=a.index, name=a.name, element=a.element,
                x=a.x, y=a.y, z=a.z,
                chain_id=ligand_chain_id,
                residue_name=a.residue_name,
                residue_number=a.residue_number,
                occupancy=a.occupancy,
                b_factor=a.b_factor,
                is_hetatm=True,
            )
            for a in ligand.atoms
        ]

    combined_atoms = list(protein_atoms) + ligand_atoms_remapped

    import tempfile
    import yaml

    work_dir = Path(tempfile.mkdtemp(prefix="boltz_ablation_"))
    try:
        protein_entry = {"id": protein_chain_id, "sequence": protein_seq}
        if msa_path:
            protein_entry["msa"] = msa_path
        yaml_data = {
            "version": 1,
            "sequences": [
                {"protein": protein_entry},
                {"ligand": {"id": ligand_chain_id, "smiles": smiles}},
            ],
            "properties": [
                {"affinity": {"binder": ligand_chain_id}},
            ],
        }
        yaml_path = work_dir / "input.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        out_dir = work_dir / "output"
        msa_dir = out_dir / "msa"
        records_dir = out_dir / "processed" / "records"
        structure_dir = out_dir / "processed" / "structures"
        processed_msa_dir = out_dir / "processed" / "msa"
        processed_constraints_dir = out_dir / "processed" / "constraints"
        processed_templates_dir = out_dir / "processed" / "templates"
        processed_mols_dir = out_dir / "processed" / "mols"
        predictions_dir = out_dir / "predictions"

        for d in [out_dir, msa_dir, records_dir, structure_dir,
                  processed_msa_dir, processed_constraints_dir,
                  processed_templates_dir, processed_mols_dir,
                  predictions_dir]:
            d.mkdir(parents=True, exist_ok=True)

        process_input(
            path=yaml_path,
            ccd=ccd,
            msa_dir=msa_dir,
            mol_dir=mol_dir,
            boltz2=True,
            use_msa_server=False,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="paired+unpaired",
            msa_server_username=None,
            msa_server_password=None,
            api_key_header=None,
            api_key_value=None,
            max_msa_seqs=8192,
            processed_msa_dir=processed_msa_dir,
            processed_constraints_dir=processed_constraints_dir,
            processed_templates_dir=processed_templates_dir,
            processed_mols_dir=processed_mols_dir,
            structure_dir=structure_dir,
            records_dir=records_dir,
        )

        record_files = list(records_dir.glob("*.json"))
        if not record_files:
            raise RuntimeError("process_input produced no records.")
        record = Record.load(record_files[0])
        processed_struct = StructureV2.load(structure_dir / f"{record.id}.npz")

        chain_id_map_yaml = {
            protein_chain_id: protein_chain_id,
            ligand_chain_id: ligand_chain_id,
        }
        yaml_chain_ids = list(chain_id_map_yaml.values())
        asym_map = build_chain_id_map(processed_struct, yaml_chain_ids)
        full_map = {}
        for pdb_cid, yaml_cid in chain_id_map_yaml.items():
            if yaml_cid in asym_map:
                full_map[pdb_cid] = asym_map[yaml_cid]

        injected, unmatched = inject_pdb_coords_into_structure(
            processed_struct, combined_atoms, full_map
        )
        save_pre_affinity_structure(injected, predictions_dir, record.id)

        input_data = load_input(
            record=record,
            target_dir=predictions_dir,
            msa_dir=processed_msa_dir,
            constraints_dir=processed_constraints_dir,
            template_dir=processed_templates_dir,
            extra_mols_dir=processed_mols_dir,
            affinity=True,
        )

        tokenized = tokenizer.tokenize(input_data)
        tokenized = cropper.crop(tokenized, max_tokens=256, max_atoms=2048)

        molecules = {}
        molecules.update(ccd)
        if input_data.extra_mols:
            molecules.update(input_data.extra_mols)
        mol_names = set(tokenized.tokens["res_name"].tolist())
        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(mol_dir, mol_names))

        random = np.random.default_rng(42)
        features = featurizer.process(
            tokenized,
            molecules=molecules,
            random=random,
            training=False,
            max_atoms=None,
            max_tokens=None,
            max_seqs=const.max_msa_seqs,
            pad_to_max_seqs=False,
            single_sequence_prop=0.0,
            compute_frames=True,
            inference_pocket_constraints=None,
            inference_contact_constraints=None,
            compute_constraint_features=True,
            override_method=None,
            compute_affinity=True,
        )

        batch = {}
        for k, v in features.items():
            if isinstance(v, Tensor):
                batch[k] = v.unsqueeze(0).to(device)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            elif k == "affinity_mw":
                batch[k] = [v]
            else:
                batch[k] = v

        return batch, None
    except Exception as e:
        return None, str(e)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _coords_affinity(batch) -> "object":
    coords_affinity = batch["coords"].detach()
    if coords_affinity.dim() == 3:
        coords_affinity = coords_affinity[None]
    elif coords_affinity.dim() == 4 and coords_affinity.shape[1] > 1:
        coords_affinity = coords_affinity[:, :1]
    return coords_affinity


# ── Core ablation runner ──────────────────────────────────────────────

def run_ablation_for_pair(
    receptor_id: str,
    receptor_path: Path,
    ligands_path: Path,
    model,
    experiments: List[AblationExperiment],
    max_ligands: Optional[int],
    recycling_steps: int,
    use_msa_server: bool,
    cache_dir: Path,
    trunk_cache_dir: Path,
    rebuild_trunk_cache: bool,
    output_path: Optional[Path] = None,
    fieldnames: Optional[List[str]] = None,
) -> tuple:
    """Run all ablation experiments for one receptor/ligand-set pair.

    Two passes:
      1. Featurize + trunk-forward every ligand once, caching the trunk
         output to ``trunk_cache_dir`` (Task 4).
      2. Featurize every ligand again (cheap) and replay every requested
         experiment's affinity head against the *cached* trunk -- for
         ``zero`` this is the query's own cached trunk; for
         ``resample``/``mean`` it additionally substitutes donor content
         from other cached ligands on the same receptor (Task 1).

    Results are written to ``output_path`` incrementally after each ligand
    so that partial results are preserved if the job is killed.  Returns
    ``(n_done, n_failed)`` counts rather than the full row list.
    """
    import numpy as np
    import torch

    from boltz.affinity_rescoring.inference import (
        _affinity_input_embed_with_ablation,
        _build_cross_pair_mask,
        _get_module,
        affinity_head_forward,
        affinity_trunk_forward,
    )
    from boltz.affinity_rescoring.mol2_parser import MOL2Parser
    from boltz.affinity_rescoring.parsers import (
        get_chain_sequences,
        get_seqres_sequences,
        parse_structure_file,
    )
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer

    mol_dir = cache_dir / "mols"
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    ccd = load_canonicals(mol_dir)
    tokenizer = Boltz2Tokenizer()
    cropper = AffinityCropper()
    featurizer = Boltz2Featurizer()

    protein_atoms, _metadata = parse_structure_file(receptor_path)
    protein_chain_id = "A"
    merged_refs = get_seqres_sequences(receptor_path)
    sequences = get_chain_sequences(protein_atoms, reference_sequences=merged_refs)
    if protein_chain_id not in sequences:
        protein_chain_id = next(iter(sequences))
    protein_seq = sequences[protein_chain_id]
    logger.info(f"[{receptor_id}] Protein chain {protein_chain_id}: {len(protein_seq)} residues")

    import tempfile
    msa_cache_dir = Path(tempfile.mkdtemp(prefix="boltz_msa_cache_"))
    msa_path = None
    if use_msa_server:
        try:
            from boltz.main import compute_msa
            target_id = f"receptor_{protein_chain_id}"
            data = {target_id: protein_seq}
            compute_msa(
                data=data, target_id=target_id, msa_dir=msa_cache_dir,
                msa_server_url="https://api.colabfold.com",
                msa_pairing_strategy="paired+unpaired",
            )
            for ext in ("*.a3m", "*.csv"):
                found = list(msa_cache_dir.glob(ext))
                if found:
                    msa_path = str(found[0].resolve())
                    break
            if msa_path:
                logger.info(f"[{receptor_id}] MSA pre-computed: {msa_path}")
        except Exception as e:
            logger.warning(f"[{receptor_id}] MSA generation failed: {e}")

    if use_msa_server and msa_path is None:
        logger.error(f"[{receptor_id}] Skipping receptor: MSA pre-computation failed.")
        return 0, 0

    mol2_parser = MOL2Parser()
    ligands = mol2_parser.extract_ligands_with_names(ligands_path)
    if max_ligands is not None:
        ligands = ligands[:max_ligands]
    logger.info(f"[{receptor_id}] {len(ligands)} ligands to process")

    def featurize(ligand):
        return _featurize_ligand(
            ligand, receptor_id, protein_chain_id, protein_seq, protein_atoms,
            msa_path, ccd, mol_dir, tokenizer, cropper, featurizer, device,
        )

    # ── Pass 1: populate the trunk cache ────────────────────────────
    n_cache_built = 0
    n_cache_failed = 0
    for lig_idx, ligand in enumerate(ligands):
        ligand_name = ligand.name
        if not rebuild_trunk_cache and trunk_cache.load_cache_entry(
            trunk_cache_dir, receptor_id, ligand_name
        ) is not None:
            continue

        batch, err = featurize(ligand)
        if err is not None:
            logger.warning(f"[{receptor_id}] cache pass: {ligand_name} featurization failed: {err}")
            n_cache_failed += 1
            continue
        try:
            trunk_out = affinity_trunk_forward(model, batch, recycling_steps=recycling_steps)
            s_inputs = _affinity_input_embed_with_ablation(model, batch)
            token_to_rep_atom = batch["token_to_rep_atom"][0].float()
            coords_affinity = _coords_affinity(batch)[0]
            token_repr_pos = token_to_rep_atom @ coords_affinity
            trunk_cache.save_cache_entry(
                trunk_cache_dir, receptor_id, ligand_name,
                z=trunk_out["z"], s_inputs=s_inputs, token_repr_pos=token_repr_pos,
                use_kernels=trunk_out["use_kernels"],
                meta={"recycling_steps": recycling_steps},
            )
            n_cache_built += 1
            logger.info(f"[{receptor_id}] cached trunk {lig_idx + 1}/{len(ligands)}: {ligand_name}")
        except Exception as e:
            logger.error(f"[{receptor_id}] cache pass: {ligand_name} trunk forward failed: {e}")
            n_cache_failed += 1

    logger.info(f"[{receptor_id}] trunk cache: {n_cache_built} built, {n_cache_failed} failed, "
                f"{len(trunk_cache.list_cached_ligands(trunk_cache_dir, receptor_id))} total cached")

    # ── Pass 2: replay every experiment against the cache ───────────
    n_done = 0
    n_failed = 0

    for lig_idx, ligand in enumerate(ligands):
        ligand_name = ligand.name
        logger.info(f"[{receptor_id}] Ligand {lig_idx + 1}/{len(ligands)}: {ligand_name}")

        entry = trunk_cache.load_cache_entry(trunk_cache_dir, receptor_id, ligand_name)
        if entry is None:
            logger.warning(f"  No cached trunk for {ligand_name}, skipping (see cache-pass log above).")
            _rows = [_error_row(receptor_id, ligand_name, exp, "no cached trunk (cache pass failed)")
                     for exp in experiments]
            _write_rows(output_path, fieldnames, _rows)
            n_failed += 1
            continue

        batch, err = featurize(ligand)
        if err is not None:
            logger.warning(f"  {ligand_name} featurization failed on replay pass: {err}")
            _rows = [_error_row(receptor_id, ligand_name, exp, err) for exp in experiments]
            _write_rows(output_path, fieldnames, _rows)
            n_failed += 1
            continue

        query_n_tokens = entry["n_tokens"]
        trunk_out = {
            "z": entry["z"].to(device=device, dtype=model_dtype).unsqueeze(0),
            "use_kernels": entry["use_kernels"],
        }

        lig_rows = []
        for exp in experiments:
            t0 = time.perf_counter()
            try:
                if exp.operator == "zero":
                    out = affinity_head_forward(
                        model, batch, trunk_out,
                        zero_z_trunk=exp.zero_z_trunk,
                        zero_s_inputs=exp.zero_s_inputs,
                        disable_distogram=(exp.distogram_mask_mode == "zero_all"),
                        zero_atom_encoder=exp.zero_atom_encoder,
                        zero_msa_profile=exp.zero_msa_profile,
                        zero_res_type=exp.zero_res_type,
                        pose_noise_sigma=exp.pose_noise_sigma,
                        pose_noise_seed=exp.pose_noise_seed,
                        pose_noise_target=exp.pose_noise_target,
                    )
                    donor_ids: Dict[str, str] = {}
                    skip_reason = None
                else:
                    out, donor_ids, skip_reason = _run_resample_or_mean(
                        model, batch, trunk_out, exp,
                        trunk_cache_dir, receptor_id, ligand_name, query_n_tokens,
                        _affinity_input_embed_with_ablation, _build_cross_pair_mask, _get_module,
                    )
                    if out is None:
                        raise RuntimeError(f"resample/mean skipped: {skip_reason}")

                elapsed = (time.perf_counter() - t0) * 1000
                lig_rows.append(_result_row(
                    receptor_id, ligand_name, exp, out, elapsed,
                    query_n_tokens, donor_ids, skip_reason,
                ))
                logger.info(
                    f"  [{exp.name}] pred={out['affinity_pred_value']:.4f} "
                    f"prob={out['affinity_probability_binary']:.4f} ({elapsed:.0f}ms)"
                )
            except Exception as e:
                logger.error(f"  [{exp.name}] FAILED: {e}")
                lig_rows.append(_error_row(receptor_id, ligand_name, exp, str(e)))

        _write_rows(output_path, fieldnames, lig_rows)
        n_done += 1

    shutil.rmtree(msa_cache_dir, ignore_errors=True)
    return n_done, n_failed


def _run_resample_or_mean(
    model, batch, trunk_out, exp: AblationExperiment,
    trunk_cache_dir: Path, receptor_id: str, ligand_name: str, query_n_tokens: int,
    _affinity_input_embed_with_ablation, _build_cross_pair_mask, _get_module,
):
    """Task 1: replay the affinity head with donor-substituted channels.

    Mirrors ``affinity_head_forward`` (src/boltz/affinity_rescoring/
    inference.py) but substitutes cached donor tensors for the channels in
    ``exp.resample_channels`` instead of zeroing them, and never touches
    the zero_*/pose_noise flags (orthogonal to this operator; unset for all
    resample/mean registry entries).
    """
    import numpy as np
    import torch

    z = trunk_out["z"]
    use_kernels = trunk_out["use_kernels"]
    device = z.device
    dtype = z.dtype

    donor_ids: Dict[str, str] = {}
    skip_reasons = []

    substituted_z = None
    substituted_s = None
    substituted_repr_pos = None

    if exp.operator == "resample":
        rng = np.random.default_rng(exp.donor_seed)
        for channel in exp.resample_channels:
            match = trunk_cache.find_donor(trunk_cache_dir, receptor_id, ligand_name, query_n_tokens, rng)
            if match is None:
                skip_reasons.append(f"{channel}: no donor available")
                continue
            donor_entry = trunk_cache.load_cache_entry(trunk_cache_dir, receptor_id, match.donor_ligand_id)
            donor_ids[channel] = match.donor_ligand_id
            substituted, delta = trunk_cache.substitute_channel(channel, query_n_tokens, donor_entry)
            if match.match_kind == "nearest":
                logger.warning(
                    f"  [{exp.name}] {ligand_name}: donor {match.donor_ligand_id} "
                    f"n_tokens={match.donor_n_tokens} != query n_tokens={query_n_tokens} "
                    f"(nearest match, {'padded' if delta > 0 else 'cropped'} by {abs(delta)})"
                )
            if channel == "z_trunk":
                substituted_z = substituted
            elif channel == "s_inputs":
                substituted_s = substituted
            elif channel == "distogram":
                substituted_repr_pos = substituted
    elif exp.operator == "mean":
        for channel in exp.resample_channels:
            donor_lig_ids = trunk_cache.all_other_ligands(trunk_cache_dir, receptor_id, ligand_name)
            if not donor_lig_ids:
                skip_reasons.append(f"{channel}: no donors available for mean")
                continue
            donor_entries = [
                trunk_cache.load_cache_entry(trunk_cache_dir, receptor_id, lid) for lid in donor_lig_ids
            ]
            donor_entries = [e for e in donor_entries if e is not None]
            donor_ids[channel] = "mean(" + ",".join(donor_lig_ids) + ")"
            substituted = trunk_cache.mean_channel(channel, query_n_tokens, donor_entries)
            if channel == "z_trunk":
                substituted_z = substituted
            elif channel == "s_inputs":
                substituted_s = substituted
            elif channel == "distogram":
                substituted_repr_pos = substituted
    else:
        raise ValueError(f"Unknown operator {exp.operator!r}")

    skip_reason = "; ".join(skip_reasons) if skip_reasons else None
    if not donor_ids:
        return None, donor_ids, skip_reason

    cross_pair_mask = _build_cross_pair_mask(batch)
    z_affinity = z * cross_pair_mask[None, :, :, None]
    if substituted_z is not None:
        z_affinity = substituted_z.to(device=device, dtype=dtype).unsqueeze(0)

    coords_affinity = _coords_affinity(batch)
    if substituted_repr_pos is not None:
        token_to_rep_atom = batch["token_to_rep_atom"][0]
        rep_atom_idx = token_to_rep_atom.argmax(dim=-1)
        donor_pos = substituted_repr_pos.to(device=coords_affinity.device, dtype=coords_affinity.dtype)
        n = min(rep_atom_idx.shape[0], donor_pos.shape[0])
        coords_affinity = coords_affinity.clone()
        coords_affinity[0, rep_atom_idx[:n]] = donor_pos[:n]

    s_inputs = _affinity_input_embed_with_ablation(model, batch)
    if substituted_s is not None:
        s_inputs = substituted_s.to(device=s_inputs.device, dtype=s_inputs.dtype).unsqueeze(0)

    module_kwargs = {"disable_distogram": False}
    results: Dict[str, float] = {}
    with torch.autocast("cuda", enabled=False):
        if model.affinity_ensemble:
            affinity_module1 = _get_module(model, "affinity_module1")
            affinity_module2 = _get_module(model, "affinity_module2")
            out1 = affinity_module1(
                s_inputs=s_inputs.detach(), z=z_affinity.detach(), x_pred=coords_affinity,
                feats=batch, multiplicity=1, use_kernels=use_kernels, **module_kwargs,
            )
            out1["affinity_probability_binary"] = torch.nn.functional.sigmoid(out1["affinity_logits_binary"])
            out2 = affinity_module2(
                s_inputs=s_inputs.detach(), z=z_affinity.detach(), x_pred=coords_affinity,
                feats=batch, multiplicity=1, use_kernels=use_kernels, **module_kwargs,
            )
            out2["affinity_probability_binary"] = torch.nn.functional.sigmoid(out2["affinity_logits_binary"])
            avg_pred = (out1["affinity_pred_value"] + out2["affinity_pred_value"]) / 2
            avg_prob = (out1["affinity_probability_binary"] + out2["affinity_probability_binary"]) / 2
            if model.affinity_mw_correction:
                model_coef, mw_coef, bias = 1.03525938, -0.59992683, 2.83288489
                mw = batch["affinity_mw"][0] ** 0.3
                avg_pred = model_coef * avg_pred + mw_coef * mw + bias
            results["affinity_pred_value"] = avg_pred.item()
            results["affinity_probability_binary"] = avg_prob.item()
        else:
            affinity_module = _get_module(model, "affinity_module")
            out = affinity_module(
                s_inputs=s_inputs.detach(), z=z_affinity.detach(), x_pred=coords_affinity,
                feats=batch, multiplicity=1, use_kernels=use_kernels, **module_kwargs,
            )
            results["affinity_pred_value"] = out["affinity_pred_value"].item()
            results["affinity_probability_binary"] = torch.nn.functional.sigmoid(
                out["affinity_logits_binary"]
            ).item()

    return results, donor_ids, skip_reason


# ── CSV row helpers ──────────────────────────────────────────────────

def _row_base(receptor_id, ligand_name, exp: AblationExperiment) -> dict:
    return {
        "receptor_id": receptor_id,
        "ligand_name": ligand_name,
        "experiment": exp.name,
        "operator": exp.operator,
        "resample_channels": ",".join(exp.resample_channels),
        "donor_seed": "" if exp.donor_seed is None else exp.donor_seed,
        "donor_complex_ids": "",
        "query_n_tokens": "",
        "pose_noise_sigma": exp.pose_noise_sigma,
        "pose_noise_target": exp.pose_noise_target,
        "pose_noise_seed": "" if exp.pose_noise_seed is None else exp.pose_noise_seed,
    }


def _error_row(receptor_id, ligand_name, exp, error) -> dict:
    row = _row_base(receptor_id, ligand_name, exp)
    row.update({
        "affinity_pred_value": "",
        "affinity_probability_binary": "",
        "error": error,
        "time_ms": "",
    })
    return row


def _result_row(receptor_id, ligand_name, exp, out, elapsed, query_n_tokens, donor_ids, skip_reason) -> dict:
    row = _row_base(receptor_id, ligand_name, exp)
    donor_str = ";".join(f"{ch}={lid}" for ch, lid in donor_ids.items())
    row.update({
        "donor_complex_ids": donor_str,
        "query_n_tokens": query_n_tokens,
        "affinity_pred_value": out["affinity_pred_value"],
        "affinity_probability_binary": out["affinity_probability_binary"],
        "error": f"partial: {skip_reason}" if skip_reason else "",
        "time_ms": f"{elapsed:.1f}",
    })
    return row


def _write_rows(output_path, fieldnames, rows):
    if output_path and fieldnames and rows:
        with open(output_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.log_level)

    import torch
    from boltz.affinity_rescoring.inference import AffinityModelManager
    from boltz.affinity_rescoring.models import DeviceOption

    logger.info("Loading model...")
    manager = AffinityModelManager(device=DeviceOption(args.device))
    model = manager.load_model(
        checkpoint_path=args.checkpoint if args.checkpoint != "auto" else None,
    )
    cache_dir = manager.cache_dir
    logger.info(f"Model loaded on {manager.device}")

    if args.all_receptors:
        pairs = discover_pairs(args.receptors_dir, args.poses_dir)
    else:
        receptor_id = args.receptor.stem.removesuffix("_receptor")
        pairs = [(receptor_id, args.receptor, args.ligands)]

    experiments = [EXPERIMENTS[name] for name in args.experiments]

    logger.info(f"Running {len(experiments)} experiment(s) on {len(pairs)} receptor(s)")
    for exp in experiments:
        flags = []
        if exp.operator != "zero":
            flags.append(f"operator={exp.operator}")
            flags.append(f"channels={','.join(exp.resample_channels)}")
            if exp.donor_seed is not None:
                flags.append(f"donor_seed={exp.donor_seed}")
        else:
            if exp.distogram_mask_mode != "none":
                flags.append(f"disto={exp.distogram_mask_mode}")
            if exp.zero_z_trunk:
                flags.append("zero_z_trunk")
            if exp.zero_s_inputs:
                flags.append("zero_s_inputs")
            if exp.zero_atom_encoder:
                flags.append("zero_atom_enc")
            if exp.zero_msa_profile:
                flags.append("zero_msa_prof")
            if exp.zero_res_type:
                flags.append("zero_res_type")
        if exp.pose_noise_sigma > 0:
            flags.append(
                f"noise={exp.pose_noise_sigma}Å/{exp.pose_noise_target}"
                + (f",seed={exp.pose_noise_seed}" if exp.pose_noise_seed is not None else "")
            )
        flag_str = ", ".join(flags) if flags else "(no ablation)"
        logger.info(f"  {exp.name:30s} → {flag_str}")

    n_total_done = 0
    n_total_failed = 0
    t_start = time.perf_counter()

    fieldnames = [
        "receptor_id", "ligand_name", "experiment", "operator", "resample_channels",
        "donor_seed", "donor_complex_ids", "query_n_tokens",
        "affinity_pred_value", "affinity_probability_binary",
        "pose_noise_sigma", "pose_noise_target", "pose_noise_seed",
        "error", "time_ms",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    config_hash = hashlib.sha256(
        ",".join(sorted(args.experiments)).encode()
    ).hexdigest()[:16]

    for pair_idx, (receptor_id, receptor_path, ligands_path) in enumerate(pairs, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Receptor {pair_idx}/{len(pairs)}: {receptor_id}")
        logger.info(f"{'=' * 60}")

        n_done, n_failed = run_ablation_for_pair(
            receptor_id=receptor_id,
            receptor_path=receptor_path,
            ligands_path=ligands_path,
            model=model,
            experiments=experiments,
            max_ligands=args.max_ligands,
            recycling_steps=args.recycling_steps,
            use_msa_server=args.use_msa_server,
            cache_dir=cache_dir,
            trunk_cache_dir=args.trunk_cache_dir,
            rebuild_trunk_cache=args.rebuild_trunk_cache,
            output_path=args.output,
            fieldnames=fieldnames,
        )
        n_total_done += n_done
        n_total_failed += n_failed

    elapsed_total = time.perf_counter() - t_start

    n_total = n_total_done + n_total_failed
    logger.info(f"\n{'=' * 60}")
    logger.info("FEATURE ABLATION COMPLETE")
    logger.info(f"  Total runs  : {n_total}")
    logger.info(f"  Succeeded   : {n_total_done}")
    logger.info(f"  Failed      : {n_total_failed}")
    logger.info(f"  Wall time   : {elapsed_total:.1f}s")
    logger.info(f"  Output      : {args.output}")
    logger.info(f"{'=' * 60}")

    meta = trunk_cache.run_metadata({
        "config_hash": config_hash,
        "experiments": args.experiments,
        "recycling_steps": args.recycling_steps,
        "max_ligands": args.max_ligands,
        "trunk_cache_dir": str(args.trunk_cache_dir),
        "checkpoint": args.checkpoint,
        "n_total": n_total,
        "n_succeeded": n_total_done,
        "n_failed": n_total_failed,
        "wall_time_s": elapsed_total,
    })
    sidecar = trunk_cache.write_json_sidecar(args.output, meta)
    logger.info(f"  Sidecar     : {sidecar}")


if __name__ == "__main__":
    main()
