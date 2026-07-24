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
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Ensure the repo's src/ is on sys.path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


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
# sweep.  Use ``--experiments`` to override.
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
    return receptors_dir, poses_dir, output_dir


def parse_args() -> argparse.Namespace:
    receptors_dir, poses_dir, output_dir = default_paths()

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
    output_path: Optional[Path] = None,
    fieldnames: Optional[List[str]] = None,
) -> tuple:
    """Run all ablation experiments for one receptor/ligand-set pair.

    Preprocesses each ligand once, then runs the affinity head repeatedly
    with different ablation flags — avoiding redundant featurization.

    Results are written to ``output_path`` incrementally after each ligand
    so that partial results are preserved if the job is killed.  Returns
    ``(n_done, n_failed)`` counts rather than the full row list.
    """
    import numpy as np
    import torch
    from torch import Tensor

    from boltz.affinity_rescoring.coord_injection import (
        build_chain_id_map,
        inject_pdb_coords_into_structure,
        save_pre_affinity_structure,
    )
    from boltz.affinity_rescoring.inference import (
        affinity_forward,
        create_affinity_yaml,
    )
    from boltz.affinity_rescoring.mol2_parser import MOL2Parser
    from boltz.affinity_rescoring.parsers import (
        get_chain_sequences,
        get_seqres_sequences,
        parse_structure_file,
    )
    from boltz.affinity_rescoring.smiles_inference import (
        infer_ligand_smiles_from_structure,
    )
    from boltz.data import const
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals, load_molecules
    from boltz.data.module.inferencev2 import load_input
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Record, StructureV2
    from boltz.main import process_input

    mol_dir = cache_dir / "mols"
    device = next(model.parameters()).device

    # Load heavy objects once (not per-ligand)
    ccd = load_canonicals(mol_dir)
    tokenizer = Boltz2Tokenizer()
    cropper = AffinityCropper()
    featurizer = Boltz2Featurizer()

    # Parse receptor
    protein_atoms, _metadata = parse_structure_file(receptor_path)
    protein_chain_id = "A"
    merged_refs = get_seqres_sequences(receptor_path)
    sequences = get_chain_sequences(protein_atoms, reference_sequences=merged_refs)

    if protein_chain_id not in sequences:
        protein_chain_id = next(iter(sequences))

    protein_seq = sequences[protein_chain_id]
    logger.info(f"[{receptor_id}] Protein chain {protein_chain_id}: "
                f"{len(protein_seq)} residues")

    # Pre-compute MSA once
    import tempfile
    msa_cache_dir = Path(tempfile.mkdtemp(prefix="boltz_msa_cache_"))
    msa_path = None
    if use_msa_server:
        try:
            from boltz.main import compute_msa
            target_id = f"receptor_{protein_chain_id}"
            data = {target_id: protein_seq}
            compute_msa(
                data=data,
                target_id=target_id,
                msa_dir=msa_cache_dir,
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
        logger.error(f"[{receptor_id}] Skipping receptor: MSA pre-computation failed "
                     "and running per-ligand would be redundant.")
        return []

    # Parse ligands
    mol2_parser = MOL2Parser()
    ligands = mol2_parser.extract_ligands_with_names(ligands_path)
    if max_ligands is not None:
        ligands = ligands[:max_ligands]
    logger.info(f"[{receptor_id}] {len(ligands)} ligands to process")

    n_done = 0
    n_failed = 0

    for lig_idx, ligand in enumerate(ligands):
        ligand_name = ligand.name
        logger.info(f"[{receptor_id}] Ligand {lig_idx + 1}/{len(ligands)}: "
                     f"{ligand_name}")

        # Convert ligand to SMILES
        smiles = None
        try:
            from boltz.affinity_rescoring.smiles_inference import (
                infer_smiles_from_atoms,
            )
            smiles = infer_smiles_from_atoms(ligand.atoms)
        except Exception:
            pass

        if smiles is None:
            logger.warning(f"  Cannot infer SMILES for {ligand_name}, skipping.")
            _failed_rows = [{
                "receptor_id": receptor_id,
                "ligand_name": ligand_name,
                "experiment": exp.name,
                "affinity_pred_value": "",
                "affinity_probability_binary": "",
                "pose_noise_sigma": exp.pose_noise_sigma,
                "pose_noise_target": exp.pose_noise_target,
                "pose_noise_seed": (
                    "" if exp.pose_noise_seed is None
                    else exp.pose_noise_seed
                ),
                "error": "SMILES inference failed",
                "time_ms": "",
            } for exp in experiments]
            if output_path and fieldnames:
                with open(output_path, "a", newline="") as _f:
                    csv.DictWriter(_f, fieldnames=fieldnames).writerows(_failed_rows)
            n_failed += 1
            continue

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

            # Preprocess
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

            # Load record + inject coords
            record_files = list(records_dir.glob("*.json"))
            if not record_files:
                raise RuntimeError("process_input produced no records.")
            record = Record.load(record_files[0])
            processed_struct = StructureV2.load(
                structure_dir / f"{record.id}.npz"
            )

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

            # Featurize
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
            tokenized = cropper.crop(
                tokenized, max_tokens=256, max_atoms=2048
            )

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

            # Move to device
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

            # ── Run each ablation experiment ─────────────────────────
            lig_rows = []
            for exp in experiments:
                t0 = time.perf_counter()
                try:
                    disable_distogram = exp.distogram_mask_mode == "zero_all"
                    out = affinity_forward(
                        model, batch,
                        recycling_steps=recycling_steps,
                        zero_z_trunk=exp.zero_z_trunk,
                        zero_s_inputs=exp.zero_s_inputs,
                        disable_distogram=disable_distogram,
                        zero_atom_encoder=exp.zero_atom_encoder,
                        zero_msa_profile=exp.zero_msa_profile,
                        zero_res_type=exp.zero_res_type,
                        pose_noise_sigma=exp.pose_noise_sigma,
                        pose_noise_seed=exp.pose_noise_seed,
                        pose_noise_target=exp.pose_noise_target,
                    )
                    elapsed = (time.perf_counter() - t0) * 1000
                    lig_rows.append({
                        "receptor_id": receptor_id,
                        "ligand_name": ligand_name,
                        "experiment": exp.name,
                        "affinity_pred_value": out["affinity_pred_value"],
                        "affinity_probability_binary": out[
                            "affinity_probability_binary"
                        ],
                        "pose_noise_sigma": exp.pose_noise_sigma,
                        "pose_noise_target": exp.pose_noise_target,
                        "pose_noise_seed": (
                            "" if exp.pose_noise_seed is None
                            else exp.pose_noise_seed
                        ),
                        "error": "",
                        "time_ms": f"{elapsed:.1f}",
                    })
                    logger.info(
                        f"  [{exp.name}] pred={out['affinity_pred_value']:.4f} "
                        f"prob={out['affinity_probability_binary']:.4f} "
                        f"({elapsed:.0f}ms)"
                    )
                except Exception as e:
                    logger.error(f"  [{exp.name}] FAILED: {e}")
                    lig_rows.append({
                        "receptor_id": receptor_id,
                        "ligand_name": ligand_name,
                        "experiment": exp.name,
                        "affinity_pred_value": "",
                        "affinity_probability_binary": "",
                        "pose_noise_sigma": exp.pose_noise_sigma,
                        "pose_noise_target": exp.pose_noise_target,
                        "pose_noise_seed": (
                            "" if exp.pose_noise_seed is None
                            else exp.pose_noise_seed
                        ),
                        "error": str(e),
                        "time_ms": "",
                    })
            # Write this ligand's rows immediately
            if output_path and fieldnames and lig_rows:
                with open(output_path, "a", newline="") as _f:
                    csv.DictWriter(_f, fieldnames=fieldnames).writerows(lig_rows)
            n_done += 1

        except Exception as e:
            logger.error(f"  Failed to process {ligand_name}: {e}")
            import traceback
            traceback.print_exc()
            _exc_rows = [{
                "receptor_id": receptor_id,
                "ligand_name": ligand_name,
                "experiment": exp.name,
                "affinity_pred_value": "",
                "affinity_probability_binary": "",
                "pose_noise_sigma": exp.pose_noise_sigma,
                "pose_noise_target": exp.pose_noise_target,
                "pose_noise_seed": (
                    "" if exp.pose_noise_seed is None
                    else exp.pose_noise_seed
                ),
                "error": str(e),
                "time_ms": "",
            } for exp in experiments]
            if output_path and fieldnames:
                with open(output_path, "a", newline="") as _f:
                    csv.DictWriter(_f, fieldnames=fieldnames).writerows(_exc_rows)
            n_failed += 1
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    shutil.rmtree(msa_cache_dir, ignore_errors=True)
    return n_done, n_failed


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


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.log_level)

    import torch
    from boltz.affinity_rescoring.inference import AffinityModelManager
    from boltz.affinity_rescoring.models import DeviceOption

    # Load model once
    logger.info("Loading model...")
    manager = AffinityModelManager(device=DeviceOption(args.device))
    model = manager.load_model(
        checkpoint_path=args.checkpoint if args.checkpoint != "auto" else None,
    )
    cache_dir = manager.cache_dir
    logger.info(f"Model loaded on {manager.device}")

    # Discover receptor/ligand pairs
    if args.all_receptors:
        pairs = discover_pairs(args.receptors_dir, args.poses_dir)
    else:
        receptor_id = args.receptor.stem.removesuffix("_receptor")
        pairs = [(receptor_id, args.receptor, args.ligands)]

    # Resolve experiments
    experiments = [EXPERIMENTS[name] for name in args.experiments]

    logger.info(f"Running {len(experiments)} experiment(s) on {len(pairs)} receptor(s)")
    for exp in experiments:
        flags = []
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
                f"noise={exp.pose_noise_sigma}\u00c5/{exp.pose_noise_target}"
                + (f",seed={exp.pose_noise_seed}"
                   if exp.pose_noise_seed is not None else "")
            )
        flag_str = ", ".join(flags) if flags else "(no ablation)"
        logger.info(f"  {exp.name:25s} → {flag_str}")

    # Run
    n_total_done = 0
    n_total_failed = 0
    t_start = time.perf_counter()

    # Write CSV header upfront; rows are appended per-ligand inside run_ablation_for_pair
    fieldnames = [
        "receptor_id", "ligand_name", "experiment",
        "affinity_pred_value", "affinity_probability_binary",
        "pose_noise_sigma", "pose_noise_target", "pose_noise_seed",
        "error", "time_ms",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for pair_idx, (receptor_id, receptor_path, ligands_path) in enumerate(
        pairs, 1
    ):
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
            output_path=args.output,
            fieldnames=fieldnames,
        )
        n_total_done += n_done
        n_total_failed += n_failed

    elapsed_total = time.perf_counter() - t_start

    # Summary
    n_total = n_total_done + n_total_failed
    n_success = n_total_done
    n_failed = n_total_failed

    logger.info(f"\n{'=' * 60}")
    logger.info("FEATURE ABLATION COMPLETE")
    logger.info(f"  Total runs  : {n_total}")
    logger.info(f"  Succeeded   : {n_success}")
    logger.info(f"  Failed      : {n_failed}")
    logger.info(f"  Wall time   : {elapsed_total:.1f}s")
    logger.info(f"  Output      : {args.output}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
