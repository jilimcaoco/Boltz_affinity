#!/usr/bin/env python3
"""Per-residue leave-one-out (LOO) saliency for the Boltz affinity head.

For each (receptor, ligand) pair this script:

1. Featurizes the input and runs the trunk **once** to obtain ``z`` and
   ``s_inputs`` (caches the heavy computation per ligand).
2. Runs the affinity head **once** with the un-modified ``z`` to get a
   baseline pIC50.
3. For each receptor token within ``--radius`` Å of any ligand atom,
   zeros that token's row *and* column in ``z`` (via a multiplicative
   mask) and re-runs **only the affinity head** with the perturbed ``z``.
4. Records ``Δ = baseline_pIC50 - loo_pIC50`` per residue.

Outputs a long-form CSV with columns:

    receptor_id, ligand_name, is_decoy, residue_index, residue_name,
    chain_id, distance_to_ligand_A, baseline_pred, loo_pred, delta_pred,
    abs_delta_pred

The residue_index column refers to the *token index in the cropped
representation*; an additional ``asym_id`` column tags the chain. The
residue-name field is filled from the structure's ``res_name`` token
metadata when available.

Usage
-----
# Single receptor, 20 actives + 20 decoys, only residues within 10 Å:
python run_residue_loo.py \
    --receptor ../receptors/AA2AR_receptor.pdb \
    --ligands  ../DOCK3.8_poses/AA2AR_poses.mol2 \
    --max-actives 20 --max-decoys 20 \
    --radius 10 \
    --output ../results/loo/AA2AR_residue_loo.csv

# All receptors, 10 ligands each:
python run_residue_loo.py \
    --all-receptors \
    --max-actives 10 --max-decoys 10 \
    --radius 8 \
    --output ../results/loo/all_residue_loo.csv

Notes
-----
- "Within ``--radius`` Å" uses the minimum distance from the
  representative atom of each receptor token to any representative atom
  of the ligand. This means the per-residue saliency is conditional on
  the docked pose.
- Each ligand pays one trunk-forward cost and N_residues head-forward
  passes. The head is dramatically cheaper than the trunk, so this
  scales well even for ~200 residues × 40 ligands.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Ensure repo src/ on sys.path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


logger = logging.getLogger(__name__)


def default_paths():
    experiment_root = _script_dir.parent
    receptors_dir = experiment_root / "receptors"
    poses_dir = experiment_root / "DOCK3.8_poses"
    output_dir = experiment_root / "results" / "loo"
    return receptors_dir, poses_dir, output_dir


def parse_args():
    receptors_dir, poses_dir, output_dir = default_paths()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--receptor", type=Path,
                       help="Single receptor PDB file.")
    group.add_argument("--all-receptors", action="store_true")

    parser.add_argument("--ligands", type=Path, default=None,
                        help="MOL2 ligand poses (required with --receptor).")
    parser.add_argument("--receptors-dir", type=Path, default=receptors_dir)
    parser.add_argument("--poses-dir", type=Path, default=poses_dir)
    parser.add_argument("--max-actives", type=int, default=20)
    parser.add_argument("--max-decoys", type=int, default=20)
    parser.add_argument("--radius", type=float, default=10.0,
                        help="Only LOO receptor tokens within this many Å of "
                             "any ligand atom (use 0 to consider all tokens).")
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--recycling-steps", type=int, default=5)
    parser.add_argument("--use-msa-server", action="store_true", default=False)
    parser.add_argument("--output", "-o", type=Path,
                        default=output_dir / "residue_loo_results.csv")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.receptor and not args.ligands:
        parser.error("--ligands is required when using --receptor.")
    return args


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def discover_pairs(receptors_dir: Path, poses_dir: Path):
    pairs = []
    for receptor_path in sorted(receptors_dir.glob("*_receptor.pdb")):
        receptor_id = receptor_path.stem.removesuffix("_receptor")
        poses_path = poses_dir / f"{receptor_id}_poses.mol2"
        if poses_path.exists():
            pairs.append((receptor_id, receptor_path, poses_path))
        else:
            logger.warning(f"No poses found for {receptor_id}, skipping.")
    return pairs


def is_decoy_name(name: str) -> bool:
    return str(name).startswith("ZINC")


def select_ligands(ligands, max_actives: int, max_decoys: int):
    """Split ligands into actives / decoys by name and cap each list."""
    actives = [l for l in ligands if not is_decoy_name(l.name)]
    decoys = [l for l in ligands if is_decoy_name(l.name)]
    return actives[:max_actives] + decoys[:max_decoys]


def _build_batch_for_ligand(
    receptor_id, receptor_path, ligand, ccd, mol_dir, tokenizer, cropper,
    featurizer, msa_path, work_root, device,
):
    """Construct the model-ready batch for one ligand pose. Returns
    ``(batch, token_meta)`` where ``token_meta`` is a list of per-token
    dicts (residue index/name/chain) for the LOO labelling. Returns
    ``(None, None)`` on failure."""
    import numpy as np
    import torch
    import yaml
    from torch import Tensor

    from boltz.affinity_rescoring.coord_injection import (
        build_chain_id_map,
        inject_pdb_coords_into_structure,
        save_pre_affinity_structure,
    )
    from boltz.affinity_rescoring.parsers import (
        get_chain_sequences,
        get_seqres_sequences,
        parse_structure_file,
    )
    from boltz.affinity_rescoring.smiles_inference import (
        infer_smiles_from_atoms,
    )
    from boltz.data import const
    from boltz.data.mol import load_molecules
    from boltz.data.module.inferencev2 import load_input
    from boltz.data.types import Record, StructureV2
    from boltz.main import process_input

    # Re-parse receptor for chain info
    protein_atoms, _ = parse_structure_file(receptor_path)
    protein_chain_id = "A"
    merged_refs = get_seqres_sequences(receptor_path)
    sequences = get_chain_sequences(protein_atoms, reference_sequences=merged_refs)
    if protein_chain_id not in sequences:
        protein_chain_id = next(iter(sequences))
    protein_seq = sequences[protein_chain_id]

    # SMILES
    try:
        smiles = infer_smiles_from_atoms(ligand.atoms)
    except Exception:
        smiles = None
    if smiles is None:
        logger.warning(f"  SMILES inference failed for {ligand.name}, skipping.")
        return None, None

    # Remap ligand atom names so Boltz recognises them. Reuses logic from
    # run_feature_ablation._remap_ligand_atoms.
    from run_feature_ablation import _remap_ligand_atoms
    ligand_chain_id = "L"
    ligand_atoms_remapped = _remap_ligand_atoms(ligand, smiles, ligand_chain_id)
    if ligand_atoms_remapped is None:
        from boltz.affinity_rescoring.models import AtomInfo as _AI
        ligand_atoms_remapped = [
            _AI(index=a.index, name=a.name, element=a.element,
                x=a.x, y=a.y, z=a.z, chain_id=ligand_chain_id,
                residue_name=a.residue_name, residue_number=a.residue_number,
                occupancy=a.occupancy, b_factor=a.b_factor, is_hetatm=True)
            for a in ligand.atoms
        ]
    combined_atoms = list(protein_atoms) + ligand_atoms_remapped

    work_dir = Path(tempfile.mkdtemp(prefix="boltz_loo_", dir=work_root))
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
            "properties": [{"affinity": {"binder": ligand_chain_id}}],
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
            path=yaml_path, ccd=ccd,
            msa_dir=msa_dir, mol_dir=mol_dir, boltz2=True,
            use_msa_server=False,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="paired+unpaired",
            msa_server_username=None, msa_server_password=None,
            api_key_header=None, api_key_value=None,
            max_msa_seqs=8192,
            processed_msa_dir=processed_msa_dir,
            processed_constraints_dir=processed_constraints_dir,
            processed_templates_dir=processed_templates_dir,
            processed_mols_dir=processed_mols_dir,
            structure_dir=structure_dir, records_dir=records_dir,
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
        full_map = {pdb_cid: asym_map[yaml_cid]
                    for pdb_cid, yaml_cid in chain_id_map_yaml.items()
                    if yaml_cid in asym_map}
        injected, _ = inject_pdb_coords_into_structure(
            processed_struct, combined_atoms, full_map
        )
        save_pre_affinity_structure(injected, predictions_dir, record.id)

        input_data = load_input(
            record=record, target_dir=predictions_dir,
            msa_dir=processed_msa_dir,
            constraints_dir=processed_constraints_dir,
            template_dir=processed_templates_dir,
            extra_mols_dir=processed_mols_dir, affinity=True,
        )
        tokenized = tokenizer.tokenize(input_data)
        tokenized = cropper.crop(tokenized, max_tokens=256, max_atoms=2048)

        # Capture per-token metadata BEFORE featurization (one row per token).
        tok_table = tokenized.tokens
        token_meta = []
        for i in range(len(tok_table)):
            try:
                token_meta.append({
                    "residue_index": int(tok_table["res_idx"][i]),
                    "residue_name": str(tok_table["res_name"][i]),
                    "asym_id": int(tok_table["asym_id"][i]),
                    "mol_type": int(tok_table["mol_type"][i]),
                })
            except Exception:
                token_meta.append({
                    "residue_index": i,
                    "residue_name": "?",
                    "asym_id": -1,
                    "mol_type": -1,
                })

        molecules = {}
        molecules.update(ccd)
        if input_data.extra_mols:
            molecules.update(input_data.extra_mols)
        mol_names = set(tokenized.tokens["res_name"].tolist())
        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(mol_dir, mol_names))

        random = np.random.default_rng(42)
        features = featurizer.process(
            tokenized, molecules=molecules, random=random,
            training=False, max_atoms=None, max_tokens=None,
            max_seqs=const.max_msa_seqs, pad_to_max_seqs=False,
            single_sequence_prop=0.0, compute_frames=True,
            inference_pocket_constraints=None,
            inference_contact_constraints=None,
            compute_constraint_features=True, override_method=None,
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
        return batch, token_meta
    except Exception as e:
        logger.error(f"  Failed to prepare batch for {ligand.name}: {e}")
        shutil.rmtree(work_dir, ignore_errors=True)
        return None, None


def _residue_distances_to_ligand(batch):
    """Return per-token min distance to any ligand atom in Å, and the
    receptor-token boolean mask. Uses representative atoms (token-level
    coordinates) for both receptor and ligand to match the affinity
    head's input geometry."""
    import torch

    token_to_rep_atom = batch["token_to_rep_atom"][0].float()  # (N_tok, N_atom)
    coords = batch["coords"][0].float()                        # (N_atom, 3)
    # If coords carries an extra ensemble dim, take first
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    rep_xyz = token_to_rep_atom @ coords                       # (N_tok, 3)

    pad_mask = batch["token_pad_mask"][0].bool()
    rec_mask = (batch["mol_type"][0] == 0) & pad_mask
    lig_mask = batch["affinity_token_mask"][0].bool() & pad_mask

    if lig_mask.sum() == 0:
        n = pad_mask.shape[0]
        return torch.full((n,), float("nan"), device=coords.device), rec_mask

    rec_xyz = rep_xyz
    lig_xyz = rep_xyz[lig_mask]                                # (N_lig, 3)
    dists = torch.cdist(rec_xyz.unsqueeze(0), lig_xyz.unsqueeze(0))[0]  # (N_tok, N_lig)
    min_dist = dists.min(dim=1).values                         # (N_tok,)
    return min_dist, rec_mask


def run_loo_for_pair(
    receptor_id, receptor_path, ligands_path, model, args, cache_dir,
):
    import numpy as np
    import torch

    from boltz.affinity_rescoring.inference import (
        affinity_head_forward, affinity_trunk_forward,
    )
    from boltz.affinity_rescoring.mol2_parser import MOL2Parser
    from boltz.affinity_rescoring.parsers import (
        get_chain_sequences, get_seqres_sequences, parse_structure_file,
    )
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer

    device = next(model.parameters()).device
    mol_dir = cache_dir / "mols"

    ccd = load_canonicals(mol_dir)
    tokenizer = Boltz2Tokenizer()
    cropper = AffinityCropper()
    featurizer = Boltz2Featurizer()

    # Pre-compute MSA once
    msa_cache_dir = Path(tempfile.mkdtemp(prefix="boltz_loo_msa_"))
    msa_path = None
    protein_atoms, _ = parse_structure_file(receptor_path)
    protein_chain_id = "A"
    merged_refs = get_seqres_sequences(receptor_path)
    sequences = get_chain_sequences(protein_atoms, reference_sequences=merged_refs)
    if protein_chain_id not in sequences:
        protein_chain_id = next(iter(sequences))
    protein_seq = sequences[protein_chain_id]

    if args.use_msa_server:
        try:
            from boltz.main import compute_msa
            target_id = f"receptor_{protein_chain_id}"
            compute_msa(
                data={target_id: protein_seq},
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
        except Exception as e:
            logger.warning(f"[{receptor_id}] MSA generation failed: {e}")
        if msa_path is None:
            logger.error(f"[{receptor_id}] Skipping — MSA pre-computation failed.")
            shutil.rmtree(msa_cache_dir, ignore_errors=True)
            return []

    # Parse ligands and select subset
    ligands = MOL2Parser().extract_ligands_with_names(ligands_path)
    ligands = select_ligands(ligands, args.max_actives, args.max_decoys)
    logger.info(f"[{receptor_id}] Processing {len(ligands)} ligands "
                f"(max actives={args.max_actives}, max decoys={args.max_decoys})")

    results = []
    work_root = Path(tempfile.mkdtemp(prefix="boltz_loo_work_"))
    try:
        for lig_idx, ligand in enumerate(ligands):
            ligand_name = ligand.name
            logger.info(f"[{receptor_id}] Ligand {lig_idx + 1}/{len(ligands)}: "
                        f"{ligand_name}")
            t0 = time.perf_counter()
            batch, token_meta = _build_batch_for_ligand(
                receptor_id, receptor_path, ligand, ccd, mol_dir,
                tokenizer, cropper, featurizer, msa_path,
                work_root=work_root, device=device,
            )
            if batch is None:
                continue

            # Trunk once
            trunk_out = affinity_trunk_forward(
                model, batch, recycling_steps=args.recycling_steps,
            )

            # Baseline pIC50
            base_out = affinity_head_forward(model, batch, trunk_out)
            baseline_pred = float(base_out["affinity_pred_value"])
            baseline_prob = float(base_out["affinity_probability_binary"])

            # Which receptor tokens are within --radius Å of the ligand?
            min_dist, rec_mask = _residue_distances_to_ligand(batch)
            min_dist_cpu = min_dist.detach().cpu().numpy()
            rec_idx_all = torch.nonzero(rec_mask, as_tuple=False).flatten().tolist()
            if args.radius > 0:
                rec_idx = [i for i in rec_idx_all
                           if np.isfinite(min_dist_cpu[i])
                           and min_dist_cpu[i] <= args.radius]
            else:
                rec_idx = rec_idx_all
            logger.info(f"  baseline pred={baseline_pred:.3f} "
                        f"prob={baseline_prob:.3f}; LOO over "
                        f"{len(rec_idx)} residues (radius={args.radius} Å)")

            n_tok = batch["token_pad_mask"].shape[-1]
            ones_mask = torch.ones(n_tok, device=device)
            for ti in rec_idx:
                loo_mask = ones_mask.clone()
                loo_mask[ti] = 0.0
                out = affinity_head_forward(
                    model, batch, trunk_out, z_token_mask=loo_mask,
                )
                loo_pred = float(out["affinity_pred_value"])
                meta = token_meta[ti] if ti < len(token_meta) else {}
                results.append({
                    "receptor_id": receptor_id,
                    "ligand_name": ligand_name,
                    "is_decoy": int(is_decoy_name(ligand_name)),
                    "token_index": ti,
                    "residue_index": meta.get("residue_index", ti),
                    "residue_name": meta.get("residue_name", ""),
                    "asym_id": meta.get("asym_id", -1),
                    "distance_to_ligand_A": float(min_dist_cpu[ti]),
                    "baseline_pred": baseline_pred,
                    "loo_pred": loo_pred,
                    "delta_pred": baseline_pred - loo_pred,
                    "abs_delta_pred": abs(baseline_pred - loo_pred),
                })

            elapsed = time.perf_counter() - t0
            logger.info(f"  {ligand_name}: {len(rec_idx)} LOO passes in "
                        f"{elapsed:.1f}s")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        shutil.rmtree(msa_cache_dir, ignore_errors=True)

    return results


def main():
    args = parse_args()
    setup_logging(args.log_level)

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "receptor_id", "ligand_name", "is_decoy",
        "token_index", "residue_index", "residue_name", "asym_id",
        "distance_to_ligand_A",
        "baseline_pred", "loo_pred", "delta_pred", "abs_delta_pred",
    ]
    with open(args.output, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    total_rows = 0
    for receptor_id, receptor_path, ligands_path in pairs:
        rows = run_loo_for_pair(
            receptor_id, receptor_path, ligands_path, model, args, cache_dir,
        )
        if rows:
            with open(args.output, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
            total_rows += len(rows)
            logger.info(f"[{receptor_id}] wrote {len(rows)} LOO rows.")

    logger.info(f"DONE: total LOO rows = {total_rows}; output = {args.output}")


if __name__ == "__main__":
    main()
