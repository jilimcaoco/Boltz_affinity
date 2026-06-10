#!/usr/bin/env python3
"""Distogram ablation study for Boltz affinity rescoring.

Runs affinity inference under different distogram masking modes to measure
how much the model relies on 3D distance information.

This script operates OUTSIDE the regular CLI — it loads the model once,
preprocesses each receptor/ligand pair once, then runs the affinity head
repeatedly with different masking modes, avoiding redundant trunk
computation and I/O.

Masking modes
-------------
- none           : baseline (full distogram)
- zero_all       : zero the entire embedded distogram
- zero_cross     : zero protein↔ligand cross pairs
- zero_ligand    : zero all pairs involving the ligand
- zero_receptor  : zero receptor–receptor pairs
- distance_cutoff: zero pairs beyond a distance threshold (default 8 Å)

Usage
-----
# Single receptor with a few ligands (quick test):
python run_distogram_ablation.py \
    --receptor ../receptors/AA2AR_receptor.pdb \
    --ligands ../DOCK3.8_poses/AA2AR_poses.mol2 \
    --output ../results/ablation/AA2AR_ablation.csv \
    --max-ligands 10

# All receptors:
python run_distogram_ablation.py \
    --all-receptors \
    --max-ligands 20 \
    --output ../results/ablation/all_ablation.csv

# Specific modes only:
python run_distogram_ablation.py \
    --receptor ../receptors/AA2AR_receptor.pdb \
    --ligands ../DOCK3.8_poses/AA2AR_poses.mol2 \
    --modes none zero_all zero_cross \
    --output ../results/ablation/AA2AR_subset.csv

# Custom distance cutoffs:
python run_distogram_ablation.py \
    --receptor ../receptors/AA2AR_receptor.pdb \
    --ligands ../DOCK3.8_poses/AA2AR_poses.mol2 \
    --distance-cutoffs 6 8 10 12 15 \
    --output ../results/ablation/AA2AR_cutoff_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure the repo's src/ is on sys.path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


logger = logging.getLogger(__name__)

# All available masking modes (excluding distance_cutoff which is parameterised)
STANDARD_MODES = ["none", "zero_all", "zero_cross", "zero_ligand", "zero_receptor"]


def default_paths():
    experiment_root = _script_dir.parent
    receptors_dir = experiment_root / "receptors"
    poses_dir = experiment_root / "DOCK3.8_poses"
    output_dir = experiment_root / "results" / "ablation"
    return receptors_dir, poses_dir, output_dir


def parse_args() -> argparse.Namespace:
    receptors_dir, poses_dir, output_dir = default_paths()

    parser = argparse.ArgumentParser(
        description="Distogram ablation study for Boltz affinity rescoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input — single receptor or all
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--receptor", type=Path,
        help="Path to a single receptor PDB file.",
    )
    group.add_argument(
        "--all-receptors", action="store_true",
        help="Run ablation on all receptor/pose pairs found in the experiment dirs.",
    )

    parser.add_argument(
        "--ligands", type=Path, default=None,
        help="Path to MOL2 ligand poses (required with --receptor).",
    )
    parser.add_argument(
        "--receptors-dir", type=Path, default=receptors_dir,
        help="Directory with *_receptor.pdb files (used with --all-receptors).",
    )
    parser.add_argument(
        "--poses-dir", type=Path, default=poses_dir,
        help="Directory with *_poses.mol2 files (used with --all-receptors).",
    )

    # Masking configuration
    parser.add_argument(
        "--modes", nargs="+", default=STANDARD_MODES,
        choices=STANDARD_MODES,
        help="Which masking modes to run.",
    )
    parser.add_argument(
        "--distance-cutoffs", nargs="*", type=float, default=None,
        help="Distance cutoff values in Å. Each becomes a separate ablation run. "
             "Omit to skip cutoff experiments.",
    )

    # Processing
    parser.add_argument(
        "--max-ligands", type=int, default=None,
        help="Limit number of ligands per receptor (for quick testing).",
    )
    parser.add_argument(
        "--checkpoint", default="auto",
        help="Model checkpoint path or 'auto'.",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu", "mps"],
    )
    parser.add_argument(
        "--recycling-steps", type=int, default=5,
    )
    parser.add_argument(
        "--use-msa-server", action="store_true", default=False,
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=Path, default=output_dir / "ablation_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

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


def discover_pairs(
    receptors_dir: Path, poses_dir: Path,
) -> List[tuple]:
    """Discover receptor/pose pairs from the experiment directories."""
    pairs = []
    for receptor_path in sorted(receptors_dir.glob("*_receptor.pdb")):
        receptor_id = receptor_path.stem.removesuffix("_receptor")
        poses_path = poses_dir / f"{receptor_id}_poses.mol2"
        if poses_path.exists():
            pairs.append((receptor_id, receptor_path, poses_path))
        else:
            logger.warning(f"No poses found for {receptor_id}, skipping.")
    return pairs


def run_ablation_for_pair(
    receptor_id: str,
    receptor_path: Path,
    ligands_path: Path,
    model,
    modes: List[str],
    distance_cutoffs: Optional[List[float]],
    max_ligands: Optional[int],
    recycling_steps: int,
    use_msa_server: bool,
    cache_dir: Path,
) -> List[Dict]:
    """Run all ablation modes for a single receptor/ligand-set pair.

    The strategy: parse the receptor and ligands once, build the YAML and
    featurise once per ligand, then run the affinity head multiple times
    with different masking modes.  This avoids re-running the expensive
    preprocessing for each mode.
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

    # Parse receptor
    protein_atoms, _metadata = parse_structure_file(receptor_path)
    protein_chain_id = "A"
    merged_refs = get_seqres_sequences(receptor_path)
    sequences = get_chain_sequences(protein_atoms, reference_sequences=merged_refs)

    if protein_chain_id not in sequences:
        # Use the first chain
        protein_chain_id = next(iter(sequences))

    protein_seq = sequences[protein_chain_id]
    logger.info(f"[{receptor_id}] Protein chain {protein_chain_id}: {len(protein_seq)} residues")

    # Pre-compute MSA once for this receptor (reused across all ligands)
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
            # Find the generated MSA file
            for ext in ("*.a3m", "*.csv"):
                found = list(msa_cache_dir.glob(ext))
                if found:
                    msa_path = str(found[0].resolve())
                    break
            if msa_path:
                logger.info(f"[{receptor_id}] MSA pre-computed: {msa_path}")
            else:
                logger.warning(f"[{receptor_id}] MSA server returned no file")
        except Exception as e:
            logger.warning(f"[{receptor_id}] MSA generation failed: {e}")

    # Parse ligands
    mol2_parser = MOL2Parser()
    ligands = mol2_parser.extract_ligands_with_names(ligands_path)
    if max_ligands is not None:
        ligands = ligands[:max_ligands]
    logger.info(f"[{receptor_id}] {len(ligands)} ligands to process")

    results = []

    for lig_idx, ligand in enumerate(ligands):
        ligand_name = ligand.name
        logger.info(f"[{receptor_id}] Ligand {lig_idx + 1}/{len(ligands)}: {ligand_name}")

        # Convert ligand to SMILES
        smiles = None
        try:
            from boltz.affinity_rescoring.smiles_inference import infer_smiles_from_atoms
            smiles = infer_smiles_from_atoms(ligand.atoms)
        except Exception:
            pass

        if smiles is None:
            logger.warning(f"  Cannot infer SMILES for {ligand_name}, skipping.")
            for mode in modes:
                results.append({
                    "receptor_id": receptor_id,
                    "ligand_name": ligand_name,
                    "mask_mode": mode,
                    "distance_cutoff": "",
                    "affinity_pred_value": "",
                    "affinity_probability_binary": "",
                    "error": "SMILES inference failed",
                })
            continue

        ligand_chain_id = "L"

        # Remap ligand atom names to canonical Boltz naming
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

        # Build YAML
        import tempfile, shutil, yaml

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
                      processed_templates_dir, processed_mols_dir, predictions_dir]:
                d.mkdir(parents=True, exist_ok=True)

            ccd = load_canonicals(mol_dir)

            process_input(
                path=yaml_path,
                ccd=ccd,
                msa_dir=msa_dir,
                mol_dir=mol_dir,
                boltz2=True,
                use_msa_server=use_msa_server,
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
            processed_struct = StructureV2.load(structure_dir / f"{record.id}.npz")

            chain_id_map_yaml = {protein_chain_id: protein_chain_id, ligand_chain_id: ligand_chain_id}
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
            tokenizer = Boltz2Tokenizer()
            cropper = AffinityCropper()
            featurizer = Boltz2Featurizer()
            canonicals = load_canonicals(mol_dir)

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
            molecules.update(canonicals)
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

            # ── Run each masking mode ────────────────────────────────
            for mode in modes:
                t0 = time.perf_counter()
                try:
                    out = affinity_forward(
                        model, batch,
                        recycling_steps=recycling_steps,
                        distogram_mask_mode=mode,
                    )
                    elapsed = (time.perf_counter() - t0) * 1000
                    results.append({
                        "receptor_id": receptor_id,
                        "ligand_name": ligand_name,
                        "mask_mode": mode,
                        "distance_cutoff": "",
                        "affinity_pred_value": out["affinity_pred_value"],
                        "affinity_probability_binary": out["affinity_probability_binary"],
                        "error": "",
                        "time_ms": f"{elapsed:.1f}",
                    })
                    logger.info(
                        f"  [{mode}] pred={out['affinity_pred_value']:.4f} "
                        f"prob={out['affinity_probability_binary']:.4f} ({elapsed:.0f}ms)"
                    )
                except Exception as e:
                    results.append({
                        "receptor_id": receptor_id,
                        "ligand_name": ligand_name,
                        "mask_mode": mode,
                        "distance_cutoff": "",
                        "affinity_pred_value": "",
                        "affinity_probability_binary": "",
                        "error": str(e),
                        "time_ms": "",
                    })

            # ── Distance cutoff sweep ────────────────────────────────
            if distance_cutoffs:
                for cutoff in distance_cutoffs:
                    t0 = time.perf_counter()
                    try:
                        out = affinity_forward(
                            model, batch,
                            recycling_steps=recycling_steps,
                            distogram_mask_mode="distance_cutoff",
                            distance_cutoff=cutoff,
                        )
                        elapsed = (time.perf_counter() - t0) * 1000
                        results.append({
                            "receptor_id": receptor_id,
                            "ligand_name": ligand_name,
                            "mask_mode": "distance_cutoff",
                            "distance_cutoff": cutoff,
                            "affinity_pred_value": out["affinity_pred_value"],
                            "affinity_probability_binary": out["affinity_probability_binary"],
                            "error": "",
                            "time_ms": f"{elapsed:.1f}",
                        })
                        logger.info(
                            f"  [cutoff={cutoff}Å] pred={out['affinity_pred_value']:.4f} "
                            f"prob={out['affinity_probability_binary']:.4f} ({elapsed:.0f}ms)"
                        )
                    except Exception as e:
                        results.append({
                            "receptor_id": receptor_id,
                            "ligand_name": ligand_name,
                            "mask_mode": "distance_cutoff",
                            "distance_cutoff": cutoff,
                            "affinity_pred_value": "",
                            "affinity_probability_binary": "",
                            "error": str(e),
                            "time_ms": "",
                        })

        except Exception as e:
            logger.error(f"  Failed to process {ligand_name}: {e}")
            import traceback
            traceback.print_exc()
            for mode in modes:
                results.append({
                    "receptor_id": receptor_id,
                    "ligand_name": ligand_name,
                    "mask_mode": mode,
                    "distance_cutoff": "",
                    "affinity_pred_value": "",
                    "affinity_probability_binary": "",
                    "error": str(e),
                    "time_ms": "",
                })
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # Clean up MSA cache
    import shutil as _shutil
    _shutil.rmtree(msa_cache_dir, ignore_errors=True)

    return results


def _remap_ligand_atoms(ligand, smiles, ligand_chain_id):
    """Remap mol2 ligand atoms to canonical Boltz naming via RDKit substructure match."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol_smiles = Chem.MolFromSmiles(smiles)
        if mol_smiles is None:
            return None
        mol_smiles = Chem.AddHs(mol_smiles)
        AllChem.EmbedMolecule(mol_smiles, randomSeed=42)
        ranks = list(AllChem.CanonicalRankAtoms(mol_smiles))

        # Build canonical atom names
        canon_names = {}
        for i, atom in enumerate(mol_smiles.GetAtoms()):
            elem = atom.GetSymbol().upper()
            rank = ranks[i]
            canon_names[i] = f"{elem}{rank + 1}"

        # Match mol2 → canonical via 3D substructure
        mol_mol2 = Chem.RWMol()
        conf = Chem.Conformer(len(ligand.atoms))
        atom_map = {}
        for idx, a in enumerate(ligand.atoms):
            rdkit_idx = mol_mol2.AddAtom(Chem.Atom(a.element))
            conf.SetAtomPosition(rdkit_idx, (a.x, a.y, a.z))
            atom_map[rdkit_idx] = idx
        mol_mol2.AddConformer(conf)

        # Try substructure match
        mol_no_h = Chem.RemoveHs(mol_smiles)
        mol2_no_h_indices = [i for i, a in enumerate(ligand.atoms) if a.element != "H"]

        # Simple approach: match by element order (fallback)
        from boltz.affinity_rescoring.models import AtomInfo as _AI

        # Use element + canonical rank matching
        remapped = []
        # Group canonical atoms by element
        elem_canon = {}
        for i, atom in enumerate(mol_smiles.GetAtoms()):
            if atom.GetAtomicNum() == 1:
                continue  # skip H
            elem = atom.GetSymbol().upper()
            if elem not in elem_canon:
                elem_canon[elem] = []
            elem_canon[elem].append((i, canon_names[i]))

        # Group mol2 atoms by element
        elem_mol2 = {}
        for a in ligand.atoms:
            elem = a.element.upper() if a.element else ""
            if elem == "H":
                continue
            if elem not in elem_mol2:
                elem_mol2[elem] = []
            elem_mol2[elem].append(a)

        for elem in elem_mol2:
            if elem not in elem_canon:
                continue
            canon_list = elem_canon[elem]
            mol2_list = elem_mol2[elem]
            # Match by order (both sorted by appearance)
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


def main():
    args = parse_args()
    setup_logging(args.log_level)

    import torch
    from boltz.affinity_rescoring.inference import AffinityModelManager
    from boltz.affinity_rescoring.models import DeviceOption

    # Load model once
    logger.info("Loading model...")
    manager = AffinityModelManager(
        device=DeviceOption(args.device),
    )
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

    logger.info(f"Running ablation on {len(pairs)} receptor(s)")
    logger.info(f"Modes: {args.modes}")
    if args.distance_cutoffs:
        logger.info(f"Distance cutoffs: {args.distance_cutoffs}")

    # Build list of all experiments
    all_modes = list(args.modes)
    n_cutoffs = len(args.distance_cutoffs) if args.distance_cutoffs else 0
    n_experiments = len(all_modes) + n_cutoffs
    logger.info(f"{n_experiments} experiment(s) per ligand")

    # Run
    all_results = []
    t_start = time.perf_counter()

    for pair_idx, (receptor_id, receptor_path, ligands_path) in enumerate(pairs, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Receptor {pair_idx}/{len(pairs)}: {receptor_id}")
        logger.info(f"{'='*60}")

        pair_results = run_ablation_for_pair(
            receptor_id=receptor_id,
            receptor_path=receptor_path,
            ligands_path=ligands_path,
            model=model,
            modes=args.modes,
            distance_cutoffs=args.distance_cutoffs,
            max_ligands=args.max_ligands,
            recycling_steps=args.recycling_steps,
            use_msa_server=args.use_msa_server,
            cache_dir=cache_dir,
        )
        all_results.extend(pair_results)

    elapsed_total = time.perf_counter() - t_start

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "receptor_id", "ligand_name", "mask_mode", "distance_cutoff",
        "affinity_pred_value", "affinity_probability_binary",
        "error", "time_ms",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Summary
    n_total = len(all_results)
    n_success = sum(1 for r in all_results if r.get("error") == "")
    n_failed = n_total - n_success

    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total runs:    {n_total}")
    logger.info(f"Successful:    {n_success}")
    logger.info(f"Failed:        {n_failed}")
    logger.info(f"Wall time:     {elapsed_total:.1f}s")
    logger.info(f"Results saved: {args.output}")

    # Quick per-mode summary
    if n_success > 0:
        from collections import defaultdict
        mode_scores = defaultdict(list)
        for r in all_results:
            if r.get("error") == "" and r["affinity_pred_value"] != "":
                key = r["mask_mode"]
                if r["distance_cutoff"] != "":
                    key = f"{key}_{r['distance_cutoff']}A"
                mode_scores[key].append(float(r["affinity_pred_value"]))

        logger.info(f"\nPer-mode summary (mean ± std of affinity_pred_value):")
        import statistics
        for mode in sorted(mode_scores.keys()):
            vals = mode_scores[mode]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            logger.info(f"  {mode:25s}  mean={mean:+.4f}  std={std:.4f}  n={len(vals)}")


if __name__ == "__main__":
    main()
