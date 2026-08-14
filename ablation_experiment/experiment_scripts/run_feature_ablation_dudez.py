#!/usr/bin/env python3
"""DUDEZ feature-ablation runner — one receptor at a time.

Differs from ``run_feature_ablation.py`` (which expects a receptor PDB +
multi-pose MOL2) by consuming the *predicted* DUDEZ Boltz outputs:

  /scratch/maom_root/maom/limcaoco/Boltz_outputs/<RECEPTOR>/
      combined_structures/<COMPOUND_ID>_model_0.cif

and the SMILES manifest at

  /home/limcaoco/turbo/limcaoco/boltz_benchmark/input_files/DUDEZ_benchmark/
      <RECEPTOR>_combined_ids.csv          (cols: SMILES, compound_ID, is_binder)

For every CIF found, this script:

1. Splits the predicted complex into chain-A protein atoms and chain-B
   ligand atoms (HETATM ``LIG1`` written by Boltz).
2. Looks up the SMILES from the receptor's *_combined_ids.csv* (no RDKit
   bond-perception — the SMILES is authoritative).
3. Re-maps the ligand atom names to canonical RDKit ranks (re-using
   ``_remap_ligand_atoms`` from the MOL2 runner).
4. Builds a Boltz YAML pinning the MSA to the cached
   ``<RECEPTOR>_mmseqs2.a3m`` so the colab MSA server is never contacted.
5. Featurises once, then runs every requested ablation against the
   affinity head.

Outputs one CSV per receptor with the same columns as the MOL2 runner
plus the receptor-of-origin and a ``compound_class`` column (active /
decoy) read from the SMILES CSV's ``is_binder`` flag.
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

# Ensure repo src/ is on sys.path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Re-use the MOL2 runner's experiment registry and ligand-atom
# canonicalisation helper.
from run_feature_ablation import (
    ALL_EXPERIMENT_NAMES,
    DEFAULT_EXPERIMENT_NAMES,
    EXPERIMENTS,
    AblationExperiment,
    _remap_ligand_atoms,
    setup_logging,
)

# DUDEZ default: channel + sub-component ablations only — pose-noise
# experiments are intentionally excluded here for computational efficiency
# and run separately via the pose-noise pipeline.
DUDEZ_DEFAULT_EXPERIMENT_NAMES: list[str] = [
    n for n in DEFAULT_EXPERIMENT_NAMES
    if not (n.startswith("lig_noise_")
            or n.startswith("rec_noise_")
            or n.startswith("all_noise_"))
]


logger = logging.getLogger(__name__)


DEFAULT_BOLTZ_OUTPUTS = Path("/scratch/maom_root/maom/limcaoco/Boltz_outputs")
DEFAULT_DUDEZ_INPUTS = Path(
    "/home/limcaoco/turbo/limcaoco/boltz_benchmark/input_files/DUDEZ_benchmark"
)
DEFAULT_MSA_DIR = Path(
    "/home/limcaoco/turbo/limcaoco/boltz_benchmark/input_files/msa"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--receptor", required=True,
        help="DUDEZ target ID (e.g. AA2AR). Used to locate "
             "<boltz-outputs>/<RECEPTOR>/combined_structures/ and "
             "<dudez-inputs>/<RECEPTOR>_combined_ids.csv.",
    )
    parser.add_argument(
        "--boltz-outputs-root", type=Path, default=DEFAULT_BOLTZ_OUTPUTS,
        help="Directory containing per-receptor Boltz prediction trees.",
    )
    parser.add_argument(
        "--dudez-inputs-root", type=Path, default=DEFAULT_DUDEZ_INPUTS,
        help="Directory containing <RECEPTOR>_combined_ids.csv files.",
    )
    parser.add_argument(
        "--msa-dir", type=Path, default=DEFAULT_MSA_DIR,
        help="Directory containing <RECEPTOR>_mmseqs2.a3m cached MSAs.",
    )
    parser.add_argument(
        "--structures-dir", type=Path, default=None,
        help="Override the per-receptor combined_structures path "
             "(default: <boltz-outputs-root>/<RECEPTOR>/combined_structures).",
    )
    parser.add_argument(
        "--smiles-csv", type=Path, default=None,
        help="Override the per-receptor SMILES CSV path "
             "(default: <dudez-inputs-root>/<RECEPTOR>_combined_ids.csv).",
    )
    parser.add_argument(
        "--msa-path", type=Path, default=None,
        help="Override the cached MSA path "
             "(default: <msa-dir>/<RECEPTOR>_mmseqs2.a3m).",
    )

    # Experiment selection — defaults to noise-free subset; pass
    # --experiments explicitly to include pose-noise experiments.
    parser.add_argument(
        "--experiments", nargs="+", default=list(DUDEZ_DEFAULT_EXPERIMENT_NAMES),
        choices=ALL_EXPERIMENT_NAMES,
        help="Which ablation experiments to run (default: channel + "
             "sub-component ablations, no pose-noise).",
    )
    parser.add_argument("--max-ligands", type=int, default=None)
    parser.add_argument(
        "--actives-only", action="store_true",
        help="Restrict to compounds with is_binder=True in the SMILES CSV.",
    )
    parser.add_argument(
        "--decoys-only", action="store_true",
        help="Restrict to compounds with is_binder=False in the SMILES CSV.",
    )
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output CSV (default: ablation_experiment/results/dudez_ablation/"
             "<RECEPTOR>_dudez_ablation.csv).",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    if args.actives_only and args.decoys_only:
        parser.error("--actives-only and --decoys-only are mutually exclusive.")
    return args


# ── Per-receptor input resolution ─────────────────────────────────────


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve and validate the four per-receptor input artefacts."""
    receptor_id = args.receptor
    structures_dir = args.structures_dir or (
        args.boltz_outputs_root / receptor_id / "combined_structures"
    )
    smiles_csv = args.smiles_csv or (
        args.dudez_inputs_root / f"{receptor_id}_combined_ids.csv"
    )
    msa_path = args.msa_path or (args.msa_dir / f"{receptor_id}_mmseqs2.a3m")

    output = args.output or (
        _repo_root / "ablation_experiment" / "results" / "dudez_ablation"
        / f"{receptor_id}_dudez_ablation.csv"
    )

    if not structures_dir.exists():
        raise FileNotFoundError(
            f"Structures directory not found: {structures_dir}. "
            "Did the per-receptor tarball get untarred? "
            f"Try: tar -xf {args.boltz_outputs_root}/{receptor_id}/"
            "combined_structures.tar.gz "
            f"-C {args.boltz_outputs_root}/{receptor_id}/"
        )
    if not smiles_csv.exists():
        raise FileNotFoundError(f"SMILES CSV not found: {smiles_csv}")
    if not msa_path.exists():
        raise FileNotFoundError(f"Cached MSA not found: {msa_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    return structures_dir, smiles_csv, msa_path, output


def load_smiles_table(smiles_csv: Path) -> dict[str, dict[str, str]]:
    """Return {compound_ID -> {'smiles': ..., 'is_binder': bool}}.

    The CSV header is ``SMILES,compound_ID,is_binder`` and several files
    have a leading whitespace before the SMILES; we strip whitespace
    everywhere so lookups are robust.
    """
    table: dict[str, dict[str, str]] = {}
    with smiles_csv.open() as fh:
        reader = csv.DictReader(fh)
        # Robust header lookup — column names may have whitespace.
        smiles_key = next(
            (k for k in (reader.fieldnames or []) if k.strip().upper() == "SMILES"),
            None,
        )
        cid_key = next(
            (k for k in (reader.fieldnames or [])
             if k.strip().lower() == "compound_id"),
            None,
        )
        binder_key = next(
            (k for k in (reader.fieldnames or [])
             if k.strip().lower() == "is_binder"),
            None,
        )
        if smiles_key is None or cid_key is None:
            raise ValueError(
                f"Cannot locate SMILES/compound_ID columns in {smiles_csv}: "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            cid = (row.get(cid_key) or "").strip()
            sm = (row.get(smiles_key) or "").strip()
            if not cid or not sm:
                continue
            is_binder = False
            if binder_key:
                v = (row.get(binder_key) or "").strip().lower()
                is_binder = v in ("true", "1", "yes", "y")
            table[cid] = {"smiles": sm, "is_binder": is_binder}
    return table


# ── Lightweight ligand shim for ``_remap_ligand_atoms`` ────────────────


class _LigandFromCif:
    """Adapter so ``_remap_ligand_atoms`` (built for MOL2 ``Ligand``) works."""
    __slots__ = ("name", "atoms")

    def __init__(self, name: str, atoms):
        self.name = name
        self.atoms = atoms


# ── Per-receptor runner ────────────────────────────────────────────────


def run_for_receptor(args: argparse.Namespace) -> int:
    """Run the requested ablation set for every CIF in the receptor."""
    import numpy as np
    import torch
    import yaml
    from torch import Tensor

    from boltz.affinity_rescoring.coord_injection import (
        build_chain_id_map,
        inject_pdb_coords_into_structure,
        save_pre_affinity_structure,
    )
    from boltz.affinity_rescoring.inference import (
        AffinityModelManager,
        affinity_forward,
    )
    from boltz.affinity_rescoring.models import DeviceOption
    from boltz.affinity_rescoring.parsers import (
        get_chain_sequences,
        get_seqres_sequences,
        parse_structure_file,
    )
    from boltz.data import const
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals, load_molecules
    from boltz.data.module.inferencev2 import load_input
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Record, StructureV2
    from boltz.main import process_input

    receptor_id = args.receptor

    # This runner only implements the ``zero`` operator: it drives the
    # affinity head straight off ``exp.zero_*``/``distogram_mask_mode``.
    # A resample/mean experiment leaves every one of those flags unset, so
    # running one here would silently produce an *unablated baseline* number
    # filed under the resample name -- corrupting any downstream Shapley/NAE
    # that consumed it. Those operators need the donor trunk cache and the
    # two-pass flow in run_feature_ablation.py. Checked before the model
    # load so this fails in seconds, not after a checkpoint load.
    non_zero_ops = [n for n in args.experiments if EXPERIMENTS[n].operator != "zero"]
    if non_zero_ops:
        logger.error(
            f"[{receptor_id}] This DUDEZ runner supports only the 'zero' ablation "
            f"operator, but {len(non_zero_ops)} requested experiment(s) use "
            f"resample/mean: {non_zero_ops[:5]}"
            f"{' ...' if len(non_zero_ops) > 5 else ''}. "
            f"Run those through run_feature_ablation.py, which owns the donor "
            f"trunk cache the resample/mean operators need."
        )
        return 2

    structures_dir, smiles_csv, msa_path, output_path = resolve_inputs(args)
    logger.info(f"[{receptor_id}] structures: {structures_dir}")
    logger.info(f"[{receptor_id}] smiles    : {smiles_csv}")
    logger.info(f"[{receptor_id}] MSA       : {msa_path}")
    logger.info(f"[{receptor_id}] output    : {output_path}")

    smiles_table = load_smiles_table(smiles_csv)
    logger.info(f"[{receptor_id}] SMILES table rows: {len(smiles_table)}")

    # ── Load model
    logger.info("Loading model ...")
    manager = AffinityModelManager(device=DeviceOption(args.device))
    model = manager.load_model(
        checkpoint_path=args.checkpoint if args.checkpoint != "auto" else None,
    )
    cache_dir = manager.cache_dir
    device = next(model.parameters()).device
    mol_dir = cache_dir / "mols"
    ccd = load_canonicals(mol_dir)
    tokenizer = Boltz2Tokenizer()
    cropper = AffinityCropper()
    featurizer = Boltz2Featurizer()

    experiments: list[AblationExperiment] = [EXPERIMENTS[n] for n in args.experiments]
    logger.info(f"Running {len(experiments)} experiments per ligand.")

    # Discover CIFs and apply class filter
    cif_files = sorted(structures_dir.glob("*.cif"))
    if not cif_files:
        logger.error(f"[{receptor_id}] No CIFs found under {structures_dir}.")
        return 2

    def _compound_id(cif_path: Path) -> str:
        # Strip Boltz's "_model_0" suffix written by the predictor
        stem = cif_path.stem
        return stem[: -len("_model_0")] if stem.endswith("_model_0") else stem

    candidates = []
    for p in cif_files:
        cid = _compound_id(p)
        if cid not in smiles_table:
            continue
        if args.actives_only and not smiles_table[cid]["is_binder"]:
            continue
        if args.decoys_only and smiles_table[cid]["is_binder"]:
            continue
        candidates.append((cid, p))
    if args.max_ligands is not None:
        candidates = candidates[: args.max_ligands]
    logger.info(
        f"[{receptor_id}] {len(candidates)} ligands to process "
        f"(actives_only={args.actives_only}, decoys_only={args.decoys_only}, "
        f"max_ligands={args.max_ligands})"
    )

    # ── CSV writer (incremental)
    fieldnames = [
        "receptor_id", "compound_id", "is_binder", "experiment",
        "affinity_pred_value", "affinity_probability_binary",
        "pose_noise_sigma", "pose_noise_target", "pose_noise_seed",
        "error", "time_ms",
    ]
    with output_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    work_root = Path(tempfile.mkdtemp(prefix=f"dudez_{receptor_id}_"))
    n_done = n_failed = 0
    t_start = time.perf_counter()
    try:
        for lig_idx, (compound_id, cif_path) in enumerate(candidates, 1):
            class_label = "active" if smiles_table[compound_id]["is_binder"] else "decoy"
            logger.info(
                f"[{receptor_id}] {lig_idx}/{len(candidates)} {compound_id} "
                f"({class_label})"
            )

            # ── Parse the predicted complex CIF
            try:
                all_atoms, _meta = parse_structure_file(cif_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"  parse failed: {exc}")
                _write_failed_rows(
                    output_path, fieldnames, receptor_id, compound_id,
                    smiles_table[compound_id]["is_binder"], experiments,
                    f"parse_structure_file: {exc}",
                )
                n_failed += 1
                continue

            protein_chain_id = "A"
            ligand_chain_id_in_cif = "B"
            protein_atoms = [a for a in all_atoms if a.chain_id == protein_chain_id]
            cif_ligand_atoms = [
                a for a in all_atoms
                if a.chain_id == ligand_chain_id_in_cif and a.is_hetatm
            ]
            if not protein_atoms or not cif_ligand_atoms:
                logger.warning(
                    f"  skipping {compound_id}: chains A={len(protein_atoms)} "
                    f"B(HETATM)={len(cif_ligand_atoms)}"
                )
                _write_failed_rows(
                    output_path, fieldnames, receptor_id, compound_id,
                    smiles_table[compound_id]["is_binder"], experiments,
                    "missing_protein_or_ligand_chain",
                )
                n_failed += 1
                continue

            merged_refs = get_seqres_sequences(cif_path)
            sequences = get_chain_sequences(
                protein_atoms, reference_sequences=merged_refs,
            )
            if protein_chain_id not in sequences:
                protein_chain_id = next(iter(sequences))
            protein_seq = sequences[protein_chain_id]

            smiles = smiles_table[compound_id]["smiles"]

            # Re-map ligand atom names to canonical RDKit ranks.
            ligand_chain_id = "L"
            shim = _LigandFromCif(compound_id, cif_ligand_atoms)
            remapped = _remap_ligand_atoms(shim, smiles, ligand_chain_id)
            if remapped is None:
                from boltz.affinity_rescoring.models import AtomInfo as _AI
                remapped = [
                    _AI(
                        index=a.index, name=a.name, element=a.element,
                        x=a.x, y=a.y, z=a.z, chain_id=ligand_chain_id,
                        residue_name=a.residue_name,
                        residue_number=a.residue_number,
                        occupancy=a.occupancy, b_factor=a.b_factor,
                        is_hetatm=True,
                    )
                    for a in cif_ligand_atoms
                ]
            combined_atoms = list(protein_atoms) + remapped

            work_dir = Path(tempfile.mkdtemp(prefix="lig_", dir=work_root))
            try:
                # ── Build YAML pinning the cached MSA path
                yaml_data = {
                    "version": 1,
                    "sequences": [
                        {
                            "protein": {
                                "id": protein_chain_id,
                                "sequence": protein_seq,
                                "msa": str(msa_path),
                            }
                        },
                        {"ligand": {"id": ligand_chain_id, "smiles": smiles}},
                    ],
                    "properties": [
                        {"affinity": {"binder": ligand_chain_id}},
                    ],
                }
                yaml_path = work_dir / "input.yaml"
                yaml_path.write_text(yaml.dump(yaml_data, default_flow_style=False))

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
                    path=yaml_path, ccd=ccd, msa_dir=msa_dir, mol_dir=mol_dir,
                    boltz2=True, use_msa_server=False,
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
                processed_struct = StructureV2.load(
                    structure_dir / f"{record.id}.npz"
                )

                chain_id_map_yaml = {
                    protein_chain_id: protein_chain_id,
                    ligand_chain_id: ligand_chain_id,
                }
                yaml_chain_ids = list(chain_id_map_yaml.values())
                asym_map = build_chain_id_map(processed_struct, yaml_chain_ids)
                full_map = {
                    pdb_cid: asym_map[yaml_cid]
                    for pdb_cid, yaml_cid in chain_id_map_yaml.items()
                    if yaml_cid in asym_map
                }
                injected, _ = inject_pdb_coords_into_structure(
                    processed_struct, combined_atoms, full_map,
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
                tokenized = cropper.crop(
                    tokenized, max_tokens=256, max_atoms=2048,
                )

                molecules = dict(ccd)
                if input_data.extra_mols:
                    molecules.update(input_data.extra_mols)
                needed = (
                    set(tokenized.tokens["res_name"].tolist()) - set(molecules.keys())
                )
                molecules.update(load_molecules(mol_dir, needed))

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

                # ── Run every requested ablation against the affinity head
                rows = []
                for exp in experiments:
                    t0 = time.perf_counter()
                    try:
                        disable_distogram = exp.distogram_mask_mode == "zero_all"
                        out = affinity_forward(
                            model, batch,
                            recycling_steps=args.recycling_steps,
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
                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        rows.append({
                            "receptor_id": receptor_id,
                            "compound_id": compound_id,
                            "is_binder": int(smiles_table[compound_id]["is_binder"]),
                            "experiment": exp.name,
                            "affinity_pred_value": out["affinity_pred_value"],
                            "affinity_probability_binary": out["affinity_probability_binary"],
                            "pose_noise_sigma": exp.pose_noise_sigma,
                            "pose_noise_target": exp.pose_noise_target,
                            "pose_noise_seed": (
                                "" if exp.pose_noise_seed is None
                                else exp.pose_noise_seed
                            ),
                            "error": "",
                            "time_ms": f"{elapsed_ms:.1f}",
                        })
                    except Exception as exc:  # noqa: BLE001
                        logger.error(f"  [{exp.name}] FAILED: {exc}")
                        rows.append({
                            "receptor_id": receptor_id,
                            "compound_id": compound_id,
                            "is_binder": int(smiles_table[compound_id]["is_binder"]),
                            "experiment": exp.name,
                            "affinity_pred_value": "",
                            "affinity_probability_binary": "",
                            "pose_noise_sigma": exp.pose_noise_sigma,
                            "pose_noise_target": exp.pose_noise_target,
                            "pose_noise_seed": (
                                "" if exp.pose_noise_seed is None
                                else exp.pose_noise_seed
                            ),
                            "error": str(exc),
                            "time_ms": "",
                        })

                with output_path.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
                n_done += 1

            except Exception as exc:  # noqa: BLE001
                logger.exception(f"  failed to process {compound_id}: {exc}")
                _write_failed_rows(
                    output_path, fieldnames, receptor_id, compound_id,
                    smiles_table[compound_id]["is_binder"], experiments, str(exc),
                )
                n_failed += 1
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"[{receptor_id}] DONE — {n_done} ok, {n_failed} failed, "
        f"{elapsed:.1f}s, output: {output_path}"
    )
    return 0


def _write_failed_rows(output_path, fieldnames, receptor_id, compound_id,
                       is_binder, experiments, message: str) -> None:
    rows = [
        {
            "receptor_id": receptor_id,
            "compound_id": compound_id,
            "is_binder": int(bool(is_binder)),
            "experiment": exp.name,
            "affinity_pred_value": "",
            "affinity_probability_binary": "",
            "pose_noise_sigma": exp.pose_noise_sigma,
            "pose_noise_target": exp.pose_noise_target,
            "pose_noise_seed": (
                "" if exp.pose_noise_seed is None else exp.pose_noise_seed
            ),
            "error": message,
            "time_ms": "",
        }
        for exp in experiments
    ]
    with output_path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    return run_for_receptor(args)


if __name__ == "__main__":
    sys.exit(main())
