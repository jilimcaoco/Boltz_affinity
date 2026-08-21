#!/usr/bin/env python3
"""DUDEZ feature-ablation runner — one receptor at a time.

**This is the primary runner for the ablation study.** It consumes
*precomputed* Boltz-2 predicted complexes rather than a receptor PDB plus a
multi-pose MOL2 (which is what ``run_feature_ablation.py`` does), because
the study's premise is that a valid structure is already in hand: the
affinity head is fed that structure directly, sidestepping diffusion
sampling entirely.

Inputs, per receptor
--------------------
  <boltz-outputs>/<RECEPTOR>/combined_structures/<COMPOUND_ID>_model_0.cif
  <dudez-inputs>/<RECEPTOR>_combined_ids.csv   (cols: SMILES, compound_ID, is_binder)
  <msa-dir>/<RECEPTOR>_mmseqs2.a3m

Why this path rather than the MOL2 runner
-----------------------------------------
1. **SMILES are authoritative** — read from the manifest, not inferred by
   RDKit bond perception, which silently drops ligands when it fails.
2. **Labels are authoritative** — ``is_binder`` comes from the manifest
   rather than being guessed from a "ZINC" name prefix.
3. **The MSA is pinned** to a cached ``.a3m``, so the ColabFold server is
   never contacted (works on air-gapped compute nodes).

Flow (mirrors run_feature_ablation.py)
--------------------------------------
Two passes per receptor:

  Pass 1  Featurize every complex and run the trunk **once**, caching the
          result to ``--trunk-cache-dir`` (see ``trunk_cache.py``).
  Pass 2  Replay only the affinity head (~10 ms) for every requested
          experiment against that cache.

This matters: the previous version of this script called
``affinity_forward`` — the *full* trunk plus head — once per experiment, so
a 14-experiment run recomputed the ~10 s trunk 14 times per ligand. The
cache also makes the ``resample``/``mean`` operators possible at all, since
they need another ligand's trunk output as a donor.

Note on comparability: ``--recycling-steps`` defaults to 5, matching
``run_feature_ablation.py``. It previously defaulted to 1 here, which made
results from the two runners silently non-comparable.

Outputs one CSV per receptor, sharing the schema of the MOL2 runner
(``ligand_name`` carries the compound ID) plus an authoritative
``is_binder`` column, so a single analysis chain reads both.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo src/ and this directory are on sys.path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np  # noqa: E402
import trunk_cache  # noqa: E402

# Re-use the MOL2 runner's registry, shared featurization, and head-replay
# machinery. The two runners differ only in how inputs are obtained.
from run_feature_ablation import (  # noqa: E402
    ALL_EXPERIMENT_NAMES,
    DEFAULT_EXPERIMENT_NAMES,
    EXPERIMENTS,
    AblationExperiment,
    _coords_affinity,
    _run_resample_or_mean,
    featurize_complex,
    remap_or_passthrough,
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

# Shared with the MOL2 runner so a single analysis chain reads both.
FIELDNAMES = [
    "receptor_id", "ligand_name", "is_binder", "experiment",
    "operator", "resample_channels", "donor_seed", "donor_complex_ids",
    "query_n_tokens",
    "affinity_pred_value", "affinity_probability_binary",
    "pose_noise_sigma", "pose_noise_target", "pose_noise_seed",
    "error", "time_ms",
]


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
    # --experiments explicitly to include pose-noise or resample/mean.
    parser.add_argument(
        "--experiments", nargs="+", default=list(DUDEZ_DEFAULT_EXPERIMENT_NAMES),
        choices=ALL_EXPERIMENT_NAMES,
        help="Which ablation experiments to run (default: channel + "
             "sub-component ablations, no pose-noise, no resample/mean).",
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
    parser.add_argument(
        "--no-mw-correction", action="store_true", default=False,
        help="Disable Boltz's post-hoc molecular-weight correction "
             "(pred = 1.035*net - 0.600*MW^0.3 + 2.833). STRONGLY RECOMMENDED "
             "for ablation work: the term is added OUTSIDE the network, so it "
             "survives every ablation -- including bias_only, which therefore "
             "ranks purely by molecular weight instead of being a random "
             "baseline. Leaving it on contaminates v(0) and every Shapley / "
             "NAE denominator built on it.",
    )
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument(
        "--recycling-steps", type=int, default=5,
        help="Trunk recycling steps. Matches run_feature_ablation.py so "
             "results from the two runners stay comparable.",
    )
    parser.add_argument(
        "--trunk-cache-dir", type=Path, default=None,
        help="Disk cache for per-(receptor,ligand) trunk output. Default: "
             "ablation_experiment/results/trunk_cache_dudez. Keep this "
             "SEPARATE from the MOL2 runner's cache -- entries are keyed on "
             "(receptor, ligand) with no record of which pose source "
             "produced them, so mixing the two would serve the wrong z.",
    )
    parser.add_argument(
        "--rebuild-trunk-cache", action="store_true", default=False,
        help="Recompute and overwrite cached trunk entries even if present.",
    )
    parser.add_argument(
        "--donor-pool-size", type=int, default=trunk_cache.DEFAULT_DONOR_POOL_SIZE,
        help="How many ligands to cache as the donor pool for resample/mean. "
             "Each entry is ~16 MB (z is N*N*token_z), so caching every ligand "
             "would cost hundreds of GB per receptor. 0 = no limit. Ignored "
             "entirely when no resample/mean experiment is requested, in which "
             "case NOTHING is written to disk.",
    )
    parser.add_argument(
        "--donor-pool-seed", type=int, default=0,
        help="Seed for which ligands land in the donor pool.",
    )
    parser.add_argument(
        "--keep-trunk-cache", action="store_true", default=False,
        help="Keep the donor-pool cache after the run. By default it is "
             "deleted once the replay pass finishes, since it is a scratch "
             "artefact reproducible from the inputs and costs ~16 MB/ligand.",
    )
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


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve and validate the per-receptor input artefacts."""
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
    trunk_cache_dir = args.trunk_cache_dir or (
        _repo_root / "ablation_experiment" / "results" / "trunk_cache_dudez"
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
    return structures_dir, smiles_csv, msa_path, output, trunk_cache_dir


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
        if binder_key is None:
            raise ValueError(
                f"No is_binder column in {smiles_csv} (got {reader.fieldnames}). "
                f"This runner uses the manifest label as ground truth rather "
                f"than guessing actives/decoys from a name prefix, so the "
                f"column is required."
            )
        for row in reader:
            cid = (row.get(cid_key) or "").strip()
            sm = (row.get(smiles_key) or "").strip()
            if not cid or not sm:
                continue
            v = (row.get(binder_key) or "").strip().lower()
            table[cid] = {"smiles": sm, "is_binder": v in ("true", "1", "yes", "y")}
    return table


# ── Lightweight ligand shim for the shared atom-remapper ───────────────


class _LigandFromCif:
    """Adapter so ``remap_or_passthrough`` (built for MOL2 ``Ligand``) works."""
    __slots__ = ("name", "atoms")

    def __init__(self, name: str, atoms):
        self.name = name
        self.atoms = atoms


# ── CSV rows ───────────────────────────────────────────────────────────


def _row_base(receptor_id, compound_id, is_binder, exp: AblationExperiment) -> dict:
    return {
        "receptor_id": receptor_id,
        "ligand_name": compound_id,
        "is_binder": int(bool(is_binder)),
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


def _error_row(receptor_id, compound_id, is_binder, exp, message) -> dict:
    row = _row_base(receptor_id, compound_id, is_binder, exp)
    row.update({
        "affinity_pred_value": "", "affinity_probability_binary": "",
        "error": message, "time_ms": "",
    })
    return row


def _result_row(receptor_id, compound_id, is_binder, exp, out, elapsed_ms,
                query_n_tokens, donor_ids, skip_reason) -> dict:
    row = _row_base(receptor_id, compound_id, is_binder, exp)
    row.update({
        "donor_complex_ids": ";".join(f"{ch}={lid}" for ch, lid in donor_ids.items()),
        "query_n_tokens": query_n_tokens,
        "affinity_pred_value": out["affinity_pred_value"],
        "affinity_probability_binary": out["affinity_probability_binary"],
        "error": f"partial: {skip_reason}" if skip_reason else "",
        "time_ms": f"{elapsed_ms:.1f}",
    })
    return row


def _write_rows(output_path: Path, rows) -> None:
    if rows:
        with output_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerows(rows)


# ── Complex parsing ────────────────────────────────────────────────────


def _compound_id_from_cif(cif_path: Path) -> str:
    """Strip Boltz's ``_model_0`` suffix written by the predictor."""
    stem = cif_path.stem
    return stem[: -len("_model_0")] if stem.endswith("_model_0") else stem


def _parse_complex(cif_path: Path, smiles: str):
    """Split a predicted complex into protein atoms + canonicalised ligand
    atoms, and recover the protein sequence. Returns
    ``(protein_atoms, ligand_atoms, protein_chain_id, protein_seq, error)``.
    """
    from boltz.affinity_rescoring.parsers import (
        get_chain_sequences,
        get_seqres_sequences,
        parse_structure_file,
    )

    try:
        all_atoms, _meta = parse_structure_file(cif_path)
    except Exception as exc:  # noqa: BLE001
        return None, None, None, None, f"parse_structure_file: {exc}"

    protein_chain_id = "A"
    ligand_chain_id_in_cif = "B"
    protein_atoms = [a for a in all_atoms if a.chain_id == protein_chain_id]
    cif_ligand_atoms = [
        a for a in all_atoms
        if a.chain_id == ligand_chain_id_in_cif and a.is_hetatm
    ]
    if not protein_atoms or not cif_ligand_atoms:
        return None, None, None, None, (
            f"missing_protein_or_ligand_chain "
            f"(A={len(protein_atoms)} B_HETATM={len(cif_ligand_atoms)})"
        )

    merged_refs = get_seqres_sequences(cif_path)
    sequences = get_chain_sequences(protein_atoms, reference_sequences=merged_refs)
    if protein_chain_id not in sequences:
        protein_chain_id = next(iter(sequences))
    protein_seq = sequences[protein_chain_id]

    shim = _LigandFromCif(_compound_id_from_cif(cif_path), cif_ligand_atoms)
    ligand_atoms = remap_or_passthrough(shim, smiles, "L")

    return protein_atoms, ligand_atoms, protein_chain_id, protein_seq, None


# ── Per-receptor runner ────────────────────────────────────────────────


def run_for_receptor(args: argparse.Namespace) -> int:
    """Run the requested ablation set for every predicted complex."""
    from boltz.affinity_rescoring.inference import (
        _affinity_input_embed_with_ablation,
        _build_cross_pair_mask,
        _get_module,
        affinity_head_forward,
        affinity_trunk_forward,
    )
    from boltz.affinity_rescoring.inference import AffinityModelManager
    from boltz.affinity_rescoring.models import DeviceOption
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer

    receptor_id = args.receptor
    structures_dir, smiles_csv, msa_path, output_path, trunk_cache_dir = resolve_inputs(args)
    logger.info(f"[{receptor_id}] structures : {structures_dir}")
    logger.info(f"[{receptor_id}] smiles     : {smiles_csv}")
    logger.info(f"[{receptor_id}] MSA        : {msa_path}")
    logger.info(f"[{receptor_id}] trunk cache: {trunk_cache_dir}")
    logger.info(f"[{receptor_id}] output     : {output_path}")

    smiles_table = load_smiles_table(smiles_csv)
    logger.info(f"[{receptor_id}] SMILES table rows: {len(smiles_table)}")

    # ── Load model
    logger.info("Loading model ...")
    manager = AffinityModelManager(device=DeviceOption(args.device))
    model = manager.load_model(
        checkpoint_path=args.checkpoint if args.checkpoint != "auto" else None,
        affinity_mw_correction=not args.no_mw_correction,
    )
    if args.no_mw_correction:
        logger.info("Molecular-weight correction DISABLED (--no-mw-correction).")
    else:
        logger.warning(
            "Molecular-weight correction is ON. It is applied outside the network, "
            "so it survives every ablation: bias_only will rank by molecular weight "
            "rather than being a random baseline. Pass --no-mw-correction for a "
            "clean v(0)."
        )
    cache_dir = manager.cache_dir
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    mol_dir = cache_dir / "mols"
    ccd = load_canonicals(mol_dir)
    tokenizer = Boltz2Tokenizer()
    cropper = AffinityCropper()
    featurizer = Boltz2Featurizer()

    experiments: list[AblationExperiment] = [EXPERIMENTS[n] for n in args.experiments]
    n_ops = len({e.operator for e in experiments})
    logger.info(
        f"Running {len(experiments)} experiments per ligand "
        f"({n_ops} operator(s): {sorted({e.operator for e in experiments})})."
    )

    # ── Discover complexes and apply class filter
    cif_files = sorted(structures_dir.glob("*.cif"))
    if not cif_files:
        logger.error(f"[{receptor_id}] No CIFs found under {structures_dir}.")
        return 2

    candidates: List[Tuple[str, Path]] = []
    n_no_manifest = 0
    for p in cif_files:
        cid = _compound_id_from_cif(p)
        if cid not in smiles_table:
            n_no_manifest += 1
            continue
        if args.actives_only and not smiles_table[cid]["is_binder"]:
            continue
        if args.decoys_only and smiles_table[cid]["is_binder"]:
            continue
        candidates.append((cid, p))
    if args.max_ligands is not None:
        candidates = candidates[: args.max_ligands]

    n_actives = sum(1 for c, _ in candidates if smiles_table[c]["is_binder"])
    logger.info(
        f"[{receptor_id}] {len(candidates)} complexes to process "
        f"({n_actives} actives / {len(candidates) - n_actives} decoys; "
        f"{n_no_manifest} CIFs skipped: not in manifest)"
    )

    def featurize(compound_id: str, cif_path: Path):
        """Parse + featurize one predicted complex. Returns (batch, error)."""
        smiles = smiles_table[compound_id]["smiles"]
        prot, lig, chain_id, seq, err = _parse_complex(cif_path, smiles)
        if err is not None:
            return None, err
        return featurize_complex(
            protein_atoms=prot,
            ligand_atoms_remapped=lig,
            smiles=smiles,
            protein_chain_id=chain_id,
            protein_seq=seq,
            msa_path=str(msa_path),
            ccd=ccd,
            mol_dir=mol_dir,
            tokenizer=tokenizer,
            cropper=cropper,
            featurizer=featurizer,
            device=device,
        )

    # ── CSV header
    with output_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    t_start = time.perf_counter()

    # ── Donor pool (storage-aware) ──────────────────────────────────
    # The disk cache exists ONLY to supply donor channels for
    # resample/mean. A zero-only run never looks at another ligand's trunk,
    # so it writes nothing at all -- the query's trunk is computed once and
    # kept in memory just long enough to replay that ligand's experiments.
    # When donors ARE needed we cache a bounded, stratified pool rather than
    # every ligand: entries are ~16 MB each (z is N*N*token_z fp16), so a
    # full cache would be hundreds of GB per receptor.
    needs_donors = any(e.operator != "zero" for e in experiments)
    donor_pool: List[str] = []
    donor_pool_obj = None
    n_cache_built = n_cache_failed = 0

    if not needs_donors:
        logger.info(
            f"[{receptor_id}] zero-operator experiments only -- no trunk cache "
            f"will be written (saves ~"
            f"{trunk_cache.format_bytes(trunk_cache.bytes_per_entry() * len(candidates))})."
        )
    else:
        strata = {cid: smiles_table[cid]["is_binder"] for cid, _ in candidates}
        donor_pool = trunk_cache.select_donor_pool(
            [cid for cid, _ in candidates],
            pool_size=args.donor_pool_size,
            rng=np.random.default_rng(args.donor_pool_seed),
            strata=strata,
        )
        est = trunk_cache.bytes_per_entry() * len(donor_pool)
        logger.info(
            f"[{receptor_id}] resample/mean requested -- caching a donor pool of "
            f"{len(donor_pool)}/{len(candidates)} ligands "
            f"(~{trunk_cache.format_bytes(est)} on disk) at {trunk_cache_dir}"
        )

        cif_by_cid = dict(candidates)
        for i, cid in enumerate(donor_pool, 1):
            if not args.rebuild_trunk_cache and trunk_cache.load_cache_entry(
                trunk_cache_dir, receptor_id, cid
            ) is not None:
                continue
            batch, err = featurize(cid, cif_by_cid[cid])
            if err is not None:
                logger.warning(f"[{receptor_id}] donor {cid}: featurization failed: {err}")
                n_cache_failed += 1
                continue
            try:
                trunk_out = affinity_trunk_forward(model, batch, recycling_steps=args.recycling_steps)
                s_inputs = _affinity_input_embed_with_ablation(model, batch)
                token_repr_pos = batch["token_to_rep_atom"][0].float() @ _coords_affinity(batch)[0]
                trunk_cache.save_cache_entry(
                    trunk_cache_dir, receptor_id, cid,
                    z=trunk_out["z"], s_inputs=s_inputs, token_repr_pos=token_repr_pos,
                    use_kernels=trunk_out["use_kernels"],
                    meta={"recycling_steps": args.recycling_steps, "source": "dudez_cif",
                          "role": "donor_pool"},
                )
                n_cache_built += 1
                logger.info(f"[{receptor_id}] cached donor {i}/{len(donor_pool)}: {cid}")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[{receptor_id}] donor {cid}: trunk forward failed: {exc}")
                n_cache_failed += 1

        logger.info(
            f"[{receptor_id}] donor cache: {n_cache_built} built, {n_cache_failed} failed, "
            f"{trunk_cache.format_bytes(trunk_cache.cache_size_bytes(trunk_cache_dir, receptor_id))} on disk"
        )
        # Load the pool into memory ONCE for the whole receptor. Doing this
        # per query was an O(n_ligands x pool_size) full-payload read that
        # dominated the entire run.
        donor_pool_obj = trunk_cache.DonorPool(trunk_cache_dir, receptor_id, donor_pool)
        logger.info(f"[{receptor_id}] donor pool resident in memory: {len(donor_pool_obj)} entries")

    # ── Replay pass ─────────────────────────────────────────────────
    # The query's own trunk is computed here and held in memory only for the
    # duration of this ligand -- it is never written to disk. Only donor-pool
    # members live on disk, and only when resample/mean is in play.
    n_done = n_failed = 0
    for idx, (compound_id, cif_path) in enumerate(candidates, 1):
        is_binder = smiles_table[compound_id]["is_binder"]
        class_label = "active" if is_binder else "decoy"
        logger.info(f"[{receptor_id}] {idx}/{len(candidates)} {compound_id} ({class_label})")

        batch, err = featurize(compound_id, cif_path)
        if err is not None:
            logger.warning(f"  featurization failed: {err}")
            _write_rows(output_path, [
                _error_row(receptor_id, compound_id, is_binder, e, err) for e in experiments
            ])
            n_failed += 1
            continue

        # Reuse the donor-pool entry when this ligand happens to be in it,
        # otherwise run the trunk once, in memory.
        entry = trunk_cache.load_cache_entry(trunk_cache_dir, receptor_id, compound_id) \
            if needs_donors else None
        try:
            if entry is not None:
                query_n_tokens = entry["n_tokens"]
                trunk_out = {
                    "z": entry["z"].to(device=device, dtype=model_dtype).unsqueeze(0),
                    "use_kernels": entry["use_kernels"],
                }
            else:
                _t = affinity_trunk_forward(model, batch, recycling_steps=args.recycling_steps)
                trunk_out = {"z": _t["z"], "use_kernels": _t["use_kernels"]}
                query_n_tokens = int(_t["z"].shape[1])
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  trunk forward failed: {exc}")
            _write_rows(output_path, [
                _error_row(receptor_id, compound_id, is_binder, e, f"trunk forward: {exc}")
                for e in experiments
            ])
            n_failed += 1
            continue

        rows = []
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
                        donor_pool_obj, receptor_id, compound_id, query_n_tokens,
                        _affinity_input_embed_with_ablation, _build_cross_pair_mask, _get_module,
                    )
                    if out is None:
                        raise RuntimeError(f"resample/mean skipped: {skip_reason}")

                elapsed_ms = (time.perf_counter() - t0) * 1000
                rows.append(_result_row(
                    receptor_id, compound_id, is_binder, exp, out, elapsed_ms,
                    query_n_tokens, donor_ids, skip_reason,
                ))
                logger.info(
                    f"  [{exp.name}] pred={out['affinity_pred_value']:.4f} "
                    f"prob={out['affinity_probability_binary']:.4f} ({elapsed_ms:.0f}ms)"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"  [{exp.name}] FAILED: {exc}")
                rows.append(_error_row(receptor_id, compound_id, is_binder, exp, str(exc)))

        _write_rows(output_path, rows)
        n_done += 1

    cache_bytes = trunk_cache.cache_size_bytes(trunk_cache_dir, receptor_id)
    if needs_donors and not args.keep_trunk_cache:
        shutil.rmtree(trunk_cache_dir / receptor_id, ignore_errors=True)
        logger.info(
            f"[{receptor_id}] removed donor cache "
            f"({trunk_cache.format_bytes(cache_bytes)} reclaimed). "
            f"Pass --keep-trunk-cache to retain it."
        )

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"[{receptor_id}] DONE — {n_done} ok, {n_failed} failed, "
        f"{n_cache_failed} cache failures, {n_no_manifest} not in manifest, "
        f"{elapsed:.1f}s, output: {output_path}"
    )

    meta = trunk_cache.run_metadata({
        "runner": "run_feature_ablation_dudez.py",
        "receptor": receptor_id,
        "config_hash": hashlib.sha256(",".join(sorted(args.experiments)).encode()).hexdigest()[:16],
        "experiments": args.experiments,
        "recycling_steps": args.recycling_steps,
        "mw_correction": not args.no_mw_correction,
        "max_ligands": args.max_ligands,
        "structures_dir": str(structures_dir),
        "smiles_csv": str(smiles_csv),
        "msa_path": str(msa_path),
        "trunk_cache_dir": str(trunk_cache_dir),
        "checkpoint": args.checkpoint,
        "n_candidates": len(candidates),
        "n_ok": n_done,
        "n_failed": n_failed,
        "n_cache_failed": n_cache_failed,
        "n_not_in_manifest": n_no_manifest,
        "needs_donors": needs_donors,
        "donor_pool_size": len(donor_pool),
        "donor_pool_seed": args.donor_pool_seed,
        "donor_cache_bytes": cache_bytes,
        "donor_cache_kept": bool(args.keep_trunk_cache),
        "wall_time_s": elapsed,
    })
    sidecar = trunk_cache.write_json_sidecar(output_path, meta)
    logger.info(f"[{receptor_id}] sidecar: {sidecar}")

    return 0


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    return run_for_receptor(args)


if __name__ == "__main__":
    sys.exit(main())
