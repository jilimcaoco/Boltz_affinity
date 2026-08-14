#!/usr/bin/env python3
"""Task 4 — trunk-cache fidelity gate.

Every downstream ablation number depends on ``run_feature_ablation.py``'s
disk-backed trunk cache (see ``trunk_cache.py``) replaying the affinity head
against a *cached* trunk output rather than a freshly computed one. A silent
mismatch between the two -- a stale cache entry, a dtype/device cast bug in
``save_cache_entry``/``load_cache_entry``, a shape bug in how ``trunk_out``
is reconstructed in ``run_ablation_for_pair`` -- would corrupt every
downstream number while every analysis script kept running happily.

This script is the check: for a stratified sample of complexes that already
have a cached trunk entry, recompute the ``baseline`` (no ablation) affinity
prediction two ways:
  1. **Uncached**: run ``affinity_trunk_forward`` + ``affinity_head_forward``
     fresh, end to end, exactly as if the cache didn't exist.
  2. **Cached-replay**: load the cached trunk entry from disk and run only
     ``affinity_head_forward`` against it -- the same path
     ``run_feature_ablation.py`` takes for every ablation experiment.

and assert they agree to within 1e-4 in the predicted pIC50-like value. Must
be run (and pass) on the same machine/checkpoint/cache the ablation results
were produced with, before any of those results are trusted.

Usage
-----
python verify_trunk_cache.py --all-receptors --n-samples 200 \\
    --trunk-cache-dir ../results/trunk_cache
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import trunk_cache  # noqa: E402
from run_feature_ablation import (  # noqa: E402
    _coords_affinity,
    _featurize_ligand,
    default_paths,
    discover_pairs,
)

logger = logging.getLogger(__name__)

DELTA_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    receptors_dir, poses_dir, output_dir, trunk_cache_dir = default_paths()
    analysis_data_dir = _script_dir.parent / "analysis_data"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--receptor", type=Path, help="Single receptor PDB file.")
    group.add_argument("--all-receptors", action="store_true")

    parser.add_argument("--ligands", type=Path, default=None)
    parser.add_argument("--receptors-dir", type=Path, default=receptors_dir)
    parser.add_argument("--poses-dir", type=Path, default=poses_dir)
    parser.add_argument("--trunk-cache-dir", type=Path, default=trunk_cache_dir)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--recycling-steps", type=int, default=5)
    parser.add_argument("--use-msa-server", action="store_true", default=False)
    parser.add_argument("--output", type=Path, default=analysis_data_dir / "cache_fidelity.csv")
    parser.add_argument("--tolerance", type=float, default=DELTA_TOLERANCE)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    if args.receptor and not args.ligands:
        parser.error("--ligands is required when using --receptor.")
    return args


def _stratified_sample(receptor_to_ligands: dict, n_samples: int, rng) -> List[Tuple[str, str]]:
    """Sample ~equally across receptors up to ``n_samples`` total pairs."""
    receptors = sorted(receptor_to_ligands.keys())
    if not receptors:
        return []
    per_receptor = max(1, n_samples // len(receptors))
    sampled = []
    for receptor_id in receptors:
        ligs = receptor_to_ligands[receptor_id]
        if not ligs:
            continue
        k = min(per_receptor, len(ligs))
        idx = rng.choice(len(ligs), size=k, replace=False)
        sampled.extend((receptor_id, ligs[i]) for i in idx)
    # top up / trim to exactly n_samples where possible
    if len(sampled) > n_samples:
        keep_idx = rng.choice(len(sampled), size=n_samples, replace=False)
        sampled = [sampled[i] for i in sorted(keep_idx)]
    return sampled


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s [%(levelname)s] %(message)s")

    import numpy as np
    import torch

    from boltz.affinity_rescoring.inference import (
        AffinityModelManager,
        affinity_head_forward,
        affinity_trunk_forward,
    )
    from boltz.affinity_rescoring.mol2_parser import MOL2Parser
    from boltz.affinity_rescoring.models import DeviceOption
    from boltz.affinity_rescoring.parsers import (
        get_chain_sequences,
        get_seqres_sequences,
        parse_structure_file,
    )
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer

    logger.info("Loading model...")
    manager = AffinityModelManager(device=DeviceOption(args.device))
    model = manager.load_model(checkpoint_path=args.checkpoint if args.checkpoint != "auto" else None)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    logger.info(f"Model loaded on {manager.device}")

    if args.all_receptors:
        pairs = discover_pairs(args.receptors_dir, args.poses_dir)
    else:
        receptor_id = args.receptor.stem.removesuffix("_receptor")
        pairs = [(receptor_id, args.receptor, args.ligands)]

    mol_dir = manager.cache_dir / "mols"
    ccd = load_canonicals(mol_dir)
    tokenizer = Boltz2Tokenizer()
    cropper = AffinityCropper()
    featurizer = Boltz2Featurizer()
    mol2_parser = MOL2Parser()

    # Only sample ligands that already have a cached trunk entry -- this
    # gate verifies cache fidelity, it does not populate the cache itself
    # (run run_feature_ablation.py first).
    receptor_to_ligands = {}
    receptor_paths = {}
    for receptor_id, receptor_path, ligands_path in pairs:
        cached = trunk_cache.list_cached_ligands(args.trunk_cache_dir, receptor_id)
        if not cached:
            logger.warning(f"[{receptor_id}] no cached trunk entries -- skipping "
                            f"(run run_feature_ablation.py first).")
            continue
        receptor_to_ligands[receptor_id] = cached
        receptor_paths[receptor_id] = (receptor_path, ligands_path)

    if not receptor_to_ligands:
        raise SystemExit(
            "No cached trunk entries found under "
            f"{args.trunk_cache_dir}. Run run_feature_ablation.py first to populate the cache."
        )

    rng = np.random.default_rng(args.seed)
    sample = _stratified_sample(receptor_to_ligands, args.n_samples, rng)
    logger.info(f"Sampled {len(sample)} (receptor, ligand) pairs across {len(receptor_to_ligands)} receptors.")

    rows = []
    n_pass = 0
    n_fail = 0
    n_error = 0

    for receptor_id, ligand_name in sample:
        receptor_path, ligands_path = receptor_paths[receptor_id]
        try:
            protein_atoms, _ = parse_structure_file(receptor_path)
            protein_chain_id = "A"
            merged_refs = get_seqres_sequences(receptor_path)
            sequences = get_chain_sequences(protein_atoms, reference_sequences=merged_refs)
            if protein_chain_id not in sequences:
                protein_chain_id = next(iter(sequences))
            protein_seq = sequences[protein_chain_id]

            msa_path = None  # verification uses the model's own single-sequence path for speed;
                              # cache entries built with an MSA should be verified with --use-msa-server.
            if args.use_msa_server:
                import tempfile
                from boltz.main import compute_msa
                msa_dir = Path(tempfile.mkdtemp(prefix="boltz_verify_msa_"))
                target_id = f"receptor_{protein_chain_id}"
                compute_msa(
                    data={target_id: protein_seq}, target_id=target_id, msa_dir=msa_dir,
                    msa_server_url="https://api.colabfold.com", msa_pairing_strategy="paired+unpaired",
                )
                found = list(msa_dir.glob("*.a3m")) or list(msa_dir.glob("*.csv"))
                if found:
                    msa_path = str(found[0].resolve())

            ligands = mol2_parser.extract_ligands_with_names(ligands_path)
            ligand = next((lg for lg in ligands if lg.name == ligand_name), None)
            if ligand is None:
                raise RuntimeError(f"ligand {ligand_name} not found in {ligands_path}")

            batch, err = _featurize_ligand(
                ligand, receptor_id, protein_chain_id, protein_seq, protein_atoms,
                msa_path, ccd, mol_dir, tokenizer, cropper, featurizer, device,
            )
            if err is not None:
                raise RuntimeError(f"featurization failed: {err}")

            # 1. Uncached: full trunk + head, fresh.
            t0 = time.perf_counter()
            trunk_out_uncached = affinity_trunk_forward(model, batch, recycling_steps=args.recycling_steps)
            out_uncached = affinity_head_forward(model, batch, trunk_out_uncached)
            t_uncached = time.perf_counter() - t0

            # 2. Cached-replay: load from disk, head only.
            entry = trunk_cache.load_cache_entry(args.trunk_cache_dir, receptor_id, ligand_name)
            if entry is None:
                raise RuntimeError("cache entry disappeared between listing and load")
            t0 = time.perf_counter()
            trunk_out_cached = {
                "z": entry["z"].to(device=device, dtype=model_dtype).unsqueeze(0),
                "use_kernels": entry["use_kernels"],
            }
            out_cached = affinity_head_forward(model, batch, trunk_out_cached)
            t_cached = time.perf_counter() - t0

            delta = abs(out_uncached["affinity_pred_value"] - out_cached["affinity_pred_value"])
            passed = delta < args.tolerance
            n_pass += int(passed)
            n_fail += int(not passed)

            rows.append({
                "receptor_id": receptor_id, "ligand_name": ligand_name,
                "uncached_pred_value": out_uncached["affinity_pred_value"],
                "cached_pred_value": out_cached["affinity_pred_value"],
                "abs_delta": delta, "tolerance": args.tolerance, "pass": passed,
                "n_tokens": entry["n_tokens"],
                "uncached_time_ms": t_uncached * 1000, "cached_time_ms": t_cached * 1000,
                "error": "",
            })
            level = logger.info if passed else logger.error
            level(f"[{receptor_id}] {ligand_name}: uncached={out_uncached['affinity_pred_value']:.6f} "
                  f"cached={out_cached['affinity_pred_value']:.6f} delta={delta:.2e} "
                  f"({'PASS' if passed else 'FAIL'}) "
                  f"[{t_uncached*1000:.0f}ms -> {t_cached*1000:.0f}ms]")

        except Exception as e:
            n_error += 1
            logger.error(f"[{receptor_id}] {ligand_name}: ERROR {e}")
            rows.append({
                "receptor_id": receptor_id, "ligand_name": ligand_name,
                "uncached_pred_value": "", "cached_pred_value": "",
                "abs_delta": "", "tolerance": args.tolerance, "pass": False,
                "n_tokens": "", "uncached_time_ms": "", "cached_time_ms": "",
                "error": str(e),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["receptor_id", "ligand_name", "uncached_pred_value", "cached_pred_value",
                  "abs_delta", "tolerance", "pass", "n_tokens",
                  "uncached_time_ms", "cached_time_ms", "error"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    logger.info(f"\n{'=' * 60}")
    logger.info("TRUNK CACHE FIDELITY GATE")
    logger.info(f"  Sampled     : {n_total}")
    logger.info(f"  Passed      : {n_pass}")
    logger.info(f"  Failed      : {n_fail}")
    logger.info(f"  Errored     : {n_error}")
    logger.info(f"  Tolerance   : |delta pIC50| < {args.tolerance}")
    logger.info(f"  Output      : {args.output}")
    logger.info(f"{'=' * 60}")

    if n_fail or n_error:
        raise SystemExit(
            f"FIDELITY GATE FAILED: {n_fail} mismatches + {n_error} errors out of {n_total}. "
            f"Do not trust ablation results produced against {args.trunk_cache_dir} until this is "
            f"fixed and this script passes clean. See {args.output} for per-complex detail."
        )
    logger.info("Fidelity gate PASSED. Cached-replay results may be trusted.")


if __name__ == "__main__":
    main()
