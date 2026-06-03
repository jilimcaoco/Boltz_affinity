#!/usr/bin/env python
"""Build per-row Boltz prediction YAMLs from the curated ChEMBL CSV.

For every row in the curated CSV, emit a Boltz input YAML containing the
target protein (taken from the per-target receptor YAML) plus the ligand
SMILES.  Two output sub-directories are produced::

    <out_dir>/DRD4/<molecule_chembl_id>.yaml
    <out_dir>/5HT2A/<molecule_chembl_id>.yaml

Run ``boltz predict <out_dir>/DRD4`` (and 5HT2A) to generate complex
structures, then point :mod:`build_lora_manifest` at the collected poses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

# Silence noisy RDKit parse warnings; we log our own rejection counts.
RDLogger.DisableLog("rdApp.*")

_LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()


def sanitize_smiles(smiles: str) -> Optional[str]:
    """Return a canonical, fully-sanitized SMILES, or ``None`` if invalid.

    Mirrors Boltz's ``standardize()`` (src/boltz/data/parse/schema.py) but with
    RDKit sanitization *enabled*, so the resulting SMILES round-trips cleanly
    through Boltz's ``Chem.MolFromSmiles(..., sanitize=False)`` +
    ``LargestFragmentChooser`` call without hitting the
    ``getNumImplicitHs() called without preceding call to calcImplicitValence()``
    precondition violation.

    Steps:
      1. Parse with full sanitization (computes implicit valences).
      2. Pick the largest covalent fragment (drops salts/counter-ions).
      3. Emit canonical isomeric SMILES.
      4. Re-parse with ``sanitize=False`` to verify Boltz's parser will accept it.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)  # sanitize=True by default
        if mol is None:
            return None
        mol = _LARGEST_FRAGMENT_CHOOSER.choose(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        canon = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if not canon:
            return None
        # Final sanity check: this is exactly what Boltz will do internally.
        probe = Chem.MolFromSmiles(canon, sanitize=False)
        if probe is None:
            return None
        # Force implicit-valence computation so the probe behaves like the
        # mol object Boltz will hand to LargestFragmentChooser.
        for atom in probe.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(probe)
        return canon
    except Exception:
        return None


def load_protein_block(receptor_yaml: Path) -> list[dict[str, Any]]:
    """Return the ``sequences`` entries from ``receptor_yaml`` minus any ligand."""
    data = yaml.safe_load(receptor_yaml.read_text())
    seqs = data.get("sequences") or []
    keep: list[dict[str, Any]] = []
    for entry in seqs:
        if "ligand" in entry:
            # drop placeholder ligands; we'll add our own
            continue
        keep.append(entry)
    if not keep:
        msg = f"No protein/RNA/DNA sequences found in {receptor_yaml}"
        raise ValueError(msg)
    return keep


def _next_chain_id(existing: list[dict[str, Any]]) -> str:
    used = set()
    for entry in existing:
        for body in entry.values():
            cid = body.get("id")
            if isinstance(cid, str):
                used.add(cid)
            elif isinstance(cid, list):
                used.update(cid)
    for c in "BCDEFGHIJKLMNOPQRSTUVWXYZ":
        if c not in used:
            return c
    msg = "Ran out of single-letter chain IDs"
    raise RuntimeError(msg)


def _inject_msa_paths(
    protein_block: list[dict[str, Any]],
    msa_dir: Path,
    target: str,
) -> list[dict[str, Any]]:
    """Return a deep-ish copy of *protein_block* with ``msa:`` fields added.

    Resolution uses the shared
    :func:`boltz.affinity_rescoring.msa_cache.find_msa` helper so files
    produced by *either* the canonical sequence-hash precompute or the
    legacy ``<target>_<chain_id>.a3m`` precompute are picked up. Chains
    with no cached MSA are left unmodified; if the resulting YAML is
    then fed to ``boltz predict`` without an MSA path, prediction will
    error out instead of silently calling the MSA server (which is
    disabled in this fork).
    """
    import copy

    try:
        from boltz.affinity_rescoring.msa_cache import find_msa
    except Exception:  # pragma: no cover
        find_msa = None  # type: ignore[assignment]

    block = copy.deepcopy(protein_block)
    for entry in block:
        if "protein" not in entry:
            continue
        body = entry["protein"]
        cid = body.get("id")
        if isinstance(cid, list):
            cid = cid[0]
        sequence = body.get("sequence")
        hit = None
        if find_msa is not None:
            hit = find_msa(
                sequence=sequence,
                msa_dirs=[msa_dir],
                chain_id=str(cid) if cid is not None else None,
                target=target,
            )
        # Backward-compat fallback (in case msa_cache import failed):
        if hit is None:
            legacy = msa_dir / f"{target}_{cid}.a3m"
            if legacy.exists() and legacy.stat().st_size > 0:
                hit = legacy.resolve()
        if hit is not None:
            body["msa"] = str(hit)
    return block


def write_input_yaml(
    out_path: Path,
    protein_block: list[dict[str, Any]],
    smiles: str,
    ligand_id: str | None = None,
    msa_dir: Path | None = None,
    target: str | None = None,
) -> None:
    seqs = (
        _inject_msa_paths(protein_block, msa_dir, target)
        if (msa_dir is not None and target is not None)
        else [dict(e) for e in protein_block]
    )
    ligand_chain = ligand_id or _next_chain_id(seqs)
    seqs.append({"ligand": {"id": ligand_chain, "smiles": smiles}})
    doc = {
        "version": 1,
        "sequences": seqs,
        "properties": [{"affinity": {"binder": ligand_chain}}],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-csv", required=True, type=Path)
    parser.add_argument("--drd4-receptor", required=True, type=Path)
    parser.add_argument("--ht2a-receptor", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Will contain DRD4/ and 5HT2A/ sub-dirs of YAMLs.")
    parser.add_argument("--keep-censored", action="store_true",
                        help="Include rows with is_censored=True. Default drops them.")
    parser.add_argument("--msa-dir", type=Path, default=None,
                        help="Directory containing precomputed {target}_{chain}.a3m files "
                             "(output of precompute_msas.py). When provided, each YAML gets "
                             "an embedded msa: path so boltz predict skips the MSA server.")
    args = parser.parse_args()

    df = pd.read_csv(args.curated_csv)
    if not args.keep_censored and "is_censored" in df.columns:
        df = df[~df["is_censored"].astype(bool)]
    # Accept either 'source_target' (current pull_chembl_affinity_data.py
    # output) or legacy 'target_label' for backward compatibility.
    if "source_target" in df.columns:
        target_col = "source_target"
    elif "target_label" in df.columns:
        target_col = "target_label"
    else:
        msg = (
            "curated CSV is missing the target-label column "
            "(expected 'source_target' or 'target_label')."
        )
        raise ValueError(msg)
    if "molecule_chembl_id" not in df.columns:
        msg = "curated CSV is missing molecule_chembl_id"
        raise ValueError(msg)

    smiles_col = "canonical_smiles_std" if "canonical_smiles_std" in df.columns else "canonical_smiles"

    targets = {
        "DRD4": load_protein_block(args.drd4_receptor),
        "5HT2A": load_protein_block(args.ht2a_receptor),
    }

    counts = {"DRD4": 0, "5HT2A": 0, "skipped": 0, "bad_smiles": 0}
    # Dedup: one prediction per (target, molecule) — multiple assays share the pose
    seen: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        tgt = str(row[target_col])
        mol = str(row["molecule_chembl_id"])
        key = (tgt, mol)
        if key in seen:
            continue
        seen.add(key)
        if tgt not in targets:
            counts["skipped"] += 1
            continue
        raw_smiles = row.get(smiles_col)
        if not isinstance(raw_smiles, str) or not raw_smiles:
            counts["skipped"] += 1
            continue
        smiles = sanitize_smiles(raw_smiles)
        if smiles is None:
            counts["bad_smiles"] += 1
            continue
        out_path = args.out_dir / tgt / f"{mol}.yaml"
        write_input_yaml(out_path, targets[tgt], smiles,
                         msa_dir=args.msa_dir, target=tgt)
        counts[tgt] += 1

    print(f"[predict-inputs] wrote DRD4={counts['DRD4']}  5HT2A={counts['5HT2A']}  "
          f"skipped={counts['skipped']}  bad_smiles={counts['bad_smiles']}")
    print(f"[predict-inputs] output dir: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
