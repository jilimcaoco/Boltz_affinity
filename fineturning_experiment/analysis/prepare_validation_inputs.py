#!/usr/bin/env python
"""Build per-ligand Boltz prediction YAMLs from the held-out validation CSVs.

Reads ``../validation_data/D4_in_vitro_data.csv`` and
``../validation_data/5ht2a_in_vitro_data.csv`` and emits

    <out_dir>/DRD4/<zinc_id>.yaml
    <out_dir>/5HT2A/<zinc_id>.yaml

plus a tidy labels CSV per target

    <labels_dir>/labels_DRD4.csv
    <labels_dir>/labels_5HT2A.csv

with the columns the evaluator consumes:

    name, smiles, exp_activity, is_binder, exp_pIC50 (DRD4 only),
    plus the raw assay columns for provenance.

The downstream :mod:`evaluate_adapters` joins these on ``name``.
"""
from __future__ import annotations

import argparse
import copy
import math
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_LARGEST = rdMolStandardize.LargestFragmentChooser()

# 5HT2A binder cutoff: ≤50% radioligand remaining = bound (active displacer).
HT2A_PERCENT_BOUND_CUTOFF = 50.0


def sanitize_smiles(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = _LARGEST.choose(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def load_protein_block(receptor_yaml: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(receptor_yaml.read_text())
    keep = [e for e in (data.get("sequences") or []) if "ligand" not in e]
    if not keep:
        raise ValueError(f"No protein sequences in {receptor_yaml}")
    return keep


def _inject_msa(block: list[dict[str, Any]], msa_dir: Optional[Path], target: str) -> list[dict[str, Any]]:
    """Attach a pre-computed MSA to every protein chain in ``block``.

    Resolves MSAs through :func:`boltz.affinity_rescoring.msa_cache.find_msa`
    so canonical hash filenames, legacy ``<target>_<chain>.a3m`` names,
    and ``$BOLTZ_MSA_CACHE_DIR`` entries are all honored uniformly.
    """
    from boltz.affinity_rescoring.msa_cache import find_msa

    out = copy.deepcopy(block)
    search_dirs = [msa_dir] if msa_dir is not None else None
    for entry in out:
        if "protein" not in entry:
            continue
        body = entry["protein"]
        cid = body.get("id")
        cid = cid[0] if isinstance(cid, list) else cid
        seq = body.get("sequence", "")
        hit = find_msa(
            sequence=seq,
            msa_dirs=search_dirs,
            chain_id=cid,
            target=target,
            allow_single_file_fallback=True,
        )
        if hit is not None:
            body["msa"] = str(Path(hit).resolve())
    return out


def _next_chain_id(seqs: list[dict[str, Any]]) -> str:
    used = set()
    for entry in seqs:
        for body in entry.values():
            cid = body.get("id")
            if isinstance(cid, str):
                used.add(cid)
            elif isinstance(cid, list):
                used.update(cid)
    for c in "BCDEFGHIJKLMNOPQRSTUVWXYZ":
        if c not in used:
            return c
    raise RuntimeError("Out of chain IDs")


def write_yaml(out_path: Path, block: list[dict[str, Any]], smiles: str) -> None:
    seqs = [dict(e) for e in copy.deepcopy(block)]
    cid = _next_chain_id(seqs)
    seqs.append({"ligand": {"id": cid, "smiles": smiles}})
    doc = {
        "version": 1,
        "sequences": seqs,
        "properties": [{"affinity": {"binder": cid}}],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _safe_name(raw: str) -> str:
    """Boltz uses the YAML stem as record id — keep it filesystem- and CLI-safe."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("_") or "unnamed"


def _parse_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def prepare_drd4(csv: Path, receptor: Path, out_dir: Path, labels_dir: Path,
                 msa_dir: Optional[Path]) -> None:
    df = pd.read_csv(csv)
    block = load_protein_block(receptor)
    block_msa = _inject_msa(block, msa_dir, "DRD4")

    labels: list[dict[str, Any]] = []
    written = skipped = 0
    seen: set[str] = set()
    for _, row in df.iterrows():
        zinc = str(row.get("zincid", "")).strip()
        smi_raw = str(row.get("smiles", "")).strip()
        if not zinc or not smi_raw:
            skipped += 1
            continue
        smi = sanitize_smiles(smi_raw)
        if smi is None:
            skipped += 1
            continue
        name = _safe_name(zinc)
        if name in seen:
            continue
        seen.add(name)

        ki_nm = _parse_float(row.get("D4 Ki(nM)"))
        binder_raw = _parse_float(row.get("Binder"))
        is_binder = int(binder_raw) if binder_raw is not None else None

        # Continuous activity: prefer pKi; fall back to % inhibition for ranking
        # (% inhibition is monotone-up with activity, opposite sign of Ki).
        if ki_nm is not None and ki_nm > 0:
            exp_activity = -math.log10(ki_nm)        # = 9 - pKi(M); higher = stronger
            exp_pic50 = 9.0 - math.log10(ki_nm)      # pKi proxy for calibration metric
        else:
            inhib = _parse_float(row.get("Inhibition (%) at 10uM"))
            if inhib is None:
                # No usable label — skip from the labels CSV (yaml still emitted for completeness)
                exp_activity = None
                exp_pic50 = None
            else:
                exp_activity = inhib / 100.0
                exp_pic50 = None

        write_yaml(out_dir / f"{name}.yaml", block_msa, smi)
        written += 1
        labels.append({
            "name": name,
            "smiles": smi,
            "exp_activity": exp_activity,
            "exp_pIC50": exp_pic50,
            "is_binder": is_binder,
            "D4_Ki_nM": ki_nm,
            "inhibition_pct_10uM": _parse_float(row.get("Inhibition (%) at 10uM")),
        })

    labels_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(labels).to_csv(labels_dir / "labels_DRD4.csv", index=False)
    print(f"[prepare] DRD4: wrote {written} YAMLs, skipped {skipped}, "
          f"labels rows = {len(labels)}")


def prepare_ht2a(csv: Path, receptor: Path, out_dir: Path, labels_dir: Path,
                 msa_dir: Optional[Path]) -> None:
    df = pd.read_csv(csv)
    block = load_protein_block(receptor)
    block_msa = _inject_msa(block, msa_dir, "5HT2A")

    labels: list[dict[str, Any]] = []
    written = skipped = 0
    seen: set[str] = set()
    for _, row in df.iterrows():
        zinc = str(row.get("ZINC ID", "")).strip()
        smi_raw = str(row.get("SMILES", "")).strip()
        if not zinc or not smi_raw or zinc.lower() == "buffer":
            skipped += 1
            continue
        smi = sanitize_smiles(smi_raw)
        if smi is None:
            skipped += 1
            continue
        name = _safe_name(zinc)
        if name in seen:
            continue
        seen.add(name)

        pct_bound = _parse_float(row.get("% bound 3[H]-LSD @ 10uM (mean)"))
        if pct_bound is None:
            skipped += 1
            continue
        # Lower % bound = more displacement = stronger binder.
        exp_activity = -pct_bound
        is_binder = int(pct_bound <= HT2A_PERCENT_BOUND_CUTOFF)

        write_yaml(out_dir / f"{name}.yaml", block_msa, smi)
        written += 1
        labels.append({
            "name": name,
            "smiles": smi,
            "exp_activity": exp_activity,
            "is_binder": is_binder,
            "percent_bound_10uM": pct_bound,
            "percent_bound_sem": _parse_float(row.get("% bound 3[H]-LSD @ 10uM (SEM)")),
        })

    labels_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(labels).to_csv(labels_dir / "labels_5HT2A.csv", index=False)
    print(f"[prepare] 5HT2A: wrote {written} YAMLs, skipped {skipped}, "
          f"labels rows = {len(labels)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drd4-csv", required=True, type=Path)
    p.add_argument("--ht2a-csv", required=True, type=Path)
    p.add_argument("--drd4-receptor", required=True, type=Path)
    p.add_argument("--ht2a-receptor", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Will contain DRD4/ and 5HT2A/ sub-dirs of YAMLs.")
    p.add_argument("--labels-dir", required=True, type=Path,
                   help="Where labels_{TARGET}.csv files are written.")
    p.add_argument("--msa-dir", type=Path, default=None,
                   help="Optional precomputed MSA directory ({target}_{chain}.a3m).")
    args = p.parse_args()

    prepare_drd4(args.drd4_csv,  args.drd4_receptor,
                 args.out_dir / "DRD4",  args.labels_dir, args.msa_dir)
    prepare_ht2a(args.ht2a_csv,  args.ht2a_receptor,
                 args.out_dir / "5HT2A", args.labels_dir, args.msa_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
