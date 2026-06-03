#!/usr/bin/env python
"""Collect Boltz predicted complexes into a flat per-target poses dir.

Boltz writes predictions to
``<out_dir>/boltz_results_<input_dir_stem>/predictions/<rec>/<rec>_model_<k>.<fmt>``.
This script symlinks the rank-0 prediction for each record into
``<poses_dir>/<rec>.pdb`` so :mod:`build_lora_manifest` can find them with
the default ``--pose-pattern`` (``{molecule_chembl_id}.pdb``).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def collect(boltz_out_dir: Path, poses_dir: Path, *,
            model_index: int = 0, copy: bool = False) -> int:
    """Return the number of poses linked / copied."""
    poses_dir.mkdir(parents=True, exist_ok=True)
    # Find the single boltz_results_* sub-dir
    candidates = list(boltz_out_dir.glob("boltz_results_*"))
    if not candidates:
        msg = f"No boltz_results_* directory under {boltz_out_dir}"
        raise FileNotFoundError(msg)
    if len(candidates) > 1:
        # Pick the most recently modified
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    results_dir = candidates[0] / "predictions"
    if not results_dir.is_dir():
        msg = f"Predictions directory not found: {results_dir}"
        raise FileNotFoundError(msg)

    linked = 0
    for rec_dir in sorted(results_dir.iterdir()):
        if not rec_dir.is_dir():
            continue
        rec = rec_dir.name
        # Try pdb first, then mmcif
        pdb_candidates = list(rec_dir.glob(f"{rec}_model_{model_index}.pdb"))
        cif_candidates = (
            list(rec_dir.glob(f"{rec}_model_{model_index}.cif"))
            + list(rec_dir.glob(f"{rec}_model_{model_index}.mmcif"))
        )
        src = pdb_candidates[0] if pdb_candidates else (cif_candidates[0] if cif_candidates else None)
        if src is None:
            print(f"[collect] WARNING no model_{model_index} for {rec}")
            continue
        # Always normalise to .pdb extension downstream
        ext = ".pdb" if src.suffix == ".pdb" else src.suffix
        dst = poses_dir / f"{rec}{ext}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy:
            shutil.copy2(src, dst)
        else:
            # Relative symlink keeps things portable across moves of the parent dir
            rel = os.path.relpath(src, dst.parent)
            os.symlink(rel, dst)
        linked += 1
    return linked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boltz-out", required=True, type=Path,
                        help="Boltz predict --out_dir for one target.")
    parser.add_argument("--poses-dir", required=True, type=Path,
                        help="Where to write {record}.pdb files.")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking.")
    args = parser.parse_args()

    n = collect(args.boltz_out, args.poses_dir, model_index=args.model_index, copy=args.copy)
    print(f"[collect] {n} pose(s) -> {args.poses_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
