#!/usr/bin/env python
"""Precompute MSAs for each unique protein chain in the receptor YAMLs.

We only have two targets (DRD4, 5HT2A) so this makes exactly one ColabFold
API call per unique protein sequence — instead of one call per ligand in the
later ``boltz predict`` step.

Outputs one ``.a3m`` file per (target, chain) in ``<msa_dir>/``:

    <msa_dir>/DRD4_A.a3m
    <msa_dir>/5HT2A_A.a3m

``prepare_predict_inputs.py --msa-dir <msa_dir>`` then embeds the absolute
path into every per-ligand YAML so ``boltz predict`` never needs to contact
the MSA server.

Usage
-----
    python precompute_msas.py \\
        --drd4-receptor  receptors/DRD4.yaml  \\
        --ht2a-receptor  receptors/5HT2A.yaml \\
        --msa-dir        msa/ \\
        [--host-url https://api.colabfold.com]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# locate boltz's precompute_msa helper
try:
    from boltz.affinity_rescoring.mmseqs2 import precompute_msa
    from boltz.affinity_rescoring.msa_cache import (
        canonical_msa_filename,
        find_msa,
        write_legacy_symlinks,
    )
except ImportError as exc:
    sys.exit(
        f"Could not import boltz.affinity_rescoring: {exc}\n"
        "Make sure the boltz package is installed in the active conda env."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _protein_chains(receptor_yaml: Path) -> list[tuple[str, str]]:
    """Return ``[(chain_id, sequence), ...]`` for every protein chain."""
    data = yaml.safe_load(receptor_yaml.read_text())
    chains: list[tuple[str, str]] = []
    for entry in data.get("sequences") or []:
        if "protein" not in entry:
            continue
        body = entry["protein"]
        seq = body.get("sequence")
        cid = body.get("id")
        if not seq:
            log.warning("Protein entry in %s has no 'sequence' field — skipping", receptor_yaml.name)
            continue
        if isinstance(cid, list):
            cid = cid[0]  # take first id for naming
        chains.append((str(cid), str(seq)))
    return chains


def _precompute_target(
    target: str,
    receptor_yaml: Path,
    msa_dir: Path,
    host_url: str,
) -> dict[str, str]:
    """Run MSA for every protein chain in *receptor_yaml*.

    Returns ``{chain_id: absolute_a3m_path}`` for use in reporting.
    Skips chains whose ``.a3m`` file already exists.
    """
    chains = _protein_chains(receptor_yaml)
    if not chains:
        log.warning("[%s] No protein chains found in %s", target, receptor_yaml)
        return {}

    result: dict[str, str] = {}
    for chain_id, sequence in chains:
        canonical_name = canonical_msa_filename(sequence)
        canonical_path = msa_dir / canonical_name
        legacy_path = msa_dir / f"{target}_{chain_id}.a3m"

        # Reuse any existing cached MSA (canonical, legacy, or one already
        # discoverable via $BOLTZ_MSA_CACHE_DIR) before hitting ColabFold.
        existing = find_msa(
            sequence=sequence,
            msa_dirs=[msa_dir],
            chain_id=chain_id,
            target=target,
        )
        if existing is not None:
            log.info("[%s] chain %s — already cached at %s", target, chain_id, existing)
            # Make sure the canonical name exists too, so downstream
            # consumers find it via sequence hash.
            if existing.resolve() != canonical_path.resolve() and not canonical_path.exists():
                try:
                    canonical_path.symlink_to(existing.name)
                except OSError:
                    canonical_path.write_bytes(existing.read_bytes())
            write_legacy_symlinks(canonical_path, chain_id=chain_id, target=target)
            result[chain_id] = str(canonical_path.resolve())
            continue

        log.info(
            "[%s] chain %s — submitting to ColabFold (seq length %d) → %s",
            target, chain_id, len(sequence), canonical_path,
        )
        precompute_msa(
            sequence=sequence,
            out_path=canonical_path,
            host_url=host_url,
        )
        # Maintain backwards-compatible filenames so older code paths
        # (legacy `<target>_<chain_id>.a3m` lookups) keep working.
        write_legacy_symlinks(canonical_path, chain_id=chain_id, target=target)
        result[chain_id] = str(canonical_path.resolve())
        log.info(
            "[%s] chain %s — done → %s (legacy alias: %s)",
            target, chain_id, canonical_path, legacy_path,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drd4-receptor", required=True, type=Path,
                        help="Boltz YAML for the DRD4 target.")
    parser.add_argument("--ht2a-receptor", required=True, type=Path,
                        help="Boltz YAML for the 5HT2A target.")
    parser.add_argument("--msa-dir", required=True, type=Path,
                        help="Directory where .a3m files will be written.")
    parser.add_argument("--host-url", default="https://api.colabfold.com",
                        help="ColabFold API endpoint (default: %(default)s).")
    args = parser.parse_args()

    args.msa_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "DRD4": args.drd4_receptor,
        "5HT2A": args.ht2a_receptor,
    }

    all_ok = True
    for tgt, yaml_path in targets.items():
        if not yaml_path.exists():
            log.error("[%s] receptor YAML not found: %s", tgt, yaml_path)
            all_ok = False
            continue
        chains = _precompute_target(tgt, yaml_path, args.msa_dir, args.host_url)
        if not chains:
            all_ok = False
        for chain_id, a3m_path in chains.items():
            print(f"[OK] {tgt} chain {chain_id} → {a3m_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
