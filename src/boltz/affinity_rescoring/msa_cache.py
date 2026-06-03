"""Unified MSA cache discovery for the Boltz_affinity fork.

This module is the *single source of truth* for locating pre-computed
MSAs across every consumer in this fork:

* ``boltz predict`` (the upstream Boltz-2 prediction path)
* ``boltz rescore *`` (the affinity-rescoring pipeline)
* ``boltz lora train`` / ``boltz lora update`` (LoRA fine-tuning)
* ``fineturning_experiment/`` helpers (``prepare_predict_inputs.py``,
  ``precompute_msas.py``)
* ``scripts/prepare_lora_manifest.py``

Why a shared module?
--------------------
The fork has historically grown three almost-but-not-quite-compatible
MSA naming conventions:

============================  ==================================
Producer                      Filename pattern
============================  ==================================
``rescorer.py`` (affinity)    ``<chain_id>.a3m`` / ``<chain_id>.csv``
``precompute_msas.py``        ``<target>_<chain_id>.a3m``
``boltz predict`` (internal)  ``<entity>.csv``
============================  ==================================

That means an MSA written by one pipeline was often invisible to the
other two, and people kept reaching for ``--use-msa-server`` to fill
the gap — which hammers the public ColabFold endpoint and gets jobs
killed after a few dozen sequences.

This module fixes that by:

1. Defining a single **canonical** filename for a given sequence —
   ``<sha256(seq)[:16]>.a3m`` — so identical sequences across datasets
   share one MSA file regardless of chain ID or target name.
2. Providing :func:`find_msa` which probes the canonical name *and*
   every legacy name in a stable order before giving up.
3. Exposing :func:`canonical_msa_path` so producers
   (``precompute_msas.py``, ``precompute_msa``) can write the canonical
   file and optionally symlink legacy names for backwards compatibility.
4. Defining :class:`MSAServerDisabledError` so every CLI surface can
   raise the *same* error message when a user passes ``--use-msa-server``.

Policy
------
The public ColabFold MMseqs2 server (`https://api.colabfold.com`) is
**off-limits** for routine inference in this fork.  The *only*
sanctioned use of :func:`boltz.affinity_rescoring.mmseqs2.precompute_msa`
is from an explicit precompute script (e.g.
``fineturning_experiment/precompute_msas.py``), run once per unique
sequence, with the resulting ``.a3m`` cached on local/turbo storage and
re-used by every downstream pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


# ── Public error ────────────────────────────────────────────────────────────

class MSAServerDisabledError(RuntimeError):
    """Raised when a user attempts to invoke the ColabFold MSA server.

    The fork policy is to *always* pre-compute MSAs with
    :func:`boltz.affinity_rescoring.mmseqs2.precompute_msa` and reuse
    the cached ``.a3m`` files.  Querying the public ColabFold endpoint
    once per ligand kills jobs at scale, so every CLI surface raises
    this error instead of silently falling back.
    """


_MSA_SERVER_DISABLED_HINT = (
    "The --use-msa-server / --use_msa_server flag is DISABLED in this fork.\n"
    "Pre-compute MSAs once per unique sequence with:\n"
    "    python -m boltz.affinity_rescoring.mmseqs2 \\\n"
    "        --sequence <SEQ> --out <cache_dir>/<name>.a3m\n"
    "or use fineturning_experiment/precompute_msas.py for batch precompute.\n"
    "Then either:\n"
    "  * embed the absolute MSA path under each protein chain's 'msa:' key in\n"
    "    your YAML input, OR\n"
    "  * point every consumer at the same cache via the BOLTZ_MSA_CACHE_DIR\n"
    "    environment variable (or --msa-directory where supported)."
)


def raise_msa_server_disabled() -> None:
    """Raise :class:`MSAServerDisabledError` with the canonical hint."""
    raise MSAServerDisabledError(_MSA_SERVER_DISABLED_HINT)


# ── Canonical naming ────────────────────────────────────────────────────────

CANONICAL_EXT = ".a3m"
"""All canonical MSA files written by this fork use the ``.a3m`` extension."""


def sequence_hash(sequence: str) -> str:
    """Return a deterministic 16-char SHA256 prefix for *sequence*.

    The sequence is uppercased and stripped of whitespace first so that
    cosmetic differences (newlines from FASTA parsing, mixed case from
    different YAML producers) all map to the same hash.
    """
    norm = "".join(sequence.split()).upper()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def canonical_msa_filename(sequence: str) -> str:
    """Return ``<sha256[:16]>.a3m`` for *sequence*."""
    return f"{sequence_hash(sequence)}{CANONICAL_EXT}"


def canonical_msa_path(sequence: str, cache_dir: str | os.PathLike[str]) -> Path:
    """Return ``<cache_dir>/<sequence_hash>.a3m``."""
    return Path(cache_dir) / canonical_msa_filename(sequence)


# ── Discovery ───────────────────────────────────────────────────────────────

def default_cache_dirs() -> list[Path]:
    """Return the cache directories implied by ``BOLTZ_MSA_CACHE_DIR``.

    The environment variable is a path-list (``os.pathsep``-separated)
    so multiple shared caches (e.g. a project cache plus a personal
    scratch cache) can be searched in priority order.
    Empty entries are skipped.
    """
    raw = os.environ.get("BOLTZ_MSA_CACHE_DIR", "")
    out: list[Path] = []
    for chunk in raw.split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            out.append(Path(chunk).expanduser())
    return out


def _candidate_names(
    sequence: Optional[str],
    chain_id: Optional[str],
    target: Optional[str],
) -> list[str]:
    """Return filename candidates to probe inside each cache directory.

    Order matters — the canonical hash-based name is preferred so that
    sequence-level dedup wins over legacy per-chain naming.  Legacy
    names are kept for backwards-compatibility with caches produced by
    earlier versions of this fork.
    """
    names: list[str] = []
    if sequence:
        names.append(canonical_msa_filename(sequence))
        # Also accept the .csv variant of the canonical name for
        # parity with the internal boltz `compute_msa` writer.
        names.append(f"{sequence_hash(sequence)}.csv")
    if target and chain_id:
        names.append(f"{target}_{chain_id}.a3m")
        names.append(f"{target}_{chain_id}.csv")
    if chain_id:
        names.append(f"{chain_id}.a3m")
        names.append(f"{chain_id}.csv")
    if target:
        names.append(f"{target}.a3m")
        names.append(f"{target}.csv")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def find_msa(
    sequence: Optional[str] = None,
    *,
    msa_dirs: Optional[Sequence[str | os.PathLike[str]]] = None,
    chain_id: Optional[str] = None,
    target: Optional[str] = None,
    allow_single_file_fallback: bool = False,
) -> Optional[Path]:
    """Locate a pre-computed MSA file for *sequence* (or *chain_id*).

    Parameters
    ----------
    sequence
        Amino-acid sequence.  When provided, the canonical
        ``<sha256[:16]>.a3m`` name is probed first.
    msa_dirs
        Directories to search, highest priority first.  If ``None`` (or
        empty), :func:`default_cache_dirs` is used.
    chain_id, target
        Optional metadata used to probe legacy filename patterns
        (``<chain_id>.a3m``, ``<target>_<chain_id>.a3m``, …).
    allow_single_file_fallback
        If True and exactly one ``*.a3m`` (or ``*.csv``) lives in a
        given directory, accept it.  This preserves the behaviour of
        the original rescorer for single-chain proteins where the
        directory contains only one MSA.

    Returns
    -------
    Path or None
        Absolute path to the first matching MSA, or ``None`` if no
        candidate exists.
    """
    dirs: list[Path] = []
    if msa_dirs:
        dirs.extend(Path(d) for d in msa_dirs if d is not None)
    dirs.extend(default_cache_dirs())
    if not dirs:
        return None

    candidates = _candidate_names(sequence, chain_id, target)
    norm_seq = (
        "".join(sequence.split()).upper() if sequence else None
    )

    for d in dirs:
        if not d.is_dir():
            continue
        for name in candidates:
            p = d / name
            if p.exists() and p.stat().st_size > 0:
                return p.resolve()
        # Sequence-content fallback: scan every .a3m/.csv in the dir and
        # accept the first one whose query (= first non-header row) matches
        # the requested sequence.  This rescues caches written with legacy
        # naming when the caller does not know the original ``target``.
        if norm_seq is not None:
            for ext in ("*.a3m", "*.csv"):
                for hit in sorted(d.glob(ext)):
                    if hit.stat().st_size == 0:
                        continue
                    if _file_query_matches(hit, norm_seq):
                        return hit.resolve()
        if allow_single_file_fallback:
            for ext in ("*.a3m", "*.csv"):
                hits = sorted(d.glob(ext))
                if len(hits) == 1 and hits[0].stat().st_size > 0:
                    return hits[0].resolve()
    return None


def _file_query_matches(path: Path, norm_seq: str) -> bool:
    """Return True if the first sequence in *path* equals *norm_seq*.

    Supports both ``.a3m`` (FASTA-style, first ``>`` block) and the
    Boltz ``.csv`` MSA format whose first data row holds the query
    sequence in column 0.  Comparison is case-insensitive and ignores
    whitespace and a3m insertion markers (lowercase letters / ``-``
    / ``.``).
    """
    try:
        with path.open("r") as fh:
            if path.suffix == ".csv":
                # Skip header row, read first data row, take column 0.
                header = fh.readline()
                if not header:
                    return False
                first = fh.readline().strip()
                if not first:
                    return False
                query = first.split(",")[0]
            else:
                # FASTA / a3m: skip header lines, accumulate the first
                # sequence block.
                query_parts: list[str] = []
                in_query = False
                for line in fh:
                    line = line.rstrip("\n")
                    if line.startswith(">"):
                        if in_query:
                            break
                        in_query = True
                        continue
                    if in_query:
                        query_parts.append(line)
                query = "".join(query_parts)
    except OSError:
        return False

    # Strip a3m insertion markers (lowercase = insertion relative to query,
    # '-' / '.' = gap) and uppercase the rest.
    cleaned = "".join(
        c for c in query if c.isalpha() and not c.islower()
    ).upper()
    if not cleaned:
        # Fall back to a tolerant comparison if the query line uses
        # gap-only or insertion-only encoding.
        cleaned = "".join(c for c in query if c.isalpha()).upper()
    return cleaned == norm_seq


def write_legacy_symlinks(
    canonical: Path,
    *,
    chain_id: Optional[str] = None,
    target: Optional[str] = None,
) -> list[Path]:
    """Create symlinks next to *canonical* for legacy naming patterns.

    This lets producers (e.g. ``precompute_msas.py``) write a single
    canonical hash-named ``.a3m`` while still satisfying older code
    paths that look for ``<target>_<chain_id>.a3m`` or
    ``<chain_id>.a3m``.

    Existing files / symlinks at the legacy paths are left untouched
    so this is safe to call repeatedly.  Returns the list of symlinks
    that were actually created.
    """
    canonical = Path(canonical)
    if not canonical.exists():
        return []

    legacy_names: list[str] = []
    if target and chain_id:
        legacy_names.append(f"{target}_{chain_id}.a3m")
    if chain_id:
        legacy_names.append(f"{chain_id}.a3m")

    created: list[Path] = []
    for name in legacy_names:
        link = canonical.parent / name
        if link.exists() or link.is_symlink():
            continue
        try:
            link.symlink_to(canonical.name)
            created.append(link)
        except OSError as exc:  # filesystem without symlink support
            logger.debug("Could not create legacy symlink %s -> %s: %s",
                         link, canonical.name, exc)
    return created


__all__ = [
    "CANONICAL_EXT",
    "MSAServerDisabledError",
    "canonical_msa_filename",
    "canonical_msa_path",
    "default_cache_dirs",
    "find_msa",
    "raise_msa_server_disabled",
    "sequence_hash",
    "write_legacy_symlinks",
]


# ── Convenience: __main__ entry-point for one-off precompute ────────────────

def _main() -> int:
    """Tiny CLI: ``python -m boltz.affinity_rescoring.mmseqs2`` analogue.

    Allows ``python -m boltz.affinity_rescoring.msa_cache --where SEQ``
    so users can quickly check the canonical filename for a sequence.
    """
    import argparse

    p = argparse.ArgumentParser(
        description="Inspect the MSA cache used by Boltz_affinity.",
    )
    p.add_argument("--sequence", "-s",
                   help="Print canonical filename for SEQUENCE and exit.")
    p.add_argument("--find", action="store_true",
                   help="Search the cache for an MSA matching --sequence.")
    p.add_argument("--cache-dir", "-d", action="append", default=[],
                   help="Additional cache directory to search (repeatable).")
    args = p.parse_args()

    if not args.sequence:
        p.error("--sequence is required")

    print(f"hash:      {sequence_hash(args.sequence)}")
    print(f"filename:  {canonical_msa_filename(args.sequence)}")

    if args.find:
        hit = find_msa(args.sequence, msa_dirs=args.cache_dir or None)
        if hit is None:
            print("found:     <none> — run precompute_msa to populate the cache")
            return 1
        print(f"found:     {hit}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
