#!/usr/bin/env python3
"""Task 4 — disk-backed trunk cache, and Task 1 donor-matching helpers for
the resample/mean ablation operators.

Why this exists
----------------
``affinity_forward()`` (src/boltz/affinity_rescoring/inference.py) runs the
full recycled trunk (MSA + Pairformer, ~10s) and then the affinity head
(~10ms) in one call. Before this module, ``run_feature_ablation.py`` called
``affinity_forward`` once *per experiment*, so every one of the ~14+
zero-ablation experiments (and now, many more resample/mean variants) reran
the expensive trunk from scratch on identical input. The trunk/head split
already exists as ``affinity_trunk_forward`` / ``affinity_head_forward`` —
this module is what actually exploits it: compute the trunk once per
(receptor, ligand), cache it to disk, and replay only the cheap head for
every experiment.

The disk cache (rather than an in-memory dict scoped to one script process)
is what makes the resample/mean operators possible at all: they need another
ligand's trunk output as a donor, and that ligand may not have been reached
yet in the query's iteration order. ``run_feature_ablation.py`` now runs two
passes per receptor — pass 1 populates the cache for every ligand, pass 2
runs every experiment (including resample/mean) against it.

Cache format
------------
One file per (receptor, ligand) at
``<cache_dir>/<receptor_id>/<ligand_id>.pt``, containing a dict with:
  - ``z``               : (N, N, token_z) float16, trunk pair representation
  - ``s_inputs``         : (N, token_s) float32, *unablated* baseline s_inputs
  - ``token_repr_pos``   : (N, 3) float32, per-token representative-atom
                           coordinates (``token_to_rep_atom @ coords``,
                           precomputed so we never need to keep full atom
                           coordinates or the token_to_rep_atom matrix around)
  - ``n_tokens``         : int
  - ``use_kernels``      : bool
  - ``meta``             : dict (git SHA, boltz version, input hash, etc.)

Only the *unablated* trunk/s_inputs/geometry are ever cached — all zero/
resample/mean substitution happens at replay time in
``run_feature_ablation.py``, never inside the cache itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── cache I/O ────────────────────────────────────────────────────────────────

def cache_entry_path(cache_dir: Path, receptor_id: str, ligand_id: str) -> Path:
    return cache_dir / receptor_id / f"{ligand_id}.pt"


def save_cache_entry(
    cache_dir: Path,
    receptor_id: str,
    ligand_id: str,
    *,
    z,
    s_inputs,
    token_repr_pos,
    use_kernels: bool,
    meta: Optional[dict] = None,
) -> Path:
    import torch

    out_path = cache_entry_path(cache_dir, receptor_id, ligand_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z0 = z[0] if z.dim() == 4 else z          # (N, N, C)
    s0 = s_inputs[0] if s_inputs.dim() == 3 else s_inputs  # (N, C)
    # Callers build this as ``token_to_rep_atom[0] @ coords[0]``, where coords
    # still carries a multiplicity axis -- so it arrives as (1, N, 3). Storing
    # it that way makes crop_or_pad_tokens pad the multiplicity axis instead of
    # the token axis.
    p0 = token_repr_pos[0] if token_repr_pos.dim() == 3 else token_repr_pos  # (N, 3)
    n_tokens = int(s0.shape[0])
    if int(p0.shape[0]) != n_tokens:
        raise ValueError(
            f"token_repr_pos has {p0.shape[0]} tokens but s_inputs has {n_tokens} "
            f"({receptor_id}/{ligand_id}); refusing to cache a misaligned entry."
        )

    payload = {
        "z": z0.detach().to("cpu", dtype=torch.float16).contiguous(),
        "s_inputs": s0.detach().to("cpu", dtype=torch.float32).contiguous(),
        "token_repr_pos": p0.detach().to("cpu", dtype=torch.float32).contiguous(),
        "n_tokens": n_tokens,
        "use_kernels": bool(use_kernels),
        "meta": meta or {},
    }
    torch.save(payload, out_path)
    update_token_index(cache_dir, receptor_id, ligand_id, n_tokens)
    return out_path


def load_cache_entry(cache_dir: Path, receptor_id: str, ligand_id: str) -> Optional[dict]:
    import torch

    path = cache_entry_path(cache_dir, receptor_id, ligand_id)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def token_index_path(cache_dir: Path, receptor_id: str) -> Path:
    return cache_dir / receptor_id / "_token_index.json"


def update_token_index(cache_dir: Path, receptor_id: str, ligand_id: str, n_tokens: int) -> None:
    """Record ligand -> n_tokens in a small sidecar index.

    Donor matching only needs the token count, but an entry is ~16 MB (z is
    N*N*token_z). Without this index, ``find_donor`` opens every entry in the
    pool just to read one integer -- an O(queries x pool) full-payload read
    that dwarfs every other cost in the run.
    """
    idx_path = token_index_path(cache_dir, receptor_id)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    index = {}
    if idx_path.exists():
        try:
            index = json.loads(idx_path.read_text())
        except Exception:
            index = {}
    index[ligand_id] = int(n_tokens)
    idx_path.write_text(json.dumps(index))


def load_token_index(cache_dir: Path, receptor_id: str) -> Dict[str, int]:
    """{ligand_id: n_tokens}. Falls back to reading payloads (slow) only for
    entries the index doesn't cover, so an index built by an older run still
    works."""
    idx_path = token_index_path(cache_dir, receptor_id)
    index: Dict[str, int] = {}
    if idx_path.exists():
        try:
            index = {k: int(v) for k, v in json.loads(idx_path.read_text()).items()}
        except Exception:
            index = {}

    missing = [l for l in list_cached_ligands(cache_dir, receptor_id) if l not in index]
    for lid in missing:
        entry = load_cache_entry(cache_dir, receptor_id, lid)
        if entry is not None:
            index[lid] = int(entry["n_tokens"])
            update_token_index(cache_dir, receptor_id, lid, index[lid])
    return index


def list_cached_ligands(cache_dir: Path, receptor_id: str) -> List[str]:
    rec_dir = cache_dir / receptor_id
    if not rec_dir.exists():
        return []
    return sorted(p.stem for p in rec_dir.glob("*.pt"))


# ── crop / pad along the token axis ─────────────────────────────────────────

def crop_or_pad_tokens(tensor, target_n: int, token_dims: Tuple[int, ...]):
    """Crop or zero-pad ``tensor`` along each axis in ``token_dims`` so that
    axis ends up with length ``target_n``. Never reshapes/interpolates — a
    donor tensor with fewer tokens than the query is zero-padded (matching
    this codebase's existing zero-ablation convention for "missing" signal);
    a donor with more tokens is truncated. Returns (tensor, delta) where
    ``delta = target_n - original_n`` (positive = padded, negative = cropped,
    zero = exact match) so callers can log what happened.
    """
    import torch

    dim0 = token_dims[0]
    original_n = tensor.shape[dim0]
    delta = target_n - original_n

    if delta == 0:
        return tensor, 0

    if delta < 0:
        # crop (truncate) each token axis to target_n
        slices = [slice(None)] * tensor.dim()
        for d in token_dims:
            slices[d] = slice(0, target_n)
        return tensor[tuple(slices)].contiguous(), delta

    # pad each token axis with zeros up to target_n
    pad_amt = delta
    out = tensor
    for d in token_dims:
        pad_shape = list(out.shape)
        pad_shape[d] = pad_amt
        zeros = torch.zeros(pad_shape, dtype=out.dtype, device=out.device)
        out = torch.cat([out, zeros], dim=d)
    return out.contiguous(), delta


# ── donor matching ──────────────────────────────────────────────────────────

def donor_rng(donor_seed: Optional[int], query_ligand_id: str):
    """RNG for one query's donor draw, seeded on ``(donor_seed, ligand)``.

    Seeding on ``donor_seed`` alone gives every query an identical RNG state,
    so ``find_donor`` returns the same rank within each token-count bucket and
    a whole receptor collapses onto ~1 donor per seed -- no donor-draw variance
    in the data, and donor identity correlated with the query's token count
    (hence its size/MW). Mixing the ligand id in restores independent draws
    while staying reproducible; ``hash()`` cannot be used because it is salted
    per process.
    """
    import numpy as np

    digest = hashlib.blake2b(query_ligand_id.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng([int(donor_seed or 0), int.from_bytes(digest, "big")])


@dataclass
class DonorMatch:
    donor_ligand_id: str
    match_kind: str        # "exact" | "nearest" | "none"
    donor_n_tokens: int
    query_n_tokens: int
    token_delta: int        # query_n_tokens - donor_n_tokens


def find_donor(
    cache_dir: Path,
    receptor_id: str,
    query_ligand_id: str,
    query_n_tokens: int,
    rng,
) -> Optional[DonorMatch]:
    """Pick one donor ligand for ``query_ligand_id`` on ``receptor_id``,
    excluding the query itself. Fallback order: exact token-count match
    (chosen uniformly at random among ties via ``rng``), else nearest token
    count, else None (caller must skip and log the reason)."""
    candidates = [lid for lid in list_cached_ligands(cache_dir, receptor_id) if lid != query_ligand_id]
    if not candidates:
        return None

    entries_n = {}
    for lid in candidates:
        entry = load_cache_entry(cache_dir, receptor_id, lid)
        if entry is None:
            continue
        entries_n[lid] = entry["n_tokens"]
    if not entries_n:
        return None

    exact = [lid for lid, n in entries_n.items() if n == query_n_tokens]
    if exact:
        chosen = exact[int(rng.integers(0, len(exact)))]
        return DonorMatch(chosen, "exact", entries_n[chosen], query_n_tokens, 0)

    nearest_lid = min(entries_n, key=lambda lid: abs(entries_n[lid] - query_n_tokens))
    n = entries_n[nearest_lid]
    return DonorMatch(nearest_lid, "nearest", n, query_n_tokens, query_n_tokens - n)


def all_other_ligands(cache_dir: Path, receptor_id: str, query_ligand_id: str) -> List[str]:
    return [lid for lid in list_cached_ligands(cache_dir, receptor_id) if lid != query_ligand_id]


# ── donor pool sizing (storage) ─────────────────────────────────────────────

DEFAULT_DONOR_POOL_SIZE = 64

def bytes_per_entry(n_tokens: int = 256, token_z: int = 128, token_s: int = 384) -> int:
    """Approximate on-disk size of one cache entry.

    ``z`` dominates: it is (N, N, token_z) float16, so it grows with the
    *square* of the token count -- ~16 MB at the cropper's 256-token limit.
    That is why the cache must never hold every ligand (see
    ``select_donor_pool``).
    """
    return n_tokens * n_tokens * token_z * 2 + n_tokens * token_s * 4 + n_tokens * 3 * 4


def cache_size_bytes(cache_dir: Path, receptor_id: Optional[str] = None) -> int:
    """Actual bytes on disk under the cache (optionally one receptor)."""
    root = cache_dir / receptor_id if receptor_id else cache_dir
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*.pt") if p.is_file())


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def select_donor_pool(
    candidate_ids: Sequence[str],
    pool_size: int = DEFAULT_DONOR_POOL_SIZE,
    rng=None,
    strata: Optional[Dict[str, object]] = None,
) -> List[str]:
    """Choose a bounded, representative subset of ligands to cache as donors.

    The cache exists solely to supply *donor* channels for the
    ``resample``/``mean`` operators -- nothing needs every ligand's trunk on
    disk. Caching all of them costs ~16 MB each (see ``bytes_per_entry``),
    which is hundreds of GB per receptor at DUD-E/DUDEZ scale; a bounded
    pool costs ~1 GB and is statistically ample, since ``resample`` draws
    only 5 donors per query and ``mean`` is a sample mean either way.

    Selection is random (seeded via ``rng``) rather than "first N", because
    candidates usually arrive sorted and the head of the list can be all
    actives -- which would make every donor an active. When ``strata`` maps
    ligand id -> group (e.g. is_binder), the pool is drawn proportionally
    from each group so it stays representative.

    ``pool_size <= 0`` means "no limit" (cache everything) -- only sensible
    for small receptors.
    """
    ids = list(candidate_ids)
    if pool_size is None or pool_size <= 0 or len(ids) <= pool_size:
        return ids

    if rng is None:
        import numpy as _np
        rng = _np.random.default_rng(0)

    if not strata:
        idx = rng.choice(len(ids), size=pool_size, replace=False)
        return [ids[i] for i in sorted(idx)]

    groups: Dict[object, List[str]] = {}
    for lid in ids:
        groups.setdefault(strata.get(lid), []).append(lid)

    chosen: List[str] = []
    total = len(ids)
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        # Proportional allocation, but never zero for a non-empty group --
        # a donor pool with no decoys in it would be unrepresentative.
        want = max(1, round(pool_size * len(members) / total))
        want = min(want, len(members))
        idx = rng.choice(len(members), size=want, replace=False)
        chosen.extend(members[i] for i in idx)

    if len(chosen) > pool_size:
        idx = rng.choice(len(chosen), size=pool_size, replace=False)
        chosen = [chosen[i] for i in sorted(idx)]
    return sorted(chosen)


# ── in-memory donor pool ────────────────────────────────────────────────────

class DonorPool:
    """Donor entries held in memory for one receptor, with a memoized mean.

    Replaces the naive per-query disk access pattern, which was the single
    dominant cost of a resample/mean run:

      * ``find_donor`` opened every pool entry (~16 MB each) just to read
        ``n_tokens`` -- now served from the token index.
      * ``mean_channel`` reloaded the entire pool for *every query ligand*,
        recomputing an average that barely changes between queries.

    Both were O(n_queries x pool_size) full-payload reads. Here the pool is
    read once (pool_size x 16 MB, ~1 GB at the default 64) and the mean is
    computed once per (channel, token_count).

    Leave-one-out: when the query is itself in the pool it must not donate to
    its own mean. Rather than recomputing, the memoized sum is adjusted --
    ``(S - x_q) / (n - 1)`` -- which is exact and O(1). Most queries are not
    in the pool (it is a small subset), so this is rare.
    """

    def __init__(self, cache_dir: Path, receptor_id: str, ligand_ids: Optional[Sequence[str]] = None):
        self.cache_dir = cache_dir
        self.receptor_id = receptor_id
        self.token_index = load_token_index(cache_dir, receptor_id)
        ids = list(ligand_ids) if ligand_ids is not None else list_cached_ligands(cache_dir, receptor_id)
        self.entries: Dict[str, dict] = {}
        for lid in ids:
            entry = load_cache_entry(cache_dir, receptor_id, lid)
            if entry is not None:
                self.entries[lid] = entry
                self.token_index.setdefault(lid, int(entry["n_tokens"]))
        self._sum_cache: Dict[Tuple[str, int], object] = {}

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def ligand_ids(self) -> List[str]:
        return sorted(self.entries)

    def find_donor(self, query_ligand_id: str, query_n_tokens: int, rng) -> Optional[DonorMatch]:
        """Same contract as the module-level ``find_donor``, but served from
        memory: exact token match preferred (uniformly at random among ties),
        else nearest, else None."""
        candidates = {lid: n for lid, n in self.token_index.items()
                      if lid != query_ligand_id and lid in self.entries}
        if not candidates:
            return None
        exact = [lid for lid, n in candidates.items() if n == query_n_tokens]
        if exact:
            chosen = sorted(exact)[int(rng.integers(0, len(exact)))]
            return DonorMatch(chosen, "exact", candidates[chosen], query_n_tokens, 0)
        nearest = min(sorted(candidates), key=lambda lid: abs(candidates[lid] - query_n_tokens))
        n = candidates[nearest]
        return DonorMatch(nearest, "nearest", n, query_n_tokens, query_n_tokens - n)

    def entry(self, ligand_id: str) -> Optional[dict]:
        return self.entries.get(ligand_id)

    def _pool_sum(self, channel: str, n_tokens: int):
        """Sum over all pool members of `channel`, matched to n_tokens."""
        import torch

        key = (channel, int(n_tokens))
        if key not in self._sum_cache:
            total = None
            for lid in self.ligand_ids:
                t, _ = substitute_channel(channel, n_tokens, self.entries[lid])
                total = t.clone().float() if total is None else total + t.float()
            self._sum_cache[key] = total
        return self._sum_cache[key]

    def mean_channel(self, channel: str, n_tokens: int, exclude: Optional[str] = None):
        """Per-position mean of `channel` over the pool, matched to
        ``n_tokens``, optionally leaving out one member."""
        if not self.entries:
            raise ValueError("DonorPool is empty; cannot compute a mean")
        total = self._pool_sum(channel, n_tokens)
        n = len(self.entries)
        if exclude is not None and exclude in self.entries:
            if n < 2:
                raise ValueError(
                    "DonorPool has only the query itself; no donors available for mean"
                )
            excl, _ = substitute_channel(channel, n_tokens, self.entries[exclude])
            return (total - excl.float()) / (n - 1)
        return total / n


# ── channel substitution ─────────────────────────────────────────────────────

def substitute_channel(
    channel: str,
    query_n_tokens: int,
    donor_entry: dict,
):
    """Return the donor tensor for ``channel`` ("z_trunk" | "s_inputs" |
    "distogram"), cropped/padded to ``query_n_tokens``, plus the token delta.
    """
    if channel == "z_trunk":
        return crop_or_pad_tokens(donor_entry["z"], query_n_tokens, token_dims=(0, 1))
    if channel == "s_inputs":
        return crop_or_pad_tokens(donor_entry["s_inputs"], query_n_tokens, token_dims=(0,))
    if channel == "distogram":
        return crop_or_pad_tokens(donor_entry["token_repr_pos"], query_n_tokens, token_dims=(0,))
    raise ValueError(f"Unknown channel {channel!r}")


def mean_channel(
    channel: str,
    query_n_tokens: int,
    donor_entries: List[dict],
):
    """Per-position mean over ``donor_entries`` for ``channel``, each
    individually cropped/padded to ``query_n_tokens`` before averaging, so
    the mean is well-defined at the query's own token count. Requires
    donor_entries to be non-empty."""
    import torch

    if not donor_entries:
        raise ValueError("mean_channel requires at least one donor entry")

    matched = [substitute_channel(channel, query_n_tokens, e)[0] for e in donor_entries]
    stacked = torch.stack(matched, dim=0)
    return stacked.mean(dim=0)


def run_metadata(recipe_extra: Optional[dict] = None) -> dict:
    """Reproducibility sidecar fields: git SHA, boltz version, and whatever
    the caller adds (config hash, seeds, input file hashes)."""
    import subprocess

    meta = {}
    try:
        meta["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        meta["git_sha"] = None
    try:
        import boltz
        meta["boltz_version"] = getattr(boltz, "__version__", None)
    except Exception:
        meta["boltz_version"] = None
    if recipe_extra:
        meta.update(recipe_extra)
    return meta


def write_json_sidecar(output_csv: Path, meta: dict) -> Path:
    sidecar_path = output_csv.with_suffix(output_csv.suffix + ".meta.json")
    with open(sidecar_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return sidecar_path
