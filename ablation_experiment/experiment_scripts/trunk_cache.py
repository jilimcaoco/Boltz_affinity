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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    n_tokens = int(s0.shape[0])

    payload = {
        "z": z0.detach().to("cpu", dtype=torch.float16).contiguous(),
        "s_inputs": s0.detach().to("cpu", dtype=torch.float32).contiguous(),
        "token_repr_pos": token_repr_pos.detach().to("cpu", dtype=torch.float32).contiguous(),
        "n_tokens": n_tokens,
        "use_kernels": bool(use_kernels),
        "meta": meta or {},
    }
    torch.save(payload, out_path)
    return out_path


def load_cache_entry(cache_dir: Path, receptor_id: str, ligand_id: str) -> Optional[dict]:
    import torch

    path = cache_entry_path(cache_dir, receptor_id, ligand_id)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


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
