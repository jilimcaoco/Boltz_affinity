"""On-disk memo of the frozen-trunk output ``z`` for affinity rescoring.

Rescoring a pose runs the full Boltz-2 trunk (input embedder → MSA module →
pairformer, ``recycling_steps + 1`` times) and then the affinity module. The
LoRA/finetune presets used for affinity adaptation (``heads``,
``heads_pairformer``) only touch modules *inside* ``affinity_module`` — the
``affinity_heads.*`` MLPs and the affinity module's own ``pairformer_stack.*``,
which is a different module from the trunk's ``pairformer_module``. The trunk
output is therefore identical for the base model and every adapted arm, yet a
grid that scores N arms over the same poses recomputes it N times.

Measured on an A40 (recycling_steps=1): the trunk is ~93% of per-pose cost and
the affinity module ~7%, so reusing ``z`` across a receptor's arms is worth
~5x on an 8-arm receptor.

This is the rescore-time counterpart of ``boltz.lora.train.TrunkFeatureCache``
and follows the same conventions: opt-in via an environment variable (unset →
behaviour byte-identical to before), atomic writes, and an optional LRU size
cap.

    BOLTZ_RESCORE_CACHE_DIR      enable, and store entries here
    BOLTZ_RESCORE_CACHE_MAX_GB   optional LRU budget (unset/0 = unbounded)

Correctness: the cache key includes a fingerprint hashed over *every trunk
parameter*, so a different base checkpoint — or an adapter that (unlike the
presets above) does reach into the trunk — produces a different key and can
never read another model's ``z``. The key also covers a digest of the input
features and the recycling depth.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

Z_CACHE_VERSION = "rescore_z_v1"

# Mirrors boltz.lora.train._TRUNK_MODULE_ATTRS: everything that runs before the
# affinity module. Kept as a local copy so importing this module does not drag
# in the training stack.
_TRUNK_MODULE_ATTRS = (
    "input_embedder", "s_init", "z_init_1", "z_init_2", "rel_pos",
    "token_bonds", "token_bonds_type", "contact_conditioning",
    "s_recycle", "s_norm", "z_recycle", "z_norm",
    "template_module", "msa_module", "pairformer_module",
)


def _unwrap(model: Any, attr: str) -> Any:
    mod = getattr(model, attr, None)
    if mod is None:
        return None
    return getattr(mod, "_orig_mod", mod)


def trunk_fingerprint(model: Any) -> str:
    """SHA-256 over every parameter feeding the pre-affinity trunk.

    Any change to the trunk — a different base checkpoint, or an adapter whose
    target spec reaches outside ``affinity_module`` — changes this digest, so a
    stale or foreign ``z`` can never be read back.
    """
    h = hashlib.sha256()
    for attr in _TRUNK_MODULE_ATTRS:
        mod = _unwrap(model, attr)
        if mod is None or not hasattr(mod, "named_parameters"):
            continue
        for name, p in sorted(mod.named_parameters(), key=lambda kv: kv[0]):
            h.update(f"{attr}.{name}:{tuple(p.shape)}:{p.dtype}".encode())
            h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def feats_digest(feats: dict[str, Any]) -> str:
    """SHA-256 over the feature tensors that determine the trunk output."""
    h = hashlib.sha256()
    for key in sorted(feats):
        if key.startswith("__"):
            continue
        v = feats[key]
        if torch.is_tensor(v):
            h.update(f"{key}:{tuple(v.shape)}:{v.dtype}".encode())
            h.update(v.detach().to("cpu").contiguous().numpy().tobytes())
        elif isinstance(v, (str, int, float, bool)):
            h.update(f"{key}:{v}".encode())
    return h.hexdigest()


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(obj, tmp)
        tmp.replace(path)
    except Exception:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise


class RescoreTrunkCache:
    """Disk-backed store of ``z``, keyed on (trunk weights, feats, recycling)."""

    def __init__(self, root: Path, trunk_sha: str, *,
                 max_bytes: Optional[int] = None) -> None:
        self.root = Path(root).expanduser()
        (self.root / "z").mkdir(parents=True, exist_ok=True)
        self.trunk_sha = trunk_sha
        self.max_bytes = max_bytes if max_bytes and max_bytes > 0 else None
        self.hits = 0
        self.misses = 0

    def key(self, digest: str, recycling_steps: int) -> str:
        raw = f"{Z_CACHE_VERSION}|{self.trunk_sha}|{digest}|rec{recycling_steps}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / "z" / f"{key}.pt"

    def load(self, key: str, device: Any) -> Optional[torch.Tensor]:
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            z = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 — partial/corrupt write → recompute
            self.misses += 1
            return None
        self.hits += 1
        return z.to(device)

    def save(self, key: str, z: torch.Tensor) -> None:
        _atomic_torch_save(z.detach().to("cpu").clone(), self._path(key))
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        if self.max_bytes is None:
            return
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for p in (self.root / "z").glob("*.pt"):
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
            total += st.st_size
        if total <= self.max_bytes:
            return
        entries.sort(key=lambda e: e[0])  # oldest mtime first
        for _mtime, size, p in entries:
            if total <= self.max_bytes:
                break
            try:
                p.unlink()
                total -= size
            except OSError:
                continue


_CACHE: Optional[RescoreTrunkCache] = None
_CACHE_MODEL_ID: Optional[int] = None


def get_cache(model: Any) -> Optional[RescoreTrunkCache]:
    """Return the process-wide cache, building it on first use for ``model``.

    Returns ``None`` when ``$BOLTZ_RESCORE_CACHE_DIR`` is unset, in which case
    callers must run the trunk normally.
    """
    global _CACHE, _CACHE_MODEL_ID
    root = os.environ.get("BOLTZ_RESCORE_CACHE_DIR", "").strip()
    if not root:
        return None
    if _CACHE is not None and _CACHE_MODEL_ID == id(model):
        return _CACHE

    max_gb = os.environ.get("BOLTZ_RESCORE_CACHE_MAX_GB", "").strip()
    max_bytes = int(float(max_gb) * 1e9) if max_gb else None
    sha = trunk_fingerprint(model)
    _CACHE = RescoreTrunkCache(Path(root), sha, max_bytes=max_bytes)
    _CACHE_MODEL_ID = id(model)
    logger.info(
        "Rescore trunk cache ENABLED at %s (trunk=%s..., budget=%s)",
        root, sha[:12], f"{max_bytes/1e9:.0f} GB" if max_bytes else "unbounded",
    )
    return _CACHE


def reset_cache() -> None:
    """Drop the memoised cache handle (used by tests)."""
    global _CACHE, _CACHE_MODEL_ID
    _CACHE = None
    _CACHE_MODEL_ID = None
