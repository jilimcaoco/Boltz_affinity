"""On-disk registry of trained LoRA adapters.

Layout under ``$BOLTZ_LORA_DIR`` (default ``~/.boltz/loras``):

    <root>/
      registry.json              # index: {name: {path, created_at, ...}}
      <name>/
        meta.json                # serialised LoRAAdapter
        adapter.pt               # torch.save of {"state_dict": {...}, "config": {...}}
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import torch

from boltz.lora.adapter import LoRAAdapter

ENV_VAR = "BOLTZ_LORA_DIR"
DEFAULT_DIR = "~/.boltz/loras"


def _default_root() -> Path:
    raw = os.environ.get(ENV_VAR, DEFAULT_DIR)
    return Path(raw).expanduser().resolve()


def hash_file(path: str | os.PathLike, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of ``path``."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            sha.update(buf)
    return sha.hexdigest()


class LoRARegistry:
    """File-system backed registry of LoRA adapters."""

    def __init__(self, root: Optional[str | os.PathLike] = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else _default_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        if not self.index_path.exists():
            self._write_index({})

    # -- index helpers ---------------------------------------------------
    def _read_index(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.index_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_index(self, data: dict[str, dict[str, Any]]) -> None:
        # Use a process- and time-unique tmp file so concurrent writers (e.g.
        # parallel SLURM jobs sharing the same BOLTZ_LORA_DIR) don't race on
        # the same ``registry.json.tmp`` path and crash with FileNotFoundError
        # when one process's os.replace consumes another's tmp file.
        tmp = self.index_path.with_suffix(
            f".json.tmp.{os.getpid()}.{time.time_ns()}"
        )
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        try:
            tmp.replace(self.index_path)
        except FileNotFoundError:
            # Another writer beat us to it and already moved an equivalent
            # bootstrap file into place. Safe to ignore.
            if not self.index_path.exists():
                raise

    # -- path helpers ----------------------------------------------------
    def adapter_dir(self, name: str) -> Path:
        return self.root / name

    def meta_path(self, name: str) -> Path:
        return self.adapter_dir(name) / "meta.json"

    def tensor_path(self, name: str) -> Path:
        return self.adapter_dir(name) / "adapter.pt"

    # -- public API ------------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        """Return a list of summary dicts, one per registered adapter."""
        idx = self._read_index()
        return [{"name": n, **info} for n, info in sorted(idx.items())]

    def exists(self, name: str) -> bool:
        return self.meta_path(name).exists() and self.tensor_path(name).exists()

    def resolve(self, name_or_path: str) -> Path:
        """Accept an adapter name (registry lookup) or a path to a directory."""
        candidate = Path(name_or_path).expanduser()
        if candidate.is_dir() and (candidate / "meta.json").exists():
            return candidate.resolve()
        if self.exists(name_or_path):
            return self.adapter_dir(name_or_path)
        msg = (
            f"Adapter '{name_or_path}' not found in registry ({self.root}) "
            "and is not a valid adapter directory."
        )
        raise FileNotFoundError(msg)

    def load(self, name_or_path: str) -> tuple[LoRAAdapter, dict[str, torch.Tensor]]:
        """Return (adapter metadata, lora state_dict) for ``name_or_path``."""
        path = self.resolve(name_or_path)
        meta = LoRAAdapter.from_dict(json.loads((path / "meta.json").read_text()))
        blob = torch.load(path / "adapter.pt", map_location="cpu", weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        return meta, state

    def save(
        self,
        adapter: LoRAAdapter,
        state_dict: dict[str, torch.Tensor],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist ``adapter`` + tensors under ``<root>/<adapter.name>/``."""
        target = self.adapter_dir(adapter.name)
        if target.exists() and not overwrite:
            msg = f"Adapter '{adapter.name}' already exists at {target}. Use overwrite=True or `lora update`."
            raise FileExistsError(msg)
        target.mkdir(parents=True, exist_ok=True)

        tensor_blob = {
            "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
            "config": adapter.config.to_dict(),
            "format_version": adapter.format_version,
        }
        torch.save(tensor_blob, target / "adapter.pt")
        (target / "meta.json").write_text(adapter.to_json())

        idx = self._read_index()
        idx[adapter.name] = {
            "path": str(target),
            "created_at": adapter.created_at,
            "updated_at": time.time(),
            "parent_adapter": adapter.parent_adapter,
            "rank": adapter.config.rank,
            "target_spec": adapter.config.target_spec,
            "num_adapted_layers": len(adapter.adapted_layers),
        }
        self._write_index(idx)
        return target

    def update_meta(self, adapter: LoRAAdapter) -> None:
        """Rewrite ``meta.json`` for an existing adapter (e.g. after a new run)."""
        target = self.adapter_dir(adapter.name)
        if not target.exists():
            msg = f"Adapter '{adapter.name}' does not exist; cannot update meta."
            raise FileNotFoundError(msg)
        (target / "meta.json").write_text(adapter.to_json())
        idx = self._read_index()
        if adapter.name in idx:
            idx[adapter.name]["updated_at"] = time.time()
            self._write_index(idx)

    def delete(self, name: str) -> None:
        target = self.adapter_dir(name)
        if target.exists():
            shutil.rmtree(target)
        idx = self._read_index()
        idx.pop(name, None)
        self._write_index(idx)


def default_registry() -> LoRARegistry:
    """Return a registry rooted at ``$BOLTZ_LORA_DIR`` (or ``~/.boltz/loras``)."""
    return LoRARegistry()


__all__ = ["DEFAULT_DIR", "ENV_VAR", "LoRARegistry", "default_registry", "hash_file"]
