"""Adapter metadata: serialisable record of a trained LoRA + its provenance."""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

BOLTZ_LORA_FORMAT_VERSION = 1


@dataclass
class LoRAConfig:
    """User-facing LoRA hyperparameters."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_spec: str = "heads_pairformer"
    target_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoRAConfig":
        return cls(**data)


@dataclass
class TrainingRun:
    """One pass of training contributing to an adapter (initial or update)."""

    started_at: float
    finished_at: Optional[float] = None
    data_manifest: Optional[str] = None
    data_sha256: Optional[str] = None
    data_rows: Optional[int] = None
    mode: str = "rescore"  # "rescore" or "full"
    loss_spec: str = "mse"
    epochs: int = 0
    learning_rate: float = 1e-4
    batch_size: int = 1
    metrics: list[dict[str, float]] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class LoRAAdapter:
    """In-memory record persisted to disk as ``meta.json`` + ``adapter.pt``.

    The actual LoRA tensors live in ``adapter.pt``; this object stores the
    metadata needed for reproducibility, provenance, and active-learning
    workflows.
    """

    name: str
    created_at: float = field(default_factory=time.time)
    boltz_version: Optional[str] = None
    base_checkpoint_sha256: Optional[str] = None
    base_checkpoint_path: Optional[str] = None
    config: LoRAConfig = field(default_factory=LoRAConfig)
    adapted_layers: list[str] = field(default_factory=list)
    history: list[TrainingRun] = field(default_factory=list)
    parent_adapter: Optional[str] = None
    format_version: int = BOLTZ_LORA_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "boltz_version": self.boltz_version,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "base_checkpoint_path": self.base_checkpoint_path,
            "config": self.config.to_dict(),
            "adapted_layers": list(self.adapted_layers),
            "history": [dataclasses.asdict(h) for h in self.history],
            "parent_adapter": self.parent_adapter,
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoRAAdapter":
        return cls(
            name=data["name"],
            created_at=data.get("created_at", time.time()),
            boltz_version=data.get("boltz_version"),
            base_checkpoint_sha256=data.get("base_checkpoint_sha256"),
            base_checkpoint_path=data.get("base_checkpoint_path"),
            config=LoRAConfig.from_dict(data.get("config", {})),
            adapted_layers=list(data.get("adapted_layers", [])),
            history=[TrainingRun(**h) for h in data.get("history", [])],
            parent_adapter=data.get("parent_adapter"),
            format_version=data.get("format_version", BOLTZ_LORA_FORMAT_VERSION),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


__all__ = ["BOLTZ_LORA_FORMAT_VERSION", "LoRAAdapter", "LoRAConfig", "TrainingRun"]
