"""Metadata records for a full fine-tune of the affinity module."""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# Re-use the same TrainingRun dataclass so that downstream tooling that
# already understands LoRA history files (notebooks, dashboards) can read
# fine-tune histories without modification.
from boltz.lora.adapter import TrainingRun  # noqa: F401  (re-exported)

BOLTZ_FINETUNE_FORMAT_VERSION = 1


@dataclass
class FinetuneConfig:
    """User-facing fine-tuning hyperparameters.

    ``target_spec`` selects which sub-tree of the affinity model is
    trainable. Built-in presets (see :mod:`boltz.finetune.targets`):

    * ``affinity_module`` *(default)* — every parameter under
      ``affinity_module*``. Largest, highest capacity, requires the most
      data to avoid catastrophic forgetting of the upstream signal.
    * ``affinity_heads`` — only the heads MLPs (mirror of the LoRA
      ``heads`` preset).
    * ``affinity_pairformer`` — only the affinity-side pairformer linears.
    * ``heads_pairformer`` — heads + pairformer (mirror of the LoRA
      ``heads_pairformer`` preset).

    ``target_patterns`` is the resolved regex tuple actually used; saved so
    we can reproduce the parameter selection at load time without depending
    on preset names that may evolve.

    ``l2_sp_weight`` records the L2-SP regularization strength used for
    this fine-tune (0.0 = disabled, the default and today's behavior). See
    :mod:`boltz.finetune.l2_sp`.
    """

    target_spec: str = "affinity_module"
    target_patterns: list[str] = field(default_factory=list)
    num_trainable_params: int = 0
    l2_sp_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinetuneConfig":
        return cls(**data)


@dataclass
class FinetuneRecord:
    """In-memory record persisted to disk as ``meta.json`` + ``weights.pt``.

    Symmetric to :class:`boltz.lora.LoRAAdapter` but holds *full* (not
    low-rank) parameter deltas.  ``weights.pt`` stores a partial state-dict
    keyed by fully-qualified parameter names in the Boltz2 model so it can
    be loaded with ``model.load_state_dict(..., strict=False)``.
    """

    name: str
    created_at: float = field(default_factory=time.time)
    boltz_version: Optional[str] = None
    base_checkpoint_sha256: Optional[str] = None
    base_checkpoint_path: Optional[str] = None
    config: FinetuneConfig = field(default_factory=FinetuneConfig)
    trained_params: list[str] = field(default_factory=list)
    history: list[TrainingRun] = field(default_factory=list)
    parent: Optional[str] = None
    format_version: int = BOLTZ_FINETUNE_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "boltz_version": self.boltz_version,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "base_checkpoint_path": self.base_checkpoint_path,
            "config": self.config.to_dict(),
            "trained_params": list(self.trained_params),
            "history": [dataclasses.asdict(h) for h in self.history],
            "parent": self.parent,
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinetuneRecord":
        return cls(
            name=data["name"],
            created_at=data.get("created_at", time.time()),
            boltz_version=data.get("boltz_version"),
            base_checkpoint_sha256=data.get("base_checkpoint_sha256"),
            base_checkpoint_path=data.get("base_checkpoint_path"),
            config=FinetuneConfig.from_dict(data.get("config", {})),
            trained_params=list(data.get("trained_params", [])),
            history=[TrainingRun(**h) for h in data.get("history", [])],
            parent=data.get("parent"),
            format_version=data.get(
                "format_version", BOLTZ_FINETUNE_FORMAT_VERSION
            ),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


__all__ = [
    "BOLTZ_FINETUNE_FORMAT_VERSION",
    "FinetuneConfig",
    "FinetuneRecord",
    "TrainingRun",
]
