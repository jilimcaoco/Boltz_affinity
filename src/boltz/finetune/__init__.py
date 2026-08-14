"""Full fine-tuning of the Boltz affinity module weights.

This package is the *non-LoRA* counterpart to :mod:`boltz.lora`. Where LoRA
trains a low-rank residual on top of frozen base weights, this module
fine-tunes the affinity-side parameters of the Boltz2 checkpoint directly
(by default ``affinity_module``/``affinity_module1``/``affinity_module2``).
It exists for two reasons:

1. It is a useful **product feature** — users with sufficient labelled data
   may prefer full fine-tuning over a LoRA adapter.
2. It is the natural **baseline** to benchmark LoRA against.

The on-disk surface and CLI ergonomics intentionally mirror :mod:`boltz.lora`
so that comparing the two is as drop-in as flipping ``--use-lora`` for
``--use-finetune``.

Public surface
--------------
* :class:`FinetuneRecord`, :class:`FinetuneConfig` — adapter metadata.
* :class:`FinetuneRegistry`, :func:`default_registry` — on-disk registry.
* :func:`load_finetune_into_model` — inference-time application.
* :func:`train_finetune` — the training entry-point used by the CLI.
"""

from boltz.finetune.adapter import (
    FinetuneConfig,
    FinetuneRecord,
    TrainingRun,
)
from boltz.finetune.apply import load_finetune_into_model
from boltz.finetune.registry import FinetuneRegistry, default_registry

__all__ = [
    "FinetuneConfig",
    "FinetuneRecord",
    "FinetuneRegistry",
    "TrainingRun",
    "default_registry",
    "load_finetune_into_model",
]
