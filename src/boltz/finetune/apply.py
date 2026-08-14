"""Apply a saved fine-tune to a loaded Boltz2 model for inference."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from boltz.finetune.adapter import FinetuneRecord
from boltz.finetune.registry import FinetuneRegistry, default_registry

logger = logging.getLogger(__name__)


def load_finetune_into_model(
    model: Any,
    name_or_path: str,
    *,
    registry: Optional[FinetuneRegistry] = None,
    strict: bool = True,
) -> FinetuneRecord:
    """Copy fine-tuned weights into ``model`` in-place.

    Parameters
    ----------
    model : nn.Module
        A loaded Boltz2 model.
    name_or_path : str
        Name (registry lookup) or path to a fine-tune directory.
    strict : bool
        If True, every key in the saved state-dict must exist on the model.
        If False, unknown keys are logged and skipped (handy when applying
        a fine-tune trained on a slightly different ensemble configuration).

    Returns
    -------
    FinetuneRecord
        Metadata of the applied fine-tune.
    """
    registry = registry or default_registry()
    record, state = registry.load(name_or_path)

    model_state = model.state_dict()
    matched: list[str] = []
    missing: list[str] = []
    shape_mismatched: list[tuple[str, tuple, tuple]] = []
    for k, v in state.items():
        if k not in model_state:
            missing.append(k)
            continue
        target = model_state[k]
        if tuple(target.shape) != tuple(v.shape):
            shape_mismatched.append((k, tuple(target.shape), tuple(v.shape)))
            continue
        with torch.no_grad():
            target.copy_(v.to(target.device, dtype=target.dtype))
        matched.append(k)

    if shape_mismatched:
        msg = (
            f"Fine-tune '{record.name}' has {len(shape_mismatched)} tensors "
            "with shape mismatches against the current model. "
            f"First few: {shape_mismatched[:3]}"
        )
        raise RuntimeError(msg)

    if missing:
        msg = (
            f"Fine-tune '{record.name}' has {len(missing)} parameter names "
            f"that do not exist on the current model (e.g. {missing[:3]}). "
            "This usually means the base checkpoint changed (compiled vs "
            "uncompiled affinity module, or ensemble vs single-module)."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    logger.info(
        "Applied fine-tune '%s' (%d params loaded, %d missing) targeting %s",
        record.name,
        len(matched),
        len(missing),
        record.config.target_spec,
    )
    return record


__all__ = ["load_finetune_into_model"]
