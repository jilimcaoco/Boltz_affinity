"""Apply a trained LoRA adapter to a Boltz2 model for inference.

Used by both ``boltz predict --use-lora NAME`` and the
``boltz rescore`` family. The function is intentionally idempotent:
calling it twice on the same model will refuse to stack adapters
unless ``allow_restack=True`` (v1 supports a single adapter at a time).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from boltz.lora.adapter import LoRAAdapter
from boltz.lora.inject import apply_lora, load_lora_state_dict
from boltz.lora.layers import LoRALinear
from boltz.lora.registry import LoRARegistry, default_registry

logger = logging.getLogger(__name__)


def _already_adapted(model: Any) -> bool:
    return any(isinstance(m, LoRALinear) for _, m in model.named_modules())


def load_adapter_into_model(
    model: Any,
    name_or_path: str,
    *,
    registry: Optional[LoRARegistry] = None,
    merge: bool = False,
    allow_restack: bool = False,
    strict: bool = True,
) -> LoRAAdapter:
    """Inject a registered LoRA adapter into ``model`` in-place.

    Parameters
    ----------
    model : nn.Module
        A loaded Boltz2 model (or any subtree containing the affinity stack).
    name_or_path : str
        Adapter name (registry lookup) or path to an adapter directory.
    registry : LoRARegistry, optional
        Custom registry. Defaults to ``$BOLTZ_LORA_DIR`` resolver.
    merge : bool
        If True, fold the LoRA update into the base weights after loading
        (useful for exporting a single merged checkpoint).
    allow_restack : bool
        If False (default), refuse to apply when the model already has a
        LoRA installed. v1 does not support adapter composition.
    strict : bool
        If True, fail when saved adapter layers don't match the model.

    Returns
    -------
    LoRAAdapter
        The loaded adapter metadata, useful for logging / downstream code.
    """
    registry = registry or default_registry()

    if _already_adapted(model):
        if not allow_restack:
            msg = (
                "Model already has a LoRA adapter installed. "
                "Adapter composition is not supported in v1. "
                "Reload the base checkpoint, or pass allow_restack=True."
            )
            raise RuntimeError(msg)

    adapter, state = registry.load(name_or_path)

    # Use the same target patterns that were used at training time so the
    # injected LoRALinear layers exactly match the saved state_dict keys.
    target_patterns = adapter.config.target_patterns or list(adapter.adapted_layers)
    if not target_patterns:
        msg = (
            f"Adapter '{adapter.name}' lists no target layers; "
            "cannot determine where to inject."
        )
        raise RuntimeError(msg)

    injected = apply_lora(
        model, target_patterns,
        r=adapter.config.rank,
        alpha=adapter.config.alpha,
        dropout=0.0,           # always disabled at inference
        freeze_base=True,
    )
    if strict and set(injected) != set(adapter.adapted_layers):
        only_now = sorted(set(injected) - set(adapter.adapted_layers))
        only_saved = sorted(set(adapter.adapted_layers) - set(injected))
        msg = (
            f"Adapter '{adapter.name}' layer mismatch.\n"
            f"  Adapted now but not in adapter: {only_now}\n"
            f"  In adapter but not adapted now: {only_saved}\n"
            "The base checkpoint may differ from the one used for training. "
            "Pass strict=False to ignore."
        )
        raise RuntimeError(msg)

    load_lora_state_dict(model, state, strict=strict)

    if merge:
        from boltz.lora.inject import merge_lora
        merge_lora(model)

    logger.info(
        "Applied LoRA adapter '%s' (rank=%d, %d layers, merge=%s)",
        adapter.name, adapter.config.rank, len(injected), merge,
    )
    return adapter


__all__ = ["load_adapter_into_model"]
