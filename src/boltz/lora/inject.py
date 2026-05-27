"""Inject LoRA adapters into an existing ``nn.Module`` tree.

The injection walks ``named_modules``, matches each ``nn.Linear`` against the
caller-supplied regex patterns, replaces it in-place with a :class:`LoRALinear`,
and freezes every other parameter on the model.

Use :func:`apply_lora` for training, :func:`merge_lora` for export, and
:func:`remove_lora` to fully restore the base model.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import nn

from boltz.lora.layers import LoRALinear, wrap_linear


def _iter_linear(model: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            yield name, module


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """Replace ``root.<qualified_name>`` with ``new_module``."""
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module  # type: ignore[index]
    else:
        setattr(parent, leaf, new_module)


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


def apply_lora(
    model: nn.Module,
    target_patterns: Iterable[str],
    *,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> list[str]:
    """Replace every matching ``nn.Linear`` in ``model`` with a ``LoRALinear``.

    Returns the sorted list of fully-qualified layer names that were adapted.
    Raises :class:`ValueError` if no layers matched.
    """
    patterns = tuple(target_patterns)
    candidates = [(n, m) for n, m in _iter_linear(model)]
    adapted: list[str] = []
    for name, layer in candidates:
        if not _matches_any(name, patterns):
            continue
        new_layer = wrap_linear(layer, r=r, alpha=alpha, dropout=dropout)
        _set_submodule(model, name, new_layer)
        adapted.append(name)

    if not adapted:
        msg = (
            "apply_lora matched zero layers. Patterns: "
            f"{list(patterns)}. Inspect model.named_modules() for valid names."
        )
        raise ValueError(msg)

    if freeze_base:
        for p_name, p in model.named_parameters():
            # Trainable: only the lora_A / lora_B parameters under wrapped layers.
            p.requires_grad = p_name.endswith(".lora_A") or p_name.endswith(".lora_B")

    return sorted(adapted)


def remove_lora(model: nn.Module) -> list[str]:
    """Replace every ``LoRALinear`` back with its frozen base layer."""
    removed: list[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            base = module.base
            for p in base.parameters():
                p.requires_grad = True
            _set_submodule(model, name, base)
            removed.append(name)
    return sorted(removed)


def merge_lora(model: nn.Module) -> list[str]:
    """Fold LoRA deltas into each base layer and replace wrappers with bases."""
    merged: list[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            module.merge_()
            base = module.base
            for p in base.parameters():
                p.requires_grad = True
            _set_submodule(model, name, base)
            merged.append(name)
    return sorted(merged)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect only the LoRA tensors (A,B) across the whole model."""
    out: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for k, v in module.lora_state_dict().items():
                out[f"{name}.{k}"] = v
    return out


def load_lora_state_dict(
    model: nn.Module, state: dict[str, torch.Tensor], *, strict: bool = True
) -> None:
    """Load LoRA tensors saved via :func:`lora_state_dict`."""
    by_layer: dict[str, dict[str, torch.Tensor]] = {}
    for key, val in state.items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            layer_name, leaf = key.rsplit(".", 1)
            by_layer.setdefault(layer_name, {})[leaf] = val

    missing: list[str] = []
    for layer_name, params in by_layer.items():
        module = model.get_submodule(layer_name)
        if not isinstance(module, LoRALinear):
            if strict:
                msg = f"Target layer '{layer_name}' is not a LoRALinear; cannot load."
                raise RuntimeError(msg)
            missing.append(layer_name)
            continue
        module.load_lora_state_dict(params)


__all__ = [
    "apply_lora",
    "load_lora_state_dict",
    "lora_state_dict",
    "merge_lora",
    "remove_lora",
]
