"""Target presets for LoRA injection and full fine-tuning of the affinity stack.

Patterns are matched against fully-qualified ``named_modules`` /
``named_parameters`` paths via :func:`re.search`.

**Scope presets** (``affinity_heads``, ``affinity_pairformer``,
``affinity_heads_pairformer``, ``affinity_module``) name module *subtrees*
inside the affinity module(s) actually used by the training forward. LoRA
adapts every ``nn.Linear`` inside them; ``boltz finetune`` trains every
parameter inside them, so the two methods cover the same part of the model.
Patterns are anchored at the affinity module, so nothing outside it (e.g.
``confidence_module.pairformer_stack``) can match.

**Deprecated presets** ``heads`` / ``heads_pairformer`` keep their original
patterns so artifacts trained with them can be reproduced. ``heads_pairformer``
targets ``to_q``/``to_k``/``to_v``/``to_out``/``to_gate``/``linear_no_bias``
names that do not exist in Boltz-2 (only ``transition*.fc*`` and the heads
match) and is unanchored (it also adapts ``confidence_module``).
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Affinity prediction heads only — small, cheap, lowest risk.
# Targets the three Sequential MLPs in ``AffinityHeadsTransformer``.
AFFINITY_HEADS_ONLY: tuple[str, ...] = (
    r"affinity_heads\.affinity_out_mlp\.\d+$",
    r"affinity_heads\.to_affinity_pred_value\.\d+$",
    r"affinity_heads\.to_affinity_pred_score\.\d+$",
    r"affinity_heads\.to_affinity_logits_binary$",
)

# Heads + pairformer linears inside the affinity module's pairformer stack.
# Captures attention projections (to_q/to_k/to_v/to_out) and transition MLPs.
AFFINITY_HEADS_AND_PAIRFORMER: tuple[str, ...] = AFFINITY_HEADS_ONLY + (
    r"pairformer_stack\..*\.to_q$",
    r"pairformer_stack\..*\.to_k$",
    r"pairformer_stack\..*\.to_v$",
    r"pairformer_stack\..*\.to_out$",
    r"pairformer_stack\..*\.to_gate$",
    r"pairformer_stack\..*transition.*\.fc\d+$",
    r"pairformer_stack\..*\.linear_no_bias\d*$",
)

TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "heads": AFFINITY_HEADS_ONLY,
    "heads_pairformer": AFFINITY_HEADS_AND_PAIRFORMER,
}

DEPRECATED_PRESETS: dict[str, str] = {
    "heads": "affinity_heads",
    "heads_pairformer": "affinity_heads_pairformer",
}

# Scope preset -> child subtrees of an affinity module ("" = the whole module).
SCOPE_PRESETS: dict[str, tuple[str, ...]] = {
    "affinity_heads": ("affinity_heads",),
    "affinity_pairformer": ("pairformer_stack",),
    "affinity_heads_pairformer": ("affinity_heads", "pairformer_stack"),
    "affinity_module": ("",),
}

# Every affinity-module attribute name Boltz-2 uses (single or ensemble).
ALL_AFFINITY_MODULES: tuple[str, ...] = (
    "affinity_module", "affinity_module1", "affinity_module2",
)


def scope_patterns(spec: str, modules: Iterable[str]) -> tuple[str, ...]:
    """Anchored regexes for scope preset ``spec`` within ``modules``.

    ``(?:\\._orig_mod)?`` keeps the patterns valid when a module is wrapped by
    ``torch.compile``.
    """
    mods = "|".join(re.escape(m) for m in modules)
    out = []
    for sub in SCOPE_PRESETS[spec]:
        tail = rf"{re.escape(sub)}\." if sub else ""
        out.append(rf"^(?:{mods})(?:\._orig_mod)?\.{tail}")
    return tuple(out)


def resolve_targets(
    spec: str | list[str] | tuple[str, ...],
    *,
    modules: Optional[Iterable[str]] = None,
    legacy: bool = False,
) -> tuple[str, ...]:
    """Resolve a preset name or explicit pattern list to a regex tuple.

    ``modules`` restricts scope presets to the affinity modules in the
    training forward (default: any affinity module). With ``legacy=True``
    scope preset names are not recognised and resolve exactly as before they
    existed (as a raw regex), so pre-v2 invocations reproduce bit-identically.
    """
    if isinstance(spec, str):
        if not legacy and spec in SCOPE_PRESETS:
            return scope_patterns(spec, modules or ALL_AFFINITY_MODULES)
        if spec in TARGET_PRESETS:
            if spec in DEPRECATED_PRESETS and not legacy:
                logger.warning(
                    "LoRA target preset %r is deprecated: several of its "
                    "patterns match no Boltz-2 layer and it also adapts "
                    "confidence_module. Use %r. (Kept unchanged so existing "
                    "adapters can be reproduced.)",
                    spec, DEPRECATED_PRESETS[spec],
                )
            return TARGET_PRESETS[spec]
        return (spec,)
    return tuple(spec)


__all__ = [
    "AFFINITY_HEADS_AND_PAIRFORMER",
    "AFFINITY_HEADS_ONLY",
    "ALL_AFFINITY_MODULES",
    "DEPRECATED_PRESETS",
    "SCOPE_PRESETS",
    "TARGET_PRESETS",
    "resolve_targets",
    "scope_patterns",
]
