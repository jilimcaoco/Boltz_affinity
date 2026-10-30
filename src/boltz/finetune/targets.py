"""Parameter-name regex presets for full fine-tuning of the affinity stack.

Patterns are matched (via :func:`re.search`) against fully-qualified names
yielded by ``model.named_parameters()``.  ``torch.compile`` wraps modules
under ``_orig_mod``, so the patterns are written to be insensitive to that
detail (e.g. ``affinity_module(?:1|2)?`` matches both compiled and
uncompiled trees).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from boltz.lora.targets import ALL_AFFINITY_MODULES, SCOPE_PRESETS, scope_patterns

logger = logging.getLogger(__name__)

# Default: all parameters under the affinity module(s). This includes the
# ensemble variants ``affinity_module1`` and ``affinity_module2`` when
# present.
ALL_AFFINITY_MODULE: tuple[str, ...] = (
    r"^(?:.*?\.)?affinity_module(?:1|2)?(?:\._orig_mod)?\.",
)

# Heads-only — direct mirror of the LoRA ``heads`` preset so the two
# baselines train the same parameter family.
AFFINITY_HEADS_ONLY: tuple[str, ...] = (
    r"affinity_module(?:1|2)?(?:\._orig_mod)?\.affinity_heads\.",
)

# Pairformer linears inside the affinity module.
AFFINITY_PAIRFORMER_ONLY: tuple[str, ...] = (
    r"affinity_module(?:1|2)?(?:\._orig_mod)?\.pairformer_stack\.",
)

AFFINITY_HEADS_AND_PAIRFORMER: tuple[str, ...] = (
    AFFINITY_HEADS_ONLY + AFFINITY_PAIRFORMER_ONLY
)

TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "affinity_module": ALL_AFFINITY_MODULE,
    "affinity_heads": AFFINITY_HEADS_ONLY,
    "affinity_pairformer": AFFINITY_PAIRFORMER_ONLY,
    "heads_pairformer": AFFINITY_HEADS_AND_PAIRFORMER,
    # Aliases that match the LoRA preset names.
    "heads": AFFINITY_HEADS_ONLY,
}


# Aliases whose LoRA preset of the same name covers a different (smaller) set
# of layers; kept with their original patterns for reproducibility.
_MISMATCHED_ALIASES: dict[str, str] = {
    "heads": "affinity_heads",
    "heads_pairformer": "affinity_heads_pairformer",
}


def resolve_targets(
    spec: str | list[str] | tuple[str, ...],
    *,
    modules: Optional[Iterable[str]] = None,
    legacy: bool = False,
) -> tuple[str, ...]:
    """Resolve a preset name, single regex, or list of regexes.

    Scope presets (``affinity_heads``, ``affinity_pairformer``,
    ``affinity_heads_pairformer``, ``affinity_module``) are shared with
    :mod:`boltz.lora.targets` and restricted to ``modules`` (the affinity
    modules in the training forward). ``legacy=True`` resolves every preset
    exactly as before (both ensemble modules), for reproducing old runs.
    """
    if isinstance(spec, str):
        if not legacy and spec in SCOPE_PRESETS:
            return scope_patterns(spec, modules or ALL_AFFINITY_MODULES)
        if spec in TARGET_PRESETS:
            if spec in _MISMATCHED_ALIASES and not legacy:
                logger.warning(
                    "Fine-tune preset %r trains every parameter under the "
                    "affinity heads%s of both ensemble modules, which is NOT "
                    "what the LoRA preset of the same name adapts. Use %r for "
                    "a preset that means the same thing in both tools.",
                    spec, " and pairformer stack" if "pairformer" in spec else "",
                    _MISMATCHED_ALIASES[spec],
                )
            return TARGET_PRESETS[spec]
        # Treat unknown strings as a comma-separated regex list, matching
        # the LoRA UX.
        if "," in spec:
            return tuple(s.strip() for s in spec.split(",") if s.strip())
        return (spec,)
    return tuple(spec)


__all__ = [
    "AFFINITY_HEADS_AND_PAIRFORMER",
    "AFFINITY_HEADS_ONLY",
    "AFFINITY_PAIRFORMER_ONLY",
    "ALL_AFFINITY_MODULE",
    "TARGET_PRESETS",
    "resolve_targets",
]
