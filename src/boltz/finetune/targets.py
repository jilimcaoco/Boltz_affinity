"""Parameter-name regex presets for full fine-tuning of the affinity stack.

Patterns are matched (via :func:`re.search`) against fully-qualified names
yielded by ``model.named_parameters()``.  ``torch.compile`` wraps modules
under ``_orig_mod``, so the patterns are written to be insensitive to that
detail (e.g. ``affinity_module(?:1|2)?`` matches both compiled and
uncompiled trees).
"""

from __future__ import annotations

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


def resolve_targets(spec: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Resolve a preset name, single regex, or list of regexes."""
    if isinstance(spec, str):
        if spec in TARGET_PRESETS:
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
