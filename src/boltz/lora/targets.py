"""Default target patterns for LoRA injection on the Boltz affinity stack.

Patterns are matched against fully-qualified ``named_modules`` paths via
:func:`re.search` (so a substring regex is enough). Use anchors (^, $) for
strict matching.
"""

from __future__ import annotations

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


def resolve_targets(spec: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Resolve a preset name or explicit pattern list to a regex tuple."""
    if isinstance(spec, str):
        if spec in TARGET_PRESETS:
            return TARGET_PRESETS[spec]
        return (spec,)
    return tuple(spec)


__all__ = [
    "AFFINITY_HEADS_AND_PAIRFORMER",
    "AFFINITY_HEADS_ONLY",
    "TARGET_PRESETS",
    "resolve_targets",
]
