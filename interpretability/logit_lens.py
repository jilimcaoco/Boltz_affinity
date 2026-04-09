"""
interpretability/logit_lens.py
==============================
"Logit lens" for the affinity pairformer stack.

The idea (borrowed from language-model interpretability) is to take the pair
representation z at an *intermediate* layer — before the stack has finished
refining it — and ask: "if the affinity head read z right now, what would it
predict?"  This tells you how the IC50 estimate evolves block-by-block and at
which layer the model has "made up its mind."

Pooling operation found in affinity.py (AffinityHeadsTransformer.forward,
lines 208-210):
    g = torch.sum(z * cross_pair_mask, dim=(1, 2))
        / (torch.sum(cross_pair_mask, dim=(1, 2)) + 1e-7)

This is a **masked mean pool** over both spatial dimensions (i, j) of the
pair representation z (B, N, N, token_z).  The mask selects ligand-receptor
cross-pair entries plus ligand-ligand entries (excluding self-pairs on the
diagonal) and has an implicit trailing dim of 1 for broadcasting against the
token_z channel dimension.

After pooling to a vector g of shape (B, token_z), the head runs two
sequential MLPs using the model's learned weights:

  1. affinity_out_mlp:          Linear(token_z, token_z) → ReLU
                                → Linear(token_z, input_token_s) → ReLU
  2. to_affinity_pred_value:    Linear(input_token_s, input_token_s) → ReLU
                                → Linear(input_token_s, input_token_s) → ReLU
                                → Linear(input_token_s, 1)

The logit lens faithfully reproduces this exact pipeline with the model's own
weights for each intermediate z.
"""

from __future__ import annotations

import torch

# Type-only import to avoid circular deps at runtime; the caller passes the
# live object in, so no actual import of AffinityModule is needed at call
# time.  The import here is purely for type-checking / IDE support.
from interpretability.hooks import InstrumentedAffinityModule


@torch.no_grad()
def compute_logit_lens(
    instrumented_module: InstrumentedAffinityModule,
    layer_z_dict: dict[int, torch.Tensor],
    interface_mask: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Project each intermediate z through the affinity head to obtain a
    pseudo-IC50 at every pairformer layer.

    Parameters
    ----------
    instrumented_module : InstrumentedAffinityModule
        The hooked wrapper.  Its ``.module`` attribute is the live
        ``AffinityModule`` whose learned weights we borrow.
    layer_z_dict : dict[int, torch.Tensor]
        Mapping ``{layer_idx: z}`` where z has shape ``(B, N, N, token_z)``.
        Typically comes from ``instrumented_module.captured_z`` after a
        forward pass.  If the dict contains key ``-1``, that entry is the
        pre-layer-0 z (before any pairformer layer has processed it) and
        will appear first in the returned trajectory.  When visualising
        the trajectory, plot the ``-1`` entry as the leftmost point.
    interface_mask : torch.Tensor
        Boolean mask of shape ``(B, N, N)`` indicating which (i, j) pair
        entries should contribute to the mean pool.  This is the caller's
        version of the ``cross_pair_mask`` computed inside
        ``AffinityHeadsTransformer.forward``.

    Returns
    -------
    dict[int, torch.Tensor]
        Mapping ``{layer_idx: pseudo_ic50}`` where ``pseudo_ic50`` has shape
        ``(B, 1)`` — one scalar per batch element.

    Physical interpretation
    -----------------------
    ``pseudo_ic50[k]`` is the binding affinity (pIC50-scale) that the model
    *would* have predicted if the pairformer stack were truncated after layer
    ``k`` and the rest of the computation (including the regression MLP) ran
    on that partially-refined z.

    *  If pseudo_ic50 converges by layer 2 of 4, the later layers are
       performing only minor refinements — the binding signal is already
       encoded early in the stack.
    *  If pseudo_ic50 changes sharply at layer k, that layer is making a
       critical structural decision about the prediction.
    *  A non-monotonic trajectory (e.g. the prediction swings then settles)
       suggests interference or error-correction dynamics between layers.

    Comparing the pseudo_ic50 trajectory across complexes with different
    affinities can reveal whether the model separates binders from
    non-binders early or late, and which layers are responsible.
    """

    # ---- Grab the real affinity head from the wrapped module ----
    affinity_heads = instrumented_module.module.affinity_heads

    # The two MLPs we need (Sequential modules with learned weights):
    #   affinity_out_mlp:       token_z  → input_token_s  (with ReLU)
    #   to_affinity_pred_value: input_token_s → 1          (with ReLU)
    affinity_out_mlp = affinity_heads.affinity_out_mlp
    to_affinity_pred_value = affinity_heads.to_affinity_pred_value

    # ---- Prepare the mask ----
    # interface_mask: (B, N, N) bool  →  (B, N, N, 1) float for broadcasting.
    mask = interface_mask.unsqueeze(-1).to(dtype=torch.float32)

    # Denominator for the mean pool — number of active entries per channel.
    # Shape: (B, 1) after summing over dims 1 and 2 (the N×N spatial grid),
    # with the trailing 1 from the unsqueeze kept for broadcasting.
    denom = mask.sum(dim=(1, 2)).clamp(min=1e-7)  # (B, 1)

    results: dict[int, torch.Tensor] = {}

    for layer_idx, z in sorted(layer_z_dict.items()):
        # z: (B, N, N, token_z)

        # 1. Masked mean pool — identical to AffinityHeadsTransformer lines
        #    208-210.
        g = (z * mask).sum(dim=(1, 2)) / denom  # (B, token_z)

        # 2. First MLP stage: token_z → input_token_s
        g = affinity_out_mlp(g)  # (B, input_token_s)

        # 3. Regression head: input_token_s → 1
        pred = to_affinity_pred_value(g)  # (B, 1)

        results[layer_idx] = pred

    return results
