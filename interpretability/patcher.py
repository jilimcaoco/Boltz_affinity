"""
interpretability/patcher.py
============================
Causal patching for the AffinityModule's pairformer stack.

Provides :func:`patch_and_measure`, which applies a targeted intervention
to a single layer or head, runs the full forward pass, and returns the
causal effect on the predicted IC50.

Sign convention
---------------
The return value is ``patched_ic50 - baseline_ic50``:

  * **Positive** → the patch *increased* the predicted IC50, i.e. the model
    now thinks binding is weaker.  The patched component was contributing
    to a *lower* (stronger-affinity) prediction.
  * **Negative** → the patch *decreased* IC50, meaning the patched component
    was *hurting* predicted affinity (or suppressing a repulsive signal).
  * **Near zero** → the component has little causal effect on the prediction.

Hook strategy
-------------
All three patch types are implemented via ``register_forward_pre_hook`` or
``register_forward_hook`` on sub-modules of the pairformer stack, **not** by
modifying weights.  Hooks are always removed in a ``finally`` block.

Patch types
-----------
``"zero_pairs"``
    Zero out z[..., i, j, :] and z[..., j, i, :] for all ``(i, j)`` where
    ``pair_mask[i, j]`` is True.  Applied as a **pre-hook** on the target
    pairformer layer so the zeroed positions propagate through that layer's
    tri_mul and tri_att operations.

``"zero_head"``
    Zero the contribution of a single triangle-attention head.  Implemented
    by hooking the ``softmax`` sub-module inside ``tri_att_start.mha``:
    the hook zeros the attention-weight slice for the target head
    (``a[..., head_idx, :, :] = 0``), which causes that head's value
    aggregation to produce zero — effectively removing its contribution
    from the output of ``linear_o``.

    The softmax module lives at::

        layer.tri_att_start.mha.softmax

    and its output has shape ``[B, N, H, N, N]`` with ``H`` at dim ``-3``.

``"swap_z"``
    Replace the pair representation ``z`` entering the target layer with a
    pre-computed alternative tensor.  Useful for target-shuffling experiments
    where you want to feed the z from a different complex into a specific
    layer.  Applied as a **pre-hook** on the pairformer layer.

    Requires ``patch_spec["swap_tensor"]`` of the same shape as z.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from interpretability.hooks import InstrumentedAffinityModule


def patch_and_measure(
    instrumented_module: InstrumentedAffinityModule,
    forward_kwargs: dict,
    patch_spec: dict,
    baseline_ic50: float,
) -> float:
    """Apply a causal patch and return its effect on predicted IC50.

    Parameters
    ----------
    instrumented_module : InstrumentedAffinityModule
        The wrapped AffinityModule (``instrumented_module.module`` is the
        underlying ``AffinityModule``).
    forward_kwargs : dict
        Keyword arguments for ``AffinityModule.forward()``
        (keys: ``s_inputs``, ``z``, ``x_pred``, ``feats``, etc.).
    patch_spec : dict
        Describes the intervention.  Required keys:

        - ``"type"`` : ``"zero_pairs"`` | ``"zero_head"`` | ``"swap_z"``
        - ``"layer_idx"`` : int — target pairformer layer index.
        - ``"head_idx"`` : int — target attention head (for ``"zero_head"``).
        - ``"pair_mask"`` : bool Tensor of shape ``(N, N)`` — positions to
          zero (for ``"zero_pairs"``).
        - ``"swap_tensor"`` : Tensor — replacement z (for ``"swap_z"``).

    baseline_ic50 : float
        The IC50 prediction from the clean (un-patched) forward pass.

    Returns
    -------
    float
        ``patched_ic50 - baseline_ic50``.

        * Positive → patch weakened predicted affinity (raised IC50).
        * Negative → patch strengthened predicted affinity (lowered IC50).
        * ≈ 0     → patched component has negligible causal effect.

    Raises
    ------
    ValueError
        If ``patch_spec["type"]`` is not one of the three supported types.
    """
    patch_type = patch_spec["type"]
    layer_idx = patch_spec.get("layer_idx", 0)

    # The underlying AffinityModule.
    affinity_module = instrumented_module.module

    # Access the target pairformer layer.
    # Path: AffinityModule.pairformer_stack.layers[layer_idx]
    #       (PairformerNoSeqLayer)
    layer = affinity_module.pairformer_stack.layers[layer_idx]

    hooks: list[torch.utils.hooks.RemovableHook] = []

    try:
        if patch_type == "zero_pairs":
            # ---------------------------------------------------------
            # Pre-hook on the PairformerNoSeqLayer: zero specified
            # (i, j) pairs in z before the layer processes it.
            #
            # PairformerNoSeqLayer.forward signature:
            #   forward(z, pair_mask, chunk_size_tri_attn, use_kernels, ...)
            # z is the first positional argument (after self).
            # ---------------------------------------------------------
            pair_mask = patch_spec["pair_mask"]  # (N, N) bool

            def _zero_pairs_pre_hook(
                mod: nn.Module,
                args: tuple,
            ) -> tuple:
                z = args[0]  # (B, N, N, token_z)
                # pair_mask: (N, N) → broadcast to (1, N, N, 1)
                mask = pair_mask.to(device=z.device, dtype=torch.bool)
                # Zero z[..., i, j, :] where mask[i, j] is True
                z = z.clone()
                z[:, mask] = 0.0
                # Also zero the transpose positions z[..., j, i, :]
                mask_T = mask.T
                z[:, mask_T] = 0.0
                return (z,) + args[1:]

            hooks.append(layer.register_forward_pre_hook(_zero_pairs_pre_hook))

        elif patch_type == "zero_head":
            # ---------------------------------------------------------
            # Hook the softmax sub-module inside tri_att_start.mha.
            #
            # The softmax output has shape [B, N, H, N, N] with H at
            # dim -3.  Zeroing a[..., head_idx, :, :] causes that
            # head's value aggregation (torch.matmul(a, v)) to produce
            # zero, effectively ablating the head's contribution to the
            # output projection (linear_o) and therefore to the
            # residual update added to z.
            #
            # Path: layer.tri_att_start.mha.softmax
            # ---------------------------------------------------------
            head_idx = patch_spec["head_idx"]
            softmax_module = layer.tri_att_start.mha.softmax

            def _zero_head_hook(
                mod: nn.Module,
                inputs: tuple,
                output: Tensor,
            ) -> Tensor:
                # output: [B, N, H, N, N]  — H at dim -3
                output = output.clone()
                output[..., head_idx, :, :] = 0.0
                return output

            hooks.append(softmax_module.register_forward_hook(_zero_head_hook))

        elif patch_type == "swap_z":
            # ---------------------------------------------------------
            # Pre-hook on the PairformerNoSeqLayer: replace z entirely
            # with a pre-computed alternative tensor.
            # ---------------------------------------------------------
            swap_tensor = patch_spec["swap_tensor"]

            def _swap_z_pre_hook(
                mod: nn.Module,
                args: tuple,
            ) -> tuple:
                replacement = swap_tensor.to(device=args[0].device)
                return (replacement,) + args[1:]

            hooks.append(layer.register_forward_pre_hook(_swap_z_pre_hook))

        else:
            raise ValueError(
                f"Unknown patch type: {patch_type!r}.  "
                f"Expected one of: 'zero_pairs', 'zero_head', 'swap_z'."
            )

        # Run the patched forward pass.
        with torch.no_grad():
            out = instrumented_module(**forward_kwargs)

        # Extract the IC50 prediction.
        # AffinityModule.forward returns:
        #   {"affinity_pred_value": Tensor(B, 1), "affinity_logits_binary": ...}
        patched_ic50 = out["affinity_pred_value"].item()

    finally:
        # Always remove hooks, even if the forward pass raises.
        for h in hooks:
            h.remove()

    return patched_ic50 - baseline_ic50
