"""L2-SP ("L2 starting point") regularization for full fine-tuning.

Unlike standard weight decay (``--weight-decay``, applied via AdamW and
pulling parameters toward **zero**), L2-SP pulls each trainable parameter
back toward its value at the *start of training* — normally the pretrained
Boltz2 checkpoint. This makes it a fair, capacity-matched comparator to
LoRA: a LoRA adapter is implicitly anchored to the base weights by
construction (it can only add a low-rank residual), whereas an unconstrained
full fine-tune can drift arbitrarily far from the pretrained model.

Reference: Xuhong et al., "Explicit Inductive Bias for Transfer Learning
with Convolutional Networks", ICML 2018.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

__all__ = ["l2_sp_penalty", "add_l2_sp_penalty"]


def l2_sp_penalty(
    current_params: Dict[str, torch.Tensor],
    initial_params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute the L2-SP penalty between current and initial parameter values.

    The penalty is the **sum** (not mean) of squared L2 distances between
    each current tensor and its initial snapshot, summed across every
    tensor in ``current_params``::

        penalty = sum_i ||current_i - initial_i||_2^2

    Sum (rather than mean) is the formulation used in the original L2-SP
    paper; callers are expected to scale the result by a small weight
    (e.g. ``l2_sp_weight``) before adding it to the primary training loss.

    Args:
        current_params: mapping from parameter name to its *live* tensor
            (as currently held by the model/optimizer).
        initial_params: mapping from the same parameter names to a
            detached snapshot taken at the start of training.

    Returns:
        A 0-dim (scalar) tensor. Requires-grad follows ``current_params``.

    Raises:
        ValueError: if ``current_params`` is empty.
        KeyError: if ``initial_params`` is missing a name present in
            ``current_params``.
    """
    if not current_params:
        msg = "l2_sp_penalty called with an empty current_params mapping."
        raise ValueError(msg)

    penalty: Optional[torch.Tensor] = None
    for name, current in current_params.items():
        if name not in initial_params:
            msg = f"initial_params is missing snapshot for parameter {name!r}."
            raise KeyError(msg)
        initial = initial_params[name].to(
            device=current.device, dtype=current.dtype,
        )
        term = (current - initial).pow(2).sum()
        penalty = term if penalty is None else penalty + term
    return penalty


def add_l2_sp_penalty(
    loss: torch.Tensor,
    l2_sp_weight: float,
    current_params: Dict[str, torch.Tensor],
    initial_params: Optional[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Add ``l2_sp_weight * l2_sp_penalty(...)`` to ``loss`` if enabled.

    When ``l2_sp_weight <= 0.0`` (the default), ``loss`` is returned
    unchanged and the penalty is never computed — this keeps the no-op
    path cheap and makes the opt-in nature of L2-SP obvious at call sites.

    Args:
        loss: the primary training loss for this step.
        l2_sp_weight: scale factor for the penalty; ``<= 0.0`` disables it.
        current_params: mapping from parameter name to its live tensor.
        initial_params: snapshot of ``current_params`` at the start of
            training, or ``None`` if L2-SP is disabled.

    Returns:
        ``loss`` unchanged, or ``loss + l2_sp_weight * penalty``.

    Raises:
        ValueError: if ``l2_sp_weight > 0.0`` but ``initial_params`` is
            ``None`` (i.e. no snapshot was captured).
    """
    if l2_sp_weight <= 0.0:
        return loss
    if initial_params is None:
        msg = (
            "l2_sp_weight > 0.0 but no initial parameter snapshot was "
            "captured at the start of training."
        )
        raise ValueError(msg)
    penalty = l2_sp_penalty(current_params, initial_params)
    return loss + l2_sp_weight * penalty
