"""Loss functions for LoRA training.

All loss functions share the contract::

    loss_fn(pred: dict[str, torch.Tensor],
            batch: dict[str, Any],
            adapter_meta: LoRAAdapter | None = None) -> torch.Tensor

``pred`` is the dict returned by ``AffinityModule.forward`` (keys include
``affinity_pred_value`` and ``affinity_logits_binary``). ``batch`` is the
batch dict yielded by :class:`boltz.lora.data.LoRADataset`; it must contain a
``target`` tensor of shape ``[B]``.

Users can register their own via :func:`load_loss_from_spec`, which accepts
either a built-in name (``"mse"``, ``"mae"``, ...) or a string of the form
``path/to/file.py:fn_name``.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

# Loss signature: (pred_dict, batch, adapter_meta) -> scalar tensor
LossFn = Callable[[Dict[str, torch.Tensor], Dict[str, Any], Optional[Any]], torch.Tensor]


def _pred_value(pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "affinity_pred_value" not in pred:
        msg = "Pred dict is missing 'affinity_pred_value'."
        raise KeyError(msg)
    v = pred["affinity_pred_value"]
    return v.squeeze(-1) if v.dim() > 1 and v.shape[-1] == 1 else v


def _target(batch: Dict[str, Any]) -> torch.Tensor:
    if "target" not in batch:
        msg = "Batch is missing 'target' tensor."
        raise KeyError(msg)
    return batch["target"].to(dtype=torch.float32)


def mse_loss(pred, batch, adapter_meta=None):  # noqa: D401, ANN001
    return F.mse_loss(_pred_value(pred).float(), _target(batch))


def mae_loss(pred, batch, adapter_meta=None):  # noqa: ANN001
    return F.l1_loss(_pred_value(pred).float(), _target(batch))


def huber_loss(pred, batch, adapter_meta=None):  # noqa: ANN001
    return F.smooth_l1_loss(_pred_value(pred).float(), _target(batch))


def bce_loss(pred, batch, adapter_meta=None):  # noqa: ANN001
    """Binary cross-entropy on ``affinity_logits_binary``."""
    if "affinity_logits_binary" not in pred:
        msg = "BCE loss requires 'affinity_logits_binary' in pred dict."
        raise KeyError(msg)
    logits = pred["affinity_logits_binary"]
    if logits.dim() > 1 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits.float(), _target(batch))


def pairwise_ranking_loss(pred, batch, adapter_meta=None):  # noqa: ANN001
    """Margin ranking over all pairs in the batch (Bradley-Terry style)."""
    v = _pred_value(pred).float()
    t = _target(batch)
    if v.numel() < 2:
        return torch.zeros((), device=v.device, dtype=v.dtype)
    diff_pred = v.unsqueeze(0) - v.unsqueeze(1)
    diff_true = t.unsqueeze(0) - t.unsqueeze(1)
    sign = torch.sign(diff_true)
    # softplus margin loss; ignores ties (sign == 0)
    mask = (sign != 0).float()
    return (F.softplus(-sign * diff_pred) * mask).sum() / mask.sum().clamp(min=1.0)


BUILTIN_LOSSES: dict[str, LossFn] = {
    "mse": mse_loss,
    "mae": mae_loss,
    "huber": huber_loss,
    "bce": bce_loss,
    "pairwise_ranking": pairwise_ranking_loss,
}


def _validate_signature(fn: Callable) -> None:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if len(params) < 2:
        msg = (
            f"Custom loss '{fn.__qualname__}' must accept at least "
            "(pred, batch[, adapter_meta]); got "
            f"{len(params)} params."
        )
        raise TypeError(msg)


def load_loss_from_spec(spec: str) -> LossFn:
    """Resolve a loss specification.

    Accepted forms:
    * ``"mse"``, ``"mae"``, ... — built-in registry lookup.
    * ``"/abs/path/file.py:fn_name"`` or ``"rel/path.py:fn"`` — import the
      file as an isolated module and return ``fn``.
    """
    if spec in BUILTIN_LOSSES:
        return BUILTIN_LOSSES[spec]

    if ":" not in spec:
        msg = (
            f"Unknown loss '{spec}'. Built-ins: {sorted(BUILTIN_LOSSES)}. "
            "For a custom loss use 'path/to/file.py:fn_name'."
        )
        raise KeyError(msg)

    path_part, fn_name = spec.rsplit(":", 1)
    path = Path(path_part).expanduser().resolve()
    if not path.exists():
        msg = f"Custom loss file not found: {path}"
        raise FileNotFoundError(msg)

    mod_name = f"_boltz_lora_user_loss_{abs(hash(str(path)))}"
    spec_obj = importlib.util.spec_from_file_location(mod_name, path)
    if spec_obj is None or spec_obj.loader is None:
        msg = f"Could not import custom loss module from {path}."
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[mod_name] = module
    spec_obj.loader.exec_module(module)

    if not hasattr(module, fn_name):
        msg = f"Module {path} does not define '{fn_name}'."
        raise AttributeError(msg)
    fn = getattr(module, fn_name)
    if not callable(fn):
        msg = f"'{fn_name}' in {path} is not callable."
        raise TypeError(msg)
    _validate_signature(fn)
    return fn  # type: ignore[return-value]


def call_loss(
    loss_fn: LossFn,
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    adapter_meta: Optional[Any] = None,
) -> torch.Tensor:
    """Call a loss fn defensively (handles 2- vs 3-arg signatures)."""
    sig = inspect.signature(loss_fn)
    if len(sig.parameters) >= 3:
        return loss_fn(pred, batch, adapter_meta)
    return loss_fn(pred, batch)  # type: ignore[call-arg]


__all__ = [
    "BUILTIN_LOSSES",
    "LossFn",
    "bce_loss",
    "call_loss",
    "huber_loss",
    "load_loss_from_spec",
    "mae_loss",
    "mse_loss",
    "pairwise_ranking_loss",
]
