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
import os
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


def _target(batch: Dict[str, Any], device: Optional[torch.device] = None) -> torch.Tensor:
    if "target" not in batch:
        msg = "Batch is missing 'target' tensor."
        raise KeyError(msg)
    t = batch["target"].to(dtype=torch.float32)
    if device is not None:
        t = t.to(device=device)
    return t


def mse_loss(pred, batch, adapter_meta=None):  # noqa: D401, ANN001
    p = _pred_value(pred).float()
    return F.mse_loss(p, _target(batch, device=p.device))


def mae_loss(pred, batch, adapter_meta=None):  # noqa: ANN001
    p = _pred_value(pred).float()
    return F.l1_loss(p, _target(batch, device=p.device))


def huber_loss(pred, batch, adapter_meta=None):  # noqa: ANN001
    p = _pred_value(pred).float()
    return F.smooth_l1_loss(p, _target(batch, device=p.device))


def bce_loss(pred, batch, adapter_meta=None):  # noqa: ANN001
    """Binary cross-entropy on ``affinity_logits_binary``."""
    if "affinity_logits_binary" not in pred:
        msg = "BCE loss requires 'affinity_logits_binary' in pred dict."
        raise KeyError(msg)
    logits = pred["affinity_logits_binary"]
    if logits.dim() > 1 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    logits = logits.float()
    return F.binary_cross_entropy_with_logits(logits, _target(batch, device=logits.device))


def pairwise_ranking_loss(pred, batch, adapter_meta=None):  # noqa: ANN001
    """Margin ranking over all pairs in the batch (Bradley-Terry style)."""
    v = _pred_value(pred).float()
    t = _target(batch, device=v.device)
    if v.numel() < 2:
        return torch.zeros((), device=v.device, dtype=v.dtype)
    diff_pred = v.unsqueeze(0) - v.unsqueeze(1)
    diff_true = t.unsqueeze(0) - t.unsqueeze(1)
    sign = torch.sign(diff_true)
    # softplus margin loss; ignores ties (sign == 0)
    mask = (sign != 0).float()
    return (F.softplus(-sign * diff_pred) * mask).sum() / mask.sum().clamp(min=1.0)


def intra_assay_huber_loss(
    pred,  # noqa: ANN001
    batch,  # noqa: ANN001
    adapter_meta=None,  # noqa: ANN001
    *,
    delta: float = 1.0,
    pair_weight: float = 2.0,
):
    """Boltz-2 style absolute + intra-assay pairwise Huber loss.

    The Boltz-2 affinity head was trained with a Huber loss on the absolute
    ``log10(IC50 in uM)`` value plus a *stronger-weighted* Huber loss on the
    pairwise differences of compounds measured **within the same assay**.
    Restricting the pairwise term to one assay cancels the Cheng-Prusoff /
    inter-assay offset, which is what lets Ki / Kd / IC50 / EC50 / AC50 / XC50
    measurements be trained jointly.

    Requirements on the batch:
      * ``batch['target']`` must be on the same scale as
        ``pred['affinity_pred_value']`` (i.e. ``log10(IC50_uM)``, lower =
        stronger).
      * ``batch['group_id']`` must be a per-row sequence of assay identifiers
        (any hashable; commonly ``assay_chembl_id``).  Rows with
        ``group_id is None`` contribute to the absolute term only.

    With batch size 1 the pairwise term is zero, so this loss reduces to a
    plain Huber loss; use a grouped sampler (or sort the manifest by
    ``group_id`` and pick batch_size >= 4) to get real pairs per step.
    """
    p = _pred_value(pred).float()
    t = _target(batch, device=p.device)
    abs_loss = F.smooth_l1_loss(p, t, beta=delta, reduction="mean")

    groups = batch.get("group_id")
    if groups is None or p.numel() < 2:
        return abs_loss

    # Build pairwise mask within each non-null assay group.
    g_list = list(groups) if not torch.is_tensor(groups) else groups.tolist()
    n = len(g_list)
    pair_l: list[int] = []
    pair_r: list[int] = []
    for i in range(n):
        gi = g_list[i]
        if gi is None or (isinstance(gi, str) and not gi):
            continue
        for j in range(i + 1, n):
            if g_list[j] == gi:
                pair_l.append(i)
                pair_r.append(j)
    if not pair_l:
        return abs_loss

    pl = torch.tensor(pair_l, device=p.device, dtype=torch.long)
    pr = torch.tensor(pair_r, device=p.device, dtype=torch.long)
    diff_pred = p.index_select(0, pl) - p.index_select(0, pr)
    diff_true = t.index_select(0, pl) - t.index_select(0, pr)
    pair_loss = F.smooth_l1_loss(diff_pred, diff_true, beta=delta, reduction="mean")

    return abs_loss + pair_weight * pair_loss


def _get_censored_mask(batch: Dict[str, Any], n: int, device: torch.device) -> torch.Tensor:
    """Return a bool tensor of shape [n] — True where the compound is right-censored."""
    raw = batch.get("is_censored")
    if raw is None:
        return torch.zeros(n, dtype=torch.bool, device=device)
    if torch.is_tensor(raw):
        return raw.to(dtype=torch.bool, device=device)
    return torch.tensor([bool(int(v or 0)) for v in raw], dtype=torch.bool, device=device)


def censored_intra_assay_huber_loss(
    pred,  # noqa: ANN001
    batch,  # noqa: ANN001
    adapter_meta=None,  # noqa: ANN001
    *,
    delta: float = 1.0,
    pair_weight: float = 2.0,
):
    """Intra-assay Huber loss extended with support for right-censored measurements.

    Right-censored means the compound was reported as "> X µM", so its true
    ``log10_aff_uM`` is *at least* the reported value.  The standard two-sided
    Huber loss penalises equally whether the model is too potent or not potent
    enough, but for a censored compound only the "too potent" direction
    violates the known constraint.

    **Absolute term** — for compound *i* with ``is_censored=1``:

    .. math::
        \\ell_i = \\text{Huber}\\!\\left(\\max(0,\\; t_i - p_i)\\right)

    i.e. zero loss when the model already predicts ≥ the censored lower bound.

    **Pairwise intra-assay term** — for pair *(i, j)* in the same assay:

    * Both uncensored → standard two-sided Huber on ``(p_i - p_j) - (t_i - t_j)``.
    * Only *j* is censored (true t_j ≥ reported t_j): the model may over-estimate
      the potency of *i* relative to *j*.  Penalise only when
      ``p_i - p_j > t_i - t_j`` (model says i is disproportionately more potent):
      ``Huber(max(0, diff_pred - diff_true))``.
    * Only *i* is censored (true t_i ≥ reported t_i): penalise only when
      ``p_i - p_j < t_i - t_j`` (model too optimistic about i):
      ``Huber(max(0, diff_true - diff_pred))``.
    * Both censored → skip (no useful ordering signal).

    The ``is_censored`` flag is read from ``batch['is_censored']`` (list of
    int or tensor, 0/1).  If absent the function degrades gracefully to the
    standard :func:`intra_assay_huber_loss`.
    """
    p = _pred_value(pred).float()
    t = _target(batch, device=p.device)
    censored = _get_censored_mask(batch, p.numel(), p.device)

    # ---- Absolute term ---------------------------------------------------- #
    uncensored_idx = (~censored).nonzero(as_tuple=True)[0]
    censored_idx = censored.nonzero(as_tuple=True)[0]

    abs_terms: list[torch.Tensor] = []
    n_terms = 0
    if uncensored_idx.numel() > 0:
        pu, tu = p.index_select(0, uncensored_idx), t.index_select(0, uncensored_idx)
        abs_terms.append(F.smooth_l1_loss(pu, tu, beta=delta, reduction="sum"))
        n_terms += uncensored_idx.numel()
    if censored_idx.numel() > 0:
        pc, tc = p.index_select(0, censored_idx), t.index_select(0, censored_idx)
        # Only penalise when model predicts too potent (pred < target_reported)
        one_sided = F.relu(tc - pc)  # > 0 only when violation
        abs_terms.append(F.smooth_l1_loss(one_sided, torch.zeros_like(one_sided), beta=delta, reduction="sum"))
        n_terms += censored_idx.numel()

    abs_loss = (
        torch.stack(abs_terms).sum() / max(n_terms, 1)
        if abs_terms
        else torch.zeros((), device=p.device)
    )

    # ---- Pairwise intra-assay term ---------------------------------------- #
    groups = batch.get("group_id")
    if groups is None or p.numel() < 2:
        return abs_loss

    g_list = list(groups) if not torch.is_tensor(groups) else groups.tolist()
    cens_list = censored.tolist()
    n = len(g_list)
    pair_l: list[int] = []
    pair_r: list[int] = []
    pair_mode: list[str] = []  # "both", "r_cens", "l_cens"
    for i in range(n):
        gi = g_list[i]
        if gi is None or (isinstance(gi, str) and not gi):
            continue
        for j in range(i + 1, n):
            if g_list[j] != gi:
                continue
            i_cens, j_cens = cens_list[i], cens_list[j]
            if i_cens and j_cens:
                continue  # both censored — no signal
            pair_l.append(i)
            pair_r.append(j)
            if not i_cens and not j_cens:
                pair_mode.append("both")
            elif j_cens:
                pair_mode.append("r_cens")
            else:
                pair_mode.append("l_cens")

    if not pair_l:
        return abs_loss

    pl_t = torch.tensor(pair_l, device=p.device, dtype=torch.long)
    pr_t = torch.tensor(pair_r, device=p.device, dtype=torch.long)
    diff_pred = p.index_select(0, pl_t) - p.index_select(0, pr_t)
    diff_true = t.index_select(0, pl_t) - t.index_select(0, pr_t)

    mode_t = [0 if m == "both" else (1 if m == "r_cens" else 2) for m in pair_mode]
    mode_t = torch.tensor(mode_t, device=p.device, dtype=torch.long)

    both_mask = (mode_t == 0)
    r_cens_mask = (mode_t == 1)
    l_cens_mask = (mode_t == 2)

    pair_terms: list[torch.Tensor] = []
    n_pairs = 0
    if both_mask.any():
        dp, dt = diff_pred[both_mask], diff_true[both_mask]
        pair_terms.append(F.smooth_l1_loss(dp, dt, beta=delta, reduction="sum"))
        n_pairs += both_mask.sum().item()
    if r_cens_mask.any():
        # right compound censored: penalise when diff_pred > diff_true
        dp, dt = diff_pred[r_cens_mask], diff_true[r_cens_mask]
        pair_terms.append(F.smooth_l1_loss(F.relu(dp - dt), torch.zeros_like(dp), beta=delta, reduction="sum"))
        n_pairs += r_cens_mask.sum().item()
    if l_cens_mask.any():
        # left compound censored: penalise when diff_pred < diff_true
        dp, dt = diff_pred[l_cens_mask], diff_true[l_cens_mask]
        pair_terms.append(F.smooth_l1_loss(F.relu(dt - dp), torch.zeros_like(dp), beta=delta, reduction="sum"))
        n_pairs += l_cens_mask.sum().item()

    pair_loss = (
        torch.stack(pair_terms).sum() / max(n_pairs, 1)
        if pair_terms
        else torch.zeros((), device=p.device)
    )

    return abs_loss + pair_weight * pair_loss


def censored_boltz2_affinity_loss(
    pred,  # noqa: ANN001
    batch,  # noqa: ANN001
    adapter_meta=None,  # noqa: ANN001
    *,
    delta: float = 1.0,
    pair_weight: float = 2.0,
    bce_weight: float = 1.0,
):
    """Censored variant of :func:`boltz2_affinity_loss`.

    Combines :func:`censored_intra_assay_huber_loss` (one-sided Huber for
    right-censored compounds) with the binary cross-entropy branch on
    ``affinity_logits_binary``.  This is the drop-in replacement for
    ``boltz2_affinity`` when the training manifest includes censored rows
    (built with ``--keep-censored``).

    Binder labels and BCE mechanics are identical to
    :func:`boltz2_affinity_loss`; the only difference is that the value/
    pairwise term uses one-sided Huber penalties for censored compounds.
    """
    value = censored_intra_assay_huber_loss(
        pred, batch, adapter_meta, delta=delta, pair_weight=pair_weight,
    )

    if "affinity_logits_binary" not in pred:
        return value

    logits = pred["affinity_logits_binary"]
    if logits.dim() > 1 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    logits = logits.float()

    target = _target(batch, device=logits.device)
    labels = _binder_labels(batch, target)
    bce = F.binary_cross_entropy_with_logits(logits, labels)

    return value + bce_weight * bce


def _binder_threshold_from_env(default: float = 1.0) -> float:
    raw = os.environ.get("BOLTZ_LORA_BINDER_THRESHOLD")
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _binder_labels(batch: Dict[str, Any], target: torch.Tensor) -> torch.Tensor:
    """Resolve binder/non-binder labels for the BCE branch.

    Preference order:
      1. Explicit ``batch['is_binder']`` (list of int/None or tensor).
         Rows with ``None`` fall back to the threshold rule.
      2. Threshold rule: ``target <= BOLTZ_LORA_BINDER_THRESHOLD`` (default
         1.0, i.e. ≤10 µM on the ``log10(IC50_uM)`` scale).
    """
    thr = _binder_threshold_from_env(default=1.0)
    fallback = (target <= thr).float()

    raw = batch.get("is_binder")
    if raw is None:
        return fallback
    if torch.is_tensor(raw):
        tens = raw.to(device=target.device, dtype=torch.float32)
        return torch.where(torch.isnan(tens), fallback, tens)
    # list form (from lora_collate)
    if not isinstance(raw, (list, tuple)):
        return fallback
    out = []
    for i, v in enumerate(raw):
        if v is None:
            out.append(float(fallback[i].item()))
        else:
            try:
                out.append(float(int(v)))
            except (TypeError, ValueError):
                out.append(float(fallback[i].item()))
    return torch.tensor(out, device=target.device, dtype=torch.float32)


def boltz2_affinity_loss(
    pred,  # noqa: ANN001
    batch,  # noqa: ANN001
    adapter_meta=None,  # noqa: ANN001
    *,
    delta: float = 1.0,
    pair_weight: float = 2.0,
    bce_weight: float = 1.0,
):
    """Boltz-2 paper multi-task affinity loss.

    Combines the two terms used to pre-train the public Boltz-2 affinity
    heads, so a LoRA adapter trained with this loss is the direct
    finetuning analogue of the vanilla checkpoint:

    .. math::
        \\mathcal{L} \;=\; \\mathcal{L}_{\\text{value}}
                       \;+\; w_{\\text{bce}} \\, \\mathcal{L}_{\\text{bce}}

    where ``L_value`` is :func:`intra_assay_huber_loss` (absolute Huber on
    ``affinity_pred_value`` plus an intra-assay pairwise Huber, weight
    ``pair_weight``) and ``L_bce`` is binary cross-entropy on
    ``affinity_logits_binary`` against the binder/non-binder label.

    Binder labels are resolved per row from ``batch['is_binder']`` if
    present, else by thresholding ``target`` at
    ``BOLTZ_LORA_BINDER_THRESHOLD`` (default 1.0; ≤10 µM on the
    ``log10(IC50_uM)`` scale).

    Notes
    -----
    The original paper trains the BCE branch with sampled decoy negatives
    drawn against each receptor.  This loss does *not* do that
    sampling itself — it consumes whatever positive/negative split exists
    in the manifest.  For a faithful reproduction, mix decoy rows
    (``target`` set to a large log10 µM value or ``is_binder=0``) into
    the training CSV upstream.
    """
    value = intra_assay_huber_loss(
        pred, batch, adapter_meta, delta=delta, pair_weight=pair_weight,
    )

    if "affinity_logits_binary" not in pred:
        # Affinity forward didn't return the binary logits (e.g. partial
        # forward in a custom hook).  Fall back to value-only.
        return value

    logits = pred["affinity_logits_binary"]
    if logits.dim() > 1 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    logits = logits.float()

    target = _target(batch, device=logits.device)
    labels = _binder_labels(batch, target)
    bce = F.binary_cross_entropy_with_logits(logits, labels)

    return value + bce_weight * bce


BUILTIN_LOSSES: dict[str, LossFn] = {
    "mse": mse_loss,
    "mae": mae_loss,
    "huber": huber_loss,
    "bce": bce_loss,
    "pairwise_ranking": pairwise_ranking_loss,
    "intra_assay_huber": intra_assay_huber_loss,
    "censored_intra_assay_huber": censored_intra_assay_huber_loss,
    "censored_boltz2_affinity": censored_boltz2_affinity_loss,
    "boltz2_affinity": boltz2_affinity_loss,
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
    "boltz2_affinity_loss",
    "call_loss",
    "censored_boltz2_affinity_loss",
    "censored_intra_assay_huber_loss",
    "huber_loss",
    "intra_assay_huber_loss",
    "load_loss_from_spec",
    "mae_loss",
    "mse_loss",
    "pairwise_ranking_loss",
]
