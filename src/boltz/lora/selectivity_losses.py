r"""Cross-receptor selectivity losses for the Boltz-2 affinity head.

Motivation
----------
The stock affinity losses in :mod:`boltz.lora.losses` fit *potency*: an
absolute Huber on ``log10(IC50_uM)`` plus an intra-assay pairwise Huber over
pairs of **ligands measured against one receptor**.  Restricting the pairwise
term to one assay is what cancels the additive assay offset and lets Ki / Kd /
IC50 be trained jointly.

Selectivity is the same idea rotated ninety degrees: pairs of **receptors
measured with one ligand**.  Scoring the same compound against both receptors
makes the difference

.. math::  \\hat{\\Delta} = \\hat{y}_A - \\hat{y}_B

invariant to every *ligand-global* nuisance the model carries — molecular
weight, size, lipophilicity, and any ligand-level calibration error.  Boltz-2's
inference-time MW correction
``1.03525938 * raw - 0.59992683 * mw**0.3 + 2.83288489`` cancels exactly in
:math:`\\hat{\\Delta}` (same ligand, same MW), up to the ``1.035`` scale, so the
selectivity readout is unaffected by whether that correction is applied.

Two things this module deliberately does **not** do
---------------------------------------------------
All terms are zero exactly when the prediction matches the measurement, so a
training curve reads as model quality rather than as batch composition.

*It does not reward a large gap.*  A term such as ``-lambda * |y_A - y_B|``
gated on a large experimental difference is minimised at infinity: nothing
anchors the magnitude, so the equilibrium is set by the weight ratio rather
than by the data, and predictions inflate without the *ordering* improving.
The cross-receptor term here is a Huber **regression of the predicted
difference onto the measured difference**, which is a proper scoring rule —
minimised at the truth.  Extra emphasis on the selective tail is available as
a sample weight (``sel_tail_gamma``), which reweights examples without moving
the optimum.

*It is not gated on large differences.*  A decision boundary needs the
non-selective compounds too; if only large-|Δ| pairs contributed, nothing would
push :math:`\\hat{\\Delta}` toward zero for equipotent ligands.  Every complete
pair contributes.

Note on arm-swap augmentation
-----------------------------
Presenting each pair in both orders is a no-op for these terms and is therefore
not implemented.  The Huber is even, so
``Huber(-x) == Huber(x)``, and soft-label BCE satisfies
``BCE(sigmoid(-a), sigmoid(-b)) == BCE(sigmoid(a), sigmoid(b))``.  Swapping the
arms negates both the prediction and the target, leaving both terms unchanged.
Arm-swap augmentation only buys something for an *asymmetric* difference head,
which this module does not add.  Pair orientation is instead fixed
deterministically (see ``reference_receptor``) so the sign convention is
stable across runs.

Batch requirements
------------------
``batch`` must carry, per row, the keys produced by
:func:`boltz.lora.data.lora_collate`:

* ``target``      — ``log10(Ki or IC50 in uM)``, lower = stronger.
* ``pair_id``     — shared by both arms of one ligand.
* ``receptor_id`` — which arm the row is.
* ``group_id``    — the panel / source shared by both arms.
* ``is_censored`` — 0/1, right-censored ("> X") measurements.

Use :class:`boltz.lora.data.PairedReceptorSampler` so both arms of a pair land
in the same mini-batch; the stock :class:`~boltz.lora.data.AssayGroupedSampler`
fills a batch from a single ``group_id`` and therefore can never produce one.

Censoring
---------
On the ``log10`` scale a right-censored row means the true target is *at least*
the reported value.  For a directed pair ``Δ = y_A - y_B``:

* ``B`` censored  → true ``Δ <= Δ_reported`` → penalise only ``relu(Δ̂ - Δ_rep)``.
* ``A`` censored  → true ``Δ >= Δ_reported`` → penalise only ``relu(Δ_rep - Δ̂)``.
* both censored   → skipped, no usable signal.

The ranking term keeps a censored pair only when the censoring direction
*reinforces* the observed sign (``B`` censored and ``Δ_rep <= 0``, or ``A``
censored and ``Δ_rep >= 0``); otherwise the sign is not determined by the data
and the pair is dropped.  For retained censored pairs the reported ``Δ`` gives
a conservative soft target, since the true ``|Δ|`` is at least as large.

Usage
-----
Every term is independently weighted, so one implementation covers joint
training and either objective alone::

    boltz lora train ... --loss <path>/selectivity_losses.py:selectivity_joint

Presets: ``selectivity_joint``, ``selectivity_rank_only``,
``selectivity_delta_only``, ``selectivity_off``.  With ``w_sel=0`` and
``w_rank=0`` the loss reduces exactly to
:func:`boltz.lora.losses.censored_boltz2_affinity_loss`, which makes the
no-selectivity control free of any implementation difference.

Every keyword can also be overridden from the environment with the
``BOLTZ_SELECTIVITY_<NAME>`` prefix (e.g. ``BOLTZ_SELECTIVITY_W_SEL=0.5``),
matching the existing ``BOLTZ_LORA_BINDER_THRESHOLD`` convention, so sweeps do
not need a new loss file per point.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
import torch.nn.functional as F

from boltz.lora.losses import _binder_labels, _get_censored_mask, _pred_value, _target

__all__ = [
    "make_selectivity_loss",
    "selectivity_delta_only",
    "selectivity_joint",
    "selectivity_off",
    "selectivity_rank_only",
]

_ENV_PREFIX = "BOLTZ_SELECTIVITY_"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(_ENV_PREFIX + name.upper())
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _as_list(batch: dict[str, Any], key: str, n: int) -> list[Any]:
    """Read a per-row column as a plain python list of length ``n``."""
    raw = batch.get(key)
    if raw is None:
        return [None] * n
    if torch.is_tensor(raw):
        return raw.detach().cpu().tolist()
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw] * n


def _huber(x: torch.Tensor, y: torch.Tensor, delta: float) -> torch.Tensor:
    return F.smooth_l1_loss(x, y, beta=delta, reduction="sum")


def _one_sided(viol: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber applied to a violation magnitude that is zero when satisfied."""
    return F.smooth_l1_loss(viol, torch.zeros_like(viol), beta=delta, reduction="sum")


# ── Term 1: censor-aware absolute (point) Huber ──────────────────────────────


def _point_term(
    p: torch.Tensor, t: torch.Tensor, censored: torch.Tensor, delta: float
) -> torch.Tensor:
    """Absolute Huber; censored rows penalised only for over-predicting potency."""
    n = p.numel()
    if n == 0:
        return torch.zeros((), device=p.device)
    total = torch.zeros((), device=p.device)
    unc = (~censored).nonzero(as_tuple=True)[0]
    cen = censored.nonzero(as_tuple=True)[0]
    if unc.numel():
        total = total + _huber(p.index_select(0, unc), t.index_select(0, unc), delta)
    if cen.numel():
        # target is a lower bound: only "predicted too potent" violates it.
        viol = F.relu(t.index_select(0, cen) - p.index_select(0, cen))
        total = total + _one_sided(viol, delta)
    return total / n


# ── Term 2: ligand-axis pairwise Huber, keyed on (group_id, receptor_id) ─────


def _same_key_pairs(keys: Sequence[Any]) -> tuple[list[int], list[int]]:
    left: list[int] = []
    right: list[int] = []
    for i, ki in enumerate(keys):
        if ki is None:
            continue
        for j in range(i + 1, len(keys)):
            if keys[j] == ki:
                left.append(i)
                right.append(j)
    return left, right


def _pairwise_term(
    p: torch.Tensor,
    t: torch.Tensor,
    censored: torch.Tensor,
    keys: Sequence[Any],
    delta: float,
) -> torch.Tensor:
    """Censor-aware Huber on differences between rows sharing ``keys``."""
    left, right = _same_key_pairs(keys)
    if not left:
        return torch.zeros((), device=p.device)

    cl = censored.tolist()
    keep_l: list[int] = []
    keep_r: list[int] = []
    mode: list[int] = []  # 0 = both known, 1 = right censored, 2 = left censored
    for i, j in zip(left, right):
        ci, cj = bool(cl[i]), bool(cl[j])
        if ci and cj:
            continue
        keep_l.append(i)
        keep_r.append(j)
        mode.append(0 if not ci and not cj else (1 if cj else 2))
    if not keep_l:
        return torch.zeros((), device=p.device)

    pl = torch.tensor(keep_l, device=p.device, dtype=torch.long)
    pr = torch.tensor(keep_r, device=p.device, dtype=torch.long)
    m = torch.tensor(mode, device=p.device, dtype=torch.long)
    dp = p.index_select(0, pl) - p.index_select(0, pr)
    dt = t.index_select(0, pl) - t.index_select(0, pr)

    total = torch.zeros((), device=p.device)
    both = m == 0
    if bool(both.any()):
        total = total + _huber(dp[both], dt[both], delta)
    rc = m == 1
    if bool(rc.any()):
        total = total + _one_sided(F.relu(dp[rc] - dt[rc]), delta)
    lc = m == 2
    if bool(lc.any()):
        total = total + _one_sided(F.relu(dt[lc] - dp[lc]), delta)
    return total / len(keep_l)


# ── Term 3+4: cross-receptor selectivity regression and soft ranking ─────────


def _orient(
    i: int,
    j: int,
    receptor_ids: Sequence[Any],
    reference_receptor: Optional[str],
) -> Optional[tuple[int, int]]:
    """Order one candidate pair, or ``None`` if it is not a cross-receptor pair."""
    ri, rj = receptor_ids[i], receptor_ids[j]
    if ri == rj:
        # Same receptor twice under one pair_id: a replicate, not a
        # cross-receptor pair. Skip rather than invent a direction.
        return None
    if reference_receptor is not None:
        if rj == reference_receptor and ri != reference_receptor:
            return j, i
        return i, j
    if ri is not None and rj is not None and str(rj) < str(ri):
        return j, i
    return i, j


def _build_directed_pairs(
    pair_ids: Sequence[Any],
    receptor_ids: Sequence[Any],
    reference_receptor: Optional[str],
) -> tuple[list[int], list[int]]:
    """Index pairs ``(a, b)`` with a deterministic receptor orientation.

    ``a`` is the row whose ``receptor_id`` equals ``reference_receptor`` when
    given, otherwise the lexicographically smaller ``receptor_id``.  A stable
    orientation keeps the sign of ``Δ`` consistent across batches, epochs and
    runs, which matters because the ranking term is not sign-symmetric across
    *different* pairs even though it is symmetric within one.
    """
    buckets: dict[Any, list[int]] = {}
    for i, pid in enumerate(pair_ids):
        if pid is None or pid == "":
            continue
        buckets.setdefault(pid, []).append(i)

    left: list[int] = []
    right: list[int] = []
    for idxs in buckets.values():
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                oriented = _orient(
                    idxs[x], idxs[y], receptor_ids, reference_receptor
                )
                if oriented is None:
                    continue
                left.append(oriented[0])
                right.append(oriented[1])
    return left, right


def _selectivity_terms(
    p: torch.Tensor,
    t: torch.Tensor,
    censored: torch.Tensor,
    pair_ids: Sequence[Any],
    receptor_ids: Sequence[Any],
    *,
    delta: float,
    tau: float,
    sel_tail_gamma: float,
    reference_receptor: Optional[str],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``(delta_regression, soft_rank, n_pairs)``."""
    zero = torch.zeros((), device=p.device)
    left, right = _build_directed_pairs(pair_ids, receptor_ids, reference_receptor)
    if not left:
        return zero, zero, 0

    cl = censored.tolist()
    reg_l: list[int] = []
    reg_r: list[int] = []
    reg_mode: list[int] = []
    for i, j in zip(left, right):
        ci, cj = bool(cl[i]), bool(cl[j])
        if ci and cj:
            continue
        reg_l.append(i)
        reg_r.append(j)
        reg_mode.append(0 if not ci and not cj else (1 if cj else 2))
    if not reg_l:
        return zero, zero, 0

    la = torch.tensor(reg_l, device=p.device, dtype=torch.long)
    lb = torch.tensor(reg_r, device=p.device, dtype=torch.long)
    m = torch.tensor(reg_mode, device=p.device, dtype=torch.long)
    d_pred = p.index_select(0, la) - p.index_select(0, lb)
    d_true = t.index_select(0, la) - t.index_select(0, lb)

    # Sample weight emphasising the selective tail without moving the optimum.
    w = 1.0 + sel_tail_gamma * d_true.detach().abs()

    # -- Δ regression (censor-aware, per-example weighted) --
    both = m == 0
    rc = m == 1  # B censored: true Δ <= Δ_rep
    lc = m == 2  # A censored: true Δ >= Δ_rep
    resid = torch.zeros_like(d_pred)
    resid[both] = d_pred[both] - d_true[both]
    resid[rc] = F.relu(d_pred[rc] - d_true[rc])
    resid[lc] = F.relu(d_true[lc] - d_pred[lc])
    per_ex = F.smooth_l1_loss(
        resid, torch.zeros_like(resid), beta=delta, reduction="none"
    )
    reg = (per_ex * w).sum() / w.sum().clamp(min=1e-8)

    # -- Soft ranking: keep a censored pair only where the bound reinforces
    #    the observed sign, so the label is actually determined by the data. --
    keep = both | (rc & (d_true <= 0)) | (lc & (d_true >= 0))
    if not bool(keep.any()):
        return reg, zero, len(reg_l)
    a = d_pred[keep] / tau
    z = d_true[keep].detach() / tau
    q = torch.sigmoid(z)
    # KL(q || sigmoid(a)) rather than plain cross-entropy.  Soft-label BCE
    # bottoms out at the binary entropy H(q), not at zero, and H(q) depends on
    # which pairs happen to be in the batch — so a raw cross-entropy makes the
    # logged loss curve move with batch composition rather than with model
    # quality.  Subtracting H(q) is constant w.r.t. the predictions, so the
    # gradients are identical, but the term is now zero exactly when the
    # predicted difference matches the measured one.
    #
    # Stable forms: log sigmoid(z) = -softplus(-z), log(1 - sigmoid(z)) = -softplus(z).
    ce = q * F.softplus(-a) + (1.0 - q) * F.softplus(a)
    entropy = q * F.softplus(-z) + (1.0 - q) * F.softplus(z)
    rank = (ce - entropy).mean()
    return reg, rank, len(reg_l)


# ── Public factory ───────────────────────────────────────────────────────────


def make_selectivity_loss(
    *,
    delta: float = 1.0,
    w_point: float = 1.0,
    w_lig: float = 2.0,
    w_sel: float = 1.0,
    w_rank: float = 1.0,
    bce_weight: float = 1.0,
    tau: float = 1.0,
    sel_tail_gamma: float = 0.0,
    reference_receptor: Optional[str] = None,
    read_env: bool = True,
) -> Callable[..., torch.Tensor]:
    """Build a selectivity loss with independently weighted terms.

    .. code-block:: text

        L = w_point * point(y_hat, y)                         # censor-aware
          + w_lig   * pairwise over (group_id, receptor_id)   # ligand axis
          + w_sel   * huber(delta_hat - delta_true)           # receptor axis
          + w_rank  * KL(sigmoid(delta_true/tau) || sigmoid(delta_hat/tau))
          + bce_weight * bce(binary_logits, is_binder)

    Parameters
    ----------
    delta:
        Huber transition point (MSE below, MAE above).
    w_point, w_lig, w_sel, w_rank, bce_weight:
        Term weights.  ``w_sel=0`` gives pure ranking, ``w_rank=0`` gives pure
        Δ regression, both positive gives the joint objective, and both zero
        reduces the loss exactly to ``censored_boltz2_affinity``.
    tau:
        Temperature of the soft ranking target, in log units.  Smaller makes
        the target closer to a hard 0/1 label.
    sel_tail_gamma:
        Per-example weight ``1 + gamma * |delta_true|`` on the Δ regression.
        ``0.0`` (default) weights every pair equally, which keeps the term an
        unweighted proper scoring rule.
    reference_receptor:
        ``receptor_id`` placed on the positive side of ``Δ``.  ``None`` orders
        each pair by lexicographic ``receptor_id``.
    read_env:
        Allow ``BOLTZ_SELECTIVITY_<NAME>`` environment overrides of the numeric
        keywords, for sweeps.

    Returns
    -------
    A ``loss_fn(pred, batch, adapter_meta=None) -> scalar`` matching the
    contract validated by :func:`boltz.lora.losses.load_loss_from_spec`.
    """
    if read_env:
        delta = _env_float("delta", delta)
        w_point = _env_float("w_point", w_point)
        w_lig = _env_float("w_lig", w_lig)
        w_sel = _env_float("w_sel", w_sel)
        w_rank = _env_float("w_rank", w_rank)
        bce_weight = _env_float("bce_weight", bce_weight)
        tau = _env_float("tau", tau)
        sel_tail_gamma = _env_float("sel_tail_gamma", sel_tail_gamma)
        reference_receptor = (
            os.environ.get(_ENV_PREFIX + "REFERENCE_RECEPTOR") or reference_receptor
        )
    if tau <= 0:
        msg = f"tau must be > 0, got {tau}"
        raise ValueError(msg)

    def selectivity_loss(pred, batch, adapter_meta=None) -> torch.Tensor:  # noqa: ANN001
        p = _pred_value(pred).float()
        t = _target(batch, device=p.device)
        n = p.numel()
        censored = _get_censored_mask(batch, n, p.device)

        total = w_point * _point_term(p, t, censored, delta)

        if w_lig and n >= 2:
            groups = _as_list(batch, "group_id", n)
            receptors = _as_list(batch, "receptor_id", n)
            # Key on (group_id, receptor_id): pairing rows across receptors
            # here would reintroduce the very offset this term cancels.
            keys = [
                None if not g else (g, r) for g, r in zip(groups, receptors)
            ]
            total = total + w_lig * _pairwise_term(p, t, censored, keys, delta)

        if (w_sel or w_rank) and n >= 2:
            reg, rank, _ = _selectivity_terms(
                p,
                t,
                censored,
                _as_list(batch, "pair_id", n),
                _as_list(batch, "receptor_id", n),
                delta=delta,
                tau=tau,
                sel_tail_gamma=sel_tail_gamma,
                reference_receptor=reference_receptor,
            )
            total = total + w_sel * reg + w_rank * rank

        if bce_weight and "affinity_logits_binary" in pred:
            logits = pred["affinity_logits_binary"]
            if logits.dim() > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            logits = logits.float()
            labels = _binder_labels(batch, _target(batch, device=logits.device))
            total = total + bce_weight * F.binary_cross_entropy_with_logits(
                logits, labels
            )

        return total

    return selectivity_loss


# ── Named presets (pre-bound so they match the (pred, batch, meta) contract) ──

selectivity_joint = make_selectivity_loss()
selectivity_rank_only = make_selectivity_loss(w_sel=0.0, w_rank=1.0)
selectivity_delta_only = make_selectivity_loss(w_sel=1.0, w_rank=0.0)
selectivity_off = make_selectivity_loss(w_sel=0.0, w_rank=0.0)
