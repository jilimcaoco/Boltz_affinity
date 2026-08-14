"""Parametric loss wrappers for the boltz2_affinity_loss delta scan.

Each function wraps :func:`boltz.lora.losses.boltz2_affinity_loss` with a
fixed ``delta`` value (the Huber transition point between quadratic and linear
penalty regions, in log10(IC50_uM) units).

The SLURM training scripts select the appropriate function via the
``--loss path/to/delta_scan_losses.py:<fn_name>`` spec, and set the
``BOLTZ_LORA_DELTA`` environment variable for documentation / provenance only.

Delta values scanned: 1.00, 0.75, 0.50, 0.25, 0.00
  * delta=1.00 — standard Huber, quadratic below 1 log-unit error (~Boltz-2
                 paper default).
  * delta=0.50 — tighter quadratic zone; penalises outliers more linearly.
  * delta=0.00 — pure MAE (L1); the smooth_l1_loss beta=0 limit reduces to
                 absolute error, most robust to outliers.
"""
from __future__ import annotations

import functools
from typing import Any, Dict, Optional

import torch

from boltz.lora.losses import boltz2_affinity_loss as _base


def _make(delta: float):
    """Return a boltz2_affinity_loss variant with fixed *delta*."""
    def _fn(
        pred: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        adapter_meta: Optional[Any] = None,
    ) -> torch.Tensor:
        return _base(pred, batch, adapter_meta, delta=delta)
    _fn.__name__ = f"boltz2_affinity_delta_{str(delta).replace('.', 'p')}"
    _fn.__doc__ = f"boltz2_affinity_loss with delta={delta}."
    return _fn


boltz2_affinity_delta_1p00 = _make(1.00)
boltz2_affinity_delta_0p75 = _make(0.75)
boltz2_affinity_delta_0p50 = _make(0.50)
boltz2_affinity_delta_0p25 = _make(0.25)
boltz2_affinity_delta_0p00 = _make(0.00)

# Convenience mapping used by run_scan.sh to resolve fn name from a delta string.
DELTA_FN_MAP: dict[str, str] = {
    "1.00": "boltz2_affinity_delta_1p00",
    "0.75": "boltz2_affinity_delta_0p75",
    "0.50": "boltz2_affinity_delta_0p50",
    "0.25": "boltz2_affinity_delta_0p25",
    "0.00": "boltz2_affinity_delta_0p00",
}
