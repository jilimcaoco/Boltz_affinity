"""Low-rank adapter linear layer.

LoRALinear wraps an existing ``nn.Linear`` (with or without bias). The base
weights are frozen; only the two low-rank factors A and B are trainable.

    y = base(x) + dropout(x) @ A^T @ B^T * (alpha / r)

A is initialised with Kaiming uniform, B with zeros, so that the adapter is
the identity at initialisation (matches the original LoRA paper).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` with an additive low-rank update.

    Parameters
    ----------
    base : nn.Linear
        The pre-existing layer to wrap. Its parameters are frozen in place.
    r : int
        Adapter rank. Must be > 0.
    alpha : float
        Scaling factor; effective scale is ``alpha / r``.
    dropout : float
        Dropout applied to the LoRA input branch only (0 disables).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            msg = f"LoRA rank must be positive, got {r}"
            raise ValueError(msg)

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r

        # Match base dtype/device so optimizer state stays on the right device.
        weight = base.weight
        factory = {"device": weight.device, "dtype": weight.dtype}

        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout: nn.Module = (
            nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        )

    @property
    def merged(self) -> bool:
        """Whether the adapter has been folded into the base weight."""
        return getattr(self, "_merged", False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + lora_out * self.scaling

    @torch.no_grad()
    def merge_(self) -> None:
        """Fold A,B into ``base.weight`` in-place; subsequent forwards skip LoRA."""
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        self._merged = True

    @torch.no_grad()
    def unmerge_(self) -> None:
        """Reverse :meth:`merge_`. Safe to call when not merged (no-op)."""
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.sub_(delta.to(self.base.weight.dtype))
        self._merged = False

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Return ONLY the trainable LoRA tensors (A,B)."""
        return {"lora_A": self.lora_A.detach().cpu(),
                "lora_B": self.lora_B.detach().cpu()}

    def load_lora_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.lora_A.copy_(state["lora_A"].to(self.lora_A))
            self.lora_B.copy_(state["lora_B"].to(self.lora_B))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, merged={self.merged}"
        )


def wrap_linear(
    layer: nn.Linear,
    r: int,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRALinear:
    """Factory used by :mod:`boltz.lora.inject`."""
    if not isinstance(layer, nn.Linear):
        msg = f"wrap_linear expected nn.Linear, got {type(layer).__name__}"
        raise TypeError(msg)
    return LoRALinear(layer, r=r, alpha=alpha, dropout=dropout)


__all__ = ["LoRALinear", "wrap_linear"]
