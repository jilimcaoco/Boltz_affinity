"""
interpretability/hooks.py
=========================
Forward-hook instrumentation for AffinityModule.

ASSUMPTION: All hooks operate under the assumption that torch.no_grad() is
active during capture. Stored tensors are .detach().clone() — they carry no
gradient graph and do not extend tensor lifetime beyond the hook call.

Attention-weight access
-----------------------
Per-head softmax attention weights are captured by hooking
    layer.tri_att_start.mha.softmax   (a _SoftmaxWeights sub-module)
    layer.tri_att_end.mha.softmax

This is made possible by a minimal, prediction-neutral change to
boltz/model/layers/triangular_attention/primitives.py:

  * A parameter-free _SoftmaxWeights(nn.Module) wraps softmax_no_cast.
  * Attention.__init__ adds self.softmax = _SoftmaxWeights().
  * Attention.forward inlines the non-kernel _attention() free-function call,
    routing the softmax step through self.softmax — numerically identical.

The hook fires at that sub-module boundary and receives the full weight tensor
of shape [*, H, Q, K] = (B, N, H, N, N), which is then split along the head
dimension.  Chunks: if chunk_size_tri_attn is active, Attention.forward is
called once per chunk, so the hook fires multiple times and only the last
chunk's weights are retained.  For the ligand-sized N values typical in
affinity prediction, chunking is not triggered.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from boltz.model.modules.affinity import AffinityModule


class InstrumentedAffinityModule(nn.Module):
    """Wraps an AffinityModule and registers forward hooks on every
    PairformerNoSeqLayer in the pairformer_stack.

    Parameters
    ----------
    module : AffinityModule
        The module to observe.  Stored as self.module; never modified.

    Attributes
    ----------
    captured_z : dict[int, torch.Tensor]
        Populated after each forward call.
        Maps layer_idx -> pair representation z exiting that block.
        Shape: (B, N, N, token_z).

    captured_attn : dict[int, dict[str, dict[int, torch.Tensor]]]
        Populated after each forward call.
        Maps layer_idx -> {
            'tri_att_start': {0: Tensor, 1: Tensor, ...},
            'tri_att_end':   {0: Tensor, 1: Tensor, ...},
        }
        Each per-head tensor has shape (B, N, N, N):
            B  = batch
            N  = token sequence length (first N: "row" being attended from)
            N  = sequence positions (Q dimension)
            N  = sequence positions (K dimension)
        These are the softmax weights BEFORE value aggregation, captured at the
        _SoftmaxWeights module boundary inside Attention.
    """

    def __init__(self, module: AffinityModule) -> None:
        super().__init__()
        # Hold a reference to the wrapped module — do NOT subclass.
        self.module = module

        self.captured_z: dict[int, torch.Tensor] = {}
        self.captured_attn: dict[int, dict[str, dict[int, torch.Tensor]]] = {}

        # Removable hook handles.
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        self._num_layers: int = 0
        self._num_heads: int | None = None

        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        layers = self.module.pairformer_stack.layers
        self._num_layers = len(layers)

        for layer_idx, layer in enumerate(layers):

            # 1. Block-level hook ----------------------------------------
            # Fires after PairformerNoSeqLayer.forward completes.
            # output is z: (B, N, N, token_z).

            def _make_layer_hook(idx: int):
                def _layer_hook(
                    mod: nn.Module,
                    inputs: tuple,
                    output: torch.Tensor,
                ) -> None:
                    self.captured_z[idx] = output.detach().clone()

                return _layer_hook

            self._hooks.append(
                layer.register_forward_hook(_make_layer_hook(layer_idx))
            )

            # 2. Triangle attention softmax hooks ------------------------
            # Hook layer.tri_att_{start,end}.mha.softmax — the _SoftmaxWeights
            # sub-module added to Attention.  Its output is the per-head weight
            # tensor of shape [*, H, Q, K] before value aggregation.
            # We split along the H dimension and store each head separately.

            def _make_softmax_hook(idx: int, name: str):
                def _softmax_hook(
                    mod: nn.Module,
                    inputs: tuple,
                    output: torch.Tensor,
                ) -> None:
                    # output: [*, H, Q, K]
                    # For triangle attention: [B, N, H, N, N]
                    # H is at dim -3.
                    n_heads = output.shape[-3]
                    if idx not in self.captured_attn:
                        self.captured_attn[idx] = {}
                    self.captured_attn[idx][name] = {
                        h: output[..., h, :, :].detach().clone()
                        for h in range(n_heads)
                    }

                return _softmax_hook

            self._hooks.append(
                layer.tri_att_start.mha.softmax.register_forward_hook(
                    _make_softmax_hook(layer_idx, "tri_att_start")
                )
            )
            self._hooks.append(
                layer.tri_att_end.mha.softmax.register_forward_hook(
                    _make_softmax_hook(layer_idx, "tri_att_end")
                )
            )

        # Derive head count for repr.
        if self._num_layers > 0:
            try:
                self._num_heads = layers[0].tri_att_start.mha.no_heads
            except AttributeError:
                self._num_heads = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Empty both capture dicts."""
        self.captured_z.clear()
        self.captured_attn.clear()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Clear captures, run the wrapped module, return output unchanged."""
        self.clear()
        return self.module.forward(*args, **kwargs)

    def remove_hooks(self) -> None:
        """Remove all registered hooks (useful for cleanup after profiling)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        heads_str = str(self._num_heads) if self._num_heads is not None else "unknown"

        # Summarise what is currently stored.
        z_keys = sorted(self.captured_z.keys())
        attn_summary = {
            k: list(v.keys()) for k, v in sorted(self.captured_attn.items())
        }

        return (
            f"InstrumentedAffinityModule(\n"
            f"  layers hooked       : {self._num_layers}\n"
            f"  heads per tri_att   : {heads_str}\n"
            f"  total hooks         : {len(self._hooks)} "
            f"(3 per layer: 1 block-z + 2 softmax)\n"
            f"  captured_z keys     : {z_keys or 'empty'}\n"
            f"  captured_attn keys  : {attn_summary or 'empty'}\n"
            f"  attn tensor shape   : (B, N, N, N) per head "
            f"[B=batch, N=seq, Q×K attention]\n"
            f"  capture point       : softmax output BEFORE value aggregation\n"
            f"  hook path           : layer.tri_att_{{start,end}}.mha.softmax\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test — instantiate with a dummy AffinityModule and print repr.
# Does NOT run a forward pass.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy = AffinityModule(
        token_s=128,
        token_z=128,
        pairformer_args={
            "num_blocks": 4,
            "dropout": 0.0,
            "pairwise_head_width": 32,
            "pairwise_num_heads": 4,
        },
        transformer_args={
            "token_s": 128,
            "num_blocks": 2,
            "num_heads": 4,
            "activation_checkpointing": False,
        },
    )

    instrumented = InstrumentedAffinityModule(dummy)
    print(instrumented)
