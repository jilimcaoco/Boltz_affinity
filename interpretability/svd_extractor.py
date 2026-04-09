"""
interpretability/svd_extractor.py
==================================
SVD decomposition of attention and multiplicative update circuits.

This module extracts the query-key (QK) and output-value (OV) circuits from
triangle attention heads, and the multiplicative gating circuits, then
decomposes them via SVD to identify principal weight directions.
"""

from __future__ import annotations

import torch
from torch import nn

from boltz.model.modules.affinity import AffinityModule


def extract_head_circuits(
    module: AffinityModule,
    layer_idx: int,
    head_idx: int,
) -> dict:
    """Extract and SVD-decompose the QK and OV circuits of a single triangle
    attention head.

    Parameters
    ----------
    module : AffinityModule
        The parent module containing the pairformer_stack.
    layer_idx : int
        Index of the PairformerNoSeqLayer block.
    head_idx : int
        Index of the attention head within the triangle attention module.

    Returns
    -------
    dict
        Containing:
          - qk_U, qk_S, qk_Vh: SVD components of W_QK (d_pair × d_pair)
          - ov_U, ov_S, ov_Vh: SVD components of W_OV (d_pair × d_pair)
          - spectrum_ratio: scalar = ov_S[0] / ov_S[1]
          - top_qk_query_dir: first row of qk_Vh (query direction)
          - top_qk_key_dir: first column of qk_U (key direction)
          - top_ov_input_dir: first row of ov_Vh (input direction)
          - top_ov_output_dir: first column of ov_U (output direction)
    """
    # Grab the triangle attention module.
    # Structure: layer.tri_att_{start,end} is the TriangleAttention;
    #            .mha is the Attention sub-module inside it.
    layer = module.pairformer_stack.layers[layer_idx]
    tri_att = layer.tri_att_start  # or tri_att_end; both have same structure
    attn_mha = tri_att.mha

    # Extract per-head projection weights.
    # The Attention class has:
    #   linear_q: nn.Linear(c_q, c_hidden * no_heads, bias=False)
    #   linear_k: nn.Linear(c_k, c_hidden * no_heads, bias=False)
    #   linear_v: nn.Linear(c_v, c_hidden * no_heads, bias=False)
    #   linear_o: nn.Linear(c_hidden * no_heads, c_q, bias=False)
    #
    # Each weight has shape (out_dim, in_dim); we slice along the output dim
    # to extract per-head chunks.  Per-head chunk has shape (c_hidden, in_dim).

    c_hidden = attn_mha.c_hidden
    c_q = attn_mha.c_q
    no_heads = attn_mha.no_heads

    # W_Q: (c_hidden * no_heads, c_q)  →  slice for head_idx
    W_Q = attn_mha.linear_q.weight[
        head_idx * c_hidden : (head_idx + 1) * c_hidden, :
    ]  # (c_hidden, c_q)

    # W_K: (c_hidden * no_heads, c_k)  →  slice for head_idx
    W_K = attn_mha.linear_k.weight[
        head_idx * c_hidden : (head_idx + 1) * c_hidden, :
    ]  # (c_hidden, c_k)

    # W_V: (c_hidden * no_heads, c_v)  →  slice for head_idx
    W_V = attn_mha.linear_v.weight[
        head_idx * c_hidden : (head_idx + 1) * c_hidden, :
    ]  # (c_hidden, c_v)

    # W_O: (c_q, c_hidden * no_heads)  →  slice for head_idx
    W_O = attn_mha.linear_o.weight[:, head_idx * c_hidden : (head_idx + 1) * c_hidden]
    # (c_q, c_hidden)

    # Compute QK and OV circuits in pair-representation space (c_q × c_q).
    #
    # QK circuit: attention score = x^T W_Q^T W_K x
    #   W_QK = W_Q^T @ W_K  →  (c_q, c_q)
    #
    # OV circuit: output contribution = W_O @ W_V @ x
    #   W_OV = W_O @ W_V  →  (c_q, c_q)

    W_QK = W_Q.T @ W_K  # (c_q, c_q)
    W_OV = W_O @ W_V    # (c_q, c_q)

    # Run SVD decomposition (full matrices).
    qk_U, qk_S, qk_Vh = torch.linalg.svd(W_QK, full_matrices=True)
    ov_U, ov_S, ov_Vh = torch.linalg.svd(W_OV, full_matrices=True)

    # Extract principal directions.
    # SVD: W_QK = U S Vh.  Score = z_i^T W_QK z_j, so position i (query)
    # aligns with U[:,0] and position j (key) aligns with Vh[0,:].
    top_qk_query_dir = qk_U[:, 0]   # left singular vector = query direction
    top_qk_key_dir = qk_Vh[0, :]    # right singular vector = key direction
    top_ov_input_dir = ov_Vh[0, :]  # First row of right-singular vectors
    top_ov_output_dir = ov_U[:, 0]  # First column of left-singular vectors

    # Spectrum ratio: ratio of first to second singular value of OV.
    spectrum_ratio = (ov_S[0] / ov_S[1].clamp(min=1e-12)).item() if len(ov_S) > 1 else float("inf")

    return {
        "qk_U": qk_U.detach(),
        "qk_S": qk_S.detach(),
        "qk_Vh": qk_Vh.detach(),
        "ov_U": ov_U.detach(),
        "ov_S": ov_S.detach(),
        "ov_Vh": ov_Vh.detach(),
        "spectrum_ratio": spectrum_ratio,
        "top_qk_query_dir": top_qk_query_dir.detach(),
        "top_qk_key_dir": top_qk_key_dir.detach(),
        "top_ov_input_dir": top_ov_input_dir.detach(),
        "top_ov_output_dir": top_ov_output_dir.detach(),
    }


def extract_triangle_mult_circuits(
    module: AffinityModule,
    layer_idx: int,
    direction: str,
) -> dict:
    """Extract and SVD-decompose the multiplicative gating circuits of a
    triangle multiplicative update.

    Parameters
    ----------
    module : AffinityModule
        The parent module.
    layer_idx : int
        Index of the PairformerNoSeqLayer block.
    direction : str
        Either "outgoing" or "incoming" to select TriangleMultiplicationOutgoing
        or TriangleMultiplicationIncoming.

    Returns
    -------
    dict
        Containing:
          - p_in_U, p_in_S, p_in_Vh: SVD of p_in.weight (2*d_pair × d_pair)
          - g_in_U, g_in_S, g_in_Vh: SVD of g_in.weight (2*d_pair × d_pair)
          - p_out_U, p_out_S, p_out_Vh: SVD of p_out.weight (d_pair × d_pair)
          - g_out_U, g_out_S, g_out_Vh: SVD of g_out.weight (d_pair × d_pair)
          - top_pin_dir, top_gin_dir, top_pout_dir, top_gout_dir:
            first singular vectors (principal directions)
    """
    layer = module.pairformer_stack.layers[layer_idx]

    # Select the appropriate triangular multiplication module.
    if direction.lower() == "outgoing":
        tri_mult = layer.tri_mul_out
    elif direction.lower() == "incoming":
        tri_mult = layer.tri_mul_in
    else:
        raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction}")

    # Extract weights from the gating and projection layers.
    # TriangleMultiplication{Outgoing,Incoming} have:
    #   p_in:   nn.Linear(dim, 2*dim, bias=False)  →  weight (2*dim, dim)
    #   g_in:   nn.Linear(dim, 2*dim, bias=False)  →  weight (2*dim, dim)
    #   p_out:  nn.Linear(dim, dim, bias=False)    →  weight (dim, dim)
    #   g_out:  nn.Linear(dim, dim, bias=False)    →  weight (dim, dim)

    p_in_weight = tri_mult.p_in.weight  # (2*dim, dim)
    g_in_weight = tri_mult.g_in.weight  # (2*dim, dim)
    p_out_weight = tri_mult.p_out.weight  # (dim, dim)
    g_out_weight = tri_mult.g_out.weight  # (dim, dim)

    # Run SVD on each.
    p_in_U, p_in_S, p_in_Vh = torch.linalg.svd(p_in_weight, full_matrices=True)
    g_in_U, g_in_S, g_in_Vh = torch.linalg.svd(g_in_weight, full_matrices=True)
    p_out_U, p_out_S, p_out_Vh = torch.linalg.svd(p_out_weight, full_matrices=True)
    g_out_U, g_out_S, g_out_Vh = torch.linalg.svd(g_out_weight, full_matrices=True)

    return {
        "p_in_U": p_in_U.detach(),
        "p_in_S": p_in_S.detach(),
        "p_in_Vh": p_in_Vh.detach(),
        "g_in_U": g_in_U.detach(),
        "g_in_S": g_in_S.detach(),
        "g_in_Vh": g_in_Vh.detach(),
        "p_out_U": p_out_U.detach(),
        "p_out_S": p_out_S.detach(),
        "p_out_Vh": p_out_Vh.detach(),
        "g_out_U": g_out_U.detach(),
        "g_out_S": g_out_S.detach(),
        "g_out_Vh": g_out_Vh.detach(),
        "top_pin_dir": p_in_Vh[0, :].detach(),
        "top_gin_dir": g_in_Vh[0, :].detach(),
        "top_pout_dir": p_out_Vh[0, :].detach(),
        "top_gout_dir": g_out_Vh[0, :].detach(),
    }
