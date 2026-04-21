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


# -----------------------------------------------------------------------
# Activation-level SVD
# -----------------------------------------------------------------------


def extract_attention_svd(
    attention_weights: torch.Tensor,
    top_k: int = 5,
) -> dict:
    """SVD-decompose a captured attention pattern matrix.

    Parameters
    ----------
    attention_weights : torch.Tensor
        Shape ``(B, N, N)`` — the softmax attention weights for one head
        at one layer.  B is typically 1 for interpretability.
    top_k : int
        Number of singular triplets to return.

    Returns
    -------
    dict
        - ``singular_values``: top-k singular values (Tensor)
        - ``query_dirs``: (top_k, N) — which query positions dominate
        - ``key_dirs``: (top_k, N) — which key positions are attended to
        - ``effective_rank``: scalar — sum(S)^2 / sum(S^2), measures
          how concentrated the attention pattern is
        - ``top_query_idx``: list[int] — argmax of each left singular
          vector (which token position is the strongest "query")
        - ``top_key_idx``: list[int] — argmax of each right singular
          vector (which token position is the strongest "key")
    """
    # Use first batch element.
    A = attention_weights[0].float()  # (N, N)

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    k = min(top_k, len(S))
    S_top = S[:k]
    U_top = U[:, :k]   # (N, k)
    Vh_top = Vh[:k, :]  # (k, N)

    # Effective rank: (sum S)^2 / sum(S^2).  High = diffuse, low = concentrated.
    s_sum = S.sum()
    s_sq_sum = (S ** 2).sum()
    effective_rank = (s_sum ** 2 / s_sq_sum.clamp(min=1e-12)).item()

    top_query_idx = U_top.abs().argmax(dim=0).tolist()  # k entries
    top_key_idx = Vh_top.abs().argmax(dim=1).tolist()    # k entries

    return {
        "singular_values": S_top.detach(),
        "query_dirs": U_top.T.detach(),  # (k, N)
        "key_dirs": Vh_top.detach(),      # (k, N)
        "effective_rank": effective_rank,
        "top_query_idx": top_query_idx,
        "top_key_idx": top_key_idx,
    }


def extract_z_delta_svd(
    layer_z_dict: dict[int, torch.Tensor],
    layer_idx: int,
    interface_mask: torch.Tensor | None = None,
    top_k: int = 5,
) -> dict:
    """SVD-decompose the z update at a specific layer.

    Computes Δz = z_layer - z_{layer-1}, reshapes to (N*N, token_z),
    and decomposes to find the principal update directions in pair space.

    Parameters
    ----------
    layer_z_dict : dict[int, torch.Tensor]
        Mapping ``{layer_idx: z}`` from captured_z.  Must contain
        ``layer_idx`` and ``layer_idx - 1``.
    layer_idx : int
        The layer whose update to analyze (must be >= 0).
    interface_mask : torch.Tensor, optional
        Boolean mask ``(B, N, N)`` — if provided, only interface pairs
        contribute to the SVD.
    top_k : int
        Number of singular triplets to return.

    Returns
    -------
    dict
        - ``singular_values``: top-k singular values
        - ``channel_dirs``: (top_k, token_z) — principal directions in
          pair-representation channel space.  Compare with OV output
          directions from weight SVD.
        - ``spatial_dirs``: (top_k, N*N) — which (i,j) pairs received the
          largest updates along each principal direction.
        - ``effective_rank``: how many directions carry the update
        - ``top_pair_idx``: list[(i, j)] — pair with largest weight in
          each principal spatial direction
    """
    prev_key = layer_idx - 1
    if prev_key not in layer_z_dict or layer_idx not in layer_z_dict:
        raise ValueError(
            f"layer_z_dict must contain keys {prev_key} and {layer_idx}. "
            f"Available: {sorted(layer_z_dict.keys())}"
        )

    z_curr = layer_z_dict[layer_idx][0].float()   # (N, N, token_z)
    z_prev = layer_z_dict[prev_key][0].float()     # (N, N, token_z)
    delta = z_curr - z_prev                         # (N, N, token_z)

    N = delta.shape[0]

    # Optionally restrict to interface pairs only.
    if interface_mask is not None:
        mask_2d = interface_mask[0].bool()  # (N, N)
        delta_flat = delta[mask_2d]  # (n_pairs, token_z)
        pair_indices = mask_2d.nonzero(as_tuple=False)  # (n_pairs, 2)
    else:
        delta_flat = delta.reshape(N * N, -1)  # (N*N, token_z)
        pair_indices = None

    if delta_flat.shape[0] == 0:
        return {
            "singular_values": torch.zeros(0),
            "channel_dirs": torch.zeros(0, delta.shape[-1]),
            "spatial_dirs": torch.zeros(0, delta_flat.shape[0]),
            "effective_rank": 0.0,
            "top_pair_idx": [],
        }

    # SVD: delta_flat = U S Vh
    # U columns: spatial directions (which pairs update most)
    # Vh rows: channel directions (what direction in token_z space)
    U, S, Vh = torch.linalg.svd(delta_flat, full_matrices=False)

    k = min(top_k, len(S))
    S_top = S[:k]
    U_top = U[:, :k]    # (n_pairs, k)
    Vh_top = Vh[:k, :]   # (k, token_z)

    s_sum = S.sum()
    s_sq_sum = (S ** 2).sum()
    effective_rank = (s_sum ** 2 / s_sq_sum.clamp(min=1e-12)).item()

    # Find which (i, j) pair has largest weight in each spatial direction.
    top_pair_flat = U_top.abs().argmax(dim=0).tolist()  # k entries
    if pair_indices is not None:
        top_pair_idx = [
            (pair_indices[idx, 0].item(), pair_indices[idx, 1].item())
            for idx in top_pair_flat
        ]
    else:
        top_pair_idx = [(idx // N, idx % N) for idx in top_pair_flat]

    return {
        "singular_values": S_top.detach(),
        "channel_dirs": Vh_top.detach(),   # (k, token_z) — compare with weight SVD
        "spatial_dirs": U_top.T.detach(),  # (k, n_pairs)
        "effective_rank": effective_rank,
        "top_pair_idx": top_pair_idx,
    }
