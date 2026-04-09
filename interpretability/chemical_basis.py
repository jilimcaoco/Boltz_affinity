"""
interpretability/chemical_basis.py
===================================
Project SVD singular vectors onto chemically interpretable directions.

IMPORTANT — embedding availability
------------------------------------
The AffinityModule (src/boltz/model/modules/affinity.py) does **not** contain
atom-type or residue-type embedding tables.  The only embedding it holds is
``dist_bin_pairwise_embed`` (distance-bin → token_z), which encodes inter-
atomic distance bins, not chemical identity.

Residue-type and atom-type information enters the model through the
**InputEmbedder** (src/boltz/model/modules/trunkv2.py), which lives in the
full Boltz2 trunk — not in the AffinityModule.  Specifically:

  * ``model.input_embedder.res_type_encoding``
        nn.Linear(const.num_tokens, token_s, bias=False)
        Weight shape: (token_s, num_tokens=33).
        Each *column* is the learned direction for one token/residue type.
        NOTE: this is in *single-representation* space (token_s), NOT in
        pair-representation space (token_z).

  * There is no standalone atom-type embedding table.  Atoms are encoded
    via an AtomEncoder (atom-level transformer), whose representations are
    aggregated before reaching the pairformer.

Because the SVD directions extracted by ``svd_extractor.py`` live in either
``c_hidden`` (head-internal) or ``d_pair`` (token_z) space, a direct cosine-
similarity comparison with the residue type encoding (token_s space) is only
valid when token_s == token_z or the caller manually provides a projection
into the matching space.

All three public functions therefore accept the embedding table as an
**explicit argument** (``embedding_table``).  A convenience wrapper tries to
pull the table from the full Boltz2 model if the caller passes one, but no
silent guessing is performed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from boltz.data import const
from boltz.model.modules.affinity import AffinityModule


# ---------------------------------------------------------------------------
# Token / residue name lookups (from boltz.data.const)
# ---------------------------------------------------------------------------

# const.tokens is the authoritative list:
#   index 0:  "<pad>"
#   index 1:  "-"
#   indices 2-22:  protein residues (ALA … VAL, UNK)
#   indices 23-27: RNA  (A, G, C, U, N)
#   indices 28-32: DNA  (DA, DG, DC, DT, DN)
#
# const.num_tokens == 33

_TOKEN_NAMES: list[str] = const.tokens  # length 33


@torch.no_grad()
def project_onto_atom_types(
    direction: torch.Tensor,
    module: AffinityModule,
    *,
    embedding_table: Optional[torch.Tensor] = None,
    atom_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute cosine similarity between *direction* and every row of an
    atom-type embedding table.

    Parameters
    ----------
    direction : torch.Tensor
        A 1-D vector of shape ``(d,)`` — typically a singular vector from
        ``svd_extractor.extract_head_circuits``.
    module : AffinityModule
        Included for API symmetry.  **Not used** because the AffinityModule
        does not contain an atom-type embedding table (see module docstring).
    embedding_table : torch.Tensor, optional
        Shape ``(n_atom_types, d)`` — each row is the learned embedding for
        one atom type.  **Required**; raises ``ValueError`` if not provided.
    atom_names : list[str], optional
        Human-readable names for each row of *embedding_table*.  Must have
        length ``n_atom_types``.  If omitted, rows are labelled
        ``"atom_0", "atom_1", …``.

    Returns
    -------
    dict[str, float]
        Mapping ``atom_type_name → cosine_similarity``, sorted descending.
    """
    # --- The AffinityModule has no atom-type embedding table. ---
    # Users must supply one explicitly (e.g. extracted from the full Boltz2
    # trunk's AtomEncoder or from an external featurisation pipeline).
    if embedding_table is None:
        raise ValueError(
            "AffinityModule does not contain an atom-type embedding table. "
            "Pass the table explicitly via the `embedding_table` argument."
        )

    return _cosine_rank(direction, embedding_table, atom_names, prefix="atom")


@torch.no_grad()
def project_onto_residue_types(
    direction: torch.Tensor,
    module: AffinityModule,
    *,
    embedding_table: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Compute cosine similarity between *direction* and every row of a
    residue-type embedding table.

    Parameters
    ----------
    direction : torch.Tensor
        A 1-D vector of shape ``(d,)``.
    module : AffinityModule
        Included for API symmetry.  **Not used** because residue-type
        embeddings live in the InputEmbedder (full Boltz2 trunk), not in
        the AffinityModule.  See module-level docstring.
    embedding_table : torch.Tensor, optional
        Shape ``(num_tokens, d)`` — each row is the learned embedding for
        one token/residue type.  When ``None``, raises ``ValueError``.

        To obtain this from a full Boltz2 model::

            # res_type_encoding.weight has shape (token_s, num_tokens).
            # Transpose so rows = token types.
            table = model.input_embedder.res_type_encoding.weight.T

        Note that the resulting vectors live in *token_s* space.  A direct
        comparison with pair-space (token_z) singular vectors is only
        meaningful when token_s == token_z or the caller has projected
        into the matching space.

    Returns
    -------
    dict[str, float]
        Mapping ``residue_name → cosine_similarity``, sorted descending.
        Names are taken from ``boltz.data.const.tokens`` (e.g. "ALA", "PHE",
        "DA", "<pad>", etc.).
    """
    # --- The AffinityModule has no residue-type embedding table. ---
    # The table lives at model.input_embedder.res_type_encoding.weight
    # (shape: token_s × num_tokens) in the full Boltz2 trunk.
    if embedding_table is None:
        raise ValueError(
            "AffinityModule does not contain a residue-type embedding table. "
            "Pass `embedding_table` explicitly.  For the Boltz2 trunk: "
            "model.input_embedder.res_type_encoding.weight.T  "
            "(shape: num_tokens × token_s)."
        )

    return _cosine_rank(direction, embedding_table, _TOKEN_NAMES, prefix="token")


@torch.no_grad()
def classify_direction(
    direction: torch.Tensor,
    module: AffinityModule,
    top_k: int = 5,
    *,
    atom_embedding_table: Optional[torch.Tensor] = None,
    atom_names: Optional[list[str]] = None,
    residue_embedding_table: Optional[torch.Tensor] = None,
) -> str:
    """Return a human-readable summary of what a singular vector points at.

    Calls :func:`project_onto_atom_types` and
    :func:`project_onto_residue_types`, then formats a short string showing
    the ``top_k`` highest-similarity entries from each.

    Parameters
    ----------
    direction : torch.Tensor
        Shape ``(d,)``.
    module : AffinityModule
        Passed through to the projection functions.
    top_k : int
        Number of entries to show per category.
    atom_embedding_table : torch.Tensor, optional
        Forwarded to :func:`project_onto_atom_types`.
    atom_names : list[str], optional
        Forwarded to :func:`project_onto_atom_types`.
    residue_embedding_table : torch.Tensor, optional
        Forwarded to :func:`project_onto_residue_types`.

    Returns
    -------
    str
        Multi-line summary, e.g.::

            Top residue types: PHE (0.79), TYR (0.76), ...
            Top atom types: C_aromatic (0.82), C_aliphatic (0.71), ...
    """
    lines: list[str] = []

    # --- residue types ---
    if residue_embedding_table is not None:
        res_sims = project_onto_residue_types(
            direction, module, embedding_table=residue_embedding_table,
        )
        top_res = list(res_sims.items())[:top_k]
        entries = ", ".join(f"{name} ({sim:.2f})" for name, sim in top_res)
        lines.append(f"Top residue types: {entries}")
    else:
        lines.append(
            "Top residue types: <skipped — no residue_embedding_table provided>"
        )

    # --- atom types ---
    if atom_embedding_table is not None:
        atom_sims = project_onto_atom_types(
            direction,
            module,
            embedding_table=atom_embedding_table,
            atom_names=atom_names,
        )
        top_atoms = list(atom_sims.items())[:top_k]
        entries = ", ".join(f"{name} ({sim:.2f})" for name, sim in top_atoms)
        lines.append(f"Top atom types: {entries}")
    else:
        lines.append(
            "Top atom types: <skipped — no atom_embedding_table provided>"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _cosine_rank(
    direction: torch.Tensor,
    embedding_table: torch.Tensor,
    names: Optional[list[str]],
    *,
    prefix: str = "item",
) -> dict[str, float]:
    """Rank rows of *embedding_table* by cosine similarity with *direction*.

    Returns a dict ``{name: similarity}`` sorted descending.
    """
    # Validate shapes.
    if direction.ndim != 1:
        raise ValueError(f"direction must be 1-D, got shape {direction.shape}")
    if embedding_table.ndim != 2:
        raise ValueError(
            f"embedding_table must be 2-D, got shape {embedding_table.shape}"
        )
    n_types, d = embedding_table.shape
    if d != direction.shape[0]:
        raise ValueError(
            f"Dimension mismatch: direction has {direction.shape[0]} elements "
            f"but embedding rows have {d}."
        )

    if names is None:
        names = [f"{prefix}_{i}" for i in range(n_types)]
    if len(names) != n_types:
        raise ValueError(
            f"names has {len(names)} entries but embedding_table has "
            f"{n_types} rows."
        )

    # Cosine similarity: cos(θ) = (a · b) / (‖a‖ ‖b‖)
    # F.cosine_similarity broadcasts: (1, d) vs (n, d) → (n,)
    sims = F.cosine_similarity(
        direction.unsqueeze(0),
        embedding_table,
        dim=1,
    )  # (n_types,)

    # Sort descending and build output dict.
    order = sims.argsort(descending=True)
    return {names[i]: sims[i].item() for i in order.tolist()}
