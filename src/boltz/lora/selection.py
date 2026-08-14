"""Active-learning candidate selection for LoRA finetuning loops.

Given a CSV of screening results (one row per scored ligand) this module
ranks candidates for the next training round using one of:

* ``uncertainty`` – highest per-row predictive uncertainty first. Falls back
  to ``abs(affinity_pred)`` when no uncertainty column is present, which
  approximates *informativeness* in score-space.
* ``diversity``   – greedy Tanimoto-furthest selection over Morgan fingerprints.
  Requires RDKit.
* ``hybrid``      – z-score-sum of uncertainty rank and diversity rank.
* ``topk``        – plain best-by-prediction (lowest ``affinity_pred``).

Inputs are assumed to be the output of a ``boltz rescore`` style screen:
the CSV must contain ``name``, ``ligand`` (SMILES), and ``affinity_pred``
columns. Extra columns are preserved on output.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = ("name", "ligand", "affinity_pred")


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = f"Scores CSV is missing required columns: {missing}"
        raise ValueError(msg)


def _rank_by_uncertainty(
    df: pd.DataFrame, uncertainty_col: str,
) -> pd.Series:
    if uncertainty_col in df.columns:
        return df[uncertainty_col].astype(float).abs()
    logger.warning(
        "Uncertainty column %r not found; falling back to |affinity_pred|.",
        uncertainty_col,
    )
    return df["affinity_pred"].astype(float).abs()


def _morgan_fps(smiles: list[str], radius: int = 2, n_bits: int = 2048):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - depends on env
        msg = "Diversity selection requires RDKit (pip install rdkit)."
        raise ImportError(msg) from exc

    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return fps


def _tanimoto(a, b) -> float:
    from rdkit import DataStructs
    return float(DataStructs.TanimotoSimilarity(a, b))


def _greedy_diverse(df: pd.DataFrame, k: int) -> list[int]:
    """Return row indices of a greedy MaxMin Tanimoto selection."""
    fps = _morgan_fps(df["ligand"].fillna("").tolist())
    valid = [i for i, fp in enumerate(fps) if fp is not None]
    if not valid:
        return []

    # Seed with the best-predicted ligand by affinity_pred (lowest = strongest).
    pred = df["affinity_pred"].astype(float)
    seed = pred.iloc[valid].idxmin()
    selected = [seed]
    remaining = [i for i in valid if i != seed]

    min_sim = {i: _tanimoto(fps[i], fps[seed]) for i in remaining}
    while len(selected) < min(k, len(valid)) and remaining:
        # Pick the candidate with the smallest current similarity to selected set.
        nxt = min(remaining, key=lambda i: min_sim[i])
        selected.append(nxt)
        remaining.remove(nxt)
        # Update min_sim against the newly added item.
        for j in remaining:
            sim = _tanimoto(fps[j], fps[nxt])
            if sim > min_sim[j]:
                min_sim[j] = sim
    return selected


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or pd.isna(std):
        return s * 0.0
    return (s - s.mean()) / std


def select_candidates(
    scores_csv: str,
    *,
    top_k: int = 50,
    strategy: str = "hybrid",
    uncertainty_col: str = "affinity_pred_std",
) -> pd.DataFrame:
    """Return the top-``k`` candidates for the next active-learning round."""
    df = pd.read_csv(scores_csv)
    _validate(df)

    if strategy == "topk":
        return df.nsmallest(top_k, "affinity_pred").reset_index(drop=True)

    if strategy == "uncertainty":
        df = df.copy()
        df["_uncertainty"] = _rank_by_uncertainty(df, uncertainty_col)
        return df.nlargest(top_k, "_uncertainty").drop(columns="_uncertainty").reset_index(drop=True)

    if strategy == "diversity":
        idx = _greedy_diverse(df, top_k)
        return df.iloc[idx].reset_index(drop=True)

    if strategy == "hybrid":
        unc = _rank_by_uncertainty(df, uncertainty_col)
        # Diversity rank: position in greedy ordering (earlier = more diverse).
        div_order = _greedy_diverse(df, len(df))
        div_rank = pd.Series(
            [len(df)] * len(df), index=df.index, dtype=float,
        )
        for pos, idx in enumerate(div_order):
            div_rank.iloc[idx] = float(pos)
        # Lower div_rank == more diverse; we want diverse first, so invert.
        score = _zscore(unc) + _zscore(-div_rank)
        out = df.copy()
        out["_select_score"] = score
        return out.nlargest(top_k, "_select_score").drop(columns="_select_score").reset_index(drop=True)

    msg = f"Unknown selection strategy: {strategy!r}"
    raise ValueError(msg)


__all__ = ["select_candidates"]
