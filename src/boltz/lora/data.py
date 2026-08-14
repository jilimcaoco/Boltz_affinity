"""Dataset for LoRA finetuning of the Boltz affinity stack.

Input CSV schema (header required):

    ligand,receptor,target[,structure][,name][,group_id]

Columns:
    ligand     : path to a single-molecule MOL2/SDF *or* a SMILES string
    receptor   : path to a protein PDB / CIF / Boltz YAML
    target     : float — the value to fit (e.g. log10(IC50_uM))
    structure  : optional — PDB/CIF of the *complex* to inject coordinates
                 (required for ``mode='rescore'``)
    name       : optional human-readable identifier; defaults to row index
    group_id   : optional assay identifier (e.g. ``assay_chembl_id``).
                 Used by :class:`AssayGroupedSampler` to fill mini-batches
                 with same-assay compounds so the intra-assay pairwise Huber
                 loss (``"intra_assay_huber"``) fires on every training step.
    is_binder  : optional binary label (0/1) for the BCE branch of the
                 Boltz-2-style multi-task loss (``"boltz2_affinity"``).
                 If absent, the loss derives it from ``target`` via
                 ``target <= BOLTZ_LORA_BINDER_THRESHOLD`` (default 1.0,
                 i.e. ≤10 µM = binder on the ``log10(IC50_uM)`` scale).

The dataset yields a dict per row::

    {
        "name": str,
        "ligand": str,
        "receptor": str,
        "structure": Optional[str],
        "group_id": Optional[str],
        "is_binder": Optional[int],
        "target": torch.Tensor (scalar float32),
        "row_index": int,
    }

Featurisation (the heavy lift converting these paths into Boltz feature
dicts) is delegated to :mod:`boltz.lora.train` so that this dataset stays
lightweight and pickle-friendly.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import math
import random as _random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import Dataset, Sampler

REQUIRED_COLUMNS = ("ligand", "receptor", "target")
OPTIONAL_COLUMNS = ("structure", "name", "group_id", "is_binder", "is_censored")


@dataclass
class LoRARow:
    """One parsed CSV row."""

    name: str
    ligand: str
    receptor: str
    target: float
    structure: Optional[str] = None
    group_id: Optional[str] = None
    is_binder: Optional[int] = None
    is_censored: int = 0
    row_index: int = 0

    def to_batch_item(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ligand": self.ligand,
            "receptor": self.receptor,
            "structure": self.structure,
            "group_id": self.group_id,
            "is_binder": self.is_binder,
            "is_censored": self.is_censored,
            "target": torch.tensor(self.target, dtype=torch.float32),
            "row_index": self.row_index,
        }


def _hash_csv(path: Path) -> str:
    sha = hashlib.sha256()
    sha.update(path.read_bytes())
    return sha.hexdigest()


def parse_lora_csv(path: str | Path) -> tuple[list[LoRARow], str]:
    """Parse a LoRA training CSV.

    Returns
    -------
    rows : list[LoRARow]
    sha256 : str
        Content hash for provenance.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        msg = f"Training CSV not found: {p}"
        raise FileNotFoundError(msg)

    rows: list[LoRARow] = []
    with p.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            msg = f"CSV {p} has no header row."
            raise ValueError(msg)
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            msg = (
                f"CSV {p} missing required columns: {missing}. "
                f"Have: {reader.fieldnames}"
            )
            raise ValueError(msg)
        for i, raw in enumerate(reader):
            try:
                target = float(raw["target"])
            except (TypeError, ValueError) as e:
                msg = f"Row {i} has non-numeric target '{raw.get('target')}': {e}"
                raise ValueError(msg) from e
            raw_is_binder = (raw.get("is_binder") or "").strip()
            if raw_is_binder == "":
                is_binder: Optional[int] = None
            else:
                try:
                    is_binder = int(float(raw_is_binder))
                except (TypeError, ValueError) as e:
                    msg = (
                        f"Row {i} has non-numeric is_binder "
                        f"'{raw_is_binder}': {e}"
                    )
                    raise ValueError(msg) from e
                if is_binder not in (0, 1):
                    msg = (
                        f"Row {i} is_binder must be 0 or 1 "
                        f"(got {is_binder})."
                    )
                    raise ValueError(msg)
            raw_is_censored = (raw.get("is_censored") or "").strip()
            is_censored = int(float(raw_is_censored)) if raw_is_censored else 0
            rows.append(
                LoRARow(
                    name=raw.get("name") or f"row_{i:06d}",
                    ligand=raw["ligand"].strip(),
                    receptor=raw["receptor"].strip(),
                    target=target,
                    structure=(raw.get("structure") or "").strip() or None,
                    group_id=(raw.get("group_id") or "").strip() or None,
                    is_binder=is_binder,
                    is_censored=is_censored,
                    row_index=i,
                )
            )
    if not rows:
        msg = f"CSV {p} contains zero data rows."
        raise ValueError(msg)
    return rows, _hash_csv(p)


class LoRADataset(Dataset[dict[str, Any]]):
    """Thin row-indexed dataset; featurisation happens in the trainer.

    Holding the heavy featurization at trainer-step time keeps this class
    cheap to instantiate, picklable for DataLoader workers, and lets us
    cache featurized inputs per ``row_index`` on demand.
    """

    def __init__(self, csv_path: str | Path, *, mode: str = "rescore") -> None:
        if mode not in {"rescore", "full"}:
            msg = f"mode must be 'rescore' or 'full', got {mode!r}"
            raise ValueError(msg)
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.mode = mode
        self.rows, self.sha256 = parse_lora_csv(self.csv_path)
        if mode == "rescore":
            missing = [r for r in self.rows if not r.structure]
            if missing:
                msg = (
                    f"mode='rescore' requires a 'structure' column for every row "
                    f"({len(missing)}/{len(self.rows)} are empty). "
                    "Use mode='full' to run the full Boltz pipeline instead."
                )
                raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx].to_batch_item()


def lora_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimal collate: stack ``target``, list-of everything else.

    Batch size will almost always be 1 for affinity training because
    featurization is expensive and per-row variable-shaped.
    """
    out: dict[str, Any] = {
        "name": [it["name"] for it in items],
        "ligand": [it["ligand"] for it in items],
        "receptor": [it["receptor"] for it in items],
        "structure": [it["structure"] for it in items],
        "group_id": [it.get("group_id") for it in items],
        "is_binder": [it.get("is_binder") for it in items],
        "row_index": [it["row_index"] for it in items],
        "target": torch.stack([it["target"] for it in items]),
    }
    return out


class AssayGroupedSampler(Sampler[list[int]]):
    """Batch sampler that fills each mini-batch from a single assay group.

    Each yielded index list is drawn entirely from one ``group_id`` (assay),
    so the Boltz-2-style ``"intra_assay_huber"`` loss always has intra-assay
    pairs to work with.  Rows whose ``group_id`` is ``None`` or empty are
    yielded as singletons.

    Parameters
    ----------
    rows:
        The :attr:`LoRADataset.rows` list.
    batch_size:
        Maximum rows per yielded batch.  Groups smaller than ``batch_size``
        are yielded as a single undersized batch.
    shuffle:
        Shuffle group order and intra-group row order each epoch.
        Call :meth:`set_epoch` at the start of each epoch for
        reproducible but varied ordering.
    seed:
        Base RNG seed; the epoch index is added before each iteration.
    """

    def __init__(
        self,
        rows: list[LoRARow],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

        grouped: dict[str, list[int]] = collections.defaultdict(list)
        ungrouped: list[int] = []
        for i, row in enumerate(rows):
            gid = row.group_id
            if gid:
                grouped[gid].append(i)
            else:
                ungrouped.append(i)
        self._grouped: dict[str, list[int]] = dict(grouped)
        self._ungrouped: list[int] = ungrouped
        self._len: int = (
            sum(math.ceil(len(v) / batch_size) for v in self._grouped.values())
            + len(ungrouped)
        )

    # Expose read-only stats for logging.
    @property
    def n_assay_groups(self) -> int:
        return len(self._grouped)

    @property
    def n_ungrouped(self) -> int:
        return len(self._ungrouped)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for reproducible shuffling."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[list[int]]:
        rng = _random.Random(self._seed + self._epoch)
        all_batches: list[list[int]] = []

        group_keys = list(self._grouped.keys())
        if self._shuffle:
            rng.shuffle(group_keys)

        for gid in group_keys:
            idxs = list(self._grouped[gid])
            if self._shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), self._batch_size):
                all_batches.append(idxs[start : start + self._batch_size])

        # Ungrouped rows as singletons, interspersed with group batches.
        for idx in self._ungrouped:
            all_batches.append([idx])

        if self._shuffle:
            rng.shuffle(all_batches)

        yield from all_batches


__all__ = [
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "AssayGroupedSampler",
    "LoRADataset",
    "LoRARow",
    "lora_collate",
    "parse_lora_csv",
]
