"""Dataset for LoRA finetuning of the Boltz affinity stack.

Input CSV schema (header required):

    ligand,receptor,target[,structure][,name]

Columns:
    ligand     : path to a single-molecule MOL2/SDF *or* a SMILES string
    receptor   : path to a protein PDB / CIF / Boltz YAML
    target     : float — the value to fit (e.g. pIC50)
    structure  : optional — PDB/CIF of the *complex* to inject coordinates
                 (required for ``mode='rescore'``)
    name       : optional human-readable identifier; defaults to row index

The dataset yields a dict per row::

    {
        "name": str,
        "ligand": str,
        "receptor": str,
        "structure": Optional[str],
        "target": torch.Tensor (scalar float32),
        "row_index": int,
    }

Featurisation (the heavy lift converting these paths into Boltz feature
dicts) is delegated to :mod:`boltz.lora.train` so that this dataset stays
lightweight and pickle-friendly.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

REQUIRED_COLUMNS = ("ligand", "receptor", "target")
OPTIONAL_COLUMNS = ("structure", "name")


@dataclass
class LoRARow:
    """One parsed CSV row."""

    name: str
    ligand: str
    receptor: str
    target: float
    structure: Optional[str] = None
    row_index: int = 0

    def to_batch_item(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ligand": self.ligand,
            "receptor": self.receptor,
            "structure": self.structure,
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
            rows.append(
                LoRARow(
                    name=raw.get("name") or f"row_{i:06d}",
                    ligand=raw["ligand"].strip(),
                    receptor=raw["receptor"].strip(),
                    target=target,
                    structure=(raw.get("structure") or "").strip() or None,
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
        "row_index": [it["row_index"] for it in items],
        "target": torch.stack([it["target"] for it in items]),
    }
    return out


__all__ = [
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "LoRADataset",
    "LoRARow",
    "lora_collate",
    "parse_lora_csv",
]
