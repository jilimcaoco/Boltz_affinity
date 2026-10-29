"""Shared fixtures.

``lora_harness`` runs :func:`boltz.lora.train.train_lora` end-to-end on CPU
with a toy model: checkpoint loading, featurisation and the trunk/affinity
forward are stubbed, so no Boltz-2 checkpoint, MSA or structure is needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.affinity_module = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))


def _feat(name: str) -> torch.Tensor:
    h = hashlib.sha256(name.encode()).digest()
    return torch.tensor([b / 255.0 - 0.5 for b in h[:4]], dtype=torch.float32)


def _base_model() -> _Toy:
    torch.manual_seed(123)
    return _Toy()


def _target(name: str) -> float:
    return float(f"{float(3 * _feat(name).sum()):.6f}")


def _forward(model, feats, *, recycling_steps, **_):  # noqa: ANN001, ARG001
    return {"affinity_pred_value": model.affinity_module(feats["x"])}


@pytest.fixture
def lora_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return ``run(**TrainArgs overrides) -> (adapter_state, last_run_meta)``.

    Attributes: ``csv``, ``names``, ``feat(name)``, ``target(name)``,
    ``base_model()``, ``write_csv(censored=(), n=24)`` and ``fail_rows`` (a
    set of row names whose featurisation raises).
    """
    import boltz.lora.train as T
    from boltz.lora.registry import LoRARegistry

    fail_rows: set[str] = set()

    def featurize(row, **_):  # noqa: ANN001
        name = row["name"][0]
        if name in fail_rows:
            msg = f"stub featurisation failure for {name}"
            raise RuntimeError(msg)
        return {"x": _feat(name).unsqueeze(0)}

    monkeypatch.delenv("BOLTZ_TRAIN_CACHE_DIR", raising=False)
    monkeypatch.setattr(T, "_load_base_model", lambda ckpt, dev: (_base_model(), "", ""))
    monkeypatch.setattr(T, "_featurize_row", featurize)
    monkeypatch.setattr(T, "_affinity_forward_trainable", _forward)

    csv = tmp_path / "train.csv"
    registry = LoRARegistry(tmp_path / "reg")

    def write_csv(censored=(), n: int = 24) -> list[str]:
        names = [f"m{i:02d}" for i in range(n)]
        header = "name,ligand,receptor,target,structure,group_id"
        lines = [header + (",is_censored" if censored else "")]
        for i, nm in enumerate(names):
            line = f"{nm},C,r.yaml,{_target(nm)},s.pdb,g{i % 3}"
            if censored:
                line += f",{int(nm in censored)}"
            lines.append(line)
        csv.write_text("\n".join(lines) + "\n")
        run.names = names
        return names

    def run(**overrides):
        kw = dict(
            name="toy", csv_path=str(csv), loss_spec="intra_assay_huber",
            target_spec="affinity_module", epochs=4, batch_size=4,
            learning_rate=1e-2, plot_loss_curve=False, device="cpu",
            overwrite=True,
        )
        kw.update(overrides)
        torch.manual_seed(0)
        T.train_lora(T.TrainArgs(**kw), registry=registry)
        _, state = registry.load(kw["name"])
        meta = json.loads((registry.adapter_dir(kw["name"]) / "meta.json").read_text())
        return state, meta["history"][-1]

    run.csv = csv
    run.feat = _feat
    run.target = _target
    run.base_model = _base_model
    run.write_csv = write_csv
    run.fail_rows = fail_rows
    write_csv()
    return run
