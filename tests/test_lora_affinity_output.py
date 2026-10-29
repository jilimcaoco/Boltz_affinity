"""``affinity_output`` (module1 vs ensemble) and the repaired apply/export CLIs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner
from torch import nn

import boltz.affinity_rescoring.inference as inference
import boltz.lora.train as T

N_TOK, MW = 6, 412.0


class _Head(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.value = nn.Linear(4, 1)
        self.logit = nn.Linear(4, 1)

    def forward(self, *, s_inputs, z, x_pred, feats, multiplicity, use_kernels):  # noqa: ANN001, ARG002
        h = s_inputs.mean(dim=1)
        return {"affinity_pred_value": self.value(h), "affinity_logits_binary": self.logit(h)}


class _EnsembleModel(nn.Module):
    def __init__(self, *, ensemble: bool = True, mw_correction: bool = True) -> None:
        super().__init__()
        self.affinity_ensemble = ensemble
        self.affinity_mw_correction = mw_correction
        self.use_kernels = False
        if ensemble:
            self.affinity_module1, self.affinity_module2 = _Head(1), _Head(2)
        else:
            self.affinity_module = _Head(1)

    def input_embedder(self, feats, affinity=False):  # noqa: ANN001, ARG002
        return feats["s"]


def _feats() -> dict:
    torch.manual_seed(0)
    return {
        "s": torch.randn(1, N_TOK, 4),
        "coords": torch.randn(1, 1, 10, 3),
        "token_pad_mask": torch.ones(1, N_TOK),
        "mol_type": torch.zeros(1, N_TOK, dtype=torch.long),
        "affinity_token_mask": torch.tensor([[0, 0, 0, 0, 1, 1]]),
        "affinity_mw": [MW],
    }


@pytest.fixture(autouse=True)
def _stub_trunk(monkeypatch):
    monkeypatch.setattr(T, "_TRAIN_CACHE", None)
    monkeypatch.setattr(
        T, "_compute_trunk_z", lambda *a, **k: torch.zeros(1, N_TOK, N_TOK, 2),
    )


def _forward(model, mode):
    return T._affinity_forward_trainable(
        model, _feats(), recycling_steps=0, affinity_output=mode,
    )


def test_module1_is_raw_first_head():
    model = _EnsembleModel()
    out = _forward(model, "module1")
    ref = model.affinity_module1(
        s_inputs=_feats()["s"], z=None, x_pred=None, feats=None, multiplicity=1, use_kernels=False,
    )
    assert torch.equal(out["affinity_pred_value"], ref["affinity_pred_value"])


def test_ensemble_matches_scoring_formula():
    model = _EnsembleModel()
    out = _forward(model, "ensemble")
    kw = dict(s_inputs=_feats()["s"], z=None, x_pred=None, feats=None, multiplicity=1, use_kernels=False)
    o1, o2 = model.affinity_module1(**kw), model.affinity_module2(**kw)
    avg = (o1["affinity_pred_value"] + o2["affinity_pred_value"]) / 2
    expected = 1.03525938 * avg - 0.59992683 * MW**0.3 + 2.83288489
    assert torch.allclose(out["affinity_pred_value"], expected)
    assert torch.allclose(
        out["affinity_pred_value"], inference.apply_affinity_mw_correction(avg, {"affinity_mw": [MW]}),
    )
    prob = (torch.sigmoid(o1["affinity_logits_binary"]) + torch.sigmoid(o2["affinity_logits_binary"])) / 2
    assert torch.allclose(torch.sigmoid(out["affinity_logits_binary"]), prob, atol=1e-6)


def test_ensemble_without_mw_correction_is_plain_average():
    model = _EnsembleModel(mw_correction=False)
    out = _forward(model, "ensemble")
    kw = dict(s_inputs=_feats()["s"], z=None, x_pred=None, feats=None, multiplicity=1, use_kernels=False)
    avg = (model.affinity_module1(**kw)["affinity_pred_value"]
           + model.affinity_module2(**kw)["affinity_pred_value"]) / 2
    assert torch.allclose(out["affinity_pred_value"], avg)


def test_ensemble_trains_both_modules():
    model = _EnsembleModel()
    _forward(model, "ensemble")["affinity_pred_value"].sum().backward()
    for mod in (model.affinity_module1, model.affinity_module2):
        assert mod.value.weight.grad is not None and mod.value.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    _forward(model, "module1")["affinity_pred_value"].sum().backward()
    assert model.affinity_module2.value.weight.grad is None


def test_ensemble_on_single_module_checkpoint_falls_back():
    model = _EnsembleModel(ensemble=False)
    assert torch.equal(
        _forward(model, "ensemble")["affinity_pred_value"],
        _forward(model, "module1")["affinity_pred_value"],
    )


def test_trainer_records_affinity_output(lora_harness):
    _, run = lora_harness(affinity_output="ensemble", epochs=1)
    assert run["affinity_output"] == "ensemble"
    _, run = lora_harness(epochs=1)
    assert run["affinity_output"] == "module1"
    with pytest.raises(ValueError, match="affinity_output"):
        lora_harness(affinity_output="both", epochs=1)


# ─── apply / export CLIs (previously imported a non-existent module) ─────────


class _FakeManager:
    def __init__(self, *a, **k) -> None:
        pass

    def load_model(self, checkpoint_path=None):  # noqa: ANN001, ARG002
        return nn.Linear(2, 2)


@pytest.fixture
def fake_models(monkeypatch):
    import boltz.finetune.apply as fapply
    import boltz.lora.apply as lapply

    monkeypatch.setattr(inference, "AffinityModelManager", _FakeManager)
    adapter = SimpleNamespace(
        name="a", config=SimpleNamespace(rank=4, target_spec="t", num_trainable_params=1e6),
        adapted_layers=["x"],
    )
    monkeypatch.setattr(lapply, "load_adapter_into_model", lambda *a, **k: adapter)
    monkeypatch.setattr(fapply, "load_finetune_into_model", lambda *a, **k: adapter)


@pytest.mark.parametrize("group", ["lora", "finetune"])
def test_apply_and_export_run(group, fake_models, tmp_path):
    if group == "lora":
        from boltz.lora.cli import lora_cli as cli
    else:
        from boltz.finetune.cli import finetune_cli as cli
    runner = CliRunner()
    res = runner.invoke(cli, ["apply", "a"])
    assert res.exit_code == 0, res.output
    assert "OK: applied 'a'" in res.output
    out = tmp_path / "merged.ckpt"
    res = runner.invoke(cli, ["export", "a", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "state_dict" in torch.load(out, weights_only=False)
