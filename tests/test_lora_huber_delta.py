"""Tests for data-derived Huber delta selection (BOLTZ_HUBER_DELTA_SPEC.md).

CPU-cheap: the trainer runs end-to-end on a toy model with featurisation,
the trunk forward and checkpoint loading stubbed out.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from click.testing import CliRunner
from torch import nn

import boltz.lora.train as T
from boltz.lora.adapter import LoRAAdapter, TrainingRun
from boltz.lora.inject import apply_lora
from boltz.lora.losses import (
    BUILTIN_LOSSES,
    bind_delta,
    call_loss,
    censored_intra_assay_huber_loss,
    huber_loss,
    intra_assay_huber_loss,
    loss_delta,
    mse_loss,
    robust_delta,
)
from boltz.lora.registry import LoRARegistry

HUBER_FAMILY = [
    "huber",
    "intra_assay_huber",
    "censored_intra_assay_huber",
    "boltz2_affinity",
    "censored_boltz2_affinity",
]
NO_DELTA = ["mse", "mae", "bce", "pairwise_ranking"]


# ─── Toy trainer harness ─────────────────────────────────────────────────────


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


def _forward(model, feats, *, recycling_steps):  # noqa: ANN001, ARG001
    return {"affinity_pred_value": model.affinity_module(feats["x"])}


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BOLTZ_TRAIN_CACHE_DIR", raising=False)
    monkeypatch.setattr(T, "_load_base_model", lambda ckpt, dev: (_base_model(), "", ""))
    monkeypatch.setattr(
        T, "_featurize_row",
        lambda row, **_: {"x": _feat(row["name"][0]).unsqueeze(0)},
    )
    monkeypatch.setattr(T, "_affinity_forward_trainable", _forward)

    names = [f"m{i:02d}" for i in range(24)]
    csv = tmp_path / "train.csv"
    lines = ["name,ligand,receptor,target,structure,group_id"]
    for i, n in enumerate(names):
        lines.append(f"{n},C,r.yaml,{float(3 * _feat(n).sum()):.6f},s.pdb,g{i % 3}")
    csv.write_text("\n".join(lines) + "\n")
    registry = LoRARegistry(tmp_path / "reg")

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
    run.names = names
    return run


def _pred_batch(n: int = 64, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pred = {"affinity_pred_value": torch.randn(n, 1, generator=g) * 2}
    batch = {
        "target": torch.randn(n, generator=g) * 2,
        "group_id": [f"g{i % 4}" for i in range(n)],
        "is_censored": [int(i % 5 == 0) for i in range(n)],
        "is_binder": [int(i % 2) for i in range(n)],
    }
    pred["affinity_logits_binary"] = torch.randn(n, 1, generator=g)
    return pred, batch


# ─── Backward compatibility ──────────────────────────────────────────────────


def test_default_path_never_binds_or_estimates(harness, monkeypatch):
    """R1: no new flags -> no residual pass, no MAD, unbound loss call."""
    def _boom(*a, **k):  # noqa: ANN002, ANN003
        raise AssertionError("must not be called in the default path")

    monkeypatch.setattr(T, "bind_delta", _boom)
    monkeypatch.setattr(T, "robust_delta", _boom)
    monkeypatch.setattr(T, "_vanilla_residuals", _boom)
    monkeypatch.setattr(T, "_batch_residuals", _boom)
    state, run = harness(val_split=0.25)
    assert run["huber_delta_mode"] == "fixed"
    assert run["huber_delta_resolved"] == 1.0
    assert run["huber_delta_per_epoch"] is None


def test_default_path_matches_explicit_unbound_trajectory(harness, monkeypatch):
    """R1: weights from a default run equal a run where call_loss is forced to
    use the raw built-in function (i.e. exactly the pre-patch loss call)."""
    state_default, _ = harness(name="a")
    raw = BUILTIN_LOSSES["intra_assay_huber"]
    seen = []

    def _call(fn, pred, batch, meta):  # noqa: ANN001
        seen.append(fn)
        return call_loss(raw, pred, batch, meta)

    monkeypatch.setattr(T, "call_loss", _call)
    state_raw, _ = harness(name="b")
    assert all(fn is raw for fn in seen)
    assert state_default.keys() == state_raw.keys()
    for k in state_default:
        assert torch.equal(state_default[k], state_raw[k]), k


@pytest.mark.parametrize("name", HUBER_FAMILY)
def test_bind_delta_one_is_identity(name):
    fn = BUILTIN_LOSSES[name]
    pred, batch = _pred_batch()
    assert torch.equal(call_loss(bind_delta(fn, 1.0), pred, batch), call_loss(fn, pred, batch))


def test_loss_defaults_unchanged():
    """R2: delta / pair_weight defaults stay 1.0 / 2.0."""
    import inspect
    for fn in (intra_assay_huber_loss, censored_intra_assay_huber_loss):
        params = inspect.signature(fn).parameters
        assert params["delta"].default == 1.0
        assert params["pair_weight"].default == 2.0
        assert params["scale_invariant"].default is False


# ─── robust_delta ────────────────────────────────────────────────────────────


def test_robust_delta_gaussian():
    s = 0.8
    r = torch.randn(200_000, generator=torch.Generator().manual_seed(0)) * s
    assert robust_delta(r, k=1.345) == pytest.approx(1.345 * s, rel=0.05)


def test_robust_delta_offset_invariant():
    r = torch.randn(5_000, generator=torch.Generator().manual_seed(1)).double()
    assert robust_delta(r + 3.7) == pytest.approx(robust_delta(r), rel=1e-9)


@pytest.mark.parametrize(("frac", "tol"), [(0.05, 0.10), (0.10, 0.20)])
def test_robust_delta_outlier_resistant(frac, tol):
    """Influence of outliers is bounded: the result does not depend on how
    large they are. The MAD's own contamination bias is ~6.5% at 5% and
    ~13-15% at 10% (one-sided), so the spec's "<10% at 10%" is relaxed to the
    tolerances it can actually meet."""
    r = torch.randn(10_000, generator=torch.Generator().manual_seed(2))
    clean = robust_delta(r)
    results = []
    for magnitude in (1e3, 1e6):
        corrupted = r.clone()
        corrupted[: int(len(r) * frac)] = magnitude
        results.append(robust_delta(corrupted))
    assert results[0] == results[1]
    assert abs(results[0] - clean) / clean < tol


@pytest.mark.parametrize("bad", [[0.1] * 7, [0.5] * 50, [float("nan")] * 20 + [1.0] * 5])
def test_robust_delta_degenerate_returns_none(bad):
    assert robust_delta(bad) is None


def test_robust_delta_floor_and_nonfinite():
    r = [0.0] * 4 + [1e-9, -1e-9] * 4 + [float("inf")]
    assert robust_delta(r, floor=1e-3) == 1e-3


def test_degenerate_falls_back_with_warning(caplog):
    args = T.TrainArgs(name="x", csv_path="x", huber_delta=0.75)
    with caplog.at_level(logging.WARNING, logger=T.logger.name):
        assert T._resolve_delta(torch.zeros(5), args, "test") == 0.75
    assert "falling back to huber_delta=0.7500" in caplog.text


# ─── Binding and dispatch ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", NO_DELTA)
def test_bind_delta_no_delta_param_is_unchanged(name):
    fn = BUILTIN_LOSSES[name]
    assert bind_delta(fn, 0.5) is fn
    assert loss_delta(fn) is None


def test_bind_delta_mse_unchanged():
    assert bind_delta(mse_loss, 2.0) is mse_loss


@pytest.mark.parametrize("name", HUBER_FAMILY)
def test_call_loss_on_bound_builtin(name):
    bound = bind_delta(BUILTIN_LOSSES[name], 0.5)
    pred, batch = _pred_batch()
    got = call_loss(bound, pred, batch, None)
    want = BUILTIN_LOSSES[name](pred, batch, None, delta=0.5)
    assert torch.equal(got, want)
    assert loss_delta(bound) == 0.5


def test_call_loss_on_bound_positional_delta_custom_loss():
    """A naive partial turns a positional-or-keyword `delta` keyword-only; the
    old len(params) >= 3 dispatch then passed adapter_meta positionally into
    it and raised. call_loss must count positional slots only."""
    def custom(pred, batch, delta=1.0):  # noqa: ANN001
        return F.smooth_l1_loss(pred["affinity_pred_value"].squeeze(-1), batch["target"], beta=delta)

    pred, batch = _pred_batch()
    bound = bind_delta(custom, 0.25)
    assert torch.equal(call_loss(bound, pred, batch, object()), custom(pred, batch, delta=0.25))


def test_call_loss_three_arg_custom_still_gets_meta():
    sentinel = object()

    def custom(pred, batch, meta, *, delta=1.0):  # noqa: ANN001
        assert meta is sentinel
        return torch.tensor(delta)

    assert call_loss(bind_delta(custom, 0.3), {}, {}, sentinel).item() == pytest.approx(0.3)


# ─── Scale relationship ──────────────────────────────────────────────────────


@pytest.mark.parametrize("d", [0.25, 0.5, 1.0, 2.0])
def test_huber_equals_delta_times_smooth_l1_convention(d):
    """Huber(x, delta) == delta * smooth_l1(x, beta=delta). The losses use the
    smooth_l1 form by default; scale_invariant=True gives textbook Huber."""
    x = torch.linspace(-4, 4, 101)
    pred, batch = {"affinity_pred_value": x}, {"target": torch.zeros_like(x)}
    smooth = F.smooth_l1_loss(x, torch.zeros_like(x), beta=d)
    textbook = F.huber_loss(x, torch.zeros_like(x), delta=d)
    assert torch.allclose(textbook, d * smooth)
    assert torch.equal(huber_loss(pred, batch, delta=d), smooth)
    assert torch.allclose(huber_loss(pred, batch, delta=d, scale_invariant=True), textbook)


@pytest.mark.parametrize("name", HUBER_FAMILY[1:3])
def test_scale_invariant_scales_value_terms(name):
    fn = BUILTIN_LOSSES[name]
    pred, batch = _pred_batch()
    plain = fn(pred, batch, delta=0.5)
    scaled = fn(pred, batch, delta=0.5, scale_invariant=True)
    assert torch.allclose(scaled, 0.5 * plain)


# ─── Mode behaviour ──────────────────────────────────────────────────────────


def test_lora_disabled_reproduces_fresh_base_model():
    base = _base_model()
    adapted = _base_model()
    apply_lora(adapted, ("affinity_module",), r=4, alpha=8.0)
    for m in adapted.modules():
        if hasattr(m, "lora_B"):
            nn.init.normal_(m.lora_B)
    x = torch.randn(16, 4)
    assert not torch.equal(adapted.affinity_module(x), base.affinity_module(x))
    with T._lora_disabled(adapted):
        assert torch.equal(adapted.affinity_module(x), base.affinity_module(x))
    assert not torch.equal(adapted.affinity_module(x), base.affinity_module(x))


def test_static_vanilla_resolves_once_from_base_residuals(harness, monkeypatch):
    bound = []
    real_bind = T.bind_delta
    monkeypatch.setattr(T, "bind_delta", lambda fn, d: bound.append(d) or real_bind(fn, d))
    _, run = harness(huber_delta_mode="static-vanilla", epochs=3)

    base = _base_model()
    with torch.no_grad():
        preds = torch.stack([base.affinity_module(_feat(n)).squeeze() for n in harness.names])
    targets = torch.tensor([float(f"{float(3 * _feat(n).sum()):.6f}") for n in harness.names])
    expected = robust_delta(preds - targets, k=1.345)

    assert bound == [pytest.approx(expected, rel=1e-6)]  # bound once, never rebound
    assert run["huber_delta_mode"] == "static-vanilla"
    assert run["huber_delta_resolved"] == pytest.approx(expected, rel=1e-6)
    assert run["huber_delta_k"] == 1.345
    assert run["huber_delta_per_epoch"] is None


def test_dynamic_mad_records_per_epoch(harness):
    _, run = harness(
        huber_delta_mode="dynamic-mad", early_stopping_rel_delta=0.01,
        early_stopping_patience=0, epochs=4,
    )
    per_epoch = run["huber_delta_per_epoch"]
    assert len(per_epoch) == len(run["metrics"]) == 4
    assert per_epoch[0] == 1.0  # seeded from huber_delta
    assert len(set(per_epoch)) >= 2
    assert run["huber_delta_resolved"] == per_epoch[-1]
    assert run["early_stopping_rel_delta"] == 0.01


def test_dynamic_mad_requires_rel_delta(harness):
    with pytest.raises(ValueError, match="early_stopping_rel_delta"):
        harness(huber_delta_mode="dynamic-mad")


def test_cli_dynamic_mad_requires_rel_delta(harness):
    from boltz.lora.cli import lora_cli

    res = CliRunner().invoke(lora_cli, [
        "train", "--name", "x", "--csv", str(harness.csv),
        "--huber-delta-mode", "dynamic-mad",
    ])
    assert res.exit_code == 2
    assert "requires --early-stopping-rel-delta" in res.output


def test_cli_rel_and_abs_min_delta_mutually_exclusive(harness):
    from boltz.lora.cli import lora_cli

    res = CliRunner().invoke(lora_cli, [
        "train", "--name", "x", "--csv", str(harness.csv),
        "--early-stopping-rel-delta", "0.01", "--early-stopping-min-delta", "0.01",
    ])
    assert res.exit_code == 2
    assert "mutually exclusive" in res.output


def test_relative_early_stopping_criterion(harness):
    # Improvement must exceed 10 * |best|: impossible, so stop after patience.
    _, run = harness(early_stopping_rel_delta=10.0, early_stopping_patience=1, epochs=5)
    assert len(run["metrics"]) == 2


def test_delta_mode_with_no_delta_loss_warns_and_proceeds(harness, monkeypatch, caplog):
    """R3: mse takes no delta -> warning, no residual pass, training runs."""
    monkeypatch.setattr(
        T, "_vanilla_residuals",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no pass expected")),
    )
    with caplog.at_level(logging.WARNING, logger=T.logger.name):
        _, run = harness(loss_spec="mse", huber_delta_mode="static-vanilla", epochs=2)
    assert "has no 'delta' parameter" in caplog.text
    assert run["huber_delta_resolved"] is None
    assert len(run["metrics"]) == 2


def test_fixed_non_default_delta_is_bound(harness):
    _, run = harness(huber_delta=0.5, epochs=1)
    assert run["huber_delta_resolved"] == 0.5


# ─── Metadata round-trip ─────────────────────────────────────────────────────


def test_training_run_roundtrip_and_forward_compat():
    run = TrainingRun(
        started_at=1.0, huber_delta_mode="dynamic-mad", huber_delta_resolved=1.2,
        huber_delta_k=1.345, huber_delta_per_epoch=[1.0, 1.2],
        early_stopping_rel_delta=0.01,
    )
    a = LoRAAdapter(name="x", history=[run])
    d = json.loads(a.to_json())
    d["history"][0]["some_future_field"] = 1
    back = LoRAAdapter.from_dict(d)
    assert back.history[0] == run
    # Metadata written before these fields existed still loads.
    legacy = {"name": "x", "history": [{"started_at": 1.0, "loss_spec": "mse"}]}
    assert LoRAAdapter.from_dict(legacy).history[0].huber_delta_mode is None
