"""Unit tests for the full-finetune subsystem.

Like ``tests/test_lora.py``, these run on a tiny toy module so they are
CPU-cheap and don't require the Boltz affinity checkpoint.  They cover the
parts of the package that don't depend on the actual Boltz2 forward
(target resolution, registry round-trip, parameter selection, apply).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from boltz.finetune import (
    FinetuneConfig,
    FinetuneRecord,
    FinetuneRegistry,
    load_finetune_into_model,
)
from boltz.finetune.l2_sp import add_l2_sp_penalty, l2_sp_penalty
from boltz.finetune.targets import TARGET_PRESETS, resolve_targets
from boltz.finetune.train import _select_trainable_params


# ─── Fixtures ───────────────────────────────────────────────────────────────


class _ToyBoltz(nn.Module):
    """Stand-in for the parts of Boltz2 that fine-tune cares about.

    Names are chosen to match the regexes used by the
    :mod:`boltz.finetune.targets` presets: a top-level ``affinity_module``
    with a ``pairformer_stack`` and ``affinity_heads`` subtree.
    """

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 4)  # NOT trainable for any preset
        self.affinity_module = nn.Module()
        self.affinity_module.pairformer_stack = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        )
        self.affinity_module.affinity_heads = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 1),
        )


# ─── Targets ────────────────────────────────────────────────────────────────


def test_target_presets_known() -> None:
    assert "affinity_module" in TARGET_PRESETS
    assert "affinity_heads" in TARGET_PRESETS
    assert "affinity_pairformer" in TARGET_PRESETS
    assert "heads_pairformer" in TARGET_PRESETS
    # ``heads`` is an alias for the LoRA preset name.
    assert TARGET_PRESETS["heads"] == TARGET_PRESETS["affinity_heads"]


def test_resolve_targets_preset_and_regex() -> None:
    assert resolve_targets("affinity_heads") == TARGET_PRESETS["affinity_heads"]
    assert resolve_targets(r"^my_custom\.layer$") == (r"^my_custom\.layer$",)
    assert resolve_targets("a,b") == ("a", "b")


def test_select_trainable_params_affinity_module() -> None:
    model = _ToyBoltz()
    sel = _select_trainable_params(
        model, resolve_targets("affinity_module")
    )
    names = {n for n, _ in sel}
    # All affinity_module params selected, trunk excluded.
    assert all(n.startswith("affinity_module.") for n in names)
    assert not any(n.startswith("trunk.") for n in names)
    # Both pairformer + heads should be in there.
    assert any("pairformer_stack" in n for n in names)
    assert any("affinity_heads" in n for n in names)


def test_select_trainable_params_heads_only_is_subset() -> None:
    model = _ToyBoltz()
    heads_names = {
        n for n, _ in _select_trainable_params(
            model, resolve_targets("affinity_heads"),
        )
    }
    pf_names = {
        n for n, _ in _select_trainable_params(
            model, resolve_targets("affinity_pairformer"),
        )
    }
    assert heads_names.isdisjoint(pf_names)
    assert heads_names | pf_names == {
        n for n, _ in _select_trainable_params(
            model, resolve_targets("heads_pairformer"),
        )
    }


def test_select_trainable_params_no_match_raises() -> None:
    model = _ToyBoltz()
    with pytest.raises(ValueError, match="zero parameters"):
        _select_trainable_params(model, ("^does_not_exist",))


# ─── Registry round-trip ────────────────────────────────────────────────────


def _make_record(name: str = "ft_v1") -> tuple[FinetuneRecord, dict[str, torch.Tensor]]:
    record = FinetuneRecord(
        name=name,
        boltz_version="test",
        config=FinetuneConfig(
            target_spec="affinity_heads",
            target_patterns=list(TARGET_PRESETS["affinity_heads"]),
            num_trainable_params=42,
        ),
        trained_params=["affinity_module.affinity_heads.0.weight"],
    )
    state = {
        "affinity_module.affinity_heads.0.weight": torch.zeros(4, 4),
    }
    return record, state


def test_registry_save_load_roundtrip(tmp_path: Path) -> None:
    reg = FinetuneRegistry(root=tmp_path)
    record, state = _make_record()
    reg.save(record, state)

    assert reg.exists("ft_v1")
    loaded, state2 = reg.load("ft_v1")
    assert loaded.name == "ft_v1"
    assert loaded.config.target_spec == "affinity_heads"
    assert loaded.trained_params == record.trained_params
    assert torch.equal(
        state2["affinity_module.affinity_heads.0.weight"],
        state["affinity_module.affinity_heads.0.weight"],
    )

    # Round-trips the index too.
    listed = reg.list()
    assert len(listed) == 1
    assert listed[0]["name"] == "ft_v1"


def test_registry_save_overwrite(tmp_path: Path) -> None:
    reg = FinetuneRegistry(root=tmp_path)
    record, state = _make_record()
    reg.save(record, state)
    with pytest.raises(FileExistsError):
        reg.save(record, state)
    reg.save(record, state, overwrite=True)


def test_registry_resolve_directory_path(tmp_path: Path) -> None:
    reg = FinetuneRegistry(root=tmp_path)
    record, state = _make_record()
    reg.save(record, state)

    # Resolving by directory path should also work.
    direct = reg.resolve(str(reg.adapter_dir("ft_v1")))
    assert direct == reg.adapter_dir("ft_v1")

    # And by name.
    by_name = reg.resolve("ft_v1")
    assert by_name == reg.adapter_dir("ft_v1")


def test_registry_delete(tmp_path: Path) -> None:
    reg = FinetuneRegistry(root=tmp_path)
    record, state = _make_record()
    reg.save(record, state)
    assert reg.exists("ft_v1")
    reg.delete("ft_v1")
    assert not reg.exists("ft_v1")
    assert reg.list() == []


# ─── Apply ──────────────────────────────────────────────────────────────────


def test_load_finetune_into_model_copies_weights(tmp_path: Path) -> None:
    reg = FinetuneRegistry(root=tmp_path)
    model = _ToyBoltz()

    # Build a record whose state-dict matches the toy model exactly.
    new_w = torch.full_like(
        model.affinity_module.affinity_heads[0].weight, 7.0
    )
    record = FinetuneRecord(
        name="ft_apply",
        boltz_version="test",
        config=FinetuneConfig(
            target_spec="affinity_heads",
            target_patterns=list(TARGET_PRESETS["affinity_heads"]),
            num_trainable_params=int(new_w.numel()),
        ),
        trained_params=["affinity_module.affinity_heads.0.weight"],
    )
    state = {"affinity_module.affinity_heads.0.weight": new_w}
    reg.save(record, state)

    loaded = load_finetune_into_model(model, "ft_apply", registry=reg)
    assert loaded.name == "ft_apply"
    assert torch.equal(
        model.affinity_module.affinity_heads[0].weight, new_w
    )
    # Untouched layer stayed unchanged.
    assert not torch.equal(
        model.trunk.weight, torch.full_like(model.trunk.weight, 7.0)
    )


def test_load_finetune_into_model_strict_missing_raises(tmp_path: Path) -> None:
    reg = FinetuneRegistry(root=tmp_path)
    model = _ToyBoltz()

    record = FinetuneRecord(
        name="ft_missing",
        boltz_version="test",
        config=FinetuneConfig(
            target_spec="affinity_heads",
            target_patterns=["whatever"],
            num_trainable_params=1,
        ),
        trained_params=["nonexistent.layer.weight"],
    )
    state = {"nonexistent.layer.weight": torch.zeros(4, 4)}
    reg.save(record, state)

    with pytest.raises(RuntimeError, match="do not exist"):
        load_finetune_into_model(model, "ft_missing", registry=reg, strict=True)

    # strict=False logs a warning but doesn't raise.
    load_finetune_into_model(
        model, "ft_missing", registry=reg, strict=False
    )


def test_load_finetune_into_model_shape_mismatch(tmp_path: Path) -> None:
    reg = FinetuneRegistry(root=tmp_path)
    model = _ToyBoltz()

    record = FinetuneRecord(
        name="ft_shape",
        boltz_version="test",
        config=FinetuneConfig(
            target_spec="affinity_heads",
            target_patterns=["whatever"],
            num_trainable_params=1,
        ),
        trained_params=["affinity_module.affinity_heads.0.weight"],
    )
    # Wrong shape: model expects (4, 4); save (8, 8).
    state = {"affinity_module.affinity_heads.0.weight": torch.zeros(8, 8)}
    reg.save(record, state)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_finetune_into_model(model, "ft_shape", registry=reg)


# ─── Adapter (de)serialisation ──────────────────────────────────────────────


def test_finetune_record_json_roundtrip() -> None:
    record, _ = _make_record()
    blob = json.loads(record.to_json())
    rebuilt = FinetuneRecord.from_dict(blob)
    assert rebuilt.name == record.name
    assert rebuilt.config.target_spec == record.config.target_spec
    assert rebuilt.trained_params == record.trained_params


def test_default_target_spec_is_full_affinity_module() -> None:
    """Sanity: the default target preset really targets *every* affinity-module
    parameter, not just the heads (this is the headline difference from the
    LoRA default and the main reason this baseline exists)."""
    model = _ToyBoltz()
    sel = _select_trainable_params(model, resolve_targets("affinity_module"))
    sel_names = {n for n, _ in sel}
    all_am_names = {
        n for n, _ in model.named_parameters()
        if n.startswith("affinity_module.")
    }
    assert sel_names == all_am_names


# ─── L2-SP regularization ───────────────────────────────────────────────────


def test_l2_sp_penalty_known_value() -> None:
    current = {
        "a": torch.tensor([1.0, 2.0, 3.0]),
        "b": torch.tensor([[0.0, 0.0]]),
    }
    initial = {
        "a": torch.tensor([0.0, 2.0, 5.0]),  # diff = [1, 0, -2] -> sq sum 5
        "b": torch.tensor([[3.0, 4.0]]),  # diff = [-3, -4] -> sq sum 25
    }
    penalty = l2_sp_penalty(current, initial)
    assert torch.allclose(penalty, torch.tensor(30.0))


def test_l2_sp_penalty_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        l2_sp_penalty({}, {})


def test_l2_sp_penalty_missing_key_raises() -> None:
    current = {"a": torch.tensor([1.0])}
    with pytest.raises(KeyError):
        l2_sp_penalty(current, {})


def test_add_l2_sp_penalty_zero_weight_is_noop() -> None:
    """Regression safety: l2_sp_weight=0.0 must reproduce prior behavior
    exactly — the penalty must not even be computed."""
    loss = torch.tensor(3.5, requires_grad=True)
    current = {"a": torch.tensor([100.0])}  # would blow up if penalty ran
    initial = None  # no snapshot exists when L2-SP is disabled
    out = add_l2_sp_penalty(loss, 0.0, current, initial)
    assert out is loss
    assert out.item() == pytest.approx(3.5)


def test_add_l2_sp_penalty_positive_weight_adds_expected_amount() -> None:
    loss = torch.tensor(3.5)
    current = {"a": torch.tensor([2.0]), "b": torch.tensor([0.0])}
    initial = {"a": torch.tensor([0.0]), "b": torch.tensor([0.0])}
    # penalty = (2-0)^2 + (0-0)^2 = 4.0
    weight = 0.5
    out_zero = add_l2_sp_penalty(loss, 0.0, current, initial)
    out_pos = add_l2_sp_penalty(loss, weight, current, initial)
    assert out_zero.item() == pytest.approx(3.5)
    assert out_pos.item() == pytest.approx(3.5 + weight * 4.0)
    assert out_pos.item() > out_zero.item()


def test_add_l2_sp_penalty_missing_snapshot_raises() -> None:
    loss = torch.tensor(1.0)
    with pytest.raises(ValueError, match="no initial parameter snapshot"):
        add_l2_sp_penalty(loss, 1.0, {"a": torch.tensor([1.0])}, None)


def test_l2_sp_pulls_trained_params_toward_initial_values() -> None:
    """End-to-end (but GPU-free, model-free) check of the actual
    regularization behaviour: a large l2_sp_weight should keep trained
    parameters measurably closer to their initial values than training
    with l2_sp_weight=0.0, on the same tiny synthetic problem."""

    def run(l2_sp_weight: float, steps: int = 60) -> float:
        torch.manual_seed(0)
        model = nn.Linear(4, 4)
        initial_params = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }
        optim = torch.optim.AdamW(model.parameters(), lr=0.1)
        x = torch.randn(8, 4)
        # Target is far from the model's initial behavior so gradient
        # descent on the primary loss alone pulls params away from init.
        y = torch.randn(8, 4) * 10.0
        for _ in range(steps):
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            current_params = dict(model.named_parameters())
            loss = add_l2_sp_penalty(
                loss, l2_sp_weight, current_params, initial_params,
            )
            loss.backward()
            optim.step()
        return sum(
            (p.detach() - initial_params[n]).pow(2).sum().item()
            for n, p in model.named_parameters()
        )

    dist_no_reg = run(0.0)
    dist_with_reg = run(10.0)
    assert dist_with_reg < dist_no_reg
