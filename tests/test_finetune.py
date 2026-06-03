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
from torch import nn

from boltz.finetune import (
    FinetuneConfig,
    FinetuneRecord,
    FinetuneRegistry,
    load_finetune_into_model,
)
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
