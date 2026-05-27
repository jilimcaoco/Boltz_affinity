"""Unit tests for the LoRA subsystem.

These tests run on a tiny toy module so they are CPU-cheap and don't require
the Boltz affinity checkpoint. Integration tests that actually load Boltz2
should live separately under ``tests/integration/`` (not added here).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from boltz.lora import (
    LoRAAdapter,
    LoRAConfig,
    LoRALinear,
    LoRARegistry,
    apply_lora,
    merge_lora,
    remove_lora,
)
from boltz.lora.data import LoRADataset, lora_collate, parse_lora_csv
from boltz.lora.inject import load_lora_state_dict, lora_state_dict
from boltz.lora.losses import (
    BUILTIN_LOSSES,
    call_loss,
    load_loss_from_spec,
)
from boltz.lora.targets import TARGET_PRESETS, resolve_targets


# ─── Fixtures ────────────────────────────────────────────────────────────────


class _ToyAffinity(nn.Module):
    """Tiny stand-in for the affinity stack: two linears + a head."""

    def __init__(self) -> None:
        super().__init__()
        self.pairformer_stack = nn.Sequential(
            nn.Linear(8, 8),  # name => pairformer_stack.0
            nn.Linear(8, 8),  # name => pairformer_stack.1
        )
        self.affinity_heads = nn.Module()
        self.affinity_heads.affinity_out_mlp = nn.Sequential(
            nn.Linear(8, 8),  # name => affinity_heads.affinity_out_mlp.0
            nn.Linear(8, 4),  # name => affinity_heads.affinity_out_mlp.2 idx? 1
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.pairformer_stack(x)
        h = self.affinity_heads.affinity_out_mlp(h)
        return {"affinity_pred_value": h.mean(dim=-1, keepdim=True)}


@pytest.fixture
def toy() -> _ToyAffinity:
    torch.manual_seed(0)
    return _ToyAffinity()


# ─── LoRALinear ──────────────────────────────────────────────────────────────


def test_lora_linear_identity_at_init():
    """At init, lora_B is zero so the layer matches its base exactly."""
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2, alpha=4.0)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x))


def test_lora_linear_changes_after_perturbing_B():
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2, alpha=4.0)
    with torch.no_grad():
        lora.lora_B.fill_(0.1)
    x = torch.randn(3, 8)
    assert not torch.allclose(lora(x), base(x))


def test_lora_linear_merge_unmerge_roundtrip():
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2, alpha=4.0)
    with torch.no_grad():
        lora.lora_A.normal_(0, 0.1)
        lora.lora_B.normal_(0, 0.1)
    x = torch.randn(3, 8)
    before = lora(x).clone()
    lora.merge_()
    assert lora.merged
    merged = lora(x)
    lora.unmerge_()
    assert not lora.merged
    restored = lora(x)
    assert torch.allclose(before, merged, atol=1e-5)
    assert torch.allclose(before, restored, atol=1e-5)


def test_lora_linear_rejects_nonpositive_rank():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), r=0)


# ─── Injection ───────────────────────────────────────────────────────────────


def test_apply_lora_matches_expected_layers(toy):
    adapted = apply_lora(
        toy,
        target_patterns=[r"affinity_heads\.affinity_out_mlp\.\d+$"],
        r=4,
        alpha=8.0,
    )
    assert adapted == sorted([
        "affinity_heads.affinity_out_mlp.0",
        "affinity_heads.affinity_out_mlp.1",
    ])
    # Only LoRA params are trainable.
    trainables = {n for n, p in toy.named_parameters() if p.requires_grad}
    assert all(n.endswith(".lora_A") or n.endswith(".lora_B") for n in trainables)
    assert len(trainables) == 4  # 2 layers × {A,B}


def test_apply_lora_raises_when_no_match(toy):
    with pytest.raises(ValueError, match="zero layers"):
        apply_lora(toy, target_patterns=[r"this_will_never_match_anything"], r=2)


def test_remove_lora_restores_originals(toy):
    apply_lora(toy, [r"affinity_heads\.affinity_out_mlp\.\d+$"], r=2)
    removed = remove_lora(toy)
    assert len(removed) == 2
    # No LoRALinear left anywhere.
    assert not any(isinstance(m, LoRALinear) for m in toy.modules())


def test_lora_state_dict_roundtrip(toy):
    apply_lora(toy, [r"pairformer_stack\.\d+$"], r=3, alpha=6.0)
    # Mutate weights so saved != re-injected default.
    for m in toy.modules():
        if isinstance(m, LoRALinear):
            with torch.no_grad():
                m.lora_A.normal_(0, 0.5)
                m.lora_B.normal_(0, 0.5)
    state = lora_state_dict(toy)
    assert all(k.endswith(".lora_A") or k.endswith(".lora_B") for k in state)

    # Build a fresh model and apply lora, then load.
    fresh = _ToyAffinity()
    apply_lora(fresh, [r"pairformer_stack\.\d+$"], r=3, alpha=6.0)
    load_lora_state_dict(fresh, state, strict=True)
    fresh_state = lora_state_dict(fresh)
    for k in state:
        assert torch.allclose(state[k], fresh_state[k])


def test_merge_lora_unwraps_modules(toy):
    apply_lora(toy, [r"pairformer_stack\.\d+$"], r=2)
    merged = merge_lora(toy)
    assert len(merged) == 2
    assert not any(isinstance(m, LoRALinear) for m in toy.modules())


# ─── Targets ─────────────────────────────────────────────────────────────────


def test_resolve_targets_presets():
    assert resolve_targets("heads") == TARGET_PRESETS["heads"]
    assert resolve_targets("heads_pairformer") == TARGET_PRESETS["heads_pairformer"]


def test_resolve_targets_custom_regex_and_list():
    assert resolve_targets(r"my\.layer$") == (r"my\.layer$",)
    assert resolve_targets(["a", "b"]) == ("a", "b")


# ─── Losses ──────────────────────────────────────────────────────────────────


def _fake_pred() -> dict[str, torch.Tensor]:
    return {
        "affinity_pred_value": torch.tensor([[0.5], [1.5], [2.0]]),
        "affinity_logits_binary": torch.tensor([[0.0], [-1.0], [2.0]]),
    }


def _fake_batch() -> dict:
    return {"target": torch.tensor([1.0, 1.0, 0.0])}


@pytest.mark.parametrize("name", sorted(BUILTIN_LOSSES))
def test_builtin_losses_produce_scalar(name):
    fn = load_loss_from_spec(name)
    loss = call_loss(fn, _fake_pred(), _fake_batch())
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_load_loss_unknown_raises():
    with pytest.raises(KeyError):
        load_loss_from_spec("not_a_real_loss")


def test_load_loss_from_file(tmp_path: Path):
    src = tmp_path / "my_loss.py"
    src.write_text(
        "import torch\n"
        "def my_loss(pred, batch, meta=None):\n"
        "    p = pred['affinity_pred_value'].squeeze(-1).float()\n"
        "    t = batch['target'].float()\n"
        "    return ((p - t) ** 2).mean() + 0.01\n"
    )
    fn = load_loss_from_spec(f"{src}:my_loss")
    loss = call_loss(fn, _fake_pred(), _fake_batch())
    assert torch.isfinite(loss)
    assert loss.item() > 0.01  # offset present


def test_load_loss_missing_function(tmp_path: Path):
    src = tmp_path / "empty.py"
    src.write_text("x = 1\n")
    with pytest.raises(AttributeError):
        load_loss_from_spec(f"{src}:nope")


def test_load_loss_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_loss_from_spec(f"{tmp_path}/no_such_file.py:fn")


# ─── Data CSV ────────────────────────────────────────────────────────────────


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_parse_lora_csv_ok(tmp_path: Path):
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, [
        {"ligand": "CCO", "receptor": "r.yaml", "target": "1.5", "structure": "s.pdb"},
        {"ligand": "c1ccccc1", "receptor": "r.yaml", "target": "0.2", "structure": "s2.pdb"},
    ])
    rows, sha = parse_lora_csv(csv_path)
    assert len(rows) == 2
    assert rows[0].target == 1.5
    assert rows[0].structure == "s.pdb"
    assert isinstance(sha, str) and len(sha) == 64


def test_parse_lora_csv_missing_required(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    _write_csv(csv_path, [{"ligand": "CCO", "target": "1.0"}])
    with pytest.raises(ValueError, match="missing required columns"):
        parse_lora_csv(csv_path)


def test_lora_dataset_rescore_requires_structure(tmp_path: Path):
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, [
        {"ligand": "CCO", "receptor": "r.yaml", "target": "1.5"},
    ])
    with pytest.raises(ValueError, match="structure"):
        LoRADataset(csv_path, mode="rescore")


def test_lora_dataset_collate(tmp_path: Path):
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, [
        {"ligand": "CCO", "receptor": "r.yaml", "target": "1.5", "structure": "s.pdb"},
        {"ligand": "CCN", "receptor": "r.yaml", "target": "0.5", "structure": "s.pdb"},
    ])
    ds = LoRADataset(csv_path, mode="rescore")
    batch = lora_collate([ds[0], ds[1]])
    assert batch["target"].shape == (2,)
    assert batch["ligand"] == ["CCO", "CCN"]


# ─── Registry ────────────────────────────────────────────────────────────────


@pytest.fixture
def registry(tmp_path: Path) -> LoRARegistry:
    return LoRARegistry(root=tmp_path / "loras")


def _make_adapter(name: str = "demo") -> tuple[LoRAAdapter, dict]:
    adapter = LoRAAdapter(
        name=name,
        boltz_version="test",
        base_checkpoint_path="/fake/ckpt.pt",
        base_checkpoint_sha256="deadbeef",
        config=LoRAConfig(
            rank=4, alpha=8.0, dropout=0.0,
            target_spec="heads",
            target_patterns=[r"affinity_heads\.affinity_out_mlp\.\d+$"],
        ),
        adapted_layers=[
            "affinity_heads.affinity_out_mlp.0",
            "affinity_heads.affinity_out_mlp.1",
        ],
    )
    state = {
        "affinity_heads.affinity_out_mlp.0.lora_A": torch.randn(4, 8),
        "affinity_heads.affinity_out_mlp.0.lora_B": torch.randn(8, 4),
        "affinity_heads.affinity_out_mlp.1.lora_A": torch.randn(4, 8),
        "affinity_heads.affinity_out_mlp.1.lora_B": torch.randn(4, 4),
    }
    return adapter, state


def test_registry_save_list_load_delete(registry):
    adapter, state = _make_adapter("alpha")
    registry.save(adapter, state)
    entries = registry.list()
    assert [e["name"] for e in entries] == ["alpha"]
    assert entries[0]["rank"] == 4

    loaded_adapter, loaded_state = registry.load("alpha")
    assert loaded_adapter.name == "alpha"
    assert loaded_adapter.config.rank == 4
    for k, v in state.items():
        assert torch.allclose(v, loaded_state[k])

    registry.delete("alpha")
    assert registry.list() == []
    assert not registry.exists("alpha")


def test_registry_save_rejects_duplicate_without_overwrite(registry):
    adapter, state = _make_adapter("dup")
    registry.save(adapter, state)
    with pytest.raises(FileExistsError):
        registry.save(adapter, state)
    # overwrite=True succeeds.
    registry.save(adapter, state, overwrite=True)


def test_registry_resolve_path(registry, tmp_path: Path):
    adapter, state = _make_adapter("p1")
    target = registry.save(adapter, state)
    # Resolve via path
    assert registry.resolve(str(target)) == target
    # Resolve via name
    assert registry.resolve("p1") == target
    # Missing → error
    with pytest.raises(FileNotFoundError):
        registry.resolve("not_a_real_adapter")


def test_registry_index_round_trips_through_disk(registry, tmp_path: Path):
    adapter, state = _make_adapter("persisted")
    registry.save(adapter, state)
    # Re-open a fresh registry pointing at the same root.
    reopened = LoRARegistry(root=registry.root)
    names = [e["name"] for e in reopened.list()]
    assert names == ["persisted"]


# ─── Adapter dataclass ───────────────────────────────────────────────────────


def test_lora_adapter_to_from_dict_roundtrip():
    adapter, _ = _make_adapter("rt")
    data = adapter.to_dict()
    blob = json.loads(json.dumps(data))
    reborn = LoRAAdapter.from_dict(blob)
    assert reborn.name == adapter.name
    assert reborn.config.rank == adapter.config.rank
    assert reborn.adapted_layers == adapter.adapted_layers
