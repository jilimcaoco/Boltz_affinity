"""Unit tests for the frozen-trunk train cache (boltz.lora.train.TrunkFeatureCache).

CPU-cheap: uses a toy module, no Boltz2 checkpoint required.
"""

from __future__ import annotations

import time

import torch
from torch import nn

from boltz.lora.train import (
    TrunkFeatureCache,
    _configure_train_cache,
    trunk_is_frozen,
)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pairformer_module = nn.Linear(4, 4)
        self.msa_module = nn.Linear(4, 4)
        self.input_embedder = nn.Linear(4, 4)
        self.affinity_module = nn.Linear(4, 4)


def _freeze_trunk(m: _ToyModel) -> None:
    for name in ("pairformer_module", "msa_module", "input_embedder"):
        for p in getattr(m, name).parameters():
            p.requires_grad = False


def test_trunk_is_frozen_detection():
    m = _ToyModel()
    _freeze_trunk(m)
    assert trunk_is_frozen(m) is True
    # A single trainable trunk param flips it.
    next(m.pairformer_module.parameters()).requires_grad = True
    assert trunk_is_frozen(m) is False


def test_feats_roundtrip_embeds_key(tmp_path):
    c = TrunkFeatureCache(tmp_path, "sha", trunk_frozen=True)
    row = {"receptor": ["/tmp/r.yaml"], "ligand": ["CCO"], "structure": ["/tmp/s.pdb"]}
    key = c.feats_key(row)
    feats = {"coords": torch.randn(1, 3), "name": ["x"]}
    assert c.load_feats(key, "cpu") is None  # miss before save
    c.save_feats(key, feats)
    got = c.load_feats(key, "cpu")
    assert got is not None
    assert torch.allclose(got["coords"], feats["coords"])
    assert got["__cache_key__"] == key


def test_z_cache_is_keyed_on_recycling(tmp_path):
    c = TrunkFeatureCache(tmp_path, "sha", trunk_frozen=True)
    key = "k"
    z = torch.randn(1, 8, 8, 4)
    c.save_z(key, 1, z)
    assert torch.allclose(c.load_z(key, 1, "cpu"), z)
    assert c.load_z(key, 3, "cpu") is None  # different recycling → miss


def test_key_changes_with_ligand_and_structure(tmp_path):
    c = TrunkFeatureCache(tmp_path, "sha", trunk_frozen=True)
    base = {"receptor": ["/tmp/r.yaml"], "ligand": ["CCO"], "structure": ["/tmp/s.pdb"]}
    k0 = c.feats_key(base)
    k1 = c.feats_key({**base, "ligand": ["CCN"]})
    assert k0 != k1


def test_budget_evicts_oldest_first(tmp_path):
    c = TrunkFeatureCache(tmp_path, "sha", trunk_frozen=True, max_bytes=3000)
    keys = [f"k{i}" for i in range(6)]
    for k in keys:
        c.save_z(k, 1, torch.randn(4, 4, 4))
        time.sleep(0.01)  # distinct mtimes for deterministic LRU order

    total = sum(p.stat().st_size for p in (c.root / "z").glob("*.pt"))
    assert total <= c.max_bytes
    assert c._z_path(keys[-1], 1).exists()   # newest survives
    assert not c._z_path(keys[0], 1).exists()  # oldest evicted


def test_unbounded_by_default_never_evicts(tmp_path):
    c = TrunkFeatureCache(tmp_path, "sha", trunk_frozen=True)  # no max_bytes
    for i in range(6):
        c.save_z(f"k{i}", 1, torch.randn(4, 4, 4))
    assert len(list((c.root / "z").glob("*.pt"))) == 6


def test_max_gb_env_var_parsed(monkeypatch, tmp_path):
    monkeypatch.setenv("BOLTZ_TRAIN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BOLTZ_TRAIN_CACHE_MAX_GB", "0.000001")  # 1000 bytes
    m = _ToyModel()
    _freeze_trunk(m)
    cache = _configure_train_cache(m, "sha")
    assert cache is not None
    assert cache.max_bytes == 1000
    monkeypatch.delenv("BOLTZ_TRAIN_CACHE_DIR", raising=False)
    monkeypatch.delenv("BOLTZ_TRAIN_CACHE_MAX_GB", raising=False)
    _configure_train_cache(m, "sha")  # reset module-global for other tests


def test_cache_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv("BOLTZ_TRAIN_CACHE_DIR", raising=False)
    m = _ToyModel()
    _freeze_trunk(m)
    assert _configure_train_cache(m, "sha") is None


def test_cache_enabled_and_frozen_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("BOLTZ_TRAIN_CACHE_DIR", str(tmp_path))
    m = _ToyModel()
    _freeze_trunk(m)
    cache = _configure_train_cache(m, "sha")
    assert cache is not None and cache.trunk_frozen is True
    # cleanup global so other tests are unaffected
    monkeypatch.delenv("BOLTZ_TRAIN_CACHE_DIR", raising=False)
    _configure_train_cache(m, "sha")
