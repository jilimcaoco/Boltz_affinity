"""Unit tests for the rescore-time trunk cache.

CPU-cheap: uses a toy module, no Boltz2 checkpoint required. The GPU-level
numerical-equivalence check lives in the LSD_finetuning project's
``tools/verify_trunk_cache.py``.
"""

from __future__ import annotations

import torch
from torch import nn

from boltz.affinity_rescoring.trunk_cache import (
    RescoreTrunkCache,
    feats_digest,
    get_cache,
    reset_cache,
    trunk_fingerprint,
)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pairformer_module = nn.Linear(4, 4)
        self.msa_module = nn.Linear(4, 4)
        self.input_embedder = nn.Linear(4, 4)
        # Affinity side: must NOT contribute to the trunk fingerprint.
        self.affinity_module = nn.Linear(4, 4)


def test_fingerprint_ignores_affinity_module():
    m = _ToyModel()
    before = trunk_fingerprint(m)
    with torch.no_grad():
        m.affinity_module.weight.add_(1.0)
    assert trunk_fingerprint(m) == before, "affinity weights must not affect the key"


def test_fingerprint_tracks_trunk_weights():
    m = _ToyModel()
    before = trunk_fingerprint(m)
    with torch.no_grad():
        m.pairformer_module.weight.add_(1.0)
    assert trunk_fingerprint(m) != before, "trunk weights must affect the key"


def test_feats_digest_is_order_independent_and_value_sensitive():
    a = {"x": torch.ones(3), "y": torch.zeros(2, 2)}
    b = {"y": torch.zeros(2, 2), "x": torch.ones(3)}
    c = {"x": torch.ones(3) * 2, "y": torch.zeros(2, 2)}
    assert feats_digest(a) == feats_digest(b)
    assert feats_digest(a) != feats_digest(c)


def test_roundtrip(tmp_path):
    cache = RescoreTrunkCache(tmp_path, "sha")
    key = cache.key("digest", 1)
    assert cache.load(key, "cpu") is None
    z = torch.randn(1, 4, 4, 8)
    cache.save(key, z)
    got = cache.load(key, "cpu")
    assert got is not None
    assert torch.equal(got, z)


def test_recycling_steps_change_the_key(tmp_path):
    cache = RescoreTrunkCache(tmp_path, "sha")
    assert cache.key("digest", 1) != cache.key("digest", 3)


def test_lru_budget_evicts(tmp_path):
    z = torch.randn(1, 32, 32, 32)  # ~128 KB
    cache = RescoreTrunkCache(tmp_path, "sha", max_bytes=200_000)
    for i in range(5):
        cache.save(cache.key(f"d{i}", 1), z)
    remaining = list((tmp_path / "z").glob("*.pt"))
    assert len(remaining) < 5, "LRU eviction should have trimmed the cache"


def test_disabled_by_default(monkeypatch):
    reset_cache()
    monkeypatch.delenv("BOLTZ_RESCORE_CACHE_DIR", raising=False)
    assert get_cache(_ToyModel()) is None


def test_enabled_by_env(monkeypatch, tmp_path):
    reset_cache()
    monkeypatch.setenv("BOLTZ_RESCORE_CACHE_DIR", str(tmp_path))
    cache = get_cache(_ToyModel())
    assert cache is not None
    reset_cache()
