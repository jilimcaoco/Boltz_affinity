"""Unit tests for trunk_cache.py: cache I/O, token crop/pad, donor matching,
and the resample/mean channel-substitution helpers (Task 1 / Task 4).

These test the pure tensor/bookkeeping logic with synthetic data -- they do
not require a GPU, a real Boltz2 checkpoint, or real featurized complexes.
End-to-end correctness against the real model is verify_trunk_cache.py's job
(Task 4's fidelity gate), which needs a checkpoint and must run on the
cluster.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_scripts"))
import trunk_cache as tc  # noqa: E402


def _make_entry(n_tokens, token_z=8, token_s=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "z": torch.randn(n_tokens, n_tokens, token_z, generator=g),
        "s_inputs": torch.randn(n_tokens, token_s, generator=g),
        "token_repr_pos": torch.randn(n_tokens, 3, generator=g),
        "n_tokens": n_tokens,
        "use_kernels": False,
        "meta": {},
    }


class TestCacheIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        z = torch.randn(1, 10, 10, 8)
        s_inputs = torch.randn(1, 10, 6)
        token_repr_pos = torch.randn(10, 3)

        tc.save_cache_entry(
            tmp_path, "AA2AR", "lig1",
            z=z, s_inputs=s_inputs, token_repr_pos=token_repr_pos,
            use_kernels=True, meta={"foo": "bar"},
        )
        entry = tc.load_cache_entry(tmp_path, "AA2AR", "lig1")
        assert entry is not None
        assert entry["n_tokens"] == 10
        assert entry["use_kernels"] is True
        assert entry["z"].shape == (10, 10, 8)
        assert entry["s_inputs"].shape == (10, 6)
        assert entry["token_repr_pos"].shape == (10, 3)
        assert entry["meta"]["foo"] == "bar"
        # z is stored fp16 for disk compactness
        assert entry["z"].dtype == torch.float16

    def test_load_missing_returns_none(self, tmp_path):
        assert tc.load_cache_entry(tmp_path, "AA2AR", "nope") is None

    def test_list_cached_ligands(self, tmp_path):
        for lig in ["lig_b", "lig_a", "lig_c"]:
            tc.save_cache_entry(
                tmp_path, "AA2AR", lig,
                z=torch.randn(1, 5, 5, 4), s_inputs=torch.randn(1, 5, 3),
                token_repr_pos=torch.randn(5, 3), use_kernels=False,
            )
        assert tc.list_cached_ligands(tmp_path, "AA2AR") == ["lig_a", "lig_b", "lig_c"]
        assert tc.list_cached_ligands(tmp_path, "NOSUCH") == []


class TestCropOrPadTokens:
    def test_exact_match_is_noop(self):
        t = torch.randn(10, 6)
        out, delta = tc.crop_or_pad_tokens(t, 10, token_dims=(0,))
        assert delta == 0
        assert torch.equal(out, t)

    def test_crop_1d(self):
        t = torch.arange(10.0).reshape(10, 1)
        out, delta = tc.crop_or_pad_tokens(t, 6, token_dims=(0,))
        assert delta == -4
        assert out.shape == (6, 1)
        assert torch.equal(out[:, 0], torch.arange(6.0))

    def test_pad_1d(self):
        t = torch.ones(4, 3)
        out, delta = tc.crop_or_pad_tokens(t, 7, token_dims=(0,))
        assert delta == 3
        assert out.shape == (7, 3)
        assert torch.equal(out[:4], t)
        assert torch.equal(out[4:], torch.zeros(3, 3))

    def test_pad_2d_square(self):
        n0 = 5
        t = torch.randn(n0, n0, 4)
        out, delta = tc.crop_or_pad_tokens(t, 8, token_dims=(0, 1))
        assert delta == 3
        assert out.shape == (8, 8, 4)
        # original block preserved exactly
        assert torch.equal(out[:n0, :n0, :], t)
        # padded region is zero
        assert torch.equal(out[n0:, :, :], torch.zeros(3, 8, 4))
        assert torch.equal(out[:, n0:, :], torch.zeros(8, 3, 4))

    def test_crop_2d_square(self):
        t = torch.randn(10, 10, 4)
        out, delta = tc.crop_or_pad_tokens(t, 6, token_dims=(0, 1))
        assert delta == -4
        assert out.shape == (6, 6, 4)
        assert torch.equal(out, t[:6, :6, :])


class TestDonorMatching:
    def test_no_candidates_returns_none(self, tmp_path):
        rng = __import__("numpy").random.default_rng(0)
        assert tc.find_donor(tmp_path, "AA2AR", "query", 100, rng) is None

    def test_excludes_self(self, tmp_path):
        import numpy as np
        tc.save_cache_entry(tmp_path, "AA2AR", "query", z=torch.randn(1, 10, 10, 4),
                             s_inputs=torch.randn(1, 10, 3), token_repr_pos=torch.randn(10, 3),
                             use_kernels=False)
        assert tc.find_donor(tmp_path, "AA2AR", "query", 10, np.random.default_rng(0)) is None

    def test_exact_match_preferred(self, tmp_path):
        import numpy as np
        for lig, n in [("query", 10), ("near", 12), ("exact", 10)]:
            tc.save_cache_entry(tmp_path, "AA2AR", lig, z=torch.randn(1, n, n, 4),
                                 s_inputs=torch.randn(1, n, 3), token_repr_pos=torch.randn(n, 3),
                                 use_kernels=False)
        match = tc.find_donor(tmp_path, "AA2AR", "query", 10, np.random.default_rng(0))
        assert match.donor_ligand_id == "exact"
        assert match.match_kind == "exact"
        assert match.token_delta == 0

    def test_nearest_fallback(self, tmp_path):
        import numpy as np
        for lig, n in [("query", 10), ("far", 30), ("near", 12)]:
            tc.save_cache_entry(tmp_path, "AA2AR", lig, z=torch.randn(1, n, n, 4),
                                 s_inputs=torch.randn(1, n, 3), token_repr_pos=torch.randn(n, 3),
                                 use_kernels=False)
        match = tc.find_donor(tmp_path, "AA2AR", "query", 10, np.random.default_rng(0))
        assert match.donor_ligand_id == "near"
        assert match.match_kind == "nearest"
        assert match.token_delta == 10 - 12

    def test_all_other_ligands_excludes_query(self, tmp_path):
        for lig, n in [("query", 10), ("a", 10), ("b", 11)]:
            tc.save_cache_entry(tmp_path, "AA2AR", lig, z=torch.randn(1, n, n, 4),
                                 s_inputs=torch.randn(1, n, 3), token_repr_pos=torch.randn(n, 3),
                                 use_kernels=False)
        others = tc.all_other_ligands(tmp_path, "AA2AR", "query")
        assert sorted(others) == ["a", "b"]


class TestChannelSubstitution:
    def test_substitute_z_trunk_matches_shape(self):
        donor = _make_entry(n_tokens=8, seed=1)
        out, delta = tc.substitute_channel("z_trunk", 12, donor)
        assert out.shape == (12, 12, 8)
        assert delta == 4

    def test_substitute_s_inputs_matches_shape(self):
        donor = _make_entry(n_tokens=8, seed=1)
        out, delta = tc.substitute_channel("s_inputs", 5, donor)
        assert out.shape == (5, 6)
        assert delta == -3

    def test_substitute_distogram_matches_shape(self):
        donor = _make_entry(n_tokens=8, seed=1)
        out, delta = tc.substitute_channel("distogram", 8, donor)
        assert out.shape == (8, 3)
        assert delta == 0

    def test_unknown_channel_raises(self):
        donor = _make_entry(n_tokens=8)
        with pytest.raises(ValueError):
            tc.substitute_channel("not_a_channel", 8, donor)

    def test_mean_channel_is_elementwise_average(self):
        e1 = _make_entry(n_tokens=6, token_s=4, seed=1)
        e2 = _make_entry(n_tokens=6, token_s=4, seed=2)
        out = tc.mean_channel("s_inputs", 6, [e1, e2])
        expected = (e1["s_inputs"] + e2["s_inputs"]) / 2
        assert torch.allclose(out, expected)

    def test_mean_channel_requires_donors(self):
        with pytest.raises(ValueError):
            tc.mean_channel("s_inputs", 6, [])

    def test_mean_channel_crops_each_donor_independently(self):
        e_small = _make_entry(n_tokens=4, token_s=3, seed=1)
        e_big = _make_entry(n_tokens=10, token_s=3, seed=2)
        out = tc.mean_channel("s_inputs", 4, [e_small, e_big])
        expected = (e_small["s_inputs"] + e_big["s_inputs"][:4]) / 2
        assert torch.allclose(out, expected)
