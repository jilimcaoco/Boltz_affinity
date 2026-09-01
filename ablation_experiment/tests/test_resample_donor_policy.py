"""Donor-selection policy for the resample operator.

Covers the registry wiring and the two policy rules that the tensor-level
tests in test_trunk_cache.py cannot see, because both live in
run_feature_ablation._run_resample_or_mean:

  * one donor per cell (not one per channel)
  * exact token-count match required (no crop/pad fallback)

Exercising _run_resample_or_mean directly needs a checkpoint and a GPU, so the
selection logic is re-derived here against a real DonorPool and the shared
registry. End-to-end fidelity is verify_trunk_cache.py's job.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SCRIPTS = Path(__file__).resolve().parent.parent / "experiment_scripts"
sys.path.insert(0, str(_SCRIPTS))

import trunk_cache as tc  # noqa: E402
from run_feature_ablation import EXPERIMENTS  # noqa: E402


def _pool(tmp_path, spec, token_z=4, token_s=3):
    """spec: {ligand_id: n_tokens} -> a DonorPool over those cached entries."""
    for lid, n in spec.items():
        tc.save_cache_entry(
            tmp_path, "AA2AR", lid,
            z=torch.randn(1, n, n, token_z),
            s_inputs=torch.randn(1, n, token_s),
            token_repr_pos=torch.randn(1, n, 3),  # production shape
            use_kernels=False,
        )
    return tc.DonorPool(tmp_path, "AA2AR", list(spec))


def _select(pool, query, n_tokens, seed):
    """The donor policy as applied in _run_resample_or_mean."""
    match = pool.find_donor(query, n_tokens, tc.donor_rng(seed, query))
    if match is None or match.match_kind != "exact":
        return None
    return match.donor_ligand_id


class TestOneDonorPerCell:
    def test_all_channels_share_one_donor(self, tmp_path):
        pool = _pool(tmp_path, {f"d{i}": 10 for i in range(8)} | {"q": 10})
        donor = _select(pool, "q", 10, 11)
        assert donor is not None
        # A cell ablating three channels substitutes from this single entry.
        entry = pool.entry(donor)
        shapes = {ch: tc.substitute_channel(ch, 10, entry)[0].shape
                  for ch in ("z_trunk", "s_inputs", "distogram")}
        assert shapes["z_trunk"] == (10, 10, 4)
        assert shapes["s_inputs"] == (10, 3)
        assert shapes["distogram"] == (10, 3)

    def test_donor_never_the_query(self, tmp_path):
        pool = _pool(tmp_path, {"q": 10, "d1": 10})
        assert _select(pool, "q", 10, 11) == "d1"

    def test_draw_is_stable_within_a_cell_and_varies_across_seeds(self, tmp_path):
        pool = _pool(tmp_path, {f"d{i}": 10 for i in range(30)} | {"q": 10})
        assert _select(pool, "q", 10, 11) == _select(pool, "q", 10, 11)
        assert len({_select(pool, "q", 10, s) for s in (11, 22, 33, 44, 55)}) > 1


class TestExactMatchRequired:
    def test_nearest_match_is_rejected(self, tmp_path):
        pool = _pool(tmp_path, {"d_near": 12, "d_far": 40})
        assert pool.find_donor("q", 10, tc.donor_rng(11, "q")).match_kind == "nearest"
        assert _select(pool, "q", 10, 11) is None

    def test_exact_match_preferred_over_nearest(self, tmp_path):
        pool = _pool(tmp_path, {"d_near": 12, "d_exact": 10})
        assert _select(pool, "q", 10, 11) == "d_exact"

    def test_empty_pool_yields_no_donor(self, tmp_path):
        pool = _pool(tmp_path, {"q": 10})
        assert _select(pool, "q", 10, 11) is None


class TestIdentityControl:
    def test_registered_and_covers_all_channels(self):
        exp = EXPERIMENTS["identity__resample__self"]
        assert exp.operator == "resample"
        assert exp.donor_self is True
        assert set(exp.resample_channels) == {"distogram", "z_trunk", "s_inputs"}

    def test_self_substitution_is_a_no_op(self, tmp_path):
        """Donating from the query itself must return exactly what baseline
        would have used -- that is the whole point of the control."""
        pool = _pool(tmp_path, {"q": 10})
        entry = pool.entry("q")
        for ch, key in (("z_trunk", "z"), ("s_inputs", "s_inputs"),
                        ("distogram", "token_repr_pos")):
            out, delta = tc.substitute_channel(ch, 10, entry)
            assert delta == 0
            assert torch.equal(out, entry[key])

    def test_resample_cells_do_not_self_donate(self):
        for name, exp in EXPERIMENTS.items():
            if name != "identity__resample__self":
                assert exp.donor_self is False, name
