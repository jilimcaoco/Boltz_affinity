"""Unit tests for compute_shapley_interactions.py (Task 2): Shapley
efficiency, Möbius sign convention on synthetic redundant/synergistic cases,
and the experiment-name parser. Pure combinatorics/math -- no GPU, no real
ablation data needed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis_scripts"))
import compute_shapley_interactions as si  # noqa: E402

D, Z, S = "distogram", "z_trunk", "s_inputs"


def _v(empty, d, z, s, dz, ds, zs, dzs):
    """Build a characteristic function dict from the 8 factorial values."""
    return {
        frozenset(): empty,
        frozenset({D}): d, frozenset({Z}): z, frozenset({S}): s,
        frozenset({D, Z}): dz, frozenset({D, S}): ds, frozenset({Z, S}): zs,
        frozenset({D, Z, S}): dzs,
    }


class TestShapleyEfficiency:
    def test_efficiency_holds_for_arbitrary_values(self):
        # Efficiency is an algebraic identity -- must hold for *any* v.
        v = _v(empty=0.0, d=3.0, z=7.0, s=2.0, dz=12.0, ds=6.0, zs=15.0, dzs=20.0)
        phi = si.shapley_values(v)
        si.assert_efficiency(phi, v)  # should not raise
        assert sum(phi.values()) == pytest.approx(v[frozenset({D, Z, S})] - v[frozenset()])

    def test_efficiency_holds_for_negative_and_zero_values(self):
        v = _v(empty=-5.0, d=-5.0, z=-5.0, s=-5.0, dz=-5.0, ds=-5.0, zs=-5.0, dzs=-5.0)
        phi = si.shapley_values(v)
        si.assert_efficiency(phi, v)
        assert all(p == pytest.approx(0.0) for p in phi.values())

    def test_symmetric_players_get_equal_shapley_value(self):
        # d, z, s enter v() symmetrically -> by the symmetry axiom each
        # should get exactly v(C)/3 when v(empty)=0.
        v = _v(empty=0.0, d=10.0, z=10.0, s=10.0, dz=20.0, ds=20.0, zs=20.0, dzs=30.0)
        phi = si.shapley_values(v)
        for c in si.CHANNELS:
            assert phi[c] == pytest.approx(10.0)

    def test_efficiency_violation_is_detected(self):
        v = _v(empty=0.0, d=3.0, z=7.0, s=2.0, dz=12.0, ds=6.0, zs=15.0, dzs=20.0)
        bad_phi = {D: 1.0, Z: 1.0, S: 1.0}  # doesn't sum to v(C)-v(empty)=20
        with pytest.raises(AssertionError):
            si.assert_efficiency(bad_phi, v)


class TestMobiusSignConvention:
    def test_redundant_channels_give_negative_interaction(self):
        # d and z carry the *same* information: either alone gets you all
        # the way to the pair's value, so together they add nothing extra.
        v = _v(empty=0.0, d=10.0, z=10.0, s=0.0, dz=10.0, ds=10.0, zs=10.0, dzs=10.0)
        mobius = si.mobius_interactions(v)
        assert mobius[(D, Z)] == pytest.approx(10.0 - 10.0 - 10.0 + 0.0)  # -10
        assert mobius[(D, Z)] < 0

    def test_synergistic_channels_give_positive_interaction(self):
        # neither d nor z alone helps, but together they unlock signal.
        v = _v(empty=0.0, d=0.0, z=0.0, s=0.0, dz=10.0, ds=0.0, zs=0.0, dzs=10.0)
        mobius = si.mobius_interactions(v)
        assert mobius[(D, Z)] == pytest.approx(10.0)
        assert mobius[(D, Z)] > 0

    def test_additive_independent_channels_give_zero_interaction(self):
        v = _v(empty=0.0, d=5.0, z=3.0, s=0.0, dz=8.0, ds=5.0, zs=3.0, dzs=8.0)
        mobius = si.mobius_interactions(v)
        assert mobius[(D, Z)] == pytest.approx(0.0, abs=1e-9)


class TestExperimentNameParser:
    def test_bare_cell_names(self):
        assert si.parse_experiment_name("baseline") == ("baseline", "zero", None)
        assert si.parse_experiment_name("bias_only") == ("bias_only", "zero", None)

    def test_resample_variant(self):
        assert si.parse_experiment_name("no_distogram__resample__d11") == ("no_distogram", "resample", 11)

    def test_mean_variant(self):
        assert si.parse_experiment_name("bias_only__mean") == ("bias_only", "mean", None)

    def test_unrelated_name_returns_none(self):
        assert si.parse_experiment_name("lig_noise_1p0_s101") is None
        assert si.parse_experiment_name("only_atom_encoder") is None


class TestCellValuesToV:
    def test_maps_cell_names_to_frozensets(self):
        cell_values = {
            "baseline": 30.0, "no_distogram": 20.0, "no_z_trunk": 10.0, "no_s_inputs": 10.0,
            "distogram_only": 10.0, "z_trunk_only": 20.0, "s_inputs_only": 20.0, "bias_only": 0.0,
        }
        v = si.cell_values_to_v(cell_values)
        assert v[frozenset({D, Z, S})] == 30.0
        assert v[frozenset()] == 0.0
        assert v[frozenset({D})] == 10.0

    def test_ignores_unknown_cell_names(self):
        v = si.cell_values_to_v({"baseline": 1.0, "not_a_cell": 99.0})
        assert frozenset({D, Z, S}) in v
        assert len(v) == 1


class TestBootstrapAggregate:
    def test_returns_nan_for_empty_input(self):
        import numpy as np
        agg = si.bootstrap_aggregate([], 100, np.random.default_rng(0))
        assert agg["n_receptors"] == 0
        import math
        assert math.isnan(agg["mean"])

    def test_mean_matches_point_estimate(self):
        import numpy as np
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        agg = si.bootstrap_aggregate(vals, 500, np.random.default_rng(1))
        assert agg["mean"] == pytest.approx(3.0)
        assert agg["ci_low"] < 3.0 < agg["ci_high"]
