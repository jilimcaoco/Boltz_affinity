"""Regression tests for the Task 0 tie-handling fix in logauc_utils.py.

Background: bootstrap_logauc used to concatenate actives before decoys and
call Python's stable list.sort() directly, so every tied score resolved in
favor of actives. On data with zero real signal (e.g. a constant model
output) this pushed logAUC toward the theoretical maximum (85.54) instead of
~0, and the exact value depended on concatenation order alone. These tests
guard against that regression.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis_scripts"))
import logauc_utils as lu  # noqa: E402


def _make_names(n_lig, n_dec):
    lig_names = [f"lig{i}" for i in range(n_lig)]
    dec_names = [f"ZINC{i:06d}" for i in range(n_dec)]
    return lig_names, dec_names


class TestConstantScoreLogAUC:
    """The `bias_only` scenario: a model with no per-ligand-varying signal
    should score as random (logAUC ~= 0), not as a perfect classifier."""

    def test_bootstrap_constant_score_is_near_zero(self):
        lig_names, dec_names = _make_names(400, 400)
        lig_scores = [(n, 0.0) for n in lig_names]
        dec_scores = [(n, 0.0) for n in dec_names]
        lig_set, dec_set = set(lig_names), set(dec_names)

        rng = np.random.default_rng(0)
        vals = lu.bootstrap_logauc(lig_scores, dec_scores, lig_set, dec_set, n_boot=300, rng=rng)
        mean_val = float(np.nanmean(vals))

        assert abs(mean_val) < 5.0, f"expected ~0, got {mean_val:.2f}"
        # The old bug pinned this near the theoretical max regardless of n.
        assert mean_val < 20.0

    def test_point_estimate_constant_score_is_near_zero(self):
        lig_names, dec_names = _make_names(500, 500)
        lig_scores = [(n, 0.0) for n in lig_names]
        dec_scores = [(n, 0.0) for n in dec_names]
        lig_set, dec_set = set(lig_names), set(dec_names)

        rng = np.random.default_rng(1)
        val = lu.point_estimate_logauc(lig_scores, dec_scores, lig_set, dec_set, rng, n_shuffles=100)
        assert abs(val) < 5.0, f"expected ~0, got {val:.2f}"

    def test_naive_stable_sort_reproduces_the_original_bug(self):
        """Documents the bug this module fixes: without shuffling ties away,
        concatenating actives first and calling a stable sort inflates
        logAUC toward the theoretical maximum on constant-score data."""
        lig_names, dec_names = _make_names(400, 400)
        lig_scores = [(n, 0.0) for n in lig_names]
        dec_scores = [(n, 0.0) for n in dec_names]
        lig_set, dec_set = set(lig_names), set(dec_names)

        naive = lig_scores + dec_scores
        naive.sort(key=lambda x: x[1])  # stable sort, no tie-break shuffle
        buggy_val = lu.compute_logauc(naive, lig_set, dec_set)

        assert buggy_val > 60.0, (
            "sanity check on the bug itself failed — if this changed, "
            "re-verify the fix is still meaningful"
        )


class TestConcatenationOrderInvariance:
    def test_order_does_not_bias_result_beyond_noise(self):
        lig_names, dec_names = _make_names(300, 300)
        lig_scores = [(n, 0.0) for n in lig_names]
        dec_scores = [(n, 0.0) for n in dec_names]
        lig_set, dec_set = set(lig_names), set(dec_names)

        def run(order, seed):
            rng = np.random.default_rng(seed)
            vals = []
            for _ in range(400):
                shuffled = lu.break_ties(order, rng)
                shuffled.sort(key=lambda x: x[1])
                vals.append(lu.compute_logauc(shuffled, lig_set, dec_set))
            return float(np.mean(vals))

        mean_ligs_first = run(lig_scores + dec_scores, seed=10)
        mean_decs_first = run(dec_scores + lig_scores, seed=20)

        assert abs(mean_ligs_first) < 5.0
        assert abs(mean_decs_first) < 5.0
        assert abs(mean_ligs_first - mean_decs_first) < 5.0


class TestBootstrapStillDetectsRealSignal:
    """Sanity check that the tie fix didn't break ordinary, well-separated
    (non-tied) data — actives should still score as a near-perfect classifier."""

    def test_well_separated_scores_score_near_max(self):
        lig_names, dec_names = _make_names(200, 200)
        rng_data = np.random.default_rng(2)
        lig_scores = [(n, float(rng_data.normal(0.0, 0.1))) for n in lig_names]
        dec_scores = [(n, float(rng_data.normal(10.0, 0.1))) for n in dec_names]
        lig_set, dec_set = set(lig_names), set(dec_names)

        rng = np.random.default_rng(3)
        vals = lu.bootstrap_logauc(lig_scores, dec_scores, lig_set, dec_set, n_boot=100, rng=rng)
        mean_val = float(np.nanmean(vals))
        assert mean_val > 60.0, f"expected near-perfect separation, got {mean_val:.2f}"


class TestTieStats:
    def test_no_ties(self):
        scores = [(f"n{i}", float(i)) for i in range(1000)]
        stats = lu.tie_stats(scores)
        assert stats["distinct_values"] == 1000
        assert stats["largest_tie_block"] == 1
        assert stats["tied_top1pct_flag"] is False

    def test_all_tied(self):
        scores = [(f"n{i}", 1.0) for i in range(500)]
        stats = lu.tie_stats(scores)
        assert stats["distinct_values"] == 1
        assert stats["largest_tie_block"] == 500
        assert stats["top1pct_tie_fraction"] == pytest.approx(1.0)
        assert stats["tied_top1pct_flag"] is True

    def test_tie_block_spanning_top_1pct(self):
        # top1pct_n for n=1000 is 10; make the 10 best-scoring entries share
        # one value so the entire top-1% window is a single tie block.
        tied_top = [(f"top{i}", 0.0) for i in range(10)]
        rest = [(f"rest{i}", float(i + 1)) for i in range(990)]
        stats = lu.tie_stats(tied_top + rest)
        assert stats["top1pct_n"] == 10
        assert stats["top1pct_tie_fraction"] == pytest.approx(1.0)
        assert stats["tied_top1pct_flag"] is True

    def test_small_tie_block_within_top_1pct_does_not_flag(self):
        # 3 of the top 10 tied at the best value, but the *boundary* (10th)
        # value is unique -- set membership at the cutoff isn't ambiguous.
        tied_top = [(f"top{i}", 0.0) for i in range(3)]
        rest = [(f"rest{i}", float(i + 1)) for i in range(997)]
        stats = lu.tie_stats(tied_top + rest)
        assert stats["top1pct_n"] == 10
        assert stats["top1pct_tie_fraction"] == pytest.approx(1 / 10)
        assert stats["tied_top1pct_flag"] is False

    def test_empty_input(self):
        stats = lu.tie_stats([])
        assert stats["n"] == 0
        assert stats["tied_top1pct_flag"] is False

    def test_small_n_perfectly_separated_does_not_falsely_flag(self):
        """Regression test: n=80 (top1pct_n = ceil(80*0.01) = 1) with all
        distinct values used to always flag tied_top1pct_flag=True, purely
        because a lone element in a 1-wide window trivially "ties with
        itself." Caught via an end-to-end smoke test on synthetic
        well-separated data that should score near the theoretical max with
        no ties anywhere."""
        scores = [(f"n{i}", float(i)) for i in range(80)]  # all distinct
        stats = lu.tie_stats(scores)
        assert stats["top1pct_n"] == 1
        assert stats["distinct_values"] == 80
        assert stats["tied_top1pct_flag"] is False

    def test_small_n_genuine_tie_at_the_top_still_flags(self):
        # top1pct_n = 1; the single best-ranked value IS genuinely
        # duplicated elsewhere in the ranking -> should still flag.
        scores = [(f"n{i}", 0.0) for i in range(2)] + [(f"n{i}", float(i)) for i in range(2, 80)]
        stats = lu.tie_stats(scores)
        assert stats["top1pct_n"] == 1
        assert stats["tied_top1pct_flag"] is True


def _make_receptor_scores(n_receptors=4, n_lig=60, n_dec=60, separation=8.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for r in range(n_receptors):
        lig_names = [f"r{r}_lig{i}" for i in range(n_lig)]
        dec_names = [f"r{r}_ZINC{i:06d}" for i in range(n_dec)]
        lig_scores = [(n, float(rng.normal(0.0, 0.3))) for n in lig_names]
        dec_scores = [(n, float(rng.normal(separation, 0.3))) for n in dec_names]
        out[f"receptor{r}"] = (lig_scores, dec_scores, set(lig_names), set(dec_names))
    return out


class TestClusterBootstrap:
    def test_resamples_receptors_not_just_ligands(self):
        """A single-receptor-set input should show the cluster bootstrap
        occasionally dropping/duplicating receptors -- i.e. its variance
        should differ from (generally exceed) the naive per-receptor-draw
        approach on a small number of receptors."""
        receptor_scores = _make_receptor_scores(n_receptors=3, separation=8.0)
        rng = np.random.default_rng(1)
        vals = lu.cluster_bootstrap_avg_logauc(receptor_scores, n_boot=500, rng=rng)
        assert len(vals) == 500
        assert all(not np.isnan(v) for v in vals)
        # well-separated data -> still near-max on average
        assert np.mean(vals) > 50.0

    def test_empty_input_returns_empty(self):
        assert lu.cluster_bootstrap_avg_logauc({}, n_boot=100, rng=np.random.default_rng(0)) == []


class TestPairedDifferenceBootstrap:
    def test_identical_arms_have_zero_difference(self):
        receptor_scores = _make_receptor_scores(n_receptors=4, seed=2)
        rng = np.random.default_rng(3)
        diffs = lu.cluster_bootstrap_paired_difference(receptor_scores, receptor_scores, n_boot=300, rng=rng)
        assert len(diffs) == 300
        assert abs(np.mean(diffs)) < 1.0  # should be ~0, small bootstrap noise only

    def test_detects_a_real_difference(self):
        arm1 = _make_receptor_scores(n_receptors=4, separation=8.0, seed=4)
        arm2 = _make_receptor_scores(n_receptors=4, separation=0.05, seed=4)  # near-random arm2
        rng = np.random.default_rng(5)
        diffs = lu.cluster_bootstrap_paired_difference(arm1, arm2, n_boot=300, rng=rng)
        # arm1 (well separated) should score much higher than arm2 (near random)
        assert np.mean(diffs) > 30.0

    def test_no_shared_receptors_returns_empty(self):
        arm1 = {"onlyA": ([("a", 0.0)], [("ZINC1", 1.0)], {"a"}, {"ZINC1"})}
        arm2 = {"onlyB": ([("b", 0.0)], [("ZINC2", 1.0)], {"b"}, {"ZINC2"})}
        assert lu.cluster_bootstrap_paired_difference(arm1, arm2, 100, np.random.default_rng(0)) == []


class TestJackknifeAndBCa:
    def test_jackknife_leave_one_out_means(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        jk = lu.jackknife_leave_one_out_means(vals)
        # leave out 1.0 -> mean(2,3,4)=3; leave out 2.0 -> mean(1,3,4)=8/3; etc.
        assert jk == pytest.approx([3.0, 8 / 3, 7 / 3, 2.0])

    def test_jackknife_too_few_points(self):
        assert lu.jackknife_leave_one_out_means([1.0]) == []

    def test_bca_interval_contains_point_estimate_region(self):
        rng = np.random.default_rng(7)
        boot_vals = rng.normal(50.0, 5.0, size=5000)
        point_estimate = 50.0
        jackknife_vals = rng.normal(50.0, 1.0, size=10)
        lo, hi = lu.bca_interval(boot_vals, point_estimate, jackknife_vals)
        assert lo < point_estimate < hi
        # roughly symmetric-ish for near-normal data with small acceleration
        assert hi - lo > 10.0

    def test_bca_empty_bootstrap_returns_nan(self):
        lo, hi = lu.bca_interval([], 1.0, [1.0, 2.0])
        assert np.isnan(lo) and np.isnan(hi)


class TestPolarityWarning:
    def test_no_warning_near_zero(self):
        assert lu.polarity_warning("Method", 2.0) is None

    def test_no_warning_strongly_positive(self):
        assert lu.polarity_warning("Method", 55.0) is None

    def test_warning_strongly_negative(self):
        msg = lu.polarity_warning("Method", -40.0)
        assert msg is not None
        assert "Method" in msg

    def test_no_warning_for_nan(self):
        assert lu.polarity_warning("Method", float("nan")) is None
