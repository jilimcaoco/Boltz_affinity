#!/usr/bin/env python3
"""Shared logAUC / ROC utilities for the ablation and method-comparison
bootstrap scripts.

Both ``compute_ablation_bootstrap.py`` and ``compute_logauc_bootstrap.py``
used to carry independent copies of this code, including a tie-handling bug:
the bootstrap resample concatenated actives before decoys and then called
Python's stable ``list.sort``, so every group of tied scores was silently
resolved in favor of actives. On data with no signal at all this pushed
logAUC from ~0 up to the theoretical maximum (85.54) as ties got denser —
see ``ablation_experiment/tests/test_logauc_utils.py`` for the regression
test. Ties are now broken by an explicit RNG shuffle before the sort, which
is unbiased in expectation and averages out across bootstrap replicates.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

LOGAUC_MAX = 1.0
LOGAUC_MIN = 0.001
RANDOM_LOGAUC = (LOGAUC_MAX - LOGAUC_MIN) / np.log(10) / np.log10(LOGAUC_MAX / LOGAUC_MIN)

# A method's mean logAUC across receptors this far below zero is far more
# consistent with an inverted sort polarity than with a genuinely bad
# classifier, and is worth a loud warning rather than a silent bad number.
POLARITY_SUSPECT_THRESHOLD = -15.0

Score = Tuple[str, float]


# ── ROC / logAUC core ───────────────────────────────────────────────────────

def do_roc(scores: Sequence[Score], lig_set, decoy_set, nbins: int = 10000):
    """Compute ROC curve points from a ranked list of (name, score) tuples.
    ``scores`` must already be sorted ascending by score (best first)."""
    num_data = len(scores)
    binsize = max(int(num_data / nbins), 1)
    num_lig = len(lig_set)
    num_dec = len(decoy_set)
    if num_lig == 0 or num_dec == 0:
        return None

    found_ligand = 0
    results = []
    for i in range(num_data):
        if i % binsize == 0:
            results.append([i - found_ligand, found_ligand])
        if scores[i][0] in lig_set:
            found_ligand += 1
    results.append([num_data - found_ligand, found_ligand])
    results.append([num_dec, num_lig])

    points = []
    for x in results:
        fpr = x[0] * 100.0 / num_dec
        tpr = x[1] * 100.0 / num_lig
        points.append([fpr, tpr])
    return points


def interpolate_curve(points):
    i = 0
    while i < len(points) and points[i][0] < 0.1:
        i += 1
    if i == 0:
        return points
    slope = (points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0] + 1e-12)
    intercept = points[i][1] - slope * points[i][0]
    point_one = [0.100001, slope * 0.100001 + intercept]
    npoints = list(points)
    npoints.insert(i, point_one)
    return npoints


def logAUC(points) -> float:
    npoints = []
    for x in points:
        if (x[0] >= LOGAUC_MIN * 100) and (x[0] <= LOGAUC_MAX * 100):
            npoints.append([x[0] / 100, x[1] / 100])

    area = 0.0
    for point2, point1 in zip(npoints[1:], npoints[:-1]):
        if point2[0] - point1[0] < 0.000001:
            continue
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        intercept = point2[1] - (dy) / (dx) * point2[0]
        area += dy / np.log(10) + intercept * (np.log10(point2[0]) - np.log10(point1[0]))

    return area / np.log10(LOGAUC_MAX / LOGAUC_MIN) - RANDOM_LOGAUC


def compute_logauc(ranked_list: Sequence[Score], lig_set, decoy_set) -> float:
    """Given a ranked (ascending, best-first) list of (name, score), compute
    adjusted logAUC * 100 (random = 0, per ``RANDOM_LOGAUC`` subtraction above)."""
    points = do_roc(ranked_list, lig_set, decoy_set)
    if points is None:
        return np.nan
    points = interpolate_curve(points)
    return logAUC(points) * 100


# ── tie handling ─────────────────────────────────────────────────────────────

def break_ties(scores: Sequence[Score], rng: np.random.Generator) -> List[Score]:
    """Shuffle a list of (name, score) tuples so a subsequent stable sort
    resolves tied scores uniformly at random instead of systematically
    favoring whichever group (actives vs. decoys) was concatenated first."""
    shuffled = list(scores)
    rng.shuffle(shuffled)
    return shuffled


def tie_stats(scores: Sequence[Score]) -> Dict[str, float]:
    """Tie-density diagnostics for a set of (name, score) tuples.

    Returns distinct-value count, largest tie block, and the fraction of the
    top 1% of the ranking (best/lowest scores, per this codebase's
    lower-is-better convention) that falls inside a single tied score value.
    A tied top-1% is a hard failure signal: it means logAUC there is
    determined by tie-break order, not by the model.
    """
    n = len(scores)
    if n == 0:
        return {
            "n": 0,
            "distinct_values": 0,
            "largest_tie_block": 0,
            "top1pct_n": 0,
            "top1pct_tie_fraction": float("nan"),
            "tied_top1pct_flag": False,
        }

    vals = np.array([s[1] for s in scores], dtype=float)
    uniq, counts = np.unique(vals, return_counts=True)
    distinct = int(len(uniq))
    largest_tie = int(counts.max())

    top_n = max(1, int(np.ceil(n * 0.01)))
    sorted_vals = np.sort(vals)  # ascending = best-first, matching do_roc's convention
    top_slice = sorted_vals[:top_n]
    boundary_val = sorted_vals[top_n - 1]  # worst-ranked score still inside the top-1% window

    # What matters for "is the top-1% cutoff arbitrary" is specifically
    # whether the *boundary* value is tied -- a tie fully contained inside
    # the window (e.g. ranks 1-3 of a top-10 window) doesn't make set
    # membership ambiguous, only internal ordering; only a tie straddling
    # the cutoff (the worst-ranked member of the window sharing its value
    # with something else) makes it genuinely arbitrary whether a compound
    # counts as "top 1%" or not.
    top1pct_tie_fraction = float(np.sum(top_slice == boundary_val)) / top_n
    # Global duplication check, not just within the window: with top_n==1
    # (n<100, ceil(n*0.01) rounds down to 1) a lone element's "fraction of a
    # 1-wide window" is trivially 1.0 whether or not it's ever duplicated,
    # so a window-only check falsely flags perfectly-separated small-n data.
    # Counting matches across the *full* ranking fixes both directions: it
    # stops the false positive when the value is unique, and still catches
    # a real duplicate elsewhere in the ranking even when top_n==1.
    boundary_tie_block_size = int(np.sum(vals == boundary_val))
    tied_top1pct_flag = bool(boundary_tie_block_size >= 2 and top1pct_tie_fraction >= 0.99)

    return {
        "n": n,
        "distinct_values": distinct,
        "largest_tie_block": largest_tie,
        "top1pct_n": top_n,
        "top1pct_tie_fraction": top1pct_tie_fraction,
        "tied_top1pct_flag": tied_top1pct_flag,
    }


# ── bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_logauc(lig_scores, decoy_scores, lig_set, decoy_set, n_boot, rng):
    """Bootstrap logAUC by resampling ligands and decoys separately.
    Ties within each resample are broken by random shuffle (see module
    docstring) before the stable sort, removing the active-favoring bias."""
    results = []
    for _ in range(n_boot):
        boot_lig_idx = rng.integers(0, len(lig_scores), size=len(lig_scores))
        boot_dec_idx = rng.integers(0, len(decoy_scores), size=len(decoy_scores))
        boot_scores = [lig_scores[i] for i in boot_lig_idx] + [decoy_scores[i] for i in boot_dec_idx]
        boot_scores = break_ties(boot_scores, rng)
        boot_scores.sort(key=lambda x: x[1])
        val = compute_logauc(boot_scores, lig_set, decoy_set)
        results.append(val)
    return results


def compute_avg_logauc_bootstrap(per_receptor_boots, n_boot, rng):
    """DEPRECATED (Task 3a): sample one logAUC from each receptor's
    per-receptor bootstrap distribution, then average, with every receptor
    always appearing exactly once. This is *not* a cluster bootstrap -- it
    never resamples which receptors are present, so it understates
    between-receptor variance and supports only "these particular
    receptors" claims, not "receptors in general." Kept only for backward
    compatibility with old callers; use ``cluster_bootstrap_avg_logauc``
    for any new population-level CI (see its docstring)."""
    receptors = list(per_receptor_boots.keys())
    if not receptors:
        return []
    avg_boots = []
    for _ in range(n_boot):
        sampled = []
        for rec in receptors:
            vals = per_receptor_boots[rec]
            idx = rng.integers(0, len(vals))
            sampled.append(vals[idx])
        avg_boots.append(np.nanmean(sampled))
    return avg_boots


# ── Task 3: proper two-stage cluster bootstrap, BCa, paired difference ─────

MIN_BOOTSTRAPS = 2000
DEFAULT_BOOTSTRAPS = 10000

ReceptorScores = Dict[str, Tuple[List[Score], List[Score], set, set]]
"""{receptor: (lig_scores, decoy_scores, lig_set, decoy_set)}"""


def cluster_bootstrap_avg_logauc(receptor_scores: ReceptorScores, n_boot: int, rng) -> List[float]:
    """Two-stage cluster bootstrap for the population-level average logAUC:
    resample receptors *with replacement* first, then resample
    ligands/decoys within each resampled receptor (each occurrence gets an
    independent within-receptor draw, even if the same receptor is drawn
    more than once). This is what actually supports a claim about
    "receptors in general" rather than just the receptors on hand -- the
    old ``compute_avg_logauc_bootstrap`` never resampled which receptors
    were included, so it could not capture between-receptor variance.
    """
    receptors = list(receptor_scores.keys())
    k = len(receptors)
    if k == 0:
        return []
    results = []
    for _ in range(n_boot):
        chosen = rng.integers(0, k, size=k)
        vals = []
        for idx in chosen:
            rec = receptors[idx]
            lig_scores, dec_scores, lig_set, dec_set = receptor_scores[rec]
            if not lig_scores or not dec_scores:
                continue
            boot_lig_idx = rng.integers(0, len(lig_scores), size=len(lig_scores))
            boot_dec_idx = rng.integers(0, len(dec_scores), size=len(dec_scores))
            boot = [lig_scores[i] for i in boot_lig_idx] + [dec_scores[i] for i in boot_dec_idx]
            boot = break_ties(boot, rng)
            boot.sort(key=lambda x: x[1])
            vals.append(compute_logauc(boot, lig_set, dec_set))
        results.append(float(np.nanmean(vals)) if vals else float("nan"))
    return results


def cluster_bootstrap_paired_difference(
    receptor_scores_arm1: ReceptorScores,
    receptor_scores_arm2: ReceptorScores,
    n_boot: int,
    rng,
) -> List[float]:
    """Bootstrap Δ = v(arm1) − v(arm2), resampling the *same* receptors and
    the *same* within-receptor ligand/decoy draw for both arms on each
    replicate, so receptor difficulty (the dominant variance component)
    cancels between arms instead of adding noise to the comparison.
    Overlapping CIs on two independently-bootstrapped arms do not imply a
    non-significant difference -- this is the correct way to test arm-vs-arm
    claims (Task 3c). Requires both arms to share the same receptor keys and,
    within each receptor, the same ligand/decoy list *order* (same compound
    at the same index) so a shared draw pairs the same compound in both arms.
    """
    receptors = sorted(set(receptor_scores_arm1) & set(receptor_scores_arm2))
    k = len(receptors)
    if k == 0:
        return []
    diffs = []
    for _ in range(n_boot):
        chosen = rng.integers(0, k, size=k)
        vals1, vals2 = [], []
        for idx in chosen:
            rec = receptors[idx]
            lig1, dec1, ligset1, decset1 = receptor_scores_arm1[rec]
            lig2, dec2, ligset2, decset2 = receptor_scores_arm2[rec]
            if not lig1 or not dec1 or not lig2 or not dec2:
                continue
            n_lig, n_dec = len(lig1), len(dec1)
            boot_lig_idx = rng.integers(0, n_lig, size=n_lig)
            boot_dec_idx = rng.integers(0, n_dec, size=n_dec)

            b1 = [lig1[i] for i in boot_lig_idx] + [dec1[i] for i in boot_dec_idx]
            b1 = break_ties(b1, rng)
            b1.sort(key=lambda x: x[1])
            vals1.append(compute_logauc(b1, ligset1, decset1))

            b2 = [lig2[i] for i in boot_lig_idx] + [dec2[i] for i in boot_dec_idx]
            b2 = break_ties(b2, rng)
            b2.sort(key=lambda x: x[1])
            vals2.append(compute_logauc(b2, ligset2, decset2))
        if vals1 and vals2:
            diffs.append(float(np.nanmean(vals1) - np.nanmean(vals2)))
        else:
            diffs.append(float("nan"))
    return diffs


def jackknife_leave_one_out_means(point_estimates: Sequence[float]) -> List[float]:
    """Leave-one-out jackknife means over a set of per-cluster (receptor)
    point estimates, used as the acceleration input to ``bca_interval``."""
    vals = np.asarray([v for v in point_estimates if not np.isnan(v)], dtype=float)
    n = len(vals)
    if n < 2:
        return []
    total = vals.sum()
    return [float((total - v) / (n - 1)) for v in vals]


def bca_interval(
    boot_vals: Sequence[float],
    point_estimate: float,
    jackknife_vals: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bias-corrected and accelerated (BCa) confidence interval.

    A plain 2.5/97.5 percentile interval from only a few thousand replicates
    is determined by a handful of order statistics and can be visibly biased
    when the bootstrap distribution is skewed (as logAUC often is near the
    theoretical max/min). BCa corrects for both median bias (``z0``) and
    skewness (acceleration ``a``, estimated here via a leave-one-receptor-out
    jackknife of the point estimate).
    """
    from scipy.stats import norm

    boot_arr = np.asarray([v for v in boot_vals if not np.isnan(v)], dtype=float)
    if len(boot_arr) == 0 or np.isnan(point_estimate):
        return (float("nan"), float("nan"))

    prop_less = np.mean(boot_arr < point_estimate)
    prop_less = min(max(prop_less, 1e-6), 1 - 1e-6)
    z0 = norm.ppf(prop_less)

    jk = np.asarray(jackknife_vals, dtype=float)
    jk = jk[~np.isnan(jk)]
    if len(jk) < 2:
        a = 0.0
    else:
        jk_mean = jk.mean()
        num = np.sum((jk_mean - jk) ** 3)
        den = 6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5)
        a = float(num / den) if den != 0 else 0.0

    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)

    def _adj(z_a):
        denom = 1 - a * (z0 + z_a)
        return z0 + z_a if denom == 0 else z0 + (z0 + z_a) / denom

    alpha1 = float(np.clip(norm.cdf(_adj(z_lo)), 0.0, 1.0))
    alpha2 = float(np.clip(norm.cdf(_adj(z_hi)), 0.0, 1.0))

    lo = float(np.percentile(boot_arr, 100 * alpha1))
    hi = float(np.percentile(boot_arr, 100 * alpha2))
    return (lo, hi)


def point_estimate_logauc(lig_scores, decoy_scores, lig_set, decoy_set, rng, n_shuffles: int = 50) -> float:
    """A single, comparatively low-variance point estimate of logAUC on the
    *observed* (non-resampled) data, averaging over ``n_shuffles`` independent
    tie-breaks rather than relying on one arbitrary sort order. Use this
    instead of a single ``compute_logauc`` call whenever ties are present and
    a non-bootstrapped point value is needed (e.g. for reporting alongside
    bootstrap CIs, or in the tie-density audit)."""
    combined = list(lig_scores) + list(decoy_scores)
    vals = []
    for _ in range(n_shuffles):
        shuffled = break_ties(combined, rng)
        shuffled.sort(key=lambda x: x[1])
        vals.append(compute_logauc(shuffled, lig_set, decoy_set))
    return float(np.nanmean(vals))


def polarity_warning(method_name: str, mean_logauc: float) -> str | None:
    """Return a warning string if ``mean_logauc`` is suspiciously far below
    zero for ``method_name`` (see ``POLARITY_SUSPECT_THRESHOLD``), else None."""
    if mean_logauc is not None and not np.isnan(mean_logauc) and mean_logauc < POLARITY_SUSPECT_THRESHOLD:
        return (
            f"[polarity-check] {method_name}: mean logAUC = {mean_logauc:.2f}, "
            f"far below 0. This is much more consistent with an inverted "
            f"score-polarity (higher/lower-is-better mixed up) than with a "
            f"genuinely anti-correlated classifier. Verify the sort direction "
            f"for {method_name} before trusting this number."
        )
    return None
