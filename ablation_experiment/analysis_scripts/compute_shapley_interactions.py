#!/usr/bin/env python3
"""Task 2 — exact Shapley decomposition and pairwise Möbius interactions
over the 2^3 factorial {distogram, z_trunk, s_inputs}.

Why this matters under the new hypothesis
------------------------------------------
If z_trunk and s_inputs are two routes to the same identity shortcut, they
are redundant: ablating either one alone shows a *small* effect because the
other compensates, and read as independent single-channel ablations that
looks like "neither channel matters" -- the opposite of the truth, and
exactly what would appear to support the old (distogram-dominant)
hypothesis. Only the full factorial, decomposed into Shapley attribution
(how much each channel is worth on average across all the coalitions it
could join) plus pairwise Möbius interaction (whether two channels help
independently, redundantly, or synergistically) can detect that.

Data
----
Reads ``analysis_data/ablation_bootstrap_results/summary_per_receptor.csv``
(written by ``compute_ablation_bootstrap.py``), which has one row per
(experiment, receptor) with a ``point_estimate_logAUC`` column. The 8
factorial-cell experiment names (baseline, no_distogram, no_z_trunk,
no_s_inputs, distogram_only, z_trunk_only, s_inputs_only, bias_only) map to
the 8 characteristic-function values v(S) for S subset of
{distogram, z_trunk, s_inputs} -- see ``CELL_KEPT_CHANNELS`` below, which
must stay in sync with ``run_feature_ablation.py``'s
``_FACTORIAL_CELLS_KEPT_CHANNELS``. Resample/mean operator variants
(``<cell>__resample__d<seed>``, ``<cell>__mean``) are parsed and analyzed
per operator, since v(∅) under ``resample`` is a materially different
number than v(∅) under ``zero`` (see Task 1).

This depends on Task 0: v(∅) is ``bias_only``, and if it is artifact-
inflated by the tie bug, the efficiency check still passes (it's an
algebraic identity, not a sanity check on the data) but every Shapley value
and Möbius interaction computed from it is still wrong. Run the Task 0b
audit and confirm bias_only is sound before trusting this script's output.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logauc_utils import bca_interval, jackknife_leave_one_out_means  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV = BASE_DIR / "analysis_data" / "ablation_bootstrap_results" / "summary_per_receptor.csv"
OUT_DIR = BASE_DIR / "analysis_data"

CHANNELS = ("distogram", "z_trunk", "s_inputs")

# Must stay in sync with run_feature_ablation.py's _FACTORIAL_CELLS_KEPT_CHANNELS.
CELL_KEPT_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "baseline":       ("distogram", "z_trunk", "s_inputs"),
    "no_distogram":   ("z_trunk", "s_inputs"),
    "no_z_trunk":     ("distogram", "s_inputs"),
    "no_s_inputs":    ("distogram", "z_trunk"),
    "distogram_only": ("distogram",),
    "z_trunk_only":   ("z_trunk",),
    "s_inputs_only":  ("s_inputs",),
    "bias_only":      (),
}

EFFICIENCY_TOLERANCE = 1e-6

_RESAMPLE_RE = re.compile(r"^(?P<cell>.+)__resample__d(?P<seed>\d+)$")
_MEAN_RE = re.compile(r"^(?P<cell>.+)__mean$")


# ── naming parser ────────────────────────────────────────────────────────────

def parse_experiment_name(name: str) -> Optional[Tuple[str, str, Optional[int]]]:
    """Return (cell, operator, seed) if ``name`` is one of the 8 factorial
    cells (bare or with a resample/mean suffix), else None."""
    if name in CELL_KEPT_CHANNELS:
        return name, "zero", None
    m = _RESAMPLE_RE.match(name)
    if m and m.group("cell") in CELL_KEPT_CHANNELS:
        return m.group("cell"), "resample", int(m.group("seed"))
    m = _MEAN_RE.match(name)
    if m and m.group("cell") in CELL_KEPT_CHANNELS:
        return m.group("cell"), "mean", None
    return None


# ── Shapley / Möbius ─────────────────────────────────────────────────────────

def shapley_values(
    v: Dict[FrozenSet[str], float], players: Tuple[str, ...] = CHANNELS
) -> Dict[str, float]:
    """Exact Shapley values: φ_c = Σ_{S⊆C\\{c}} w(|S|)·[v(S∪{c}) − v(S)],
    w(|S|) = |S|!(n−|S|−1)!/n!. Requires ``v`` to contain all 2^n subsets."""
    n = len(players)
    phi = {}
    for c in players:
        others = [p for p in players if p != c]
        total = 0.0
        for r in range(len(others) + 1):
            for combo in itertools.combinations(others, r):
                S = frozenset(combo)
                weight = (math.factorial(len(S)) * math.factorial(n - len(S) - 1)) / math.factorial(n)
                total += weight * (v[S | {c}] - v[S])
        phi[c] = total
    return phi


def assert_efficiency(
    phi: Dict[str, float], v: Dict[FrozenSet[str], float], players: Tuple[str, ...] = CHANNELS,
    tol: float = EFFICIENCY_TOLERANCE, context: str = "",
) -> None:
    """Σ_c φ_c == v(C) − v(∅), the Shapley efficiency axiom. This is an
    algebraic identity of the formula itself -- true for *any* v -- so a
    failure here means the implementation has a bug (wrong subset
    enumeration, wrong weights, missing coalition), not that the data is
    unusual. Fails loudly by design."""
    lhs = sum(phi.values())
    rhs = v[frozenset(players)] - v[frozenset()]
    if abs(lhs - rhs) > tol:
        raise AssertionError(
            f"Shapley efficiency violated{' for ' + context if context else ''}: "
            f"sum(phi)={lhs!r} != v(C)-v(empty)={rhs!r} (tol={tol}). "
            f"This indicates a bug in shapley_values(), not unusual data."
        )


def mobius_interactions(
    v: Dict[FrozenSet[str], float], players: Tuple[str, ...] = CHANNELS
) -> Dict[Tuple[str, str], float]:
    """Pairwise Möbius interaction I(c,c') = v({c,c'}) − v({c}) − v({c'}) + v(∅).
    Sign convention: negative = redundant (channels overlap -- together they
    add less than the sum of their solo effects), positive = synergistic
    (together they add more than the sum)."""
    empty = v[frozenset()]
    out = {}
    for c1, c2 in itertools.combinations(players, 2):
        pair = frozenset({c1, c2})
        out[(c1, c2)] = v[pair] - v[frozenset({c1})] - v[frozenset({c2})] + empty
    return out


def cell_values_to_v(cell_values: Dict[str, float]) -> Dict[FrozenSet[str], float]:
    """Map {cell_name: value} (using the 8 factorial-cell names) to
    {frozenset(kept_channels): value}, the characteristic-function form
    shapley_values/mobius_interactions expect."""
    v = {}
    for cell, val in cell_values.items():
        kept = CELL_KEPT_CHANNELS.get(cell)
        if kept is None:
            continue
        v[frozenset(kept)] = val
    return v


# ── bootstrap aggregate over receptors ──────────────────────────────────────

def bootstrap_aggregate(per_receptor_values: List[float], n_boot: int, rng, alpha: float = 0.05):
    """Cluster-bootstrap the cross-receptor mean of an already-computed
    per-receptor statistic (a Shapley value or Möbius interaction), with a
    BCa CI. This resamples receptors, not raw ligand-level data -- the
    per-receptor Shapley/Möbius values are themselves derived from
    already-bootstrapped logAUC point estimates upstream."""
    vals = np.asarray([v for v in per_receptor_values if not np.isnan(v)], dtype=float)
    k = len(vals)
    if k == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "bca_low": float("nan"), "bca_high": float("nan"), "n_receptors": 0}
    point_estimate = float(vals.mean())
    boots = [float(vals[rng.integers(0, k, size=k)].mean()) for _ in range(n_boot)]
    jackknife_vals = jackknife_leave_one_out_means(vals.tolist())
    bca_lo, bca_hi = bca_interval(boots, point_estimate, jackknife_vals, alpha=alpha)
    return {
        "mean": point_estimate,
        "ci_low": float(np.percentile(boots, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "bca_low": bca_lo, "bca_high": bca_hi,
        "n_receptors": k,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=IN_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR,
                         help="Directory to write shapley_values.csv and mobius_interactions.csv into.")
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(
            f"{args.input} not found. Run compute_ablation_bootstrap.py first "
            f"(this script reads its summary_per_receptor.csv)."
        )

    df = pd.read_csv(args.input)
    df["parsed"] = df["experiment"].apply(parse_experiment_name)
    df = df[df["parsed"].notna()]
    if df.empty:
        raise SystemExit(
            f"No factorial-cell experiments found in {args.input}. Expected names like "
            f"{list(CELL_KEPT_CHANNELS)} (bare, __resample__dN, or __mean)."
        )
    df["cell"], df["operator"], df["donor_seed"] = zip(*df["parsed"])

    rng = np.random.default_rng(args.seed)
    phi_rows = []
    mobius_rows = []

    for (operator, donor_seed), group in df.groupby(["operator", "donor_seed"], dropna=False):
        # groupby on a column that mixes None (zero/mean rows) and int
        # (resample rows) upcasts to float64, turning None into NaN --
        # `nan is not None` is True, so every zero/mean label downstream
        # would otherwise render as "dnan" instead of the bare operator name.
        donor_seed = None if pd.isna(donor_seed) else int(donor_seed)
        receptors = sorted(group["receptor"].unique())
        per_receptor_phi: Dict[str, List[float]] = {c: [] for c in CHANNELS}
        per_receptor_mobius: Dict[Tuple[str, str], List[float]] = {
            pair: [] for pair in itertools.combinations(CHANNELS, 2)
        }
        n_receptors_ok = 0

        for receptor in receptors:
            rec_rows = group[group["receptor"] == receptor]
            cell_values = dict(zip(rec_rows["cell"], rec_rows["point_estimate_logAUC"]))
            missing = set(CELL_KEPT_CHANNELS) - set(cell_values)
            if missing:
                print(f"  [{operator}{'/d' + str(donor_seed) if donor_seed is not None else ''}] "
                      f"{receptor}: missing cells {sorted(missing)}, skipping (fail loudly, not silently)")
                continue

            v = cell_values_to_v(cell_values)
            phi = shapley_values(v)
            assert_efficiency(phi, v, context=f"operator={operator} donor_seed={donor_seed} receptor={receptor}")
            mobius = mobius_interactions(v)

            for c in CHANNELS:
                per_receptor_phi[c].append(phi[c])
            for pair, val in mobius.items():
                per_receptor_mobius[pair].append(val)
            n_receptors_ok += 1

            for c in CHANNELS:
                phi_rows.append({
                    "operator": operator, "donor_seed": donor_seed, "receptor": receptor,
                    "channel": c, "phi": phi[c], "is_aggregate": False,
                })
            for pair, val in mobius.items():
                mobius_rows.append({
                    "operator": operator, "donor_seed": donor_seed, "receptor": receptor,
                    "channel_pair": f"{pair[0]}+{pair[1]}", "interaction": val,
                    "sign": "redundant" if val < 0 else ("synergistic" if val > 0 else "none"),
                    "is_aggregate": False,
                })

        label = f"{operator}{'/d' + str(donor_seed) if donor_seed is not None else ''}"
        print(f"[{label}] {n_receptors_ok}/{len(receptors)} receptors had all 8 cells; "
              f"efficiency check passed for all.")

        if n_receptors_ok == 0:
            # Emitting all-NaN aggregates here would look like a completed
            # analysis while carrying no information -- and downstream
            # (NAE, reporting) would happily consume the NaNs. Skip the
            # operator entirely and say so.
            print(f"  [{label}] SKIPPED: no receptor had all 8 factorial cells, so no Shapley "
                  f"decomposition is possible for this operator. Check that the 8 cell "
                  f"experiments {sorted(CELL_KEPT_CHANNELS)} were all run.")
            continue

        for c in CHANNELS:
            agg = bootstrap_aggregate(per_receptor_phi[c], args.n_bootstraps, rng)
            phi_rows.append({
                "operator": operator, "donor_seed": donor_seed, "receptor": "ALL",
                "channel": c, "phi": agg["mean"], "is_aggregate": True,
                "ci_low": agg["ci_low"], "ci_high": agg["ci_high"],
                "bca_low": agg["bca_low"], "bca_high": agg["bca_high"],
                "n_receptors": agg["n_receptors"],
            })
        for pair, vals in per_receptor_mobius.items():
            agg = bootstrap_aggregate(vals, args.n_bootstraps, rng)
            sign = "redundant" if agg["mean"] < 0 else ("synergistic" if agg["mean"] > 0 else "none")
            mobius_rows.append({
                "operator": operator, "donor_seed": donor_seed, "receptor": "ALL",
                "channel_pair": f"{pair[0]}+{pair[1]}", "interaction": agg["mean"], "sign": sign,
                "is_aggregate": True,
                "ci_low": agg["ci_low"], "ci_high": agg["ci_high"],
                "bca_low": agg["bca_low"], "bca_high": agg["bca_high"],
                "n_receptors": agg["n_receptors"],
            })

    if not phi_rows:
        raise SystemExit(
            f"No operator had a single receptor with all 8 factorial cells present, so no "
            f"Shapley decomposition could be computed from {args.input}. The 8 cells are "
            f"{sorted(CELL_KEPT_CHANNELS)} -- confirm run_feature_ablation.py was run with "
            f"all of them (they are all in its default experiment list)."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phi_df = pd.DataFrame(phi_rows)
    mobius_df = pd.DataFrame(mobius_rows)
    phi_df.to_csv(args.output_dir / "shapley_values.csv", index=False)
    mobius_df.to_csv(args.output_dir / "mobius_interactions.csv", index=False)

    print(f"\nWrote {len(phi_df)} Shapley rows to shapley_values.csv")
    print(f"Wrote {len(mobius_df)} Möbius interaction rows to mobius_interactions.csv")

    agg_phi = phi_df[phi_df["is_aggregate"]]
    if not agg_phi.empty:
        print("\nAggregate Shapley shares (zero operator, if present):")
        zero_agg = agg_phi[agg_phi["operator"] == "zero"]
        for _, r in zero_agg.iterrows():
            print(f"  {r['channel']:12s} φ = {r['phi']:.2f}  BCa 95% CI [{r['bca_low']:.2f}, {r['bca_high']:.2f}]")

    agg_mob = mobius_df[mobius_df["is_aggregate"]]
    if not agg_mob.empty:
        print("\nAggregate pairwise interactions (zero operator, if present):")
        zero_agg = agg_mob[agg_mob["operator"] == "zero"]
        for _, r in zero_agg.iterrows():
            print(f"  {r['channel_pair']:25s} I = {r['interaction']:+.2f} ({r['sign']})  "
                  f"BCa 95% CI [{r['bca_low']:.2f}, {r['bca_high']:.2f}]")


if __name__ == "__main__":
    main()
