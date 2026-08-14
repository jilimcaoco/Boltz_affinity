#!/usr/bin/env python3
"""Plots for the refactored ablation outputs (deliverable #9).

The pre-existing plot_ridgeline.py / plot_scatter.py cover the *method
comparison* panel (DiffDock / DOCK3.8 / OG_Affinity / Boltz rescoring) and
know nothing about the ablation columns added by this refactor. This script
covers the new ones:

  1. shapley_attribution.png     phi per channel with BCa CIs        (Task 2)
  2. mobius_interactions.png     pairwise redundancy/synergy         (Task 2)
  3. normalized_ablation_effect.png  NAE per channel x receptor class (Task 6)
  4. operator_contrast.png       zero vs resample vs mean            (Task 1)
  5. structure_attributable.png  raw logAUC vs 2D-corrected excess   (Task 5)
  6. tie_density_diagnostic.png  distinct values / flagged conditions (Task 0b)

Every panel is optional: each reads its own input and is skipped with a
message if that input is absent (e.g. no resample runs yet -> no operator
contrast; no pose files -> no structure-attributable panel). Missing inputs
are a normal intermediate state in this pipeline, not an error.

Colour: the three information channels are a *categorical* encoding -- the
colour identifies the channel, never its rank, and the assignment is fixed
so a channel keeps its colour across every figure here. Slots 1-3 of the
reference palette (blue/orange/aqua) are used in fixed order; that triple is
documented as passing the all-pairs CVD and normal-vision separation floors
in both light and dark modes. Aqua sits under 3:1 contrast on a light
surface, so every bar also carries a direct value label rather than relying
on fill colour alone.

Usage:
    python plot_ablation_summary.py
    python plot_ablation_summary.py --analysis-dir /path/to/analysis_data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS_DIR = BASE_DIR / "analysis_data"

# Categorical slots 1-3, fixed order, assigned by entity (channel) not rank.
CHANNEL_COLORS = {
    "distogram": "#2a78d6",   # slot 1, blue
    "z_trunk":   "#eb6834",   # slot 2, orange
    "s_inputs":  "#1baf7a",   # slot 3, aqua
}
CHANNEL_ORDER = ["distogram", "z_trunk", "s_inputs"]

# Diverging pair for the signed Mobius interaction (redundant vs synergistic)
# and a neutral for "no interaction" -- polarity, so two hues + neutral, never
# a categorical ramp.
COLOR_REDUNDANT = "#2a78d6"
COLOR_SYNERGISTIC = "#eb6834"
COLOR_NEUTRAL = "#8a8a86"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dcdcd8"

ADJUSTED_NOTE = "adjusted logAUC, random = 0"


def _style_axes(ax, xlabel: str = "", ylabel: str = "", title: str = "", subtitle: str = ""):
    """Recessive grid and axes; the data carries the emphasis.

    ``subtitle`` is rendered inside the title block rather than as separate
    axes-coordinate text -- placing it at y=1.02 independently collided with
    the title at most figure heights.
    """
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=TEXT_SECONDARY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=TEXT_SECONDARY)
    if title:
        ax.set_title(title, fontsize=12.5, color=TEXT_PRIMARY, loc="left",
                      pad=22 if subtitle else 10)
    if subtitle:
        # Sits in the gap opened by the title pad above, in axes coords.
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=9, color=TEXT_SECONDARY, va="bottom", ha="left")


def _label_beyond(ax, value, err_hi, err_lo, pos, span, horizontal=True):
    """Place a value label clear of its error-bar whisker rather than at the
    bar end, so the two never overlap."""
    pad = span * 0.03
    if value >= 0:
        at = value + (err_hi if not np.isnan(err_hi) else 0) + pad
        ha = "left"
    else:
        at = value - (err_lo if not np.isnan(err_lo) else 0) - pad
        ha = "right"
    if horizontal:
        ax.text(at, pos, f"{value:.1f}", va="center", ha=ha,
                fontsize=9.5, color=TEXT_PRIMARY)
    else:
        ax.text(pos, at, f"{value:.1f}", ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9.5, color=TEXT_PRIMARY)


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def _read(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  skip: {path.name} not found")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"  skip: {path.name} is empty")
        return None
    return df


# ── 1. Shapley attribution ──────────────────────────────────────────────────

def plot_shapley(analysis_dir: Path, out_dir: Path):
    df = _read(analysis_dir / "shapley_values.csv")
    if df is None:
        return False
    agg = df[(df["is_aggregate"]) & (df["operator"] == "zero")]
    if agg.empty:
        print("  skip: no aggregate zero-operator Shapley rows")
        return False

    agg = agg.set_index("channel").reindex([c for c in CHANNEL_ORDER if c in set(agg["channel"])])
    y = np.arange(len(agg))
    vals = agg["phi"].to_numpy()
    lo = vals - agg["bca_low"].to_numpy()
    hi = agg["bca_high"].to_numpy() - vals

    fig, ax = plt.subplots(figsize=(7.5, 0.62 * len(agg) + 2.0))
    ax.barh(y, vals, height=0.34, color=[CHANNEL_COLORS[c] for c in agg.index])
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=TEXT_SECONDARY,
                elinewidth=1.4, capsize=3.5)
    ax.axvline(0, color=TEXT_SECONDARY, lw=1)

    span = float(np.nanmax(np.abs(vals))) or 1.0
    for yi, v, e_hi, e_lo in zip(y, vals, hi, lo):
        _label_beyond(ax, v, e_hi, e_lo, yi, span)

    ax.set_yticks(y)
    ax.set_yticklabels(agg.index, fontsize=10, color=TEXT_PRIMARY)
    ax.invert_yaxis()
    ax.margins(x=0.14)
    _style_axes(ax, xlabel=f"Shapley value φ  —  {ADJUSTED_NOTE}",
                title="Channel attribution — exact Shapley decomposition",
                subtitle="zero operator; error bars are cluster-bootstrap BCa 95% CIs")
    _save(fig, out_dir / "shapley_attribution.png")
    return True


# ── 2. Mobius interactions ──────────────────────────────────────────────────

def plot_mobius(analysis_dir: Path, out_dir: Path):
    df = _read(analysis_dir / "mobius_interactions.csv")
    if df is None:
        return False
    agg = df[(df["is_aggregate"]) & (df["operator"] == "zero")]
    if agg.empty:
        print("  skip: no aggregate zero-operator Mobius rows")
        return False

    y = np.arange(len(agg))
    vals = agg["interaction"].to_numpy()
    lo = vals - agg["bca_low"].to_numpy()
    hi = agg["bca_high"].to_numpy() - vals
    colors = [COLOR_REDUNDANT if v < 0 else (COLOR_SYNERGISTIC if v > 0 else COLOR_NEUTRAL)
              for v in vals]

    fig, ax = plt.subplots(figsize=(8.0, 0.62 * len(agg) + 2.2))
    ax.barh(y, vals, height=0.34, color=colors)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=TEXT_SECONDARY,
                elinewidth=1.4, capsize=3.5)
    ax.axvline(0, color=TEXT_SECONDARY, lw=1)

    span = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
    for yi, v, e_hi, e_lo in zip(y, vals, hi, lo):
        _label_beyond(ax, v, e_hi, e_lo, yi, span or 1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(agg["channel_pair"], fontsize=10, color=TEXT_PRIMARY)
    ax.invert_yaxis()
    ax.margins(x=0.16)
    _style_axes(ax, xlabel=f"Möbius interaction I  —  {ADJUSTED_NOTE}",
                title="Pairwise channel interaction",
                subtitle="negative = redundant (overlapping information)   ·   positive = synergistic")
    _save(fig, out_dir / "mobius_interactions.png")
    return True


# ── 3. Normalized ablation effect ───────────────────────────────────────────

def plot_nae(analysis_dir: Path, out_dir: Path):
    df = _read(analysis_dir / "normalized_ablation_effect.csv")
    if df is None:
        return False
    class_path = analysis_dir / "receptor_class_map.csv"
    if class_path.exists():
        df = df.merge(pd.read_csv(class_path), on="receptor", how="left")
    else:
        df["receptor_class"] = "all"
    df["receptor_class"] = df["receptor_class"].fillna("other")
    df = df[df["NAE"].notna()]
    if df.empty:
        print("  skip: every NAE row is NaN (near-zero usable-signal denominator?)")
        return False

    pivot = df.pivot_table(index="receptor_class", columns="channel", values="NAE", aggfunc="mean")
    channels = [c for c in CHANNEL_ORDER if c in pivot.columns]
    classes = list(pivot.index)

    x = np.arange(len(classes))
    width = 0.8 / max(len(channels), 1)

    fig, ax = plt.subplots(figsize=(1.9 * len(classes) + 3.5, 4.6))
    for i, ch in enumerate(channels):
        offs = (i - (len(channels) - 1) / 2) * width
        vals = pivot[ch].to_numpy()
        ax.bar(x + offs, vals, width=width * 0.92, label=ch, color=CHANNEL_COLORS[ch])
        for xi, v in zip(x + offs, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8, color=TEXT_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10, color=TEXT_PRIMARY)
    ax.axhline(0, color=TEXT_SECONDARY, lw=1)
    _style_axes(ax, ylabel="NAE  (fraction of usable signal destroyed)",
                title="Normalized ablation effect by receptor class",
                subtitle="NAE = [v(full) − v(ablate c)] / [v(full) − v(bias_only)]")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.16)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncols=len(channels))
    _save(fig, out_dir / "normalized_ablation_effect.png")
    return True


# ── 4. Operator contrast (Task 1 headline) ─────────────────────────────────

def plot_operator_contrast(analysis_dir: Path, out_dir: Path):
    summary = _read(analysis_dir / "ablation_bootstrap_results" / "summary_per_receptor.csv")
    if summary is None:
        return

    import re
    def parse(name):
        if re.match(r"^.+__resample__d\d+$", str(name)):
            return re.sub(r"__resample__d\d+$", "", name), "resample"
        if str(name).endswith("__mean"):
            return re.sub(r"__mean$", "", name), "mean"
        return name, "zero"

    parsed = summary["experiment"].apply(parse)
    summary = summary.assign(cell=[p[0] for p in parsed], operator=[p[1] for p in parsed])

    cells = ["no_distogram", "no_z_trunk", "no_s_inputs"]
    summary = summary[summary["cell"].isin(cells)]
    if summary.empty or summary["operator"].nunique() < 2:
        print("  skip: no resample/mean runs present, nothing to contrast against zero")
        return False

    pivot = summary.pivot_table(index="cell", columns="operator",
                                values="point_estimate_logAUC", aggfunc="mean")
    operators = [o for o in ("zero", "resample", "mean") if o in pivot.columns]
    cells_present = [c for c in cells if c in pivot.index]
    pivot = pivot.reindex(cells_present)

    # Operators are levels of one factor, not entities -- encode them by
    # lightness of a single hue rather than by three unrelated categorical
    # hues, so the channel colours stay free to mean "channel".
    op_shades = {"zero": "#0b3f7d", "resample": "#2a78d6", "mean": "#8fbdee"}

    x = np.arange(len(cells_present))
    width = 0.8 / max(len(operators), 1)

    fig, ax = plt.subplots(figsize=(2.2 * len(cells_present) + 3.5, 4.8))
    for i, op in enumerate(operators):
        offs = (i - (len(operators) - 1) / 2) * width
        vals = pivot[op].to_numpy()
        ax.bar(x + offs, vals, width=width * 0.92, label=op, color=op_shades.get(op, COLOR_NEUTRAL))
        for xi, v in zip(x + offs, vals):
            if not np.isnan(v):
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=8, color=TEXT_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(cells_present, fontsize=10, color=TEXT_PRIMARY)
    ax.axhline(0, color=TEXT_SECONDARY, lw=1)
    _style_axes(ax, ylabel=ADJUSTED_NOTE,
                title="Ablation operator contrast — zero vs. donor substitution",
                subtitle="a large zero−resample gap means the zero effect reflects "
                         "off-distribution brittleness, not lost signal")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.16)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncols=len(operators))
    _save(fig, out_dir / "operator_contrast.png")
    return True


# ── 5. Structure-attributable excess ────────────────────────────────────────

def plot_structure_attributable(analysis_dir: Path, out_dir: Path):
    df = _read(analysis_dir / "ablation_with_property_control.csv")
    if df is None:
        return False
    if "structure_attributable_excess" not in df.columns:
        print("  skip: no structure_attributable_excess column")
        return False

    agg = (df.groupby("experiment")[["point_estimate_logAUC", "structure_attributable_excess"]]
             .mean().dropna())
    if agg.empty:
        print("  skip: no rows with both raw logAUC and 2D ceiling")
        return False
    agg = agg.sort_values("point_estimate_logAUC")

    y = np.arange(len(agg))
    raw = agg["point_estimate_logAUC"].to_numpy()
    excess = agg["structure_attributable_excess"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(agg) + 2.4))
    # Dumbbell: the gap between raw and 2D-corrected IS the quantity of
    # interest, so encode it as a connecting segment rather than two bars.
    for yi, r, e in zip(y, raw, excess):
        ax.plot([e, r], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(raw, y, s=52, color="#2a78d6", label="raw logAUC", zorder=2)
    ax.scatter(excess, y, s=52, color="#eb6834", label="structure-attributable (− 2D ceiling)", zorder=2)
    ax.axvline(0, color=TEXT_SECONDARY, lw=1)

    ax.set_yticks(y)
    ax.set_yticklabels(agg.index, fontsize=9, color=TEXT_PRIMARY)
    _style_axes(ax, xlabel=ADJUSTED_NOTE,
                title="Structure-attributable excess over the 2D ligand-only ceiling")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    _save(fig, out_dir / "structure_attributable.png")
    return True


# ── 6. Tie-density diagnostic ───────────────────────────────────────────────

def plot_tie_density(analysis_dir: Path, out_dir: Path):
    df = _read(analysis_dir / "tie_density_audit.csv")
    if df is None:
        return False
    df = df[df["source"] == "ablation"]
    if df.empty:
        print("  skip: no ablation rows in the tie audit")
        return False

    agg = df.groupby("condition").agg(
        distinct=("distinct_values", "min"),
        flagged=("tied_top1pct_flag", "any"),
        status=("status", lambda s: "unusable" if "unusable" in set(s)
                else ("suspect" if "suspect" in set(s) else "ok")),
    ).sort_values("distinct")

    status_color = {"ok": "#1baf7a", "suspect": "#eda100", "unusable": "#e34948"}
    y = np.arange(len(agg))

    vals = agg["distinct"].to_numpy(dtype=float)
    # Fix the x-window first, so the threshold rules below are guaranteed to
    # fall inside it. Drawing a rule outside the window pushes its label off
    # the axes and stretches the whole figure.
    x_lo = min(8.0, vals.min() * 0.6)
    x_hi = vals.max() * 2.6

    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(agg) + 2.6))
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)

    # The two audit thresholds, shown rather than described. Labels sit at
    # the bottom of the plotting area in axes coords, clear of the subtitle.
    for xpos, color, text in ((10, "#e34948", "unusable < 10"),
                               (100, "#eda100", "suspect < 100")):
        if x_lo < xpos < x_hi:
            ax.axvline(xpos, color=color, lw=1, ls="--", zorder=2)
            ax.annotate(f" {text}", xy=(xpos, 0), xycoords=("data", "axes fraction"),
                        xytext=(2, 4), textcoords="offset points",
                        fontsize=8, color=color, va="bottom", ha="left")

    # Dot plot, not bars: this axis is logarithmic, and bar length encodes
    # magnitude measured from a zero baseline -- which a log axis does not
    # have, making bar lengths here meaningless. Dots encode position only.
    for yi, v in zip(y, vals):
        ax.plot([x_lo, v], [yi, yi], color=GRID, lw=1, zorder=1)
    ax.scatter(vals, y, s=70, zorder=3,
               color=[status_color[s] for s in agg["status"]])
    for yi, v, s in zip(y, vals, agg["status"]):
        ax.text(v * 1.12, yi, f"{int(v)}  ({s})", va="center", ha="left",
                fontsize=8, color=TEXT_PRIMARY)

    ax.set_yticks(y)
    ax.set_yticklabels(agg.index, fontsize=9, color=TEXT_PRIMARY)
    ax.set_ylim(-0.8, len(agg) - 0.2)
    _style_axes(ax, xlabel="distinct score values (min across receptors, log scale)",
                title="Tie-density diagnostic — score resolution per condition",
                subtitle="a condition with few distinct scores has its logAUC decided by "
                         "tie-break order, not by the model")
    _save(fig, out_dir / "tie_density_diagnostic.png")
    return True


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Default: <analysis-dir>/graphs")
    args = parser.parse_args()

    out_dir = args.output_dir or (args.analysis_dir / "graphs")

    print(f"Reading from {args.analysis_dir}")
    print(f"Writing plots to {out_dir}")

    panels = [
        ("Shapley attribution", plot_shapley),
        ("Möbius interactions", plot_mobius),
        ("Normalized ablation effect", plot_nae),
        ("Operator contrast", plot_operator_contrast),
        ("Structure-attributable excess", plot_structure_attributable),
        ("Tie-density diagnostic", plot_tie_density),
    ]

    # Count panels that actually rendered, not new files on disk -- a re-run
    # overwrites existing PNGs, which a file-count delta reports as zero.
    n_ok = 0
    n_skipped = 0
    n_failed = 0
    for label, fn in panels:
        print(f"\n{label}:")
        try:
            if fn(args.analysis_dir, out_dir):
                n_ok += 1
            else:
                n_skipped += 1
        except Exception as e:
            # One bad panel must not cost the others -- the inputs are
            # produced by independent upstream stages.
            print(f"  FAILED: {e}")
            n_failed += 1

    print(f"\n{n_ok} plot(s) written to {out_dir}"
          f"  ({n_skipped} skipped for missing input, {n_failed} failed)")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
