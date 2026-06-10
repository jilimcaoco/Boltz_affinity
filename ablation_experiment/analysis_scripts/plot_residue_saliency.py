#!/usr/bin/env python3
"""Aggregate residue LOO results and produce saliency heatmaps.

Inputs
------
- ``residue_loo_results.csv`` produced by ``run_residue_loo.py`` with
  columns ``receptor_id, ligand_name, is_decoy, token_index,
  residue_index, residue_name, asym_id, distance_to_ligand_A,
  baseline_pred, loo_pred, delta_pred, abs_delta_pred``.

Outputs (per receptor)
----------------------
1. ``saliency_by_residue.csv``: per (receptor, residue) — mean and std
   of ``delta_pred`` and ``abs_delta_pred``, split by class (active /
   decoy / all), plus the residue's min distance to the ligand.
2. ``saliency_heatmap_<RECEPTOR>.png``: residues × ligands heatmap of
   ``delta_pred``. Residues sorted by mean |Δ|. Actives and decoys
   plotted in separate column blocks.
3. ``saliency_top10_<RECEPTOR>.csv``: top-10 residues per receptor by
   mean |Δ|, with both active and decoy mean Δ for comparison.

Usage
-----
python plot_residue_saliency.py \
    --input ../results/loo/AA2AR_residue_loo.csv \
    --output-dir ../analysis_data/residue_loo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per (receptor_id, token_index) mean/std of |Δ| and Δ, split by class."""
    rows = []
    for (rec, tok), grp in df.groupby(["receptor_id", "token_index"]):
        meta_row = grp.iloc[0]
        residue_index = meta_row["residue_index"]
        residue_name = meta_row["residue_name"]
        asym_id = meta_row["asym_id"]
        dist = meta_row["distance_to_ligand_A"]
        all_abs = grp["abs_delta_pred"]
        all_delta = grp["delta_pred"]
        act = grp[grp["is_decoy"] == 0]
        dec = grp[grp["is_decoy"] == 1]
        rows.append({
            "receptor_id": rec,
            "token_index": tok,
            "residue_index": residue_index,
            "residue_name": residue_name,
            "asym_id": asym_id,
            "distance_to_ligand_A": dist,
            "n_ligands": len(grp),
            "n_actives": len(act),
            "n_decoys": len(dec),
            "mean_abs_delta_all": float(all_abs.mean()),
            "mean_delta_all": float(all_delta.mean()),
            "std_delta_all": float(all_delta.std(ddof=1) if len(all_delta) > 1 else float("nan")),
            "mean_abs_delta_actives": (
                float(act["abs_delta_pred"].mean()) if len(act) else float("nan")
            ),
            "mean_delta_actives": (
                float(act["delta_pred"].mean()) if len(act) else float("nan")
            ),
            "mean_abs_delta_decoys": (
                float(dec["abs_delta_pred"].mean()) if len(dec) else float("nan")
            ),
            "mean_delta_decoys": (
                float(dec["delta_pred"].mean()) if len(dec) else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def heatmap_for_receptor(df_rec: pd.DataFrame, out_path: Path,
                         top_k: int | None = None):
    """Heatmap: residues (sorted by mean |Δ|) × ligands, actives then decoys."""
    if df_rec.empty:
        return

    # Compute residue ordering by overall mean |Δ|
    agg = (df_rec.groupby("token_index")["abs_delta_pred"].mean()
                .sort_values(ascending=False))
    if top_k is not None:
        agg = agg.head(top_k)
    residue_order = agg.index.tolist()

    # Split actives vs decoys; preserve a deterministic ligand order.
    act_ligs = sorted(df_rec.loc[df_rec["is_decoy"] == 0, "ligand_name"].unique())
    dec_ligs = sorted(df_rec.loc[df_rec["is_decoy"] == 1, "ligand_name"].unique())
    ligand_order = act_ligs + dec_ligs

    pivot = df_rec.pivot_table(
        index="token_index", columns="ligand_name",
        values="delta_pred", aggfunc="mean",
    )
    pivot = pivot.reindex(index=residue_order, columns=ligand_order)
    mat = pivot.to_numpy(dtype=float)

    # Symmetric color scale centred at 0
    finite = mat[np.isfinite(mat)]
    if finite.size:
        vmax = float(np.percentile(np.abs(finite), 99))
        if vmax == 0:
            vmax = 1e-6
    else:
        vmax = 1.0

    fig_h = max(4, 0.18 * len(residue_order))
    fig_w = max(8, 0.18 * len(ligand_order))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # Y labels = residue_name + residue_index
    y_labels = []
    for ti in residue_order:
        sub = df_rec[df_rec["token_index"] == ti].iloc[0]
        y_labels.append(f"{sub['residue_name']}{int(sub['residue_index'])}")
    ax.set_yticks(np.arange(len(residue_order)))
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xticks(np.arange(len(ligand_order)))
    ax.set_xticklabels(ligand_order, rotation=90, fontsize=5)

    # Divider line between actives and decoys
    if act_ligs and dec_ligs:
        ax.axvline(len(act_ligs) - 0.5, color="k", lw=1.0)
        ax.text(len(act_ligs) / 2 - 0.5, -0.6, "actives",
                ha="center", fontsize=8)
        ax.text(len(act_ligs) + len(dec_ligs) / 2 - 0.5, -0.6, "decoys",
                ha="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δ pIC50 (baseline − LOO)")
    receptor_id = df_rec["receptor_id"].iloc[0]
    ax.set_title(f"Per-residue LOO saliency — {receptor_id}", fontsize=11)
    ax.set_xlabel("ligand")
    ax.set_ylabel("residue (token-ordered)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    default_input = here.parent / "results" / "loo" / "residue_loo_results.csv"
    default_out = here.parent / "analysis_data" / "residue_loo"
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--top-k", type=int, default=40,
                        help="Plot only the top-K residues by mean |Δ|.")
    parser.add_argument("--receptors", nargs="+", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    if args.receptors:
        df = df[df["receptor_id"].isin(args.receptors)]
    if df.empty:
        raise SystemExit("No LOO rows after filtering.")

    agg = aggregate(df)
    agg.sort_values(
        ["receptor_id", "mean_abs_delta_all"], ascending=[True, False],
        inplace=True,
    )
    out_csv = args.output_dir / "saliency_by_residue.csv"
    agg.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(agg)} rows)")

    for receptor_id, df_rec in df.groupby("receptor_id"):
        heatmap_path = args.output_dir / f"saliency_heatmap_{receptor_id}.png"
        heatmap_for_receptor(df_rec, heatmap_path, top_k=args.top_k)
        print(f"  wrote {heatmap_path}")

        top = (agg[agg["receptor_id"] == receptor_id]
               .head(10)[[
                   "residue_name", "residue_index", "asym_id",
                   "distance_to_ligand_A", "mean_abs_delta_all",
                   "mean_delta_all", "mean_delta_actives",
                   "mean_delta_decoys",
               ]])
        top_path = args.output_dir / f"saliency_top10_{receptor_id}.csv"
        top.to_csv(top_path, index=False)
        print(f"  wrote {top_path}")


if __name__ == "__main__":
    main()
