#!/usr/bin/env python
"""Evaluate fine-tuned LoRA adapters vs vanilla Boltz-2 on held-out validation data.

Inputs (per target):
    scores/{TARGET}_vanilla.csv       — output of `boltz rescore batch` (no LoRA / no FT)
    scores/{TARGET}_<adapter>.csv     — one CSV per adapter name (any name is accepted)
    labels/labels_{TARGET}.csv        — produced by prepare_validation_inputs.py

Model discovery is automatic: every ``{TARGET}_*.csv`` present in ``--scores-dir``
is included.  ``vanilla`` is always listed first; the rest appear in alphabetical
order.  Adapter names are taken verbatim from the filename stem (strip
``{target}_`` prefix), so ``DRD4_drd4_v2_rank_64.csv`` → model tag
``drd4_v2_rank_64``.

Versioned outputs are written to subdirectories named by ``--run-id`` so that
results from different adapter sets never overwrite each other:

    metrics/<run_id>/metrics_summary.csv
    metrics/<run_id>/per_compound_{TARGET}.csv
    plots/<run_id>/<lots of pngs, see README>

Pass ``--run-id ""`` to write directly into ``--metrics-dir`` / ``--plots-dir``
(backward-compatible behaviour).

Reports ranking, regression, and retrieval metrics (with bootstrap 95 % CIs)
plus per-compound delta diagnostics.  No scikit-learn dependency: only numpy,
pandas, scipy, matplotlib.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Metric primitives  (no sklearn — keeps env-deps minimal on HPC)
# ─────────────────────────────────────────────────────────────────────────────

def _rank_desc(scores: np.ndarray) -> np.ndarray:
    """Return 1-based ranks where the largest score gets rank 1 (ties = average)."""
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=float)
    np.add.at(sums, inv, ranks)
    means = sums / counts
    return means[inv]


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.concatenate(([0.0], tps / P))
    fpr = np.concatenate(([0.0], fps / N))
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    P = y.sum()
    if P == 0:
        return np.array([0.0]), np.array([1.0]), float("nan")
    tps = np.cumsum(y)
    precision = tps / np.arange(1, len(y) + 1)
    recall = tps / P
    # Average precision (step-wise integration over recall).
    ap = float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))
    return recall, precision, ap


def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """BEDROC (Truchon & Bayly 2007)."""
    N = len(y_true)
    n = int(y_true.sum())
    if n == 0 or n == N:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    ranks = np.where(y_true[order] == 1)[0] + 1  # 1-based
    Ra = n / N
    Sa = float(np.sum(np.exp(-alpha * ranks / N)))
    factor = Ra * np.sinh(alpha / 2.0) / (np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * Ra))
    return Sa * factor / n + 1.0 / (1.0 - np.exp(alpha * (1.0 - Ra)))


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    N = len(y_true)
    n = int(y_true.sum())
    k = max(1, int(math.ceil(frac * N)))
    if n == 0 or N == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    hits_top = int(y_true[order][:k].sum())
    return (hits_top / k) / (n / N)


def semilog_auc(y_true: np.ndarray, y_score: np.ndarray, lam: float = 0.001) -> float:
    """Semi-log ROC AUC integrated on log10(FPR) ∈ [log10(lam), 0] (DUDE-Z style)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr >= lam
    if mask.sum() < 2:
        return float("nan")
    x = np.log10(np.clip(fpr[mask], lam, 1.0))
    y = tpr[mask]
    auc = float(np.trapz(y, x)) / (-np.log10(lam))
    return auc


def _dcg(rels: np.ndarray) -> float:
    # Standard DCG with log2(i+1) discount, 1-based positions.
    positions = np.arange(1, len(rels) + 1, dtype=float)
    return float(np.sum(rels / np.log2(positions + 1.0)))


def ndcg(y_true_relevance: np.ndarray, y_score: np.ndarray,
         k: Optional[int] = None) -> float:
    """Normalized Discounted Cumulative Gain.

    ``y_true_relevance`` should be non-negative graded relevance (higher = more
    active).  For continuous bioactivity we shift to be ≥ 0 (subtract the
    minimum) so DCG is well-defined; for binary labels this reduces to the
    standard binary NDCG.  ``k`` is the cutoff (``None`` = full list).
    """
    rel = np.asarray(y_true_relevance, dtype=float)
    rel = rel - rel.min()  # shift to non-negative
    if rel.sum() == 0:
        return float("nan")
    n = len(rel)
    kk = n if k is None else min(int(k), n)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    dcg = _dcg(rel[order][:kk])
    ideal = _dcg(np.sort(rel)[::-1][:kk])
    if ideal <= 0:
        return float("nan")
    return dcg / ideal


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    func: Callable[[np.ndarray, np.ndarray], float],
    a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0,
) -> tuple[float, float, float]:
    """Returns (point, lo95, hi95). Resamples indices with replacement."""
    rng = np.random.default_rng(seed)
    point = func(a, b)
    if not np.isfinite(point) or len(a) < 3:
        return point, float("nan"), float("nan")
    boots = np.empty(n)
    idx_pool = np.arange(len(a))
    for i in range(n):
        idx = rng.choice(idx_pool, size=len(a), replace=True)
        try:
            boots[i] = func(a[idx], b[idx])
        except Exception:
            boots[i] = np.nan
    boots = boots[np.isfinite(boots)]
    if len(boots) < 10:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────────
# Score loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_scores(path: Path, model_tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep only successful rows.
    if "validation_status" in df.columns:
        df = df[df["validation_status"].astype(str).str.upper() == "SUCCESS"]
    if "affinity_pred" not in df.columns:
        raise ValueError(f"{path} missing 'affinity_pred' column")
    keep = ["id", "affinity_pred"]
    if "affinity_probability_binary" in df.columns:
        keep.append("affinity_probability_binary")
    df = df[keep].rename(columns={
        "affinity_pred": f"pred_pIC50_{model_tag}",
        "affinity_probability_binary": f"prob_binder_{model_tag}",
    })
    return df


# Fixed colours for well-known model tags; all others are assigned dynamically.
_FIXED_COLORS: dict[str, str] = {
    "vanilla":  "tab:gray",
    "lora":     "tab:red",
    "finetune": "tab:blue",
}
# Ordered palette for dynamically-named adapters (cycles if > len).
_DYNAMIC_PALETTE = [
    "tab:orange", "tab:green", "tab:purple", "tab:brown",
    "tab:pink",   "tab:olive", "tab:cyan",
]


def _model_color(model: str, all_models: list[str]) -> str:
    """Return a stable matplotlib color for *model*.

    Known model names get a fixed color; unknown names are assigned a color
    from ``_DYNAMIC_PALETTE`` in the order they appear in *all_models*.
    """
    if model in _FIXED_COLORS:
        return _FIXED_COLORS[model]
    dynamic = [m for m in all_models if m not in _FIXED_COLORS]
    idx = dynamic.index(model) % len(_DYNAMIC_PALETTE)
    return _DYNAMIC_PALETTE[idx]


def _discover_models(target: str, scores_dir: Path) -> list[str]:
    """Return all model tags that have a score CSV for *target*.

    Globs ``{target}_*.csv`` in *scores_dir*, strips the ``{target}_`` prefix
    to obtain the model tag, and returns the list with ``vanilla`` always
    first followed by the rest in alphabetical order.

    Raises ``FileNotFoundError`` if no CSVs are found or if
    ``{target}_vanilla.csv`` is absent.
    """
    csvs = sorted(scores_dir.glob(f"{target}_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No score CSVs found for target '{target}' in {scores_dir}"
        )
    prefix = f"{target}_"
    tags = [p.stem[len(prefix):] for p in csvs]
    if "vanilla" not in tags:
        raise FileNotFoundError(
            f"Required file '{target}_vanilla.csv' not found in {scores_dir}"
        )
    ordered = ["vanilla"] + sorted(t for t in tags if t != "vanilla")
    return ordered


def _merge(target: str, scores_dir: Path, labels_dir: Path,
           models: list[str]) -> pd.DataFrame:
    labels = pd.read_csv(labels_dir / f"labels_{target}.csv")
    df = labels
    for tag in models:
        scores = _load_scores(scores_dir / f"{target}_{tag}.csv", tag)
        df = df.merge(scores, left_on="name", right_on="id", how="inner").drop(columns=["id"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-target metrics
# ─────────────────────────────────────────────────────────────────────────────

# Boltz outputs `affinity_pred` ≈ predicted log10(IC50_µM) (lower = stronger).
# Convert to a "higher = stronger" score for ranking against exp_activity.
def _ranker(pred_log10_ic50_um: np.ndarray) -> np.ndarray:
    return -pred_log10_ic50_um


def _safe_corr(fn, x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    r = fn(x, y)
    return float(r.correlation) if hasattr(r, "correlation") else float(r[0])


def compute_metrics(df: pd.DataFrame, target: str, n_boot: int = 1000,
                    models: Optional[list[str]] = None) -> pd.DataFrame:
    rows: list[dict] = []
    if models is None:
        models = ["vanilla", "lora"]
    for model in models:
        pred_col = f"pred_pIC50_{model}"
        score = _ranker(df[pred_col].to_numpy(float))

        # Ranking on continuous activity (drop NaNs).  These are the
        # **headline** metrics — calibration matters far less than rank order.
        mask = df["exp_activity"].notna()
        if mask.sum() >= 3:
            a = df.loc[mask, "exp_activity"].to_numpy(float)
            s = score[mask.to_numpy()]
            spearman, sp_lo, sp_hi = bootstrap_ci(
                lambda x, y: _safe_corr(stats.spearmanr, x, y), a, s, n_boot)
            kendall, kt_lo, kt_hi = bootstrap_ci(
                lambda x, y: _safe_corr(stats.kendalltau, x, y), a, s, n_boot)
            pearson = _safe_corr(stats.pearsonr, a, s)
            ndcg_full, ndcg_lo, ndcg_hi = bootstrap_ci(
                lambda x, y: ndcg(x, y, None), a, s, n_boot)
            ndcg_10 = ndcg(a, s, 10)
            ndcg_20 = ndcg(a, s, 20)
            ndcg_top5pct  = ndcg(a, s, max(1, int(math.ceil(0.05 * len(a)))))
            ndcg_top10pct = ndcg(a, s, max(1, int(math.ceil(0.10 * len(a)))))
        else:
            spearman = sp_lo = sp_hi = float("nan")
            kendall = kt_lo = kt_hi = pearson = float("nan")
            ndcg_full = ndcg_lo = ndcg_hi = float("nan")
            ndcg_10 = ndcg_20 = ndcg_top5pct = ndcg_top10pct = float("nan")

        # Regression vs experimental pIC50 (only DRD4 has it).
        if "exp_pIC50" in df.columns and df["exp_pIC50"].notna().sum() >= 3:
            m = df["exp_pIC50"].notna()
            # Boltz `affinity_pred` is log10(IC50_µM); convert to pIC50 = 6 - log10(IC50_µM).
            pred_pic50 = 6.0 - df.loc[m, pred_col].to_numpy(float)
            obs = df.loc[m, "exp_pIC50"].to_numpy(float)
            rmse = float(np.sqrt(np.mean((pred_pic50 - obs) ** 2)))
            mae  = float(np.mean(np.abs(pred_pic50 - obs)))
        else:
            rmse = mae = float("nan")

        # Retrieval on binary label.
        b_mask = df["is_binder"].notna()
        if b_mask.sum() >= 5 and df.loc[b_mask, "is_binder"].nunique() == 2:
            y = df.loc[b_mask, "is_binder"].to_numpy(int)
            sc = score[b_mask.to_numpy()]
            _, _, auc = roc_curve(y, sc)
            _, _, ap  = pr_curve(y, sc)
            bed = bedroc(y, sc, alpha=20.0)
            ef1   = enrichment_factor(y, sc, 0.01)
            ef5   = enrichment_factor(y, sc, 0.05)
            ef10  = enrichment_factor(y, sc, 0.10)
            slauc = semilog_auc(y, sc, lam=0.001)
            _, auc_lo, auc_hi = bootstrap_ci(lambda yy, ss: roc_curve(yy, ss)[2], y, sc, n_boot)
            _, bed_lo, bed_hi = bootstrap_ci(lambda yy, ss: bedroc(yy, ss, 20.0), y, sc, n_boot)
            _, ef1_lo, ef1_hi = bootstrap_ci(lambda yy, ss: enrichment_factor(yy, ss, 0.01), y, sc, n_boot)
        else:
            auc = ap = bed = ef1 = ef5 = ef10 = slauc = float("nan")
            auc_lo = auc_hi = bed_lo = bed_hi = ef1_lo = ef1_hi = float("nan")

        rows.append({
            "target": target, "model": model,
            "n_compounds": int(len(df)),
            "n_with_activity": int(mask.sum()),
            "n_binders": int(df["is_binder"].fillna(0).sum()) if "is_binder" in df else 0,
            # Headline ranking metrics
            "spearman": spearman, "spearman_lo": sp_lo, "spearman_hi": sp_hi,
            "kendall": kendall, "kendall_lo": kt_lo, "kendall_hi": kt_hi,
            "pearson": pearson,
            "ndcg": ndcg_full, "ndcg_lo": ndcg_lo, "ndcg_hi": ndcg_hi,
            "ndcg_top5pct":  ndcg_top5pct,
            "ndcg_top10pct": ndcg_top10pct,
            "ndcg_at_10": ndcg_10, "ndcg_at_20": ndcg_20,
            # Calibration (secondary — only meaningful when units match)
            "rmse_pIC50": rmse, "mae_pIC50": mae,
            # Retrieval / virtual screening
            "roc_auc": auc, "roc_auc_lo": auc_lo, "roc_auc_hi": auc_hi,
            "pr_auc": ap,
            "bedroc_20": bed, "bedroc_20_lo": bed_lo, "bedroc_20_hi": bed_hi,
            "ef_1pct":  ef1, "ef_1pct_lo": ef1_lo, "ef_1pct_hi": ef1_hi,
            "ef_5pct":  ef5, "ef_10pct": ef10,
            "semilog_auc": slauc,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _scatter(df: pd.DataFrame, target: str, plots: Path,
             models: list[str]) -> None:
    mask = df["exp_activity"].notna()
    if mask.sum() < 3:
        return
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True, squeeze=False)
    axes = axes[0]
    obs = df.loc[mask, "exp_activity"].to_numpy(float)
    binder = df.loc[mask, "is_binder"].fillna(-1).astype(int).to_numpy()
    for ax, model in zip(axes, models):
        pred = -df.loc[mask, f"pred_pIC50_{model}"].to_numpy(float)
        rho = _safe_corr(stats.spearmanr, obs, pred)
        for cls, color, label in [(1, "tab:red", "binder"),
                                  (0, "tab:gray", "non-binder"),
                                  (-1, "tab:blue", "unlabeled")]:
            sel = binder == cls
            if sel.any():
                ax.scatter(obs[sel], pred[sel], s=18, alpha=0.7, c=color, label=label)
        ax.set_xlabel("experimental activity (higher = stronger)")
        ax.set_title(f"{target} · {model}\nρ = {rho:.3f}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("predicted score (−Boltz pIC50, higher = stronger)")
    fig.suptitle(f"{target}: {' vs '.join(models)} — predicted vs experimental")
    fig.tight_layout()
    fig.savefig(plots / f"scatter_{target}.png", dpi=180)
    plt.close(fig)


def _roc(df: pd.DataFrame, target: str, plots: Path,
         models: list[str]) -> None:
    m = df["is_binder"].notna()
    if m.sum() < 5 or df.loc[m, "is_binder"].nunique() != 2:
        return
    y = df.loc[m, "is_binder"].to_numpy(int)
    fig, ax = plt.subplots(figsize=(5, 5))
    for model in models:
        color = _model_color(model, models)
        sc = -df.loc[m, f"pred_pIC50_{model}"].to_numpy(float)
        fpr, tpr, auc = roc_curve(y, sc)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{model} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="k", ls="--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{target} — ROC")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots / f"roc_{target}.png", dpi=180); plt.close(fig)


def _pr(df: pd.DataFrame, target: str, plots: Path,
        models: list[str]) -> None:
    m = df["is_binder"].notna()
    if m.sum() < 5 or df.loc[m, "is_binder"].nunique() != 2:
        return
    y = df.loc[m, "is_binder"].to_numpy(int)
    fig, ax = plt.subplots(figsize=(5, 5))
    for model in models:
        color = _model_color(model, models)
        sc = -df.loc[m, f"pred_pIC50_{model}"].to_numpy(float)
        rec, prec, ap = pr_curve(y, sc)
        ax.plot(rec, prec, color=color, lw=2, label=f"{model} (AP={ap:.3f})")
    base = y.mean()
    ax.axhline(base, color="k", ls="--", lw=1, label=f"random (AP={base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{target} — Precision-Recall")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots / f"pr_{target}.png", dpi=180); plt.close(fig)


def _enrichment(df: pd.DataFrame, target: str, plots: Path,
                models: list[str]) -> None:
    m = df["is_binder"].notna()
    if m.sum() < 5 or df.loc[m, "is_binder"].nunique() != 2:
        return
    y = df.loc[m, "is_binder"].to_numpy(int)
    N = len(y); n = int(y.sum())
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fracs = np.linspace(1 / N, 1.0, N)
    for model in models:
        color = _model_color(model, models)
        sc = -df.loc[m, f"pred_pIC50_{model}"].to_numpy(float)
        order = np.argsort(-sc, kind="mergesort")
        hits_cum = np.cumsum(y[order]) / n
        ax.plot(fracs, hits_cum, color=color, lw=2, label=model)
    ax.plot(fracs, fracs, color="k", ls="--", lw=1, label="random")
    ax.set_xscale("log"); ax.set_xlim(max(1 / N, 1e-3), 1.0)
    ax.set_xlabel("Fraction of library screened (log)")
    ax.set_ylabel("Fraction of binders recovered")
    ax.set_title(f"{target} — enrichment")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(plots / f"enrichment_{target}.png", dpi=180); plt.close(fig)


def _bars(metrics: pd.DataFrame, target: str, plots: Path,
          models: list[str]) -> None:
    sub = metrics[metrics["target"] == target].set_index("model")
    fields = [
        ("spearman", "spearman_lo", "spearman_hi", "Spearman ρ"),
        ("kendall",  "kendall_lo",  "kendall_hi",  "Kendall τ"),
        ("ndcg",     "ndcg_lo",     "ndcg_hi",     "NDCG"),
        ("roc_auc",  "roc_auc_lo",  "roc_auc_hi",  "ROC-AUC"),
        ("bedroc_20","bedroc_20_lo","bedroc_20_hi","BEDROC (α=20)"),
        ("ef_1pct",  "ef_1pct_lo",  "ef_1pct_hi",  "EF @ 1 %"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(fields))
    n_mod = len(models)
    w = 0.8 / max(n_mod, 1)
    for i, model in enumerate(models):
        if model not in sub.index:
            continue
        vals = [sub.loc[model, k] for k, _, _, _ in fields]
        los  = [sub.loc[model, l] for _, l, _, _ in fields]
        his  = [sub.loc[model, h] for _, _, h, _ in fields]
        err_lo = [v - lo if np.isfinite(lo) else 0 for v, lo in zip(vals, los)]
        err_hi = [hi - v if np.isfinite(hi) else 0 for v, hi in zip(vals, his)]
        offset = (i - (n_mod - 1) / 2.0) * w
        ax.bar(x + offset, vals, w,
               yerr=[err_lo, err_hi], capsize=3,
               label=model, color=_model_color(model, models))
    ax.set_xticks(x); ax.set_xticklabels([f[3] for f in fields], rotation=15)
    ax.set_title(f"{target}: ranking metrics (bootstrap 95 % CI)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(plots / f"metrics_bars_{target}.png", dpi=180); plt.close(fig)


def _delta_hist(df: pd.DataFrame, target: str, plots: Path,
                models: list[str]) -> None:
    """One ΔpIC50 panel per non-vanilla model (vs vanilla)."""
    others = [m for m in models if m != "vanilla"]
    if not others:
        return
    fig, axes = plt.subplots(1, len(others), figsize=(6 * len(others), 4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, others):
        delta = (-df[f"pred_pIC50_{model}"]) - (-df["pred_pIC50_vanilla"])
        if "is_binder" in df.columns and df["is_binder"].notna().any():
            for cls, color, label in [(1, "tab:red", "binders"),
                                      (0, "tab:gray", "non-binders")]:
                sel = df["is_binder"] == cls
                if sel.any():
                    ax.hist(delta[sel].dropna(), bins=25, alpha=0.6,
                            color=color, label=label)
        else:
            ax.hist(delta.dropna(), bins=25, color="tab:blue")
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel(f"Δ score ({model} − vanilla)")
        ax.set_title(f"{target}: Δ ({model} − vanilla)")
        ax.legend(); ax.grid(alpha=0.3)
    axes[0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(plots / f"delta_hist_{target}.png", dpi=180)
    plt.close(fig)


def _rank_rank(df: pd.DataFrame, target: str, plots: Path,
               models: list[str]) -> None:
    """Rank-rank scatter of every non-vanilla model against vanilla."""
    others = [m for m in models if m != "vanilla"]
    if not others:
        return
    rv = _rank_desc(-df["pred_pIC50_vanilla"].to_numpy(float))
    fig, axes = plt.subplots(1, len(others), figsize=(5.5 * len(others), 5.5),
                             squeeze=False)
    axes = axes[0]
    binder = df["is_binder"].fillna(-1).astype(int).to_numpy() if "is_binder" in df.columns \
        else np.full(len(df), -1)
    for ax, model in zip(axes, others):
        rl = _rank_desc(-df[f"pred_pIC50_{model}"].to_numpy(float))
        for cls, color, label in [(1, "tab:red", "binder"),
                                  (0, "tab:gray", "non-binder"),
                                  (-1, "tab:blue", "unlabeled")]:
            sel = binder == cls
            if sel.any():
                ax.scatter(rv[sel], rl[sel], s=18, alpha=0.7, c=color, label=label)
        lim = max(rv.max(), rl.max())
        ax.plot([1, lim], [1, lim], color="k", ls="--", lw=1)
        ax.set_xlabel("vanilla rank (1 = best)")
        ax.set_ylabel(f"{model} rank (1 = best)")
        ax.set_title(f"{target}: vanilla vs {model}")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / f"rank_rank_{target}.png", dpi=180)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def run_target(target: str, scores_dir: Path, labels_dir: Path,
               metrics_dir: Path, plots_dir: Path, n_boot: int) -> Optional[pd.DataFrame]:
    models = _discover_models(target, scores_dir)
    print(f"[{target}] models: {models}")
    try:
        df = _merge(target, scores_dir, labels_dir, models)
    except FileNotFoundError as e:
        print(f"[{target}] SKIP: {e}")
        return None
    if df.empty:
        print(f"[{target}] SKIP: empty merge")
        return None
    df.to_csv(metrics_dir / f"per_compound_{target}.csv", index=False)

    metrics = compute_metrics(df, target, n_boot=n_boot, models=models)
    _scatter(df, target, plots_dir, models)
    _roc(df, target, plots_dir, models)
    _pr(df, target, plots_dir, models)
    _enrichment(df, target, plots_dir, models)
    _delta_hist(df, target, plots_dir, models)
    _rank_rank(df, target, plots_dir, models)
    _bars(metrics, target, plots_dir, models)
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores-dir",  required=True, type=Path)
    p.add_argument("--labels-dir",  required=True, type=Path)
    p.add_argument("--metrics-dir", required=True, type=Path)
    p.add_argument("--plots-dir",   required=True, type=Path)
    p.add_argument("--targets", nargs="+", default=["DRD4", "5HT2A"])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument(
        "--run-id", default="",
        help="Optional subdirectory name that namespaces metrics and plots so "
             "previous results are not overwritten (e.g. 'drd4_v2_vs_v1').  "
             "Leave empty to write directly into --metrics-dir / --plots-dir.",
    )
    args = p.parse_args()

    metrics_dir = args.metrics_dir / args.run_id if args.run_id else args.metrics_dir
    plots_dir   = args.plots_dir   / args.run_id if args.run_id else args.plots_dir
    # Redirect for the rest of the function.
    args.metrics_dir = metrics_dir
    args.plots_dir   = plots_dir

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[pd.DataFrame] = []
    for tgt in args.targets:
        m = run_target(tgt, args.scores_dir, args.labels_dir,
                       args.metrics_dir, args.plots_dir, args.n_boot)
        if m is not None:
            all_metrics.append(m)

    if all_metrics:
        summary = pd.concat(all_metrics, ignore_index=True)
        out = args.metrics_dir / "metrics_summary.csv"
        summary.to_csv(out, index=False)
        print(f"[evaluate] wrote {out}")
        # Pretty console summary.
        cols = ["target", "model", "n_compounds", "n_binders",
                "spearman", "kendall", "ndcg", "ndcg_top10pct",
                "roc_auc", "bedroc_20", "ef_1pct", "ef_5pct", "semilog_auc"]
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(summary[cols].to_string(index=False))
    else:
        print("[evaluate] No metrics produced (no merges succeeded).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
