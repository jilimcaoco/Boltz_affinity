# Adapter evaluation: vanilla Boltz-2 vs LoRA-finetuned heads

This directory benchmarks the fine-tuned `drd4_v1` and `5ht2a_v1` LoRA
adapters against the vanilla Boltz-2 affinity head on **held-out in-vitro
data** (`../validation_data/D4_in_vitro_data.csv`,
`../validation_data/5ht2a_in_vitro_data.csv`).

The key question this answers:

> Does fine-tuning produce a more accurate **ranking** of compounds by
> bioactivity than the stock Boltz-2 affinity head?

**Ranking is the success criterion, not absolute calibration.** Predicted
pIC50 values from Boltz are not expected to be in the same units / on the
same scale as the assay readouts (Ki, % displacement at a single dose) used
here — what we care about is whether the experimental rank order is
reproduced by the model. All headline metrics therefore measure ordinal
agreement (Spearman ρ, Kendall τ, NDCG) or top-of-list retrieval
(ROC-AUC, BEDROC, EF, semi-logAUC). RMSE/MAE on pIC50 are reported only
when the experimental units allow it (DRD4 Ki) and only as secondary
diagnostics.

## Strategy

Both models score the **same predicted poses**, so any difference in
ranking comes purely from the LoRA-modified affinity head — not from
differences in the diffusion-sampled complex geometry.

```
                ┌──────────────────────────────┐
                │  validation_data/*.csv        │
                │  (ZINC ID, SMILES, in vitro)  │
                └───────────────┬───────────────┘
                                │ prepare_validation_inputs.py
                                ▼
              poses_inputs/{DRD4,5HT2A}/<id>.yaml
                                │ boltz predict   (vanilla, H200)
                                ▼
                 poses/{DRD4,5HT2A}/<id>.pdb           ← shared input
                       │                       │
        ┌──────────────┘                       └──────────────┐
        │ boltz rescore batch                  boltz rescore batch│ --use-lora
        ▼                                                       ▼
 scores/{TARGET}_vanilla.csv               scores/{TARGET}_lora.csv
        │                                                       │
        └────────────────────┬──────────────────────────────────┘
                             │ evaluate_adapters.py
                             ▼
              metrics/*.csv  +  plots/*.png
```

## Files

| File | Role |
|---|---|
| `prepare_validation_inputs.py` | Validation CSV → per-ligand Boltz YAMLs + a tidy `labels_{TARGET}.csv`. |
| `01_prepare_inputs.slurm` | CPU wrapper around the above. |
| `02_predict_poses.slurm` | `boltz predict` on validation YAMLs (H200). |
| `03_rescore_vanilla_lora.slurm` | `boltz rescore batch` × 2 per target (vanilla + LoRA), H200. |
| `evaluate_adapters.py` | Merge predictions + labels, compute metrics, write plots. |
| `04_evaluate.slurm` | CPU wrapper around `evaluate_adapters.py`. |
| `run_analysis.sh` | Submit the four jobs as a dependency chain. |

All GPU jobs request `--account=maom --partition=maom-h200`.

## Metrics

For each (target, model) pair we report (with 1 000-sample bootstrap 95% CIs):

**Headline — rank-order agreement (continuous label)**
- **Spearman ρ** — monotone correlation between predicted score and
  experimental activity. Primary success metric.
- **Kendall τ** — fraction of concordant pairs (more conservative than
  Spearman, less sensitive to ties).
- **NDCG** — Normalized Discounted Cumulative Gain on the full list, using
  shifted-to-non-negative experimental activity as graded relevance.
  Heavier weight on top-ranked items than Spearman, so it directly
  rewards getting the *strongest* binders to the top.
- **NDCG@10, NDCG@20, NDCG@top-5 %, NDCG@top-10 %** — the same metric
  cut to the top-k positions; the most actionable variants for screening.
- Pearson r is reported for reference but is **not** a success criterion.

**Retrieval / virtual-screening (binary label)**
- ROC-AUC, PR-AUC, average precision.
- BEDROC (α = 20.0, Truchon & Bayly 2007).
- Enrichment factor at 1 %, 5 %, 10 % of the screen.
- semi-logAUC (the metric used in the DUDE-Z / TLDR papers, integrated 0.001–1).

**Calibration (secondary, diagnostic only)**
- RMSE / MAE on `pred_pIC50` vs `exp_pIC50` (DRD4 only — 5HT2A label is a
  single-point displacement assay, not a true pIC50).

**Per-compound diagnostics**
- ΔpIC50 = pred_LoRA − pred_vanilla, broken down by ground-truth activity class.
- Rank-rank scatter (vanilla rank vs LoRA rank).

## Visualisations

`plots/` will contain, **per target**:

| File | What it shows |
|---|---|
| `scatter_{target}.png` | side-by-side scatter of predicted vs experimental, colour = binder class. |
| `roc_{target}.png` | ROC curves (vanilla vs LoRA) on the same axes. |
| `pr_{target}.png` | Precision–recall curves overlaid. |
| `enrichment_{target}.png` | Hits-recovered vs fraction-screened (log-x). |
| `metrics_bars_{target}.png` | Bar chart of point metrics with bootstrap CIs. |
| `delta_hist_{target}.png` | Histogram of per-compound ΔpIC50 (LoRA − vanilla), split by binder class. |
| `rank_rank_{target}.png` | Rank-rank scatter highlighting compounds that changed quartile. |

A combined `metrics_summary.csv` is also written.

## Bioactivity label conventions

We map every experimental readout to **two** columns the metrics code consumes:

- `exp_activity` — higher = more active (used for ranking metrics).
- `is_binder`    — binary label for retrieval metrics.

| Target | Source column | `exp_activity` | `is_binder` (default cutoff) |
|---|---|---|---|
| DRD4   | `D4 Ki(nM)` (continuous, lower = stronger) | `-log10(Ki_nM)` | column `Binder` (0/1) from the CSV |
| 5HT2A  | `% bound 3[H]-LSD @ 10uM (mean)` (lower = more displacement = stronger) | `-percent_bound` | `percent_bound ≤ 50 %` |

For DRD4 we *additionally* compute `exp_pIC50 = 9 − log10(Ki_nM)` and use it
for RMSE; for 5HT2A no calibration metric is reported because the assay only
yields a single dose-point.

## Submission

```bash
cd fineturning_experiment/analysis
chmod +x run_analysis.sh
./run_analysis.sh                  # full chain
./run_analysis.sh --from 02        # skip prepare
./run_analysis.sh --from 03        # skip prepare + predict (reuse poses)
./run_analysis.sh --from 04        # only re-run evaluation
```
