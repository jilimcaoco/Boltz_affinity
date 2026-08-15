# Ablation experiment

Mechanistic ablation study of the Boltz-2 affinity head: which information
channel — the trunk pair representation (`z_trunk`), the single/atom-level
representation (`s_inputs`), or the explicit 3D distogram — actually drives
the model's affinity predictions, and whether the head is taking a
receptor/ligand-*identity* shortcut instead of genuinely reading the
predicted complex geometry.

**Read this before touching results.** A tie-handling bug in the logAUC
bootstrap (fixed; see `logauc_utils.py`) previously inflated near-constant
score distributions — most notably `bias_only`, the anchor of every
normalized effect size in this study — toward the theoretical maximum
logAUC. Before trusting any existing result, run
`analysis_scripts/audit_tie_density.py` against it and check
`analysis_data/tie_density_audit.csv`.

## Data

Results (`results/`) and derived analysis outputs (`analysis_data/`) are
gitignored and live on the training cluster, not in this checkout — only
the scripts and tests are tracked. Everything below describes what those
scripts do; to actually run them you need the full `boltz` environment
(a GPU + checkpoint for the `experiment_scripts/`, just `numpy`/`pandas`/
`scipy`/`rdkit`/`scikit-learn` for the `analysis_scripts/`).

## Pipeline

```
experiment_scripts/
  run_feature_ablation.py     Main driver. Featurizes each (receptor, ligand),
                               caches the trunk output to disk once, then
                               replays every requested ablation experiment's
                               affinity head against the cache.
  trunk_cache.py               Disk cache (keyed on receptor+ligand) and the
                               token crop/pad + donor-matching helpers the
                               resample/mean operators use.
  verify_trunk_cache.py        Fidelity gate: recomputes a stratified sample
                               uncached and asserts it matches the cached
                               replay to |Δ pIC50| < 1e-4. Must pass before
                               any result from the cache is trusted.
  run_distogram_ablation.py    DEFERRED — do not extend.
  run_residue_loo.py           DEFERRED — do not extend.

analysis_scripts/
  logauc_utils.py               Shared logAUC/ROC/bootstrap core (tie-aware,
                                 cluster bootstrap, BCa, paired-difference).
                                 Both bootstrap scripts import from here so
                                 the tie fix has one place to live.
  audit_tie_density.py          Task 0b/0c: tie-density audit -- run this
                                 first, on real data, before anything else.
  compute_ablation_bootstrap.py Per-receptor + population-level (cluster)
                                 bootstrapped logAUC for every ablation
                                 experiment, plus paired differences vs a
                                 baseline arm.
  compute_logauc_bootstrap.py   Same core, for the 4-method score comparison
                                 (DiffDock / DOCK3.8 / OG_Affinity / Boltz
                                 rescoring) rather than the ablation matrix.
  compute_shapley_interactions.py
                                 Exact Shapley attribution + pairwise Möbius
                                 interaction over the 2^3 factorial.
  compute_property_control.py   Ligand-property residual control + 2D-only
                                 ECFP4 ceiling + AVE bias.
  compute_effect_normalization.py
                                 Normalized Ablation Effect (NAE), receptor
                                 class breakdowns, rank preservation.
  plot_ablation_summary.py      Plots over the new columns: Shapley
                                 attribution, Möbius interactions, NAE by
                                 receptor class, zero-vs-resample operator
                                 contrast, structure-attributable excess,
                                 tie-density diagnostic. Each panel is
                                 skipped (not failed) when its input is
                                 absent, which is a normal intermediate
                                 state here.
  combine_dudez_results.py      Merges the per-receptor DUDEZ CSVs into the
                                 single table the rest of the chain reads.
  compute_distribution_stats.py, combine_scores.py, bootstrap_tldr.py,
  plot_ridgeline.py, plot_scatter.py
                                 Supporting utilities, unchanged. The two
                                 plot_* scripts cover the *method
                                 comparison* panel and know nothing about
                                 the ablation columns — that's
                                 plot_ablation_summary.py's job.
  plot_residue_saliency.py      DEFERRED — do not extend.

run_analysis.sh                 The CPU analysis chain (stages 0-6) with the
                                 tie-audit gate. See "Running it" below.
```

## Running it

Two stages, split by what needs a GPU. There are two input flows; **the
DUDEZ flow is the primary one.**

### Input flows

| Flow | Runner | Input | Labels |
|---|---|---|---|
| **DUDEZ (primary)** | `run_feature_ablation_dudez.py` | precomputed Boltz-2 predicted complexes (`.cif`) + SMILES manifest + cached MSA | authoritative `is_binder` from the manifest |
| MOL2 (legacy) | `run_feature_ablation.py` | static receptor PDB + multi-pose MOL2 | inferred from a `ZINC` name prefix |

The DUDEZ flow is preferred because the study's premise is that a valid
structure is already in hand — the head is fed that structure directly,
sidestepping diffusion. It also avoids two silent failure modes: RDKit
SMILES inference dropping ligands, and the name-prefix heuristic
mislabelling decoys that aren't named `ZINC*` (which would leave a receptor
with zero decoys).

Both runners share the trunk cache, the operator machinery, the output
schema, and the whole analysis chain. **Keep their trunk caches separate**
(`--trunk-cache-dir`): entries are keyed on `(receptor, ligand)` with no
record of which pose source produced them.

### Storage

A cached trunk entry is **~16 MB** — `z` is `(N, N, token_z)` fp16, so it
grows with the *square* of the token count (256 tokens × 128 channels).
Caching every ligand would cost 80–320 GB per receptor, i.e. multiple TB
across 43 receptors. Two things keep that bounded:

- **A zero-only run writes nothing to disk.** The cache exists solely to
  supply donor channels for `resample`/`mean`; a zero-operator experiment
  never reads another ligand's trunk. The query's trunk is computed once
  and held in memory only for that ligand.
- **When donors are needed, only a bounded pool is cached** —
  `--donor-pool-size` (default 64), drawn at random and stratified by
  `is_binder` so the pool isn't all actives. `resample` draws 5 donors per
  query and `mean` is a sample mean, so a pool of 64 is ample. That's ~1 GB
  per receptor instead of hundreds, and it's deleted at the end of the run
  unless you pass `--keep-trunk-cache`.

The SLURM script puts the cache on `$SCRATCH_BASE` (not the repo tree),
removes structures it extracted itself unless `KEEP_STRUCTURES=1`, and
prints `du -sh` of both before cleanup. The tarball stays the source of
truth — nothing extracted is worth keeping.

**1. GPU — one array task per receptor**

```bash
sbatch slurm_scripts/feature_ablation_dudez.slurm
```

Untars structures, runs the ablation, then the trunk-cache fidelity gate
(hard stop if cached replay disagrees with an uncached recomputation).
Writes `results/dudez_ablation/<RECEPTOR>_dudez_ablation.csv`.

For the legacy MOL2 flow instead: `sbatch slurm_scripts/feature_ablation.slurm`.

**2. CPU — `run_analysis.sh`**

```bash
cd ablation_experiment && ./run_analysis.sh --from-dudez
```

`--from-dudez` first merges the per-receptor CSVs into the single table the
chain reads (stage 0a). Omit it for the MOL2 flow, which already writes one
combined file.

Writes `analysis_data/tie_density_audit.csv` and **stops for review** — a
tie-dense `bias_only` is the expected *finding*, not something a script can
adjudicate, and `bias_only` is the `v(∅)` anchor under every Shapley value
and NAE denominator downstream. Once you've read it:

```bash
./run_analysis.sh --from-dudez --tie-audit-reviewed
```

which runs bootstrap → Shapley/Möbius → property control → normalization →
distribution stats. Stage 1 (bootstrap) is blocking; stages 2–5 are
independent, so one failing doesn't cost you the others — failures are
collected and re-reported at the end with a non-zero exit. `--from N`
resumes at a stage; `--n-bootstraps 2000` gives a ~5× faster pass than the
10,000 default (roughly 1.5–2 h over ~14 experiments × ~18 receptors).

Both gates exist because a silent failure here is worse than a loud one: a
stale cache or an artifact-inflated `bias_only` would corrupt every
downstream number while every script kept running happily.

## Ablation operators

Every one of the 8 factorial cells over `{distogram, z_trunk, s_inputs}`
(`baseline`, the 3 `no_*`, the 3 `*_only`, `bias_only`) ablates its excluded
channel(s) under one of three operators:

| Operator | Behavior | Confound it isolates |
|---|---|---|
| `zero` | Replace with zeros (default; bare experiment names) | Ambiguous between "channel carries signal" and "network reacts badly off-distribution" |
| `resample` | Replace with the same channel from a *different* ligand on the *same* receptor, token-count matched (5 donor draws, seeds 11/22/33/44/55) | Off-distribution brittleness — donor content is something the network has plausibly seen |
| `mean` | Replace with the per-position mean over all other cached donors on the same receptor | Same as resample, deterministic single point instead of 5 draws |

`zero` is retained deliberately, not because it's flawed — the `zero` −
`resample` gap is itself an informative measure of the head's
off-distribution brittleness.

## Naming convention

- Bare cell name (`no_z_trunk`, `bias_only`, ...) = `zero` operator.
- `<cell>__resample__d<seed>` = `resample` operator, e.g.
  `no_distogram__resample__d11`.
- `<cell>__mean` = `mean` operator, e.g. `bias_only__mean`.

Legacy bare names are permanent aliases for `zero` — old output CSVs and
notebooks built against them stay valid.

## Factorial cell → experiment name

| Cell (kept channels) | Experiment name | Shapley/Möbius role |
|---|---|---|
| `{distogram, z_trunk, s_inputs}` | `baseline` | v(C) — full model |
| `{z_trunk, s_inputs}` | `no_distogram` | v(C \ {distogram}) |
| `{distogram, s_inputs}` | `no_z_trunk` | v(C \ {z_trunk}) |
| `{distogram, z_trunk}` | `no_s_inputs` | v(C \ {s_inputs}) |
| `{distogram}` | `distogram_only` | v({distogram}) |
| `{z_trunk}` | `z_trunk_only` | v({z_trunk}) |
| `{s_inputs}` | `s_inputs_only` | v({s_inputs}) |
| `{}` | `bias_only` | v(∅) |

`s_inputs` sub-components (`atom_encoder`, `msa_profile`, `res_type`) have
their own `no_*`/`only_*` zero-operator experiments, gated behind the Task 7
trigger rule (`|NAE(s_inputs)| >= 0.20` or Shapley share > 20%, `zero`
operator, cluster CI excluding zero) — not run unless that trigger fires.

## Adjusted-logAUC convention

`logauc_utils.logAUC()` subtracts `RANDOM_LOGAUC` internally, so **every**
logAUC value anywhere in this pipeline already has **random = 0** baked in.
This has been misread before — when in doubt, `bias_only` under a sound
tie-fixed bootstrap should sit near 0, not near a nonzero "random" baseline.
`compute_effect_normalization.py`'s adjusted-convention check spot-checks
this on every run.

## Output columns: primary vs. diagnostic

**Primary** (what to read for the actual finding):
- `point_estimate_logAUC`, `mean_cluster_avg_logAUC` + `cluster_ci_bca_*`
  (population-level claim) — `compute_ablation_bootstrap.py`
- `NAE` — `compute_effect_normalization.py`
- `phi` (Shapley value), `interaction` + `sign` (Möbius) — `compute_shapley_interactions.py`
- `structure_attributable_excess`, `residual_logauc` — `compute_property_control.py`

**Diagnostic** (sanity checks, not the headline number):
- `distinct_score_values`, `largest_tie_block`, `top1pct_tie_fraction`,
  `tied_top1pct_flag` — tie-density; a `True` flag means don't trust the
  logAUC next to it until re-audited
- `deprecated_naive_*` columns in `summary_avg_logAUC.csv` — kept only for
  continuity with pre-Task-3 output, not a valid population-level CI
- `denominator_near_zero_flag` — NAE is `NaN`, not divided, when this is set
- `ave_bias`, `logauc_2d` — context for `structure_attributable_excess`,
  not a headline number on their own
- `spearman_rho` (rank preservation) — receptor-normalized companion metric,
  low priority per the original spec
- `receptor_class` — best-effort DUD-E lookup (`compute_effect_normalization.py`'s
  `DUDE_RECEPTOR_CLASS`), not independently re-verified against the current
  DUD-E/DUDEZ target list; override via `--receptor-class-map`

## Reproducibility

`run_feature_ablation.py` writes a `<output>.meta.json` sidecar per run with
git SHA, `boltz` version, config hash, seeds, and run counts. Every
resample/mean row records `donor_complex_ids` (which donor ligand supplied
each substituted channel) so any ablation is exactly reproducible.

## Out of scope

Deferred by decision: distogram spatial masks / distance-cutoff sweep, pose
perturbation beyond the existing `pose_noise_*` sweep, per-residue saliency,
cross-receptor resample donors. See the code review handoff this pipeline
was built from for the full list — don't extend the corresponding scripts.
