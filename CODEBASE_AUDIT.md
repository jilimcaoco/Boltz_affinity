# Codebase audit — Boltz_affinity (branch `TRIAGE-LORA`, 2026-10-29)

Scope: everything this fork adds on top of upstream Boltz — `src/boltz/lora`,
`src/boltz/finetune`, `src/boltz/affinity_rescoring`, the experiment folders
(`ablation_experiment/`, `fineturning_experiment/`), `slurm_scripts/`, `scripts/`,
docs and repo hygiene. Upstream model/data code was only examined where the fork
calls into it.

Labels: **[C]** confirmed by reading the code (items marked ✔ were independently
re-checked); **[S]** suspected — the mechanism is real but the impact depends on
data/config. Line numbers are against the working tree at the time of the audit.

---

## Status — fix round 1

| Item | Status |
|---|---|
| §0 #1 / §1.1 `is_censored` dropped by `lora_collate` | **Fixed.** Collated and kept aligned on row drop in both trainers; warning when a `censored_*` loss sees no censored rows |
| §0 #8 / §3.10 censored manifests without censored rows | **Fixed (pipeline).** `KEEP_CENSORED=1` in `config.env` feeds `--keep-censored` to 03 and 04a/04b. You need to rerun 03 → 04a/04b to regenerate poses/manifests |
| §0 #5 / §1.5 `apply`/`export` import error | **Fixed.** The import now points at `affinity_rescoring.inference`, and CLI tests invoke the commands |
| §0 #3 / §2.1 order-based residue injection | **Fixed.** Positional name check, then realignment on the PDB's own `(resnum, icode)`, then a CA–CA continuity check. Unresolvable chains raise `ResidueMappingError`, which skips the row in training and fails the pose with its reason in rescoring. Training chain mapping is now by name |
| §2.14 insertion codes dropped | **Fixed** for injection (`AtomInfo.insertion_code`); altlocs still last-wins |
| §0 #2 / §1.2 train vs score head mismatch | **Opt-in fix.** `--affinity-output ensemble` trains on the MW-corrected ensemble value scoring reports; the default `module1` is unchanged |

Default (no new flag) training stays bit-identical to before this round, verified against the pre-patch baseline.

## Status — fix round 2 (trainer v2; see `docs/MIGRATION_TRAINER_V2.md`)

| Item | Status |
|---|---|
| Affinity pairformer stack never received gradient (reentrant activation checkpointing on a frozen input) — every pre-v2 adapter/fine-tune was heads-only (verified: all 150 LSD adapters, all 8 LSD fine-tunes, this repo's 4 + 1) | **Fixed.** Non-reentrant checkpointing (`pairformer.py`); `--legacy-trainer` reproduces the old behaviour bit-identically (verified vs `1b7e973`, incl. RNG) |
| §0 #2 / §1.2 train vs score output | **Fixed by default.** `--affinity-output` now defaults to `ensemble` |
| LoRA `heads_pairformer` dead patterns + `confidence_module` leak; LoRA/finetune presets of the same name meant different scopes | **Fixed.** Anchored scope presets shared by both tools (`affinity_heads`, `affinity_pairformer`, `affinity_heads_pairformer`, `affinity_module`), restricted to the modules the forward uses; old names deprecated with a warning |
| No guard against silently frozen parameters | **Added.** Gradient-coverage check after epoch 1 (fails when a targeted scope in the forward gets no gradient), recorded as `grad_coverage` |
| v2 adapters unloadable (empty metadata) | **Fixed.** Loader falls back to the layers named in `adapter.pt`; recovery scripts now write layer lists and count layers, not tensors |
| No code provenance for an unpinned editable install | **Fixed.** `meta.json` records `trainer_semantics`, `target_modules`, `boltz_git_sha`, `boltz_git_dirty`, `torch_version` |
| `boltz finetune` had no validation (LoRA vs FT early-stopped on different criteria) | **Fixed.** `--val-csv` / `--val-split`; shared validation and batch-forward helpers replace three copies of the loop |
| §3 evaluation: SMILES auto-inference failures; silent row loss in the cross-model join | **Fixed.** Rescoring passes label SMILES; evaluator reports per-model `n_success` |
| §2.23 / config tests | **Fixed.** `recycling_steps` fallback, `DEFAULT_CONFIG` mutation, stale test asserts |
| `tests/test_regression.py` imports `boltz.main.MODEL_URL`, which no longer exists | Open |

---

## 0. Fix-first list

| # | Area | Problem | Why it matters |
|---|------|---------|----------------|
| 1 | lora | `lora_collate` drops `is_censored` ✔ | `censored_intra_assay_huber` / `censored_boltz2_affinity` silently behave exactly like the uncensored losses |
| 2 | lora ↔ rescoring | Training optimises `affinity_module1` raw output; inference uses the module1/module2 ensemble mean **plus** MW correction | The adapter is trained on a different quantity from the one it is scored on; LoRA layers in `affinity_module2` never get gradient |
| 3 | rescoring | Protein coord injection pairs residues by order, not residue number | Any chain with gaps / SEQRES replacement / CA-less residues gets coordinates shifted onto the wrong residues, silently |
| 4 | rescoring | `rescore_pdb` never renames ligand atoms to Boltz's canonical names | Non-MOL2 inputs: ligand atoms unmatched → placed at origin, still reported SUCCESS |
| 5 | lora/finetune CLI | `apply` / `export` import `boltz.affinity_rescoring.model_manager`, which does not exist ✔ | `boltz lora apply|export`, `boltz finetune apply|export` crash with `ModuleNotFoundError` |
| 6 | huber_delta_scan | Tag mismatch (`1p0` vs `1p00`) ✔, metric-name mismatch, vanilla rows not filtered | The delta-scan table/plots drop 3/5 deltas, only 2 metrics plot, and "best delta" can land on a vanilla point |
| 7 | repo | ~2.3 GB of `.pt` weights committed and pushed (`fineturning_experiment/{adapters,finetunes}`) ✔ | Clone size; requires history rewrite or LFS to undo |
| 8 | experiments | "censored" LoRA config never passes `--keep-censored` | Combined with #1, the `*_censored_data` adapters contain no censored data *and* the loss ignores censoring |
| 9 | lora | `pairwise_ranking` returns a no-grad zero for 1-row batches ✔ | `loss.backward()` raises at the default `--batch-size 1` |
| 10 | rescoring | `validation.py:205` compares `element.upper()` against a mixed-case set (`"Cl"`) ✔ | Every Cl/Br/Fe/Zn/Mg/Na/Se atom flagged invalid; STRICT mode fails halogenated ligands |

---

## 1. `src/boltz/lora` and `src/boltz/finetune`

### High
1. **[C✔] `is_censored` never reaches the loss.** `LoRARow.to_batch_item` emits it (data.py:82) but `lora_collate` (data.py:202-212) doesn't copy it, so `_get_censored_mask` always sees `None`. *Not fixed in this patch* — fixing it changes the training of every existing censored-loss arm; needs its own decision.
2. **[C] Train/inference head mismatch.** `_affinity_forward_trainable` uses only `affinity_module1` when `affinity_ensemble` is set and returns raw output (train.py `_affinity_forward_trainable`), while inference averages module1+module2 and applies `1.035*avg − 0.6*mw^0.3 + 2.83` (affinity_rescoring/inference.py:610-652; `AffinityModelManager` defaults `affinity_mw_correction=True`). Module2's LoRA B stays at zero; the finetune `affinity_module` preset counts module2 params that never move.
3. **[C✔] `pairwise_ranking` crashes on single-row batches** (losses.py `pairwise_ranking_loss`: `torch.zeros(())` has no grad).
4. **[C] Ungrouped `pairwise_ranking` validation is always 0.** With `batch_size>1` and no `group_id`, train uses random multi-row batches but val falls back to `batch_size=1` → every val batch is 0 → early stopping locks onto epoch 1.
5. **[C✔] `apply`/`export` import a non-existent module** (lora/cli.py:325,348; finetune/cli.py:335,359).
6. **[C✔] `bce` loss uses the regression target as the BCE label** (losses.py `bce_loss`); `_binder_labels` exists but isn't used here. The unit test passes only because it uses 0/1 targets.

### Medium
7. **[C] All-failed epoch reports loss 0.0** (`sum([])/max(1,0)`), which early stopping treats as a new best.
8. **[C] Early-stopping monitor can silently switch from val to train loss** if every val row fails in one epoch.
9. **[C] `except Exception` around featurisation** swallows CUDA OOM / device errors / programming bugs as per-row warnings; no skip counter or failure threshold.
10. **[C] `--val-split` takes the last N CSV rows.** The comment calls this "stratified", but it isn't stratified or shuffled; with an assay-sorted CSV an assay straddles train/val.
11. **[C] Receptor YAML that already contains a ligand overrides the CSV SMILES** (`_materialize_row_yaml`) — every row featurises the same ligand while the cache key still varies.
12. **[C] `data.py` docstring** says ligand may be MOL2/SDF and receptor PDB/CIF; the trainer only accepts SMILES + YAML.
13. **[C] Resume never checks the checkpoint matches the current args** (rank, targets, CSV sha, loss). Re-running a crashed name with new hyper-params silently continues the old run.
14. **[C] No seed control.** LoRA A init, `DataLoader(shuffle=True)` and the hard-coded `AssayGroupedSampler` seed are not exposed; resume doesn't restore shuffle state.
15. **[C] Cache on/off changes semantics.** With `BOLTZ_TRAIN_CACHE_DIR` set, the frozen trunk runs in eval mode (no dropout); without it, trunk dropout is active.
16. **[C] `--target-spec` help** advertises an `affinity_heads` preset and comma lists that `lora/targets.resolve_targets` doesn't implement (finetune's version does).
17. **[C] Train-cache eviction is FIFO, not LRU** — `load_*` never touches mtime.
18. **[C] Val-loss aggregation** is a mean of batch means (unequal grouped batches weighted equally) and depends on `--batch-size`, so `val_loss` isn't comparable across batch sizes.
19. **[S] Unanchored target regexes** (`pairformer_stack\..*`) also match `confidence_module.pairformer_stack` if loaded.
20. **[S] z-cache key** falls back to `"nockpt"`, ignores MSA file contents, and uses whole-second mtime.
21. **[C] Training featurises without `fork_rng` + `_featurizer_seed()`** that rescoring uses, so `ref_pos` augmentation differs between train and eval.

### Low
22. `boltz lora update` silently ignores `--rank/--alpha/--dropout/--target-spec` (inherits parent); `--val-csv`/`--val-split` "mutually exclusive" is not enforced.
23. `boltz lora apply` docstring says it uses the adapter's base checkpoint by default — it passes `None`; no `base_checkpoint_sha256` check.
24. `inject.load_lora_state_dict`: `missing` list built and never used; strict mode misses layers that received no tensors.
25. Registry writes (`adapter.pt`, `meta.json`, `registry.json`) are non-atomic; concurrent SLURM arms can lose index updates.
26. Training `recycling_steps=3` vs inference default 5 **[S]**.
27. Per-epoch checkpoints (incl. optimiser state + `es_best_state`) are never cleaned up after a successful save.
28. `selection.py`: hybrid MaxMin is O(N²) over the whole CSV; `|affinity_pred|` fallback isn't an uncertainty; `idxmin` label used as a position.

---

## 2. `src/boltz/affinity_rescoring`

### High
1. **[C] Residue-order coordinate injection** (coord_injection.py:73-115, parsers.py:439-449, rescorer.py:325-328) — shifts coordinates after any gap, SEQRES substitution, CA-less or unmapped non-standard residue, or in-chain HETATM. Backbone names still match, so it is silent.
2. **[C] `rescore_pdb` skips ligand atom renaming** (rescorer.py:381-391 vs 1173) — unmatched atoms zeroed, result still SUCCESS; `pocket_proximity_report` not surfaced in `AffinityResult`.
3. **[C] `affinity_std` is wrong with MW correction on** (inference.py:638-652; rescorer.py:417-422): it mixes the corrected mean with raw head values, so it mostly measures the correction offset. rescorer.py:1223 uses a different formula.
4. **[C/S] MSA cache accepts files without checking the sequence** (msa_cache.py:155-165, 248-262): legacy `A.a3m` / single-file fallback can hand one receptor another protein's MSA in a shared cache.

### Medium
5. **[C✔] Two-letter elements flagged invalid** (validation.py:205).
6. **[C] Clash check allocates N×N×3 float64** (~2.4 GB at 10k atoms) per pose; unseeded `np.random` (validation.py:221-234).
7. **[C] `rescore manifest` maps SMILES to chain `"L"`**, which never matches → silent fallback to auto-inference (cli.py:709); `fmt` unused.
8. **[C] `_run_direct_prediction` swallows all exceptions** → "Affinity prediction failed. Check logs." with the cause lost.
9. **[C] `RescoreConfig.checkpoint` ignored** when a config is passed (rescorer.py:179-183).
10. **[C] `DEFAULT_CONFIG` mutated in place** via shallow copy (config.py:78-96,209); CLI never uses `load_config`.
11. **[C] Multipocket sorts worst-first by default** (log10 IC50, lower = better) and ranks NaN first; `interface_iptm` always NaN (looked up by chain letter, written by index); calls ColabFold when `--msa` omitted, contradicting the MSA-server policy.
12. **[C] Chain classification is whole-chain only** — HETATM ligands inside protein chains are missed; buffer molecules (GOL/EDO/PEG) become ligand chains and may be chosen as binder.
13. **[C] SMILES auto-inference** can return a connectivity-only (all single bonds, charge 0) SMILES as success.
14. **[C] Insertion codes dropped and all altlocs kept** (parsers.py:102-117).
15. **[C] SQLite export** fails on duplicate file stems (`--recursive`) and leaks the connection.
16. **[C] MSA CSV content check reads the `key` column**, so CSV matching never succeeds.
17. **[S] Rescore trunk-cache handle memoised on `id(model)`**; FIFO eviction; full directory scan per save.

### Low
18. `inference.py:88` `total_mem` → `total_memory` (exception swallowed).
19. `export.py`: `--sort-by ligand_name` raises inside `math.isnan` (silently unsorted); HTML unescaped; `_safe_float(None)` raises.
20. Unused/never-filled fields: `validation_level` param, `featurization_time_ms`, `model_checkpoint_sha256` (but the whole multi-GB checkpoint is still hashed); `RescoreConfig.max_tokens/max_atoms/...` unused (256/2048 hard-coded); `ULTRA_FAST_RECYCLING_STEPS` is accidentally a pydantic field.
21. `_mol2_to_smiles` sets no formal charges and returns unsanitised SMILES on failure.
22. `boltz rescore` registration in `main.py:817-822` is wrapped in `except ImportError: pass`, hiding real import errors.
23. **Pre-existing test failures:** `test_affinity_rescoring.py::TestConfiguration::test_load_default_config` and `test_affinity_integration.py::TestConfigIntegration::test_full_config_round_trip` expect `recycling_steps == 5`; the config default is 3.

---

## 3. Experiments, SLURM, scripts

### High
1. **[C✔] Delta-scan tag mismatch.** `analyse_delta_scan.py:49-50` builds `1p0`, `0p5`, `0p0`; `04_evaluate.slurm:50` writes `1p00`, `0p50`, `0p00`. Only `0p75`/`0p25` load.
2. **[C] Delta-scan metric names** (`spearman_rho`, `ef_0.01`, `ap`, `rmse`, `*_ci_lo/_ci_hi`) don't match `evaluate_adapters.py` output (`spearman`, `ef_1pct`, `pr_auc`, `rmse_pIC50`, `*_lo/_hi`).
3. **[C] "Skip vanilla rows" filter is a no-op** (`analyse_delta_scan.py:254-256`): vanilla rows also carry `delta`.
4. **[C] Tracked SLURM scripts pass `--use-msa-server`**, which the fork's rescore CLI rejects: `boltz_screen_gpu`, `boltz_screen_peroxide`, `run_subposes`, `run_affinity_prediction_peroxide`.
5. **[C] Ablation scripts bypass the MSA guard** and call ColabFold directly (`run_feature_ablation.py`, `run_distogram_ablation.py`; via `feature_ablation.slurm`, `residue_loo.slurm`, `distogram_ablation.slurm`).
6. **[C] `config.env:88-89` lists adapters that don't exist** (`*_v3_rank_32_updated_loss`); `analysis/03_rescore_vanilla_lora.slurm` runs under `set -e`, so it aborts before later targets. The adapters 05a/05b actually train are not in the list.
7. **[C] `5ht2a_full_v1` is registered but has no `meta.json`/`weights.pt`**, so `--use-finetune 5ht2a_full_v1` fails.
8. **[C] SLURM headers reference untracked helpers**: `submit_*_ablation_dudez.sh` (ignored by `ablation_experiment/.gitignore`), `submit_boltz.py` (root `.gitignore`).
9. **[C] `distogram_ablation.slurm`** uses the legacy `experiment_files/...` layout.

### Medium
10. **[C] "censored" adapters trained without censored rows** (no `--keep-censored` in 04a/04b) — see also §1.1.
11. **[C] Delta scan is not comparable to the main experiment**: it uses `boltz2_affinity` while the main runs use `censored_boltz2_affinity`, and its hyper-params are copied rather than sourced from the parent config.
12. **[C] `huber_delta_scan/04_evaluate.slurm`** leaks temp dirs (`continue 2` skips cleanup; `trap` overwritten per iteration).
13. **[C] `adapters/{drd4,5ht2a}_v2_rank_64` are actually rank 32** (meta.json/registry.json).
14. **[C] Default output path** in `run_feature_ablation_dudez.py:104` differs from what the DUDE-Z SLURM scripts use.
15. **[C] Hard-coded `/home/limcaoco`, `/scratch/...`, `/nfs/turbo/...` paths** throughout configs, SLURM, analysis scripts and both `registry.json` files.

### Low
16. CPU-only jobs (01, 02, 04a/04b at 98 GB/24 h for CSV building, analysis 01/04, delta-scan 04) request `maom-h200`.
17. `--time=00:60:00` (non-canonical) in four screen scripts **[S]**.
18. `evaluate_adapters.py`: ties broken by input order; `semilog_auc` doesn't interpolate at λ; no paired bootstrap for vanilla-vs-adapter; `np.trapz` deprecated; `pred_pIC50_*` columns actually hold log10(IC50 µM).
19. `compute_logauc_bootstrap.py` `is_decoy` lacks `str()` and fails on NaN or numeric IDs.

---

## 4. Redundancies

| Duplicated thing | Locations | Suggested consolidation |
|---|---|---|
| Single/multi-row featurise→forward→subset block | `lora/train.py` (train loop), `lora/train.py` (new val loop), `finetune/train.py:322-401` | One `_forward_batch(model, batch, …) -> (pred, batch_for_loss) | None` helper (~300 lines removed). Finetune also lacks validation entirely — already diverged |
| Early stopping, resume, loader construction | `lora/train.py` vs `finetune/train.py` | Shared epoch driver |
| Registry | `lora/registry.py:50-178` vs `finetune/registry.py:34-170` | Generic base class parameterised by tensor filename |
| Adapter (de)serialisation | `lora/adapter.py` vs `finetune/adapter.py` | Shared mixin |
| `resolve_targets` | `lora/targets.py` vs `finetune/targets.py` (already drifted: comma split only in finetune) | One implementation |
| CLI scaffolding | `_setup_logging`, ~80% of `_train_options`, `list/show/rm/apply/export`, MSA-server guard ×4 | Shared option factory + command helpers |
| Featurisation pipeline | `lora/train._featurize_row` vs `affinity_rescoring/inference.run_direct_affinity_inference` (already diverged on RNG) | Single featurise function used by both |
| Affinity forward | `inference.py` (trunk+head), `Boltz2.forward`, `lora/train.py`, `model/modules/affinity.py`; MW coefficients hard-coded twice | One forward + one MW-correction function |
| z/trunk cache | `affinity_rescoring/trunk_cache.py` vs `lora/train.TrunkFeatureCache` (+ same FIFO eviction bug) | One cache class |
| `run_mmseqs2` | `affinity_rescoring/mmseqs2.py` is a verbatim copy of `data/msa/mmseqs2.py` | Import upstream; keep only `precompute_msa` |
| RDKit builders, YAML builders (×3 each), residue / water / ion tables | rescorer, smiles_inference, multipocket, parsers, validation, mol2_parser | One module each; the string-concatenated YAML writer breaks on SMILES containing `'` |
| Small helpers in `lora/train.py` | `_save_checkpoint`≈`_atomic_torch_save`; `_unwrap`≈`_unwrap_module`; `x[0] if isinstance(x, list) else x` ×8 | Collapse |
| Analysis helpers | `clean_ligand_name` ×7, `is_decoy` ×6, ROC/logAUC ×2, `ridgeline_plot` ×3, `bootstrap_ci` ×2 | Import from `logauc_utils.py` |
| Ablation runners | `run_feature_ablation.py` / `run_distogram_ablation.py` / `run_residue_loo.py` share `discover_pairs`, `_remap_ligand_atoms`, … by copy | Shared module |
| SLURM scripts | `{feature,mean,replacement}_ablation_dudez.slurm`; per-target 04a/04b, 05a/05b, 05c/05d, delta-scan 01/02 | One script with a `MODE`/`TARGET` variable or array |
| Delta list | Defined 6× (`config.env` + each of 01-04 + `analyse_delta_scan.py`); exported bash arrays don't reach child processes anyway | Single source |
| `config.env` | `*_FINETUNE_NAME` defined twice (lines 5-6 and 128-129) | Delete one |
| One-off recovery scripts | `recover_adapter.py` / `recover_5ht2a_adapter.py` / `register_v2_adapters.py` | Archive or parameterise |

**Dead code / unused:** `hash_file` import, `import copy as _copy`, the no-op `train_dataset = train_dataset`, `_resolve_yaml(work_dir)`, the `use_msa_server` parameter, the `mode="full"` branches in both trainers, `StructureNormalizer`, `sorted_resolved`, ~25 unused imports in `rescorer.py` (and missing imports for `LigandStructure`/`AtomInfo` annotations), and the orphaned `boltz_lora/` directory (its compat shim imports a non-existent function; `MIGRATION.md` links to files that don't exist).

---

## 5. Repo hygiene

1. **~2.3 GB of weights committed and pushed** (`fineturning_experiment/finetunes` 1.6 GB, `adapters` 743 MB; commits `f9c0876`, `3329168`). Needs `git filter-repo`/BFG or LFS; add `*.pt`, `adapters/`, `finetunes/` to `.gitignore`.
2. `.gitignore` has `*.png`/`*.csv`, yet many PNG/CSV files are tracked; new diagnostics will be silently ignored.
3. Docs contradict the MSA policy (`docs/affinity_rescoring_userguide.md` documents `--use-msa-server` as working in ~10 places; README still explains server credentials).
4. Missing dependencies in `pyproject.toml`: `matplotlib` (used by `lora/train.py`), `chembl_webresource_client`, `seaborn`, `statsmodels`, `tqdm`. No `[analysis]` extra.
5. Stray files: `slurm_scripts/feature_ablation_dudez.slurm~`, root-level `plot_score_comparison.py` and `Affinity_module_rescoring.txt`, emacs autosave `#submit_feature_ablation_dudez.sh#`, nested `logs/` dirs created by relative `--output` paths.
6. Misspelled `fineturning_experiment/` is baked into `.gitignore`, `config.env` and absolute registry paths — renaming needs coordinated edits.
7. No `testpaths` in `pyproject.toml`, so `pytest tests/` skips `ablation_experiment/tests` (68 pass there).
8. Local environment note: on CPU login nodes `tests/test_kernels.py`, `tests/test_regression.py` and `tests/model/layers/*` fail at collection (no NVIDIA driver; `matplotlib` needs a newer `libstdc++` than `/lib64` provides).

---

## 6. Test gaps

- No collate/sampler tests (would have caught §1.1); `AssayGroupedSampler` untested.
- Loss tests only check "finite scalar" on 3 ungrouped rows with 0/1 targets — no pair-mask, censoring, `_binder_labels` or n=1 backward tests.
- No CLI tests that actually invoke commands (an import smoke test would catch §1.5).
- No tests for `coord_injection` (the highest-risk rescoring path), `msa_cache.find_msa`, multipocket, end-to-end `AffinityRescorer` with a mocked model, or numerical parity of `affinity_forward` vs `Boltz2.forward`.
- No resume/overwrite/`val_split` tests for `train_lora` and nothing for `train_finetune`. (`tests/test_lora_huber_delta.py`, added with the Huber patch, now provides a stubbed end-to-end `train_lora` harness that these can reuse.)
