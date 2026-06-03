# DRD4 + 5HT2A LoRA fine-tuning experiment

End-to-end pipeline for fine-tuning the Boltz-2 affinity head with LoRA on
held-out human GPCR data (dopamine D4 + serotonin 5-HT2A) curated from
ChEMBL.  **No external docking step required** — Boltz itself generates the
complex structures used as training inputs.

In addition to the LoRA adapters, the pipeline trains a **full fine-tune
control** per target (`05c_finetune_drd4.slurm`, `05d_finetune_5ht2a.slurm`)
on the same manifests with the same loss. This gives a three-way comparison
at evaluation time:

| Model | Parameter footprint | Where artefacts land |
|---|---|---|
| `vanilla` | 0 (frozen Boltz-2)            | — |
| `lora`    | ~rank · 2 · d (small adapter) | `$BOLTZ_LORA_DIR` (= `adapters/`) |
| `finetune`| every weight under `--target-spec` | `$BOLTZ_FINETUNE_DIR` (= `finetunes/`) |

The full-FT job acts as a strict capacity ceiling for the LoRA adapter:
same data, same loss, same target sub-tree — only the parameterisation
differs.

## Files

| File | Purpose |
|---|---|
| `config.env` | Single place for all paths and hyperparameters. **Edit this first.** |
| `01_pull_chembl.slurm` | Curate ChEMBL CSV via `pull_chembl_affinity_data.py`. |
| `prepare_predict_inputs.py` | Curated CSV → per-ligand Boltz input YAMLs. |
| `collect_predicted_poses.py` | Flatten `boltz predict` output → `poses/{target}/{molecule}.pdb`. |
| `02_predict_poses.slurm` | Build inputs + run `boltz predict` for both targets + collect. **H200 GPU.** |
| `build_lora_manifest.py` | Curated CSV + poses dirs → LoRA manifest CSV. |
| `03_prepare_manifest.slurm` | SLURM wrapper around `build_lora_manifest.py`. |
| `04_train_lora.slurm` | Train the LoRA adapter on H200 with intra-assay Huber loss. |
| `run_pipeline.sh` | Submit all four steps as a dependency chain. |

All SLURM jobs use `--account=maom --partition=maom-h200`.

## Pipeline overview

```
┌────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│ 01 pull        │ → │ 02 predict     │ → │ 03 build       │ → │ 04 train       │
│    ChEMBL      │   │    poses       │   │    manifest    │   │    LoRA        │
│ (CPU)          │   │ (H200 GPU)     │   │ (CPU)          │   │ (H200 GPU)     │
│ curated.csv    │   │ poses/*/*.pdb  │   │ manifest.csv   │   │ adapter.pt     │
└────────────────┘   └────────────────┘   └────────────────┘   └────────────────┘
```

Step 02 replaces what used to be a manual DOCK3.8 / DiffDock job: for every
unique `(target, ligand)` pair in the curated CSV it
1. builds a Boltz input YAML by appending the SMILES to the per-target
   receptor YAML (`prepare_predict_inputs.py`),
2. runs `boltz predict <input_dir>` once per target,
3. symlinks the top-rank model into a flat `poses/{target}/{molecule}.pdb`
   so step 03 finds them with the default pose pattern.

## Setup

1. Edit `config.env`:
   - Confirm `CONDA_ENV` matches the env that has `boltz`,
     `chembl_webresource_client`, `rdkit`, `pandas`, `pyyaml`.
   - Point `DRD4_RECEPTOR_YAML` / `HT2A_RECEPTOR_YAML` at your prepared
     Boltz YAMLs.  These must contain the protein chain(s); any `ligand`
     entries are ignored.  See `examples/affinity.yaml` for the schema.
     If you do not have local MSAs, run `02_precompute_msas.slurm` first;
     `--use-msa-server` is disabled in this fork.
   - Optionally tune `PREDICT_RECYCLING_STEPS`, `PREDICT_SAMPLING_STEPS`,
     `PREDICT_DIFFUSION_SAMPLES` (cost / quality trade-off).
2. `logs/` is created automatically by `run_pipeline.sh`.

## Submission

```bash
cd fineturning_experiment
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Submits all four jobs as a `--dependency=afterok` chain.  Skip earlier
steps once their outputs already exist:

```bash
./run_pipeline.sh --from 02      # skip ChEMBL pull
./run_pipeline.sh --from 03      # skip pull + predict
./run_pipeline.sh --from 04      # only retrain
```

Or submit any single step manually:

```bash
sbatch 01_pull_chembl.slurm
sbatch 02_predict_poses.slurm
sbatch 03_prepare_manifest.slurm
sbatch 04_train_lora.slurm
```

## What the trainer sees

After step 03, `lora_manifest.csv` has the columns the LoRA trainer expects:

| Column | Source | Notes |
|---|---|---|
| `name` | `{src}_{molecule_chembl_id}_{assay_chembl_id}` | unique row id |
| `ligand` | `canonical_smiles_std` | RDKit-canonicalized SMILES |
| `receptor` | per-target Boltz YAML | from `config.env` |
| `target` | `log10_aff_uM` = `log10(IC50_µM)` | **Boltz-2 affinity-head scale, lower=stronger** |
| `structure` | `{poses_dir}/{molecule_chembl_id}.pdb` | Boltz-predicted complex (step 02) |
| `group_id` | `assay_chembl_id` | unit of the intra-assay pairwise Huber loss |

Censored (`is_censored=True`) rows are dropped during manifest build since
none of the built-in losses handle right-censoring correctly.

## Loss

Step 04 uses the built-in `intra_assay_huber` loss (Huber on absolute
`log10(IC50_µM)` + `2× Huber` on pairwise intra-assay differences).  This
matches the Boltz-2 training recipe and lets us mix Ki / Kd / IC50 / EC50 /
AC50 / XC50 readouts because the pairwise term cancels the Cheng-Prusoff
offset.

`04_train_lora.slurm` sets `--batch-size 8`, and the trainer auto-engages
`AssayGroupedSampler` whenever `batch_size > 1` and the manifest has
`group_id` populated — so every step gets up to 8 same-assay compounds and
the pairwise Huber term actually fires.

## Outputs

After the chain finishes:

```
fineturning_experiment/
├── data/chembl_D4_5HT2A_curated.csv
├── predict_inputs/{DRD4,5HT2A}/<molecule>.yaml
├── predict_outputs/{DRD4,5HT2A}/boltz_results_*/...
├── poses/{DRD4,5HT2A}/<molecule>.pdb           # symlinks into predict_outputs
├── manifests/lora_manifest.csv
├── adapters/                                    # registry; honoured via BOLTZ_LORA_DIR
│   └── drd4_5ht2a_v1/
│       ├── meta.json
│       └── adapter.pt
└── logs/*.log
```

Use the adapter at inference with:

```bash
boltz rescore receptor \
    --receptor DRD4_for_eval.yaml \
    --ligands heldout_drd4_ligands.mol2 \
    --use-lora drd4_5ht2a_v1 \
    -o drd4_heldout_scored.csv
```
