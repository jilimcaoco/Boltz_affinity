# Boltz Affinity Fine-tuning — User Guide

This module fine-tunes the **weights of the Boltz-2 affinity module
directly**, as a counterpart to the LoRA adapter system documented in
[lora_userguide.md](lora_userguide.md). Use it when you want either:

1. **A baseline** to benchmark LoRA against on the same data, loss, and
   featurization pipeline.
2. **Higher capacity than LoRA** when you have enough labelled data that
   over-fitting is no longer the dominant risk.

The CLI ergonomics, registry layout, training CSV schema, loss registry,
and inference hooks intentionally mirror `boltz lora …` so that swapping
between the two is a single flag change.

---

## Contents

1. [When to use this vs LoRA](#when-to-use-this-vs-lora)
2. [Quickstart](#quickstart)
3. [Training](#training)
4. [Target presets](#target-presets)
5. [Continuing training](#continuing-training)
6. [Using a fine-tune at inference](#using-a-fine-tune-at-inference)
7. [Registry & storage](#registry--storage)
8. [CLI reference](#cli-reference)
9. [Programmatic API](#programmatic-api)
10. [Comparing LoRA vs full fine-tune](#comparing-lora-vs-full-fine-tune)

---

## When to use this vs LoRA

| Concern | LoRA (`boltz lora`) | Full fine-tune (`boltz finetune`) |
|---|---|---|
| Trainable params | ~1–5% of affinity module | up to **100%** of affinity module |
| Data requirement | works at ~10² labels | recommend ≥ 10³–10⁴ labels |
| Risk of forgetting | very low (base frozen) | non-trivial; use small LR + early stopping |
| Multiple targets per base ckpt | yes (one adapter per task) | no (each fine-tune is a full set of weights) |
| Storage per task | ~MBs | ~hundreds of MB |
| Default LR | `1e-4` | `1e-5` (10× lower) |

If unsure, train both with the same CSV and the same loss and compare.
That comparison is the headline use-case this module exists for.

---

## Quickstart

```bash
# 1. Author a training CSV (same schema as `boltz lora train`).
# 2. Train.
boltz finetune train \
    --name my_kinase_full_v1 \
    --csv data/kinase_train.csv \
    --loss huber --epochs 5 \
    --target-spec affinity_module \
    --learning-rate 1e-5

# 3. Use it during rescoring or prediction.
boltz rescore receptor \
    --receptor receptor.pdb --ligands ligands.mol2 \
    --use-finetune my_kinase_full_v1 \
    -o scored.csv
```

---

## Training

### CSV manifest

The training CSV is **identical** to the LoRA one — see the
[LoRA user guide](lora_userguide.md#training-a-new-adapter) for the schema.
Required columns: `ligand`, `receptor`, `target`. Optional: `structure`
(required for `mode=rescore`), `name`, `group_id`, `is_binder`,
`is_censored`. The same `parse_lora_csv` parser and `LoRADataset` are used
under the hood, so any CSV that works for `boltz lora train` works for
`boltz finetune train`.

### Training command

```bash
boltz finetune train \
    --name my_kinase_full_v1 \
    --csv data/kinase_train.csv \
    --loss huber \
    --target-spec affinity_module \
    --epochs 10 \
    --learning-rate 1e-5 \
    --weight-decay 1e-4 \
    --recycling-steps 3 \
    --device auto \
    --notes "Full-FT baseline vs LoRA on March'26 kinase data"
```

Key knobs:

- `--target-spec` — selects which sub-tree is unfrozen. Default
  `affinity_module` unfreezes everything under `affinity_module*`. See
  [Target presets](#target-presets) below.
- `--learning-rate` — default `1e-5`. Full fine-tuning is far more
  sensitive to LR than LoRA; start low.
- `--gradient-clip` — defaults to 1.0. Lower if you see loss spikes.
- `--early-stopping-patience` — defaults to 5 epochs; the trainer keeps a
  running snapshot of the best-loss weights and restores them on stop.
- `--use-msa-server` — **DISABLED in this fork**, same policy as LoRA.
  Pre-compute MSAs into `$BOLTZ_MSA_CACHE_DIR`.

Output: a new fine-tune under `$BOLTZ_FINETUNE_DIR/my_kinase_full_v1/`
containing `meta.json` (provenance), `weights.pt` (a partial state-dict
keyed by Boltz2 parameter names, holding only the trained tensors), and
`loss_curve.png`.

### Custom losses

The full set of built-in losses (`mse`, `mae`, `huber`, `bce`,
`pairwise_ranking`, `intra_assay_huber`, `boltz2_affinity`) and the
`path/to/file.py:function_name` extension hook are shared with
`boltz.lora.losses` — see the [LoRA user guide](lora_userguide.md#custom-losses)
for the contract. Custom loss functions that accept a third
`adapter_meta` argument will receive `None` from the fine-tune trainer
(the metadata is LoRA-specific).

---

## Target presets

`--target-spec` resolves to a tuple of regexes that are matched (via
`re.search`) against `model.named_parameters()`:

| Preset | What it unfreezes |
|---|---|
| `affinity_module` *(default)* | every parameter under `affinity_module*` (incl. ensemble variants `affinity_module1`/`affinity_module2`) |
| `affinity_heads` (alias `heads`) | only the heads MLPs — direct mirror of the LoRA `heads` preset |
| `affinity_pairformer` | only the affinity-side pairformer linears |
| `heads_pairformer` | heads + pairformer (mirror of LoRA `heads_pairformer`) |

You can also pass an explicit regex (`r"^affinity_module\.affinity_heads\..*\.weight$"`)
or a comma-separated list of regexes for full control.

The two narrowest presets (`heads`, `heads_pairformer`) are the
**apples-to-apples baseline** for the corresponding LoRA presets — same
parameter family, but trained as full updates instead of low-rank
residuals.

---

## Continuing training

`boltz finetune update` resumes from an existing fine-tune's weights,
appends a new `TrainingRun` entry to its history, and (optionally) saves
under a child name so the parent is preserved. Same UX as
`boltz lora update`:

```bash
# Round 1: initial fine-tune
boltz finetune train --name pose_full_v1 --csv round1.csv --epochs 5

# Round 2: resume on additional data, save under a new name
boltz finetune update --name pose_full_v1 --new-name pose_full_v2 \
    --csv round2.csv --epochs 5
```

Note that `--target-spec` and any other parameter-selection knobs are
**inherited from the parent** during `update` so the saved tensors stay
shape-compatible with the existing entry.

---

## Using a fine-tune at inference

### Rescoring

All four `boltz rescore` subcommands accept `--use-finetune`:

```bash
boltz rescore pdb --input complex.pdb --use-finetune my_kinase_full_v1
boltz rescore receptor --receptor r.pdb --ligands l.mol2 --use-finetune my_kinase_full_v1
boltz rescore batch --input-dir poses/ --use-finetune my_kinase_full_v1
boltz rescore manifest --manifest m.yaml --use-finetune my_kinase_full_v1
```

`--use-finetune` and `--use-lora` are **mutually exclusive** — adapter
composition is not supported.

You can also set `BOLTZ_RESCORE_FINETUNE=<name>` to apply a fine-tune
implicitly (analogous to `BOLTZ_RESCORE_LORA`).

### Prediction

`boltz predict --use_finetune <name>` applies the fine-tune to the
affinity model after both the structure-prediction and affinity
checkpoints are loaded.

---

## Registry & storage

```
$BOLTZ_FINETUNE_DIR (default ~/.boltz/finetunes)/
  registry.json
  <name>/
    meta.json         # FinetuneRecord (provenance, history)
    weights.pt        # {"state_dict": <partial>, "config": {...}}
    loss_curve.png    # optional plot
```

`weights.pt` stores **only the parameters that were trained**, keyed by
their fully-qualified names in the Boltz2 model. At apply time we walk
the saved keys and `copy_` each tensor into the live model's
`state_dict()`. This keeps fine-tunes small (`affinity_heads`-only is a
few MB) and preserves the rest of the base checkpoint intact.

---

## CLI reference

```
boltz finetune train     # train a fresh fine-tune
boltz finetune update    # continue training an existing fine-tune
boltz finetune list      # list registered fine-tunes
boltz finetune show NAME # pretty-print metadata
boltz finetune apply NAME            # dry-run load (compatibility check)
boltz finetune export NAME --out X.ckpt  # export full merged checkpoint
boltz finetune rm NAME   # remove
```

Run `boltz finetune <cmd> --help` for full option listings.

---

## Programmatic API

```python
from boltz.finetune import (
    FinetuneRegistry, default_registry, load_finetune_into_model,
)
from boltz.finetune.train import FinetuneArgs, train_finetune
from boltz.affinity_rescoring.model_manager import AffinityModelManager

# Train
record = train_finetune(FinetuneArgs(
    name="my_v1",
    csv_path="data/train.csv",
    target_spec="affinity_module",
    learning_rate=1e-5,
    epochs=5,
))

# Apply at inference
mgr = AffinityModelManager()
model = mgr.load_model()
load_finetune_into_model(model, "my_v1")
```

---

## Comparing LoRA vs full fine-tune

The recommended head-to-head protocol:

1. Pick a single training CSV and split it into train/validation.
2. Train a LoRA adapter:
   ```bash
   boltz lora train --name X_lora --csv train.csv --target-spec heads --loss huber
   ```
3. Train the matching full fine-tune on the same parameter family:
   ```bash
   boltz finetune train --name X_full --csv train.csv --target-spec heads --loss huber
   ```
4. Score the same validation set twice:
   ```bash
   boltz rescore receptor ... --use-lora    X_lora -o val_lora.csv
   boltz rescore receptor ... --use-finetune X_full -o val_full.csv
   ```
5. Compare R² / Spearman / EF1% on the held-out set. Same featurization,
   same forward, same loss → any difference is attributable to the
   parameterisation choice.

The two systems share the differentiable forward
(`boltz.lora.train._affinity_forward_trainable`), the featurizer
(`boltz.lora.train._featurize_row`), the dataset/sampler
(`boltz.lora.data`), and the loss registry (`boltz.lora.losses`), so the
comparison is apples-to-apples by construction.
