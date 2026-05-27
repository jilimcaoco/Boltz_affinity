# Boltz LoRA Adapters — User Guide

This module adds parameter-efficient finetuning to the Boltz2 **affinity**
stack via Low-Rank Adaptation (LoRA). You can train small adapters on your
own labelled data, register them by name, and load them on demand from
either `boltz predict` or any `boltz rescore` subcommand without ever
modifying the base checkpoint on disk.

---

## Contents

1. [Concepts](#concepts)
2. [Quickstart](#quickstart)
3. [Training a new adapter](#training-a-new-adapter)
4. [Custom losses](#custom-losses)
5. [Continuing training (active-learning loop)](#continuing-training-active-learning-loop)
6. [Using an adapter at inference](#using-an-adapter-at-inference)
7. [Registry, storage & provenance](#registry-storage--provenance)
8. [CLI reference](#cli-reference)
9. [Programmatic API](#programmatic-api)
10. [Limitations](#limitations)

---

## Concepts

A LoRA adapter replaces selected `nn.Linear` layers in the Boltz2 affinity
model with low-rank residual updates of the form

$$y = W_0 x + \frac{\alpha}{r}\, B A\, x,$$

where $A \in \mathbb{R}^{r \times d_\text{in}}$ and $B \in \mathbb{R}^{d_\text{out} \times r}$
are the only trainable parameters. $B$ is zero at init, so an untrained
adapter is the identity. Typical adapter sizes are 1–5% of the affinity
model's params.

**Targets.** Two presets are shipped:

| Preset | Layers wrapped |
|---|---|
| `heads` | The three MLPs inside `AffinityHeadsTransformer` (smallest, lowest risk). |
| `heads_pairformer` *(default)* | `heads` + attention/transition linears in the affinity-side pairformer stack. |

You can also pass an explicit regex (matched with `re.search` against
`model.named_modules()` names) or a list of regexes for full control.

---

## Quickstart

```bash
# 1. Author a training CSV (see "Training a new adapter" for the schema).
# 2. Train.
boltz lora train \
    --name my_kinase_v1 \
    --csv data/kinase_train.csv \
    --loss huber --rank 8 --epochs 5

# 3. Use it during rescoring or prediction.
boltz rescore receptor \
    --receptor receptor.pdb --ligands ligands.mol2 \
    --use-lora my_kinase_v1 \
    -o scored.csv
```

---

## Training a new adapter

### CSV manifest

The training CSV must have a header row with these columns:

| Column | Required? | Description |
|---|---|---|
| `ligand` | yes | Path to MOL2/SDF or a raw SMILES string. |
| `receptor` | yes | Path to a **Boltz YAML** (v1 constraint — see [Limitations](#limitations)). |
| `target` | yes | Float to fit (e.g. pIC50, ΔG). |
| `structure` | for `mode=rescore` | Path to the complex PDB/CIF whose coordinates you want to inject. |
| `name` | optional | Free-form identifier; defaults to row index. |

Extra columns are ignored.

Minimal example:

```csv
name,ligand,receptor,target,structure
mol_001,CCO,recipe.yaml,7.2,poses/mol_001.pdb
mol_002,c1ccccc1N,recipe.yaml,5.4,poses/mol_002.pdb
```

### Training command

```bash
boltz lora train \
    --name my_kinase_v1 \
    --csv data/kinase_train.csv \
    --loss huber \
    --rank 8 --alpha 16 --dropout 0.0 \
    --target-spec heads_pairformer \
    --epochs 10 --learning-rate 1e-4 \
    --recycling-steps 3 \
    --device auto \
    --notes "Initial run on March'26 kinase data"
```

Useful options:

- `--target-spec` — `heads`, `heads_pairformer`, or any regex. Try
  `heads` first if you have ≤ ~1k labelled pairs.
- `--rank` — 4–16 covers most use cases. Higher rank ⇒ more capacity ⇒
  more risk of overfitting on small data.
- `--gradient-clip` — defaults to 1.0; loosen for very small losses.
- `--use-msa-server` — if your receptor YAML hasn't been pre-computed
  with MSAs, enable this to call the MSA server on the fly.
- `--checkpoint` — override the base affinity checkpoint (defaults to
  the cached `boltz2_aff.ckpt`).

Output: a new adapter under `$BOLTZ_LORA_DIR/my_kinase_v1/` containing
`meta.json` (provenance) and `adapter.pt` (LoRA tensors).

---

## Custom losses

Built-in registry: `mse`, `mae`, `huber`, `bce`, `pairwise_ranking`.

For a custom loss, write a Python file and pass `path/to/file.py:function`:

```python
# my_loss.py
import torch
import torch.nn.functional as F

def asymmetric_mse(pred, batch, adapter_meta=None):
    """Penalise under-prediction 2x harder than over-prediction."""
    p = pred["affinity_pred_value"].squeeze(-1).float()
    t = batch["target"].float()
    err = p - t
    return torch.where(err < 0, 2.0 * err.pow(2), err.pow(2)).mean()
```

```bash
boltz lora train ... --loss /abs/path/my_loss.py:asymmetric_mse
```

**Contract.** The function must accept `(pred, batch)` or
`(pred, batch, adapter_meta)`. `pred` is the dict returned by the affinity
module (`affinity_pred_value`, `affinity_logits_binary`, …); `batch`
contains a `target` tensor of shape `[B]` plus the per-row metadata
(ligand, receptor, structure, name).

---

## Continuing training (active-learning loop)

`boltz lora update` resumes from an existing adapter's weights, appends a
new `TrainingRun` entry to its history, and (optionally) saves under a
child name so the parent is preserved.

Round-by-round pattern:

```bash
# Round 1: train initial adapter
boltz lora train --name pose_v1 --csv round1.csv --epochs 5

# Round 2: screen a large library
boltz rescore receptor \
    --receptor target.yaml --ligands lib.mol2 \
    --use-lora pose_v1 -o screen.csv

# Pick informative + diverse candidates for the next labelled set
boltz lora select \
    --scores screen.csv --top-k 96 \
    --strategy hybrid \
    --out next_round_candidates.csv

# (assay/label them externally, then build round2.csv)

# Round 3: continue training under a new name to keep round 1 around
boltz lora update --name pose_v1 --new-name pose_v2 \
    --csv round2.csv --epochs 5
```

Selection strategies (`boltz lora select`):

| Strategy | When to use |
|---|---|
| `topk` | Pure exploitation — best-predicted ligands. |
| `uncertainty` | Most uncertain (largest `affinity_pred_std`, or `|affinity_pred|` if no std column). |
| `diversity` | Maximise Tanimoto-furthest-first (greedy MaxMin). Requires RDKit. |
| `hybrid` *(default)* | z-score(uncertainty) + z-score(−diversity rank). |

---

## Using an adapter at inference

All inference entry points accept an adapter by **name** (registry lookup)
or by **absolute path** to an adapter directory.

```bash
# Single-pose rescoring
boltz rescore pdb complex.pdb --use-lora my_kinase_v1

# Batch
boltz rescore batch --input-dir poses/ --use-lora my_kinase_v1

# Receptor + MOL2 library
boltz rescore receptor --receptor target.yaml --ligands lib.mol2 \
    --use-lora my_kinase_v1

# YAML manifest
boltz rescore manifest --manifest jobs.yaml --use-lora my_kinase_v1

# Full Boltz predict (structure + affinity)
boltz predict --use_lora my_kinase_v1 input.yaml
```

You can also set `BOLTZ_RESCORE_LORA=my_kinase_v1` to apply an adapter
globally to any `boltz rescore` invocation without modifying flags.

To produce a *merged* checkpoint (LoRA folded into the base weight matrix,
no adapter needed at inference):

```bash
boltz lora export my_kinase_v1 --out merged.ckpt
```

---

## Registry, storage & provenance

Default root: `~/.boltz/loras`. Override with `BOLTZ_LORA_DIR=…`.

Layout:

```
$BOLTZ_LORA_DIR/
  registry.json                 # name → {path, created_at, rank, …}
  my_kinase_v1/
    meta.json                   # full LoRAAdapter (history, sha256, layers)
    adapter.pt                  # LoRA tensors + config blob
```

Every successful training run records:

- The hash + path of the base checkpoint (`base_checkpoint_sha256`).
- The hash of the training CSV (`data_sha256`) — so you always know which
  data produced which adapter.
- Hyperparameters, per-epoch loss, mode, and any free-form `--notes`.
- Parent adapter name (when produced by `update`).

Inspect with:

```bash
boltz lora list
boltz lora show my_kinase_v1     # pretty-prints meta.json
```

---

## CLI reference

```text
boltz lora train     Train a fresh LoRA adapter.
boltz lora update    Continue training an existing adapter on new data.
boltz lora list      List registered adapters.
boltz lora show      Pretty-print an adapter's metadata.
boltz lora apply     Dry-run: verify an adapter loads cleanly.
boltz lora export    Export a merged checkpoint (base + LoRA folded in).
boltz lora rm        Remove an adapter from the registry.
boltz lora select    Rank candidates for the next active-learning round.
```

Run `boltz lora <command> --help` for full option listings.

---

## Programmatic API

```python
from boltz.lora import (
    LoRARegistry,
    apply_lora,
    load_adapter_into_model,
    merge_lora,
)
from boltz.lora.train import TrainArgs, train_lora

# Train
adapter = train_lora(TrainArgs(
    name="my_adapter",
    csv_path="data.csv",
    loss_spec="huber",
    rank=8, epochs=5,
))

# List / load
registry = LoRARegistry()        # honours BOLTZ_LORA_DIR
for entry in registry.list():
    print(entry["name"], entry["rank"])

# Apply at inference
from boltz.affinity_rescoring.model_manager import AffinityModelManager
model = AffinityModelManager().load_model()
load_adapter_into_model(model, "my_adapter")
```

Lower-level building blocks:

- `LoRALinear` — the wrapper module.
- `apply_lora(model, patterns, *, r, alpha, dropout, freeze_base)` —
  in-place injection.
- `remove_lora(model)` / `merge_lora(model)` — inverse operations.
- `lora_state_dict(model)` / `load_lora_state_dict(model, state)` —
  save/load only the A/B factors.

---

## Limitations

- **`mode='full'`** (training through the diffusion stack) is *not* wired
  up yet. The training entry point raises `NotImplementedError`. Use
  `mode='rescore'` with explicit `structure` paths in your CSV, which
  mirrors `boltz rescore` and is the recommended workflow.
- **Receptor input format**: the training CSV's `receptor` column must
  point to a Boltz YAML (the same kind you pass to `boltz predict`).
  Automatic conversion from a raw PDB receptor + ligand SMILES is out of
  scope for v1; materialise the YAML first or reuse one of the
  examples under `examples/`.
- **Compatibility**: an adapter is tied to the affinity checkpoint it
  was trained against. The base checkpoint SHA-256 is recorded in
  `meta.json`; the loader currently does *not* hard-fail on mismatch (it
  only re-injects into matching layer names), but mismatched
  checkpoints will degrade quality silently. Stay on the same base
  weights or retrain.
- **Multi-adapter stacking** is not supported in v1: loading an adapter
  refuses to overwrite an already-injected one unless
  `allow_restack=True` is passed to the programmatic API.

---

## Troubleshooting

**`apply_lora matched zero layers`** — your target spec didn't match any
linear layer name. Run `python -c "from boltz...; print([n for n,_ in model.named_modules()])"`
or use one of the presets (`heads`, `heads_pairformer`).

**Loss won't go down** — check that `structure` paths actually contain
the docked complex (not the apo receptor), and that `target` is in a
scale matched by your loss (e.g. don't use raw IC50 in nM with MSE —
convert to pIC50).

**RDKit ImportError on `boltz lora select --strategy diversity`** —
install RDKit (`pip install rdkit`) or fall back to `--strategy uncertainty`.
