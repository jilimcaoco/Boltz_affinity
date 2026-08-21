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
4. [Selectivity (cross-receptor) training](#selectivity-cross-receptor-training)
5. [Custom losses](#custom-losses)
6. [Continuing training (active-learning loop)](#continuing-training-active-learning-loop)
7. [Using an adapter at inference](#using-an-adapter-at-inference)
8. [Registry, storage & provenance](#registry-storage--provenance)
9. [CLI reference](#cli-reference)
10. [Programmatic API](#programmatic-api)
11. [Limitations](#limitations)

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
| `group_id` | optional | Assay identifier. Batched together by `AssayGroupedSampler` so the intra-assay pairwise Huber term always has pairs. For selectivity training this should instead hold the **panel / source** shared by both arms of a pair (see below). |
| `is_binder` | optional | 0/1 label for the BCE branch of the Boltz-2-style multi-task losses. Falls back to `target <= BOLTZ_LORA_BINDER_THRESHOLD`. |
| `is_censored` | optional | 0/1 flag for a right-censored measurement (reported as "> X"). Consumed by the `censored_*` and `selectivity_*` losses. |
| `pair_id` | optional | Shared by the rows of one ligand scored against two receptors. Enables cross-receptor selectivity training. |
| `receptor_id` | optional | Which arm a row is (e.g. `MOR` / `DOR`). Required alongside `pair_id`. |

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
- `--use-msa-server` — **DISABLED in this fork.** Pre-compute MSAs
  with `python -m boltz.affinity_rescoring.mmseqs2` (or
  `fineturning_experiment/precompute_msas.py`) and either embed the
  `.a3m` path under each protein chain's `msa:` key in the receptor
  YAML or expose the cache via `$BOLTZ_MSA_CACHE_DIR`. The trainer
  auto-resolves missing MSAs from the cache before featurizing each
  row, so a single cache satisfies prediction, rescoring, and LoRA
  training.
- `--checkpoint` — override the base affinity checkpoint (defaults to
  the cached `boltz2_aff.ckpt`).

Output: a new adapter under `$BOLTZ_LORA_DIR/my_kinase_v1/` containing
`meta.json` (provenance) and `adapter.pt` (LoRA tensors).

---

## Selectivity (cross-receptor) training

The stock losses fit *potency*: an absolute Huber plus an intra-assay pairwise
Huber over pairs of **ligands measured against one receptor**. Selectivity is
the same idea rotated ninety degrees — pairs of **receptors measured with one
ligand**. Scoring the same compound against both receptors makes the difference
`Δ̂ = ŷ_A − ŷ_B` invariant to every ligand-global nuisance (MW, size,
lipophilicity); Boltz-2's inference-time MW correction cancels exactly in `Δ̂`.

### Manifest

Add one row per (ligand, receptor) arm, joined by `pair_id`:

```csv
name,ligand,receptor,target,structure,pair_id,receptor_id,group_id,is_censored
naltrindole_MOR,<smiles>,mor.yaml,-0.3,poses/MOR/naltrindole.pdb,naltrindole,MOR,pdsp_src_812,0
naltrindole_DOR,<smiles>,dor.yaml,-2.6,poses/DOR/naltrindole.pdb,naltrindole,DOR,pdsp_src_812,0
fentanyl_MOR,<smiles>,mor.yaml,-2.4,poses/MOR/fentanyl.pdb,fentanyl,MOR,pdsp_src_931,0
fentanyl_DOR,<smiles>,dor.yaml,1.1,poses/DOR/fentanyl.pdb,fentanyl,DOR,pdsp_src_931,1
```

`target` stays on the head's native scale, `log10(Ki or IC50 in µM)`, lower =
stronger. From a Ki in nM that is `log10(Ki_nM / 1000)`.

Two conventions matter:

- **`group_id` holds the panel/source, not a per-receptor assay id.** The
  ligand-axis pairwise term keys on `(group_id, receptor_id)`, so it never
  pairs rows measured against *different* receptors — doing so would
  reintroduce exactly the assay offset that term exists to cancel.
- **Both arms should come from one source.** A cross-receptor pair is by
  construction cross-assay, so the offset cancellation that justifies the
  ligand-axis term does not transfer. Matched-panel data (both Ki values from
  one submission, one species, recorded radioligands) is what keeps the
  difference meaningful.

When `pair_id` is populated and `--batch-size > 1`, the trainer automatically
switches from `AssayGroupedSampler` to `PairedReceptorSampler`, which keeps
both arms of a pair in the same mini-batch. The stock sampler fills a batch
from a single `group_id` and therefore could never produce one — the
selectivity term would silently stay at zero.

### Losses

| Preset | Objective |
|---|---|
| `selectivity_joint` | Δ regression **and** soft ranking |
| `selectivity_delta_only` | Calibrated Δ magnitude |
| `selectivity_rank_only` | Ordering only — robust to per-receptor offsets and censoring |
| `selectivity_off` | Reduces *exactly* to `censored_boltz2_affinity` (the control arm) |

```bash
boltz lora train ... --batch-size 8 --loss selectivity_joint
```

Every term is independently weighted:

```
L = w_point * point(ŷ, y)                          # censor-aware absolute Huber
  + w_lig   * pairwise over (group_id, receptor_id) # ligand axis
  + w_sel   * huber(Δ̂ − Δ)                          # receptor axis
  + w_rank  * KL(σ(Δ/τ) ‖ σ(Δ̂/τ))                   # soft ordering
  + bce_weight * bce(binary_logits, is_binder)
```

Sweep without writing a loss file per point using
`BOLTZ_SELECTIVITY_<NAME>` environment overrides (`W_SEL`, `W_RANK`, `W_LIG`,
`W_POINT`, `DELTA`, `TAU`, `SEL_TAIL_GAMMA`, `BCE_WEIGHT`,
`REFERENCE_RECEPTOR`), or build one directly with
`boltz.lora.selectivity_losses.make_selectivity_loss(...)`.

Two deliberate design choices:

- **No "reward a large gap" term.** `−λ·|ŷ_A − ŷ_B|` is minimised at infinity:
  nothing anchors the magnitude, so predictions inflate without the *ordering*
  improving. The cross-receptor term is a Huber regression of the predicted
  difference onto the measured one — a proper scoring rule. To emphasise the
  selective tail, use `sel_tail_gamma`, which reweights examples without moving
  the optimum.
- **No gating on large differences.** A decision boundary needs the
  non-selective compounds too; every complete pair contributes, so equipotent
  ligands actively pull `Δ̂` toward zero.

Censoring is handled directionally: on the log scale a censored row is a lower
bound, so for `Δ = y_A − y_B` a censored `B` means the true `Δ` is at most the
reported one and only over-prediction is penalised. Pairs with both arms
censored are skipped. The ranking term keeps a censored pair only when the
bound reinforces the observed sign.

> Arm-swap augmentation is intentionally **not** implemented: the Huber is even
> and soft-label KL is symmetric, so presenting a pair in the other order is a
> mathematical no-op. Pair orientation is fixed deterministically instead
> (`reference_receptor`, else lexicographic `receptor_id`).

---

## Custom losses

Built-in registry: `mse`, `mae`, `huber`, `bce`, `pairwise_ranking`,
`intra_assay_huber`, `censored_intra_assay_huber`, `boltz2_affinity`,
`censored_boltz2_affinity`, plus the `selectivity_*` presets above.

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

A loss in an installed module can also be referenced by dotted path:

```bash
boltz lora train ... --loss boltz.lora.selectivity_losses:selectivity_joint
```

**Contract.** The function must accept `(pred, batch)` or
`(pred, batch, adapter_meta)`. `pred` is the dict returned by the affinity
module (`affinity_pred_value`, `affinity_logits_binary`, …); `batch`
contains a `target` tensor of shape `[B]` plus the per-row metadata
(ligand, receptor, structure, name, `group_id`, `is_binder`, `is_censored`,
`pair_id`, `receptor_id`).

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
