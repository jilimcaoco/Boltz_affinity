# Trainer v2: what changed and what it means for existing results

Applies to `boltz lora train|update` and `boltz finetune train|update` on branch
`TRIAGE-LORA`. Rescoring (`boltz rescore`, `boltz predict --use_lora`) is
**unchanged**.

## The defect

The Boltz-2 affinity checkpoint enables activation checkpointing in both
affinity pairformer stacks (`pairformer_args.activation_checkpointing: true`).
`PairformerNoSeqModule` used `torch.utils.checkpoint.checkpoint` in its default
*reentrant* mode. When a checkpointed layer's input does not require grad —
always the case when fine-tuning on top of the frozen trunk — reentrant
checkpointing returns an output with no autograd graph, so **no parameter inside
the stack, LoRA or base, ever received a gradient**. Only the output heads,
which sit after the stack, trained.

Measured on the trained artifacts:

| Artifacts | Trained parameters that actually changed |
|---|---|
| All 150 LoRA adapters in `LSD_finetuning` (main 46, `censored_loss_experiment` 56, `gamma_lambda_sweep_experiment` 48) | `lora_B` nonzero only in `affinity_module1.affinity_heads` |
| All 8 LSD full fine-tunes (`*_ft_full_huber`, `*_ft_l2sp_huber`) | 10 of 18 `affinity_module1.affinity_heads` tensors; all 296 stack tensors and all of `affinity_module2` bit-identical to the base checkpoint |
| This repo's 4 adapters and `drd4_full_v1` | same pattern |

Two further issues compounded it:

- **Trained ≠ scored output.** Training fit `affinity_module1`'s raw value;
  rescoring reports `1.035·(m1 + m2)/2 − 0.600·MW^0.3 + 2.833`. Because
  `affinity_module2` never changed, a learned shift Δ in module1 appears as
  ≈ 0.52·Δ in scored values.
- **Presets.** The LoRA `heads_pairformer` preset's attention/multiplication
  patterns match no Boltz-2 layer, and it also adapts `confidence_module`
  (48 inert layers). The finetune preset of the same name trains every head
  and stack parameter, so "LoRA vs full-FT" arms differed in scope.

**Interpretation of existing results:** every adapter and fine-tune trained
before trainer v2 is a *heads-only change to `affinity_module1`*, attenuated
≈ 0.52× at scoring. Rank, rank-64 and rank-384 arms differ only in head-adapter
capacity; LoRA-vs-full-FT comparisons compare two heads-only fits. The score
CSVs themselves are valid records of those models (rescoring is unchanged),
including all `*_vanilla.csv`.

## What v2 changes

| | Before (and `--legacy-trainer`) | Trainer v2 default |
|---|---|---|
| Activation checkpointing in pairformer stacks | reentrant (stack frozen in practice) | non-reentrant (stack trains; same memory) |
| `--affinity-output` | `module1` | `ensemble` (the value rescoring reports) |
| Target presets | `heads`, `heads_pairformer` | `affinity_heads`, `affinity_pairformer`, `affinity_heads_pairformer` (LoRA default), `affinity_module` |
| Gradient-coverage guard | — | run fails if a targeted scope in the forward gets no gradient in epoch 1 |
| `boltz finetune` validation | none (early-stops on train loss) | `--val-csv` / `--val-split`, same criterion as LoRA |
| Provenance in `meta.json` | `boltz_version` only | `trainer_semantics`, `affinity_output`, `target_modules`, `grad_coverage`, `boltz_git_sha`, `boltz_git_dirty`, `torch_version` |

The new presets are anchored at the affinity modules actually used by the
training forward (`affinity_module1` and `affinity_module2` for `ensemble`,
`affinity_module1` for `module1`), and mean the same module subtrees for LoRA
(every `nn.Linear` inside) and full fine-tuning (every parameter inside).
Old preset names still resolve to their original patterns, with a warning.

## Reproducing an existing arm

```bash
boltz lora train ... --legacy-trainer --target-spec heads_pairformer
boltz finetune train ... --legacy-trainer --target-spec heads_pairformer
```

`--legacy-trainer` restores reentrant checkpointing, module1 output and pre-v2
preset resolution. It is bit-identical to commit `1b7e973` (verified on real
`AffinityModule`s, including RNG state with dropout). GPU nondeterminism can
still prevent byte-identical re-runs of the original jobs.

## Running new arms

- Use new arm tags (e.g. suffix `_v2`) so v2 and pre-v2 score CSVs never mix
  in one evaluation; `evaluate_adapters.py` discovers models by file name.
- Prefer `--target-spec affinity_heads_pairformer` (LoRA and finetune) and the
  default `--affinity-output ensemble`.
- Pass `--val-csv` to `boltz finetune train` as well, so both methods
  early-stop on validation loss.
- Expect longer epochs: the backward pass now runs through 8 + 4 pairformer
  blocks. Activation memory stays bounded (non-reentrant checkpointing still
  stores only layer inputs), but many more parameters now train: with the
  Boltz-2 checkpoint, `affinity_heads_pairformer` adapts 294 Linears in
  ensemble mode (`affinity_module1`: 184 stack + 9 heads; `affinity_module2`:
  92 + 9), versus the 9 that actually trained before.
- `rank` now matters for the stack; revisit rank sweeps rather than reusing
  conclusions drawn from heads-only runs.

## Recording what code trained an artifact

`LSD_finetuning` uses this repository as an unpinned editable install, so a
`git pull` here changes the next LSD job. Every new `meta.json` records
`history[-1].boltz_git_sha` and `boltz_git_dirty`; check them when collecting
results, and record the commit used for each sweep in the experiment's config.
