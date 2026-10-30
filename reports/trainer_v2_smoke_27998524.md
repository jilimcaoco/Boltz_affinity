# Trainer v2 smoke test: results and interpretation

**Job:** `trainer_v2_smoke` (`27998524`)  
**Run date:** 2026-10-29  
**Report date:** 2026-10-30  
**Verdict:** **PASS for the trainer-v2 behavior and end-to-end smoke path.** This is a
functional check, not evidence that v2 improves affinity-prediction accuracy.

## 1. Executive summary

The job completed all three training arms, inspected the saved artifacts, and rescored
the selected validation structures. The artifacts show the distinction the migration was
intended to restore: v2 LoRA and full fine-tuning changed the affinity pairformer stacks
and both affinity modules, while the legacy arm changed only `affinity_module1` heads.
The legacy run also logged the expected no-gradient warnings for its frozen stack and
out-of-forward targets.

Both vanilla and v2-LoRA rescoring produced 18 successful rows for the same 18 unique
structures. The smoke script selected 20 validation rows, but two structure names were
duplicated, so only 18 distinct PDB inputs were rescored. The mean absolute prediction
difference on the paired successful rows was **0.564**; this establishes that the adapter
affects scored outputs, not that those outputs are more accurate.

The main limitations are the very small, group-clustered subset; only two epochs per arm;
no censored training examples; and a dirty checkout. Treat the results as confirmation of
mechanics, not as a model comparison or a reproducibility record for a clean commit.

## 2. Test setup

| Item | Observed configuration |
|---|---|
| GPU | NVIDIA H200 NVL, 143,771 MiB reported total memory |
| Training subset | 64 rows from 2 assay groups |
| Validation subset | 32 rows from 3 assay groups |
| Training duration | 2 epochs per arm; batch size 8; 1 recycling step |
| Loss | `censored_intra_assay_huber` |
| v2 LoRA | Rank 8, alpha 16; default `affinity_heads_pairformer`; ensemble output |
| Legacy LoRA | Rank 8, alpha 16; `--legacy-trainer`; `heads_pairformer` |
| v2 full fine-tune | `affinity_heads_pairformer`; ensemble output; validation CSV supplied |
| Source revision | `1b7e973b592513231565ea9032cbce6691b1e7dc`, with dirty working tree |

The run directory was
`/scratch/maom_root/maom/limcaoco/trainer_v2_smoke_27998524`. The job header reported
20 modified tracked files; artifact provenance also records `boltz_git_dirty=true`.
Consequently, the commit hash alone does not identify the exact code that ran.

## 3. Did the migration mitigation work?

The migration note describes a defect in the old reentrant activation-checkpoint path:
when the frozen trunk supplied inputs without gradients, the checkpointed pairformer
stack did not get an autograd graph. The v2 path uses non-reentrant checkpointing,
selects affinity modules matching the output being trained, and records gradient
coverage. The artifact comparisons provide direct evidence for the expected behavior.

| Arm | Artifact evidence | Interpretation |
|---|---|---|
| `lora_v2` | 294 adapted Linear layers. Nonzero `lora_B`: module 1 stack 184/184 and heads 5/9; module 2 stack 92/92 and heads 5/9. Stack gradient coverage was 368/368 and 184/184 tensors; each head scope was 10/18. | Both affinity stacks and both modules participated. |
| `ft_v2` | Changed vs. base: module 1 stack 296/296 and heads 10/18; module 2 stack 148/148 and heads 10/18. | Full fine-tuning updated both stacks and both heads. |
| `lora_legacy` | Nonzero `lora_B`: module 1 heads 5/9; module 1 stack 0/24; module 2 stack and heads 0; confidence stack 0. | The legacy run reproduces the heads-only behavior. Its no-gradient warnings were marked expected for legacy mode. |

The v2 LoRA gradient-coverage record reports gradients for every stack tensor and 10
of 18 trainable tensors in each head scope. The artifact check independently confirms
nonzero updates in the stack and five of nine adapted head linears per affinity module.
The legacy trace emitted PyTorch's “None of the inputs have requires_grad=True” checkpoint
warning and explicitly reported zero coverage in its affinity stack.

This is a strong smoke-level confirmation of the documented mitigation. It does not
establish that every head tensor must receive gradients; the check here is that the
intended stack and affinity-module scopes are active and their artifacts changed.

## 4. Training behavior

Losses below are the values recorded after each epoch. These short runs should not be
compared as a performance ranking: the legacy arm uses different trainer/output
semantics, and the arms have very few training examples.

| Arm | Epoch 1 train loss | Epoch 1 val loss | Epoch 2 train loss | Epoch 2 val loss |
|---|---:|---:|---:|---:|
| `lora_v2` | 1.87793 | 1.83996 | 1.71011 | 1.82152 |
| `lora_legacy` | 2.09044 | 1.81807 | 2.12949 | 1.81082 |
| `ft_v2` | 1.90277 | 1.82832 | 1.58361 | 1.86374 |

V2 LoRA training and validation losses both decreased slightly over the two epochs.
Full fine-tuning's training loss fell while its validation loss rose from 1.82832 to
1.86374, a warning sign of validation regression at epoch 2 in this tiny run. The
legacy arm's training loss rose while its validation loss improved slightly. None of
these two-epoch trajectories should be interpreted as a stable generalization estimate.

## 5. Runtime and memory

| Arm | Wall time | Peak GPU memory |
|---|---:|---:|
| `lora_v2` | 353 s | 35,521 MiB |
| `lora_legacy` | 42 s | 5,125 MiB |
| `ft_v2` | 103 s | 8,901 MiB |

The higher v2-LoRA cost is consistent with the broader active target scope: it adapts
294 linears across both affinity modules, versus 102 legacy adapters whose stack and
some other targets received no gradients. These are measurements from a single short
smoke job, not throughput estimates for full datasets.

## 6. Rescoring check

The job selected 20 validation rows. Two structure stems occurred twice, leaving **18
unique input structures**. The resulting vanilla and `lora_v2` CSVs each contain those
same 18 IDs, and all 18 rows in each file have `validation_status=SUCCESS`.

| Check | Result |
|---|---:|
| Selected validation rows | 20 |
| Unique structure IDs / input PDBs | 18 |
| Vanilla successful rows | 18 / 18 |
| V2-LoRA successful rows | 18 / 18 |
| Shared successful rows | 18 |
| Mean absolute `affinity_pred` difference, v2 LoRA vs vanilla | 0.564 |
| Median / maximum absolute difference | 0.599 / 0.750 |

Three rows in each score CSV carry an `INVALID_ELEMENTS` warning for chlorine (`Cl`)
while still having successful validation status. This merits attention in any broader
chemistry-focused evaluation, but it did not prevent the smoke test's paired rescoring.
The observed prediction differences confirm a changed score path; without labels and an
adequately sized held-out evaluation, they do not imply improved accuracy.

## 7. Caveats and follow-up

1. **Small, clustered subset.** The 64 training rows span only two assay groups and the
   32 validation rows span three. Two epochs and this subset validate execution and
   parameter movement, not generalization.
2. **Censoring was not exercised.** The log says none of the 64 training rows had
   `is_censored=1`; the selected censored-Huber loss therefore reduced to the uncensored
   loss. This test does not validate right-censored examples or their handling.
3. **Dirty source tree.** The run records commit `1b7e973...` and `boltz_git_dirty=true`.
   Preserve the exact patch or rerun from a clean, pinned revision before treating this
   as a reproducible benchmark.
4. **Duplicate structure names.** The rescore subset has 20 CSV rows but only 18 unique
   structure stems. Future checks should report the unique input count explicitly (or
   deduplicate rows) so the requested and actual pose counts are unambiguous.
5. **Compatibility warning.** The log notes that the checkpoint was produced with
   PyTorch Lightning `2.5.0.post0`, newer than the environment's `2.5.0`. Loading and
   scoring completed, but matching the checkpoint's dependency version is preferable
   for controlled runs.

**Recommended next step:** run a larger v2 validation/training job from a clean pinned
revision, with enough independent assay groups and censored examples if censor handling
is in scope. Use a new `_v2` arm tag, retain the generated `meta.json` provenance, and
compare held-out metrics against vanilla rather than relying on training loss or score
shifts alone.

## 8. Sources

- Smoke-test log: [`trainer_v2_smoke_27998524.log`](../trainer_v2_smoke_27998524.log)
- Migration and mitigation notes: [`MIGRATION_TRAINER_V2.md`](../docs/MIGRATION_TRAINER_V2.md)
- Smoke-test job script: [`trainer_v2_smoke.slurm`](../scripts/verify/trainer_v2_smoke.slurm)
- Saved run artifacts: `/scratch/maom_root/maom/limcaoco/trainer_v2_smoke_27998524`
