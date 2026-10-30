# Spec: data-derived Huber `delta` for `boltz.lora` training

**Target repo:** `/home/limcaoco/turbo/limcaoco/Boltz_affinity` (branch `TRIAGE-LORA`)
**Requested by:** LSD_finetuning project
**Status:** implement as specified; do not redesign without flagging

---

## 1. Objective

Add two new ways to set the Huber transition point `delta` in
`boltz.lora` training, alongside the existing fixed value:

| mode | behaviour |
|---|---|
| `fixed` | **current behaviour.** `delta` is a constant (default 1.0). Must remain the default. |
| `static-vanilla` | compute `delta` **once**, before training, from the **un-adapted (vanilla) model's residuals** on the training set, then hold it constant |
| `dynamic-mad` | recompute `delta` from the current model's residuals **once per epoch** |

All three must be selectable from the `boltz lora train` CLI and recorded in the
adapter's `meta.json`.

---

## 2. Why — measured background

Read this before implementing; it determines several requirements below.

### 2.1 The targets are z-scored, so the residual scale is nearly constant

Training targets in this project are `-zscore(potency)` within `group_id`, so the
target standard deviation is ~1.0 **by construction**. Measured residual scale
(`pred - target`) on three receptors, for the un-adapted model and a trained LoRA:

| receptor | model | pred std | residual MAD | `sigma_hat = MAD/0.6745` | `delta = 1.345*sigma_hat` |
|---|---|---:|---:|---:|---:|
| D4 | vanilla | 0.611 | 0.714 | 1.058 | **1.423** |
| D4 | trained | 0.542 | 0.725 | 1.074 | **1.445** |
| SIGMA2 | vanilla | 0.603 | 0.711 | 1.054 | **1.417** |
| SIGMA2 | trained | 0.556 | 0.644 | 0.954 | **1.283** |
| AMPC | vanilla | 0.599 | 0.469 | 0.696 | **0.936** |
| AMPC | trained | 0.534 | 0.497 | 0.737 | **0.992** |

Two consequences:

- **Vanilla and trained estimates agree to within 1.5-9%.** The vanilla estimate is
  therefore an adequate, and importantly *non-circular*, reference. (Estimating
  `delta` from the trained model is circular: the model depended on the `delta`
  used to train it.)
- **A MAD-derived `delta` will almost always land in 0.9-1.5 for this pipeline.**
  Use this as a sanity check: if the implementation returns a value far outside
  that band for z-scored targets, something is wrong.

### 2.2 `smooth_l1(beta=d)` is NOT `Huber(delta=d)`

Every Huber loss in `src/boltz/lora/losses.py` currently uses
`F.smooth_l1_loss(..., beta=delta)`. The relationship is

```
Huber(x, delta) = delta * smooth_l1(x, beta=delta)
```

They coincide **only at `delta = 1.0`**. Verified numerically (`ratio huber/smooth_l1`
= 0.250, 0.500, 1.000, 2.000 at `delta` = 0.25, 0.5, 1.0, 2.0).

This is harmless today because every completed run used `delta = 1.0`. It stops being
harmless the moment `delta` varies, because the loss magnitude then scales as `1/delta`.
Measured in a sweep that varied `delta` over 0.25-2.0: median best `val_loss` moved
0.670 -> 0.228, a **2.9x magnitude gradient**, which interacted with the *absolute*
`--early-stopping-min-delta` and made high-`delta` arms stop ~2 epochs earlier than
low-`delta` arms. That confounded the axis under test.

**Therefore: if `delta` is allowed to vary, early stopping must not use a bare absolute
threshold.** See requirement R4.

### 2.3 `delta = 1.0` is not degenerate

For the record, since it affects whether this work is worth doing: at `delta = 1.0`,
66% (D4) / 70% (SIGMA2) / 75% (AMPC) of held-out residuals fall in the quadratic branch,
so 25-34% are in the robust linear branch. The Huber function is already active at the
current default. The motivation for this patch is **principled, auditable parameter
selection and robustness to a change of data regime** (upcoming ChEMBL-scale training,
where the residual scale may genuinely differ), *not* a known deficiency at `delta = 1.0`.

---

## 3. Non-negotiable constraints

**R1 — Exact backward compatibility.** Over 110 trained adapters in the downstream
project depend on current behaviour. With no new flags passed, training must produce
**bit-identical** results to the current code. The default mode is `fixed` with
`delta = 1.0`, and in that path no residual pass, no MAD computation, and no change to
the loss call may occur.

**R2 — Do not change the existing loss functions' defaults.** `delta: float = 1.0` and
`pair_weight: float = 2.0` stay as they are in
`censored_intra_assay_huber_loss` and `intra_assay_huber_loss`.

**R3 — Losses that do not accept `delta` must be unaffected.** `mse`, `mae`, `bce`,
`pairwise_ranking` take no `delta`. Binding must be conditional on the parameter
existing in the signature; passing a delta mode with one of these losses should emit a
warning and proceed, not crash.

**R4 — A varying `delta` requires a scale-stable stopping criterion.** For
`dynamic-mad`, the absolute `--early-stopping-min-delta` comparison is invalid because
the monitored loss changes scale between epochs. Implement **one** of:

- (preferred) `--early-stopping-rel-delta FLOAT`: stop when
  `improvement < rel_delta * abs(current_best)`; mutually exclusive with the absolute
  flag, and **required** when `--huber-delta-mode dynamic-mad` is used; or
- freeze `delta` after a warmup of N epochs and document that the first N epochs'
  losses are not comparable.

Pick the first unless there is a concrete reason not to. For `static-vanilla` this does
not apply (delta is constant within a run), but note that loss magnitudes are then not
comparable *across* runs with different receptors.

**R5 — Everything must be recorded.** The resolved numeric `delta`, the mode, the
constant `k`, and (for `dynamic-mad`) the per-epoch `delta` values must be written into
the adapter `meta.json` run history. A downstream audit must be able to confirm which
`delta` each arm actually used. This is not optional — a prior sweep in this project
was only auditable because per-arm parameters had been recorded.

---

## 4. Required changes, by file

### 4.1 `src/boltz/lora/losses.py`

1. Add a module-level helper:

```python
def robust_delta(residuals, k: float = 1.345, floor: float = 1e-3) -> float:
    """Huber transition point from a robust scale estimate of `residuals`.

    sigma_hat = MAD / 0.6745  (consistent with sigma for Gaussian data)
    delta     = k * sigma_hat

    k = 1.345 gives 95% asymptotic efficiency under Gaussian errors while
    bounding the influence of outliers (Huber's standard tuning constant).
    """
```

   - MAD must be taken about the **median of the residuals**, not about zero, so a
     systematic prediction offset does not inflate the scale estimate. (This matters:
     measured residual means are non-zero, e.g. -0.310 for a trained D4 arm.)
   - Guard against degenerate input: if fewer than 8 finite residuals, or
     `MAD == 0`, return `None` and let the caller fall back to the fixed default
     with a warning.
   - Clamp the result to `>= floor`.

2. Add a binder with a stable signature:

```python
def bind_delta(loss_fn: LossFn, delta: float) -> LossFn:
    """Return `loss_fn` with `delta` bound, or `loss_fn` unchanged if it has no
    `delta` parameter."""
```

   Implement with `inspect.signature` + `functools.partial`, and ensure the returned
   object still satisfies the 2-vs-3-arg dispatch in `call_loss` (line ~514). Add a
   test for that specifically — `call_loss` introspects signatures and a naive
   `partial` can change what it sees.

3. **Optional but recommended**, gated behind a keyword defaulting to current
   behaviour: a `scale_invariant: bool = False` parameter on the Huber-family losses
   that multiplies the result by `delta`, recovering textbook `Huber(delta)` semantics.
   Default `False` preserves R1. Document the `Huber = delta * smooth_l1` relationship
   in the module docstring either way.

4. Export the new helpers in `__all__`.

### 4.2 `src/boltz/lora/train.py`

1. Extend `TrainConfig` (fields are around lines 50-78) with:

```python
huber_delta_mode: str = "fixed"        # fixed | static-vanilla | dynamic-mad
huber_delta: float = 1.0               # used when mode == "fixed"
huber_delta_k: float = 1.345           # robust tuning constant
early_stopping_rel_delta: float = 0.0  # 0 disables; see R4
```

2. **Resolving `delta` for `static-vanilla`.** Before LoRA is injected, run one
   no-grad forward pass over the training set with the **un-adapted** model, collect
   `pred - target` residuals, and call `robust_delta`. Requirements:

   - It must be genuinely vanilla — either computed before injection, or with adapter
     scaling forced to zero. Verify by asserting the pass produces identical predictions
     to a freshly loaded base model.
   - Reuse the existing trunk/feature cache so this costs one cached epoch, not a full
     recompute.
   - Log the resolved value at INFO: `"Huber delta (static-vanilla): sigma_hat=%.4f -> delta=%.4f (k=%.3f, n=%d)"`.
   - On failure (too few residuals, zero MAD), warn and fall back to `huber_delta`.

3. **Resolving `delta` for `dynamic-mad`.** At the **end of each epoch**, compute
   residuals from that epoch's training predictions (accumulate them during the epoch —
   do not run a second pass) and rebind the loss for the next epoch. Per-batch
   recomputation is explicitly **not** wanted: with batch sizes of 32-64 the MAD
   estimate is too noisy.

4. Bind the resolved `delta` at the loss-load site (line ~1030,
   `loss_fn = load_loss_from_spec(args.loss_spec)`), and rebind each epoch in dynamic
   mode. Both the training call (line ~1180) and the validation call (line ~1277) must
   use the **same** bound `delta` within an epoch.

5. Early stopping (lines ~1300-1325): add the relative-criterion branch from R4.

6. Append to the run metadata: `huber_delta_mode`, `huber_delta_resolved`,
   `huber_delta_k`, and for dynamic mode a `huber_delta_per_epoch` list.

### 4.3 `src/boltz/lora/cli.py`

Add click options next to the existing `--loss` (line ~58) and
`--early-stopping-min-delta` (line ~94), with `show_default=True`:

```
--huber-delta-mode [fixed|static-vanilla|dynamic-mad]   (default: fixed)
--huber-delta FLOAT                                     (default: 1.0)
--huber-delta-k FLOAT                                   (default: 1.345)
--early-stopping-rel-delta FLOAT                        (default: 0.0)
```

Wire them into both `TrainConfig` construction sites (lines ~164 and ~237).
Error clearly if `--huber-delta-mode dynamic-mad` is given without
`--early-stopping-rel-delta` (R4).

### 4.4 `src/boltz/lora/adapter.py`

Ensure the new metadata fields survive the round-trip to `meta.json` (loss metadata is
handled around line 42).

---

## 5. Tests

Add to the existing test suite; all must pass.

**Backward compatibility (highest priority)**
1. Training with no new flags produces an adapter whose weights are bit-identical to
   the pre-patch code on a fixed seed. If a full training run is too slow for CI, assert
   equality of the loss value and the first optimizer step instead.
2. `bind_delta(fn, 1.0)` applied to each Huber-family loss returns exactly the same
   value as the unbound call.

**`robust_delta`**
3. Gaussian input: for `r ~ N(0, s)` with large n, `robust_delta(r, k=1.345)` is within
   5% of `1.345 * s`.
4. Offset invariance: adding a constant to all residuals does not change the result
   (confirms MAD is about the median).
5. Outlier resistance: corrupting 10% of residuals to large values changes the result
   by less than 10%.
6. Degenerate input (n < 8, or all-identical residuals) returns `None` and the caller
   falls back to the fixed default with a warning.

**Binding and dispatch**
7. `bind_delta` on a loss with no `delta` parameter (`mse_loss`) returns the function
   unchanged and does not raise.
8. `call_loss` works on a bound loss — explicitly covers the signature-introspection
   path at line ~514.

**Mode behaviour**
9. `static-vanilla`: `delta` is resolved once and is identical at every epoch; the
   recorded `huber_delta_resolved` matches a hand-computed `robust_delta` on the
   vanilla residuals.
10. `dynamic-mad`: `huber_delta_per_epoch` has one entry per epoch and at least two
    distinct values on a task where residuals shrink.
11. `dynamic-mad` without `--early-stopping-rel-delta` raises a clear error.

**Scale relationship**
12. Assert `huber_loss(x, delta=d) == d * smooth_l1(x, beta=d)` for
    `d in {0.25, 0.5, 1.0, 2.0}`, documenting the convention in the test name.

---

## 6. Acceptance criteria

- [ ] Default path bit-identical to pre-patch (R1), demonstrated by test 1
- [ ] All three modes selectable from `boltz lora train` and recorded in `meta.json` (R5)
- [ ] `dynamic-mad` cannot be used with a bare absolute early-stopping threshold (R4)
- [ ] Losses without a `delta` parameter are unaffected (R3)
- [ ] On z-scored targets, `static-vanilla` resolves `delta` into roughly **0.9-1.5**
      (§2.1) — if it does not, stop and report rather than shipping
- [ ] Full existing test suite still passes
- [ ] `delta` resolution adds at most one cached forward pass over the training set

---

## 7. Out of scope

Do not change in this patch:

- the `pair_weight` / `lambda_pair` term or its default of 2.0
- the `AssayGroupedSampler` or any batching behaviour
- the target construction (`-zscore` within `group_id`)
- the set of built-in losses, beyond the additions above
- left-censoring support (tracked separately downstream)

If implementing R4 appears to require broader changes to the early-stopping logic than
described, stop and report rather than refactoring it.
