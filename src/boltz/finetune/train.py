"""Full fine-tuning trainer for the Boltz affinity stack.

Twin of :mod:`boltz.lora.train` — same CSV manifest, same losses, same
``mode='rescore'`` differentiable forward — but instead of injecting LoRA
adapters and freezing the base, we directly mark a subset of the *base*
parameters as trainable and let AdamW update them.

We deliberately reuse :func:`boltz.lora.train._forward_batch` (featurisation
+ :func:`boltz.lora.train._affinity_forward_trainable`) and its validation /
gradient-coverage helpers so the two training modes share an identical
compute graph and stopping criterion.  Anything that improves
featurisation or the forward pass automatically benefits both.

Saved artefact: a *partial* state-dict (only the trained tensors) plus
provenance metadata.  Re-applying it during inference is a
``model.load_state_dict(..., strict=False)``-style copy through
:func:`boltz.finetune.apply.load_finetune_into_model`.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from boltz import __version__ as _BOLTZ_VERSION  # type: ignore[attr-defined]

from boltz.finetune.adapter import (
    FinetuneConfig,
    FinetuneRecord,
    TrainingRun,
)
from boltz.finetune.l2_sp import add_l2_sp_penalty
from boltz.finetune.registry import FinetuneRegistry, default_registry
from boltz.finetune.targets import resolve_targets

# Reuse the LoRA-side machinery — these helpers are mode-agnostic.
from boltz.lora.data import (
    AssayGroupedSampler,
    LoRADataset,
    lora_collate,
    warn_if_censored_loss_without_censored_rows,
)
from boltz.lora.losses import LossFn, call_loss, load_loss_from_spec
from boltz.lora.provenance import repo_state
from boltz.lora.train import (
    GradCoverage,
    _build_val_loader,
    _checkpoint_path_for_epoch,
    _configure_train_cache,
    _dataset_rows,
    _forward_batch,
    _iter_checkpoint_paths,
    _latest_checkpoint_path,
    _load_base_model,
    _load_checkpoint,
    _pick_device,
    _plot_loss_curve,
    _run_validation,
    _save_checkpoint,
    _split_train_val,
    affinity_modules_for,
    check_grad_coverage,
    resolve_affinity_output,
    set_checkpoint_mode,
)

logger = logging.getLogger(__name__)


@dataclass
class FinetuneArgs:
    """User-facing knobs surfaced via the CLI."""

    name: str
    csv_path: str
    mode: str = "rescore"
    loss_spec: str = "mse"
    target_spec: str = "affinity_module"
    epochs: int = 5
    learning_rate: float = 1e-5  # smaller than LoRA default — full FT is risky
    batch_size: int = 1
    weight_decay: float = 0.0
    l2_sp_weight: float = 0.0
    gradient_clip: float = 1.0
    recycling_steps: int = 3
    device: str = "auto"
    checkpoint: Optional[str] = None
    use_msa_server: bool = False
    overwrite: bool = False
    notes: Optional[str] = None
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.005
    plot_loss_curve: bool = True
    # See boltz.lora.train.TrainArgs.affinity_output / legacy_trainer.
    affinity_output: Optional[str] = None
    legacy_trainer: bool = False
    # Validation, as in boltz.lora.train.TrainArgs: explicit CSV, or the last
    # ``val_split`` fraction of training rows. val_loss then drives early
    # stopping, so LoRA and full fine-tune arms stop on the same criterion.
    val_csv_path: Optional[str] = None
    val_split: float = 0.0


# ── Parameter selection ─────────────────────────────────────────────────────


def _select_trainable_params(
    model: torch.nn.Module, patterns: Iterable[str]
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return the named parameters whose name matches any of ``patterns``."""
    pats = tuple(patterns)
    if not pats:
        msg = "No target patterns supplied for fine-tune parameter selection."
        raise ValueError(msg)
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for n, p in model.named_parameters():
        if any(re.search(pat, n) for pat in pats):
            selected.append((n, p))
    if not selected:
        msg = (
            "Fine-tune target patterns matched zero parameters. "
            f"Patterns={list(pats)}. "
            "Inspect model.named_parameters() for valid names."
        )
        raise ValueError(msg)
    return selected


def _freeze_all_then_unfreeze(
    model: torch.nn.Module, trainable_names: set[str]
) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = n in trainable_names


# ── Base-model loader ────────────────────────────────────────────────────────
# `_load_base_model` is imported from boltz.lora.train above to avoid a
# byte-for-byte duplicate; both trainers share the same loader contract.


# ── Trainer ─────────────────────────────────────────────────────────────────


def train_finetune(
    args: FinetuneArgs,
    *,
    registry: Optional[FinetuneRegistry] = None,
    init_state: Optional[dict[str, torch.Tensor]] = None,
    parent: Optional[str] = None,
    prior_history: Optional[list[TrainingRun]] = None,
) -> FinetuneRecord:
    """Run a full fine-tune end-to-end and persist the artefact."""
    if args.mode == "full":
        msg = (
            "mode='full' (full Boltz pipeline incl. diffusion) is not wired "
            "up yet. Use mode='rescore' with a 'structure' column."
        )
        raise NotImplementedError(msg)
    if args.mode != "rescore":
        msg = f"Unknown mode {args.mode!r}; expected 'rescore' or 'full'."
        raise ValueError(msg)

    requested_output = args.affinity_output
    args = dataclasses.replace(
        args,
        affinity_output=resolve_affinity_output(args.affinity_output, args.legacy_trainer),
    )

    registry = registry or default_registry()
    device = _pick_device(args.device)

    dataset = LoRADataset(args.csv_path, mode=args.mode)
    train_dataset, val_dataset = _split_train_val(
        dataset, mode=args.mode,
        val_csv_path=args.val_csv_path, val_split=args.val_split,
    )
    _train_rows = _dataset_rows(train_dataset, dataset)

    has_groups = any(r.group_id for r in _train_rows)
    warn_if_censored_loss_without_censored_rows(args.loss_spec, _train_rows)
    _sampler: Optional[AssayGroupedSampler] = None
    if args.batch_size > 1 and has_groups:
        _sampler = AssayGroupedSampler(
            _train_rows, batch_size=args.batch_size, shuffle=True,
        )
        loader = DataLoader(
            train_dataset, batch_sampler=_sampler,
            collate_fn=lora_collate, num_workers=0,
        )
        logger.info(
            "AssayGroupedSampler: %d assay groups, %d ungrouped rows.",
            _sampler.n_assay_groups, _sampler.n_ungrouped,
        )
    else:
        loader = DataLoader(
            train_dataset, batch_size=max(1, args.batch_size), shuffle=True,
            collate_fn=lora_collate, num_workers=0,
        )
    val_loader: Optional[DataLoader] = (
        _build_val_loader(val_dataset, dataset, args.batch_size)
        if val_dataset is not None else None
    )

    model, ckpt_path, ckpt_sha = _load_base_model(args.checkpoint, device)
    model.train()

    set_checkpoint_mode(model, legacy=args.legacy_trainer)
    target_modules = affinity_modules_for(model, args.affinity_output)
    target_patterns = resolve_targets(
        args.target_spec, modules=target_modules, legacy=args.legacy_trainer,
    )
    logger.info(
        "Trainer semantics: %s (affinity_output=%s, target modules=%s).",
        "legacy" if args.legacy_trainer else "v2", args.affinity_output,
        ",".join(target_modules),
    )
    if requested_output == "ensemble" and not getattr(model, "affinity_ensemble", False):
        logger.warning(
            "affinity_output='ensemble' requested but the checkpoint has a "
            "single affinity module; training on its output (same as module1)."
        )
    selected = _select_trainable_params(model, target_patterns)
    trainable_names = {n for n, _ in selected}
    _freeze_all_then_unfreeze(model, trainable_names)
    model = model.to(device)
    trainable_param_dict = dict(selected)
    _configure_train_cache(model, ckpt_sha)

    # L2-SP anchor: snapshot values right after loading the base checkpoint
    # (before any --update resume state below), so the penalty always pulls
    # toward the pretrained values for the lifetime of this fine-tune.
    initial_params: Optional[dict[str, torch.Tensor]] = None
    if args.l2_sp_weight > 0.0:
        initial_params = {
            n: p.detach().clone() for n, p in trainable_param_dict.items()
        }

    # Optionally resume from a previous fine-tune's weights.
    if init_state is not None:
        model_state = model.state_dict()
        loaded = 0
        for k, v in init_state.items():
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
                with torch.no_grad():
                    model_state[k].copy_(v.to(model_state[k].device,
                                              dtype=model_state[k].dtype))
                loaded += 1
        logger.info("Loaded %d / %d tensors from init state.",
                    loaded, len(init_state))

    n_params = sum(p.numel() for _, p in selected)
    logger.info(
        "Fine-tune target-spec=%r matched %d tensors (%.2fM params). "
        "Sample: %s",
        args.target_spec, len(selected), n_params / 1e6,
        [n for n, _ in selected[:3]],
    )

    optim = torch.optim.AdamW(
        [p for _, p in selected],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loss_fn: LossFn = load_loss_from_spec(args.loss_spec)
    cache_dir = Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()

    record = FinetuneRecord(
        name=args.name,
        boltz_version=str(_BOLTZ_VERSION),
        base_checkpoint_path=ckpt_path,
        base_checkpoint_sha256=ckpt_sha,
        config=FinetuneConfig(
            target_spec=args.target_spec,
            target_patterns=list(target_patterns),
            num_trainable_params=int(n_params),
            l2_sp_weight=args.l2_sp_weight,
        ),
        trained_params=sorted(trainable_names),
        parent=parent,
    )
    if prior_history:
        record.history.extend(prior_history)

    run = TrainingRun(
        started_at=time.time(),
        data_manifest=str(dataset.csv_path),
        data_sha256=dataset.sha256,
        data_rows=len(dataset),
        mode=args.mode,
        loss_spec=args.loss_spec,
        epochs=args.epochs,
        affinity_output=args.affinity_output,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        notes=args.notes,
        trainer_semantics="legacy" if args.legacy_trainer else "v2",
        target_modules=list(target_modules),
        **repo_state(),
    )
    grad_coverage = GradCoverage(list(selected))

    _es_best_loss: float = float("inf")
    _es_patience_counter: int = 0
    _es_best_state: Optional[dict[str, torch.Tensor]] = None
    _es_best_epoch: int = 0

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    ckpt_dir = registry.adapter_dir(args.name)

    # Guard: refuse to overwrite a *completed* fine-tune (meta.json present)
    # unless --overwrite was passed. Checked before creating the checkpoint dir
    # so the error is raised before any work is done. Mirrors train_lora; the
    # final registry.save below always uses overwrite=True because this trainer
    # itself owns ckpt_dir (it writes the per-epoch checkpoints into it).
    if not args.overwrite and (ckpt_dir / "meta.json").exists():
        msg = (
            f"Fine-tune '{args.name}' already exists at {ckpt_dir}. "
            "Use --overwrite or `boltz finetune update` to continue training."
        )
        raise FileExistsError(msg)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0

    if args.overwrite:
        for ckpt_file in _iter_checkpoint_paths(ckpt_dir):
            ckpt_file.unlink(missing_ok=True)
        logger.info("Removed existing checkpoints (--overwrite).")

    ckpt_path = _latest_checkpoint_path(ckpt_dir)
    _ckpt = _load_checkpoint(ckpt_path) if ckpt_path is not None else None
    if _ckpt is not None:
        logger.info(
            "Resuming full fine-tune from checkpoint: %d epoch(s) already completed.",
            _ckpt["epoch_completed"] + 1,
        )
        _ckpt_sd = _ckpt["model_state"]
        _live_sd = model.state_dict()
        with torch.no_grad():
            for _k, _v in _ckpt_sd.items():
                if _k in _live_sd:
                    _live_sd[_k].copy_(_v.to(_live_sd[_k].device, dtype=_live_sd[_k].dtype))
        try:
            optim.load_state_dict(_ckpt["optim_state"])
        except Exception as _oe:
            logger.warning("Could not restore optimizer state (%s) — optimizer reset.", _oe)
        run.metrics = _ckpt["metrics"]
        run.started_at = _ckpt.get("started_at", run.started_at)
        start_epoch = _ckpt["epoch_completed"] + 1
        _es_best_loss = _ckpt.get("es_best_loss", float("inf"))
        _es_patience_counter = _ckpt.get("es_patience_counter", 0)
        _es_best_state = _ckpt.get("es_best_state")
        _es_best_epoch = _ckpt.get("es_best_epoch", 0)
        run.grad_coverage = _ckpt.get("grad_coverage")
        logger.info(
            "Resumed: start_epoch=%d, es_best_loss=%.5f, es_patience=%d.",
            start_epoch, _es_best_loss, _es_patience_counter,
        )

    fwd_kwargs = dict(
        device=device, cache_dir=cache_dir, use_msa_server=args.use_msa_server,
        recycling_steps=args.recycling_steps, affinity_output=args.affinity_output,
    )
    for epoch in range(start_epoch, args.epochs):
        if _sampler is not None:
            _sampler.set_epoch(epoch)
        epoch_losses: list[float] = []
        for batch in loader:
            optim.zero_grad(set_to_none=True)
            fwd = _forward_batch(model, batch, **fwd_kwargs)
            if fwd is None:
                continue
            pred_combined, batch_for_loss, _ = fwd
            # ``call_loss`` accepts the LoRA adapter as a third arg purely
            # for metadata; we don't have one, so pass None — losses that
            # don't use it ignore it, and those that do (rare) will get a
            # clear AttributeError.
            loss = call_loss(loss_fn, pred_combined, batch_for_loss, None)
            loss = add_l2_sp_penalty(
                loss, args.l2_sp_weight, trainable_param_dict, initial_params,
            )
            loss.backward()
            if run.grad_coverage is None:
                grad_coverage.update()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in selected], args.gradient_clip,
                )
            optim.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        if run.grad_coverage is None and epoch_losses:
            run.grad_coverage = grad_coverage.summary()
            check_grad_coverage(
                run.grad_coverage, forward_modules=target_modules,
                legacy=args.legacy_trainer,
            )
        # Validation is scored on the task loss only (no L2-SP penalty), so it
        # is comparable with LoRA arms' val_loss.
        mean_val_loss: Optional[float] = (
            _run_validation(model, val_loader, loss_fn, None, **fwd_kwargs)
            if val_loader is not None else None
        )
        if mean_val_loss is not None:
            logger.info(
                "epoch %d / %d  mean_loss=%.5f  val_loss=%.5f",
                epoch + 1, args.epochs, mean_loss, mean_val_loss,
            )
            run.metrics.append({
                "epoch": float(epoch), "loss": mean_loss, "val_loss": mean_val_loss,
            })
        else:
            logger.info(
                "epoch %d / %d  mean_loss=%.5f",
                epoch + 1, args.epochs, mean_loss,
            )
            run.metrics.append({"epoch": float(epoch), "loss": mean_loss})

        _es_stop = False
        if args.early_stopping_patience > 0:
            # val_loss when a validation set is present, else train loss.
            _es_monitor = mean_val_loss if mean_val_loss is not None else mean_loss
            if _es_monitor < _es_best_loss - args.early_stopping_min_delta:
                _es_best_loss = _es_monitor
                _es_patience_counter = 0
                _es_best_epoch = epoch + 1
                _es_best_state = {
                    k: model.state_dict()[k].detach().cpu().clone()
                    for k in trainable_names
                }
                logger.info(
                    "Early stopping: new best loss=%.5f at epoch %d",
                    _es_best_loss, _es_best_epoch,
                )
            else:
                _es_patience_counter += 1
                logger.info(
                    "Early stopping: no improvement for %d/%d epochs "
                    "(best=%.5f at epoch %d)",
                    _es_patience_counter, args.early_stopping_patience,
                    _es_best_loss, _es_best_epoch,
                )
                if _es_patience_counter >= args.early_stopping_patience:
                    logger.info(
                        "Early stopping triggered at epoch %d/%d — "
                        "restoring best weights from epoch %d (loss=%.5f).",
                        epoch + 1, args.epochs, _es_best_epoch,
                        _es_best_loss,
                    )
                    _es_stop = True

        # ── Checkpoint (one file per completed epoch) ───────────────────────────
        ckpt_path = _checkpoint_path_for_epoch(ckpt_dir, epoch)
        _save_checkpoint(ckpt_path, {
            "epoch_completed": epoch,
            "model_state": {
                k: model.state_dict()[k].detach().cpu().clone()
                for k in trainable_names
                if k in model.state_dict()
            },
            "optim_state": optim.state_dict(),
            "metrics": run.metrics,
            "started_at": run.started_at,
            "es_best_loss": _es_best_loss,
            "es_patience_counter": _es_patience_counter,
            "es_best_epoch": _es_best_epoch,
            "es_best_state": _es_best_state,
            "grad_coverage": run.grad_coverage,
        })
        logger.debug("Checkpoint saved after epoch %d.", epoch + 1)

        # ── Live loss curve (updated after every epoch) ──────────────────────
        if args.plot_loss_curve:
            _plot_loss_curve(
                run.metrics,
                ckpt_dir / "loss_curve.png",
                title=f"Affinity Fine-tune Loss — {args.name}",
            )

        if _es_stop:
            break

    # Restore best-seen weights when early stopping is active.
    if args.early_stopping_patience > 0 and _es_best_state is not None:
        sd = model.state_dict()
        with torch.no_grad():
            for k, v in _es_best_state.items():
                if k in sd:
                    sd[k].copy_(v.to(sd[k].device, dtype=sd[k].dtype))

    run.finished_at = time.time()
    record.history.append(run)

    # Persist only the trained tensors (partial state-dict).
    full_state = model.state_dict()
    partial_state = {
        k: full_state[k].detach().cpu().clone()
        for k in trainable_names if k in full_state
    }
    # ckpt_dir already exists (created for checkpoints above), so always
    # overwrite here; the completion guard before training already rejected a
    # finished arm re-run without --overwrite. Mirrors train_lora.
    registry.save(record, partial_state, overwrite=True)

    return record


def prepare_update(
    name: str, *, registry: Optional[FinetuneRegistry] = None,
) -> tuple[FinetuneRecord, dict[str, torch.Tensor]]:
    """Load an existing fine-tune so it can be resumed by :func:`train_finetune`."""
    registry = registry or default_registry()
    return registry.load(name)


__all__ = ["FinetuneArgs", "prepare_update", "train_finetune"]
