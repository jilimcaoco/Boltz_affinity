"""Full fine-tuning trainer for the Boltz affinity stack.

Twin of :mod:`boltz.lora.train` — same CSV manifest, same losses, same
``mode='rescore'`` differentiable forward — but instead of injecting LoRA
adapters and freezing the base, we directly mark a subset of the *base*
parameters as trainable and let AdamW update them.

We deliberately reuse :func:`boltz.lora.train._featurize_row` and
:func:`boltz.lora.train._affinity_forward_trainable` so the two training
modes share an identical compute graph.  Anything that improves
featurisation or the forward pass automatically benefits both.

Saved artefact: a *partial* state-dict (only the trained tensors) plus
provenance metadata.  Re-applying it during inference is a
``model.load_state_dict(..., strict=False)``-style copy through
:func:`boltz.finetune.apply.load_finetune_into_model`.
"""

from __future__ import annotations

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
from boltz.finetune.registry import FinetuneRegistry, default_registry
from boltz.finetune.targets import resolve_targets

# Reuse the LoRA-side machinery — these helpers are mode-agnostic.
from boltz.lora.data import AssayGroupedSampler, LoRADataset, lora_collate
from boltz.lora.losses import LossFn, call_loss, load_loss_from_spec
from boltz.lora.train import (
    _affinity_forward_trainable,
    _extract_row_from_batch,
    _featurize_row,
    _pick_device,
    _plot_loss_curve,
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


def _load_base_model(
    checkpoint: Optional[str], device: torch.device
) -> tuple[Any, str, str]:
    """Load the Boltz2 affinity checkpoint via the existing manager."""
    from boltz.affinity_rescoring.inference import AffinityModelManager

    mgr = AffinityModelManager(device=str(device))
    model = mgr.load_model(checkpoint_path=checkpoint or "auto")
    ckpt_path = str(getattr(mgr, "_checkpoint_path", checkpoint or ""))
    ckpt_sha = str(getattr(mgr, "_checkpoint_sha256", ""))
    return model, ckpt_path, ckpt_sha


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

    registry = registry or default_registry()
    device = _pick_device(args.device)

    dataset = LoRADataset(args.csv_path, mode=args.mode)

    has_groups = any(r.group_id for r in dataset.rows)
    _sampler: Optional[AssayGroupedSampler] = None
    if args.batch_size > 1 and has_groups:
        _sampler = AssayGroupedSampler(
            dataset.rows, batch_size=args.batch_size, shuffle=True,
        )
        loader = DataLoader(
            dataset, batch_sampler=_sampler,
            collate_fn=lora_collate, num_workers=0,
        )
        logger.info(
            "AssayGroupedSampler: %d assay groups, %d ungrouped rows.",
            _sampler.n_assay_groups, _sampler.n_ungrouped,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=max(1, args.batch_size), shuffle=True,
            collate_fn=lora_collate, num_workers=0,
        )

    model, ckpt_path, ckpt_sha = _load_base_model(args.checkpoint, device)
    model.train()

    target_patterns = resolve_targets(args.target_spec)
    selected = _select_trainable_params(model, target_patterns)
    trainable_names = {n for n, _ in selected}
    _freeze_all_then_unfreeze(model, trainable_names)
    model = model.to(device)

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
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        notes=args.notes,
    )

    _es_best_loss: float = float("inf")
    _es_patience_counter: int = 0
    _es_best_state: Optional[dict[str, torch.Tensor]] = None
    _es_best_epoch: int = 0

    for epoch in range(args.epochs):
        if _sampler is not None:
            _sampler.set_epoch(epoch)
        epoch_losses: list[float] = []
        for batch in loader:
            target = batch["target"].to(device)
            n_rows = (
                len(batch["name"]) if isinstance(batch["name"], list) else 1
            )
            optim.zero_grad(set_to_none=True)

            if n_rows == 1:
                try:
                    feats = _featurize_row(
                        batch, cache_dir=cache_dir,
                        use_msa_server=args.use_msa_server, device=device,
                    )
                except Exception as exc:  # noqa: BLE001
                    name = (
                        batch["name"][0]
                        if isinstance(batch["name"], list) else batch["name"]
                    )
                    logger.warning(
                        "Skipping row %r: featurization failed (%s)",
                        name, exc,
                    )
                    continue
                pred = _affinity_forward_trainable(
                    model, feats, recycling_steps=args.recycling_steps,
                )
                pred_combined = pred
            else:
                all_pred_values: list[torch.Tensor] = []
                all_logits_binary: list[torch.Tensor] = []
                kept_indices: list[int] = []
                for i in range(n_rows):
                    row_i = _extract_row_from_batch(batch, i)
                    try:
                        feats_i = _featurize_row(
                            row_i, cache_dir=cache_dir,
                            use_msa_server=args.use_msa_server, device=device,
                        )
                    except Exception as exc:  # noqa: BLE001
                        name = (
                            row_i["name"][0]
                            if isinstance(row_i["name"], list)
                            else row_i["name"]
                        )
                        logger.warning(
                            "Skipping row %r: featurization failed (%s)",
                            name, exc,
                        )
                        continue
                    pred_i = _affinity_forward_trainable(
                        model, feats_i,
                        recycling_steps=args.recycling_steps,
                    )
                    all_pred_values.append(pred_i["affinity_pred_value"])
                    if "affinity_logits_binary" in pred_i:
                        all_logits_binary.append(
                            pred_i["affinity_logits_binary"]
                        )
                    kept_indices.append(i)
                if not all_pred_values:
                    logger.warning(
                        "Skipping batch: no rows survived featurization"
                    )
                    continue
                if len(kept_indices) != n_rows:
                    keep_t = torch.tensor(
                        kept_indices, device=target.device, dtype=torch.long,
                    )
                    target = target.index_select(0, keep_t)
                    if isinstance(batch.get("target"), torch.Tensor):
                        batch["target"] = batch["target"].index_select(
                            0, keep_t.to(batch["target"].device)
                        )
                    for k in ("group_id", "is_binder", "name"):
                        v = batch.get(k)
                        if isinstance(v, list):
                            batch[k] = [v[i] for i in kept_indices]
                pred_combined = {
                    "affinity_pred_value": torch.cat(all_pred_values, dim=0),
                }
                if (
                    all_logits_binary
                    and len(all_logits_binary) == len(kept_indices)
                ):
                    pred_combined["affinity_logits_binary"] = torch.cat(
                        all_logits_binary, dim=0,
                    )

            batch_for_loss = {**batch, "target": target}
            # ``call_loss`` accepts the LoRA adapter as a third arg purely
            # for metadata; we don't have one, so pass None — losses that
            # don't use it ignore it, and those that do (rare) will get a
            # clear AttributeError.
            loss = call_loss(loss_fn, pred_combined, batch_for_loss, None)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in selected], args.gradient_clip,
                )
            optim.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        logger.info(
            "epoch %d / %d  mean_loss=%.5f",
            epoch + 1, args.epochs, mean_loss,
        )
        run.metrics.append({"epoch": float(epoch), "loss": mean_loss})

        if args.early_stopping_patience > 0:
            if mean_loss < _es_best_loss - args.early_stopping_min_delta:
                _es_best_loss = mean_loss
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
    registry.save(record, partial_state, overwrite=args.overwrite)

    if args.plot_loss_curve and run.metrics:
        _plot_loss_curve(
            run.metrics,
            registry.adapter_dir(args.name) / "loss_curve.png",
            title=f"Affinity Fine-tune Loss — {args.name}",
        )

    return record


def prepare_update(
    name: str, *, registry: Optional[FinetuneRegistry] = None,
) -> tuple[FinetuneRecord, dict[str, torch.Tensor]]:
    """Load an existing fine-tune so it can be resumed by :func:`train_finetune`."""
    registry = registry or default_registry()
    return registry.load(name)


__all__ = ["FinetuneArgs", "prepare_update", "train_finetune"]
