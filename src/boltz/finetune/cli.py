"""Command-line interface for the full fine-tune subsystem.

Exposed under ``boltz finetune …`` and registered from
:mod:`boltz.main`. Mirrors :mod:`boltz.lora.cli` so the two paths are
trivially comparable.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@click.group("finetune")
def finetune_cli() -> None:
    """Train, manage, and apply Boltz affinity-module fine-tunes.

    Full (non-LoRA) fine-tuning of the affinity stack. See
    ``docs/finetune_userguide.md`` for the design rationale and the
    head-to-head comparison protocol against ``boltz lora``.
    """


# ─── train / update shared options ──────────────────────────────────────────


def _train_options(fn):
    decorators = [
        click.option(
            "--csv", "csv_path", required=True,
            type=click.Path(exists=True, dir_okay=False),
            help="Training manifest. Same schema as `boltz lora train` "
                 "(ligand, receptor, target, structure, ...).",
        ),
        click.option(
            "--mode", default="rescore",
            type=click.Choice(["rescore", "full"]),
            help="rescore = trunk + affinity head with injected coords "
                 "(needs structure col); full = entire Boltz pipeline "
                 "(not yet wired).",
        ),
        click.option(
            "--loss", "loss_spec", default="mse",
            help="Built-in (mse|mae|huber|bce|pairwise_ranking|"
                 "intra_assay_huber|boltz2_affinity) or "
                 "'path/to/file.py:function_name'.",
        ),
        click.option(
            "--target-spec", default="affinity_module", show_default=True,
            help="Preset (affinity_module|affinity_heads|"
                 "affinity_pairformer|heads_pairformer), regex, or "
                 "comma-separated regex list. "
                 "Matched against model.named_parameters().",
        ),
        click.option("--epochs", default=5, type=int, show_default=True),
        click.option(
            "--learning-rate", "--lr", "learning_rate",
            default=1e-5, type=float, show_default=True,
            help="Default is 10x lower than LoRA — full fine-tuning is "
                 "much more sensitive to high LRs.",
        ),
        click.option("--batch-size", default=1, type=int, show_default=True),
        click.option(
            "--weight-decay", default=0.0, type=float, show_default=True,
        ),
        click.option(
            "--gradient-clip", default=1.0, type=float, show_default=True,
        ),
        click.option(
            "--recycling-steps", default=3, type=int, show_default=True,
        ),
        click.option(
            "--device", default="auto",
            type=click.Choice(["auto", "cuda", "cpu", "mps"]),
        ),
        click.option(
            "--checkpoint", default=None, type=str,
            help="Override path to base affinity checkpoint.",
        ),
        click.option(
            "--use-msa-server", is_flag=True, default=False,
            help="[DISABLED in this fork] Pre-compute MSAs and embed "
                 "their paths in the receptor YAML.",
        ),
        click.option("--notes", default=None, type=str),
        click.option(
            "--early-stopping-patience", default=5, type=int,
            show_default=True,
            help="Stop if mean loss does not improve by at least "
                 "--early-stopping-min-delta for this many consecutive "
                 "epochs. 0 = disabled.",
        ),
        click.option(
            "--early-stopping-min-delta", default=0.005, type=float,
            show_default=True,
        ),
        click.option(
            "--plot-loss-curve/--no-plot-loss-curve",
            default=True, show_default=True,
        ),
        click.option("--log-level", default="INFO"),
    ]
    for dec in reversed(decorators):
        fn = dec(fn)
    return fn


# ─── train ──────────────────────────────────────────────────────────────────


@finetune_cli.command("train")
@click.option("--name", required=True,
              help="Fine-tune name (registry key).")
@click.option("--overwrite", is_flag=True, default=False,
              help="Replace an existing fine-tune with this name.")
@_train_options
def train_cmd(
    name: str,
    overwrite: bool,
    csv_path: str,
    mode: str,
    loss_spec: str,
    target_spec: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    gradient_clip: float,
    recycling_steps: int,
    device: str,
    checkpoint: Optional[str],
    use_msa_server: bool,
    notes: Optional[str],
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    plot_loss_curve: bool,
    log_level: str,
) -> None:
    """Train a fresh affinity-module fine-tune."""
    _setup_logging(log_level)
    if use_msa_server:
        from boltz.affinity_rescoring.msa_cache import (
            raise_msa_server_disabled,
        )
        try:
            raise_msa_server_disabled()
        except Exception as exc:  # noqa: BLE001
            raise click.UsageError(str(exc)) from exc

    from boltz.finetune.train import FinetuneArgs, train_finetune

    args = FinetuneArgs(
        name=name, csv_path=csv_path, mode=mode, loss_spec=loss_spec,
        target_spec=target_spec, epochs=epochs,
        learning_rate=learning_rate, batch_size=batch_size,
        weight_decay=weight_decay, gradient_clip=gradient_clip,
        recycling_steps=recycling_steps, device=device,
        checkpoint=checkpoint, use_msa_server=use_msa_server,
        overwrite=overwrite, notes=notes,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        plot_loss_curve=plot_loss_curve,
    )
    record = train_finetune(args)
    click.echo(
        f"Saved fine-tune '{record.name}' "
        f"(target={record.config.target_spec}, "
        f"params={record.config.num_trainable_params/1e6:.2f}M)."
    )


# ─── update ─────────────────────────────────────────────────────────────────


@finetune_cli.command("update")
@click.option("--name", required=True,
              help="Existing fine-tune to continue training.")
@click.option("--new-name", default=None,
              help="If given, save as a child under this name "
                   "(default: overwrite original).")
@_train_options
def update_cmd(
    name: str,
    new_name: Optional[str],
    csv_path: str,
    mode: str,
    loss_spec: str,
    target_spec: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    gradient_clip: float,
    recycling_steps: int,
    device: str,
    checkpoint: Optional[str],
    use_msa_server: bool,
    notes: Optional[str],
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    plot_loss_curve: bool,
    log_level: str,
) -> None:
    """Continue training an existing fine-tune on new data."""
    _setup_logging(log_level)
    if use_msa_server:
        from boltz.affinity_rescoring.msa_cache import (
            raise_msa_server_disabled,
        )
        try:
            raise_msa_server_disabled()
        except Exception as exc:  # noqa: BLE001
            raise click.UsageError(str(exc)) from exc

    from boltz.finetune.registry import default_registry
    from boltz.finetune.train import (
        FinetuneArgs, prepare_update, train_finetune,
    )

    registry = default_registry()
    parent_record, init_state = prepare_update(name, registry=registry)

    args = FinetuneArgs(
        name=new_name or name,
        csv_path=csv_path, mode=mode, loss_spec=loss_spec,
        # Inherit target spec from parent so injection stays compatible.
        target_spec=parent_record.config.target_spec,
        epochs=epochs, learning_rate=learning_rate,
        batch_size=batch_size, weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        recycling_steps=recycling_steps, device=device,
        checkpoint=checkpoint, use_msa_server=use_msa_server,
        overwrite=(new_name is None), notes=notes,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        plot_loss_curve=plot_loss_curve,
    )
    record = train_finetune(
        args, registry=registry, init_state=init_state,
        parent=parent_record.name,
        prior_history=list(parent_record.history),
    )
    click.echo(
        f"Updated fine-tune saved as '{record.name}' "
        f"(parent='{parent_record.name}', runs={len(record.history)})."
    )


# ─── list / show / rm ───────────────────────────────────────────────────────


@finetune_cli.command("list")
@click.option("--json", "as_json", is_flag=True, default=False)
def list_cmd(as_json: bool) -> None:
    """List registered fine-tunes."""
    from boltz.finetune.registry import default_registry
    entries = default_registry().list()
    if as_json:
        click.echo(json.dumps(entries, indent=2, sort_keys=True))
        return
    if not entries:
        click.echo("(no fine-tunes registered)")
        return
    click.echo(f"{'NAME':30s}  {'PARAMS':>10s}  {'TENSORS':>7s}  TARGETS")
    for e in entries:
        click.echo(
            f"{e['name']:30s}  "
            f"{e.get('num_trainable_params', '?'):>10}  "
            f"{e.get('num_tensors', '?'):>7}  "
            f"{e.get('target_spec', '?')}"
        )


@finetune_cli.command("show")
@click.argument("name")
def show_cmd(name: str) -> None:
    """Pretty-print a fine-tune's metadata."""
    from boltz.finetune.registry import default_registry
    record, _ = default_registry().load(name)
    click.echo(json.dumps(asdict(record), indent=2, sort_keys=True, default=str))


@finetune_cli.command("rm")
@click.argument("name")
@click.option("--yes", is_flag=True, default=False,
              help="Skip interactive confirmation.")
def rm_cmd(name: str, yes: bool) -> None:
    """Remove a fine-tune from the registry."""
    from boltz.finetune.registry import default_registry
    registry = default_registry()
    if not registry.exists(name):
        click.echo(f"Fine-tune '{name}' not found.", err=True)
        sys.exit(1)
    if not yes:
        click.confirm(f"Delete fine-tune '{name}'?", abort=True)
    registry.delete(name)
    click.echo(f"Removed '{name}'.")


# ─── apply / export ─────────────────────────────────────────────────────────


@finetune_cli.command("apply")
@click.argument("name")
@click.option("--checkpoint", default=None,
              help="Optional override of the base checkpoint path.")
def apply_cmd(name: str, checkpoint: Optional[str]) -> None:
    """Dry-run: verify a fine-tune loads cleanly into the affinity model."""
    from boltz.affinity_rescoring.model_manager import AffinityModelManager
    from boltz.finetune.apply import load_finetune_into_model

    mgr = AffinityModelManager()
    model = mgr.load_model(checkpoint_path=checkpoint)
    record = load_finetune_into_model(model, name)
    click.echo(
        f"OK: applied '{record.name}' "
        f"(target={record.config.target_spec}, "
        f"params={record.config.num_trainable_params/1e6:.2f}M)."
    )


@finetune_cli.command("export")
@click.argument("name")
@click.option("--out", "out_path", required=True,
              type=click.Path(dir_okay=False),
              help="Output .ckpt path with fine-tuned weights merged.")
@click.option("--checkpoint", default=None,
              help="Override base checkpoint path.")
def export_cmd(name: str, out_path: str, checkpoint: Optional[str]) -> None:
    """Export a full checkpoint with fine-tuned weights applied."""
    import torch

    from boltz.affinity_rescoring.model_manager import AffinityModelManager
    from boltz.finetune.apply import load_finetune_into_model

    mgr = AffinityModelManager()
    model = mgr.load_model(checkpoint_path=checkpoint)
    load_finetune_into_model(model, name)

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, out)
    click.echo(f"Merged checkpoint written to {out}")


__all__ = ["finetune_cli"]
