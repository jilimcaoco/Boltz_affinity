"""Command-line interface for the LoRA subsystem.

Exposed under ``boltz lora …``. The group is registered from
:mod:`boltz.main` when this module imports cleanly.

Commands
--------
* ``train``    – train a fresh adapter from a CSV manifest.
* ``update``   – continue training an existing adapter on new data.
* ``list``     – list registered adapters.
* ``show``     – pretty-print an adapter's metadata.
* ``apply``    – sanity-check applying an adapter to a checkpoint (no inference).
* ``rm``       – remove an adapter from the registry.
* ``export``   – export a *merged* checkpoint (base + LoRA folded in).
* ``select``   – rank candidate ligands for active-learning rounds.
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


@click.group("lora")
def lora_cli() -> None:
    """Train, manage, and apply Boltz LoRA adapters."""


# ─── train ───────────────────────────────────────────────────────────────────


def _train_options(fn):
    """Shared Click options between ``train`` and ``update``."""
    decorators = [
        click.option("--csv", "csv_path", required=True,
                     type=click.Path(exists=True, dir_okay=False),
                     help="Training manifest. Columns: ligand, receptor, target, "
                          "structure (rescore mode), optional name."),
        click.option("--mode", default="rescore",
                     type=click.Choice(["rescore", "full"]),
                     help="rescore = trunk + affinity head (needs structure col); "
                          "full = entire Boltz pipeline (not yet wired)."),
        click.option("--loss", "loss_spec", default="mse",
                     help="Built-in (mse|mae|huber|bce|pairwise_ranking) or "
                          "'path/to/file.py:function_name'."),
        click.option("--rank", default=8, type=int, show_default=True),
        click.option("--alpha", default=16.0, type=float, show_default=True),
        click.option("--dropout", default=0.0, type=float, show_default=True),
        click.option("--target-spec", default="heads_pairformer",
                     show_default=True,
                     help="Preset name (affinity_heads|heads_pairformer), regex, "
                          "or comma-separated regex list."),
        click.option("--epochs", default=5, type=int, show_default=True),
        click.option("--learning-rate", "--lr", "learning_rate",
                     default=1e-4, type=float, show_default=True),
        click.option("--batch-size", default=1, type=int, show_default=True),
        click.option("--weight-decay", default=0.0, type=float, show_default=True),
        click.option("--gradient-clip", default=1.0, type=float, show_default=True),
        click.option("--recycling-steps", default=3, type=int, show_default=True),
        click.option("--device", default="auto",
                     type=click.Choice(["auto", "cuda", "cpu", "mps"])),
        click.option("--checkpoint", default=None, type=str,
                     help="Override path to base affinity checkpoint."),
        click.option("--use-msa-server", is_flag=True, default=False),
        click.option("--notes", default=None, type=str),
        click.option("--log-level", default="INFO"),
    ]
    for dec in reversed(decorators):
        fn = dec(fn)
    return fn


@lora_cli.command("train")
@click.option("--name", required=True, help="Adapter name (used as registry key).")
@click.option("--overwrite", is_flag=True, default=False,
              help="Replace an existing adapter with this name.")
@_train_options
def train_cmd(
    name: str,
    overwrite: bool,
    csv_path: str,
    mode: str,
    loss_spec: str,
    rank: int,
    alpha: float,
    dropout: float,
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
    log_level: str,
) -> None:
    """Train a fresh LoRA adapter."""
    _setup_logging(log_level)
    from boltz.lora.train import TrainArgs, train_lora

    args = TrainArgs(
        name=name, csv_path=csv_path, mode=mode, loss_spec=loss_spec,
        rank=rank, alpha=alpha, dropout=dropout, target_spec=target_spec,
        epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
        weight_decay=weight_decay, gradient_clip=gradient_clip,
        recycling_steps=recycling_steps, device=device, checkpoint=checkpoint,
        use_msa_server=use_msa_server, overwrite=overwrite, notes=notes,
    )
    adapter = train_lora(args)
    click.echo(f"Saved adapter '{adapter.name}' "
               f"(rank={adapter.config.rank}, layers={len(adapter.adapted_layers)}).")


# ─── update ──────────────────────────────────────────────────────────────────


@lora_cli.command("update")
@click.option("--name", required=True,
              help="Existing adapter to continue training.")
@click.option("--new-name", default=None,
              help="If given, save as a child adapter under this name "
                   "(default: overwrite original).")
@_train_options
def update_cmd(
    name: str,
    new_name: Optional[str],
    csv_path: str,
    mode: str,
    loss_spec: str,
    rank: int,
    alpha: float,
    dropout: float,
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
    log_level: str,
) -> None:
    """Continue training an existing adapter on new (or additional) data."""
    _setup_logging(log_level)
    from boltz.lora.registry import default_registry
    from boltz.lora.train import TrainArgs, prepare_update, train_lora

    registry = default_registry()
    parent_adapter, init_state = prepare_update(name, registry=registry)

    # Inherit structural config from the parent to keep injection compatible.
    args = TrainArgs(
        name=new_name or name,
        csv_path=csv_path,
        mode=mode,
        loss_spec=loss_spec,
        rank=parent_adapter.config.rank,
        alpha=parent_adapter.config.alpha,
        dropout=parent_adapter.config.dropout,
        target_spec=parent_adapter.config.target_spec,
        epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
        weight_decay=weight_decay, gradient_clip=gradient_clip,
        recycling_steps=recycling_steps, device=device, checkpoint=checkpoint,
        use_msa_server=use_msa_server,
        overwrite=(new_name is None),
        notes=notes,
    )
    adapter = train_lora(
        args,
        registry=registry,
        init_state=init_state,
        parent_adapter=parent_adapter.name,
        prior_history=list(parent_adapter.history),
    )
    click.echo(f"Updated adapter saved as '{adapter.name}' "
               f"(parent='{parent_adapter.name}', runs={len(adapter.history)}).")


# ─── list / show / rm ────────────────────────────────────────────────────────


@lora_cli.command("list")
@click.option("--json", "as_json", is_flag=True, default=False)
def list_cmd(as_json: bool) -> None:
    """List registered adapters."""
    from boltz.lora.registry import default_registry
    entries = default_registry().list()
    if as_json:
        click.echo(json.dumps(entries, indent=2, sort_keys=True))
        return
    if not entries:
        click.echo("(no adapters registered)")
        return
    click.echo(f"{'NAME':30s}  {'RANK':>4s}  {'LAYERS':>6s}  TARGETS")
    for e in entries:
        click.echo(
            f"{e['name']:30s}  {e.get('rank', '?'):>4}  "
            f"{e.get('num_adapted_layers', '?'):>6}  {e.get('target_spec', '?')}"
        )


@lora_cli.command("show")
@click.argument("name")
def show_cmd(name: str) -> None:
    """Pretty-print an adapter's metadata."""
    from boltz.lora.registry import default_registry
    registry = default_registry()
    adapter, _ = registry.load(name)
    click.echo(json.dumps(asdict(adapter), indent=2, sort_keys=True, default=str))


@lora_cli.command("rm")
@click.argument("name")
@click.option("--yes", is_flag=True, default=False,
              help="Skip interactive confirmation.")
def rm_cmd(name: str, yes: bool) -> None:
    """Remove an adapter from the registry."""
    from boltz.lora.registry import default_registry
    registry = default_registry()
    if not registry.exists(name):
        click.echo(f"Adapter '{name}' not found.", err=True)
        sys.exit(1)
    if not yes:
        click.confirm(f"Delete adapter '{name}'?", abort=True)
    registry.delete(name)
    click.echo(f"Removed '{name}'.")


# ─── apply / export ──────────────────────────────────────────────────────────


@lora_cli.command("apply")
@click.argument("name")
@click.option("--checkpoint", default=None,
              help="Optional checkpoint path (uses adapter's base by default). "
                   "This command performs a dry-run load to verify compatibility.")
def apply_cmd(name: str, checkpoint: Optional[str]) -> None:
    """Dry-run: verify an adapter loads cleanly into the affinity model."""
    from boltz.affinity_rescoring.model_manager import AffinityModelManager
    from boltz.lora.apply import load_adapter_into_model

    mgr = AffinityModelManager()
    model = mgr.load_model(checkpoint_path=checkpoint)
    adapter = load_adapter_into_model(model, name)
    click.echo(
        f"OK: applied '{adapter.name}' "
        f"(rank={adapter.config.rank}, layers={len(adapter.adapted_layers)})."
    )


@lora_cli.command("export")
@click.argument("name")
@click.option("--out", "out_path", required=True,
              type=click.Path(dir_okay=False),
              help="Output .ckpt path with LoRA weights merged into the base.")
@click.option("--checkpoint", default=None,
              help="Override base checkpoint path.")
def export_cmd(name: str, out_path: str, checkpoint: Optional[str]) -> None:
    """Export a merged checkpoint (base + LoRA folded into linear weights)."""
    import torch

    from boltz.affinity_rescoring.model_manager import AffinityModelManager
    from boltz.lora.apply import load_adapter_into_model

    mgr = AffinityModelManager()
    model = mgr.load_model(checkpoint_path=checkpoint)
    load_adapter_into_model(model, name, merge=True)

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, out)
    click.echo(f"Merged checkpoint written to {out}")


# ─── select (active learning) ────────────────────────────────────────────────


@lora_cli.command("select")
@click.option("--scores", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="CSV of screen results. Must contain columns: name, ligand "
                   "(SMILES), affinity_pred.")
@click.option("--out", "out_path", required=True,
              type=click.Path(dir_okay=False),
              help="Output CSV ranked for the next active-learning round.")
@click.option("--top-k", default=50, type=int, show_default=True,
              help="Number of candidates to select.")
@click.option("--strategy", default="hybrid",
              type=click.Choice(["uncertainty", "diversity", "hybrid", "topk"]),
              show_default=True)
@click.option("--uncertainty-col", default="affinity_pred_std",
              help="Column with per-row uncertainty (if absent, falls back to "
                   "abs(affinity_pred) for 'uncertainty' strategy).")
def select_cmd(
    scores: str,
    out_path: str,
    top_k: int,
    strategy: str,
    uncertainty_col: str,
) -> None:
    """Rank candidate ligands for the next active-learning iteration."""
    from boltz.lora.selection import select_candidates

    selected = select_candidates(
        scores_csv=scores,
        top_k=top_k,
        strategy=strategy,
        uncertainty_col=uncertainty_col,
    )
    selected.to_csv(out_path, index=False)
    click.echo(f"Selected {len(selected)} candidates -> {out_path}")


__all__ = ["lora_cli"]
