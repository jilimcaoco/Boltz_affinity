"""LoRA finetuning trainer for the Boltz affinity stack.

Two training modes:

* ``rescore`` (default, recommended): trunk + affinity head with user-provided
  complex coordinates. Mirrors :func:`boltz.affinity_rescoring.inference.run_direct_affinity_inference`
  but built as a *differentiable* training step. Each CSV row must include a
  ``structure`` column.
* ``full``: the full Boltz forward (trunk + diffusion + affinity). This is
  significantly more expensive and currently routes through the existing
  ``scripts/train`` Lightning machinery; we expose a clear hook but raise a
  ``NotImplementedError`` until that integration lands (tracked in the user
  guide).

Provenance: every successful run appends a :class:`TrainingRun` record to the
adapter's ``meta.json`` via :class:`LoRARegistry`.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from boltz import __version__ as _BOLTZ_VERSION  # type: ignore[attr-defined]
from boltz.lora.adapter import LoRAAdapter, LoRAConfig, TrainingRun
from boltz.lora.data import LoRADataset, lora_collate
from boltz.lora.inject import apply_lora, load_lora_state_dict, lora_state_dict
from boltz.lora.losses import LossFn, call_loss, load_loss_from_spec
from boltz.lora.registry import LoRARegistry, default_registry, hash_file
from boltz.lora.targets import resolve_targets

logger = logging.getLogger(__name__)


@dataclass
class TrainArgs:
    """User-facing knobs surfaced via the CLI."""

    name: str
    csv_path: str
    mode: str = "rescore"
    loss_spec: str = "mse"
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_spec: str = "heads_pairformer"
    epochs: int = 5
    learning_rate: float = 1e-4
    batch_size: int = 1
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    recycling_steps: int = 3
    device: str = "auto"
    checkpoint: Optional[str] = None
    use_msa_server: bool = False
    overwrite: bool = False
    notes: Optional[str] = None


def _pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ── Featurisation helpers ────────────────────────────────────────────────────


def _resolve_yaml(receptor: str, work_dir: Path) -> Path:
    """For v1 LoRA training, the ``receptor`` column must be a Boltz YAML.

    The training CSV mirrors the input format of ``boltz predict`` so users
    can reuse the YAMLs they already author. Auto-conversion from a raw PDB
    receptor + SMILES ligand is intentionally out of scope for v1; use the
    ``boltz rescore`` family to materialise these YAMLs first if needed.
    """
    p = Path(receptor).expanduser()
    if not p.exists():
        msg = f"Receptor YAML not found: {p}"
        raise FileNotFoundError(msg)
    if p.suffix.lower() not in {".yaml", ".yml"}:
        msg = (
            f"LoRA training expects 'receptor' to be a Boltz YAML "
            f"(got {p.suffix!r}). See docs/lora_userguide.md."
        )
        raise ValueError(msg)
    return p


def _featurize_row(
    row: dict[str, Any],
    *,
    cache_dir: Path,
    use_msa_server: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Build a feature batch dict for one training row.

    Reuses the affinity_rescoring pipeline up to (but not including) the
    forward pass. The returned dict has all tensors moved to ``device``.
    """
    from boltz.affinity_rescoring.coord_injection import (
        build_chain_id_map,
        inject_pdb_coords_into_structure,
        save_pre_affinity_structure,
    )
    from boltz.affinity_rescoring.parsers import parse_structure_file
    from boltz.data import const
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals, load_molecules
    from boltz.data.module.inferencev2 import load_input
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Record, StructureV2
    from boltz.main import process_input

    import numpy as np

    receptor = row["receptor"][0] if isinstance(row["receptor"], list) else row["receptor"]
    ligand = row["ligand"][0] if isinstance(row["ligand"], list) else row["ligand"]
    structure = row["structure"][0] if isinstance(row["structure"], list) else row["structure"]
    del ligand  # YAML already references the ligand; kept for future modes

    work_dir = Path(tempfile.mkdtemp(prefix="boltz_lora_row_"))
    try:
        yaml_path = _resolve_yaml(receptor, work_dir)

        out_dir = work_dir / "output"
        msa_dir = out_dir / "msa"
        records_dir = out_dir / "processed" / "records"
        structure_dir = out_dir / "processed" / "structures"
        processed_msa_dir = out_dir / "processed" / "msa"
        processed_constraints_dir = out_dir / "processed" / "constraints"
        processed_templates_dir = out_dir / "processed" / "templates"
        processed_mols_dir = out_dir / "processed" / "mols"
        predictions_dir = out_dir / "predictions"
        for d in [
            out_dir, msa_dir, records_dir, structure_dir,
            processed_msa_dir, processed_constraints_dir,
            processed_templates_dir, processed_mols_dir, predictions_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        mol_dir = cache_dir / "mols"
        ccd = load_canonicals(mol_dir)
        process_input(
            path=yaml_path, ccd=ccd, msa_dir=msa_dir, mol_dir=mol_dir,
            boltz2=True, use_msa_server=use_msa_server,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="paired+unpaired",
            msa_server_username=None, msa_server_password=None,
            api_key_header=None, api_key_value=None, max_msa_seqs=8192,
            processed_msa_dir=processed_msa_dir,
            processed_constraints_dir=processed_constraints_dir,
            processed_templates_dir=processed_templates_dir,
            processed_mols_dir=processed_mols_dir,
            structure_dir=structure_dir, records_dir=records_dir,
        )

        record = Record.load(next(records_dir.glob("*.json")))
        proc = StructureV2.load(structure_dir / f"{record.id}.npz")

        pdb_atoms, _meta = parse_structure_file(structure)
        chain_ids_in_pdb = sorted({a.chain_id for a in pdb_atoms})
        asym_map = build_chain_id_map(proc, chain_ids_in_pdb)
        full_map = {cid: asym_map[cid] for cid in chain_ids_in_pdb if cid in asym_map}
        injected, _ = inject_pdb_coords_into_structure(proc, pdb_atoms, full_map)
        save_pre_affinity_structure(injected, predictions_dir, record.id)

        tokenizer = Boltz2Tokenizer()
        cropper = AffinityCropper()
        featurizer = Boltz2Featurizer()

        input_data = load_input(
            record=record, target_dir=predictions_dir, msa_dir=processed_msa_dir,
            constraints_dir=processed_constraints_dir,
            template_dir=processed_templates_dir,
            extra_mols_dir=processed_mols_dir, affinity=True,
        )
        tokenized = tokenizer.tokenize(input_data)
        tokenized = cropper.crop(tokenized, max_tokens=256, max_atoms=2048)

        molecules = dict(ccd)
        if input_data.extra_mols:
            molecules.update(input_data.extra_mols)
        needed = set(tokenized.tokens["res_name"].tolist()) - set(molecules.keys())
        molecules.update(load_molecules(mol_dir, needed))

        random = np.random.default_rng(42)
        features = featurizer.process(
            tokenized, molecules=molecules, random=random,
            training=False, max_atoms=None, max_tokens=None,
            max_seqs=const.max_msa_seqs, pad_to_max_seqs=False,
            single_sequence_prop=0.0, compute_frames=True,
            inference_pocket_constraints=None,
            inference_contact_constraints=None,
            compute_constraint_features=True, override_method=None,
            compute_affinity=True,
        )

        batch: dict[str, Any] = {}
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(device)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            elif k == "affinity_mw":
                batch[k] = [v]
            else:
                batch[k] = v
        return batch
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


# ── Differentiable trunk + affinity ──────────────────────────────────────────


def _affinity_forward_trainable(
    model: Any,
    feats: dict[str, Any],
    *,
    recycling_steps: int,
) -> dict[str, torch.Tensor]:
    """Same compute graph as ``affinity_forward`` but allowing autograd.

    Only the LoRA parameters carry ``requires_grad=True``; everything else is
    frozen, so the backward graph through the trunk is effectively a no-op for
    parameter updates but is still required to flow gradients back to adapters
    when targets include trunk-side pairformer linears.
    """
    if "coords" not in feats:
        raise RuntimeError("LoRA training requires injected 'coords' in features.")

    def _unwrap(attr: str) -> Any:
        mod = getattr(model, attr)
        return mod._orig_mod if hasattr(mod, "_orig_mod") else mod  # noqa: SLF001

    s_inputs = model.input_embedder(feats)
    s_init = model.s_init(s_inputs)
    z_init = (
        model.z_init_1(s_inputs)[:, :, None]
        + model.z_init_2(s_inputs)[:, None, :]
        + model.rel_pos(feats)
        + model.token_bonds(feats["token_bonds"].float())
    )
    if model.bond_type_feature:
        z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
    z_init = z_init + model.contact_conditioning(feats)

    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    use_kernels = getattr(model, "use_kernels", False)
    msa_module = _unwrap("msa_module")
    pairformer_module = _unwrap("pairformer_module")

    for _ in range(recycling_steps + 1):
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z_init + model.z_recycle(model.z_norm(z))
        if getattr(model, "use_templates", False):
            template_module = _unwrap("template_module")
            z = z + template_module(z, feats, pair_mask, use_kernels=use_kernels)
        z = z + msa_module(z, s_inputs, feats, use_kernels=use_kernels)
        s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask, use_kernels=use_kernels)

    pad_token_mask = feats["token_pad_mask"][0]
    rec_mask = (feats["mol_type"][0] == 0) * pad_token_mask
    lig_mask = feats["affinity_token_mask"][0].to(torch.bool) * pad_token_mask
    cross_pair_mask = (
        lig_mask[:, None] * rec_mask[None, :]
        + rec_mask[:, None] * lig_mask[None, :]
        + lig_mask[:, None] * lig_mask[None, :]
    )
    z_affinity = z * cross_pair_mask[None, :, :, None]

    coords_affinity = feats["coords"]
    if coords_affinity.dim() == 3:
        coords_affinity = coords_affinity[None]
    elif coords_affinity.dim() == 4 and coords_affinity.shape[1] > 1:
        coords_affinity = coords_affinity[:, :1]

    s_inputs = model.input_embedder(feats, affinity=True)

    affinity_attr = "affinity_module1" if getattr(model, "affinity_ensemble", False) else "affinity_module"
    affinity_module = _unwrap(affinity_attr)
    out = affinity_module(
        s_inputs=s_inputs, z=z_affinity, x_pred=coords_affinity,
        feats=feats, multiplicity=1, use_kernels=use_kernels,
    )
    return out


# ── Trainer ──────────────────────────────────────────────────────────────────


def _load_base_model(checkpoint: Optional[str], device: torch.device) -> tuple[Any, str, str]:
    """Load the Boltz2 affinity checkpoint via the existing manager."""
    from boltz.affinity_rescoring.inference import AffinityModelManager

    mgr = AffinityModelManager(device=str(device))
    model = mgr.load_model(checkpoint_path=checkpoint or "auto")
    ckpt_path = str(getattr(mgr, "_checkpoint_path", checkpoint or ""))
    ckpt_sha = str(getattr(mgr, "_checkpoint_sha256", ""))
    return model, ckpt_path, ckpt_sha


def train_lora(
    args: TrainArgs,
    *,
    registry: Optional[LoRARegistry] = None,
    init_state: Optional[dict[str, torch.Tensor]] = None,
    parent_adapter: Optional[str] = None,
    prior_history: Optional[list[TrainingRun]] = None,
) -> LoRAAdapter:
    """Run a LoRA training job end-to-end and persist the adapter.

    Used by both ``boltz lora train`` (fresh adapter) and ``boltz lora update``
    (after :func:`prepare_update`).

    Parameters
    ----------
    init_state:
        Optional LoRA state dict to load *after* injection. Used by ``update``
        to resume from an existing adapter's weights.
    parent_adapter:
        Name of the adapter this run derives from (recorded for provenance).
    prior_history:
        Previously-recorded :class:`TrainingRun` entries to preserve on the
        new adapter's history.
    """
    if args.mode == "full":
        msg = (
            "mode='full' (full Boltz pipeline incl. diffusion) is not wired up "
            "yet. Use mode='rescore' with a 'structure' column. The integration "
            "hook lives in scripts/train/train.py."
        )
        raise NotImplementedError(msg)

    if args.mode != "rescore":
        msg = f"Unknown mode {args.mode!r}; expected 'rescore' or 'full'."
        raise ValueError(msg)

    registry = registry or default_registry()
    device = _pick_device(args.device)

    dataset = LoRADataset(args.csv_path, mode=args.mode)
    loader = DataLoader(
        dataset, batch_size=max(1, args.batch_size), shuffle=True,
        collate_fn=lora_collate, num_workers=0,
    )

    model, ckpt_path, ckpt_sha = _load_base_model(args.checkpoint, device)
    model.train()  # enables grad; LoRA params are the only trainable ones

    target_patterns = resolve_targets(args.target_spec)
    adapted = apply_lora(
        model, target_patterns,
        r=args.rank, alpha=args.alpha, dropout=args.dropout, freeze_base=True,
    )
    model = model.to(device)
    if init_state is not None:
        load_lora_state_dict(model, init_state, strict=False)
        logger.info("Loaded initial LoRA state (%d tensors).", len(init_state))
    logger.info("LoRA adapted %d layers: e.g. %s", len(adapted), adapted[:3])

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        msg = "No trainable parameters after LoRA injection."
        raise RuntimeError(msg)
    optim = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    loss_fn: LossFn = load_loss_from_spec(args.loss_spec)
    cache_dir = Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()

    adapter = LoRAAdapter(
        name=args.name,
        boltz_version=str(_BOLTZ_VERSION),
        base_checkpoint_path=ckpt_path,
        base_checkpoint_sha256=ckpt_sha,
        config=LoRAConfig(
            rank=args.rank, alpha=args.alpha, dropout=args.dropout,
            target_spec=args.target_spec, target_patterns=list(target_patterns),
        ),
        adapted_layers=adapted,
        parent_adapter=parent_adapter,
    )
    if prior_history:
        adapter.history.extend(prior_history)
    run = TrainingRun(
        started_at=time.time(), data_manifest=str(dataset.csv_path),
        data_sha256=dataset.sha256, data_rows=len(dataset), mode=args.mode,
        loss_spec=args.loss_spec, epochs=args.epochs,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        notes=args.notes,
    )

    for epoch in range(args.epochs):
        epoch_losses: list[float] = []
        for batch in loader:
            target = batch["target"].to(device)
            feats = _featurize_row(
                batch, cache_dir=cache_dir,
                use_msa_server=args.use_msa_server, device=device,
            )
            pred = _affinity_forward_trainable(
                model, feats, recycling_steps=args.recycling_steps,
            )
            batch_for_loss = {"target": target, **batch}
            loss = call_loss(loss_fn, pred, batch_for_loss, adapter)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optim.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        logger.info("epoch %d / %d  mean_loss=%.5f", epoch + 1, args.epochs, mean_loss)
        run.metrics.append({"epoch": float(epoch), "loss": mean_loss})

    run.finished_at = time.time()
    adapter.history.append(run)

    state = lora_state_dict(model)
    registry.save(adapter, state, overwrite=args.overwrite)
    return adapter


def prepare_update(
    name: str, *, registry: Optional[LoRARegistry] = None,
) -> tuple[LoRAAdapter, dict[str, torch.Tensor]]:
    """Load an existing adapter so it can be resumed by :func:`train_lora`."""
    registry = registry or default_registry()
    return registry.load(name)


__all__ = ["TrainArgs", "prepare_update", "train_lora"]
