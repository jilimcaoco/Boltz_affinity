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

import hashlib
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
from boltz.lora.data import AssayGroupedSampler, LoRADataset, lora_collate
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
    # Early stopping: halt if loss does not improve by at least min_delta for
    # `patience` consecutive epochs. Set patience=0 to disable.
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.005
    # Loss curve: save loss_curve.png to the adapter directory after training.
    plot_loss_curve: bool = True
    # Validation: path to a separate validation CSV, or fraction [0,1) of
    # training rows to hold out.  When both are None/0 no validation is run.
    val_csv_path: Optional[str] = None
    val_split: float = 0.0


def _pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _plot_loss_curve(
    metrics: list[dict],  # must contain at least "epoch" and "loss" keys
    output_path: Path,
    *,
    title: str = "LoRA Training Loss",
) -> None:
    """Save a train (+optional val) loss-vs-epoch PNG to *output_path*.

    Requires ``matplotlib``. If it is not installed a warning is logged and
    the function returns silently so training is never blocked.
    When metrics dicts contain a ``val_loss`` key the validation curve is
    plotted on the same axes alongside the training curve.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive; safe on SLURM / headless nodes
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed — skipping loss curve plot. "
            "Install it with: pip install matplotlib"
        )
        return

    epochs = [int(m["epoch"]) + 1 for m in metrics]
    losses = [m["loss"] for m in metrics]
    val_losses = [m.get("val_loss") for m in metrics]
    has_val = any(v is not None for v in val_losses)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        epochs, losses,
        marker="o", linewidth=1.8, markersize=5, color="#2979ff", label="train loss",
    )

    if has_val:
        val_y = [v if v is not None else float("nan") for v in val_losses]
        ax.plot(
            epochs, val_y,
            marker="s", linewidth=1.8, markersize=5, color="#e65100",
            linestyle="--", label="val loss",
        )
        # Mark best val epoch
        finite_val = [(i, v) for i, v in enumerate(val_y) if not (v != v)]  # filter nan
        if finite_val:
            best_vi, best_vv = min(finite_val, key=lambda x: x[1])
            ax.axvline(x=epochs[best_vi], color="#e65100", linestyle=":", linewidth=1, alpha=0.7)
            ax.annotate(
                f"val best: {best_vv:.4f}\n(epoch {epochs[best_vi]})",
                xy=(epochs[best_vi], best_vv),
                xytext=(8, -18),
                textcoords="offset points",
                fontsize=8,
                color="#e65100",
            )
    else:
        # Annotate the best (minimum) training epoch
        min_idx = losses.index(min(losses))
        ax.axvline(x=epochs[min_idx], color="#d32f2f", linestyle="--", linewidth=1, alpha=0.7)
        ax.annotate(
            f"best: {losses[min_idx]:.4f}\n(epoch {epochs[min_idx]})",
            xy=(epochs[min_idx], losses[min_idx]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8,
            color="#d32f2f",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Loss curve saved to %s", output_path)


# ── Checkpoint helpers ───────────────────────────────────────────────────────

_CKPT_PREFIX = "checkpoint_epoch_"
_CKPT_SUFFIX = ".pt"


def _checkpoint_path_for_epoch(directory: Path, epoch_completed: int) -> Path:
    return directory / f"{_CKPT_PREFIX}{epoch_completed + 1:04d}{_CKPT_SUFFIX}"


def _iter_checkpoint_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob(f"{_CKPT_PREFIX}*{_CKPT_SUFFIX}"))


def _latest_checkpoint_path(directory: Path) -> Optional[Path]:
    candidates = _iter_checkpoint_paths(directory)
    return candidates[-1] if candidates else None


def _save_checkpoint(path: Path, data: dict) -> None:
    """Atomically write a torch-pickled checkpoint at *path*.

    Uses a sibling temp file + POSIX rename so that a crash mid-write never
    leaves a corrupt checkpoint. Safe on all POSIX filesystems (SLURM scratch,
    NFS turbo volumes, etc.).
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(data, tmp)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _load_checkpoint(path: Path) -> Optional[dict]:
    """Load a checkpoint saved by :func:`_save_checkpoint`.

    Returns ``None`` (and logs a warning) when the file is absent or corrupt so
    that callers can fall back to a clean training start without crashing.
    """
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # noqa: S614
    except Exception as exc:
        logger.warning(
            "Could not load checkpoint %s (%s) — starting from scratch.", path, exc
        )
        return None


def _extract_row_from_batch(batch: dict[str, Any], i: int) -> dict[str, Any]:
    """Slice row *i* from a collated batch dict produced by :func:`lora_collate`.

    Lists are sliced to a one-element list so that :func:`_featurize_row`'s
    existing ``isinstance(v, list) → v[0]`` guards work unchanged.
    Tensors are sliced on dim-0.  All other values are passed through.
    """
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, list):
            out[k] = [v[i]]
        elif isinstance(v, torch.Tensor) and v.dim() >= 1:
            out[k] = v[i : i + 1]
        else:
            out[k] = v
    return out


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


def _next_chain_id(used: set[str]) -> str:
    for c in "BCDEFGHIJKLMNOPQRSTUVWXYZ":
        if c not in used:
            return c
    msg = "Ran out of single-letter chain IDs while assembling LoRA YAML"
    raise RuntimeError(msg)


def _materialize_row_yaml(
    receptor_yaml: Path,
    ligand_smiles: str,
    work_dir: Path,
) -> Path:
    """Combine a receptor YAML with a ligand SMILES into a per-row YAML.

    The training receptor YAMLs only contain the protein chain(s) (so they
    can be reused across many ligands). The Boltz-2 affinity pipeline
    additionally needs (a) the ligand as a chain entry and (b) a
    ``properties.affinity.binder`` declaration pointing at that chain so the
    tokenizer sets ``affinity_mask`` and ``AffinityCropper`` can locate the
    ligand. We append those here without mutating the original file.

    Any protein chain in the receptor YAML that does not already have an
    ``msa:`` entry is resolved against the shared MSA cache
    (:func:`boltz.affinity_rescoring.msa_cache.find_msa`) so LoRA training
    never needs ``--use-msa-server``.
    """
    import copy
    import yaml

    from boltz.affinity_rescoring.msa_cache import find_msa

    data = yaml.safe_load(receptor_yaml.read_text()) or {}
    sequences = list(data.get("sequences") or [])

    used_ids: set[str] = set()
    has_ligand = False
    binder_chain: Optional[str] = None
    for entry in sequences:
        for kind, body in entry.items():
            cid = body.get("id")
            if isinstance(cid, str):
                used_ids.add(cid)
            elif isinstance(cid, list):
                used_ids.update(cid)
            if kind == "ligand":
                has_ligand = True
                if binder_chain is None and isinstance(cid, str):
                    binder_chain = cid

    if not has_ligand:
        binder_chain = _next_chain_id(used_ids)
        sequences.append({
            "ligand": {"id": binder_chain, "smiles": ligand_smiles},
        })

    new_data = copy.deepcopy(data)
    new_data["sequences"] = sequences

    # Auto-fill missing MSA paths from the shared cache so we never need
    # --use-msa-server. Receptor YAMLs that already declare ``msa:`` are
    # left untouched.
    target_hint = receptor_yaml.stem
    for entry in new_data["sequences"]:
        body = entry.get("protein")
        if not body or body.get("msa"):
            continue
        seq = body.get("sequence")
        cid = body.get("id")
        if isinstance(cid, list):
            cid = cid[0] if cid else None
        hit = find_msa(
            sequence=seq,
            chain_id=str(cid) if cid is not None else None,
            target=target_hint,
        )
        if hit is not None:
            body["msa"] = str(hit)

    properties = list(new_data.get("properties") or [])
    has_affinity = any("affinity" in p for p in properties)
    if not has_affinity and binder_chain is not None:
        properties.append({"affinity": {"binder": binder_chain}})
        new_data["properties"] = properties

    out = work_dir / f"{receptor_yaml.stem}_with_ligand.yaml"
    out.write_text(yaml.safe_dump(new_data, sort_keys=False))
    return out


# ── Frozen-trunk feature / z cache (opt-in via $BOLTZ_TRAIN_CACHE_DIR) ────────
#
# rescore-mode training re-featurises every row AND re-runs the frozen Boltz
# trunk (MSA + pairformer, `recycling_steps+1` passes) on EVERY epoch — even
# though, when the trainable parameters live entirely inside the affinity
# module (target-spec affinity_module / affinity_heads / affinity_pairformer /
# heads_pairformer), both the features and the trunk output ``z`` are constant
# across epochs and across concurrently-training arms that share a base
# checkpoint. This cache memoises both to disk so each pose is featurised +
# trunk-forwarded ONCE, then reused by every later epoch and every other arm.
#
# Disabled by default (env unset → behaviour byte-identical to before). When
# enabled AND the trunk is frozen, the trunk is evaluated deterministically
# (eval mode + no_grad), so the cached ``z`` is exact and reusable; this drops
# frozen-trunk dropout, the intended semantics for a frozen feature extractor.

FEAT_CACHE_VERSION = "feat_v1"
Z_CACHE_VERSION = "z_v1"

_TRUNK_MODULE_ATTRS = (
    "input_embedder", "s_init", "z_init_1", "z_init_2", "rel_pos",
    "token_bonds", "token_bonds_type", "contact_conditioning",
    "s_recycle", "s_norm", "z_recycle", "z_norm",
    "template_module", "msa_module", "pairformer_module",
)


def _unwrap_module(model: Any, attr: str) -> Any:
    mod = getattr(model, attr, None)
    if mod is None:
        return None
    return getattr(mod, "_orig_mod", mod)


def trunk_is_frozen(model: Any) -> bool:
    """True when no parameter feeding the pre-affinity trunk requires grad.

    Only then is the trunk output ``z`` a constant function of the (fixed)
    features and base weights, hence safe to cache across epochs.
    """
    for attr in _TRUNK_MODULE_ATTRS:
        mod = _unwrap_module(model, attr)
        if mod is None or not hasattr(mod, "parameters"):
            continue
        for p in mod.parameters():
            if p.requires_grad:
                return False
    return True


def _stat_sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{os.path.abspath(path)}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return f"{path}:missing"


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(obj, tmp)
        tmp.replace(path)
    except Exception:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise


class TrunkFeatureCache:
    """On-disk memo of per-row features and (frozen) trunk output ``z``.

    ``max_bytes`` (optional) caps the combined size of the ``feats/`` and
    ``z/`` subdirectories; once exceeded, the least-recently-used files (by
    mtime) are evicted after each write until back under budget. This bounds
    scratch usage for long-running / many-arm grids without requiring the
    user to babysit disk space. ``None``/``0`` = unbounded (previous
    behaviour, still the default).
    """

    def __init__(self, root: Path, ckpt_sha: str, *, trunk_frozen: bool,
                max_bytes: Optional[int] = None) -> None:
        self.root = Path(root).expanduser()
        (self.root / "feats").mkdir(parents=True, exist_ok=True)
        (self.root / "z").mkdir(parents=True, exist_ok=True)
        self.ckpt_sha = ckpt_sha or "nockpt"
        self.trunk_frozen = trunk_frozen
        self.max_bytes = max_bytes if max_bytes and max_bytes > 0 else None

    @staticmethod
    def _first(x: Any) -> Any:
        return x[0] if isinstance(x, list) else x

    def feats_key(self, row: dict[str, Any]) -> str:
        rec = _stat_sig(str(self._first(row["receptor"])))
        struct = _stat_sig(str(self._first(row["structure"])))
        lig = str(self._first(row["ligand"]))
        raw = f"{FEAT_CACHE_VERSION}|{rec}|{lig}|{struct}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _feats_path(self, key: str) -> Path:
        return self.root / "feats" / f"{key}.pt"

    def _z_path(self, key: str, recycling_steps: int) -> Path:
        h = hashlib.sha256(
            f"{Z_CACHE_VERSION}|{key}|{self.ckpt_sha}|rec{recycling_steps}".encode()
        ).hexdigest()
        return self.root / "z" / f"{h}.pt"

    def load_feats(self, key: str, device: Any) -> Optional[dict[str, Any]]:
        p = self._feats_path(key)
        if not p.exists():
            return None
        try:
            blob = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 — partial/corrupt write → recompute
            return None
        out = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in blob.items()}
        out["__cache_key__"] = key
        return out

    def save_feats(self, key: str, feats: dict[str, Any]) -> None:
        cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
               for k, v in feats.items() if k != "__cache_key__"}
        _atomic_torch_save(cpu, self._feats_path(key))
        self._enforce_budget()

    def load_z(self, key: str, recycling_steps: int, device: Any) -> Optional[torch.Tensor]:
        p = self._z_path(key, recycling_steps)
        if not p.exists():
            return None
        try:
            return torch.load(p, map_location="cpu", weights_only=False).to(device)
        except Exception:  # noqa: BLE001
            return None

    def save_z(self, key: str, recycling_steps: int, z: torch.Tensor) -> None:
        _atomic_torch_save(z.detach().cpu(), self._z_path(key, recycling_steps))
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        """Evict least-recently-used cache files until under ``max_bytes``.

        Scans both subdirectories together (a feats/z pair for the same pose
        can be evicted independently; either half is cheaply recomputed on
        the next miss). Runs once per cache MISS (i.e. once per unique pose,
        not per epoch), so the directory scan cost is amortised.
        """
        if self.max_bytes is None:
            return
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for sub in ("feats", "z"):
            for p in (self.root / sub).glob("*.pt"):
                try:
                    st = p.stat()
                except OSError:
                    continue
                entries.append((st.st_mtime, st.st_size, p))
                total += st.st_size
        if total <= self.max_bytes:
            return
        entries.sort(key=lambda e: e[0])  # oldest mtime first
        for _mtime, size, p in entries:
            if total <= self.max_bytes:
                break
            try:
                p.unlink()
                total -= size
            except OSError:
                continue
        if total > self.max_bytes:
            logger.warning(
                "Train cache still %.1f GB over budget (%.1f GB) after "
                "evicting all removable files — a single pose's cache "
                "entries may exceed the budget alone.",
                (total - self.max_bytes) / 1e9, self.max_bytes / 1e9,
            )


# Module-global cache, configured per-process by train_lora / train_finetune.
_TRAIN_CACHE: Optional[TrunkFeatureCache] = None


def _configure_train_cache(model: Any, ckpt_sha: str) -> Optional[TrunkFeatureCache]:
    """Enable the disk cache when ``$BOLTZ_TRAIN_CACHE_DIR`` is set.

    ``$BOLTZ_TRAIN_CACHE_MAX_GB`` (optional float) caps total cache size with
    LRU eviction; unset/0 = unbounded.
    """
    global _TRAIN_CACHE
    root = os.environ.get("BOLTZ_TRAIN_CACHE_DIR", "").strip()
    if not root:
        _TRAIN_CACHE = None
        return None
    frozen = trunk_is_frozen(model)
    max_gb_raw = os.environ.get("BOLTZ_TRAIN_CACHE_MAX_GB", "").strip()
    max_bytes = int(float(max_gb_raw) * 1e9) if max_gb_raw else None
    _TRAIN_CACHE = TrunkFeatureCache(Path(root), ckpt_sha, trunk_frozen=frozen,
                                     max_bytes=max_bytes)
    logger.info(
        "Train cache ENABLED at %s (trunk_frozen=%s → z-cache %s; budget=%s).",
        root, frozen, "ON" if frozen else "OFF (feature-cache only)",
        f"{max_bytes/1e9:.0f} GB" if max_bytes else "unbounded",
    )
    return _TRAIN_CACHE


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

    When the frozen-trunk cache is active (``$BOLTZ_TRAIN_CACHE_DIR`` set) the
    deterministic feature dict is memoised to disk, so re-featurisation is
    skipped on every epoch after the first (and shared across arms).
    """
    cache = _TRAIN_CACHE
    _key = cache.feats_key(row) if cache is not None else None
    if cache is not None and _key is not None:
        _cached = cache.load_feats(_key, device)
        if _cached is not None:
            return _cached

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

    work_dir = Path(tempfile.mkdtemp(prefix="boltz_lora_row_"))
    try:
        receptor_yaml = _resolve_yaml(receptor, work_dir)
        yaml_path = _materialize_row_yaml(receptor_yaml, ligand, work_dir)

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
        # use_msa_server is *always* False in this fork. The shared cache is
        # consulted via _materialize_row_yaml -> find_msa, so by the time we
        # reach process_input every protein chain should already carry an
        # `msa:` path.
        del use_msa_server  # parameter kept for backward-compat only
        process_input(
            path=yaml_path, ccd=ccd, msa_dir=msa_dir, mol_dir=mol_dir,
            boltz2=True, use_msa_server=False,
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

        record_files = sorted(records_dir.glob("*.json"))
        if not record_files:
            msg = (
                f"process_input produced no records for {yaml_path}. "
                f"This usually means the receptor YAML is missing 'msa:' "
                f"entries and the shared MSA cache "
                f"(BOLTZ_MSA_CACHE_DIR) has no pre-computed file for the "
                f"sequence. Pre-compute the MSA with "
                f"fineturning_experiment/precompute_msas.py (or "
                f"`python -m boltz.affinity_rescoring.mmseqs2 --sequence "
                f"<SEQ> --out <cache>/<name>.a3m`) and re-run."
            )
            raise RuntimeError(msg)
        record = Record.load(record_files[0])
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
        if cache is not None and _key is not None:
            cache.save_feats(_key, batch)
            batch["__cache_key__"] = _key
        return batch
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


# ── Differentiable trunk + affinity ──────────────────────────────────────────


def _compute_trunk_z(
    model: Any,
    feats: dict[str, Any],
    *,
    recycling_steps: int,
    deterministic: bool,
    unwrap,
) -> torch.Tensor:
    """Run the (frozen) trunk to produce the post-recycling pair rep ``z``.

    ``deterministic=True`` (used only when the trunk is frozen and caching is
    active) evaluates the trunk in eval mode under ``no_grad`` so the returned
    ``z`` is a reusable constant; otherwise the graph is built exactly as
    before (grad flows through any trunk-side adapter linears).
    """
    use_kernels = getattr(model, "use_kernels", False)

    def _body() -> torch.Tensor:
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
        msa_module = unwrap("msa_module")
        pairformer_module = unwrap("pairformer_module")

        for _ in range(recycling_steps + 1):
            s = s_init + model.s_recycle(model.s_norm(s))
            z = z_init + model.z_recycle(model.z_norm(z))
            if getattr(model, "use_templates", False):
                template_module = unwrap("template_module")
                z = z + template_module(z, feats, pair_mask, use_kernels=use_kernels)
            z = z + msa_module(z, s_inputs, feats, use_kernels=use_kernels)
            s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask, use_kernels=use_kernels)
        return z

    if not deterministic:
        return _body()

    mods = [m for m in (_unwrap_module(model, a) for a in _TRUNK_MODULE_ATTRS)
            if m is not None and hasattr(m, "train")]
    prev = [(m, m.training) for m in mods]
    for m in mods:
        m.eval()
    try:
        with torch.no_grad():
            return _body().detach()
    finally:
        for m, was_training in prev:
            m.train(was_training)


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

    When the frozen-trunk cache is active and the trunk carries no trainable
    parameters, the post-recycling ``z`` is loaded from disk (computed once)
    instead of being re-derived each epoch.
    """
    if "coords" not in feats:
        raise RuntimeError("LoRA training requires injected 'coords' in features.")

    def _unwrap(attr: str) -> Any:
        mod = getattr(model, attr)
        return mod._orig_mod if hasattr(mod, "_orig_mod") else mod  # noqa: SLF001

    cache = _TRAIN_CACHE
    _key = feats.pop("__cache_key__", None) if cache is not None else None
    use_cache_z = cache is not None and _key is not None and cache.trunk_frozen
    device = feats["coords"].device
    use_kernels = getattr(model, "use_kernels", False)

    z = cache.load_z(_key, recycling_steps, device) if use_cache_z else None
    if z is None:
        z = _compute_trunk_z(
            model, feats, recycling_steps=recycling_steps,
            deterministic=use_cache_z, unwrap=_unwrap,
        )
        if use_cache_z:
            cache.save_z(_key, recycling_steps, z)

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

    # ── Build train / val datasets ────────────────────────────────────────────
    # Priority: explicit val CSV > random split > no validation.
    val_dataset: Optional[LoRADataset] = None
    if args.val_csv_path:
        val_dataset = LoRADataset(args.val_csv_path, mode=args.mode)
        train_dataset = dataset
        logger.info(
            "Validation CSV: %d rows (%s)",
            len(val_dataset), args.val_csv_path,
        )
    elif args.val_split and 0.0 < args.val_split < 1.0:
        import math as _math
        import copy as _copy
        n_val = max(1, int(_math.ceil(len(dataset) * args.val_split)))
        n_train = len(dataset) - n_val
        if n_train < 1:
            logger.warning(
                "val_split=%.2f leaves 0 training rows — ignoring split.",
                args.val_split,
            )
            train_dataset = dataset
        else:
            # Deterministic stratified split: last n_val rows become validation.
            # Rows are already randomised by the sampler each epoch, so order
            # here is just insertion order from the CSV (stable, reproducible).
            import torch.utils.data as _tud
            train_dataset = _tud.Subset(dataset, list(range(n_train)))
            val_dataset_sub = _tud.Subset(dataset, list(range(n_train, len(dataset))))
            # Wrap in a thin proxy so __len__ / __getitem__ are consistent
            val_dataset = val_dataset_sub  # type: ignore[assignment]
            logger.info(
                "val_split=%.2f → %d train rows, %d val rows.",
                args.val_split, n_train, n_val,
            )
            train_dataset = train_dataset  # noqa: PLW0127 (assigned above)
    else:
        train_dataset = dataset

    # ── Build data loaders ────────────────────────────────────────────────────
    # Use the assay-grouped batch sampler when group_id is populated and
    # batch_size > 1 so that the intra-assay pairwise Huber loss always has
    # same-assay pairs within every mini-batch.
    _train_rows = (
        train_dataset.rows  # LoRADataset
        if hasattr(train_dataset, "rows")
        else [dataset.rows[i] for i in train_dataset.indices]  # Subset
    )
    has_groups = any(r.group_id for r in _train_rows)
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

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
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
    _configure_train_cache(model, ckpt_sha)
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

    # Early stopping state — uses val_loss when a val set is present, else train loss.
    _es_best_loss: float = float("inf")
    _es_patience_counter: int = 0
    _es_best_state: Optional[dict[str, torch.Tensor]] = None
    _es_best_epoch: int = 0

    # ── Checkpoint / resume ────────────────────────────────────────────────────
    ckpt_dir = registry.adapter_dir(args.name)

    # Guard: refuse to overwrite a *completed* adapter (meta.json present) unless
    # the user explicitly passed --overwrite.  We check here — before creating the
    # checkpoint directory — so the error is raised before any work is done.
    if not args.overwrite and (ckpt_dir / "meta.json").exists():
        msg = (
            f"Adapter '{args.name}' already exists at {ckpt_dir}. "
            "Use --overwrite or `boltz lora update` to continue training."
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
            "Resuming LoRA training from checkpoint: %d epoch(s) already completed.",
            _ckpt["epoch_completed"] + 1,
        )
        load_lora_state_dict(model, _ckpt["model_state"], strict=False)
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
        logger.info(
            "Resumed: start_epoch=%d, es_best_loss=%.5f, es_patience=%d.",
            start_epoch, _es_best_loss, _es_patience_counter,
        )

    for epoch in range(start_epoch, args.epochs):
        if _sampler is not None:
            _sampler.set_epoch(epoch)
        epoch_losses: list[float] = []
        for batch in loader:
            target = batch["target"].to(device)
            n_rows = len(batch["name"]) if isinstance(batch["name"], list) else 1

            optim.zero_grad(set_to_none=True)

            if n_rows == 1:
                # Fast path: single row — featurize once, forward once.
                try:
                    feats = _featurize_row(
                        batch, cache_dir=cache_dir,
                        use_msa_server=args.use_msa_server, device=device,
                    )
                except Exception as exc:  # noqa: BLE001
                    name = batch["name"][0] if isinstance(batch["name"], list) else batch["name"]
                    logger.warning("Skipping row %r: featurization failed (%s)", name, exc)
                    continue
                pred = _affinity_forward_trainable(
                    model, feats, recycling_steps=args.recycling_steps,
                )
                pred_combined = pred
            else:
                # Multi-row path: featurize and forward each row keeping all
                # computation graphs live simultaneously so the pairwise loss
                # term can back-prop through all of them in one .backward().
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
                        name = row_i["name"][0] if isinstance(row_i["name"], list) else row_i["name"]
                        logger.warning("Skipping row %r: featurization failed (%s)", name, exc)
                        continue
                    pred_i = _affinity_forward_trainable(
                        model, feats_i, recycling_steps=args.recycling_steps,
                    )
                    all_pred_values.append(pred_i["affinity_pred_value"])
                    if "affinity_logits_binary" in pred_i:
                        all_logits_binary.append(pred_i["affinity_logits_binary"])
                    kept_indices.append(i)
                if not all_pred_values:
                    logger.warning("Skipping batch: no rows survived featurization")
                    continue
                # Subset the per-row target/group/is_binder lists to kept rows.
                if len(kept_indices) != n_rows:
                    keep_t = torch.tensor(kept_indices, device=target.device, dtype=torch.long)
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
                    "affinity_pred_value": torch.cat(all_pred_values, dim=0)
                }
                if all_logits_binary and len(all_logits_binary) == len(kept_indices):
                    pred_combined["affinity_logits_binary"] = torch.cat(
                        all_logits_binary, dim=0
                    )

            batch_for_loss = {**batch, "target": target}
            loss = call_loss(loss_fn, pred_combined, batch_for_loss, adapter)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optim.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        # ── Validation pass (no grad) ──────────────────────────────────────
        mean_val_loss: Optional[float] = None
        if val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_target = val_batch["target"].to(device)
                    try:
                        val_feats = _featurize_row(
                            val_batch, cache_dir=cache_dir,
                            use_msa_server=args.use_msa_server, device=device,
                        )
                    except Exception as exc:  # noqa: BLE001
                        vname = (
                            val_batch["name"][0]
                            if isinstance(val_batch["name"], list)
                            else val_batch["name"]
                        )
                        logger.warning(
                            "Val: skipping row %r: featurization failed (%s)",
                            vname, exc,
                        )
                        continue
                    val_pred = _affinity_forward_trainable(
                        model, val_feats, recycling_steps=args.recycling_steps,
                    )
                    val_batch_for_loss = {**val_batch, "target": val_target}
                    val_loss_val = call_loss(loss_fn, val_pred, val_batch_for_loss, adapter)
                    val_epoch_losses.append(float(val_loss_val.detach().cpu().item()))
            model.train()
            if val_epoch_losses:
                mean_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)

        # ── Log epoch summary ──────────────────────────────────────────────
        if mean_val_loss is not None:
            logger.info(
                "epoch %d / %d  train_loss=%.5f  val_loss=%.5f",
                epoch + 1, args.epochs, mean_loss, mean_val_loss,
            )
            run.metrics.append({
                "epoch": float(epoch),
                "loss": mean_loss,
                "val_loss": mean_val_loss,
            })
        else:
            logger.info("epoch %d / %d  train_loss=%.5f", epoch + 1, args.epochs, mean_loss)
            run.metrics.append({"epoch": float(epoch), "loss": mean_loss})

        # ── Early stopping ───────────────────────────────────────────────────
        _es_stop = False
        if args.early_stopping_patience > 0:
            # Use val_loss for early stopping when available, else train loss.
            _es_monitor = mean_val_loss if mean_val_loss is not None else mean_loss
            if _es_monitor < _es_best_loss - args.early_stopping_min_delta:
                _es_best_loss = _es_monitor
                _es_patience_counter = 0
                _es_best_epoch = epoch + 1
                _es_best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in lora_state_dict(model).items()
                }
                logger.info(
                    "Early stopping: new best %s=%.5f at epoch %d",
                    "val_loss" if mean_val_loss is not None else "train_loss",
                    _es_best_loss, _es_best_epoch,
                )
            else:
                _es_patience_counter += 1
                logger.info(
                    "Early stopping: no improvement for %d/%d epochs "
                    "(best %s=%.5f at epoch %d)",
                    _es_patience_counter, args.early_stopping_patience,
                    "val_loss" if mean_val_loss is not None else "train_loss",
                    _es_best_loss, _es_best_epoch,
                )
                if _es_patience_counter >= args.early_stopping_patience:
                    logger.info(
                        "Early stopping triggered at epoch %d/%d — "
                        "restoring best weights from epoch %d (loss=%.5f).",
                        epoch + 1, args.epochs, _es_best_epoch, _es_best_loss,
                    )
                    _es_stop = True

        # ── Checkpoint (one file per completed epoch) ───────────────────────────
        ckpt_path = _checkpoint_path_for_epoch(ckpt_dir, epoch)
        _save_checkpoint(ckpt_path, {
            "epoch_completed": epoch,
            "model_state": {
                k: v.detach().cpu().clone()
                for k, v in lora_state_dict(model).items()
            },
            "optim_state": optim.state_dict(),
            "metrics": run.metrics,
            "started_at": run.started_at,
            "es_best_loss": _es_best_loss,
            "es_patience_counter": _es_patience_counter,
            "es_best_epoch": _es_best_epoch,
            "es_best_state": _es_best_state,
        })
        logger.debug("Checkpoint saved after epoch %d.", epoch + 1)

        # ── Live loss curve (updated after every epoch) ──────────────────────
        if args.plot_loss_curve:
            _plot_loss_curve(
                run.metrics,
                ckpt_dir / "loss_curve.png",
                title=f"LoRA Training Loss — {args.name}",
            )

        if _es_stop:
            break

    # Restore the best-seen weights when early stopping is active
    if args.early_stopping_patience > 0 and _es_best_state is not None:
        load_lora_state_dict(model, _es_best_state, strict=False)

    run.finished_at = time.time()
    adapter.history.append(run)

    state = lora_state_dict(model)
    # The adapter directory already exists (created for checkpoints), so always
    # pass overwrite=True here.  The guard against stomping a completed adapter
    # was already enforced above before training started.
    registry.save(adapter, state, overwrite=True)

    return adapter


def prepare_update(
    name: str, *, registry: Optional[LoRARegistry] = None,
) -> tuple[LoRAAdapter, dict[str, torch.Tensor]]:
    """Load an existing adapter so it can be resumed by :func:`train_lora`."""
    registry = registry or default_registry()
    return registry.load(name)


__all__ = ["TrainArgs", "prepare_update", "train_lora"]
