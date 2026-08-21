"""
Inference engine for affinity rescoring.

*** AFFINITY-ONLY MODE ***
This module runs ONLY the trunk + affinity head of Boltz-2.
Diffusion and confidence modules are intentionally bypassed.
All coordinates come from pre-existing PDB/CIF structures.

Manages model loading, device detection, checkpoint management,
and direct affinity inference (trunk + affinity head) for the
Boltz-2 affinity module.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from boltz.affinity_rescoring.trunk_cache import (
    feats_digest,
    get_cache as get_trunk_cache,
)

from boltz.affinity_rescoring.models import (
    DeviceOption,
    compute_file_sha256,
)

logger = logging.getLogger(__name__)


class AffinityModelManager:
    """
    Manages model lifecycle with caching and device management.

    Responsibilities:
    - Download/cache checkpoints
    - Validate checkpoint integrity (SHA256)
    - Load model on appropriate device
    - Handle device fallback (GPU → CPU)
    """

    # Default checkpoint URLs (same as main.py) I had to put this here cause of cache errors that may occur. 
    CHECKPOINT_URLS = {
        "gateway": "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
        "huggingface": "https://huggingface.co/boltz-community/boltz2/resolve/main/boltz2_aff.ckpt",
    }

    def __init__(
        self,
        device: DeviceOption = DeviceOption.AUTO,
        cache_dir: Optional[str] = None,
    ):
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.cache_dir = Path(cache_dir or os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._checkpoint_path: Optional[Path] = None
        self._checkpoint_sha256: str = ""

    def _resolve_device(self, device: DeviceOption) -> str:
        """Resolve device with fallback chain."""
        if device == DeviceOption.CUDA:
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        elif device == DeviceOption.MPS:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return "cpu"
        elif device == DeviceOption.CPU:
            return "cpu"
        else:  # AUTO
            if torch.cuda.is_available():
                # Check GPU memory (need at least 4GB)
                try:
                    mem = torch.cuda.get_device_properties(0).total_mem
                    if mem >= 4 * 1024**3:
                        return "cuda"
                    else:
                        logger.info(f"GPU memory {mem / 1024**3:.1f}GB < 4GB. Using CPU.")
                except Exception:
                    pass
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

    def get_checkpoint_path(
        self, checkpoint: str = "auto"
    ) -> Path:
        """
        Resolve checkpoint path. Downloads if 'auto' and not cached.
        """
        if checkpoint != "auto":
            path = Path(checkpoint)
            if not path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {path}. "
                    f"Verify path or use --checkpoint auto to download."
                )
            return path

        # Check cache
        cached = self.cache_dir / "boltz2_aff.ckpt"
        if cached.exists():
            logger.info(f"Using cached checkpoint: {cached}")
            return cached

        # Download
        logger.info("Checkpoint not found. Downloading boltz2_aff.ckpt...")
        return self._download_checkpoint(cached)

    def _download_checkpoint(self, target: Path) -> Path:
        """Download checkpoint with fallback URLs."""
        for name, url in self.CHECKPOINT_URLS.items():
            try:
                logger.info(f"Trying {name}: {url}")
                import urllib.request
                tmp = target.with_suffix(".tmp")
                urllib.request.urlretrieve(url, str(tmp))
                shutil.move(str(tmp), str(target))
                logger.info(f"Downloaded checkpoint to {target}")
                return target
            except Exception as e:
                logger.warning(f"Failed to download from {name}: {e}")
                continue

        raise RuntimeError(
            "Failed to download checkpoint from all sources. "
            "Check internet connection or provide --checkpoint path."
        )

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        affinity_mw_correction: bool = True,
    ):
        """
        Load the Boltz-2 affinity model.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to checkpoint. Uses 'auto' if None.
        affinity_mw_correction : bool
            Whether to apply molecular weight correction.

        Returns
        -------
        model : Boltz2
            Loaded model in eval mode.
        """
        from boltz.model.models.boltz2 import Boltz2

        ckpt_path = self.get_checkpoint_path(checkpoint_path or "auto")
        self._checkpoint_path = ckpt_path
        self._checkpoint_sha256 = compute_file_sha256(ckpt_path)

        logger.info(f"Loading model from {ckpt_path} on {self.device}")

        # Affinity-only predict args.
        # NOTE: diffusion / confidence parameters are intentionally omitted.
        # This module runs trunk + affinity head ONLY — no diffusion, no
        # confidence module.  Setting diffusion/sampling values here would
        # have no effect since affinity_forward() bypasses those stages.
        predict_args = {
            "recycling_steps": 5,
            "sampling_steps": 0,      # unused — affinity-only mode
            "diffusion_samples": 0,   # unused — affinity-only mode
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        # ── Patch checkpoint to strip unknown kwargs ──────────────
        # Newer checkpoints may store hyperparameters (e.g.
        # mse_rotational_alignment) that the current source code's
        # AtomDiffusion.__init__() does not accept.  Rather than
        # modifying Boltz source, we strip them from the checkpoint
        # in memory before loading.
        _STRIP_DIFFUSION_KEYS = {"mse_rotational_alignment"}

        # PyTorch ≥2.6 changed weights_only default to True.  This checkpoint
        # is from a trusted internal source and serialises multiple omegaconf
        # internal types (DictConfig, ListConfig, ContainerMetadata, …).
        # Rather than maintaining an ever-growing allowlist, load with
        # weights_only=False which is the correct approach for trusted ckpts.
        ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        patched = False
        if "hyper_parameters" in ckpt_data:
            hp = ckpt_data["hyper_parameters"]
            if "diffusion_process_args" in hp:
                for key in _STRIP_DIFFUSION_KEYS:
                    if key in hp["diffusion_process_args"]:
                        del hp["diffusion_process_args"][key]
                        patched = True
                        logger.info(
                            f"Stripped unsupported checkpoint hparam: "
                            f"diffusion_process_args.{key}"
                        )
            # ── Force v2 pairformer/MSA paths ──────────────────────
            # The Boltz-2 affinity checkpoint was trained with the v2
            # attention/MSA blocks but stores `pairformer_args` /
            # `msa_args` without the `v2: True` flag.  Rebuilding the
            # model from these hparams without the flag picks the v1
            # AttentionPairBias (with an extra `norm_s` LayerNorm) and
            # produces a state-dict mismatch.  Force v2 here.
            # Note: hparams are typically stored as omegaconf DictConfig,
            # so we use duck-typing rather than isinstance(dict).
            for _args_key in ("pairformer_args", "msa_args"):
                _args = hp.get(_args_key)
                if _args is None:
                    continue
                try:
                    _has_v2 = bool(_args.get("v2", False)) if hasattr(_args, "get") else False
                    if not _has_v2:
                        _args["v2"] = True
                        patched = True
                        logger.info(
                            f"Forced checkpoint hparam {_args_key}.v2 = True "
                            f"(Boltz-2 uses v2 attention blocks)"
                        )
                except Exception as _e:  # pragma: no cover
                    logger.warning(
                        f"Could not patch {_args_key}.v2 in checkpoint hparams: {_e}"
                    )

        if patched:
            # Save patched checkpoint to a temp file for loading
            import tempfile as _tmpmod
            _tmp_fd, _tmp_ckpt = _tmpmod.mkstemp(suffix=".ckpt")
            os.close(_tmp_fd)
            torch.save(ckpt_data, _tmp_ckpt)
            _load_path = _tmp_ckpt
        else:
            _load_path = str(ckpt_path)
            _tmp_ckpt = None

        del ckpt_data  # free memory

        try:
            model = Boltz2.load_from_checkpoint(
                _load_path,
                strict=True,
                map_location="cpu",  # Load to CPU first, then move
                predict_args=predict_args,
                affinity_mw_correction=affinity_mw_correction,
            )
            model.eval()
            model = model.to(self.device)
            self._model = model
            logger.info(f"Model loaded successfully (device={self.device})")
            return model

        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                logger.warning(f"GPU loading failed: {e}. Falling back to CPU.")
                self.device = "cpu"
                model = Boltz2.load_from_checkpoint(
                    _load_path,
                    strict=True,
                    map_location="cpu",
                    predict_args=predict_args,
                    affinity_mw_correction=affinity_mw_correction,
                )
                model.eval()
                self._model = model
                return model
            raise RuntimeError(
                f"Model loading failed: {e}. "
                f"Verify CUDA/torch installation: pip install torch --force-reinstall"
            ) from e

        finally:
            # Clean up temp patched checkpoint
            if _tmp_ckpt is not None:
                try:
                    os.remove(_tmp_ckpt)
                except OSError:
                    pass

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256


def create_affinity_yaml(
    protein_sequence: str,
    ligand_smiles: str,
    protein_chain_id: str = "A",
    ligand_chain_id: str = "B",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Create a Boltz affinity YAML input file.

    Parameters
    ----------
    protein_sequence : str
        Amino acid sequence
    ligand_smiles : str
        SMILES string for ligand
    protein_chain_id : str
        Chain ID for protein
    ligand_chain_id : str
        Chain ID for ligand (binder)
    output_path : Path, optional
        Where to write the YAML. Uses temp file if None.

    Returns
    -------
    Path
        Path to created YAML file
    """
    import yaml

    data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": protein_chain_id, "sequence": protein_sequence}},
            {"ligand": {"id": ligand_chain_id, "smiles": ligand_smiles}},
        ],
        "properties": [
            {"affinity": {"binder": ligand_chain_id}},
        ],
    }

    if output_path is None:
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        output_path = Path(tmp)

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    return output_path


# ─── Direct Affinity Inference (Plan B) ──────────────────────────────────────


def _featurizer_seed() -> int:
    """Global-torch-RNG seed for rescore featurisation (see run_direct_affinity_inference)."""
    try:
        return int(os.environ.get("BOLTZ_FEATURIZER_SEED", "42"))
    except ValueError:
        return 42


def _get_module(model: Any, attr: str) -> Any:
    """Get a model submodule, unwrapping torch.compile if needed."""
    mod = getattr(model, attr)
    if hasattr(mod, "_orig_mod"):
        return mod._orig_mod  # noqa: SLF001
    return mod


def _affinity_input_embed_with_ablation(
    model: Any,
    feats: Dict[str, Tensor],
    *,
    zero_atom_encoder: bool = False,
    zero_msa_profile: bool = False,
    zero_res_type: bool = False,
) -> Tensor:
    """Compute ``model.input_embedder(feats, affinity=True)`` with optional
    sub-component ablations.

    The InputEmbedder additively combines three paths:

        s = atom_encoder(feats) + res_type_encoding(res_type)
              + msa_profile_encoding(profile, deletion_mean)

    Each ``zero_*`` flag subtracts the corresponding contribution from
    the recomputed ``s_inputs`` tensor so that channel is zeroed before
    being passed to the affinity head.
    """
    s_inputs = model.input_embedder(feats, affinity=True)

    if not (zero_atom_encoder or zero_msa_profile or zero_res_type):
        return s_inputs

    embedder = model.input_embedder
    if hasattr(embedder, "_orig_mod"):
        embedder = embedder._orig_mod  # noqa: SLF001

    if zero_res_type:
        contrib = embedder.res_type_encoding(feats["res_type"].float())
        s_inputs = s_inputs - contrib

    if zero_msa_profile:
        profile = feats["profile_affinity"]
        deletion_mean = feats["deletion_mean_affinity"].unsqueeze(-1)
        contrib = embedder.msa_profile_encoding(
            torch.cat([profile, deletion_mean], dim=-1)
        )
        s_inputs = s_inputs - contrib

    if zero_atom_encoder:
        # Recompute the atom-encoder contribution exactly as InputEmbedder
        # does, then subtract it. This avoids replicating the full forward
        # pass while still cleanly removing that pathway.
        q, c, p, to_keys = embedder.atom_encoder(feats)
        atom_enc_bias = embedder.atom_enc_proj_z(p)
        a, _, _, _ = embedder.atom_attention_encoder(
            feats=feats,
            q=q,
            c=c,
            atom_enc_bias=atom_enc_bias,
            to_keys=to_keys,
        )
        s_inputs = s_inputs - a

    return s_inputs


@torch.inference_mode()
def affinity_trunk_forward(
    model: Any,
    feats: Dict[str, Tensor],
    recycling_steps: int = 3,
) -> Dict[str, Tensor]:
    """Run only the trunk (input embedder + recycled msa/pairformer) and
    return the intermediate tensors needed by the affinity head.

    This is the front half of :func:`affinity_forward`; useful when the
    same trunk output should be reused across many affinity-head
    invocations (e.g. residue leave-one-out studies).

    Returns
    -------
    dict
        ``{"z": Tensor, "s": Tensor, "use_kernels": bool}`` where
        ``z`` has shape ``(B, N, N, token_z)`` and ``s`` has shape
        ``(B, N, token_s)`` from the trunk's last recycling iteration.
    """
    if "coords" not in feats:
        raise RuntimeError(
            "AFFINITY-ONLY MODE: 'coords' not found in features. "
            "PDB coordinates must be injected before calling "
            "affinity_trunk_forward(). This function does NOT run diffusion."
        )
    model.eval()

    s_inputs = model.input_embedder(feats)
    s_init = model.s_init(s_inputs)
    z_init = (
        model.z_init_1(s_inputs)[:, :, None]
        + model.z_init_2(s_inputs)[:, None, :]
    )
    relative_position_encoding = model.rel_pos(feats)
    z_init = z_init + relative_position_encoding
    z_init = z_init + model.token_bonds(feats["token_bonds"].float())
    if model.bond_type_feature:
        z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
    z_init = z_init + model.contact_conditioning(feats)

    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)

    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    use_kernels = model.use_kernels

    msa_module = _get_module(model, "msa_module")
    pairformer_module = _get_module(model, "pairformer_module")

    for _ in range(recycling_steps + 1):
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z_init + model.z_recycle(model.z_norm(z))
        if model.use_templates:
            template_module = _get_module(model, "template_module")
            z = z + template_module(z, feats, pair_mask, use_kernels=use_kernels)
        z = z + msa_module(z, s_inputs, feats, use_kernels=use_kernels)
        s, z = pairformer_module(
            s, z, mask=mask, pair_mask=pair_mask, use_kernels=use_kernels,
        )

    return {"z": z, "s": s, "use_kernels": use_kernels}


def _build_cross_pair_mask(feats: Dict[str, Tensor]) -> Tensor:
    """Construct the (N, N) cross-pair mask used by the affinity head."""
    pad_token_mask = feats["token_pad_mask"][0]
    rec_mask = (feats["mol_type"][0] == 0) * pad_token_mask
    lig_mask = feats["affinity_token_mask"][0].to(torch.bool) * pad_token_mask
    return (
        lig_mask[:, None] * rec_mask[None, :]
        + rec_mask[:, None] * lig_mask[None, :]
        + lig_mask[:, None] * lig_mask[None, :]
    )


@torch.inference_mode()
def affinity_head_forward(
    model: Any,
    feats: Dict[str, Tensor],
    trunk_out: Dict[str, Tensor],
    *,
    z_token_mask: Optional[Tensor] = None,
    zero_z_trunk: bool = False,
    zero_s_inputs: bool = False,
    disable_distogram: bool = False,
    pose_noise_sigma: float = 0.0,
    pose_noise_seed: Optional[int] = None,
    pose_noise_target: str = "ligand",
    zero_atom_encoder: bool = False,
    zero_msa_profile: bool = False,
    zero_res_type: bool = False,
) -> Dict[str, Any]:
    """Run only the affinity head, given pre-computed trunk outputs.

    Parameters
    ----------
    model, feats : see :func:`affinity_forward`.
    trunk_out : dict
        Output of :func:`affinity_trunk_forward` (must contain ``z`` and
        ``use_kernels``).
    z_token_mask : Tensor or None
        Optional per-token multiplicative mask of shape ``(N_tokens,)``
        applied to ``z`` as an outer product ``mask[:,None] * mask[None,:]``
        before passing into the affinity head. Used by the leave-one-out
        residue saliency analysis to zero a single residue's row/col.
    Other kwargs : same as :func:`affinity_forward`.
    """
    z = trunk_out["z"]
    use_kernels = trunk_out["use_kernels"]

    cross_pair_mask = _build_cross_pair_mask(feats)
    z_affinity = z * cross_pair_mask[None, :, :, None]

    if z_token_mask is not None:
        z_token_mask = z_token_mask.to(z_affinity.device, z_affinity.dtype)
        loo_mask = z_token_mask[:, None] * z_token_mask[None, :]
        z_affinity = z_affinity * loo_mask[None, :, :, None]

    if zero_z_trunk:
        z_affinity = torch.zeros_like(z_affinity)

    coords_affinity = feats["coords"].detach()
    if coords_affinity.dim() == 3:
        coords_affinity = coords_affinity[None]
    elif coords_affinity.dim() == 4 and coords_affinity.shape[1] > 1:
        coords_affinity = coords_affinity[:, :1]

    if pose_noise_sigma > 0.0:
        atom_to_token = feats["atom_to_token"]
        if pose_noise_target == "ligand":
            tok_mask = feats["affinity_token_mask"][0].to(torch.bool)
        elif pose_noise_target == "receptor":
            pad_tok = feats["token_pad_mask"][0]
            tok_mask = ((feats["mol_type"][0] == 0) * pad_tok).to(torch.bool)
        elif pose_noise_target == "all":
            tok_mask = feats["token_pad_mask"][0].to(torch.bool)
        else:
            raise ValueError(
                f"Unknown pose_noise_target={pose_noise_target!r}"
            )
        n_tok_mask = tok_mask.shape[0]
        a2t = atom_to_token[0][..., :n_tok_mask].to(torch.bool)
        atom_mask = a2t.any(dim=-1) & (a2t & tok_mask.unsqueeze(0)).any(dim=-1)
        atom_mask_b = atom_mask.view(1, 1, -1, 1).to(coords_affinity.dtype)
        if pose_noise_seed is not None:
            gen = torch.Generator(device=coords_affinity.device)
            gen.manual_seed(int(pose_noise_seed))
            noise = torch.randn(
                coords_affinity.shape,
                generator=gen,
                device=coords_affinity.device,
                dtype=coords_affinity.dtype,
            )
        else:
            noise = torch.randn_like(coords_affinity)
        coords_affinity = coords_affinity + atom_mask_b * (
            noise * float(pose_noise_sigma)
        )

    s_inputs = _affinity_input_embed_with_ablation(
        model, feats,
        zero_atom_encoder=zero_atom_encoder,
        zero_msa_profile=zero_msa_profile,
        zero_res_type=zero_res_type,
    )
    if zero_s_inputs:
        s_inputs = torch.zeros_like(s_inputs)

    module_kwargs = {"disable_distogram": disable_distogram}
    results: Dict[str, Any] = {}

    with torch.autocast("cuda", enabled=False):
        if model.affinity_ensemble:
            affinity_module1 = _get_module(model, "affinity_module1")
            affinity_module2 = _get_module(model, "affinity_module2")

            out1 = affinity_module1(
                s_inputs=s_inputs.detach(),
                z=z_affinity.detach(),
                x_pred=coords_affinity,
                feats=feats,
                multiplicity=1,
                use_kernels=use_kernels,
                **module_kwargs,
            )
            out1["affinity_probability_binary"] = torch.nn.functional.sigmoid(
                out1["affinity_logits_binary"]
            )
            out2 = affinity_module2(
                s_inputs=s_inputs.detach(),
                z=z_affinity.detach(),
                x_pred=coords_affinity,
                feats=feats,
                multiplicity=1,
                use_kernels=use_kernels,
                **module_kwargs,
            )
            out2["affinity_probability_binary"] = torch.nn.functional.sigmoid(
                out2["affinity_logits_binary"]
            )
            avg_pred = (out1["affinity_pred_value"] + out2["affinity_pred_value"]) / 2
            avg_prob = (
                out1["affinity_probability_binary"]
                + out2["affinity_probability_binary"]
            ) / 2
            if model.affinity_mw_correction:
                model_coef = 1.03525938
                mw_coef = -0.59992683
                bias = 2.83288489
                mw = feats["affinity_mw"][0] ** 0.3
                avg_pred = model_coef * avg_pred + mw_coef * mw + bias
            results["affinity_pred_value"] = avg_pred.item()
            results["affinity_probability_binary"] = avg_prob.item()
            results["affinity_pred_value1"] = out1["affinity_pred_value"].item()
            results["affinity_pred_value2"] = out2["affinity_pred_value"].item()
            results["affinity_probability_binary1"] = out1[
                "affinity_probability_binary"
            ].item()
            results["affinity_probability_binary2"] = out2[
                "affinity_probability_binary"
            ].item()
        else:
            affinity_module = _get_module(model, "affinity_module")
            out = affinity_module(
                s_inputs=s_inputs.detach(),
                z=z_affinity.detach(),
                x_pred=coords_affinity,
                feats=feats,
                multiplicity=1,
                use_kernels=use_kernels,
                **module_kwargs,
            )
            results["affinity_pred_value"] = out["affinity_pred_value"].item()
            results["affinity_probability_binary"] = torch.nn.functional.sigmoid(
                out["affinity_logits_binary"]
            ).item()

    return results


@torch.inference_mode()
def affinity_forward(
    model: Any,
    feats: Dict[str, Tensor],
    recycling_steps: int = 3,
    *,
    zero_z_trunk: bool = False,
    zero_s_inputs: bool = False,
    disable_distogram: bool = False,
    pose_noise_sigma: float = 0.0,
    pose_noise_seed: Optional[int] = None,
    pose_noise_target: str = "ligand",
    zero_atom_encoder: bool = False,
    zero_msa_profile: bool = False,
    zero_res_type: bool = False,
) -> Dict[str, Any]:
    """Run trunk + affinity head, skipping diffusion and confidence.

    This replicates the relevant parts of ``Boltz2.forward()`` without
    running the diffusion or confidence modules.  The input ``feats``
    must already contain ``coords`` (from PDB coordinate injection)
    which will be used directly as ``x_pred`` for the affinity head.

    *** SAFETY: This function NEVER calls diffusion or confidence. ***
    If you see results from this function, they came from affinity-only
    inference (trunk + affinity head with injected PDB coordinates).

    Parameters
    ----------
    model : Boltz2
        A loaded Boltz2 model in eval mode.
    feats : dict
        Feature dictionary produced by the data pipeline
        (tokenize → crop → featurize) with PDB coordinates.
    recycling_steps : int
        Number of recycling iterations for the trunk.
    zero_z_trunk : bool
        If True, zero the pair representation ``z`` passed to the affinity
        head (ablation: removes trunk pair signal).
    zero_s_inputs : bool
        If True, zero the single representation ``s_inputs`` passed to the
        affinity head (ablation: removes per-token single signal).
    disable_distogram : bool
        If True, skip the distogram contribution inside the affinity
        module (ablation: removes pose-distance signal).
    pose_noise_sigma : float
        Gaussian noise standard deviation (Å) added to atom coordinates
        before the affinity head. ``0`` disables noise.
    pose_noise_seed : int or None
        RNG seed for reproducibility of the noise. If ``None``, draws
        from the global torch RNG.
    pose_noise_target : {"ligand", "all", "receptor"}
        Which atoms to perturb. Default ``"ligand"`` perturbs only atoms
        belonging to the ligand (affinity binder).
    zero_atom_encoder, zero_msa_profile, zero_res_type : bool
        Sub-component ablations of the affinity-pass ``InputEmbedder``
        output. Each one zeroes one of the three additive paths inside
        ``InputEmbedder.forward`` (atom encoder, MSA profile, residue
        type). Implemented by recomputing the affinity-pass embedding
        with the targeted feature tensor in ``feats`` zeroed.

    Returns
    -------
    dict
        Affinity predictions including ``affinity_pred_value``,
        ``affinity_probability_binary``, and ensemble values if applicable.

    Raises
    ------
    RuntimeError
        If ``feats`` does not contain ``coords`` (PDB coordinates must
        be injected before calling this function).
    """
    # Single source of truth: run the trunk, then the head.  The two
    # split helpers below carry the actual implementation; this entry
    # point is preserved for backward compatibility and stays the
    # canonical call for one-shot trunk + head inference.
    #
    # The trunk is unaffected by the affinity LoRA/finetune presets (they only
    # touch modules inside ``affinity_module``), so its output can be memoised
    # and reused across every arm scored on the same pose. Opt-in via
    # $BOLTZ_RESCORE_CACHE_DIR; see boltz.affinity_rescoring.trunk_cache.
    cache = get_trunk_cache(model)
    cache_key = None
    if cache is not None:
        cache_key = cache.key(feats_digest(feats), recycling_steps)
        z = cache.load(cache_key, feats["token_pad_mask"].device)
        if z is not None:
            return affinity_head_forward(
                model, feats, {"z": z, "use_kernels": model.use_kernels},
                zero_z_trunk=zero_z_trunk,
                zero_s_inputs=zero_s_inputs,
                disable_distogram=disable_distogram,
                pose_noise_sigma=pose_noise_sigma,
                pose_noise_seed=pose_noise_seed,
                pose_noise_target=pose_noise_target,
                zero_atom_encoder=zero_atom_encoder,
                zero_msa_profile=zero_msa_profile,
                zero_res_type=zero_res_type,
            )

    trunk_out = affinity_trunk_forward(
        model, feats, recycling_steps=recycling_steps,
    )
    if cache is not None and cache_key is not None:
        cache.save(cache_key, trunk_out["z"])
    return affinity_head_forward(
        model, feats, trunk_out,
        zero_z_trunk=zero_z_trunk,
        zero_s_inputs=zero_s_inputs,
        disable_distogram=disable_distogram,
        pose_noise_sigma=pose_noise_sigma,
        pose_noise_seed=pose_noise_seed,
        pose_noise_target=pose_noise_target,
        zero_atom_encoder=zero_atom_encoder,
        zero_msa_profile=zero_msa_profile,
        zero_res_type=zero_res_type,
    )


def run_direct_affinity_inference(
    model: Any,
    yaml_path: Path,
    pdb_atoms: list,
    chain_id_map: Dict[str, int],
    cache_dir: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    recycling_steps: int = 3,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete direct affinity inference pipeline.

    Steps:
    1. Run ``process_input`` to preprocess the YAML (structures, MSA, molecules)
    2. Inject PDB coordinates into the processed structure
    3. Save as pre_affinity npz
    4. Load through the affinity data pipeline (tokenize, crop, featurize)
    5. Run ``affinity_forward`` (trunk + affinity head, no diffusion)

    Parameters
    ----------
    model : Boltz2
        Loaded model in eval mode.
    yaml_path : Path
        Path to the affinity YAML input file.
    pdb_atoms : list
        Parsed PDB atoms (AtomInfo namedtuples).
    chain_id_map : dict
        Mapping from PDB chain_id → YAML chain_id. This is used to
        build the full chain_id → asym_id mapping after processing.
    cache_dir : Path, optional
        Boltz cache directory (default: ``~/.boltz``).
    work_dir : Path, optional
        Working directory for intermediate files. Created as tmpdir if None.
    use_msa_server : bool
        Whether to use the MSA server.
    msa_server_url : str
        URL for the MSA server.
    recycling_steps : int
        Number of trunk recycling steps.
    device : str, optional
        Device string (e.g. "cuda", "cpu"). Inferred from model if None.

    Returns
    -------
    dict
        Affinity prediction results.
    """
    from boltz.affinity_rescoring.coord_injection import (
        build_chain_id_map,
        check_unresolved_near_pocket,
        inject_pdb_coords_into_structure,
        save_pre_affinity_structure,
    )
    from boltz.data import const
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals, load_molecules
    from boltz.data.module.inferencev2 import load_input
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Record, StructureV2
    from boltz.main import process_input

    if cache_dir is None:
        cache_dir = Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()

    mol_dir = cache_dir / "mols"
    ccd_path = cache_dir / "ccd.pkl"
    own_work_dir = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="boltz_direct_"))

    try:
        # ── Step 1: Preprocess YAML ──────────────────────────────────
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

        # Load CCD
        ccd = load_canonicals(mol_dir)

        process_input(
            path=yaml_path,
            ccd=ccd,
            msa_dir=msa_dir,
            mol_dir=mol_dir,
            boltz2=True,
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy="paired+unpaired",
            msa_server_username=None,
            msa_server_password=None,
            api_key_header=None,
            api_key_value=None,
            max_msa_seqs=8192,
            processed_msa_dir=processed_msa_dir,
            processed_constraints_dir=processed_constraints_dir,
            processed_templates_dir=processed_templates_dir,
            processed_mols_dir=processed_mols_dir,
            structure_dir=structure_dir,
            records_dir=records_dir,
        )

        # ── Step 2: Load record and inject coordinates ───────────────
        record_files = list(records_dir.glob("*.json"))
        if not record_files:
            raise RuntimeError("process_input produced no records.")
        record = Record.load(record_files[0])

        processed_struct = StructureV2.load(structure_dir / f"{record.id}.npz")

        # Build the chain_id_map from YAML chain IDs to asym_ids
        yaml_chain_ids = list(chain_id_map.values())
        asym_map = build_chain_id_map(processed_struct, yaml_chain_ids)

        # Build full map: PDB chain_id → asym_id
        full_map: Dict[str, int] = {}
        for pdb_cid, yaml_cid in chain_id_map.items():
            if yaml_cid in asym_map:
                full_map[pdb_cid] = asym_map[yaml_cid]

        injected, unmatched_indices = inject_pdb_coords_into_structure(
            processed_struct, pdb_atoms, full_map
        )

        # ── Step 2b: Check if unresolved residues are near pocket ────
        pocket_report = check_unresolved_near_pocket(
            injected, unmatched_indices, threshold=10.0
        )
        if unmatched_indices:
            logger.warning(
                f"{len(unmatched_indices)} atoms had no PDB match and were "
                f"zeroed out. Atom indices: {unmatched_indices[:20]}"
                + ("..." if len(unmatched_indices) > 20 else "")
            )

        # ── Step 3: Save pre_affinity structure ──────────────────────
        save_pre_affinity_structure(injected, predictions_dir, record.id)

        # ── Step 4: Featurize through affinity pipeline ──────────────
        tokenizer = Boltz2Tokenizer()
        cropper = AffinityCropper()
        featurizer = Boltz2Featurizer()
        canonicals = load_canonicals(mol_dir)

        input_data = load_input(
            record=record,
            target_dir=predictions_dir,
            msa_dir=processed_msa_dir,
            constraints_dir=processed_constraints_dir,
            template_dir=processed_templates_dir,
            extra_mols_dir=processed_mols_dir,
            affinity=True,
        )

        tokenized = tokenizer.tokenize(input_data)
        tokenized = cropper.crop(tokenized, max_tokens=256, max_atoms=2048)

        molecules = {}
        molecules.update(canonicals)
        if input_data.extra_mols:
            molecules.update(input_data.extra_mols)
        mol_names = set(tokenized.tokens["res_name"].tolist())
        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(mol_dir, mol_names))

        random = np.random.default_rng(42)
        # The featuriser applies center_random_augmentation() to ref_pos — a
        # training-time roto-translation that is NOT gated on training=False and
        # draws from the *global* torch RNG. Left alone, scoring the same pose
        # twice yields different ref_pos (up to ~15 A) and different affinity
        # predictions. Fork the RNG so this call is reproducible without
        # disturbing the caller's random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_featurizer_seed())
            features = featurizer.process(
                tokenized,
                molecules=molecules,
                random=random,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=None,
                inference_contact_constraints=None,
                compute_constraint_features=True,
                override_method=None,
                compute_affinity=True,
            )

        # ── Step 5: Move features to device and run forward ──────────
        if device is None:
            device = next(model.parameters()).device
        else:
            device = torch.device(device)

        batch = {}
        for k, v in features.items():
            if isinstance(v, Tensor):
                batch[k] = v.unsqueeze(0).to(device)  # add batch dim
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            elif k == "affinity_mw":
                # Must be a list so feats["affinity_mw"][0] works
                # (matches collate() in inferencev2.py which keeps it as list)
                batch[k] = [v]
            else:
                batch[k] = v

        results = affinity_forward(model, batch, recycling_steps=recycling_steps)

        # Attach pocket proximity report for benchmarking
        results["pocket_proximity_report"] = {
            "n_unresolved_residues": pocket_report.n_unresolved_residues,
            "n_near_pocket": pocket_report.n_near_pocket,
            "threshold_angstrom": pocket_report.threshold_angstrom,
            "near_pocket_residues": [
                {
                    "chain": r.chain_name,
                    "residue_index": r.residue_index,
                    "unresolved_atoms": r.n_unresolved_atoms,
                    "total_atoms": r.n_total_atoms,
                    "min_dist_to_ligand": r.min_distance_to_ligand,
                    "nearest_ligand_chain": r.nearest_ligand_chain,
                }
                for r in pocket_report.near_pocket_residues
            ],
        }

        logger.info(
            f"Direct affinity inference complete: "
            f"pred={results.get('affinity_pred_value', 'N/A'):.3f}, "
            f"prob={results.get('affinity_probability_binary', 'N/A'):.3f}"
        )

        return results

    finally:
        if own_work_dir:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
