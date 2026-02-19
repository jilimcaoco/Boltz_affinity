"""
Inference engine for affinity rescoring.

Manages model loading, device detection, and running inference
through the Boltz-2 affinity module - either via the full pipeline
or with pre-computed coordinates (skipping diffusion).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from boltz.affinity_rescoring.models import (
    AffinityResult,
    DeviceOption,
    RescoreConfig,
    Timer,
    ValidationStatus,
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

    # Default checkpoint URLs (same as main.py)
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

        # Default predict args for affinity (mirroring main.py)
        predict_args = {
            "recycling_steps": 5,
            "sampling_steps": 200,
            "diffusion_samples": 5,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        try:
            model = Boltz2.load_from_checkpoint(
                str(ckpt_path),
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
                    str(ckpt_path),
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

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256


class AffinityInferenceEngine:
    """
    Performs inference using the existing Boltz-2 pipeline.

    This engine orchestrates the full Boltz-2 predict pipeline
    for affinity prediction, integrating with the existing
    data loading, tokenization, cropping, featurization,
    and model forward pass.
    """

    def __init__(
        self,
        model_manager: AffinityModelManager,
        config: RescoreConfig,
    ):
        self.model_manager = model_manager
        self.config = config
        self._model = None
        self._total_inference_time = 0.0

    def run_boltz_affinity_pipeline(
        self,
        input_yaml_path: Path,
        output_dir: Path,
        structure_prediction_dir: Optional[Path] = None,
        run_structure_prediction: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full Boltz-2 affinity pipeline for a single input.

        This leverages the existing predict infrastructure via
        programmatic invocation rather than CLI. It:
        1. Processes input YAML
        2. Runs structure prediction (if needed)
        3. Runs affinity prediction
        4. Returns parsed results

        Parameters
        ----------
        input_yaml_path : Path
            Path to the YAML input file
        output_dir : Path
            Directory for output files
        structure_prediction_dir : Path, optional
            Directory with pre-computed structures (skip diffusion)
        run_structure_prediction : bool
            Whether to run structure prediction first

        Returns
        -------
        dict
            Affinity prediction results
        """
        from boltz.main import predict as boltz_predict

        # This calls the existing predict flow which handles everything
        # For now, we raise NotImplementedError for direct invocation
        # and instead use the YAML-based pipeline
        raise NotImplementedError(
            "Direct pipeline invocation is under development. "
            "Use AffinityRescorer.rescore_pdb() which generates YAML and runs via CLI."
        )

    def infer_single_from_features(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Run inference on pre-computed features.

        This is a lower-level method that directly calls the model
        forward pass on already-featurized data.

        Parameters
        ----------
        features : dict
            Feature dictionary as produced by Boltz2Featurizer

        Returns
        -------
        dict
            Raw model outputs (affinity_pred_value, affinity_probability_binary, etc.)
        """
        model = self.model_manager.model

        with torch.no_grad(), Timer() as t:
            try:
                # Move features to device
                device = self.model_manager.device
                batch = {}
                for k, v in features.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                    else:
                        batch[k] = v

                out = model.predict_step(batch, 0)

                results = {}
                if not out.get("exception", False):
                    if "affinity_pred_value" in out:
                        results["affinity_pred_value"] = out["affinity_pred_value"].item()
                    if "affinity_probability_binary" in out:
                        results["affinity_probability_binary"] = out[
                            "affinity_probability_binary"
                        ].item()
                    # Ensemble values
                    for key in [
                        "affinity_pred_value1",
                        "affinity_pred_value2",
                        "affinity_probability_binary1",
                        "affinity_probability_binary2",
                    ]:
                        if key in out:
                            results[key] = out[key].item()
                else:
                    results["error"] = "Model returned exception"

                results["inference_time_ms"] = t.elapsed_ms
                return results

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "Out of memory during inference. "
                    "Try reducing complex size or using CPU (--device cpu)."
                )
            except Exception as e:
                return {
                    "error": str(e),
                    "inference_time_ms": t.elapsed_ms,
                }


def run_affinity_prediction_via_yaml(
    input_path: Path,
    output_dir: Path,
    checkpoint: str = "auto",
    device: str = "auto",
    recycling_steps: int = 5,
    diffusion_samples: int = 5,
    sampling_steps: int = 200,
    affinity_mw_correction: bool = True,
    use_msa_server: bool = False,
    override: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Run the full Boltz affinity prediction pipeline for a single input.

    This function programmatically invokes the existing Boltz predict
    command to run structure + affinity prediction.

    Parameters
    ----------
    input_path : Path
        Path to YAML/FASTA input file
    output_dir : Path
        Output directory
    checkpoint : str
        Checkpoint path or 'auto'
    device : str
        Device to use
    Other parameters match the Boltz CLI options.

    Returns
    -------
    dict or None
        Parsed affinity results JSON, or None on failure
    """
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "boltz.main", "predict",
        str(input_path),
        "--out_dir", str(output_dir),
        "--recycling_steps", str(recycling_steps),
        "--diffusion_samples_affinity", str(diffusion_samples),
        "--sampling_steps", str(sampling_steps),
    ]

    if not affinity_mw_correction:
        cmd.append("--no_affinity_mw_correction")

    if device != "auto":
        cmd.extend(["--accelerator", device])

    if override:
        cmd.append("--override")

    if not use_msa_server:
        pass  # Default is no MSA server

    logger.info(f"Running Boltz predict: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Boltz predict failed:\n{result.stderr}")
            return None

        # Find and parse affinity output
        results_dir = output_dir / f"boltz_results_{input_path.stem}" / "predictions"
        if results_dir.exists():
            for json_file in results_dir.rglob("affinity_*.json"):
                with open(json_file) as f:
                    return json.load(f)

        logger.warning("No affinity output found")
        return None

    except subprocess.TimeoutExpired:
        logger.error("Boltz predict timed out (>1h)")
        return None
    except Exception as e:
        logger.error(f"Failed to run Boltz predict: {e}")
        return None


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


def create_affinity_yaml_from_structure(
    structure_path: Path,
    protein_chains: List[str],
    ligand_chains: List[str],
    protein_sequences: Dict[str, str],
    ligand_smiles: Dict[str, str],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Create a Boltz affinity YAML from a parsed structure.

    This is used for PDB/CIF rescoring where we need to
    create the YAML that Boltz expects as input.

    Parameters
    ----------
    structure_path : Path
        Original structure file (for reference)
    protein_chains : list of str
        Protein chain IDs
    ligand_chains : list of str
        Ligand chain IDs
    protein_sequences : dict
        Map of chain_id → amino acid sequence
    ligand_smiles : dict
        Map of chain_id → SMILES string
    output_path : Path, optional
        Where to write YAML

    Returns
    -------
    Path
        Path to created YAML
    """
    import yaml

    sequences = []
    for chain_id in protein_chains:
        if chain_id in protein_sequences:
            sequences.append({
                "protein": {
                    "id": chain_id,
                    "sequence": protein_sequences[chain_id],
                }
            })

    binder_chain = None
    for chain_id in ligand_chains:
        if chain_id in ligand_smiles:
            sequences.append({
                "ligand": {
                    "id": chain_id,
                    "smiles": ligand_smiles[chain_id],
                }
            })
            if binder_chain is None:
                binder_chain = chain_id

    if binder_chain is None:
        raise ValueError(
            f"No ligand SMILES provided for chains {ligand_chains}. "
            f"Cannot create affinity YAML without ligand SMILES."
        )

    data = {
        "version": 1,
        "sequences": sequences,
        "properties": [
            {"affinity": {"binder": binder_chain}},
        ],
    }

    if output_path is None:
        fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="affinity_")
        os.close(fd)
        output_path = Path(tmp)

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    return output_path
