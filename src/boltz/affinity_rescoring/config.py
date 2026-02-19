"""
Configuration management for affinity rescoring.

Supports YAML-based configuration with environment variable overrides
and multiple deployment modes (development, testing, production).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from boltz.affinity_rescoring.models import (
    DeviceOption,
    OutputFormat,
    RescoreConfig,
    ValidationLevel,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "mode": "development",
    "model": {
        "checkpoint": "auto",
        "device": "auto",
    },
    "inference": {
        "recycling_steps": 5,
        "diffusion_samples": 5,
        "sampling_steps": 200,
        "affinity_mw_correction": True,
    },
    "validation": {
        "level": "moderate",
    },
    "output": {
        "format": "csv",
        "include_metadata": True,
        "include_diagnostics": True,
    },
    "processing": {
        "max_tokens": 256,
        "max_atoms": 2048,
        "max_tokens_protein": 200,
    },
    "logging": {
        "level": "INFO",
    },
}


def load_config(config_path: Optional[str | Path] = None) -> RescoreConfig:
    """
    Load configuration from YAML file with environment variable overrides.

    Priority (highest to lowest):
    1. Environment variables (BOLTZ_RESCORE_*)
    2. Config file values
    3. Default values

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config file. Searches for 'rescore_config.yaml'
        in current directory if not provided.

    Returns
    -------
    RescoreConfig
        Validated configuration object
    """
    config_data = dict(DEFAULT_CONFIG)

    # Load from file
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                file_config = yaml.safe_load(f)
            if file_config:
                _deep_merge(config_data, file_config)
            logger.info(f"Loaded config from {path}")
    else:
        # Auto-discover config
        for name in ["rescore_config.yaml", "rescore_config.yml", ".rescore.yaml"]:
            if Path(name).exists():
                with open(name) as f:
                    file_config = yaml.safe_load(f)
                if file_config:
                    _deep_merge(config_data, file_config)
                logger.info(f"Auto-discovered config: {name}")
                break

    # Apply environment overrides
    _apply_env_overrides(config_data)

    # Build RescoreConfig
    return RescoreConfig(
        checkpoint=config_data.get("model", {}).get("checkpoint", "auto"),
        device=DeviceOption(config_data.get("model", {}).get("device", "auto")),
        recycling_steps=config_data.get("inference", {}).get("recycling_steps", 5),
        diffusion_samples=config_data.get("inference", {}).get("diffusion_samples", 5),
        sampling_steps=config_data.get("inference", {}).get("sampling_steps", 200),
        affinity_mw_correction=config_data.get("inference", {}).get("affinity_mw_correction", True),
        validation_level=ValidationLevel(
            config_data.get("validation", {}).get("level", "moderate")
        ),
        output_format=OutputFormat(
            config_data.get("output", {}).get("format", "csv")
        ),
        include_metadata=config_data.get("output", {}).get("include_metadata", True),
        include_diagnostics=config_data.get("output", {}).get("include_diagnostics", True),
        max_tokens=config_data.get("processing", {}).get("max_tokens", 256),
        max_atoms=config_data.get("processing", {}).get("max_atoms", 2048),
        max_tokens_protein=config_data.get("processing", {}).get("max_tokens_protein", 200),
        log_level=config_data.get("logging", {}).get("level", "INFO"),
    )


def save_config(config: RescoreConfig, output_path: str | Path) -> None:
    """Save configuration to YAML file."""
    config_data = {
        "mode": "development",
        "model": {
            "checkpoint": config.checkpoint,
            "device": config.device.value,
        },
        "inference": {
            "recycling_steps": config.recycling_steps,
            "diffusion_samples": config.diffusion_samples,
            "sampling_steps": config.sampling_steps,
            "affinity_mw_correction": config.affinity_mw_correction,
        },
        "validation": {
            "level": config.validation_level.value,
        },
        "output": {
            "format": config.output_format.value,
            "include_metadata": config.include_metadata,
            "include_diagnostics": config.include_diagnostics,
        },
        "processing": {
            "max_tokens": config.max_tokens,
            "max_atoms": config.max_atoms,
            "max_tokens_protein": config.max_tokens_protein,
        },
        "logging": {
            "level": config.log_level,
        },
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Config saved to {path}")


def generate_default_config(output_path: str | Path = "rescore_config.yaml") -> Path:
    """Generate a default configuration file."""
    path = Path(output_path)

    with open(path, "w") as f:
        f.write("# Boltz Affinity Rescoring Configuration\n")
        f.write("# See IMPLEMENTATION_SPECIFICATION.md for details\n\n")
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

    return path


def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override dict into base dict (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_env_overrides(config: Dict) -> None:
    """Apply environment variable overrides to config."""
    env_map = {
        "BOLTZ_RESCORE_CHECKPOINT": ("model", "checkpoint"),
        "BOLTZ_RESCORE_DEVICE": ("model", "device"),
        "BOLTZ_RESCORE_RECYCLING_STEPS": ("inference", "recycling_steps"),
        "BOLTZ_RESCORE_DIFFUSION_SAMPLES": ("inference", "diffusion_samples"),
        "BOLTZ_RESCORE_VALIDATION": ("validation", "level"),
        "BOLTZ_RESCORE_OUTPUT_FORMAT": ("output", "format"),
        "BOLTZ_RESCORE_LOG_LEVEL": ("logging", "level"),
    }

    for env_var, (section, key) in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            if section not in config:
                config[section] = {}
            # Try to convert to appropriate type
            if key in ("recycling_steps", "diffusion_samples", "sampling_steps",
                       "max_tokens", "max_atoms", "max_tokens_protein"):
                try:
                    value = int(value)
                except ValueError:
                    pass
            elif key == "affinity_mw_correction":
                value = value.lower() in ("true", "1", "yes")
            config[section][key] = value
            logger.debug(f"Env override: {env_var}={value}")
