"""
Core data models for the affinity rescoring system.

Defines Pydantic models for validation, dataclasses for results,
and typed structures used across all layers.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Enums ───────────────────────────────────────────────────────────────────


class ValidationLevel(str, Enum):
    """Validation strictness level."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class OutputFormat(str, Enum):
    """Supported output formats."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    SQLITE = "sqlite"
    EXCEL = "excel"


class DeviceOption(str, Enum):
    """Device selection."""
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class ValidationStatus(str, Enum):
    """Per-complex validation status."""
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    FAILED = "FAILED"


class ChainType(str, Enum):
    """Chain classification."""
    PROTEIN = "protein"
    LIGAND = "ligand"
    UNKNOWN = "unknown"


# ─── Validation Issue ────────────────────────────────────────────────────────


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: str
    severity: str  # "error", "warning", "info"
    message: str
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


# ─── Validation Report ───────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    """Aggregated validation results."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def add_issue(self, code: str, severity: str, message: str,
                  details: Optional[Dict[str, Any]] = None) -> None:
        self.issues.append(ValidationIssue(code, severity, message, details))
        if severity == "error":
            self.is_valid = False

    def add_recovery(self, action: str) -> None:
        self.recovery_actions.append(action)


# ─── Chain Assignment ────────────────────────────────────────────────────────


@dataclass
class ChainAssignment:
    """Result of chain identification."""
    protein_chains: List[str]
    ligand_chains: List[str]
    confidence: float
    reasoning: str
    chain_types: Dict[str, ChainType] = field(default_factory=dict)


# ─── Atom / Structure Info ───────────────────────────────────────────────────


@dataclass
class AtomInfo:
    """Per-atom information."""
    index: int
    name: str
    element: str
    x: float
    y: float
    z: float
    chain_id: str
    residue_name: str
    residue_number: int
    occupancy: float = 1.0
    b_factor: float = 0.0
    is_hetatm: bool = False


@dataclass
class LigandStructure:
    """Extracted ligand structure from MOL2 or structure file."""
    name: str
    atoms: List[AtomInfo]
    bonds: List[tuple]  # [(atom_idx_1, atom_idx_2, bond_type), ...]
    partial_charges: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)

    @property
    def elements(self) -> List[str]:
        return [a.element for a in self.atoms]


# ─── Affinity Result ─────────────────────────────────────────────────────────


@dataclass
class AffinityResult:
    """Complete result with metadata for a single complex."""

    # Identification
    id: str
    source_file: str = ""

    # Core predictions
    affinity_pred: float = float("nan")
    affinity_std: float = float("nan")
    affinity_probability_binary: float = float("nan")

    # Ensemble values (if applicable)
    affinity_pred_value1: Optional[float] = None
    affinity_pred_value2: Optional[float] = None
    affinity_probability_binary1: Optional[float] = None
    affinity_probability_binary2: Optional[float] = None

    # Complex metadata
    protein_chain: str = ""
    ligand_chains: List[str] = field(default_factory=list)
    protein_residue_count: int = 0
    ligand_atom_count: int = 0

    # Processing metadata
    processing_time_ms: float = 0.0
    featurization_time_ms: float = 0.0
    inference_time_ms: float = 0.0

    # Validation
    validation_status: ValidationStatus = ValidationStatus.SUCCESS
    validation_issues: List[str] = field(default_factory=list)

    # Warnings/flags
    warnings: List[str] = field(default_factory=list)
    error_message: str = ""

    # Reproducibility
    model_checkpoint: str = ""
    model_checkpoint_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for tabular export."""
        return {
            "id": self.id,
            "source_file": self.source_file,
            "affinity_pred": self.affinity_pred,
            "affinity_std": self.affinity_std,
            "affinity_probability_binary": self.affinity_probability_binary,
            "protein_chain": self.protein_chain,
            "ligand_chains": ",".join(self.ligand_chains),
            "protein_residue_count": self.protein_residue_count,
            "ligand_atom_count": self.ligand_atom_count,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "featurization_time_ms": round(self.featurization_time_ms, 2),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "validation_status": self.validation_status.value,
            "validation_issues": ";".join(self.validation_issues),
            "warnings": ";".join(self.warnings),
            "error_message": self.error_message,
        }


@dataclass
class LigandScore:
    """Score for a single ligand in receptor-based rescoring."""

    # Identification
    ligand_name: str

    # Core predictions
    affinity_score: float = float("nan")
    affinity_uncertainty: float = float("nan")
    confidence: float = float("nan")

    # Complex metadata
    n_atoms: int = 0
    receptor_contacts: Optional[int] = None
    rotatable_bonds: Optional[int] = None

    # Validation
    validation_status: ValidationStatus = ValidationStatus.SUCCESS
    validation_issues: str = ""
    error_message: str = ""

    # Timing
    processing_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ligand_name": self.ligand_name,
            "affinity_score": self.affinity_score,
            "affinity_uncertainty": self.affinity_uncertainty,
            "confidence": self.confidence,
            "n_atoms": self.n_atoms,
            "validation_status": self.validation_status.value,
            "validation_issues": self.validation_issues,
            "receptor_contacts": self.receptor_contacts,
            "rotatable_bonds": self.rotatable_bonds,
            "processing_ms": round(self.processing_ms, 2),
            "error_message": self.error_message,
        }


# ─── Batch Result Summary ────────────────────────────────────────────────────


@dataclass
class BatchSummary:
    """Summary statistics for a batch of predictions."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    warnings_count: int = 0
    mean_affinity: float = float("nan")
    std_affinity: float = float("nan")
    min_affinity: float = float("nan")
    max_affinity: float = float("nan")
    inference_time_total_s: float = 0.0
    throughput_complexes_per_second: float = 0.0


# ─── Configuration ───────────────────────────────────────────────────────────


class RescoreConfig(BaseModel):
    """Configuration for affinity rescoring."""

    # Model
    checkpoint: str = "auto"
    device: DeviceOption = DeviceOption.AUTO

    # Inference
    recycling_steps: int = 5
    diffusion_samples: int = 5
    sampling_steps: int = 200
    affinity_mw_correction: bool = True

    # Validation
    validation_level: ValidationLevel = ValidationLevel.MODERATE

    # Output
    output_format: OutputFormat = OutputFormat.CSV
    include_metadata: bool = True
    include_diagnostics: bool = True

    # Processing
    max_tokens: int = 256
    max_atoms: int = 2048
    max_tokens_protein: int = 200

    # Logging
    log_level: str = "INFO"

    @field_validator("recycling_steps")
    @classmethod
    def validate_recycling_steps(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError(f"recycling_steps must be 1-20, got {v}")
        return v

    @field_validator("diffusion_samples")
    @classmethod
    def validate_diffusion_samples(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"diffusion_samples must be 1-100, got {v}")
        return v


# ─── Utility Functions ───────────────────────────────────────────────────────


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    path = Path(filepath)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class Timer:
    """Simple context-manager timer."""

    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
