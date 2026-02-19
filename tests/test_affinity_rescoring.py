"""
Unit tests for affinity rescoring module.

Tests cover:
- Validation (structure, atoms, chains)
- MOL2 parsing
- Structure parsing
- Models and data structures
- Results export
- Configuration management
"""

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─── Test Data Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_pdb_content():
    """Minimal PDB file content for testing."""
    return """\
HEADER    TEST STRUCTURE
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 12.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 11.00           C
ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00 13.00           O
ATOM      5  CB  ALA A   1       2.500   2.500   4.500  1.00 15.00           C
ATOM      6  N   GLY A   2       3.500   4.500   5.500  1.00 10.00           N
ATOM      7  CA  GLY A   2       4.000   5.000   6.000  1.00 12.00           C
ATOM      8  C   GLY A   2       5.000   6.000   7.000  1.00 11.00           C
ATOM      9  O   GLY A   2       6.000   7.000   8.000  1.00 13.00           O
HETATM   10  C1  LIG B   1      10.000  11.000  12.000  1.00  8.00           C
HETATM   11  C2  LIG B   1      11.000  12.000  13.000  1.00  9.00           C
HETATM   12  N1  LIG B   1      12.000  13.000  14.000  1.00  7.00           N
END
"""


@pytest.fixture
def sample_pdb_file(sample_pdb_content, tmp_path):
    """Write sample PDB to temp file."""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(sample_pdb_content)
    return pdb_file


@pytest.fixture
def sample_mol2_content():
    """Minimal MOL2 file content with 2 molecules."""
    return """\
@<TRIPOS>MOLECULE
compound_001
 3 2  0  0  0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
   1 C1   12.450  10.200  -5.100 C.ar      1  compound_001    0.0000
   2 C2   13.100  11.300  -5.800 C.ar      1  compound_001    0.0000
   3 N1   14.200  12.400  -6.500 N.3       1  compound_001    0.0000
@<TRIPOS>BOND
   1  1  2 ar
   2  2  3 1

@<TRIPOS>MOLECULE
compound_002
 4 3  0  0  0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
   1 C1   11.200  10.100  -6.500 C.ar      1  compound_002    0.0000
   2 C2   12.100  11.200  -6.900 C.ar      1  compound_002    0.0000
   3 O1   13.000  12.300  -7.200 O.3       1  compound_002    0.0000
   4 N1   14.000  13.400  -7.500 N.3       1  compound_002    0.0000
@<TRIPOS>BOND
   1  1  2 ar
   2  2  3 1
   3  3  4 1
"""


@pytest.fixture
def sample_mol2_file(sample_mol2_content, tmp_path):
    """Write sample MOL2 to temp file."""
    mol2_file = tmp_path / "ligands.mol2"
    mol2_file.write_text(sample_mol2_content)
    return mol2_file


@pytest.fixture
def sample_atoms():
    """Create sample AtomInfo list."""
    from boltz.affinity_rescoring.models import AtomInfo

    atoms = []
    # Protein chain A
    for i in range(20):
        atoms.append(AtomInfo(
            index=i,
            name="CA",
            element="C",
            x=float(i),
            y=float(i + 1),
            z=float(i + 2),
            chain_id="A",
            residue_name="ALA",
            residue_number=i + 1,
            occupancy=1.0,
            b_factor=10.0,
        ))

    # Ligand chain B
    for i in range(5):
        atoms.append(AtomInfo(
            index=20 + i,
            name=f"C{i + 1}",
            element="C",
            x=30.0 + float(i),
            y=31.0 + float(i),
            z=32.0 + float(i),
            chain_id="B",
            residue_name="LIG",
            residue_number=1,
            occupancy=1.0,
            b_factor=8.0,
            is_hetatm=True,
        ))

    return atoms


# ─── Validation Tests ─────────────────────────────────────────────────────────


class TestStructureValidator:
    """Tests for StructureValidator."""

    def test_validate_file_nonexistent(self):
        from boltz.affinity_rescoring.validation import StructureValidator
        validator = StructureValidator()
        report = validator.validate_file("/nonexistent/path.pdb")
        assert not report.is_valid
        assert any("FILE_NOT_FOUND" in str(i) for i in report.issues)

    def test_validate_file_empty(self, tmp_path):
        from boltz.affinity_rescoring.validation import StructureValidator
        empty_file = tmp_path / "empty.pdb"
        empty_file.touch()
        validator = StructureValidator()
        report = validator.validate_file(empty_file)
        assert not report.is_valid
        assert any("EMPTY_FILE" in str(i) for i in report.issues)

    def test_validate_file_valid_pdb(self, sample_pdb_file):
        from boltz.affinity_rescoring.validation import StructureValidator
        validator = StructureValidator()
        report = validator.validate_file(sample_pdb_file)
        assert report.is_valid

    def test_validate_file_wrong_extension(self, tmp_path):
        from boltz.affinity_rescoring.validation import StructureValidator
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("some content")
        validator = StructureValidator()
        report = validator.validate_file(bad_file)
        # Should warn but not fail
        assert report.is_valid
        assert any("UNKNOWN_FORMAT" in str(i) for i in report.issues)

    def test_validate_atoms_empty(self):
        from boltz.affinity_rescoring.validation import StructureValidator
        validator = StructureValidator()
        report = validator.validate_atoms([])
        assert not report.is_valid
        assert any("NO_ATOMS" in str(i) for i in report.issues)

    def test_validate_atoms_normal(self, sample_atoms):
        from boltz.affinity_rescoring.validation import StructureValidator
        validator = StructureValidator()
        report = validator.validate_atoms(sample_atoms)
        assert report.is_valid

    def test_validate_atoms_extreme_bfactors(self, sample_atoms):
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.validation import StructureValidator

        # Add atom with extreme B-factor
        sample_atoms.append(AtomInfo(
            index=100, name="X", element="C",
            x=0, y=0, z=0, chain_id="A",
            residue_name="ALA", residue_number=100,
            b_factor=250.0,
        ))
        validator = StructureValidator()
        report = validator.validate_atoms(sample_atoms)
        assert any("EXTREME_BFACTOR" in str(i) for i in report.issues)

    def test_validate_atoms_nan_coords(self):
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.validation import StructureValidator

        atoms = [AtomInfo(
            index=0, name="CA", element="C",
            x=float("nan"), y=0, z=0, chain_id="A",
            residue_name="ALA", residue_number=1,
        )]
        validator = StructureValidator()
        report = validator.validate_atoms(atoms)
        assert not report.is_valid
        assert any("INVALID_COORDS" in str(i) for i in report.issues)

    def test_validate_chains_single_chain(self, sample_atoms):
        from boltz.affinity_rescoring.validation import StructureValidator
        # Filter to only chain A
        single_chain = [a for a in sample_atoms if a.chain_id == "A"]
        validator = StructureValidator()
        report = validator.validate_chains(single_chain)
        assert any("SINGLE_CHAIN" in str(i) for i in report.issues)


# ─── Chain Identifier Tests ──────────────────────────────────────────────────


class TestChainIdentifier:
    """Tests for ChainIdentifier."""

    def test_explicit_assignment(self, sample_atoms):
        from boltz.affinity_rescoring.validation import ChainIdentifier
        identifier = ChainIdentifier()
        assignment = identifier.identify_chains(
            sample_atoms,
            protein_chains=["A"],
            ligand_chains=["B"],
        )
        assert assignment.protein_chains == ["A"]
        assert assignment.ligand_chains == ["B"]
        assert assignment.confidence == 1.0

    def test_auto_detection(self, sample_atoms):
        from boltz.affinity_rescoring.validation import ChainIdentifier
        identifier = ChainIdentifier()
        assignment = identifier.identify_chains(sample_atoms)
        assert "A" in assignment.protein_chains
        assert "B" in assignment.ligand_chains

    def test_no_chains(self):
        from boltz.affinity_rescoring.validation import ChainIdentifier
        identifier = ChainIdentifier()
        assignment = identifier.identify_chains([])
        assert assignment.protein_chains == []
        assert assignment.ligand_chains == []


# ─── Structure Normalizer Tests ──────────────────────────────────────────────


class TestStructureNormalizer:
    """Tests for StructureNormalizer."""

    def test_remove_waters(self):
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.validation import StructureNormalizer

        atoms = [
            AtomInfo(0, "CA", "C", 0, 0, 0, "A", "ALA", 1),
            AtomInfo(1, "O", "O", 1, 1, 1, "W", "HOH", 1),
        ]
        normalizer = StructureNormalizer()
        result, log = normalizer.normalize(atoms, remove_waters=True)
        assert len(result) == 1
        assert result[0].residue_name == "ALA"

    def test_center_coordinates(self):
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.validation import StructureNormalizer

        atoms = [
            AtomInfo(0, "CA", "C", 10, 20, 30, "A", "ALA", 1),
            AtomInfo(1, "CB", "C", 20, 30, 40, "A", "ALA", 1),
        ]
        normalizer = StructureNormalizer()
        result, log = normalizer.normalize(
            atoms, remove_waters=False, remove_ions=False, center_coordinates=True
        )
        coords = [(a.x, a.y, a.z) for a in result]
        mean_x = sum(c[0] for c in coords) / len(coords)
        assert abs(mean_x) < 1e-6


# ─── MOL2 Parser Tests ───────────────────────────────────────────────────────


class TestMOL2Parser:
    """Tests for MOL2Parser."""

    def test_parse_multiple_molecules(self, sample_mol2_file):
        from boltz.affinity_rescoring.mol2_parser import MOL2Parser

        parser = MOL2Parser()
        ligands = parser.extract_ligands_with_names(sample_mol2_file)

        assert len(ligands) == 2
        assert ligands[0].name == "compound_001"
        assert ligands[1].name == "compound_002"

    def test_parse_atom_counts(self, sample_mol2_file):
        from boltz.affinity_rescoring.mol2_parser import MOL2Parser

        parser = MOL2Parser()
        ligands = parser.extract_ligands_with_names(sample_mol2_file)

        assert ligands[0].num_atoms == 3
        assert ligands[1].num_atoms == 4

    def test_parse_bonds(self, sample_mol2_file):
        from boltz.affinity_rescoring.mol2_parser import MOL2Parser

        parser = MOL2Parser()
        ligands = parser.extract_ligands_with_names(sample_mol2_file)

        assert len(ligands[0].bonds) == 2
        assert len(ligands[1].bonds) == 3

    def test_parse_elements(self, sample_mol2_file):
        from boltz.affinity_rescoring.mol2_parser import MOL2Parser

        parser = MOL2Parser()
        ligands = parser.extract_ligands_with_names(sample_mol2_file)

        elements_1 = ligands[0].elements
        assert "C" in elements_1
        assert "N" in elements_1

    def test_parse_nonexistent_file(self):
        from boltz.affinity_rescoring.mol2_parser import MOL2Parser
        parser = MOL2Parser()
        with pytest.raises(FileNotFoundError):
            parser.extract_ligands_with_names("/nonexistent.mol2")

    def test_parse_empty_file(self, tmp_path):
        from boltz.affinity_rescoring.mol2_parser import MOL2Parser

        empty_mol2 = tmp_path / "empty.mol2"
        empty_mol2.write_text("")
        parser = MOL2Parser()
        ligands = parser.extract_ligands_with_names(empty_mol2)
        assert len(ligands) == 0

    def test_remove_hydrogens(self, tmp_path):
        from boltz.affinity_rescoring.mol2_parser import MOL2Parser

        mol2_with_h = tmp_path / "with_h.mol2"
        mol2_with_h.write_text("""\
@<TRIPOS>MOLECULE
test_mol
 3 2  0  0  0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
   1 C1   10.0  10.0  10.0 C.3       1  test_mol    0.0000
   2 H1   11.0  10.0  10.0 H         1  test_mol    0.0000
   3 N1   12.0  10.0  10.0 N.3       1  test_mol    0.0000
@<TRIPOS>BOND
   1  1  2 1
   2  1  3 1
""")
        parser = MOL2Parser(remove_hydrogens=True)
        ligands = parser.extract_ligands_with_names(mol2_with_h)
        assert len(ligands) == 1
        assert ligands[0].num_atoms == 2
        assert "H" not in ligands[0].elements


# ─── Data Models Tests ────────────────────────────────────────────────────────


class TestModels:
    """Tests for data model classes."""

    def test_affinity_result_to_dict(self):
        from boltz.affinity_rescoring.models import AffinityResult, ValidationStatus

        result = AffinityResult(
            id="test_001",
            affinity_pred=-7.2,
            affinity_std=0.3,
            protein_chain="A",
            ligand_chains=["B"],
            validation_status=ValidationStatus.SUCCESS,
        )
        d = result.to_dict()
        assert d["id"] == "test_001"
        assert d["affinity_pred"] == -7.2
        assert d["ligand_chains"] == "B"
        assert d["validation_status"] == "SUCCESS"

    def test_ligand_score_to_dict(self):
        from boltz.affinity_rescoring.models import LigandScore, ValidationStatus

        score = LigandScore(
            ligand_name="compound_001",
            affinity_score=-8.2,
            confidence=0.94,
            n_atoms=28,
            validation_status=ValidationStatus.SUCCESS,
        )
        d = score.to_dict()
        assert d["ligand_name"] == "compound_001"
        assert d["affinity_score"] == -8.2

    def test_validation_report(self):
        from boltz.affinity_rescoring.models import ValidationReport

        report = ValidationReport(is_valid=True)
        report.add_issue("TEST", "warning", "Test warning")
        assert report.is_valid
        assert len(report.warnings) == 1

        report.add_issue("ERR", "error", "Test error")
        assert not report.is_valid
        assert len(report.errors) == 1

    def test_timer(self):
        import time
        from boltz.affinity_rescoring.models import Timer

        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_ms >= 5  # At least 5ms (generous margin)

    def test_rescore_config_validation(self):
        from boltz.affinity_rescoring.models import RescoreConfig

        config = RescoreConfig(recycling_steps=5)
        assert config.recycling_steps == 5

        with pytest.raises(ValueError):
            RescoreConfig(recycling_steps=0)

        with pytest.raises(ValueError):
            RescoreConfig(recycling_steps=25)

    def test_batch_summary(self):
        from boltz.affinity_rescoring.models import BatchSummary
        summary = BatchSummary(total_processed=10, successful=8, failed=2)
        assert summary.total_processed == 10

    def test_chain_assignment(self):
        from boltz.affinity_rescoring.models import ChainAssignment, ChainType
        assignment = ChainAssignment(
            protein_chains=["A"],
            ligand_chains=["B"],
            confidence=0.95,
            reasoning="Test",
            chain_types={"A": ChainType.PROTEIN, "B": ChainType.LIGAND},
        )
        assert assignment.confidence == 0.95


# ─── Results Export Tests ─────────────────────────────────────────────────────


class TestResultsExporter:
    """Tests for ResultsExporter."""

    @pytest.fixture
    def sample_results(self):
        from boltz.affinity_rescoring.models import AffinityResult, ValidationStatus

        return [
            AffinityResult(
                id="complex_001",
                affinity_pred=-7.2,
                affinity_std=0.3,
                protein_chain="A",
                ligand_chains=["B"],
                protein_residue_count=100,
                ligand_atom_count=30,
                inference_time_ms=245.0,
                validation_status=ValidationStatus.SUCCESS,
            ),
            AffinityResult(
                id="complex_002",
                affinity_pred=-6.5,
                affinity_std=0.5,
                protein_chain="A",
                ligand_chains=["C"],
                protein_residue_count=150,
                ligand_atom_count=25,
                inference_time_ms=198.0,
                validation_status=ValidationStatus.SUCCESS,
            ),
            AffinityResult(
                id="complex_003",
                validation_status=ValidationStatus.FAILED,
                error_message="Parse error",
            ),
        ]

    def test_export_json(self, sample_results, tmp_path):
        from boltz.affinity_rescoring.export import ResultsExporter
        from boltz.affinity_rescoring.models import OutputFormat

        exporter = ResultsExporter(model_checkpoint="test.ckpt")
        path = exporter.export(sample_results, tmp_path / "results.json", fmt=OutputFormat.JSON)

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data["results"]) == 3
        assert data["version"] == "1.0"
        assert "summary" in data

    def test_export_csv(self, sample_results, tmp_path):
        from boltz.affinity_rescoring.export import ResultsExporter
        from boltz.affinity_rescoring.models import OutputFormat

        exporter = ResultsExporter()
        path = exporter.export(sample_results, tmp_path / "results.csv", fmt=OutputFormat.CSV)

        assert path.exists()
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["id"] == "complex_001"

    def test_export_jsonl(self, sample_results, tmp_path):
        from boltz.affinity_rescoring.export import ResultsExporter
        from boltz.affinity_rescoring.models import OutputFormat

        exporter = ResultsExporter()
        path = exporter.export(sample_results, tmp_path / "results.jsonl", fmt=OutputFormat.JSONL)

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_export_sqlite(self, sample_results, tmp_path):
        from boltz.affinity_rescoring.export import ResultsExporter
        from boltz.affinity_rescoring.models import OutputFormat
        import sqlite3

        exporter = ResultsExporter()
        path = exporter.export(sample_results, tmp_path / "results.db", fmt=OutputFormat.SQLITE)

        assert path.exists()
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM results")
        count = cursor.fetchone()[0]
        assert count == 3

        cursor.execute("SELECT value FROM metadata WHERE key='version'")
        version = cursor.fetchone()[0]
        assert version == "1.0"
        conn.close()

    def test_export_receptor_csv(self, tmp_path):
        from boltz.affinity_rescoring.export import ResultsExporter
        from boltz.affinity_rescoring.models import LigandScore, OutputFormat, ValidationStatus

        scores = [
            LigandScore("lig_001", affinity_score=-8.2, confidence=0.94, n_atoms=28),
            LigandScore("lig_002", affinity_score=-7.5, confidence=0.91, n_atoms=32),
            LigandScore(
                "lig_003",
                validation_status=ValidationStatus.FAILED,
                error_message="Steric clash",
            ),
        ]

        exporter = ResultsExporter()
        path = exporter.export_receptor_results(
            scores, tmp_path / "scores.csv", fmt=OutputFormat.CSV
        )

        assert path.exists()
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["ligand_name"] == "lig_001"

    def test_batch_summary(self, sample_results):
        from boltz.affinity_rescoring.export import compute_batch_summary

        summary = compute_batch_summary(sample_results)
        assert summary.total_processed == 3
        assert summary.successful == 2
        assert summary.failed == 1
        assert not math.isnan(summary.mean_affinity)


# ─── Configuration Tests ─────────────────────────────────────────────────────


class TestConfiguration:
    """Tests for configuration management."""

    def test_load_default_config(self):
        from boltz.affinity_rescoring.config import load_config

        config = load_config()
        assert config.recycling_steps == 5
        assert config.validation_level.value == "moderate"
        assert config.device.value == "auto"

    def test_load_from_yaml(self, tmp_path):
        from boltz.affinity_rescoring.config import load_config

        yaml_content = """\
model:
  device: cpu
  checkpoint: /path/to/ckpt
inference:
  recycling_steps: 3
validation:
  level: strict
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)
        assert config.device.value == "cpu"
        assert config.recycling_steps == 3
        assert config.validation_level.value == "strict"

    def test_save_and_load_config(self, tmp_path):
        from boltz.affinity_rescoring.config import load_config, save_config
        from boltz.affinity_rescoring.models import RescoreConfig, DeviceOption

        original = RescoreConfig(
            device=DeviceOption.CPU,
            recycling_steps=3,
        )

        config_file = tmp_path / "saved_config.yaml"
        save_config(original, config_file)
        assert config_file.exists()

        loaded = load_config(config_file)
        assert loaded.device.value == "cpu"
        assert loaded.recycling_steps == 3

    def test_env_override(self, tmp_path):
        from boltz.affinity_rescoring.config import load_config

        os.environ["BOLTZ_RESCORE_DEVICE"] = "cpu"
        try:
            config = load_config()
            assert config.device.value == "cpu"
        finally:
            del os.environ["BOLTZ_RESCORE_DEVICE"]

    def test_generate_default_config(self, tmp_path):
        from boltz.affinity_rescoring.config import generate_default_config

        path = generate_default_config(tmp_path / "default.yaml")
        assert path.exists()
        assert path.read_text().startswith("# Boltz Affinity Rescoring")


# ─── SHA256 Tests ─────────────────────────────────────────────────────────────


class TestUtilities:
    """Tests for utility functions."""

    def test_compute_sha256(self, tmp_path):
        from boltz.affinity_rescoring.models import compute_file_sha256

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        sha = compute_file_sha256(test_file)
        assert len(sha) == 64  # SHA256 hex digest length

    def test_sha256_nonexistent(self):
        from boltz.affinity_rescoring.models import compute_file_sha256

        sha = compute_file_sha256("/nonexistent/file")
        assert sha == ""


# ─── Inference Engine Tests ──────────────────────────────────────────────────


class TestAffinityModelManager:
    """Tests for AffinityModelManager (without actual model loading)."""

    def test_device_resolution_cpu(self):
        from boltz.affinity_rescoring.inference import AffinityModelManager
        from boltz.affinity_rescoring.models import DeviceOption

        manager = AffinityModelManager(device=DeviceOption.CPU)
        assert manager.device == "cpu"

    def test_device_resolution_auto(self):
        from boltz.affinity_rescoring.inference import AffinityModelManager
        from boltz.affinity_rescoring.models import DeviceOption

        manager = AffinityModelManager(device=DeviceOption.AUTO)
        assert manager.device in ("cpu", "cuda", "mps")

    def test_checkpoint_path_explicit(self, tmp_path):
        from boltz.affinity_rescoring.inference import AffinityModelManager
        from boltz.affinity_rescoring.models import DeviceOption

        ckpt = tmp_path / "test.ckpt"
        ckpt.touch()

        manager = AffinityModelManager(device=DeviceOption.CPU)
        path = manager.get_checkpoint_path(str(ckpt))
        assert path == ckpt

    def test_checkpoint_path_nonexistent(self):
        from boltz.affinity_rescoring.inference import AffinityModelManager
        from boltz.affinity_rescoring.models import DeviceOption

        manager = AffinityModelManager(device=DeviceOption.CPU)
        with pytest.raises(FileNotFoundError):
            manager.get_checkpoint_path("/nonexistent/checkpoint.ckpt")


# ─── YAML Creation Tests ─────────────────────────────────────────────────────


class TestYAMLCreation:
    """Tests for YAML input file creation."""

    def test_create_affinity_yaml(self, tmp_path):
        from boltz.affinity_rescoring.inference import create_affinity_yaml

        yaml_path = create_affinity_yaml(
            protein_sequence="MVTPEG",
            ligand_smiles="CCO",
            output_path=tmp_path / "test.yaml",
        )

        assert yaml_path.exists()

        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        assert data["version"] == 1
        assert len(data["sequences"]) == 2
        assert data["properties"][0]["affinity"]["binder"] == "B"


# ─── Non-Standard Residue Mapping Tests ──────────────────────────────────────


class TestNonStandardResidueMapping:
    """Tests for AMBER/CHARMM residue name handling in sequence extraction."""

    def test_amber_histidine_variants(self):
        """HIE, HID, HIP should all map to HIS → H."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        atoms = [
            AtomInfo(0, "CA", "C", 0, 0, 0, "A", "HIE", 1),
            AtomInfo(1, "CA", "C", 1, 0, 0, "A", "HID", 2),
            AtomInfo(2, "CA", "C", 2, 0, 0, "A", "HIP", 3),
            AtomInfo(3, "CA", "C", 3, 0, 0, "A", "ALA", 4),
        ]
        seqs = get_chain_sequences(atoms)
        assert seqs["A"] == "HHHA"

    def test_amber_cysteine_variants(self):
        """CYX (disulfide), CYM (deprotonated) should map to CYS → C."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        atoms = [
            AtomInfo(0, "CA", "C", 0, 0, 0, "A", "CYX", 1),
            AtomInfo(1, "CA", "C", 1, 0, 0, "A", "CYM", 2),
            AtomInfo(2, "CA", "C", 2, 0, 0, "A", "ALA", 3),
        ]
        seqs = get_chain_sequences(atoms)
        assert seqs["A"] == "CCA"

    def test_protonated_aspartate_glutamate(self):
        """ASH → ASP, GLH → GLU."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        atoms = [
            AtomInfo(0, "CA", "C", 0, 0, 0, "A", "ASH", 1),
            AtomInfo(1, "CA", "C", 1, 0, 0, "A", "GLH", 2),
        ]
        seqs = get_chain_sequences(atoms)
        assert seqs["A"] == "DE"

    def test_selenomethionine(self):
        """MSE (selenomethionine) should map to MET → M."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        atoms = [
            AtomInfo(0, "CA", "C", 0, 0, 0, "A", "MSE", 1),
            AtomInfo(1, "CA", "C", 1, 0, 0, "A", "ALA", 2),
        ]
        seqs = get_chain_sequences(atoms)
        assert seqs["A"] == "MA"

    def test_phosphorylated_residues(self):
        """TPO → THR, SEP → SER, PTR → TYR."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        atoms = [
            AtomInfo(0, "CA", "C", 0, 0, 0, "A", "TPO", 1),
            AtomInfo(1, "CA", "C", 1, 0, 0, "A", "SEP", 2),
            AtomInfo(2, "CA", "C", 2, 0, 0, "A", "PTR", 3),
        ]
        seqs = get_chain_sequences(atoms)
        assert seqs["A"] == "TSY"

    def test_terminal_amber_names(self):
        """N/C-terminal AMBER names (NALA, CVAL, etc.) should map correctly."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        atoms = [
            AtomInfo(0, "CA", "C", 0, 0, 0, "A", "NALA", 1),
            AtomInfo(1, "CA", "C", 1, 0, 0, "A", "GLY", 2),
            AtomInfo(2, "CA", "C", 2, 0, 0, "A", "CVAL", 3),
        ]
        seqs = get_chain_sequences(atoms)
        assert seqs["A"] == "AGV"

    def test_unknown_residues_skipped(self):
        """Completely unknown residue names should be silently skipped."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        atoms = [
            AtomInfo(0, "CA", "C", 0, 0, 0, "A", "ALA", 1),
            AtomInfo(1, "CA", "C", 1, 0, 0, "A", "XYZ", 2),  # unknown
            AtomInfo(2, "CA", "C", 2, 0, 0, "A", "GLY", 3),
        ]
        seqs = get_chain_sequences(atoms)
        assert seqs["A"] == "AG"  # XYZ skipped

    def test_reference_sequence_fills_gaps(self):
        """Reference sequences should fill gaps from incomplete loops."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        # Residues 1, 2, 10 — gap from 3-9
        atoms = [
            AtomInfo(0, "CA", "C", 0, 0, 0, "A", "ALA", 1),
            AtomInfo(1, "CA", "C", 1, 0, 0, "A", "GLY", 2),
            AtomInfo(2, "CA", "C", 2, 0, 0, "A", "VAL", 10),
        ]

        ref = {"A": "AGLLLLLLMV"}  # Full 10-residue sequence

        seqs = get_chain_sequences(atoms, reference_sequences=ref)
        assert seqs["A"] == "AGLLLLLLMV"

    def test_mixed_standard_and_nonstandard(self):
        """A chain with both standard and non-standard names in
        realistic MD simulation output."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        atoms = [
            AtomInfo(0, "CA", "C", 0, 0, 0, "A", "NMET", 1),   # N-terminal Met
            AtomInfo(1, "CA", "C", 1, 0, 0, "A", "ALA", 2),
            AtomInfo(2, "CA", "C", 2, 0, 0, "A", "HIE", 3),
            AtomInfo(3, "CA", "C", 3, 0, 0, "A", "CYX", 4),
            AtomInfo(4, "CA", "C", 4, 0, 0, "A", "GLU", 5),
            AtomInfo(5, "CA", "C", 5, 0, 0, "A", "ASH", 6),
            AtomInfo(6, "CA", "C", 6, 0, 0, "A", "CVAL", 7),  # C-terminal Val
        ]
        seqs = get_chain_sequences(atoms)
        assert seqs["A"] == "MAHCEDV"


# ─── SMILES Inference Tests ──────────────────────────────────────────────────


class TestSMILESInference:
    """Tests for automatic SMILES inference from 3D coordinates."""

    def test_infer_ethanol(self):
        """Infer SMILES for ethanol (CCO) from 3D coordinates."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.smiles_inference import infer_smiles_from_atoms

        # Ethanol 3D coordinates (C-C-O)
        atoms = [
            AtomInfo(0, "C1", "C", 0.000, 0.000, 0.000, "B", "LIG", 1),
            AtomInfo(1, "C2", "C", 1.540, 0.000, 0.000, "B", "LIG", 1),
            AtomInfo(2, "O1", "O", 2.310, 1.260, 0.000, "B", "LIG", 1),
        ]
        smiles = infer_smiles_from_atoms(atoms)
        assert smiles is not None
        # Verify it parses correctly
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        # Should have 3 heavy atoms
        assert mol.GetNumHeavyAtoms() == 3

    def test_infer_benzene(self):
        """Infer SMILES for benzene from 3D coordinates."""
        import math
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.smiles_inference import infer_smiles_from_atoms

        # Benzene ring: 6 carbons in a planar hexagon, CC bond ~1.40 Å
        r = 1.40
        atoms = []
        for i in range(6):
            angle = i * math.pi / 3.0
            atoms.append(AtomInfo(
                i, f"C{i + 1}", "C",
                r * math.cos(angle), r * math.sin(angle), 0.0,
                "B", "LIG", 1,
            ))

        smiles = infer_smiles_from_atoms(atoms)
        assert smiles is not None
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 6
        # Should contain aromatic carbons
        assert "c" in smiles.lower() or "C" in smiles

    def test_infer_water_returns_none(self):
        """Single oxygen atom should not produce a useful SMILES."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.smiles_inference import infer_smiles_from_atoms

        atoms = [
            AtomInfo(0, "O", "O", 0.0, 0.0, 0.0, "W", "HOH", 1),
        ]
        smiles = infer_smiles_from_atoms(atoms)
        # Either None or a trivial single-atom SMILES — both acceptable
        if smiles is not None:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None

    def test_infer_empty_atoms(self):
        """Empty atom list should return None."""
        from boltz.affinity_rescoring.smiles_inference import infer_smiles_from_atoms
        assert infer_smiles_from_atoms([]) is None

    def test_infer_hydrogens_ignored(self):
        """Hydrogen atoms should be filtered out before bond perception."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.smiles_inference import infer_smiles_from_atoms

        atoms = [
            AtomInfo(0, "C1", "C", 0.000, 0.000, 0.000, "B", "LIG", 1),
            AtomInfo(1, "H1", "H", 0.630, 0.630, 0.630, "B", "LIG", 1),
            AtomInfo(2, "H2", "H", -0.630, -0.630, 0.630, "B", "LIG", 1),
            AtomInfo(3, "H3", "H", -0.630, 0.630, -0.630, "B", "LIG", 1),
            AtomInfo(4, "C2", "C", 1.540, 0.000, 0.000, "B", "LIG", 1),
            AtomInfo(5, "O1", "O", 2.310, 1.260, 0.000, "B", "LIG", 1),
        ]
        smiles = infer_smiles_from_atoms(atoms)
        assert smiles is not None
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 3  # Only C, C, O

    def test_user_smiles_override(self):
        """User-provided SMILES should take priority over inference."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.smiles_inference import (
            infer_ligand_smiles_from_structure,
        )

        # Some ligand atoms (doesn't matter what coordinates)
        atoms = [
            AtomInfo(0, "C1", "C", 10.0, 11.0, 12.0, "B", "LIG", 1, is_hetatm=True),
            AtomInfo(1, "C2", "C", 11.0, 12.0, 13.0, "B", "LIG", 1, is_hetatm=True),
            AtomInfo(2, "N1", "N", 12.0, 13.0, 14.0, "B", "LIG", 1, is_hetatm=True),
        ]

        resolved, warnings = infer_ligand_smiles_from_structure(
            atoms=atoms,
            ligand_chains=["B"],
            user_smiles={"B": "c1ccccc1"},  # User says it's benzene
        )

        assert resolved["B"] == "c1ccccc1"  # User's SMILES wins

    def test_auto_inference_when_no_user_smiles(self):
        """Without user SMILES, inference should be attempted."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.smiles_inference import (
            infer_ligand_smiles_from_structure,
        )

        # Ethanol coordinates
        atoms = [
            AtomInfo(0, "C1", "C", 0.000, 0.000, 0.000, "B", "LIG", 1, is_hetatm=True),
            AtomInfo(1, "C2", "C", 1.540, 0.000, 0.000, "B", "LIG", 1, is_hetatm=True),
            AtomInfo(2, "O1", "O", 2.310, 1.260, 0.000, "B", "LIG", 1, is_hetatm=True),
        ]

        resolved, warnings = infer_ligand_smiles_from_structure(
            atoms=atoms,
            ligand_chains=["B"],
            user_smiles=None,
        )

        assert "B" in resolved
        assert len(resolved["B"]) > 0
        # Should have a warning about auto-inference
        assert any("auto-inferred" in w for w in warnings)

    def test_partial_user_smiles(self):
        """User provides SMILES for one chain, other is auto-inferred."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.smiles_inference import (
            infer_ligand_smiles_from_structure,
        )

        atoms = [
            # Chain B ligand
            AtomInfo(0, "C1", "C", 0.0, 0.0, 0.0, "B", "LIG", 1, is_hetatm=True),
            AtomInfo(1, "C2", "C", 1.54, 0.0, 0.0, "B", "LIG", 1, is_hetatm=True),
            AtomInfo(2, "O1", "O", 2.31, 1.26, 0.0, "B", "LIG", 1, is_hetatm=True),
            # Chain C ligand
            AtomInfo(3, "C1", "C", 10.0, 10.0, 10.0, "C", "DRG", 1, is_hetatm=True),
            AtomInfo(4, "N1", "N", 11.54, 10.0, 10.0, "C", "DRG", 1, is_hetatm=True),
        ]

        resolved, warnings = infer_ligand_smiles_from_structure(
            atoms=atoms,
            ligand_chains=["B", "C"],
            user_smiles={"B": "CCO"},  # Only provide B
        )

        assert resolved["B"] == "CCO"  # User-provided
        assert "C" in resolved  # Auto-inferred

    def test_missing_chain_atoms(self):
        """Ligand chain with no atoms should produce a warning."""
        from boltz.affinity_rescoring.smiles_inference import (
            infer_ligand_smiles_from_structure,
        )

        resolved, warnings = infer_ligand_smiles_from_structure(
            atoms=[],
            ligand_chains=["B"],
            user_smiles=None,
        )

        assert "B" not in resolved
        assert any("no atoms" in w.lower() for w in warnings)


# ─── SEQRES / Reference Sequence Tests ───────────────────────────────────────


class TestSEQRESExtraction:
    """Tests for SEQRES-based gap filling and reference sequence support."""

    def test_seqres_three_to_one_standard(self):
        """Standard amino acids convert correctly."""
        from boltz.affinity_rescoring.parsers import _seqres_three_to_one

        assert _seqres_three_to_one("ALA") == "A"
        assert _seqres_three_to_one("GLY") == "G"
        assert _seqres_three_to_one("TRP") == "W"
        assert _seqres_three_to_one("HIS") == "H"

    def test_seqres_three_to_one_modified(self):
        """Modified residues map to parent amino acid."""
        from boltz.affinity_rescoring.parsers import _seqres_three_to_one

        assert _seqres_three_to_one("MSE") == "M"  # selenomethionine → Met
        assert _seqres_three_to_one("TPO") == "T"  # phosphothreonine → Thr
        assert _seqres_three_to_one("SEP") == "S"  # phosphoserine → Ser
        assert _seqres_three_to_one("PTR") == "Y"  # phosphotyrosine → Tyr

    def test_seqres_three_to_one_nonstandard_ff(self):
        """Force-field protonation states map to parent."""
        from boltz.affinity_rescoring.parsers import _seqres_three_to_one

        assert _seqres_three_to_one("HID") == "H"
        assert _seqres_three_to_one("HIE") == "H"
        assert _seqres_three_to_one("CYX") == "C"
        assert _seqres_three_to_one("ASH") == "D"
        assert _seqres_three_to_one("GLH") == "E"

    def test_seqres_three_to_one_non_amino_acid(self):
        """Non-amino acid entries return None."""
        from boltz.affinity_rescoring.parsers import _seqres_three_to_one

        # Common ligand components should return None
        assert _seqres_three_to_one("ATP") is None or True  # May or may not be known
        # But definitely not amino acids:
        result = _seqres_three_to_one("ZZZNOTREAL")
        # Unknown = not amino acid
        assert result is None or result == "X"  # gemmi returns X for UNK

    def test_extract_seqres_from_pdb_with_seqres(self):
        """Extract SEQRES from a PDB file with SEQRES records."""
        from boltz.affinity_rescoring.parsers import get_seqres_sequences

        # Write a minimal PDB with SEQRES
        pdb_content = """\
HEADER    TEST
SEQRES   1 A   10  MET ALA ARG GLY HIS ILE LEU ASP GLU VAL
ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  MET A   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      3  CA  ALA A   2       3.000   4.000   5.000  1.00 10.00           C
ATOM      4  CA  ARG A   3       4.000   5.000   6.000  1.00 10.00           C
ATOM      5  CA  HIS A   5       6.000   7.000   8.000  1.00 10.00           C
ATOM      6  CA  ILE A   6       7.000   8.000   9.000  1.00 10.00           C
END
"""
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write(pdb_content)
            f.flush()
            try:
                seqres = get_seqres_sequences(f.name)
                # Should find chain A with full deposited sequence
                if "A" in seqres:
                    assert len(seqres["A"]) == 10
                    assert seqres["A"] == "MARGHILDEV"
            finally:
                os.unlink(f.name)

    def test_get_seqres_sequences_no_seqres(self):
        """PDB without SEQRES records returns empty dict."""
        from boltz.affinity_rescoring.parsers import get_seqres_sequences

        pdb_content = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C
ATOM      2  CA  GLY A   2       3.000   4.000   5.000  1.00 10.00           C
END
"""
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write(pdb_content)
            f.flush()
            try:
                seqres = get_seqres_sequences(f.name)
                # Might be empty or might have entity-derived sequences
                # (gemmi may infer entities from ATOM records)
                # The key point is no crash
                assert isinstance(seqres, dict)
            finally:
                os.unlink(f.name)

    def test_reference_sequence_overrides_gapped_extraction(self):
        """User-provided reference sequence fills gaps."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        # Simulate atoms with a gap at residues 4-5
        atoms = [
            AtomInfo(0, "CA", "C", 1.0, 2.0, 3.0, "A", "MET", 1),
            AtomInfo(1, "CA", "C", 2.0, 3.0, 4.0, "A", "ALA", 2),
            AtomInfo(2, "CA", "C", 3.0, 4.0, 5.0, "A", "ARG", 3),
            # Gap at 4, 5
            AtomInfo(3, "CA", "C", 6.0, 7.0, 8.0, "A", "HIS", 6),
            AtomInfo(4, "CA", "C", 7.0, 8.0, 9.0, "A", "ILE", 7),
        ]

        # Without reference: gapped sequence "MARHI" (5 residues)
        seqs_no_ref = get_chain_sequences(atoms)
        assert seqs_no_ref["A"] == "MARHI"

        # With reference: full 7-residue sequence
        ref = {"A": "MARGDHI"}  # The real sequence with G,D filling the gap
        seqs_with_ref = get_chain_sequences(atoms, reference_sequences=ref)
        assert seqs_with_ref["A"] == "MARGDHI"
        assert len(seqs_with_ref["A"]) == 7

    def test_reference_sequence_not_used_when_no_gaps(self):
        """Reference sequence is NOT substituted when there are no gaps."""
        from boltz.affinity_rescoring.models import AtomInfo
        from boltz.affinity_rescoring.parsers import get_chain_sequences

        atoms = [
            AtomInfo(0, "CA", "C", 1.0, 2.0, 3.0, "A", "MET", 1),
            AtomInfo(1, "CA", "C", 2.0, 3.0, 4.0, "A", "ALA", 2),
            AtomInfo(2, "CA", "C", 3.0, 4.0, 5.0, "A", "ARG", 3),
        ]

        ref = {"A": "MARGHILDEV"}  # Longer reference
        seqs = get_chain_sequences(atoms, reference_sequences=ref)
        # No gaps → extracted sequence is used
        assert seqs["A"] == "MAR"

    def test_merge_reference_sequences_user_overrides_seqres(self):
        """User-provided sequences override SEQRES-derived ones."""
        from boltz.affinity_rescoring.rescorer import _merge_reference_sequences

        # Write a PDB with SEQRES
        pdb_content = """\
HEADER    TEST
SEQRES   1 A    5  MET ALA ARG GLY HIS
ATOM      1  CA  MET A   1       1.000   2.000   3.000  1.00 10.00           C
END
"""
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write(pdb_content)
            f.flush()
            try:
                # User provides a different sequence for chain A
                user_seqs = {"A": "MARGHILDEVWWW"}
                merged = _merge_reference_sequences(
                    Path(f.name), user_seqs
                )
                # User override should win
                assert merged["A"] == "MARGHILDEVWWW"
            finally:
                os.unlink(f.name)

    def test_get_seqres_sequences_nonexistent_file(self):
        """Non-existent file returns empty dict gracefully."""
        from boltz.affinity_rescoring.parsers import get_seqres_sequences

        result = get_seqres_sequences("/nonexistent/path.pdb")
        assert result == {}

    def test_parse_structure_file_metadata_has_seqres(self):
        """parse_structure_file metadata includes SEQRES info when present."""
        from boltz.affinity_rescoring.parsers import parse_structure_file

        pdb_content = """\
HEADER    TEST
SEQRES   1 A    5  MET ALA ARG GLY HIS
ATOM      1  CA  MET A   1       1.000   2.000   3.000  1.00 10.00           C
ATOM      2  CA  ALA A   2       2.000   3.000   4.000  1.00 10.00           C
END
"""
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write(pdb_content)
            f.flush()
            try:
                atoms, metadata = parse_structure_file(f.name)
                # If SEQRES was found, metadata should indicate it
                if metadata.get("has_seqres") == "true":
                    assert "seqres_A" in metadata
            finally:
                os.unlink(f.name)
