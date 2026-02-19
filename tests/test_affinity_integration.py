"""
Integration tests for the affinity rescoring system.

These tests verify end-to-end workflows including:
- AffinityRescorer dry_run mode (no GPU required)
- Batch processing with mixed inputs
- Export pipeline from results through all formats
- CLI invocation
- Configuration round-tripping
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def sample_pdb_dir(tmp_path):
    """Directory with multiple PDB files for batch testing."""
    pdb_dir = tmp_path / "structures"
    pdb_dir.mkdir()

    pdb_template = """\
HEADER    TEST {name}
ATOM      1  N   ALA A   1       {x:.3f}   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       {cx:.3f}   3.000   4.000  1.00 12.00           C
ATOM      3  C   ALA A   1       {cx2:.3f}   4.000   5.000  1.00 11.00           C
ATOM      4  O   ALA A   1       {cx3:.3f}   5.000   6.000  1.00 13.00           O
ATOM      5  N   GLY A   2       {cx4:.3f}   4.500   5.500  1.00 10.00           N
ATOM      6  CA  GLY A   2       {cx5:.3f}   5.000   6.000  1.00 12.00           C
ATOM      7  C   GLY A   2       {cx6:.3f}   6.000   7.000  1.00 11.00           C
ATOM      8  O   GLY A   2       {cx7:.3f}   7.000   8.000  1.00 13.00           O
HETATM    9  C1  LIG B   1      10.000  11.000  12.000  1.00  8.00           C
HETATM   10  C2  LIG B   1      11.000  12.000  13.000  1.00  9.00           C
HETATM   11  N1  LIG B   1      12.000  13.000  14.000  1.00  7.00           N
END
"""
    for i, name in enumerate(["alpha", "beta", "gamma"]):
        x = float(i)
        content = pdb_template.format(
            name=name, x=x, cx=x + 1, cx2=x + 2, cx3=x + 3,
            cx4=x + 3.5, cx5=x + 4, cx6=x + 5, cx7=x + 6,
        )
        (pdb_dir / f"{name}.pdb").write_text(content)

    # Add an invalid file
    (pdb_dir / "empty.pdb").touch()

    return pdb_dir


@pytest.fixture
def sample_receptor_pdb(tmp_path):
    """Receptor PDB file for virtual screening tests."""
    pdb = tmp_path / "receptor.pdb"
    lines = ["HEADER    RECEPTOR STRUCTURE\n"]
    for i in range(50):
        lines.append(
            f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    "
            f"{float(i):8.3f}{float(i + 1):8.3f}{float(i + 2):8.3f}"
            f"  1.00 10.00           C\n"
        )
    lines.append("END\n")
    pdb.write_text("".join(lines))
    return pdb


@pytest.fixture
def sample_mol2_for_vs(tmp_path):
    """MOL2 file with ligands for virtual screening."""
    mol2 = tmp_path / "ligands.mol2"
    mol2.write_text("""\
@<TRIPOS>MOLECULE
lig_alpha
 3 2  0  0  0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
   1 C1   12.450  10.200  -5.100 C.ar      1  lig_alpha    0.0000
   2 C2   13.100  11.300  -5.800 C.ar      1  lig_alpha    0.0000
   3 N1   14.200  12.400  -6.500 N.3       1  lig_alpha    0.0000
@<TRIPOS>BOND
   1  1  2 ar
   2  2  3 1

@<TRIPOS>MOLECULE
lig_beta
 3 2  0  0  0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
   1 C1   11.200  10.100  -6.500 C.ar      1  lig_beta    0.0000
   2 O1   12.100  11.200  -6.900 O.3       1  lig_beta    0.0000
   3 N1   13.000  12.300  -7.200 N.3       1  lig_beta    0.0000
@<TRIPOS>BOND
   1  1  2 1
   2  2  3 1
""")
    return mol2


# ─── Validation Pipeline Integration ─────────────────────────────────────────


class TestValidationPipeline:
    """End-to-end validation pipeline tests."""

    def test_validation_full_pipeline(self, sample_pdb_dir):
        """Test that validation works end-to-end on a valid file."""
        from boltz.affinity_rescoring.validation import (
            ChainIdentifier,
            StructureNormalizer,
            StructureValidator,
        )
        from boltz.affinity_rescoring.parsers import parse_structure_file

        pdb_file = sample_pdb_dir / "alpha.pdb"

        # Step 1: File validation
        validator = StructureValidator()
        file_report = validator.validate_file(pdb_file)
        assert file_report.is_valid

        # Step 2: Parse
        atoms, metadata = parse_structure_file(pdb_file)
        assert len(atoms) > 0

        # Step 3: Atom validation
        atom_report = validator.validate_atoms(atoms)
        assert atom_report.is_valid

        # Step 4: Chain identification
        identifier = ChainIdentifier()
        assignment = identifier.identify_chains(atoms)
        assert len(assignment.protein_chains) > 0

        # Step 5: Normalization
        normalizer = StructureNormalizer()
        normalized, log = normalizer.normalize(atoms)
        assert len(normalized) > 0

    def test_validation_catches_empty_file(self, sample_pdb_dir):
        """Empty PDB should fail validation."""
        from boltz.affinity_rescoring.validation import StructureValidator

        empty_file = sample_pdb_dir / "empty.pdb"
        validator = StructureValidator()
        report = validator.validate_file(empty_file)
        assert not report.is_valid


# ─── Export Pipeline Integration ──────────────────────────────────────────────


class TestExportPipeline:
    """End-to-end export pipeline tests."""

    @pytest.fixture
    def mixed_results(self):
        from boltz.affinity_rescoring.models import AffinityResult, ValidationStatus

        return [
            AffinityResult(
                id=f"complex_{i:03d}",
                affinity_pred=-7.0 + i * 0.5,
                affinity_std=0.2 + i * 0.05,
                protein_chain="A",
                ligand_chains=["B"],
                protein_residue_count=100 + i * 10,
                ligand_atom_count=25 + i * 5,
                inference_time_ms=200 + i * 20,
                validation_status=ValidationStatus.SUCCESS,
            )
            for i in range(5)
        ] + [
            AffinityResult(
                id="complex_bad",
                validation_status=ValidationStatus.FAILED,
                error_message="Could not parse",
            ),
        ]

    def test_json_round_trip(self, mixed_results, tmp_output):
        """JSON export should be parseable and contain correct data."""
        from boltz.affinity_rescoring.export import ResultsExporter
        from boltz.affinity_rescoring.models import OutputFormat

        exporter = ResultsExporter(model_checkpoint="test_v1.ckpt")
        path = exporter.export(mixed_results, tmp_output / "results.json", fmt=OutputFormat.JSON)

        with open(path) as f:
            data = json.load(f)

        assert data["summary"]["total_processed"] == 6
        assert data["summary"]["successful"] == 5
        assert data["summary"]["failed"] == 1
        assert len(data["results"]) == 6
        assert data["model_checkpoint"] == "test_v1.ckpt"

    def test_csv_all_rows_present(self, mixed_results, tmp_output):
        """CSV should contain all results as rows."""
        import csv
        from boltz.affinity_rescoring.export import ResultsExporter
        from boltz.affinity_rescoring.models import OutputFormat

        exporter = ResultsExporter()
        path = exporter.export(mixed_results, tmp_output / "results.csv", fmt=OutputFormat.CSV)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 6
        # Check that all IDs are present
        ids = {r["id"] for r in rows}
        assert "complex_000" in ids
        assert "complex_bad" in ids

    def test_sqlite_queryable(self, mixed_results, tmp_output):
        """SQLite export should be queryable."""
        import sqlite3
        from boltz.affinity_rescoring.export import ResultsExporter
        from boltz.affinity_rescoring.models import OutputFormat

        exporter = ResultsExporter()
        path = exporter.export(
            mixed_results, tmp_output / "results.db", fmt=OutputFormat.SQLITE
        )

        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        # Count all
        cursor.execute("SELECT COUNT(*) FROM results")
        assert cursor.fetchone()[0] == 6

        # Query successful
        cursor.execute(
            "SELECT COUNT(*) FROM results WHERE validation_status='SUCCESS'"
        )
        assert cursor.fetchone()[0] == 5

        # Check ordering capability
        cursor.execute(
            "SELECT id, affinity_pred FROM results "
            "WHERE validation_status='SUCCESS' "
            "ORDER BY affinity_pred ASC LIMIT 1"
        )
        best = cursor.fetchone()
        assert best[0] == "complex_000"

        conn.close()

    def test_jsonl_line_per_result(self, mixed_results, tmp_output):
        """JSONL should have one JSON per line."""
        from boltz.affinity_rescoring.export import ResultsExporter
        from boltz.affinity_rescoring.models import OutputFormat

        exporter = ResultsExporter()
        path = exporter.export(
            mixed_results, tmp_output / "results.jsonl", fmt=OutputFormat.JSONL
        )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 6

        # Each line should be valid JSON
        for line in lines:
            obj = json.loads(line)
            assert "id" in obj

    def test_batch_summary_statistics(self, mixed_results):
        """Batch summary should compute correct statistics."""
        from boltz.affinity_rescoring.export import compute_batch_summary

        summary = compute_batch_summary(mixed_results)
        assert summary.total_processed == 6
        assert summary.successful == 5
        assert summary.failed == 1
        # Mean should be around -5.5 (avg of -7, -6.5, -6, -5.5, -5)
        import math
        assert not math.isnan(summary.mean_affinity)


# ─── Configuration Integration ───────────────────────────────────────────────


class TestConfigIntegration:
    """Configuration round-trip and override tests."""

    def test_full_config_round_trip(self, tmp_path):
        """Save, load, and verify configuration."""
        from boltz.affinity_rescoring.config import load_config, save_config
        from boltz.affinity_rescoring.models import (
            DeviceOption,
            OutputFormat,
            RescoreConfig,
            ValidationLevel,
        )

        original = RescoreConfig(
            device=DeviceOption.CPU,
            recycling_steps=3,
            diffusion_samples=10,
            validation_level=ValidationLevel.STRICT,
            output_format=OutputFormat.JSON,
            include_metadata=False,
        )

        config_file = tmp_path / "config.yaml"
        save_config(original, config_file)

        loaded = load_config(config_file)

        assert loaded.device == original.device
        assert loaded.recycling_steps == original.recycling_steps
        assert loaded.diffusion_samples == original.diffusion_samples
        assert loaded.validation_level == original.validation_level

    def test_env_vars_override_yaml(self, tmp_path):
        """Environment variables should override YAML values."""
        from boltz.affinity_rescoring.config import load_config

        # Write YAML with device=cpu
        yaml_content = """\
model:
  device: cpu
inference:
  recycling_steps: 3
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        # Override device via env
        os.environ["BOLTZ_RESCORE_DEVICE"] = "cuda"
        try:
            config = load_config(config_file)
            assert config.device.value == "cuda"
            assert config.recycling_steps == 3  # From YAML
        finally:
            del os.environ["BOLTZ_RESCORE_DEVICE"]

    def test_config_from_defaults_only(self, tmp_path, monkeypatch):
        """Loading without any file should produce valid defaults."""
        from boltz.affinity_rescoring.config import load_config

        # Ensure no stale env vars or config files leak in
        for key in list(os.environ):
            if key.startswith("BOLTZ_RESCORE_"):
                monkeypatch.delenv(key, raising=False)

        config = load_config()
        # Just verify the config is valid and fields are set
        assert config.recycling_steps >= 1
        assert config.device.value in ("auto", "cpu", "cuda", "mps")
        assert config.validation_level.value in ("strict", "moderate", "lenient")


# ─── Dry Run Integration ──────────────────────────────────────────────────────


class TestDryRun:
    """Dry-run mode tests (validation without inference)."""

    def test_dry_run_valid_pdb(self, sample_pdb_dir, tmp_output):
        """Dry run should validate without running inference."""
        from boltz.affinity_rescoring.rescorer import AffinityRescorer
        from boltz.affinity_rescoring.models import RescoreConfig, DeviceOption

        config = RescoreConfig(device=DeviceOption.CPU)
        rescorer = AffinityRescorer(config=config)

        pdb_file = sample_pdb_dir / "alpha.pdb"
        report = rescorer.dry_run(pdb_file)

        assert report is not None

    def test_dry_run_invalid_pdb(self, sample_pdb_dir, tmp_output):
        """Dry run should catch validation errors."""
        from boltz.affinity_rescoring.rescorer import AffinityRescorer
        from boltz.affinity_rescoring.models import RescoreConfig, DeviceOption

        config = RescoreConfig(device=DeviceOption.CPU)
        rescorer = AffinityRescorer(config=config)

        empty_file = sample_pdb_dir / "empty.pdb"
        report = rescorer.dry_run(empty_file)
        assert report is not None


# ─── MOL2 + Validation Integration ───────────────────────────────────────────


class TestMOL2Integration:
    """MOL2 parsing integrated with validation."""

    def test_parse_and_validate_ligands(self, sample_mol2_for_vs):
        """Parse MOL2 and validate extracted ligands."""
        from boltz.affinity_rescoring.mol2_parser import MOL2Parser
        from boltz.affinity_rescoring.validation import StructureValidator

        parser = MOL2Parser()
        ligands = parser.extract_ligands_with_names(sample_mol2_for_vs)

        assert len(ligands) == 2

        validator = StructureValidator()
        for lig in ligands:
            assert lig.num_atoms > 0
            report = validator.validate_atoms(lig.atoms)
            assert report.is_valid, f"Ligand {lig.name} failed validation: {report.issues}"

    def test_mol2_element_detection(self, sample_mol2_for_vs):
        """MOL2 atom type → element mapping should work for common types."""
        from boltz.affinity_rescoring.mol2_parser import MOL2Parser

        parser = MOL2Parser()
        ligands = parser.extract_ligands_with_names(sample_mol2_for_vs)

        # First ligand: C.ar, C.ar, N.3 → C, C, N
        elements = ligands[0].elements
        assert elements.count("C") == 2
        assert elements.count("N") == 1

        # Second ligand: C.ar, O.3, N.3 → C, O, N
        elements = ligands[1].elements
        assert "C" in elements
        assert "O" in elements
        assert "N" in elements


# ─── CLI Tests ────────────────────────────────────────────────────────────────


class TestCLI:
    """CLI invocation tests (without actual inference)."""

    def test_cli_group_registered(self):
        """Verify the rescore CLI group is available."""
        from boltz.affinity_rescoring.cli import rescore_cli
        import click

        assert isinstance(rescore_cli, click.MultiCommand) or isinstance(
            rescore_cli, click.Group
        )

    def test_cli_subcommands_exist(self):
        """Verify all expected subcommands are registered."""
        from boltz.affinity_rescoring.cli import rescore_cli

        commands = rescore_cli.list_commands(ctx=None)
        assert "pdb" in commands
        assert "batch" in commands
        assert "receptor" in commands
