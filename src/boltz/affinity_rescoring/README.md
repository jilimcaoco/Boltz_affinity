# Boltz Affinity Rescoring

A production-ready module for rescoring protein-ligand complexes using the Boltz-2 affinity prediction model.

## Overview

The affinity rescoring system integrates with the existing Boltz-2 pipeline to predict binding affinities (pKd) for protein-ligand complexes. It supports three primary workflows:

- **Single Complex**: Score one protein-ligand complex from a PDB/CIF file
- **Batch Processing**: Score all complexes in a directory  
- **Virtual Screening**: Score multiple ligands (MOL2) against a single receptor

## Installation

The module is included in the Boltz package. Additional dependencies:

```bash
pip install -r requirements_affinity.txt
```

Or install via the main package:

```bash
pip install -e ".[affinity]"
```

## Quick Start

### CLI Usage

```bash
# Single complex
boltz rescore pdb --input complex.pdb --output results.json

# Batch mode
boltz rescore batch --input structures/ --output batch_results.csv

# Virtual screening
boltz rescore receptor --receptor receptor.pdb --ligands ligands.mol2 --output scores.csv

# Dry run (validate only, no inference)
boltz rescore pdb --input complex.pdb --dry-run
```

### Python API

```python
from boltz.affinity_rescoring import AffinityRescorer
from boltz.affinity_rescoring.models import RescoreConfig, DeviceOption

# Configure
config = RescoreConfig(device=DeviceOption.AUTO)
rescorer = AffinityRescorer(config=config)

# Score a single complex
result = rescorer.rescore_pdb("complex.pdb")
print(f"Predicted pKd: {result.affinity_pred:.2f}")

# Batch processing
results = rescorer.rescore_directory("structures/")
rescorer.export_results(results, "output.csv")

# Virtual screening
scores = rescorer.rescore_receptor(
    receptor_path="receptor.pdb",
    ligands_path="ligands.mol2",
)
```

## Architecture

```
src/boltz/affinity_rescoring/
├── __init__.py          # Package exports
├── models.py            # Data models (Pydantic + dataclasses)
├── validation.py        # Structure validation & chain identification
├── parsers.py           # PDB/CIF file parsing (gemmi)
├── mol2_parser.py       # MOL2 multi-molecule parser
├── inference.py         # Model loading & inference engine
├── export.py            # Multi-format results export
├── rescorer.py          # Main orchestrator
├── cli.py               # Click CLI commands
└── config.py            # YAML configuration management
```

### Layer Responsibilities

| Layer | Module | Purpose |
|-------|--------|---------|
| **Input** | `parsers.py`, `mol2_parser.py` | Parse structure files |
| **Validation** | `validation.py` | Validate atoms, chains, coordinates |
| **Core** | `rescorer.py` | Orchestrate the pipeline |
| **Inference** | `inference.py` | Device management, model loading, prediction |
| **Output** | `export.py` | JSON, CSV, JSONL, Parquet, SQLite, Excel |
| **Config** | `config.py` | YAML + env var configuration |
| **CLI** | `cli.py` | Command-line interface |

## Configuration

### YAML Configuration

```yaml
model:
  device: auto          # auto | cpu | cuda | mps
  checkpoint: auto      # auto-downloads or path to .ckpt

inference:
  recycling_steps: 5
  diffusion_samples: 5
  sampling_steps: 200
  affinity_mw_correction: true

validation:
  level: moderate       # strict | moderate | lenient

output:
  format: csv
  include_metadata: true
```

Save to `~/.boltz/rescore_config.yaml` or pass via `--config`.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `BOLTZ_RESCORE_CHECKPOINT` | Path to affinity checkpoint |
| `BOLTZ_RESCORE_DEVICE` | Override device selection |
| `BOLTZ_RESCORE_VALIDATION` | Validation level |
| `BOLTZ_RESCORE_OUTPUT_FORMAT` | Default output format |

## Validation

Three strictness levels control input validation:

- **STRICT**: All checks enforced, warnings treated as errors
- **MODERATE** (default): Standard checks, warnings logged  
- **LENIENT**: Minimal checks, best-effort processing

Validation checks include:
- File existence, size, format
- Coordinate validity (NaN/Inf detection)
- B-factor ranges
- Element type verification
- Steric clash detection
- Multi-chain presence
- Sequence gap detection

## Output Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| JSON | `.json` | Full structured output with metadata |
| JSONL | `.jsonl` | Streaming / line-by-line processing |
| CSV | `.csv` | Spreadsheet analysis |
| Parquet | `.parquet` | Large-scale data analysis |
| SQLite | `.db` | Queryable database |
| Excel | `.xlsx` | Reports with multiple sheets |

## Testing

```bash
# Unit tests
pytest tests/test_affinity_rescoring.py -v

# Integration tests
pytest tests/test_affinity_integration.py -v

# All affinity tests
pytest tests/test_affinity_*.py -v
```

## API Reference

### `AffinityRescorer`

Main orchestrator class.

| Method | Description |
|--------|-------------|
| `rescore_pdb(path)` | Score single complex |
| `rescore_batch(paths)` | Score list of files |
| `rescore_directory(dir)` | Scan and score directory |
| `rescore_receptor(receptor, ligands)` | Virtual screening |
| `dry_run(path)` | Validate without inference |
| `export_results(results, path)` | Export to file |

### `AffinityResult`

Dataclass holding prediction results.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Complex identifier |
| `affinity_pred` | `float` | Predicted pKd |
| `affinity_std` | `float` | Prediction uncertainty |
| `affinity_probability_binary` | `float` | Binary binding probability |
| `protein_chain` | `str` | Protein chain ID |
| `ligand_chains` | `List[str]` | Ligand chain IDs |
| `validation_status` | `ValidationStatus` | SUCCESS/FAILED/SKIPPED |

### `LigandScore`

Dataclass for virtual screening results.

| Field | Type | Description |
|-------|------|-------------|
| `ligand_name` | `str` | Ligand identifier |
| `affinity_score` | `float` | Predicted pKd |
| `confidence` | `float` | Prediction confidence |
| `n_atoms` | `int` | Heavy atom count |

## Troubleshooting

**Checkpoint download fails**: Set `BOLTZ_RESCORE_CHECKPOINT=/path/to/boltz2_aff.ckpt` or download manually from HuggingFace.

**GPU out of memory**: Use `--device cpu` or reduce complex size. The affinity cropper limits to 256 tokens / 2048 atoms.

**MOL2 parsing errors**: Ensure MOL2 follows standard Tripos format with `@<TRIPOS>MOLECULE`, `@<TRIPOS>ATOM`, and `@<TRIPOS>BOND` sections.

**Empty results**: Check validation report with `--dry-run`. Common issues: missing HETATM records, single-chain structures, non-standard residue names.
