# Boltz Affinity Rescoring

Affinity-only rescoring for protein-ligand complexes using the Boltz-2 affinity module. **Diffusion and confidence modules are intentionally disabled** — all predictions use pre-existing 3D coordinates fed directly through the trunk + affinity head.

## Overview

Three workflows:

- **Single Complex**: Score one protein-ligand complex from a PDB/CIF file
- **Batch Processing**: Score all complexes in a directory  
- **Virtual Screening**: Score multiple ligands (MOL2) against a single receptor

## Installation

```bash
# Install the full package (editable)
pip install -e .

# Or with affinity extras
pip install -e ".[affinity]"
```

The first run will auto-download the `boltz2_aff.ckpt` checkpoint (~1.5 GB) to `~/.boltz/`.

## Quick Start — CLI

```bash
# ─── Single complex ────────────────────────────────────────────
# Score a protein-ligand PDB. Auto-detects protein/ligand chains.
boltz rescore pdb --input complex.pdb

# Specify chains explicitly + JSON output
boltz rescore pdb --input complex.pdb \
  --protein-chain A --ligand-chains B \
  --output result.json --output-format json

# Provide ligand SMILES manually (if auto-inference fails)
boltz rescore pdb --input complex.pdb \
  --ligand-smiles '{"B": "CCO"}' \
  --output result.json

# Provide the full biological sequence (for structures with
# missing loops / incomplete SEQRES records)
boltz rescore pdb --input complex.pdb \
  --reference-sequence '{"A": "MKTLLILTLVVA...FULL_SEQ_HERE"}'

# Dry run — validate inputs without loading the model
boltz rescore pdb --input complex.pdb --dry-run

# ─── Batch mode ────────────────────────────────────────────────
# Score every PDB/CIF in a directory
boltz rescore batch --input-dir ./structures/ --output batch_results.csv

# Recursive scan + Excel output
boltz rescore batch --input-dir ./structures/ \
  --recursive --output results.xlsx --output-format excel

# ─── Virtual screening (receptor + multi-ligand MOL2) ─────────
# Single receptor PDB scored against all ligands in a MOL2 file
boltz rescore receptor \
  --receptor receptor.pdb \
  --ligands docked_poses.mol2 \
  --output scores.csv

# Sort by score, Excel output, strict validation
boltz rescore receptor \
  --receptor receptor.pdb \
  --ligands library.mol2 \
  --output AA2AR_screening_results.csv \
  --output-format csv \
  --sort-by affinity_score \

# With explicit protein chain
boltz rescore receptor \
  --receptor complex.pdb \
  --protein-chain A \
  --ligands compounds.mol2 \
  --output scores.csv

# ─── Manifest mode ─────────────────────────────────────────────
# Score complexes listed in a YAML manifest
boltz rescore manifest --manifest complexes.yaml --output-dir ./scores/

# ─── Common options (apply to all commands) ────────────────────
#   --device cpu|cuda|mps|auto      Device (default: auto)
#   --validation strict|moderate|lenient  (default: moderate)
#   --checkpoint /path/to/ckpt      Custom checkpoint (default: auto-download)
#   --log-level DEBUG|INFO|WARNING|ERROR
```

## Quick Start — Python API

```python
from boltz.affinity_rescoring import AffinityRescorer

# Initialize (auto-downloads checkpoint on first use)
rescorer = AffinityRescorer(device="auto")

# ─── Single complex ───────────────────────────────────────────
result = rescorer.rescore_pdb("complex.pdb")
print(f"Predicted pKd: {result.affinity_pred:.2f}")
print(f"Binding probability: {result.affinity_probability_binary:.2f}")

# With explicit chains and SMILES
result = rescorer.rescore_pdb(
    "complex.pdb",
    protein_chain="A",
    ligand_chains=["B"],
    ligand_smiles={"B": "CCO"},
)

# ─── Batch ────────────────────────────────────────────────────
results = rescorer.rescore_directory("structures/", recursive=True)
rescorer.export_results(results, "output.csv", format="csv")

# ─── Virtual screening ───────────────────────────────────────
scores = rescorer.rescore_receptor(
    receptor_path="receptor.pdb",
    ligands_path="ligands.mol2",
    output_path="scores.csv",
    sort_by="affinity_score",
)

# Access individual scores
for s in scores:
    print(f"{s.ligand_name}: {s.affinity_score:.3f} (conf={s.confidence:.3f})")

# ─── Dry run (validate only) ─────────────────────────────────
report = rescorer.dry_run("complex.pdb")
print(report)  # chains, atom counts, validation issues
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
  recycling_steps: 5              # trunk recycling iterations
  affinity_mw_correction: true    # molecular weight correction

validation:
  level: moderate       # strict | moderate | lenient

output:
  format: csv
  include_metadata: true
```

Save to `~/.boltz/rescore_config.yaml` or pass via `--config`.

> **Note:** `diffusion_samples` and `sampling_steps` are not available. This module is affinity-only — no diffusion pipeline is executed.

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
