# Boltz-2 Affinity Rescoring

Score protein–ligand binding affinities (predicted pKd) using the Boltz-2 affinity module. Accepts PDB/CIF complexes or receptor + MOL2 ligands and outputs per-complex affinity predictions with uncertainty estimates.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [boltz rescore pdb](#boltz-rescore-pdb)
  - [boltz rescore batch](#boltz-rescore-batch)
  - [boltz rescore receptor](#boltz-rescore-receptor)
  - [boltz rescore manifest](#boltz-rescore-manifest)
- [Python API](#python-api)
- [Input Preparation](#input-preparation)
  - [PDB/CIF files](#pdbcif-files)
  - [Handling incomplete structures](#handling-incomplete-structures)
  - [Ligand SMILES](#ligand-smiles)
  - [MOL2 ligands](#mol2-ligands)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Boltz-2 affinity module is a deep learning system that predicts protein–ligand binding affinities by:

1. **Extracting** the protein sequence and ligand SMILES from your structure files
2. **Predicting** the complex structure from scratch using the Boltz-2 diffusion model
3. **Scoring** the predicted complex with the AffinityModule to estimate pKd

Boltz-2 is a **sequence-based structure predictor** — it takes an amino acid sequence and ligand SMILES as input, folds the complex from scratch via diffusion, and then predicts affinity. It does not refine input coordinates. This means the quality of the input *sequence* matters far more than the quality of the input *coordinates*.

### Key features

- **Automatic chain detection** — protein and ligand chains identified automatically
- **Automatic SMILES inference** — ligand SMILES inferred from HETATM 3D coordinates via RDKit bond perception
- **SEQRES gap filling** — missing loops automatically filled from PDB SEQRES records
- **Non-standard residue handling** — 50+ AMBER/CHARMM/GROMACS residue names (HIE, CYX, ASH, etc.) mapped to standard amino acids
- **Multiple output formats** — JSON, CSV, Parquet, SQLite, Excel
- **Virtual screening** — score one receptor against many ligands from a MOL2 file
- **Batch processing** — process entire directories of PDB/CIF files

---

## Installation

### Prerequisites

- Python 3.10–3.12
- CUDA GPU recommended (CPU works but is significantly slower)

### Install Boltz-2

```bash
# From PyPI (recommended)


# Or from source
git clone https://github.com/jwohlwend/boltz.git
cd boltz && pip install -e .[cuda]
```

For CPU-only or non-CUDA hardware, omit `[cuda]`.

### Install affinity rescoring dependencies

```bash
pip install -r requirements_affinity.txt
```

This installs additional packages needed by the rescoring module (gemmi, rdkit, polars, openpyxl, etc.). Most are already included in the base Boltz-2 install.

### Verify installation

```bash
# Check that the rescore CLI is registered
boltz rescore --help

# Quick validation of a PDB file (no GPU required)
boltz rescore pdb --input complex.pdb --dry-run
```

### Model checkpoints

Checkpoints are downloaded automatically on first use when `--checkpoint auto` (the default). They are cached in `~/.boltz/`. Two checkpoints are used:

| Checkpoint | Purpose | Size |
|---|---|---|
| `boltz2_conf.ckpt` | Structure prediction (diffusion) | ~1 GB |
| `boltz2_aff.ckpt` | Affinity prediction | ~200 MB |

To use local checkpoints instead:

```bash
boltz rescore pdb -i complex.pdb --checkpoint /path/to/checkpoints/
```

---

## Quick Start

### Score a single complex

```bash
boltz rescore pdb -i complex.pdb -o result.json
```

Output:
```
Affinity: 6.4231 (probability: 0.8712)
Uncertainty: ±0.3142
Results saved to: result.json
```

### Score a receptor against many ligands

```bash
boltz rescore receptor \
  --receptor receptor.pdb \
  --ligands docking_poses.mol2 \
  -o scores.csv
```

### Validate inputs before running

```bash
boltz rescore pdb -i complex.pdb --dry-run
```

---

## CLI Reference

All rescoring commands are under `boltz rescore`. Run `boltz rescore --help` for the full list.

### `boltz rescore pdb`

Score a single protein–ligand complex from a PDB or CIF file.

```bash
boltz rescore pdb [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `-i, --input` | *(required)* | Path to PDB or CIF structure file |
| `-o, --output` | `<stem>_affinity.<fmt>` | Output file path |
| `--protein-chain` | auto-detect | Protein chain ID (e.g. `A`) |
| `--ligand-chains` | auto-detect | Comma-separated ligand chain IDs (e.g. `B,C`) |
| `--ligand-smiles` | auto-infer | Ligand SMILES as JSON (e.g. `'{"B": "CCO"}'`) |
| `--reference-sequence` | SEQRES | Full sequence(s) as JSON (e.g. `'{"A": "MKTL..."}'`) |
| `--output-format` | `json` | One of: `json`, `csv`, `parquet`, `sqlite`, `excel` |
| `--device` | `auto` | One of: `auto`, `cuda`, `cpu`, `mps` |
| `--validation` | `moderate` | One of: `strict`, `moderate`, `lenient` |
| `--checkpoint` | `auto` | Path to checkpoint dir, or `auto` to download |
| `--use-msa-server` | off | Use ColabFold MSA server for sequence search |
| `--dry-run` | off | Validate inputs without running inference |
| `--log-level` | `INFO` | One of: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

**Examples:**

```bash
# Basic usage — everything auto-detected
boltz rescore pdb -i 1a2b.pdb

# Specify chain and SMILES explicitly
boltz rescore pdb -i complex.pdb \
  --protein-chain A \
  --ligand-chains B \
  --ligand-smiles '{"B": "c1ccc(NC(=O)c2ccccc2)cc1"}'

# Handle incomplete PDB with reference sequence
boltz rescore pdb -i incomplete.pdb \
  --reference-sequence '{"A": "MKTLLILAVLCLGFA..."}'

# Run on CPU with verbose logging
boltz rescore pdb -i complex.pdb --device cpu --log-level DEBUG
```

---

### `boltz rescore batch`

Score all PDB/CIF files in a directory.

```bash
boltz rescore batch [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--input-dir` | *(required)* | Directory containing PDB/CIF files |
| `-o, --output` | `batch_results.csv` | Output file path |
| `--output-format` | `csv` | One of: `json`, `csv`, `parquet`, `sqlite`, `excel` |
| `--recursive` | off | Scan subdirectories recursively |
| `--device` | `auto` | Compute device |
| `--validation` | `moderate` | Validation strictness |
| `--checkpoint` | `auto` | Checkpoint path |
| `--ligand-smiles` | auto-infer | SMILES JSON (applied to all complexes) |
| `--use-msa-server` | off | Use MSA server |
| `--log-level` | `INFO` | Logging level |

**Examples:**

```bash
# Score all PDB files in a directory
boltz rescore batch --input-dir ./complexes/ -o results.csv

# Recursive scan with Parquet output
boltz rescore batch --input-dir ./data/ --recursive \
  --output-format parquet -o results.parquet
```

---

### `boltz rescore receptor`

Score a fixed receptor against multiple ligands from a MOL2 file. This is the primary virtual screening workflow.

```bash
boltz rescore receptor [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--receptor` | *(required)* | Path to receptor PDB/CIF file |
| `--ligands` | *(required)* | Path to MOL2 file with ligands |
| `-o, --output` | `scores.csv` | Output file path |
| `--protein-chain` | auto-detect | Protein chain ID |
| `--output-format` | `csv` | One of: `json`, `csv`, `parquet`, `excel` |
| `--device` | `auto` | Compute device |
| `--validation` | `moderate` | Validation strictness |
| `--checkpoint` | `auto` | Checkpoint path |
| `--sort-by` | `affinity_score` | Sort column: `affinity_score`, `confidence`, `ligand_name`, `n_atoms` |
| `--reference-sequence` | SEQRES | Full sequence(s) as JSON |
| `--log-level` | `INFO` | Logging level |

**Examples:**

```bash
# Virtual screening with ranked output
boltz rescore receptor \
  --receptor protein.pdb \
  --ligands docking_poses.mol2 \
  -o ranked_hits.csv \
  --sort-by affinity_score

# With explicit chain and reference sequence for incomplete receptor
boltz rescore receptor \
  --receptor receptor_prepared.pdb \
  --ligands ligands.mol2 \
  --protein-chain A \
  --reference-sequence '{"A": "FULL_SEQUENCE_HERE"}'
```

---

### `boltz rescore manifest`

Score complexes listed in a YAML manifest file. Useful for automated pipelines.

```bash
boltz rescore manifest [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--manifest` | *(required)* | Path to YAML manifest file |
| `--output-dir` | `./scores` | Output directory |
| `--output-format` | `json` | One of: `json`, `csv`, `parquet` |
| `--device` | `auto` | Compute device |
| `--checkpoint` | `auto` | Checkpoint path |
| `--use-msa-server` | off | Use MSA server |
| `--log-level` | `INFO` | Logging level |

**Manifest format:**

```yaml
complexes:
  - pdb: /path/to/complex1.pdb
    protein_chain: A
    ligand_chains: "B"
    ligand_smiles:
      B: "CCO"

  - pdb: /path/to/complex2.pdb
    # Chains auto-detected, SMILES auto-inferred

  - pdb: /path/to/complex3.cif
    protein_chain: C
```

**Example:**

```bash
boltz rescore manifest --manifest screening.yaml --output-dir ./results/
```

---

## Python API

For programmatic access, use the `AffinityRescorer` class directly.

### Single complex

```python
from boltz.affinity_rescoring import AffinityRescorer

rescorer = AffinityRescorer(
    checkpoint="auto",
    device="cuda",
    validation_level="moderate",
)

result = rescorer.rescore_pdb(
    "complex.pdb",
    protein_chain="A",
    ligand_chains=["B"],
    ligand_smiles={"B": "c1ccccc1"},  # optional
    output_path="result.json",
    output_format="json",
)

print(f"pKd: {result.affinity_pred:.2f} ± {result.affinity_std:.2f}")
print(f"Binder probability: {result.affinity_probability_binary:.2%}")
print(f"Status: {result.validation_status.value}")
```

### Virtual screening

```python
scores = rescorer.rescore_receptor(
    receptor_path="receptor.pdb",
    ligands_path="ligands.mol2",
    protein_chain="A",
    output_path="scores.csv",
    output_format="csv",
    sort_by="affinity_score",
)

for s in scores[:10]:
    print(f"{s.ligand_name}: {s.affinity_score:.3f} (conf={s.confidence:.3f})")
```

### Batch processing

```python
results = rescorer.rescore_directory(
    "complexes/",
    output_path="results.csv",
    output_format="csv",
    recursive=True,
)
```

### Dry run (validation only)

```python
report = rescorer.dry_run("complex.pdb")
print(report["chains"])      # detected protein/ligand chains
print(report["sequences"])   # extracted sequence lengths
print(report["issues"])      # validation warnings
```

### Handling incomplete structures

```python
# Provide full reference sequence for structures with missing loops
result = rescorer.rescore_pdb(
    "incomplete_complex.pdb",
    reference_sequences={"A": "MKTLLIFAVLCLGFA...FULL_SEQUENCE"},
)
```

---

## Input Preparation

### PDB/CIF files

The rescorer accepts standard PDB (`.pdb`, `.ent`) and mmCIF (`.cif`, `.mmcif`, `.pdbx`) files. The file must contain:

- **At least one protein chain** with standard or AMBER/CHARMM-style amino acid residue names
- **At least one ligand** as HETATM records (for PDB mode) or in a separate MOL2 file (for receptor mode)

**Supported residue name formats:**

| Convention | Examples | Mapped to |
|---|---|---|
| Standard PDB | ALA, GLY, HIS | Standard 20 AAs |
| AMBER protonation | HIE, HID, HIP, CYX, ASH, GLH | Parent AA |
| AMBER terminal | NALA, CVAL, NMET, CPRO | Parent AA |
| Modified residues | MSE, TPO, SEP, PTR | Parent AA |
| CHARMM | HSD, HSE, HSP, CYM, LYN | Parent AA |

### Handling incomplete structures

PDB files from crystallography often have **missing loops** — residues that were not resolved in the electron density. If these gaps are not addressed, the model will treat gap-flanking residues as direct neighbors, which can reduce prediction quality.

The rescoring module fills gaps automatically using this priority order:

1. **User-provided `--reference-sequence`** (always wins)
2. **SEQRES records** from the PDB file (the deposited biological sequence)
3. **ATOM-derived sequence** (fallback — may contain gaps)

**When to provide a reference sequence:**

- Your PDB lacks SEQRES records (common with AMBER-prepared files, homology models, or docked complexes)
- The SEQRES records are incorrect or incomplete
- You want to use a specific isoform or mutant sequence

**Where to find the full sequence:**

- **UniProt**: Search for your protein at [uniprot.org](https://www.uniprot.org)
- **PDB SEQRES**: Run `grep ^SEQRES your_file.pdb` to check if records exist
- **RCSB**: The "Sequence" tab on any PDB entry page

**Example:** A PDB with residues 1–50 and 61–100 (missing loop at 51–60):

```bash
# Without reference: model sees 90 residues, gap-flanking residues
# treated as adjacent — may degrade prediction near the gap

# With reference: model sees full 100-residue sequence — Boltz-2
# predicts the complete structure including the missing loop
boltz rescore pdb -i complex.pdb \
  --reference-sequence '{"A": "FULL_100_RESIDUE_SEQUENCE"}'
```

### Ligand SMILES

Ligand SMILES are resolved in this priority order:

1. **User-provided `--ligand-smiles`** (always wins)
2. **Auto-inferred** from HETATM 3D coordinates via RDKit `rdDetermineBonds`
3. **Failure** with an actionable error message

**When to provide SMILES explicitly:**

- The auto-inferred SMILES is incorrect (check warnings in output)
- The ligand has unusual bond orders that cannot be inferred from geometry
- You want to use a different protonation state or tautomer

**Format:** JSON mapping chain IDs to SMILES strings:

```bash
# Single ligand
--ligand-smiles '{"B": "c1ccc(NC(=O)c2ccccc2)cc1"}'

# Multiple ligands
--ligand-smiles '{"B": "CCO", "C": "c1ccccc1"}'
```

### MOL2 ligands

For virtual screening with `boltz rescore receptor`, ligands are provided in Tripos MOL2 format. Each `@<TRIPOS>MOLECULE` block in the file is treated as a separate ligand. SMILES are extracted from the MOL2 bond table; if that fails, 3D bond perception is used as a fallback.

Standard MOL2 output from docking programs (AutoDock Vina, GOLD, Glide, rDock, etc.) is supported.

---

## Configuration

### Config file

Create a `rescore_config.yaml` in your working directory for persistent settings:

```yaml
# rescore_config.yaml
mode: production

model:
  checkpoint: auto
  device: auto

inference:
  recycling_steps: 5
  diffusion_samples: 5
  sampling_steps: 200
  affinity_mw_correction: true

validation:
  level: moderate

output:
  format: csv
  include_metadata: true
  include_diagnostics: true

processing:
  max_tokens: 256
  max_atoms: 2048
  max_tokens_protein: 200

logging:
  level: INFO
```

The file is auto-discovered from the current directory. Accepted names: `rescore_config.yaml`, `rescore_config.yml`, `.rescore.yaml`.

### Environment variables

Override any setting via environment variables (highest priority):

| Variable | Setting |
|---|---|
| `BOLTZ_RESCORE_CHECKPOINT` | model.checkpoint |
| `BOLTZ_RESCORE_DEVICE` | model.device |
| `BOLTZ_RESCORE_RECYCLING_STEPS` | inference.recycling_steps |
| `BOLTZ_RESCORE_DIFFUSION_SAMPLES` | inference.diffusion_samples |
| `BOLTZ_RESCORE_VALIDATION` | validation.level |
| `BOLTZ_RESCORE_OUTPUT_FORMAT` | output.format |
| `BOLTZ_RESCORE_LOG_LEVEL` | logging.level |

**Example:**

```bash
BOLTZ_RESCORE_DEVICE=cpu boltz rescore pdb -i complex.pdb
```

### Priority order

Settings are resolved as: **Environment variables** > **Config file** > **CLI flags** > **Defaults**.

---

## Output Formats

| Format | Extension | Use case |
|---|---|---|
| JSON | `.json` | Single results, programmatic consumption |
| CSV | `.csv` | Tabular analysis, spreadsheets |
| Parquet | `.parquet` | Large-scale analytics, DataFrame workflows |
| SQLite | `.db` | Queryable database, integration with pipelines |
| Excel | `.xlsx` | Reports, sharing with collaborators |

### JSON output structure

```json
{
  "id": "complex_1",
  "source_file": "/path/to/complex.pdb",
  "affinity_pred": 6.423,
  "affinity_std": 0.314,
  "affinity_probability_binary": 0.871,
  "protein_chain": "A",
  "ligand_chains": "B",
  "protein_residue_count": 285,
  "ligand_atom_count": 32,
  "processing_time_ms": 12543.2,
  "validation_status": "SUCCESS",
  "warnings": ""
}
```

### CSV column reference

| Column | Type | Description |
|---|---|---|
| `id` | str | Complex identifier (stem of input filename) |
| `affinity_pred` | float | Predicted pKd value |
| `affinity_std` | float | Uncertainty estimate (std across diffusion samples) |
| `affinity_probability_binary` | float | Probability of being a binder (0–1) |
| `protein_chain` | str | Protein chain ID used |
| `ligand_chains` | str | Comma-separated ligand chain IDs |
| `protein_residue_count` | int | Number of protein residues |
| `ligand_atom_count` | int | Number of ligand atoms |
| `processing_time_ms` | float | Total processing time in milliseconds |
| `validation_status` | str | `SUCCESS`, `WARNING`, or `FAILED` |
| `warnings` | str | Semicolon-separated warning messages |
| `error_message` | str | Error details (empty on success) |

---

## Examples

### 1. Basic single-complex rescoring

```bash
boltz rescore pdb -i examples/complex.pdb -o result.json
```

### 2. Virtual screening pipeline

```bash
# Step 1: Prepare receptor (strip waters, add H if needed)
# Step 2: Dock ligands with your favorite tool → output.mol2
# Step 3: Rescore with Boltz-2

boltz rescore receptor \
  --receptor receptor_prepared.pdb \
  --ligands docked_poses.mol2 \
  -o boltz_scores.csv \
  --sort-by affinity_score \
  --device cuda
```

### 3. Batch processing a dataset

```bash
boltz rescore batch \
  --input-dir ./pdbbind_refined/ \
  --recursive \
  --output-format parquet \
  -o pdbbind_scores.parquet \
  --device cuda
```

### 4. Handling an AMBER-prepared PDB

AMBER-prepared PDBs often have non-standard residue names and no SEQRES records:

```bash
boltz rescore pdb \
  -i amber_complex.pdb \
  --reference-sequence '{"A": "MKTLLIFAVLCLGFA..."}' \
  --ligand-smiles '{"B": "CC(=O)Oc1ccccc1C(=O)O"}' \
  --validation lenient
```

### 5. Using the Python API in a Jupyter notebook

```python
from boltz.affinity_rescoring import AffinityRescorer

rescorer = AffinityRescorer(device="cuda")

# Score a set of PDB files
import glob
results = []
for pdb in glob.glob("complexes/*.pdb"):
    r = rescorer.rescore_pdb(pdb)
    results.append({"file": pdb, "pKd": r.affinity_pred, "prob": r.affinity_probability_binary})

import pandas as pd
df = pd.DataFrame(results).sort_values("pKd", ascending=False)
print(df.head(20))
```

### 6. Manifest-based pipeline

```yaml
# screening.yaml
complexes:
  - pdb: data/CDK2_inhibitor1.pdb
    protein_chain: A
    ligand_smiles:
      B: "CC(=O)Nc1ccc2[nH]cc(-c3ccncc3)c2c1"

  - pdb: data/CDK2_inhibitor2.pdb

  - pdb: data/CDK2_inhibitor3.cif
    ligand_chains: "B,C"
```

```bash
boltz rescore manifest --manifest screening.yaml --output-dir results/
```

---

## Troubleshooting

### "No protein chains detected"

The rescorer could not identify a protein chain. Common causes:

- All residues have non-standard names not in the mapping table
- The file only contains HETATM records
- The chain has fewer than 10 residues

**Fix:** Specify the chain explicitly with `--protein-chain A`.

### "No ligand detected"

No HETATM group was identified as a ligand. Common causes:

- Ligand is recorded as ATOM (not HETATM)
- Ligand was stripped during preparation
- All HETATM records are water/ions

**Fix:** Specify ligand chains with `--ligand-chains B` or use the receptor workflow with a MOL2 file.

### "Could not determine ligand SMILES"

Auto-inference from 3D coordinates failed. Common causes:

- Very few heavy atoms (< 3)
- Unusual element types or missing coordinates
- Metal-complex ligands

**Fix:** Provide SMILES explicitly: `--ligand-smiles '{"B": "SMILES_HERE"}'`

### "Chain A: N gap(s) totaling M missing residues"

The PDB has missing loops. The rescorer automatically uses SEQRES records to fill them. If SEQRES records are absent:

**Fix:** Provide the full sequence: `--reference-sequence '{"A": "FULL_SEQ"}'`

### Slow performance on CPU

Boltz-2 structure prediction is computationally intensive. Expected times:

| Hardware | ~300 residues | ~500 residues |
|---|---|---|
| NVIDIA A100 | ~30s | ~60s |
| NVIDIA RTX 3090 | ~60s | ~120s |
| CPU (Apple M2) | ~10min | ~30min |

**Fix:** Use `--device cuda` with a CUDA-capable GPU.

### Out of memory (OOM)

The complex exceeds the token/atom budget. Boltz-2 affinity cropping limits:

- `max_tokens`: 256 (residues + ligand tokens)
- `max_atoms`: 2048
- `max_tokens_protein`: 200

Large proteins are automatically cropped to the binding pocket neighborhood. If OOM persists, reduce `diffusion_samples` in the config.

---

## Architecture

```
boltz rescore pdb -i complex.pdb
        │
        ▼
┌─────────────────────┐
│  Structure Parsing   │  gemmi: PDB/CIF → atoms, metadata, SEQRES
│  (parsers.py)        │  Non-standard residue name mapping
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Validation          │  File integrity, atom counts, chain detection
│  (validation.py)     │  Strictness: strict / moderate / lenient
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Sequence Resolution │  SEQRES > ATOM records > reference override
│  (parsers.py)        │  Gap detection and filling
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  SMILES Resolution   │  User SMILES > rdDetermineBonds > error
│  (smiles_inference)  │  3-tier bond perception fallback
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Boltz-2 Prediction  │  YAML → boltz predict → structure + affinity
│  (inference.py)      │  Diffusion sampling → AffinityModule → pKd
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Results Export       │  JSON / CSV / Parquet / SQLite / Excel
│  (export.py)         │  Batch summaries, metadata, diagnostics
└─────────────────────┘
```

---

## License

MIT — see [LICENSE](LICENSE).
