# Affinity Rescoring — User Guide

This guide covers every feature of the Boltz-2 affinity rescoring module in depth. For a quick overview, see [AFFINITY_README.md](../AFFINITY_README.md).

---

## Table of Contents

1. [Getting Started](#1-getting-started)
   - [System requirements](#system-requirements)
   - [Installation](#installation)
   - [First run](#first-run)
2. [CLI Command Reference](#2-cli-command-reference)
   - [boltz rescore pdb](#boltz-rescore-pdb)
   - [boltz rescore batch](#boltz-rescore-batch)
   - [boltz rescore receptor](#boltz-rescore-receptor)
   - [boltz rescore manifest](#boltz-rescore-manifest)
3. [Python API Reference](#3-python-api-reference)
   - [AffinityRescorer](#affinityrescorer)
   - [Data models](#data-models)
   - [Configuration class](#configuration-class)
4. [Working with Structures](#4-working-with-structures)
   - [Supported file formats](#supported-file-formats)
   - [Chain detection logic](#chain-detection-logic)
   - [Non-standard residue mapping](#non-standard-residue-mapping)
5. [Sequence Handling](#5-sequence-handling)
   - [Why sequences matter](#why-sequences-matter)
   - [SEQRES extraction](#seqres-extraction)
   - [Reference sequence override](#reference-sequence-override)
   - [Gap detection and filling](#gap-detection-and-filling)
6. [Ligand SMILES](#6-ligand-smiles)
   - [Auto-inference pipeline](#auto-inference-pipeline)
   - [Manual SMILES override](#manual-smiles-override)
   - [Multi-ligand complexes](#multi-ligand-complexes)
7. [MOL2 Ligand Files](#7-mol2-ligand-files)
8. [Configuration](#8-configuration)
   - [YAML config file](#yaml-config-file)
   - [Environment variables](#environment-variables)
   - [All settings reference](#all-settings-reference)
9. [Output Formats](#9-output-formats)
   - [JSON](#json)
   - [CSV](#csv)
   - [Parquet](#parquet)
   - [SQLite](#sqlite)
   - [Excel](#excel)
10. [Validation](#10-validation)
11. [Advanced Workflows](#11-advanced-workflows)
    - [Virtual screening](#virtual-screening)
    - [Batch rescoring a benchmark set](#batch-rescoring-a-benchmark-set)
    - [Manifest-based automation](#manifest-based-automation)
    - [Integration with docking tools](#integration-with-docking-tools)
12. [Performance and Resource Usage](#12-performance-and-resource-usage)
13. [Troubleshooting](#13-troubleshooting)
14. [FAQ](#14-faq)

---

## 1. Getting Started

### System requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 or 3.12 |
| GPU | — (CPU works) | NVIDIA GPU with ≥12 GB VRAM |
| RAM | 8 GB | 16 GB+ |
| Disk | 2 GB (for checkpoints) | 5 GB |
| CUDA | — | 11.8+ |

### Installation

#### Option A: PyPI (simplest)

```bash
pip install boltz[cuda] -U
pip install -r requirements_affinity.txt
```

#### Option B: From source (development)

```bash
git clone https://github.com/jwohlwend/boltz.git
cd boltz
pip install -e .[cuda]
pip install -r requirements_affinity.txt
```

#### Option C: Conda environment (isolated)

```bash
conda create -n boltz python=3.11 -y
conda activate boltz
pip install boltz[cuda] -U
pip install -r requirements_affinity.txt
```

#### CPU-only

Omit `[cuda]` from the install command:

```bash
pip install boltz -U
```

#### Dependency details

Core dependencies (installed automatically):

| Package | Version | Purpose |
|---|---|---|
| torch | ≥ 2.2 | Deep learning |
| numpy | ≥ 1.26, < 2.0 | Numerics |
| gemmi | 0.6.5 | PDB/CIF parsing |
| rdkit | ≥ 2024.3.2 | Chemistry / SMILES inference |
| click | 8.1.7 | CLI framework |
| biopython | 1.84 | Sequence utilities |
| pyyaml | 6.0.2 | Config parsing |
| pydantic | ≥ 2.0 | Data validation |

Optional dependencies (from `requirements_affinity.txt`):

| Package | Purpose |
|---|---|
| polars | Fast DataFrame ops for Parquet/CSV |
| openpyxl | Excel output |
| tqdm | Progress bars |

### First run

```bash
# 1. Verify the CLI is available
boltz rescore --help

# 2. Validate a PDB file (no GPU needed)
boltz rescore pdb -i your_complex.pdb --dry-run

# 3. Run a real prediction
boltz rescore pdb -i your_complex.pdb -o result.json
```

On the first real run, checkpoints (~1.2 GB total) are automatically downloaded to `~/.boltz/`.

---

## 2. CLI Command Reference

### boltz rescore pdb

Scores a single protein–ligand complex.

**Synopsis:**

```
boltz rescore pdb -i <PDB_FILE> [OPTIONS]
```

**All options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `-i, --input` | PATH | *required* | Input PDB or CIF file |
| `-o, --output` | PATH | auto-generated | Output file path. Default: `<stem>_affinity.<format>` |
| `--protein-chain` | TEXT | auto-detect | Protein chain ID to use |
| `--ligand-chains` | TEXT | auto-detect | Comma-separated ligand chain IDs |
| `--ligand-smiles` | TEXT | auto-infer | JSON dict of chain_id → SMILES |
| `--reference-sequence` | TEXT | SEQRES | JSON dict of chain_id → full amino acid sequence |
| `--output-format` | CHOICE | `json` | `json` \| `csv` \| `parquet` \| `sqlite` \| `excel` |
| `--device` | CHOICE | `auto` | `auto` \| `cuda` \| `cpu` \| `mps` |
| `--validation` | CHOICE | `moderate` | `strict` \| `moderate` \| `lenient` |
| `--checkpoint` | TEXT | `auto` | Path to checkpoint directory or `auto` |
| `--use-msa-server` | FLAG | false | Enable ColabFold MSA server |
| `--dry-run` | FLAG | false | Validate without inference |
| `--log-level` | CHOICE | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

**Command templates:**

```bash
# Minimal
boltz rescore pdb -i complex.pdb

# Full control
boltz rescore pdb \
  --input complex.pdb \
  --output result.json \
  --protein-chain A \
  --ligand-chains B,C \
  --ligand-smiles '{"B": "CC(=O)Oc1ccccc1C(=O)O", "C": "O=C([O-])CC(O)(CC([O-])=O)C([O-])=O"}' \
  --reference-sequence '{"A": "MKTLLIFAVLCLGFAVDMKVVRQSML..."}' \
  --output-format json \
  --device cuda \
  --validation strict \
  --checkpoint /data/models/boltz/ \
  --use-msa-server \
  --log-level DEBUG
```

---

### boltz rescore batch

Processes all PDB/CIF files in a directory.

**Synopsis:**

```
boltz rescore batch --input-dir <DIR> [OPTIONS]
```

**All options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | PATH | *required* | Directory with PDB/CIF files |
| `-o, --output` | PATH | `batch_results.csv` | Output file |
| `--output-format` | CHOICE | `csv` | `json` \| `csv` \| `parquet` \| `sqlite` \| `excel` |
| `--recursive` | FLAG | false | Process subdirectories too |
| `--device` | CHOICE | `auto` | Compute device |
| `--validation` | CHOICE | `moderate` | Validation level |
| `--checkpoint` | TEXT | `auto` | Checkpoint path |
| `--ligand-smiles` | TEXT | auto-infer | JSON map of chain → SMILES (shared for all files) |
| `--use-msa-server` | FLAG | false | Use MSA server |
| `--log-level` | CHOICE | `INFO` | Logging level |

**Command templates:**

```bash
# Score all PDBs in a flat directory
boltz rescore batch --input-dir ./complexes/ -o results.csv

# Recursive with Parquet output
boltz rescore batch \
  --input-dir ./data/pdbs/ \
  --recursive \
  --output-format parquet \
  -o all_scores.parquet \
  --device cuda

# With explicit SMILES for a shared ligand
boltz rescore batch \
  --input-dir ./complexes/ \
  --ligand-smiles '{"B": "CCCCCCCCCC"}'
```

---

### boltz rescore receptor

Scores one receptor against multiple ligands. Primary virtual screening workflow.

**Synopsis:**

```
boltz rescore receptor --receptor <PDB> --ligands <MOL2> [OPTIONS]
```

**All options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--receptor` | PATH | *required* | Receptor PDB/CIF file |
| `--ligands` | PATH | *required* | MOL2 file with ligand structures |
| `-o, --output` | PATH | `scores.csv` | Output file |
| `--protein-chain` | TEXT | auto-detect | Protein chain ID |
| `--output-format` | CHOICE | `csv` | `json` \| `csv` \| `parquet` \| `excel` |
| `--device` | CHOICE | `auto` | Compute device |
| `--validation` | CHOICE | `moderate` | Validation level |
| `--checkpoint` | TEXT | `auto` | Checkpoint path |
| `--sort-by` | CHOICE | `affinity_score` | Sort column |
| `--reference-sequence` | TEXT | SEQRES | JSON reference sequence |
| `--log-level` | CHOICE | `INFO` | Logging level |

**Sort options:**

| Value | Meaning |
|---|---|
| `affinity_score` | Sort by predicted pKd (descending — highest affinity first) |
| `confidence` | Sort by prediction confidence (descending) |
| `ligand_name` | Sort by ligand name (alphabetical) |
| `n_atoms` | Sort by ligand size (ascending) |

**Command templates:**

```bash
# Basic virtual screening
boltz rescore receptor \
  --receptor protein.pdb \
  --ligands docked.mol2 \
  -o ranked.csv

# Sort by confidence with Excel output
boltz rescore receptor \
  --receptor protein.pdb \
  --ligands library.mol2 \
  --sort-by confidence \
  --output-format excel \
  -o library_scores.xlsx

# With reference sequence for incomplete receptor
boltz rescore receptor \
  --receptor receptor.pdb \
  --ligands ligands.mol2 \
  --protein-chain A \
  --reference-sequence '{"A": "MKTL..."}'
```

---

### boltz rescore manifest

Processes complexes defined in a YAML manifest.

**Synopsis:**

```
boltz rescore manifest --manifest <YAML> [OPTIONS]
```

**All options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--manifest` | PATH | *required* | YAML manifest file |
| `--output-dir` | PATH | `./scores` | Output directory |
| `--output-format` | CHOICE | `json` | Output format per complex |
| `--device` | CHOICE | `auto` | Compute device |
| `--checkpoint` | TEXT | `auto` | Checkpoint path |
| `--use-msa-server` | FLAG | false | Use MSA server |
| `--log-level` | CHOICE | `INFO` | Logging level |

**Manifest YAML schema:**

```yaml
complexes:
  - pdb: <path>              # required
    protein_chain: <id>      # optional (auto-detect)
    ligand_chains: <ids>     # optional (auto-detect)
    ligand_smiles:           # optional (auto-infer)
      <chain_id>: <smiles>
    reference_sequences:     # optional (SEQRES)
      <chain_id>: <sequence>
```

**Command template:**

```bash
boltz rescore manifest \
  --manifest my_screening.yaml \
  --output-dir ./results/ \
  --output-format csv \
  --device cuda
```

---

## 3. Python API Reference

### AffinityRescorer

```python
from boltz.affinity_rescoring import AffinityRescorer
```

**Constructor:**

```python
AffinityRescorer(
    checkpoint: str = "auto",
    device: str = "auto",
    validation_level: str = "moderate",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `checkpoint` | str | `"auto"` | Path to checkpoint dir or `"auto"` to download |
| `device` | str | `"auto"` | `"auto"`, `"cuda"`, `"cpu"`, `"mps"` |
| `validation_level` | str | `"moderate"` | `"strict"`, `"moderate"`, `"lenient"` |

**Methods:**

#### `rescore_pdb()`

```python
result = rescorer.rescore_pdb(
    input_path: str,
    protein_chain: str = None,           # auto-detect if None
    ligand_chains: list[str] = None,     # auto-detect if None
    output_path: str = None,
    output_format: str = "json",
    ligand_smiles: dict[str, str] = None,
    use_msa_server: bool = False,
    reference_sequences: dict[str, str] = None,
) -> AffinityResult
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_path` | str | *required* | PDB/CIF file path |
| `protein_chain` | str | None | Protein chain (None = auto) |
| `ligand_chains` | list[str] | None | Ligand chains (None = auto) |
| `output_path` | str | None | Save results to file |
| `output_format` | str | `"json"` | Output format |
| `ligand_smiles` | dict | None | Chain → SMILES mapping |
| `use_msa_server` | bool | False | Use ColabFold MSA server |
| `reference_sequences` | dict | None | Chain → full sequence mapping |

**Returns:** `AffinityResult`

#### `rescore_receptor()`

```python
scores = rescorer.rescore_receptor(
    receptor_path: str,
    ligands_path: str,
    protein_chain: str = None,
    output_path: str = None,
    output_format: str = "csv",
    sort_by: str = "affinity_score",
    reference_sequences: dict[str, str] = None,
) -> list[LigandScore]
```

**Returns:** `list[LigandScore]` sorted by `sort_by`.

#### `rescore_directory()`

```python
results = rescorer.rescore_directory(
    input_dir: str,
    output_path: str = None,
    output_format: str = "csv",
    recursive: bool = False,
    ligand_smiles: dict[str, str] = None,
    use_msa_server: bool = False,
) -> list[AffinityResult]
```

#### `rescore_batch()`

```python
results = rescorer.rescore_batch(
    pdb_files: list[str],
    output_path: str = None,
    output_format: str = "csv",
    ligand_smiles: dict[str, str] = None,
    use_msa_server: bool = False,
) -> list[AffinityResult]
```

#### `dry_run()`

```python
report = rescorer.dry_run(
    input_path: str,
    protein_chain: str = None,
    ligand_chains: list[str] = None,
) -> dict
```

**Returns:** A dict with keys:
- `chains` — detected protein/ligand chains
- `sequences` — extracted sequences and lengths
- `issues` — validation warnings and errors
- `summary` — human-readable summary

### Data models

#### `AffinityResult`

| Field | Type | Description |
|---|---|---|
| `id` | str | Complex identifier |
| `source_file` | str | Input file path |
| `affinity_pred` | float | Predicted pKd |
| `affinity_std` | float | Uncertainty (std dev) |
| `affinity_probability_binary` | float | Binder probability (0–1) |
| `affinity_pred_ensemble` | float \| None | Ensemble pKd |
| `affinity_std_ensemble` | float \| None | Ensemble uncertainty |
| `protein_chain` | str | Protein chain used |
| `ligand_chains` | str | Ligand chain(s) used |
| `protein_residue_count` | int | Residue count |
| `ligand_atom_count` | int | Ligand atom count |
| `processing_time_ms` | float | Processing time (ms) |
| `validation_status` | ValidationStatus | SUCCESS / WARNING / FAILED |
| `warnings` | str | Warning messages |
| `error_message` | str | Error details |
| `metadata` | dict | Additional metadata |

#### `LigandScore`

| Field | Type | Description |
|---|---|---|
| `ligand_name` | str | Ligand name from MOL2 |
| `ligand_smiles` | str | SMILES string |
| `affinity_score` | float | Predicted pKd |
| `affinity_std` | float | Uncertainty |
| `confidence` | float | Prediction confidence |
| `n_atoms` | int | Number of atoms |
| `rank` | int | Rank position |
| `processing_time_ms` | float | Processing time |
| `validation_status` | str | Validation result |
| `warnings` | str | Warning messages |
| `error_message` | str | Error details |

#### `BatchSummary`

| Field | Type | Description |
|---|---|---|
| `total` | int | Total complexes processed |
| `succeeded` | int | Successful predictions |
| `failed` | int | Failed predictions |
| `mean_affinity` | float | Mean pKd across successful |
| `std_affinity` | float | Std dev of pKd |
| `mean_time_ms` | float | Mean processing time |
| `total_time_ms` | float | Total wall time |

### Configuration class

#### `RescoreConfig`

Pydantic model for inference configuration:

```python
from boltz.affinity_rescoring.models import RescoreConfig

config = RescoreConfig(
    checkpoint="auto",
    device="auto",
    recycling_steps=5,          # 1–20
    diffusion_samples=5,        # 1–100
    sampling_steps=200,
    affinity_mw_correction=True,
    validation_level="moderate",
    output_format="json",
    max_tokens=256,
    max_atoms=2048,
    max_tokens_protein=200,
    log_level="INFO",
)
```

---

## 4. Working with Structures

### Supported file formats

| Extension | Format | Notes |
|---|---|---|
| `.pdb` | PDB | Standard protein-ligand complexes |
| `.ent` | PDB | Alternate extension for PDB format |
| `.cif` | mmCIF | Modern PDB archive format |
| `.mmcif` | mmCIF | Alternate extension |
| `.pdbx` | PDBx/mmCIF | Extended mmCIF |
| `.mol2` | Tripos MOL2 | Ligands only (for receptor workflow) |

Files are parsed using [gemmi](https://gemmi.readthedocs.io/) v0.6.5, which handles both PDB and mmCIF formats natively.

### Chain detection logic

When `--protein-chain` and `--ligand-chains` are not specified, the rescorer auto-detects chains:

1. **Protein chains**: Chains where the majority of residues are standard amino acids (>50% of residues)
2. **Ligand chains**: HETATM groups that are not water (HOH, WAT), common ions (NA, CL, CA, MG, ZN, FE, K), or common buffers (SO4, PO4, GOL, EDO, ACT)
3. **Selection**: The longest protein chain and the largest ligand chain(s) within contact distance (< 5 Å) are chosen

If auto-detection fails or selects the wrong chains, specify them explicitly.

### Non-standard residue mapping

The parser automatically maps 50+ non-standard residue names to their parent amino acid:

**AMBER conventions:**

| Non-standard | → Standard | Note |
|---|---|---|
| HIE, HID, HIP | HIS | Histidine protonation states |
| CYX, CYM | CYS | Disulfide, deprotonated cys |
| ASH | ASP | Protonated aspartate |
| GLH | GLU | Protonated glutamate |
| LYN | LYS | Deprotonated lysine |
| NALA, NARG, ... | ALA, ARG, ... | N-terminal variants |
| CALA, CARG, ... | ALA, ARG, ... | C-terminal variants |

**CHARMM conventions:**

| Non-standard | → Standard | Note |
|---|---|---|
| HSD, HSE, HSP | HIS | Histidine tautomers |
| LSN | LYS | Deprotonated lysine |

**Modified residues:**

| Non-standard | → Standard | Note |
|---|---|---|
| MSE | MET | Selenomethionine |
| TPO | THR | Phosphothreonine |
| SEP | SER | Phosphoserine |
| PTR | TYR | Phosphotyrosine |
| MLY | LYS | Methylated lysine |
| CSO | CYS | S-hydroxycysteine |

---

## 5. Sequence Handling

### Why sequences matter

Boltz-2 is a **sequence-to-structure** model. It takes amino acid sequences and ligand SMILES, predicts a 3D structure from scratch, then estimates affinity. The input coordinates are used only to:

1. **Identify chains** — which chain is the protein, which is the ligand
2. **Extract sequences** — the 1D amino acid sequence from residue names
3. **Infer SMILES** — ligand connectivity from 3D coordinates

The completeness and accuracy of the **sequence** directly impacts prediction quality. A sequence with gaps (missing loops) causes the model's relative position encoding to treat gap-flanking residues as neighbors, which distorts the predicted structure.

### SEQRES extraction

PDB files contain SEQRES records that specify the complete biological sequence, including residues not resolved in the electron density. The rescorer extracts these automatically via `gemmi.Entity.full_sequence`.

```
SEQRES   1 A  150  MET LYS THR LEU LEU ILE PHE ALA VAL LEU CYS LEU GLY
SEQRES   2 A  150  PHE ALA VAL ASP MET ...
```

**Verification:**

```bash
# Check if your PDB has SEQRES records
grep ^SEQRES your_file.pdb | head

# Count SEQRES residues vs ATOM residues per chain
boltz rescore pdb -i your_file.pdb --dry-run
```

### Reference sequence override

When SEQRES records are absent or incorrect, provide the full sequence via `--reference-sequence`:

```bash
boltz rescore pdb -i complex.pdb \
  --reference-sequence '{"A": "MKTLLIFAVLCLGFAVDMKVVRQSML..."}'
```

The value is a JSON dictionary mapping chain IDs to one-letter amino acid sequences. Each referenced chain's sequence **fully replaces** any SEQRES-derived or ATOM-derived sequence for that chain.

**Multiple chains:**

```bash
--reference-sequence '{"A": "MKTL...", "B": "GPDA..."}'
```

**Python API:**

```python
result = rescorer.rescore_pdb(
    "complex.pdb",
    reference_sequences={"A": "MKTLLIFAVLCLGFAVDMKVVRQSML..."},
)
```

### Gap detection and filling

The sequence resolution pipeline:

```
Input PDB
   │
   ├─ Has reference_sequences for chain? ──YES──► Use that sequence
   │                                       
   NO
   │
   ├─ Has SEQRES for chain? ──YES──► Use SEQRES sequence
   │
   NO
   │
   └─ Fall back to ATOM-derived sequence (may contain gaps)
       │
       └── Log warning: "N gap(s) totaling M missing residues"
```

When a reference sequence or SEQRES is used, the ATOM records from the PDB are mapped onto the full sequence to determine correct residue indices. This ensures the model's `RelativePositionEncoder` generates accurate position embeddings.

---

## 6. Ligand SMILES

### Auto-inference pipeline

When no SMILES is provided, the rescorer automatically infers SMILES from the ligand's 3D coordinates using RDKit's `rdDetermineBonds` module. This is a three-tier process:

1. **Tier 1 — DetermineBonds with hydrogens**: If the structure includes hydrogen atoms, use full `rdDetermineBonds.DetermineBonds()` on all atoms. This is the most accurate method.

2. **Tier 2 — DetermineBonds on heavy atoms only**: Strip hydrogens, apply `DetermineBonds()` on heavy atoms only. Works for most drug-like molecules.

3. **Tier 3 — DetermineConnectivity fallback**: If bond order determination fails (common for metal complexes or unusual elements), fall back to `DetermineConnectivity()` which assigns single bonds based on distance, then try to infer bond orders heuristically.

If all tiers fail, the rescorer reports an error with guidance to provide SMILES manually.

### Manual SMILES override

Provide SMILES explicitly when:

- Auto-inference produces incorrect SMILES (check the `inferred_smiles` field in verbose output)
- The ligand has unusual chemistry (metal centers, radicals)
- You want a specific tautomer or protonation state
- You want to avoid the (small) computational cost of bond perception

**CLI format:**

```bash
--ligand-smiles '{"B": "c1ccc(NC(=O)c2ccccc2)cc1"}'
```

**Python format:**

```python
result = rescorer.rescore_pdb(
    "complex.pdb",
    ligand_smiles={"B": "c1ccc(NC(=O)c2ccccc2)cc1"},
)
```

### Multi-ligand complexes

For complexes with multiple ligand chains, provide SMILES for each:

```bash
--ligand-smiles '{"B": "CC(=O)Oc1ccccc1C(=O)O", "C": "O"}'
```

If you only provide SMILES for some chains, the remaining chains use auto-inference.

---

## 7. MOL2 Ligand Files

The `boltz rescore receptor` command reads ligands from Tripos MOL2 files. Each `@<TRIPOS>MOLECULE` block is treated as a separate ligand.

**Expected MOL2 structure:**

```
@<TRIPOS>MOLECULE
ligand_1
 32 33 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1          1.0000    2.0000    3.0000 C.ar      1 LIG1       -0.0328
      ...
@<TRIPOS>BOND
     1     1     2 ar
     ...
```

**Multi-molecule files** are supported — each `@<TRIPOS>MOLECULE` block is scored independently against the receptor.

**Compatibility:** Standard MOL2 output from docking programs works directly:

- AutoDock Vina
- GOLD (CCDC)
- Glide (Schrödinger)
- rDock
- PLANTS
- Open Babel conversions (`obabel -o mol2`)

SMILES are derived from the MOL2 bond table. If the bond table is incomplete, 3D coordinates are used for bond perception (same as PDB auto-inference).

---

## 8. Configuration

### YAML config file

Create a config file in your working directory. Auto-discovered filenames:

- `rescore_config.yaml` (preferred)
- `rescore_config.yml`
- `.rescore.yaml`

**Full example with all settings:**

```yaml
# Mode: "production" for normal use, "development" for extra diagnostics
mode: production

model:
  # Path to checkpoint directory, or "auto" for automatic download
  checkpoint: auto
  # Compute device: auto, cuda, cpu, mps
  device: auto

inference:
  # Number of recycling steps in the structure module (1-20)
  recycling_steps: 5
  # Number of diffusion samples to generate (1-100)
  # More samples = better uncertainty estimates but slower
  diffusion_samples: 5
  # Number of diffusion sampling steps
  sampling_steps: 200
  # Apply molecular weight correction to affinity prediction
  affinity_mw_correction: true

validation:
  # strict: Fail on any issue
  # moderate: Warn on issues, fail on critical
  # lenient: Warn only, never fail on validation
  level: moderate

output:
  # Default output format when not specified via CLI
  format: csv
  # Include processing metadata in output
  include_metadata: true
  # Include diagnostic information
  include_diagnostics: true

processing:
  # Maximum number of tokens (residues + ligand tokens) — crops if exceeded
  max_tokens: 256
  # Maximum number of atoms
  max_atoms: 2048
  # Maximum protein tokens (crops to binding pocket neighborhood)
  max_tokens_protein: 200

logging:
  # DEBUG, INFO, WARNING, ERROR
  level: INFO
```

**Load config programmatically:**

```python
from boltz.affinity_rescoring.config import load_config, generate_default_config

# Load from auto-discovered file + env vars
config = load_config()

# Generate default config file
generate_default_config("rescore_config.yaml")
```

### Environment variables

Environment variables take highest priority and override both the config file and CLI flags.

| Variable | Maps to | Example |
|---|---|---|
| `BOLTZ_RESCORE_CHECKPOINT` | `model.checkpoint` | `/data/checkpoints/` |
| `BOLTZ_RESCORE_DEVICE` | `model.device` | `cuda` |
| `BOLTZ_RESCORE_RECYCLING_STEPS` | `inference.recycling_steps` | `10` |
| `BOLTZ_RESCORE_DIFFUSION_SAMPLES` | `inference.diffusion_samples` | `10` |
| `BOLTZ_RESCORE_VALIDATION` | `validation.level` | `strict` |
| `BOLTZ_RESCORE_OUTPUT_FORMAT` | `output.format` | `parquet` |
| `BOLTZ_RESCORE_LOG_LEVEL` | `logging.level` | `DEBUG` |

**Usage:**

```bash
# One-off override
BOLTZ_RESCORE_DEVICE=cpu BOLTZ_RESCORE_LOG_LEVEL=DEBUG boltz rescore pdb -i complex.pdb

# Persistent (add to .bashrc / .zshrc)
export BOLTZ_RESCORE_CHECKPOINT=/data/models/boltz/
export BOLTZ_RESCORE_DEVICE=cuda
```

### All settings reference

| Setting | Type | Default | Range | Description |
|---|---|---|---|---|
| `model.checkpoint` | str | `auto` | — | Checkpoint path or `auto` |
| `model.device` | str | `auto` | auto/cuda/cpu/mps | Compute device |
| `inference.recycling_steps` | int | 5 | 1–20 | Structure module recycling |
| `inference.diffusion_samples` | int | 5 | 1–100 | Diffusion sample count |
| `inference.sampling_steps` | int | 200 | — | Denoising steps |
| `inference.affinity_mw_correction` | bool | true | — | MW correction |
| `validation.level` | str | moderate | strict/moderate/lenient | Validation strictness |
| `output.format` | str | csv | json/csv/parquet/sqlite/excel | Default format |
| `output.include_metadata` | bool | true | — | Include metadata |
| `output.include_diagnostics` | bool | true | — | Include diagnostics |
| `processing.max_tokens` | int | 256 | — | Max tokens |
| `processing.max_atoms` | int | 2048 | — | Max atoms |
| `processing.max_tokens_protein` | int | 200 | — | Max protein tokens |
| `logging.level` | str | INFO | DEBUG/INFO/WARNING/ERROR | Log level |

---

## 9. Output Formats

### JSON

Best for single results and programmatic consumption.

```json
{
  "id": "complex_1",
  "source_file": "/path/to/complex.pdb",
  "affinity_pred": 6.423,
  "affinity_std": 0.314,
  "affinity_probability_binary": 0.871,
  "affinity_pred_ensemble": 6.512,
  "affinity_std_ensemble": 0.298,
  "protein_chain": "A",
  "ligand_chains": "B",
  "protein_residue_count": 285,
  "ligand_atom_count": 32,
  "processing_time_ms": 12543.2,
  "validation_status": "SUCCESS",
  "warnings": "",
  "error_message": "",
  "metadata": {
    "checkpoint": "boltz2_aff.ckpt",
    "device": "cuda:0",
    "recycling_steps": 5,
    "diffusion_samples": 5
  }
}
```

### CSV

Best for batch results, spreadsheet analysis, DataFrame loading.

```csv
id,source_file,affinity_pred,affinity_std,affinity_probability_binary,protein_chain,ligand_chains,protein_residue_count,ligand_atom_count,processing_time_ms,validation_status,warnings,error_message
complex_1,/path/to/complex.pdb,6.423,0.314,0.871,A,B,285,32,12543.2,SUCCESS,,
complex_2,/path/to/complex2.pdb,4.812,0.521,0.432,A,C,312,28,13201.8,WARNING,Chain B excluded: too few atoms,
```

**Load with pandas:**

```python
import pandas as pd
df = pd.read_csv("results.csv")
strong_binders = df[df.affinity_pred > 6.0].sort_values("affinity_pred", ascending=False)
```

### Parquet

Best for large-scale analytics (thousands of complexes). Column-oriented, compressed.

```python
import polars as pl
df = pl.read_parquet("results.parquet")
```

### SQLite

Best for queryable storage and pipeline integration.

```bash
sqlite3 results.db "SELECT id, affinity_pred FROM results WHERE affinity_pred > 6.0 ORDER BY affinity_pred DESC"
```

### Excel

Best for reports and sharing with non-programming collaborators. Requires `openpyxl`.

Output: `.xlsx` file with formatted columns.

---

## 10. Validation

The rescorer validates inputs before running inference. Validation strictness is controlled by `--validation`:

| Level | Behavior |
|---|---|
| **strict** | Fail on any issue — malformed files, small ligands, unusual atoms, etc. Best for production pipelines where data quality must be guaranteed. |
| **moderate** | Warn on minor issues, fail only on critical problems (no protein chain, unreadable file). Good default for exploratory work. |
| **lenient** | Warn on all issues, never fail on validation alone. Use with caution — predictions for severely problematic inputs may be unreliable. |

**What is validated:**

| Check | strict | moderate | lenient |
|---|---|---|---|
| File readability | fail | fail | fail |
| Protein chain found | fail | fail | warn |
| Ligand chain found | fail | fail | warn |
| Minimum protein size (>10 residues) | fail | warn | warn |
| Minimum ligand size (>3 atoms) | fail | warn | warn |
| Max token budget | fail | warn | warn |
| SMILES validity | fail | warn | warn |
| Unknown atom elements | fail | warn | pass |
| Sequence gaps detected | warn | warn | pass |

---

## 11. Advanced Workflows

### Virtual screening

Score a library of docked compounds against a target protein:

```bash
# 1. Prepare receptor
#    - Remove waters and ions
#    - Optionally add hydrogens (not required by Boltz-2)

# 2. Dock your compound library
#    - Use AutoDock Vina, GOLD, Glide, etc.
#    - Export results as MOL2

# 3. Rescore with Boltz-2
boltz rescore receptor \
  --receptor target_prepared.pdb \
  --ligands docked_library.mol2 \
  --sort-by affinity_score \
  --output-format csv \
  -o boltz_ranking.csv \
  --device cuda

# 4. Compare rankings
python - <<'EOF'
import pandas as pd

docking = pd.read_csv("docking_scores.csv")
boltz = pd.read_csv("boltz_ranking.csv")
merged = docking.merge(boltz, on="ligand_name", suffixes=("_dock", "_boltz"))

# Consensus scoring: top 100 from each, intersect
top_dock = set(docking.nlargest(100, "score").ligand_name)
top_boltz = set(boltz.nlargest(100, "affinity_score").ligand_name)
consensus = top_dock & top_boltz
print(f"Consensus hits: {len(consensus)}")
EOF
```

### Batch rescoring a benchmark set

Process an entire dataset like PDBbind:

```bash
# Structure: pdbbind/1a2b/1a2b_complex.pdb, pdbbind/2c3d/2c3d_complex.pdb, ...
boltz rescore batch \
  --input-dir ./pdbbind/ \
  --recursive \
  --output-format parquet \
  -o pdbbind_boltz_scores.parquet \
  --device cuda

# Analyze
python -c "
import polars as pl
df = pl.read_parquet('pdbbind_boltz_scores.parquet')
print(df.describe())
print(f'Success rate: {(df[\"validation_status\"] == \"SUCCESS\").mean():.1%}')
print(f'Mean pKd: {df[\"affinity_pred\"].mean():.2f}')
"
```

### Manifest-based automation

For complex pipelines with per-complex settings:

```yaml
# screening_campaign.yaml
complexes:
  - pdb: data/target1/complex_A.pdb
    protein_chain: A
    ligand_smiles:
      B: "CC(=O)Nc1ccc2c(c1)oc1ccccc12"

  - pdb: data/target1/complex_B.pdb
    protein_chain: A
    ligand_chains: "C"

  - pdb: data/target2/homology_model.pdb
    reference_sequences:
      A: "MKTLLIFAVLCLGFAVDM..."
    ligand_smiles:
      B: "c1cc(F)cc(NC(=O)c2ccncc2)c1"
```

```bash
boltz rescore manifest \
  --manifest screening_campaign.yaml \
  --output-dir results/ \
  --output-format json \
  --device cuda
```

### Integration with docking tools

**AutoDock Vina → Boltz-2:**

```bash
# Vina outputs individual PDBQT files — convert to MOL2
obabel vina_out/*.pdbqt -o mol2 -O all_poses.mol2 --separate

boltz rescore receptor \
  --receptor receptor.pdb \
  --ligands all_poses.mol2 \
  -o vina_rescored.csv
```

**GOLD → Boltz-2:**

```bash
# GOLD outputs a ranked MOL2 directly
boltz rescore receptor \
  --receptor protein.pdb \
  --ligands gold_soln_m1.mol2 \
  -o gold_rescored.csv
```

**Glide → Boltz-2:**

```bash
# Convert Glide output to MOL2
$SCHRODINGER/utilities/structconvert glide_results.maegz -o glide_poses.mol2

boltz rescore receptor \
  --receptor protein.pdb \
  --ligands glide_poses.mol2 \
  -o glide_rescored.csv
```

---

## 12. Performance and Resource Usage

### Expected timings

Per-complex prediction time (including structure prediction + affinity scoring):

| Hardware | ~200 residues | ~300 residues | ~500 residues |
|---|---|---|---|
| NVIDIA A100 (80 GB) | ~20 s | ~30 s | ~60 s |
| NVIDIA RTX 4090 | ~25 s | ~40 s | ~80 s |
| NVIDIA RTX 3090 | ~40 s | ~60 s | ~120 s |
| Apple M2 (CPU) | ~5 min | ~10 min | ~30 min |
| Intel CPU | ~8 min | ~15 min | ~45 min |

Times scale roughly as $O(n^2)$ with the number of tokens due to attention layers.

### Memory usage

| Setting | GPU VRAM | System RAM |
|---|---|---|
| Default (256 tokens) | ~6 GB | ~4 GB |
| Large complex (512 tokens) | ~12 GB | ~6 GB |
| Batch (sequential) | ~6 GB | ~8 GB |

### Optimizing throughput

1. **Use GPU** — 10–50× faster than CPU
2. **Reduce diffusion_samples** — 1 sample is ~5× faster than 5, with less reliable uncertainty
3. **Use batch mode** — amortizes model loading time
4. **Use Parquet output** — faster than CSV for large result sets
5. **Run on multiple GPUs** — split your input directory and run parallel instances

---

## 13. Troubleshooting

### Common errors

| Error | Cause | Fix |
|---|---|---|
| `No protein chains detected` | No chain has >10 standard amino acid residues | Use `--protein-chain` |
| `No ligand detected` | No HETATM groups recognized as ligand | Use `--ligand-chains` or receptor mode |
| `Could not determine ligand SMILES` | Bond perception failed on 3D coordinates | Use `--ligand-smiles` |
| `SMILES validation failed` | Provided SMILES is invalid | Check SMILES string |
| `Token budget exceeded` | Complex too large for model | Will be auto-cropped; reduce max_tokens if OOM |
| `Chain X: N gap(s)` | Missing loops detected | Use `--reference-sequence` or check SEQRES |
| `Checkpoint not found` | Invalid checkpoint path | Use `--checkpoint auto` |
| `CUDA out of memory` | GPU VRAM insufficient | Use smaller `diffusion_samples` or CPU |

### Debugging

Enable verbose logging:

```bash
boltz rescore pdb -i complex.pdb --log-level DEBUG
```

Run a dry run to check inputs without GPU:

```bash
boltz rescore pdb -i complex.pdb --dry-run
```

Check what chains and sequences were detected:

```python
rescorer = AffinityRescorer(device="cpu")
report = rescorer.dry_run("complex.pdb")
print(report)
```

### Getting help

```bash
# General help
boltz rescore --help

# Command-specific help
boltz rescore pdb --help
boltz rescore batch --help
boltz rescore receptor --help
boltz rescore manifest --help
```

---

## 14. FAQ

**Q: Does Boltz-2 use my input coordinates?**
A: No. Boltz-2 is a sequence-to-structure model — it predicts the 3D structure from scratch. Your PDB coordinates are only used to identify chains, extract sequences, and infer SMILES. The quality of the *sequence* matters, not the coordinates.

**Q: Is the predicted pKd calibrated?**
A: The pKd values are predictions from the trained model. They correlate with experimental binding affinities but are not perfectly calibrated. Use `affinity_probability_binary` for classification (binder/non-binder) decisions.

**Q: Can I use AlphaFold-predicted structures?**
A: Yes, but note that only the sequence from the AlphaFold model is used — the predicted coordinates are discarded by Boltz-2. You could equivalently provide just the FASTA sequence and a ligand SMILES.

**Q: What's the difference between `affinity_pred` and `affinity_pred_ensemble`?**
A: `affinity_pred` uses a single pass through the model. `affinity_pred_ensemble` averages predictions across multiple diffusion samples, which is generally more robust.

**Q: Can I score protein–protein interactions?**
A: No. The affinity module is trained on protein–small molecule interactions only.

**Q: What size proteins/ligands are supported?**
A: Proteins up to ~200 residues (in the binding pocket neighborhood) and ligands up to ~100 heavy atoms. Larger proteins are automatically cropped to the binding pocket. The absolute limits are `max_tokens=256` and `max_atoms=2048`.

**Q: Do I need an MSA server?**
A: No. The `--use-msa-server` flag is optional and uses ColabFold's MMseqs2 server for evolutionary sequence search. It may improve predictions for some proteins but adds latency.

**Q: Can I run on Apple Silicon (M1/M2/M3)?**
A: Yes. Use `--device mps` for Metal Performance Shaders acceleration, or `--device cpu` for standard CPU execution. MPS support depends on your PyTorch version.
