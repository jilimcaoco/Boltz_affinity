# Affinity Rescoring User Guide

The affinity rescoring module scores protein–ligand binding affinity using the
Boltz-2 trunk and pairformer architecture with a dedicated affinity head. Unlike
the `boltz predict` pipeline, it **does not run diffusion or confidence
estimation**. Instead, it injects experimentally determined coordinates directly
from PDB/CIF and MOL2 files and evaluates binding affinity in a single forward
pass.

The model outputs a predicted pIC50 value. To convert to binding free energy:

$$\Delta G = (6 - \text{pIC50}) \times 1.364 \;\text{kcal/mol}$$

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Commands](#cli-commands)
   - [rescore receptor](#rescore-receptor) (primary workflow)
   - [rescore pdb](#rescore-pdb)
   - [rescore batch](#rescore-batch)
   - [rescore manifest](#rescore-manifest)
4. [MSA Handling](#msa-handling)
5. [MOL2 Input Format](#mol2-input-format)
6. [Output Formats](#output-formats)
7. [Validation Levels](#validation-levels)
8. [Python API](#python-api)
9. [SLURM Batch Script](#slurm-batch-script)
10. [Troubleshooting](#troubleshooting)

---

## Installation

Requires Python ≥3.10, <3.13.

```bash
# Clone and install
git clone <repository-url> && cd Boltz_affinity
pip install -e .

# Install affinity-specific extras (polars, openpyxl, etc.)
pip install -r requirements_affinity.txt
```

Key dependencies: `torch>=2.2`, `rdkit>=2024.3.2`, `biopython==1.84`,
`gemmi==0.6.5`, `pytorch-lightning==2.5.0`.

The affinity checkpoint (`boltz2_aff.ckpt`) is downloaded automatically on
first use to `~/.boltz/`. To use a local checkpoint, pass `--checkpoint <path>`.

---

## Quick Start

The standard use case is scoring a receptor PDB against multiple ligand poses
in a multi-molecule MOL2 file.

> **MSA policy.** `--use-msa-server` is **disabled** in this fork — every
> MSA must be pre-computed once and reused (see
> [Pre-computing MSAs](prediction.md#pre-computing-msas)). Pass the
> directory containing the cached `.a3m` files via `--msa-directory`
> (or export `BOLTZ_MSA_CACHE_DIR=/shared/msa_cache`).

```bash
# 1. One-time precompute (only needed once per unique sequence):
python -m boltz.affinity_rescoring.mmseqs2 \
    --sequence "$(cat receptor_seq.txt)" \
    --cache-dir /shared/msa_cache

# 2. Routine scoring — never touches the MSA server:
boltz rescore receptor \
    --receptor protein.pdb \
    --ligands  ligands.mol2 \
    --output   scores.csv \
    --msa-directory /shared/msa_cache
```

This will:

1. Parse the receptor PDB and extract the protein chain.
2. Split the MOL2 file into individual ligands.
3. For each ligand: infer SMILES from 3D coordinates, build a Boltz-2 input
   YAML, inject experimental coordinates, and run affinity inference.
4. Write sorted results to `scores.csv`.

---

## CLI Commands

All commands are subcommands of `boltz rescore`:

```
boltz rescore <COMMAND> [OPTIONS]
```

### `rescore receptor`

**Score one receptor against many ligands.** This is the primary workflow for
virtual screening and FEP benchmark rescoring.

```bash
boltz rescore receptor \
    --receptor    /path/to/receptor.pdb \
    --ligands     /path/to/ligands.mol2 \
    --output      scores.csv \
    --output-format csv \
    --use-msa-server \
    --device cuda
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--receptor` | Path | *required* | Receptor PDB or CIF file. |
| `--ligands` | Path | *required* | Multi-molecule MOL2 file with ligand poses. |
| `--output` / `-o` | str | `scores.csv` | Output file path. |
| `--protein-chain` | str | auto-detect | Protein chain ID in the receptor file. |
| `--output-format` | `json\|csv\|parquet\|excel` | `csv` | Output format. |
| `--device` | `auto\|cuda\|cpu\|mps` | `auto` | Inference device. |
| `--validation` | `strict\|moderate\|lenient` | `moderate` | Validation strictness. |
| `--checkpoint` | str | `auto` | Path to checkpoint, or `auto` to download. |
| `--sort-by` | `affinity_score\|confidence\|ligand_name\|n_atoms` | `affinity_score` | Sort column for results. |
| `--reference-sequence` | str | `None` | Full biological sequence as JSON, e.g. `'{"A": "MKTL..."}'`. Overrides SEQRES. |
| `--use-msa-server` | flag | `False` | Generate MSA via ColabFold server. |
| `--msa-directory` | Path | `None` | Directory with pre-computed MSAs or cache location. See [MSA Handling](#msa-handling). |
| `--log-level` | `DEBUG\|INFO\|WARNING\|ERROR` | `INFO` | Logging verbosity. |

#### Examples

Score with pre-computed MSA (HPC, no internet):

```bash
boltz rescore receptor \
    --receptor  receptor.pdb \
    --ligands   docked_poses.mol2 \
    --output    results.csv \
    --msa-directory /scratch/msa_cache/ \
    --device cuda
```

Score with explicit protein chain and custom checkpoint:

```bash
boltz rescore receptor \
    --receptor     complex.pdb \
    --ligands      ligands.mol2 \
    --protein-chain A \
    --checkpoint   /models/boltz2_aff.ckpt \
    --output       scores.parquet \
    --output-format parquet
```

### `rescore pdb`

**Score a single PDB/CIF file** containing both protein and ligand.

```bash
boltz rescore pdb \
    --input  complex.pdb \
    --output result.json \
    --use-msa-server
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--input` / `-i` | Path | *required* | PDB or CIF structure file. |
| `--output` / `-o` | str | `<input>_affinity.<fmt>` | Output file path. |
| `--protein-chain` | str | auto-detect | Protein chain ID. |
| `--ligand-chains` | str | auto-detect | Comma-separated ligand chain IDs. |
| `--ligand-smiles` | str | auto-infer | SMILES as JSON: `'{"B": "CCO"}'`. |
| `--output-format` | `json\|csv\|parquet\|sqlite\|excel` | `json` | Output format. |
| `--device` | `auto\|cuda\|cpu\|mps` | `auto` | Inference device. |
| `--validation` | `strict\|moderate\|lenient` | `moderate` | Validation strictness. |
| `--checkpoint` | str | `auto` | Checkpoint path. |
| `--dry-run` | flag | `False` | Validate inputs without running inference. |
| `--use-msa-server` | flag | `False` | Generate MSA via server. |
| `--reference-sequence` | str | `None` | Override sequence as JSON. |
| `--log-level` | `DEBUG\|INFO\|WARNING\|ERROR` | `INFO` | Logging verbosity. |

### `rescore batch`

**Score all PDB/CIF files in a directory.**

```bash
boltz rescore batch \
    --input-dir  structures/ \
    --output     batch_results.csv \
    --recursive \
    --use-msa-server
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | Path | *required* | Directory containing PDB/CIF files. |
| `--output` / `-o` | str | `batch_results.csv` | Output file path. |
| `--output-format` | `json\|csv\|parquet\|sqlite\|excel` | `csv` | Output format. |
| `--recursive` | flag | `False` | Scan subdirectories. |
| `--device` | `auto\|cuda\|cpu\|mps` | `auto` | Inference device. |
| `--validation` | `strict\|moderate\|lenient` | `moderate` | Validation strictness. |
| `--checkpoint` | str | `auto` | Checkpoint path. |
| `--ligand-smiles` | str | auto-infer | SMILES JSON applied to all files. |
| `--use-msa-server` | flag | `False` | Generate MSA via server. |
| `--log-level` | `DEBUG\|INFO\|WARNING\|ERROR` | `INFO` | Logging verbosity. |

### `rescore manifest`

**Score complexes from a YAML manifest file.** Each entry specifies a
receptor–ligand pair.

```bash
boltz rescore manifest \
    --manifest  complexes.yaml \
    --output-dir ./scores \
    --use-msa-server
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--manifest` | Path | *required* | YAML manifest listing complexes. |
| `--output-dir` | str | `./scores` | Output directory. |
| `--output-format` | `json\|csv\|parquet` | `json` | Output format. |
| `--device` | str | `auto` | Inference device. |
| `--checkpoint` | str | `auto` | Checkpoint path. |
| `--use-msa-server` | flag | `False` | Generate MSA via server. |
| `--log-level` | str | `INFO` | Logging verbosity. |

---

## MSA Handling

The affinity model requires a Multiple Sequence Alignment (MSA) for the protein
sequence. There are three strategies:

### 1. Server-generated MSA (simplest)

Use `--use-msa-server` to query the ColabFold MMseqs2 server automatically.
Requires internet access.

```bash
boltz rescore receptor \
    --receptor receptor.pdb \
    --ligands  ligands.mol2 \
    --use-msa-server
```

### 2. Pre-computed MSA directory (HPC recommended)

Generate MSAs on a login node or local machine, then pass the directory to
the rescorer. The directory should contain `.a3m` files (one per protein
sequence). For multi-chain proteins, use `.csv` format with `sequence` and
`key` columns.

```bash
# Step 1: Compute MSA on login node (internet available)
boltz rescore receptor \
    --receptor receptor.pdb \
    --ligands  first_ligand.mol2 \
    --use-msa-server \
    --msa-directory /scratch/msa_cache/

# Step 2: Reuse cached MSA on compute node (no internet)
boltz rescore receptor \
    --receptor receptor.pdb \
    --ligands  all_ligands.mol2 \
    --msa-directory /scratch/msa_cache/
```

When `--msa-directory` is provided:

- If the directory contains matching MSA files, they are used directly.
- If `--use-msa-server` is also set and no cached MSA exists, the server
  result is saved to the directory for future reuse.
- The MSA is resolved **once** and shared across all ligands for the same
  receptor, avoiding redundant computation.

### 3. YAML-level MSA specification

When using the manifest command or Python API, you can set the `msa` field per
protein entity in the input YAML:

```yaml
sequences:
  - protein:
      id: A
      sequence: MKTLVL...
      msa: /path/to/protein.a3m
  - ligand:
      id: B
      smiles: 'CCO'
properties:
  - affinity:
      binder: B
```

---

## MOL2 Input Format

The `--ligands` option for `rescore receptor` accepts a standard Sybyl MOL2
file containing one or more molecules. Each molecule block starts with
`@<TRIPOS>MOLECULE`.

```
@<TRIPOS>MOLECULE
LIG_001
 23 24 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1         1.2345   2.3456   3.4567 C.ar      1 LIG_001    -0.0352
      2 C2         1.5678   2.6789   3.7890 C.ar      1 LIG_001    -0.0618
      ...
@<TRIPOS>BOND
     1     1     2 ar
     2     2     3 ar
     ...
@<TRIPOS>MOLECULE
LIG_002
...
```

The parser automatically:

- Splits multi-molecule files into individual ligands.
- Removes waters, ions, and common solvents.
- Preserves ligand names from the MOLECULE record.
- Extracts atom coordinates, bonds, and partial charges.
- Infers SMILES from 3D coordinates using RDKit `rdDetermineBonds` (3-attempt
  fallback: all atoms → heavy-only → connectivity-only).

---

## Output Formats

Six export formats are supported:

| Format | Extension | Description |
|---|---|---|
| JSON | `.json` | Nested structure with metadata, summary, and results. |
| JSONL | `.jsonl` | One JSON object per line (streaming-friendly). |
| CSV | `.csv` | Flat tabular format via `csv.DictWriter`. |
| Parquet | `.parquet` | Columnar format via polars (or pandas + pyarrow). |
| SQLite | `.sqlite` | Database with `results` table (indexed on `affinity_pred`, `validation_status`) and `metadata` table. |
| Excel | `.xlsx` | Multi-sheet workbook (All Results, Successful, Failed, Metadata). |

For `rescore receptor`, the output contains one row per ligand with columns:

| Column | Description |
|---|---|
| `ligand_name` | Name from MOL2 molecule record. |
| `affinity_score` | Predicted pIC50. |
| `affinity_uncertainty` | Score uncertainty (ensemble std). |
| `confidence` | Confidence estimate. |
| `n_atoms` | Number of ligand atoms. |
| `n_unresolved_near_pocket` | Unresolved protein residues near the binding site. |
| `validation_status` | `SUCCESS`, `WARNING`, or `FAILED`. |
| `validation_issues` | Description of any issues. |
| `processing_ms` | Wall-clock time per ligand. |

---

## Validation Levels

Three levels control how strictly inputs are checked:

| Level | Behavior |
|---|---|
| `strict` | Rejects files with any warnings (alt locations, missing atoms, high B-factors). |
| `moderate` | Accepts minor issues with warnings. Rejects invalid coordinates (NaN/Inf) and missing chains. |
| `lenient` | Accepts most inputs. Only rejects files that cannot be parsed at all. |

Checks performed: file format headers, coordinate ranges, NaN/Inf coordinates,
occupancy values, B-factors (>200 Å² triggers warning), element symbols, steric
clashes (<1.0 Å), chain continuity gaps (>5 residues).

---

## Python API

```python
from boltz.affinity_rescoring import AffinityRescorer

# Initialize (downloads checkpoint on first use)
rescorer = AffinityRescorer(
    checkpoint="auto",       # or path to .ckpt file
    device="auto",           # "cuda", "cpu", "mps"
    validation_level="moderate",
)

# Score receptor against multiple MOL2 ligands
scores = rescorer.rescore_receptor(
    receptor_path="protein.pdb",
    ligands_path="ligands.mol2",
    output_path="scores.csv",
    output_format="csv",
    sort_by="affinity_score",
    use_msa_server=True,
    # msa_directory="/scratch/msa_cache/",  # for HPC
)

# Each score is a LigandScore dataclass
for s in scores:
    print(f"{s.ligand_name}: pIC50={s.affinity_score:.2f} "
          f"± {s.affinity_uncertainty:.2f}")
```

### Single PDB scoring

```python
result = rescorer.rescore_pdb(
    input_path="complex.pdb",
    use_msa_server=True,
)
print(f"Affinity: {result.affinity_pred:.3f}")
```

### Batch scoring

```python
results = rescorer.rescore_directory(
    input_dir="structures/",
    output_path="batch.csv",
    output_format="csv",
    recursive=True,
    use_msa_server=True,
)
```

### Dry run (validate without inference)

```python
report = rescorer.dry_run(
    input_path="complex.pdb",
    protein_chain="A",
    ligand_chains=["B"],
)
```

### Export results

```python
rescorer.export_results(
    results=results,
    output_path="results.parquet",
    format="parquet",
)
```

---

## SLURM Batch Script

Below is an example SLURM script for running `rescore receptor` on an HPC
cluster. It uses a two-stage workflow: first generate the MSA on a login/GPU
node with internet, then score ligands on a GPU compute node.

### Stage 1: Cache MSA (run on login node or node with internet)

```bash
boltz rescore receptor \
    --receptor  /data/receptor.pdb \
    --ligands   /data/first_ligand.mol2 \
    --msa-directory /scratch/$USER/msa_cache/ \
    --use-msa-server \
    --device cpu \
    --output /dev/null
```

### Stage 2: SLURM job for scoring

```bash
#!/bin/bash
#SBATCH --job-name=boltz-rescore
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rescore_%j.out
#SBATCH --error=logs/rescore_%j.err

# ── Configuration ────────────────────────────────────────────
RECEPTOR="/data/receptor.pdb"
LIGANDS="/data/ligands.mol2"
MSA_DIR="/scratch/$USER/msa_cache"
CHECKPOINT="/scratch/$USER/.boltz/boltz2_aff.ckpt"
OUTPUT_DIR="/scratch/$USER/results"
CONDA_ENV="boltz_env"

# ── Environment ──────────────────────────────────────────────
module load cuda/12.1
source activate "$CONDA_ENV"

mkdir -p "$OUTPUT_DIR" logs

# ── Run ──────────────────────────────────────────────────────
echo "Starting affinity rescoring: $(date)"
echo "Receptor:  $RECEPTOR"
echo "Ligands:   $LIGANDS"
echo "MSA dir:   $MSA_DIR"

boltz rescore receptor \
    --receptor       "$RECEPTOR" \
    --ligands        "$LIGANDS" \
    --output         "$OUTPUT_DIR/scores.csv" \
    --output-format  csv \
    --msa-directory  "$MSA_DIR" \
    --checkpoint     "$CHECKPOINT" \
    --device         cuda \
    --sort-by        affinity_score \
    --log-level      INFO

echo "Completed: $(date)"
echo "Results:   $OUTPUT_DIR/scores.csv"
```

Submit with:

```bash
sbatch rescore.slurm
```

### Array job (multiple receptors)

For scoring multiple receptor–ligand sets, use a SLURM array job with a
manifest TSV:

```bash
#!/bin/bash
#SBATCH --job-name=boltz-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/rescore_%A_%a.out
#SBATCH --error=logs/rescore_%A_%a.err

# manifest.tsv columns: receptor_pdb  mol2_file  output_name
MANIFEST="manifest.tsv"
CONDA_ENV="boltz_env"
MSA_DIR="/scratch/$USER/msa_cache"
CHECKPOINT="/scratch/$USER/.boltz/boltz2_aff.ckpt"
RESULTS_DIR="/scratch/$USER/results"

source activate "$CONDA_ENV"
mkdir -p "$RESULTS_DIR" logs

# Read row for this array task (skip header)
LINE=$(awk -v idx="$SLURM_ARRAY_TASK_ID" 'NR==idx+2' "$MANIFEST")
RECEPTOR=$(echo "$LINE" | cut -f1)
LIGANDS=$(echo "$LINE" | cut -f2)
NAME=$(echo "$LINE" | cut -f3)

echo "Task $SLURM_ARRAY_TASK_ID: $NAME"

boltz rescore receptor \
    --receptor       "$RECEPTOR" \
    --ligands        "$LIGANDS" \
    --output         "$RESULTS_DIR/${NAME}_scores.csv" \
    --output-format  csv \
    --msa-directory  "$MSA_DIR" \
    --checkpoint     "$CHECKPOINT" \
    --device         cuda \
    --sort-by        affinity_score

echo "Done: $NAME"
```

---

## Troubleshooting

### `TypeError: AtomDiffusion.__init__() got an unexpected keyword argument 'mse_rotational_alignment'`

This occurs when the checkpoint contains hyperparameters that the current code
version does not expect. The rescoring module patches the checkpoint
automatically in `inference.py` — ensure you are using the latest version of
this module. If the error persists, verify that you are importing from
`boltz.affinity_rescoring` and not calling `Boltz2.load_from_checkpoint`
directly.

### MSA errors on HPC compute nodes

Compute nodes typically lack internet access. Pre-compute MSAs on a login node
using `--use-msa-server --msa-directory /path/to/cache/`, then reference the
same directory on compute nodes without `--use-msa-server`.

### SMILES inference failures

If a ligand's SMILES cannot be inferred from its 3D coordinates (unusual
elements, missing hydrogens, fragmented molecules), provide them explicitly:

```bash
boltz rescore pdb \
    --input complex.pdb \
    --ligand-smiles '{"B": "c1ccc(cc1)CC(=O)O"}' \
    --use-msa-server
```

### `n_unresolved_near_pocket` warnings

This field reports how many protein residues near the binding pocket had atoms
that could not be matched to the Boltz-2 internal representation (e.g. missing
side-chain atoms in the PDB). Values > 0 indicate potential inaccuracy.
Consider using a more complete structure or filling missing atoms with a tool
like PDBFixer.

### Validation failures

Relax validation if your structures have minor issues:

```bash
boltz rescore receptor \
    --receptor receptor.pdb \
    --ligands  ligands.mol2 \
    --validation lenient \
    --use-msa-server
```

### Configuration via environment variables

The rescoring module respects `BOLTZ_RESCORE_*` environment variables. For
example:

```bash
export BOLTZ_RESCORE_DEVICE=cuda
export BOLTZ_RESCORE_CHECKPOINT=/models/boltz2_aff.ckpt
export BOLTZ_RESCORE_LOG_LEVEL=DEBUG
```

Alternatively, create a `rescore_config.yaml` in your working directory to
set defaults persistently.
