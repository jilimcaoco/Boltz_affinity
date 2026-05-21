# Boltz Affinity Rescoring

Affinity-only rescoring for protein-ligand complexes using the Boltz-2 affinity module, plus a full-pipeline mode for alternative binding pocket discovery.

## Overview

Four workflows:

- **Single Complex**: Score one protein-ligand complex from a PDB/CIF file
- **Batch Processing**: Score all complexes in a directory  
- **Virtual Screening**: Score multiple ligands (MOL2) against a single receptor
- **Multi-Pocket**: Run the full Boltz-2 structure prediction with N simultaneous ligand copies to discover alternative binding modes, then score each pocket with the affinity head

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

# ─── Multi-pocket (alternative binding mode discovery) ──────────
# Run full Boltz-2 structure prediction with N ligand copies;
# extract each predicted binding pose and score it.

# From a PDB file (receptor structure known)
boltz rescore multipocket \
  --receptor receptor.pdb \
  --ligand-smiles 'CC1=CC=CC=C1' \
  --n-pockets 5 \
  --output-dir ./mp_results/

# From a sequence only (no structure needed — Boltz folds it)
boltz rescore multipocket \
  --receptor-sequence 'MKTLLILTLVVVTIVCLDLGYT...' \
  --ligand-smiles 'CC1=CC=CC=C1' \
  --n-pockets 5 \
  --output-dir ./mp_seq_results/ \
  --protein-chain A \
  --use-msa-server

# Override protein chain, use pre-downloaded checkpoints
boltz rescore multipocket \
  --receptor receptor.pdb \
  --ligand-smiles 'CC1=CC=CC=C1' \
  --n-pockets 8 \
  --output-dir ./mp_results/ \
  --protein-chain A \
  --checkpoint /path/to/boltz2_conf.ckpt \
  --affinity-checkpoint /path/to/boltz2_aff.ckpt \
  --device cuda \
  --sort-by affinity_probability_binary

# Use MSA server for better structure quality; skip keeping raw Boltz outputs
boltz rescore multipocket \
  --receptor receptor.pdb \
  --ligand-smiles 'CC1=CC=CC=C1' \
  --n-pockets 5 \
  --output-dir ./mp_results/ \
  --use-msa-server \
  --no-keep-boltz-outputs

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

# ─── Multi-pocket pipeline ────────────────────────────────────
from boltz.affinity_rescoring import MultiPocketPipeline

pipeline = MultiPocketPipeline(
    affinity_checkpoint="auto",
    device="auto",
)
report = pipeline.run(
    receptor="receptor.pdb",
    ligand_smiles="CC1=CC=CC=C1",
    n_pockets=5,
    output_dir="./mp_results/",
    sort_by="affinity_pred",
)

# Summarise results
print(f"Pockets extracted: {report.n_pockets_extracted}")
for p in report.pockets:
    print(
        f"  pocket {p.pocket_id} (chain {p.chain_id}): "
        f"pKd={p.affinity_pred:.2f}, P(bind)={p.affinity_probability_binary:.2f}, "
        f"iPTM={p.interface_iptm:.3f}"
    )
# Outputs: mp_results/multipocket_scores.csv
#          mp_results/multipocket_report.html
#          mp_results/structures/pocket_NN_X.pdb  (one per pocket)
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
├── rescorer.py          # Main orchestrator (single / batch / receptor)
├── multipocket.py       # Multi-pocket prediction + scoring pipeline
├── cli.py               # Click CLI commands
└── config.py            # YAML configuration management
```

### Layer Responsibilities

| Layer | Module | Purpose |
|-------|--------|---------|
| **Input** | `parsers.py`, `mol2_parser.py` | Parse structure files |
| **Validation** | `validation.py` | Validate atoms, chains, coordinates |
| **Core** | `rescorer.py` | Orchestrate single / batch / receptor rescoring |
| **Multi-Pocket** | `multipocket.py` | Full prediction + pocket extraction + scoring |
| **Inference** | `inference.py` | Device management, model loading, prediction |
| **Output** | `export.py` | JSON, CSV, JSONL, Parquet, SQLite, Excel, HTML |
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

### `MultiPocketPipeline`

Full Boltz-2 structure prediction + per-pocket rescoring.

```python
from boltz.affinity_rescoring import MultiPocketPipeline

pipeline = MultiPocketPipeline(
    affinity_checkpoint="auto",   # or path to boltz2_aff.ckpt
    structure_checkpoint=None,    # or path to boltz2_conf.ckpt
    device="auto",
    cache_dir=None,               # defaults to ~/.boltz
)

# ── From a PDB file ────────────────────────────────────────────────
report = pipeline.run(
    receptor="receptor.pdb",
    ligand_smiles="CC1=CC=CC=C1",
    n_pockets=5,
    output_dir="./mp_results/",
    sort_by="affinity_pred",
)

# ── From a sequence (no PDB needed) ──────────────────────────────
report = pipeline.run(
    receptor_sequence="MKTLLILTLVVVTIVCLDLGYT...",
    ligand_smiles="CC1=CC=CC=C1",
    n_pockets=5,
    output_dir="./mp_seq_results/",
    protein_chain="A",            # defaults to 'A' when no PDB given
    use_msa_server=True,
)
```

**`run()` parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `receptor` | `str / Path / None` | PDB or CIF file. Either this or `receptor_sequence` required. |
| `receptor_sequence` | `str / None` | Plain amino-acid sequence. Mutually exclusive with `receptor`. |
| `ligand_smiles` | `str` | SMILES of the ligand (replicated N times). |
| `n_pockets` | `int` | Number of ligand copies (1..25). |
| `output_dir` | `str / Path` | Output directory. |
| `protein_chain` | `str / None` | Chain ID; auto-detected from PDB, defaults to `'A'` for sequence input. |
| `recycling_steps` | `int` | Trunk recycling iterations (default 3). |
| `sampling_steps` | `int` | Diffusion steps (default 200). |
| `use_msa_server` | `bool` | Query MMseqs2 server for MSA. |
| `sort_by` | `str` | Ranking column. |
| `ascending` | `bool` | Sort direction. |
| `keep_boltz_outputs` | `bool` | Keep raw Boltz prediction dir. |
| `reference_sequence` | `str / None` | Full-sequence override for PDB with gaps. |

**`run()` returns a `MultiPocketReport`:**

| Field | Type | Description |
|-------|------|-------------|
| `receptor` | `str` | Path to source receptor (or `"sequence-only"`) |
| `ligand_smiles` | `str` | Ligand SMILES used |
| `n_pockets_requested` | `int` | N requested |
| `n_pockets_extracted` | `int` | N successfully extracted |
| `protein_chain` | `str` | Identified protein chain |
| `pockets` | `List[PocketResult]` | Per-pocket results, sorted |
| `csv_path` | `str` | Path to ranked CSV |
| `html_path` | `str` | Path to HTML report |
| `structures_dir` | `str` | Directory of extracted PDB files |
| `boltz_prediction_dir` | `str` | Raw Boltz output directory |
| `total_time_s` | `float` | Wall-clock time |

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

### `PocketResult`

Dataclass for one predicted binding pocket (multi-pocket pipeline).

| Field | Type | Description |
|-------|------|-------------|
| `pocket_id` | `int` | 0-based pocket index |
| `chain_id` | `str` | Ligand chain ID in prediction |
| `structure_path` | `str` | Path to extracted pocket PDB |
| `affinity_pred` | `float` | Affinity-rescored pKd |
| `affinity_std` | `float` | Prediction uncertainty |
| `affinity_probability_binary` | `float` | Binding probability |
| `boltz_confidence_score` | `float` | Boltz-2 overall confidence score |
| `interface_iptm` | `float` | Per-pocket interface iPTM from Boltz |
| `plddt_mean` | `float` | Mean pLDDT across pocket |
| `n_ligand_atoms` | `int` | Heavy atom count |
| `confidence_json_path` | `str` | Path to raw Boltz confidence JSON |
| `validation_status` | `ValidationStatus` | SUCCESS/FAILED |

## Multi-Pocket: How It Works

The multi-pocket pipeline is fundamentally different from the rescoring workflows above:

1. **Full Boltz-2 prediction** — `boltz predict` is called via subprocess with the receptor sequence and N copies of the ligand specified by identical SMILES but distinct chain IDs. The diffusion module places all N copies simultaneously, allowing the model to find N distinct plausible binding sites (pockets).
2. **Pocket extraction** — `gemmi` reads the predicted PDB and writes one new PDB per ligand chain (`pocket_00_B.pdb`, `pocket_01_C.pdb`, …), each containing the full protein and a single ligand copy.
3. **Affinity rescoring** — `AffinityRescorer` (trunk + affinity head, no diffusion) scores each pocket PDB with the ligand SMILES provided. The affinity model is loaded once and reused for all N pockets.
4. **Confidence annotation** — the `pair_chains_iptm` value for the `[protein, ligand_chain]` pair is extracted from the Boltz confidence JSON and recorded as `interface_iptm`, providing a structure-quality metric independent of the affinity score.
5. **Ranked outputs** — results are sorted by `--sort-by` and written as:
   - `multipocket_scores.csv` — one row per pocket, all metrics
   - `multipocket_report.html` — self-contained ranked table with links to structure files
   - `structures/pocket_NN_X.pdb` — one extracted complex per pocket

> **Note on pocket diversity**: N simultaneous copies encourage the model to explore different sites. However, for highly symmetric or single-site binders, multiple copies may converge to the same pocket. Visual inspection of the extracted PDB files is recommended.

> **Chain ID limit**: A maximum of 25 ligand copies is supported (single-character chain IDs `A`–`Z`, minus the protein chain).

## Troubleshooting

**Checkpoint download fails**: Set `BOLTZ_RESCORE_CHECKPOINT=/path/to/boltz2_aff.ckpt` or download manually from HuggingFace.

**GPU out of memory**: Use `--device cpu` or reduce complex size. The affinity cropper limits to 256 tokens / 2048 atoms. For multi-pocket with large receptors, start with `--n-pockets 3` before scaling up.

**MOL2 parsing errors**: Ensure MOL2 follows standard Tripos format with `@<TRIPOS>MOLECULE`, `@<TRIPOS>ATOM`, and `@<TRIPOS>BOND` sections.

**Empty results**: Check validation report with `--dry-run`. Common issues: missing HETATM records, single-chain structures, non-standard residue names.

**Multi-pocket: `boltz predict` fails**: Check `boltz_prediction/` for Boltz logs. Common issues: invalid SMILES, sequence too long, missing MSA (use `--use-msa-server`), or GPU out of memory.

**Multi-pocket: prediction directory not found**: Boltz output directory naming follows `boltz_results_{target_id}/predictions/{target_id}/`. If the receptor filename contains special characters, these are sanitized to `[A-Za-z0-9-_]`. Check `--output-dir/boltz_prediction/` for the actual directory created.
