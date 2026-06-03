<div align="center">
  <div>&nbsp;</div>
  <img src="docs/boltz2_title.png" width="300"/>
  <img src="https://model-gateway.boltz.bio/a.png?x-pxid=bce1627f-f326-4bff-8a97-45c6c3bc929d" />

[Boltz-1](https://doi.org/10.1101/2024.11.19.624167) | [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) |
[Slack](https://boltz.bio/join-slack) <br> <br>
</div>



![](docs/boltz1_pred_figure.png)


## Introduction

Boltz is a family of models for biomolecular interaction prediction. Boltz-1 was the first fully open source model to approach AlphaFold3 accuracy. Our latest work Boltz-2 is a new biomolecular foundation model that goes beyond AlphaFold3 and Boltz-1 by jointly modeling complex structures and binding affinities, a critical component towards accurate molecular design. Boltz-2 is the first deep learning model to approach the accuracy of physics-based free-energy perturbation (FEP) methods, while running 1000x faster — making accurate in silico screening practical for early-stage drug discovery.

All the code and weights are provided under MIT license, making them freely available for both academic and commercial uses. For more information about the model, see the [Boltz-1](https://doi.org/10.1101/2024.11.19.624167) and [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) technical reports. To discuss updates, tools and applications join our [Slack channel](https://boltz.bio/join-slack).

## Installation & Getting Started

### Prerequisites

- Python 3.10–3.12
- CUDA-capable GPU recommended (CPU works but is significantly slower)
- [conda](https://docs.conda.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html) for environment isolation

### 1. Clone the repository

```bash
git clone https://github.com/jilimcaoco/Boltz_affinity.git
cd Boltz_affinity
```

### 2. Create a fresh Python environment

```bash
# conda (recommended)
conda create -n boltz python=3.11 -y
conda activate boltz

# or venv
python -m venv .venv && source .venv/bin/activate
```

### 3. Install the package

**With CUDA support (recommended for GPU users):**

```bash
pip install -e ".[cuda]"
```

**CPU-only or non-CUDA GPU (e.g. Apple Silicon):**

```bash
pip install -e .
```

### 4. Verify the installation

```bash
boltz --help        # should print the CLI help
boltz rescore --help
```

### 5. Download model weights

Weights are downloaded automatically on first use to `~/.boltz/`. To trigger the download explicitly:

```bash
# Structure prediction checkpoint (~3 GB)
boltz predict --help   # triggers lazy download check

# Affinity rescoring checkpoint (~1.5 GB) — downloaded on first rescore call
boltz rescore pdb --help
```

To use a custom cache directory, set `BOLTZ_CACHE_DIR` before running any command.

### 6. Run your first prediction

> **MSA policy.** `--use_msa_server` is **disabled** in this fork because
> querying the public ColabFold MMseqs2 endpoint per ligand/receptor pair
> kills jobs at scale. Pre-compute one `.a3m` per unique sequence with
> `python -m boltz.affinity_rescoring.mmseqs2` (or
> `fineturning_experiment/precompute_msas.py`) and either embed the path
> under each protein chain's `msa:` key or expose it via
> `$BOLTZ_MSA_CACHE_DIR`. See
> [docs/prediction.md#pre-computing-msas](docs/prediction.md#pre-computing-msas).

**Structure prediction from a YAML input** (with a pre-computed MSA):

```bash
export BOLTZ_MSA_CACHE_DIR=/shared/msa_cache
boltz predict examples/prot.yaml --output-dir ./output/
```

**Affinity prediction (protein + ligand SMILES):**

```bash
boltz predict examples/affinity.yaml --output-dir ./output/
```

The `examples/affinity.yaml` file looks like this — adapt it for your own system:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVG...   # your protein sequence
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
properties:
  - affinity:
      binder: B
```

**Rescore a pre-docked complex (no diffusion):**

```bash
boltz rescore pdb --input complex.pdb
```

See the [Affinity Rescoring & Multi-Pocket Pipeline](#affinity-rescoring--multi-pocket-pipeline) section below for all rescoring workflows.

---

### Installing from PyPI (upstream Boltz-2 only)

If you only need the base Boltz-2 model without the affinity rescoring extensions in this repo:

```bash
pip install boltz[cuda] -U
```

## Inference

You can run inference using Boltz with:

```
boltz predict input_path
```

`input_path` should point to a YAML file, or a directory of YAML files for batched processing, describing the biomolecules you want to model and the properties you want to predict (e.g. affinity). Each protein chain in the YAML must reference a pre-computed `.a3m` MSA (see [Pre-computing MSAs](docs/prediction.md#pre-computing-msas)); `--use_msa_server` is disabled. To see all available options: `boltz predict --help`. By default, the `boltz` command will run the latest version of the model.


### Binding Affinity Prediction
There are two main predictions in the affinity output: `affinity_pred_value` and `affinity_probability_binary`. They are trained on largely different datasets, with different supervisions, and should be used in different contexts. The `affinity_probability_binary` field should be used to detect binders from decoys, for example in a hit-discovery stage. Its value ranges from 0 to 1 and represents the predicted probability that the ligand is a binder. The `affinity_pred_value` aims to measure the specific affinity of different binders and how this changes with small modifications of the molecule. This should be used in ligand optimization stages such as hit-to-lead and lead-optimization. It reports a binding affinity value as `log10(IC50)`, derived from an `IC50` measured in `μM`. More details on how to run affinity predictions and parse the output can be found in our [prediction instructions](docs/prediction.md).

## Authentication to MSA Server

When using the `--use_msa_server` option with a server that requires authentication, you can provide credentials in one of two ways. More information is available in our [prediction instructions](docs/prediction.md).

## Affinity Rescoring & Multi-Pocket Pipeline

This repository includes an extended affinity rescoring module (`boltz rescore`) that adds four extra workflows on top of the standard `boltz predict` pipeline — all affinity-only, no diffusion is re-run except in `multipocket`:

| Command | Description |
|---------|-------------|
| `boltz rescore pdb` | Score a single pre-existing PDB/CIF complex (no diffusion) |
| `boltz rescore batch` | Score all PDB/CIF files in a directory |
| `boltz rescore receptor` | Virtual screening: score many MOL2 ligand poses against one receptor |
| `boltz rescore multipocket` | **Alternative binding mode discovery**: run a full Boltz-2 prediction with N simultaneous ligand copies, extract each binding pocket, and score with the affinity head |

The first run will auto-download the `boltz2_aff.ckpt` checkpoint (~1.5 GB) to `~/.boltz/`.

### Single Complex (`boltz rescore pdb`)

Score a pre-docked protein-ligand complex directly from a PDB or CIF file.

```bash
# Auto-detect protein/ligand chains and print scores
boltz rescore pdb --input complex.pdb

# Specify chains explicitly and write JSON output
boltz rescore pdb --input complex.pdb \
  --protein-chain A --ligand-chains B \
  --output result.json --output-format json

# Provide SMILES manually if auto-inference fails
boltz rescore pdb --input complex.pdb \
  --ligand-smiles '{"B": "CCO"}' \
  --output result.json

# Provide the full biological sequence (for structures with missing loops)
boltz rescore pdb --input complex.pdb \
  --reference-sequence '{"A": "MKTLLILTLVVA...FULL_SEQ_HERE"}'

# Dry run — validate inputs without loading the model
boltz rescore pdb --input complex.pdb --dry-run
```

### Batch Scoring (`boltz rescore batch`)

Score every PDB/CIF file in a directory in one pass.

```bash
# Score all structures in a directory
boltz rescore batch --input-dir ./structures/ --output batch_results.csv

# Recursive scan with Excel output
boltz rescore batch --input-dir ./structures/ \
  --recursive --output results.xlsx --output-format excel
```

### Virtual Screening (`boltz rescore receptor`)

Score many docked ligand poses from a MOL2 file against a single receptor structure.

```bash
# Score all poses in a MOL2 file against one receptor
boltz rescore receptor \
  --receptor receptor.pdb \
  --ligands docked_poses.mol2 \
  --output scores.csv

# Sort results by score, specify protein chain explicitly
boltz rescore receptor \
  --receptor receptor.pdb \
  --protein-chain A \
  --ligands library.mol2 \
  --output screening_results.csv \
  --sort-by affinity_score
```

### Manifest Mode (`boltz rescore manifest`)

Score complexes listed in a YAML manifest file.

```bash
boltz rescore manifest --manifest complexes.yaml --output-dir ./scores/
```

### Multi-Pocket — Alternative Binding Mode Discovery (`boltz rescore multipocket`)

Run a full Boltz-2 structure prediction with N simultaneous ligand copies, extract each predicted binding pocket, and score each with the affinity head.

```bash
# Predict 5 alternative binding poses and score each (from PDB)
boltz rescore multipocket \
  --receptor receptor.pdb \
  --ligand-smiles 'CC1=CC=CC=C1' \
  --n-pockets 5 \
  --output-dir ./mp_results/

# Fold from sequence only (no structure required)
boltz rescore multipocket \
  --receptor-sequence 'MKTLLILTLVVVTIVCLDLGYT...' \
  --ligand-smiles 'CC1=CC=CC=C1' \
  --n-pockets 5 \
  --output-dir ./mp_results/ \
  --use-msa-server

# Rank by binding probability; use pre-downloaded checkpoints
boltz rescore multipocket \
  --receptor receptor.pdb \
  --ligand-smiles 'CC1=CC=CC=C1' \
  --n-pockets 8 \
  --output-dir ./mp_results/ \
  --checkpoint /path/to/boltz2_conf.ckpt \
  --affinity-checkpoint /path/to/boltz2_aff.ckpt \
  --sort-by affinity_probability_binary
```

Output:
- `mp_results/structures/pocket_NN_X.pdb` — one extracted complex per pocket
- `mp_results/multipocket_scores.csv` — per-pocket affinity scores and confidence metrics
- `mp_results/multipocket_report.html` — self-contained ranked HTML report

### Common Options

All `boltz rescore` subcommands accept these flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `auto` | `cpu`, `cuda`, `mps`, or `auto` |
| `--validation` | `moderate` | `strict`, `moderate`, or `lenient` |
| `--checkpoint` | auto-download | Path to a custom `.ckpt` file |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |

### Python API

```python
from boltz.affinity_rescoring import AffinityRescorer

rescorer = AffinityRescorer(device="auto")

# Single complex
result = rescorer.rescore_pdb("complex.pdb")
print(f"Predicted pKd: {result.affinity_pred:.2f}")
print(f"Binding probability: {result.affinity_probability_binary:.2f}")

# Batch
results = rescorer.rescore_directory("structures/", recursive=True)
rescorer.export_results(results, "output.csv", format="csv")

# Virtual screening
scores = rescorer.rescore_receptor(
    receptor_path="receptor.pdb",
    ligands_path="ligands.mol2",
    output_path="scores.csv",
    sort_by="affinity_score",
)

# Multi-pocket
from boltz.affinity_rescoring import MultiPocketPipeline

pipeline = MultiPocketPipeline(affinity_checkpoint="auto", device="auto")
report = pipeline.run(
    receptor="receptor.pdb",
    ligand_smiles="CC1=CC=CC=C1",
    n_pockets=5,
    output_dir="./mp_results/",
    sort_by="affinity_pred",
)
for p in report.pockets:
    print(
        f"pocket {p.pocket_id}: pKd={p.affinity_pred:.2f}, "
        f"P(bind)={p.affinity_probability_binary:.2f}"
    )
```

### Configuration

Persistent settings can be placed in `~/.boltz/rescore_config.yaml` or passed via `--config`:

```yaml
model:
  device: auto
  checkpoint: auto

inference:
  recycling_steps: 5
  affinity_mw_correction: true

validation:
  level: moderate   # strict | moderate | lenient

output:
  format: csv
  include_metadata: true
```

Environment variable overrides: `BOLTZ_RESCORE_CHECKPOINT`, `BOLTZ_RESCORE_DEVICE`, `BOLTZ_RESCORE_VALIDATION`, `BOLTZ_RESCORE_OUTPUT_FORMAT`.

### Output Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| JSON | `.json` | Full structured output with metadata |
| JSONL | `.jsonl` | Streaming / line-by-line processing |
| CSV | `.csv` | Spreadsheet analysis |
| Parquet | `.parquet` | Large-scale data analysis |
| SQLite | `.db` | Queryable database |
| Excel | `.xlsx` | Reports with multiple sheets |

## Evaluation

⚠️ **Coming soon: updated evaluation code for Boltz-2!**

To encourage reproducibility and facilitate comparison with other models, on top of the existing Boltz-1 evaluation pipeline, we will soon provide the evaluation scripts and structural predictions for Boltz-2, Boltz-1, Chai-1 and AlphaFold3 on our test benchmark dataset, and our affinity predictions on the FEP+ benchmark, CASP16 and our MF-PCBA test set.

![Affinity test sets evaluations](docs/pearson_plot.png)
![Test set evaluations](docs/plot_test_boltz2.png)


## Training

⚠️ **Coming soon: updated training code for Boltz-2!**

If you're interested in retraining the model, currently for Boltz-1 but soon for Boltz-2, see our [training instructions](docs/training.md).

For parameter-efficient finetuning of the **affinity** stack on your own
labelled data (LoRA adapters, active-learning loops, custom losses), see
the [LoRA user guide](docs/lora_userguide.md).


## Contributing

We welcome external contributions and are eager to engage with the community. Connect with us on our [Slack channel](https://boltz.bio/join-slack) to discuss advancements, share insights, and foster collaboration around Boltz-2.

On recent NVIDIA GPUs, Boltz leverages the acceleration provided by [NVIDIA  cuEquivariance](https://developer.nvidia.com/cuequivariance) kernels. Boltz also runs on Tenstorrent hardware thanks to a [fork](https://github.com/moritztng/tt-boltz) by Moritz Thüning.

## License

Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.


## Cite

If you use this code or the models in your research, please cite the following papers:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

In addition if you use the automatic MSA generation, please cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```
