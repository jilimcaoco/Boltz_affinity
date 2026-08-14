# TRIAGED — Migration Handoff

**Target repo**: `github.com/maomlab/TRIAGED` (new branch)
**Source fork**: this repo (a Boltz-2 fork with two add-on modules)
**Status of this handoff**: modules in this fork have been prepared for extraction — the next agent's job is to land them in the TRIAGED repo and build the surrounding suite.

---

## 1. Product Vision

TRIAGED is a **virtual-screening suite for structural biology labs** that wraps Boltz-2, DOCK 3.8, and custom analysis tools behind a Snakemake workflow engine and an optional Streamlit GUI. It targets **single-lab, single-user** installs on a SLURM HPC cluster. The Streamlit layer exists so non-terminal-competent collaborators can submit jobs and view results without command-line knowledge.

### Design constraints
- **Single-user per install.** No auth, no multi-tenant concerns.
- **SLURM HPC** is the compute target. Snakemake profile at `config/slurm/`.
- **Vanilla `pip install boltz==2.2.1`** — TRIAGED must NOT require a boltz fork.
- **DOCK 3.8** path is configured at install time (not a pip dep; external binary).
- **SQLite + SQLAlchemy ORM** for the compound/receptor/job/result registry (chosen over PostgreSQL because zero-admin; ORM keeps the door open to PG later via connection string).

### Suite components (to build in TRIAGED)
| Package | Purpose | Source |
|---|---|---|
| `triaged_rescoring` | Boltz-2 affinity-only scoring (trunk + affinity head, no diffusion) | Move from `src/boltz/affinity_rescoring/` in this fork |
| `triaged_lora` | LoRA fine-tuning of the affinity module | Rename from `boltz_lora/` in this fork |
| `triaged_analysis` | Ligand clustering, VS post-analysis, reports | New |
| `triaged_db` | SQLAlchemy models + Alembic migrations | New |
| `triaged_app` | Streamlit GUI (7 pages: dashboard, libraries, submit, LoRA, results, settings) | New |
| `workflows/` | Snakemake rules + SLURM profile | New |

### Phase order agreed with user
1. **Decouple from fork** (this handoff's scope — mostly done)
2. **Compound/receptor DB** (SQLAlchemy + SQLite)
3. **Snakemake workflows** (msa, featurize, dock, rescore, lora_train, lora_eval, virtual_screen, cluster, report)
4. **Streamlit GUI**

---

## 2. What this fork contains

### Fork divergence vs vanilla boltz 2.2.1
A full `diff -rq` against `~/Projects/boltz` (vanilla 2.2.1) showed the fork is **cleanly additive**:
- **No changes** to `src/boltz/model/` or `src/boltz/data/` (model and data pipeline are byte-identical to upstream).
- **One file modified**: [src/boltz/main.py](src/boltz/main.py) — an 8-line defensive registration hook for the `rescore` subcommand (wrapped in `try/except ImportError`). This hook is **fork-only** and should **not** be carried into TRIAGED; the new `triaged_rescoring` package ships its own standalone CLI entry point.
- **One directory added**: [src/boltz/affinity_rescoring/](src/boltz/affinity_rescoring/) — 14 files, the affinity-rescoring module.

### The two modules to migrate
Key source files (paths are relative to this fork):

**affinity_rescoring** (`src/boltz/affinity_rescoring/`)
- [`__init__.py`](src/boltz/affinity_rescoring/__init__.py) — public API surface
- [`rescorer.py`](src/boltz/affinity_rescoring/rescorer.py) — `AffinityRescorer` high-level API
- [`inference.py`](src/boltz/affinity_rescoring/inference.py) — `AffinityModelManager`, `affinity_forward`, `run_direct_affinity_inference`
- [`featurize.py`](src/boltz/affinity_rescoring/featurize.py) — `featurize_complex`, `featurize_from_yaml`
- [`coord_injection.py`](src/boltz/affinity_rescoring/coord_injection.py) — inject pre-existing PDB coords into Boltz batch
- [`validation.py`](src/boltz/affinity_rescoring/validation.py) — `StructureValidator`, `ChainIdentifier`
- [`parsers.py`](src/boltz/affinity_rescoring/parsers.py) — PDB/CIF parsing
- [`mol2_parser.py`](src/boltz/affinity_rescoring/mol2_parser.py) — MOL2 ligand parsing
- [`smiles_inference.py`](src/boltz/affinity_rescoring/smiles_inference.py) — infer SMILES from 3D coords
- [`models.py`](src/boltz/affinity_rescoring/models.py) — `AffinityResult`, `BatchSummary`, `RescoreConfig`, `AtomInfo`, `LigandStructure`
- [`config.py`](src/boltz/affinity_rescoring/config.py) — YAML config loader
- [`export.py`](src/boltz/affinity_rescoring/export.py) — `ResultsExporter` (JSON/JSONL/CSV/Parquet/SQLite/Excel)
- [`cli.py`](src/boltz/affinity_rescoring/cli.py) — `rescore_cli` (currently registered into `boltz` CLI via main.py hook)

**boltz_lora** (`boltz_lora/`)
- See [boltz_lora/IMPLEMENTATION_REPORT.md](boltz_lora/IMPLEMENTATION_REPORT.md) for the full design doc.
- Entry points: [`cli.py`](boltz_lora/src/boltz_lora/cli.py), [`activation.py`](boltz_lora/src/boltz_lora/activation.py)
- Core math: [`lora.py`](boltz_lora/src/boltz_lora/lora.py), [`injection.py`](boltz_lora/src/boltz_lora/injection.py), [`targeting.py`](boltz_lora/src/boltz_lora/targeting.py)
- Training: [`training/`](boltz_lora/src/boltz_lora/training/)
- Registry: [`registry.py`](boltz_lora/src/boltz_lora/registry.py) (stores adapters at `$BOLTZ_LORA_HOME`, default `~/.boltz_lora`)

---

## 3. Preparation work already completed (in this fork)

These changes have been applied so the next agent can extract the modules cleanly:

### 3.1 `affinity_rescoring` converted to relative internal imports
Every `from boltz.affinity_rescoring.X import Y` inside the package was rewritten to `from .X import Y`. This means **renaming the package to `triaged_rescoring` no longer requires grepping for the old name in internal code** — just rename the directory.

Verified via AST parse + individual submodule imports (`models`, `parsers`, `mol2_parser`, `smiles_inference`, `validation`, `config`, `export` all import cleanly in Python 3.13 with torch only).

### 3.2 `boltz_lora` cross-package dependency centralised
The only cross-package import (`boltz_lora → boltz.affinity_rescoring`) was moved into a single shim module: [boltz_lora/src/boltz_lora/_rescoring_compat.py](boltz_lora/src/boltz_lora/_rescoring_compat.py). When migrating, change only that one function body from `from boltz.affinity_rescoring import featurize_complex` to `from triaged_rescoring import featurize_complex` (or whatever the final name is).

### 3.3 Verified all boltz2 internal imports resolve against vanilla 2.2.1
These are the **external-to-the-modules** boltz imports, all of which exist in vanilla boltz 2.2.1 at `~/Projects/boltz`:

| Import | File in vanilla boltz |
|---|---|
| `boltz.model.models.boltz2.Boltz2` | `src/boltz/model/models/boltz2.py:40` |
| `boltz.data.const` | `src/boltz/data/const.py` |
| `boltz.data.crop.affinity.AffinityCropper` | `src/boltz/data/crop/affinity.py:11` |
| `boltz.data.feature.featurizerv2.Boltz2Featurizer` | `src/boltz/data/feature/featurizerv2.py:2160` |
| `boltz.data.mol.load_canonicals, load_molecules` | `src/boltz/data/mol.py:42, 16` |
| `boltz.data.module.inferencev2.load_input` | `src/boltz/data/module/inferencev2.py:27` |
| `boltz.data.tokenize.boltz2.Boltz2Tokenizer` | `src/boltz/data/tokenize/boltz2.py:379` |
| `boltz.data.types.Manifest, Record, StructureV2` | `src/boltz/data/types.py:654, 573, 323` |
| `boltz.main.process_input, compute_msa` | `src/boltz/main.py:525, 415` |
| `boltz.data.parse.schema.standardize` | `src/boltz/data/parse/schema.py:1837` |

**No compatibility shim needed for boltz2 internals.** Pin the dep to `boltz>=2.2.1,<3` in each new package's `pyproject.toml`.

---

## 4. Migration playbook (next agent's job)

### Step 1 — Create the TRIAGED repo layout
```
TRIAGED/
├── pyproject.toml               # optional monorepo root
├── config/
│   ├── triaged.yaml             # paths to boltz cache, DOCK3.8 binary, SLURM partition
│   └── slurm/                   # Snakemake SLURM profile
├── packages/
│   ├── triaged_rescoring/
│   ├── triaged_lora/
│   ├── triaged_analysis/
│   ├── triaged_db/
│   └── triaged_app/
├── workflows/                   # Snakemake rules
├── tests/
└── docs/
```

### Step 2 — Move `affinity_rescoring` → `triaged_rescoring`
1. `cp -r <fork>/src/boltz/affinity_rescoring/ <triaged>/packages/triaged_rescoring/src/triaged_rescoring/`
2. Rename the one import in [boltz_lora/_rescoring_compat.py](boltz_lora/src/boltz_lora/_rescoring_compat.py) → `triaged_rescoring`.
3. Write `packages/triaged_rescoring/pyproject.toml` with:
   - `name = "triaged_rescoring"`
   - `dependencies = ["boltz>=2.2.1,<3", "click", "pyyaml", "pydantic", "rdkit>=2024.3.2", "biopython"]`
   - `[project.scripts] triaged-rescore = "triaged_rescoring.cli:rescore_cli"`
4. **Do NOT** carry over the 8-line `main.py` hook from the fork — `triaged_rescoring` ships its own CLI.
5. Run `tests/test_affinity_rescoring.py` and `tests/test_affinity_integration.py` from this fork against the new package to confirm parity.

### Step 3 — Move `boltz_lora` → `triaged_lora`
1. `cp -r <fork>/boltz_lora/src/boltz_lora/ <triaged>/packages/triaged_lora/src/triaged_lora/`
2. Rename internal self-references: `grep -r "boltz_lora" packages/triaged_lora/src/` → rewrite to `triaged_lora` (the test suite, cli entry point, and internal absolute imports all reference the package name).
3. Rewrite the one line in `_rescoring_compat.py` (see Step 2).
4. Migrate `pyproject.toml` — update package name, rename entry point `boltz-lora` → `triaged-lora`, pin `boltz>=2.2.1,<3` + `triaged_rescoring`.
5. Update `$BOLTZ_LORA_HOME` references → `$TRIAGED_LORA_HOME` (default `~/.triaged/lora`) in `registry.py`.
6. Run the 49-test suite from `boltz_lora/tests/` to confirm parity.

### Step 4 — DB package (`triaged_db`)
SQLAlchemy 2.0 declarative models, one-to-many relationships, Alembic migrations. Schema from the plan:
- `Target`: id, name, pdb_path, sequence, metadata_json
- `Compound`: id, smiles, name, mol_file_path, fingerprint_blob
- `Assay`: compound_id, target_id, affinity_value, unit, source, split_label
- `Job`: id, type (screen/rescore/lora_train), status, submitted_at, slurm_job_id, snakemake_run_id
- `Result`: job_id, compound_id, target_id, affinity_pred, affinity_std, prob_binary, output_path
- `Feature`: result_id, name, value_json

### Step 5 — Snakemake workflows
Rules: `msa`, `featurize`, `dock` (DOCK 3.8 binary), `rescore` (triaged_rescoring), `lora_train`, `lora_eval`, `virtual_screen` (dock→rescore DAG), `cluster`, `report`. SLURM profile with per-rule resources (GPU for rescore/lora_train).

### Step 6 — Streamlit app
7 pages: Dashboard, Compound Library, Receptor Library, Submit Screen, LoRA Training, Results Viewer, Settings. Job tracking polls `sacct` and reads the job table from `triaged_db`.

---

## 5. Things to watch out for

- **The `try/except ImportError` in the fork's `src/boltz/main.py` rescore hook is too broad** — it silently swallows errors inside `affinity_rescoring/__init__.py`. This is fork-only and won't be in TRIAGED, but flag it if you ever port the pattern elsewhere.
- **`coord_injection.py` imports from `boltz.data.types`** — do not move those types around; they're public boltz API in 2.2.1 and must stay as absolute imports of upstream boltz.
- **LoRA targeting uses hard-coded module path patterns** like `*affinity_module[12]*` in [boltz_lora/src/boltz_lora/targeting.py](boltz_lora/src/boltz_lora/targeting.py). These must match the vanilla boltz 2.2.1 `Boltz2` model attribute names (`affinity_module1`, `affinity_module2`) — verified present in vanilla, but re-verify if boltz bumps to 2.3+.
- **`boltz_lora/examples/infer_with_adapter.py`** has a `try: from boltz.affinity_rescoring.inference import run_direct_affinity_inference` inside a function. Update when migrating.
- **Model checkpoint SHA256 verification** in [boltz_lora/src/boltz_lora/registry.py](boltz_lora/src/boltz_lora/registry.py) hashes the base boltz checkpoint. Vanilla boltz downloads `boltz2_aff.ckpt` to `~/.boltz`. The hash check logic needs no change, but the default cache location should follow `$BOLTZ_CACHE` env var (already the case).
- **Tests require installing boltz + boltz_lora + pytest** in a venv — this fork's `.venv` currently has only `torch`. The migration agent will need to set up a proper dev environment (`pip install boltz==2.2.1 pytest && pip install -e packages/triaged_rescoring -e packages/triaged_lora`).

---

## 6. Project-specific context (carry forward)

### 6.1 Interpretability work (pre-existing in this fork)
The fork also contains a parallel interpretability effort for the affinity module. See [/memories/repo/project-goal.md](/memories/repo/project-goal.md) and the [interpretability/](interpretability/) directory. **Out of scope** for the initial TRIAGED migration, but should eventually live as a `triaged_interpretability` package.

### 6.2 Existing datasets
- `datasets/fep_benchmark_annotated.csv` + `datasets/fep_benchmark_structures.csv` — FEP benchmark complexes for validation
- `datasets/raw/` — CIF files for 8 targets (bace, cdk2, jnk1, mcl1, p38, ptp1b, thrombin, tyk2)
- Useful for integration tests in TRIAGED

### 6.3 Full plan document
The canonical multi-phase plan (DB design, workflow list, Streamlit pages, design decisions) is in the agent session memory under `/memories/session/plan.md`. Reproduce the key decisions in TRIAGED's own docs before starting.
