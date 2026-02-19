# Production-Ready Affinity Rescoring Implementation Specification

## EXECUTIVE SUMMARY

Implement a protein-ligand affinity rescoring system that leverages the Boltz-2 affinity module. The system should prioritize robustness, correctness, and the ability to benchmark the impact of removing the diffusion step from the pipeline.

**Input Formats**: PDB, CIF, mmCIF, PDBx, MOL2  
**Output Formats**: JSON, CSV, SQLite, parquet, Excel  
**Supported Scenarios**: Single complex, batch processing, directory scanning, receptor-based rescoring

---

## ARCHITECTURAL REQUIREMENTS

### 1. Core System Design

Implement a layered architecture with clear separation of concerns:

```
User Interface Layer (CLI/API)
    ↓
Input Validation & Normalization Layer
    ↓
Structure Parsing & Curation Layer
    ↓
Featurization & Preprocessing Layer
    ↓
Inference Engine Layer (Affinity Module)
    ↓
Results Aggregation & Output Layer
```

Key principles:
- **Immutability**: Do not modify input structures
- **Fail-safe defaults**: Reasonable fallbacks for all optional parameters
- **Explicit error messages**: Every failure point provides actionable feedback
- **Atomic operations**: Each complex processing is independent; batch failures don't cascade
- **Composability**: Each layer should be independently testable and swappable

### 2. Dependency Management

**Required packages** (specify exact versions):
- `torch>=2.0.0` (with CUDA support detection)
- `biopython>=1.81` (structure parsing)
- `numpy>=1.24.0`
- `click>=8.1.0` (CLI)
- `pydantic>=2.0.0` (validation)
- `polars>=0.19.0` (fast dataframe operations)
- `tqdm>=4.65.0` (progress bars)
- `pyyaml>=6.0` (config)

Create `requirements_affinity.txt` with pinned versions and optional extras:
```
torch>=2.0.0; platform_system != 'Darwin' or python_version < '3.11'
torch>=2.0.0; platform_system == 'Darwin' and python_version >= '3.11'
# GPU support variants...
```

---

## INPUT VALIDATION & PREPROCESSING

### 1. Structure File Validation

Implement comprehensive validation pipeline with detailed error recovery:

```python
class StructureValidator:
    """
    Validates and normalizes PDB/CIF files with recovery strategies.
    
    Validation checklist:
    - File existence and readability
    - Format correctness (magic numbers, headers)
    - Atom record integrity (coordinates, element, occupancy)
    - Chain continuity and gaps
    - Non-standard residues and ligands
    - Crystallographic metadata (resolution, R-free, B-factors)
    - Missing atoms/density indicators
    """
    
    # Validation levels
    STRICT = 1      # Reject any deviations
    MODERATE = 2    # Accept common artifacts, warn on issues
    LENIENT = 3     # Accept anything parseable with minimal warnings
    
    # Auto-recovery strategies
    STRATEGIES = {
        'missing_chain_info': 'infer_from_context',
        'alt_locations': 'use_highest_occupancy',
        'missing_atoms': 'skip_residue',
        'non_standard_residues': 'treat_as_hetatm',
        'coordinate_outliers': 'flag_for_review',
    }
```

**Specific validations**:
- ✓ Coordinate ranges (should be reasonable in Angstroms, typically -500 to +500)
- ✓ Occupancy values (0.0-1.0 range, warn if all non-1.0)
- ✓ B-factor ranges (typically 0-100, flag extreme values >200)
- ✓ Element symbols (must be in periodic table)
- ✓ Residue sequence continuity (flag gaps >5 residues)
- ✓ Steric clashes (detect >100 atoms within 1Å of each other)
- ✓ Chain breaks (detect backbone distance outliers)
- ✓ Heteroatom identity (distinguish ligands from water/ions)

### 2. Protein-Ligand Chain Identification

Auto-detect chains with explicit override capability:

```python
class ChainIdentifier:
    """
    Intelligently identifies protein vs. ligand chains with validation.
    
    Detection heuristics (in priority order):
    1. User-provided explicit mapping
    2. Chain name conventions (A-Z for proteins, L/X/Y for ligands)
    3. Residue type analysis (>90% standard amino acids = protein)
    4. Molecular weight/atom count
    5. Secondary structure prediction (alpha/beta = protein)
    6. Connectivity analysis (bipartite graph properties)
    """
    
    def identify_chains(
        self,
        structure,
        protein_chains: Optional[list] = None,
        ligand_chains: Optional[list] = None,
        auto_detect: bool = True,
        confidence_threshold: float = 0.95,
    ) -> ChainAssignment:
        """
        Returns assignment with confidence scores and reasoning.
        
        Raises InvalidChainAssignmentError if confidence<threshold
        and no user override provided.
        """
```

**Edge cases**:
- Covalently bound ligands (peptidomimetics)
- Multi-protein complexes (which is primary target?)
- Peptide ligands vs. protein chains
- Metal coordination spheres
- Non-standard ligands (RNA, DNA, polysaccharides)

### 3. Receptor-Based Rescoring (Single Receptor, Multiple Ligands)

For virtual screening workflows: one protein receptor scored against many ligands in `.mol2` format.

```python
class ReceptorRescorer:
    """
    Efficiently scores a fixed receptor against multiple ligands.
    
    Usage pattern:
    boltz rescore receptor --receptor protein.pdb --ligands screen.mol2 --output scores.csv
    
    Workflow:
    1. Load & validate receptor (once)
    2. Extract ligand structures from MOL2 (preserves names)
    3. For each ligand:
       - Validate structure (same rules as regular rescoring)
       - Compute features against fixed receptor
       - Run inference
       - Collect result with ligand name
    4. Output as named sheet (CSV/Excel/Parquet)
    """
    
    def rescore_batch(
        self,
        receptor_path: str,              # PDB/CIF of protein
        ligands_path: str,               # MOL2 with multiple molecules
        protein_chain: Optional[str] = None,
        output_path: str = "scores.csv",
        output_format: str = "csv",      # csv, excel, parquet
        validation_level: str = "moderate",
        include_pose_geometry: bool = True,
    ) -> ResultsDataFrame:
        """
        Process multiple ligands against single receptor.
        
        Returns structured results with ligand names as index.
        """
```

**MOL2 File Format Parsing**:

Standard MOL2 files can contain multiple `@<TRIPOS>MOLECULE` records. Each molecule includes:
- `name` field (used as ligand identifier in output)
- `atom` coordinates (already in receptor coordinate frame)
- Optional comment field (preserved in output metadata)

```python
class MOL2Parser:
    """
    Extracts individual ligand structures from multi-molecule MOL2 files.
    
    MOL2 structure:
    @<TRIPOS>MOLECULE
    ligand_name
    n_atoms n_bonds ...
    SMALL
    ...
    @<TRIPOS>ATOM
    atom_id atom_name x y z atom_type charge
    ...
    @<TRIPOS>BOND
    ...
    
    Handles:
    - Multiple molecules in single file
    - Preserves ligand names (crucial for output)
    - Validates atom coordinates and connectivity
    - Detects partial charges (used for better scoring)
    - Handles common artifact molecules (solvent, salt removed)
    """
    
    def extract_ligands_with_names(
        self,
        mol2_path: str,
        remove_water: bool = True,
        remove_ions: bool = True,
    ) -> List[Tuple[str, Structure]]:  # [(ligand_name, structure), ...]
        """
        Extract all ligands preserving their names.
        Returns list of (name, structure) tuples.
        """
```

### 4. Structure Curation & Normalization

Apply standardization transformations:

```python
class StructureNormalizer:
    """
    Standardizes structures for consistent model input.
    
    Operations:
    - Center coordinates (subtract centroid)
    - Remove water and salt molecules
    - Add missing hydrogens (or flag for user decision)
    - Handle alternate conformations (choose maximum occupancy)
    - Normalize non-standard amino acids to closest standard
    - Apply symmetry operations if needed
    - Handle disulfide bonds correctly
    - Identify and extract cofactors/prosthetic groups
    """
    
    def normalize(
        self,
        structure,
        remove_waters: bool = True,
        remove_ions: bool = True,
        add_hydrogens: bool = False,  # Expensive; off by default
        center_coordinates: bool = True,
        normalize_residues: bool = True,
        max_coordinate_range: float = 500.0,
    ) -> NormalizedStructure:
        """
        Returns normalized structure with metadata on all transformations.
        
        Returns detailed transformation log for reproducibility.
        """
```

---

## FEATURIZATION & MODEL INPUT PREPARATION

### 1. Efficient Feature Extraction

Implement cached, vectorized featurization:

```python
class AffinityFeaturizer:
    """
    Extracts features for affinity module with caching and batching.
    
    Pipeline:
    1. Distance matrix computation (vectorized einsum)
    2. Pairwise distance binning (using same bins as training)
    3. Pair feature construction (attention masks, relative features)
    4. Token embedding extraction (from encoder cache or fresh)
    5. Atom-to-token mapping (for readout)
    6. Validation against training distribution
    """
    
    def __init__(self, model_config):
        # Pre-compute binning boundaries, normalization constants, etc.
        self.distance_bins = torch.linspace(2.0, 22.0, 64)
        self.atom_type_encoding = {...}  # From training
        
    def featurize(
        self,
        protein_coords: np.ndarray,     # (n_prot_atoms, 3)
        ligand_coords: np.ndarray,      # (n_lig_atoms, 3)
        protein_atoms: List[AtomInfo],
        ligand_atoms: List[AtomInfo],
        protein_sequence: str,          # For embeddings
    ) -> FeatureDict:
        """
        Compute all required features for affinity prediction.
        """
        # Compute embeddings
        embeddings = self._compute_embeddings(...)
```

### 2. Feature Validation & Distribution Checking

Ensure features match training distribution:

```python
class FeatureValidator:
    """
    Validates that computed features are within expected ranges.
    
    Checks:
    - Distance matrix statistics (mean, std, min, max)
    - Feature dimensionality
    - NaN/Inf detection
    - Distribution shift detection (KL divergence from training)
    - Outlier detection (> 5 sigma from training distribution)
    """
    
    def validate(self, features: FeatureDict) -> ValidationReport:
        """
        Returns detailed report with warnings/errors.
        
        Warning if:
        - Complex significantly larger/smaller than training set
        - Unusual element compositions
        - Extreme distance distributions
        
        Error if:
        - NaN/Inf detected
        - Feature dimensions mismatch
        - Critical metadata missing
        """
```

---

## INFERENCE ENGINE

### 1. Model Loading & State Management

Robust checkpoint handling:

```python
class AffinityModelManager:
    """
    Manages model lifecycle with caching and device management.
    
    Responsibilities:
    - Download/cache checkpoints
    - Validate checkpoint integrity (SHA256)
    - Load model on appropriate device
    - Handle device fallback (GPU → CPU)
    - Track inference statistics
    """
    
    def __init__(self):
        self.device = self._auto_detect_device()
        self.checkpoint_cache = {}
        self.model_cache = {}
        
    def _auto_detect_device(self) -> str:
        """
        Smart device detection with user override capability.
        
        Priority:
        1. User-specified device
        2. CUDA if available and memory > 4GB
        3. MPS (Metal Performance Shaders) on Mac
        4. CPU fallback
        """
        
    def load_model(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> AffinityModule:
        """
        Load model on appropriate device.
        
        Applies:
        - Model evaluation mode (no dropout/batch norm updates)
        """
```

### 2. Batch Inference with Memory Management

efficient GPU memory handling:

```python
class AffinityInferenceEngine:
    """
    Performs inference with memory-aware batching and error recovery.
    
    Handles:
    - Inference on different devices
    - Batch-wise processing of complexes
    - Inference time tracking per complex
    """
    
    def infer_batch(
        self,
        features_list: List[FeatureDict],
        batch_size: int = 1,
    ) -> List[AffinityResult]:
        """
        Infer on batch of complexes.
        
        Returns list of results with inference metadata:
        - affinity_pred: float
        - affinity_std: float
        - inference_time_ms: float
        """
        results = []
        for features in features_list:
            result = self._infer_single(features)
            results.append(result)
        return results
```

---

## RESULTS AGGREGATION & OUTPUT

### 1. Flexible Output Formats

Support multiple output formats with schema validation:

```python
class ResultsExporter:
    """
    Exports results in multiple formats with consistent schema.
    
    Formats supported:
    - JSON (human-readable, nested)
    - JSONL (streaming, one result per line)
    - CSV (tabular, spreadsheet-compatible)
    - Parquet (columnar, efficient storage)
    - SQLite (queryable, with indexes)
    - HDF5 (scientific, with groups)
    """
    
    def export(
        self,
        results: List[AffinityResult],
        output_path: str,
        format: str = "json",
        include_metadata: bool = True,
        include_diagnostics: bool = True,
        compress: bool = True,
        validation_level: str = "strict",
    ) -> ExportReport:
        """
        Export with schema validation and metadata.
        
        Output schema:
        {
          "version": "2.0",
          "timestamp": "2026-02-18T...",
          "model_checkpoint": "...",
          "affinity_rescoring_version": "1.0",
          "results": [
            {
              "id": "complex_001",
              "pdb_id": "XXXX",
              "protein_chain": "A",
              "ligand_chains": ["B"],
              "affinity_pred": -7.2,
              "affinity_std": 0.3,
              "affinity_percentiles": {
                "p25": -7.5, "p50": -7.2, "p75": -6.9, "p95": -6.5
              },
              "num_protein_atoms": 1245,
              "num_ligand_atoms": 42,
              "inference_time_ms": 3.2,
              "model_confidence": 0.87,
              "warnings": [],
              "metadata": {
                "complex_size_percentile": 0.65,
                "ligand_complexity": "moderate",
                "protein_secondary_structure": {...}
              }
            }
          ],
          "summary": {
            "total_processed": 100,
            "successful": 98,
            "failed": 2,
            "mean_affinity": -7.15,
            "std_affinity": 0.82,
            "inference_time_total_s": 245.3,
            "throughput_complexes_per_second": 0.41
          }
        }
        """
```

### 2. Metadata & Interpretability

Capture rich metadata for each prediction:

```python
@dataclass
class AffinityResult:
    """Complete result with metadata."""
    
    # Core predictions
    id: str
    affinity_pred: float
    affinity_std: float
    
    # Complex metadata
    protein_residue_count: int
    ligand_atom_count: int
    
    # Processing metadata
    processing_time_ms: float
    featurization_time_ms: float
    inference_time_ms: float
    
    # Warnings/flags
    warnings: List[str]
    
    # Reproducibility
    model_checkpoint_sha256: str
    random_seed: int
    feature_hash: str  # For validation
```

---

## USER INTERFACE

### 1. CLI Command Design

Comprehensive CLI with sensible defaults:

```bash
# Single complex
boltz rescore pdb --input complex.pdb --output result.json

# Batch from directory
boltz rescore batch --input_dir ./pdbs --output results.csv --recursive

# From manifest
boltz rescore manifest --manifest complexes.yaml --output_dir ./scores

# Receptor-based: single receptor scored against multiple ligands
boltz rescore receptor --receptor protein.pdb --ligands molecules.mol2 --output scores.csv

# Streaming API
boltz rescore stream --input pdb_list.txt --output_dir results --workers 4

# OPTIONS (all with sensible defaults)
--protein-chain [A]                 Chain ID for protein
--ligand-chains [all_others]        Comma-separated ligand chain IDs
--output-format [csv]               json, csv, parquet, sqlite, excel
--device [auto]                     cuda, cpu, auto
--validation [moderate]             strict, moderate, lenient
--output-dir [.]                    Output directory
--log-level [INFO]                  DEBUG, INFO, WARNING, ERROR
--checkpoint [auto]                 Custom checkpoint path
--dry-run [false]                   Validate inputs without running inference
--ligand-name-field [name]          Field for ligand names in MOL2 (for receptor command)
--sheet-format                      For receptor command: format as named sheet (CSV default)
```

### 2. Receptor-Based Rescoring Usage Examples

**Virtual Screening Workflow**: Single receptor, multiple test ligands

```bash
# Basic usage - score all ligands in molecules.mol2 against receptor
boltz rescore receptor \
  --receptor kinase_1a0q.pdb \
  --ligands test_set.mol2 \
  --output kinase_scores.csv

# Specify protein chain explicitly
boltz rescore receptor \
  --receptor complex.pdb \
  --protein-chain A \
  --ligands ligands.mol2 \
  --output results.csv

# Output as Excel with sorting and filtering metadata
boltz rescore receptor \
  --receptor receptor.pdb \
  --ligands screening_library.mol2 \
  --output results.xlsx \
  --output-format excel \
  --sort-by affinity_score

# Strict validation for publication-quality results
boltz rescore receptor \
  --receptor protein.pdb \
  --ligands compounds.mol2 \
  --output high_confidence_scores.csv \
  --validation strict

# Large-scale screening with Parquet (efficient for 1M+ ligands)
boltz rescore receptor \
  --receptor active_site.pdb \
  --ligands huge_library.mol2 \
  --output library_scores.parquet \
  --output-format parquet \
  --device cuda
```

**Input File Format - MOL2 with Multiple Molecules**:

```
# molecules.mol2 (multi-record format)

@<TRIPOS>MOLECULE
compound_001
 28 28  0  0  0
SMALL
NO_CHARGES
@<TRIPOS>ATOM
   1 C1   12.450  10.200  -5.100 C.ar      1  compound_001    0.0000
   2 C2   13.100  11.300  -5.800 C.ar      1  compound_001    0.0000
   ...
@<TRIPOS>BOND
   1  1  2 ar
   ...

@<TRIPOS>MOLECULE
compound_002
 32 30  0  0  0
SMALL
NO_CHARGES
@<TRIPOS>ATOM
   1 C1   11.200  10.100  -6.500 C.ar      1  compound_002    0.0000
   2 C2   12.100  11.200  -6.900 C.ar      1  compound_002    0.0000
   ...
```

**Output File Format - CSV Sheet**:

The output is a tabular format ideal for further analysis:

```csv
ligand_name,affinity_score,affinity_uncertainty,confidence,n_atoms,validation_status,validation_issues,receptor_contacts,rotatable_bonds,processing_ms,error_message
compound_001,-8.2,0.15,0.94,28,SUCCESS,,12,5,245,
compound_002,-7.5,0.22,0.91,32,SUCCESS,Alt_location_present,14,7,267,
compound_003,-6.8,0.18,0.89,25,SUCCESS,,11,3,198,
compound_004,NaN,NaN,NaN,18,FAILED,Steric_clash_detected,,N/A,N/A,Atoms at 0.51Å distance (expected >2.0Å)
compound_005,-7.9,0.19,0.93,31,SUCCESS,,13,6,254,
```

**Accessing Results Programmatically**:

```python
# Python - load results for analysis
import pandas as pd

results = pd.read_csv("kinase_scores.csv")

# Find top 10 hits
top_hits = results.nsmallest(10, "affinity_score")
print(top_hits[["ligand_name", "affinity_score", "confidence"]])

# Filter for high-confidence results
winners = results[
    (results["validation_status"] == "SUCCESS") & 
    (results["confidence"] > 0.9)
]

# Write as Excel with formatting
winners.to_excel("winners.xlsx", sheet_name="Hits", index=False)
```

### 3. Error Handling & User Feedback

Clear, actionable error messages:

```python
class UserFeedbackEngine:
    """
    Provides clear error messages and recovery suggestions.
    
    Error categories with specific guidance:
    
    INPUT_ERRORS:
    - "File not found: {path}. Check path spelling and permissions."
    - "Invalid PDB format: {reason}. Use PDB validation tools: validate.rcsb.org"
    - "No ligand detected. Provide explicit --ligand-chains or check file."
    
    PROCESSING_ERRORS:
    - "Out of memory: Complex too large ({n_atoms} atoms). Reduce to <50k or use CPU."
    - "NaN detected in features. Chain breaks detected at residues: {list}. Repair structure."
    - "Steric clash detected. Possible coordinate errors or unfixed overlay."
    
    MODEL_ERRORS:
    - "Checkpoint not found. Downloading... (requires internet)"
    - "Model loading failed: {reason}. Verify CUDA/torch installation: pip install torch --force-reinstall"
    
    OUTPUT_ERRORS:
    - "Output directory not writable: {dir}. Check permissions: chmod u+w {dir}"
    """
    
    def provide_guidance(self, error_code: str, context: dict) -> str:
        """Returns user-friendly error message with recovery steps."""

### 3. Structured Results Output (Receptor Rescoring)

For receptor-based rescoring, output as named sheets:

```python
class ResultsExporter:
    """
    Exports results in multiple formats with ligand names as primary key.
    
    Receptor rescoring output (example):
    
    CSV Format (scores.csv):
    ligand_name,affinity_score,confidence,n_atoms,status,validation_issue
    compound_001,-8.2,0.94,32,SUCCESS,
    compound_002,-7.1,0.87,28,SUCCESS,
    compound_003,NaN,NaN,15,FAILED,Steric_clash_detected
    compound_004,-6.5,0.91,25,SUCCESS,Alt_location_used
    
    Excel Format (scores.xlsx):
    - Sheet "Results": ligand_name | affinity_score | confidence | n_atoms | status
    - Sheet "Metadata": receptor_info, validation_level, processing_date
    - Sheet "Issues": ligand_name | issue_type | severity | details
    
    Parquet Format (scores.parquet):
    - Partitioned by validation_status for easy filtering
    - Preserves all metadata as columns
    - Efficient for large-scale screening (1M+ ligands)
    """
    
    def export_receptor_results(
        self,
        results: List[LigandScore],
        output_path: str,
        format: str = "csv",  # csv, excel, parquet
        include_failed: bool = True,
        sort_by: str = "affinity_score",  # affinity_score, confidence, ligand_name
        descending: bool = False,
    ) -> None:
        """
        Export results as structured table with ligand names prominent.
        
        Automatically handles:
        - Sorting (by score, confidence, or name)
        - Failed ligands (separate sheet or marked with status)
        - Formatting (colors in Excel for easy visualization)
        - Metadata preservation (receptor info, validation settings)
        """
```

**Output Schema for Receptor Command**:

```
CSV Output (scores.csv):
ligand_name,affinity_score,confidence,n_atoms,validation_status,issues,contacts,rotatable_bonds,ms_per_ligand

compound_001,-8.2,0.94,32,SUCCESS,,12,5,245
compound_002,-7.1,0.87,28,SUCCESS,Alt_location_used,14,3,198
compound_003,NaN,NaN,15,FAILED,Steric_clash_detected,,N/A,N/A
compound_004,-6.5,0.91,25,SUCCESS,,11,7,267

Important columns for virtual screening:
- ligand_name: Identifies compound in your library
- affinity_score: Predicted binding affinity (lower = better)
- confidence: Model confidence in prediction (0-1)
- validation_status: SUCCESS, WARNING, or FAILED
- issues: Any problems detected during processing
- contacts: Number of receptor residues within 5Å (proximate to ligand)
```

---

## TESTING & VALIDATION

### 1. Comprehensive Testing Strategy

```python
class AffinityRescoreTestSuite:
    """
    Multi-level testing for production readiness.
    
    UNIT TESTS:
    - ✓ Validator handles all edge cases
    - ✓ Featurizer produces expected shapes
    - ✓ Distance computation matches reference
    - ✓ Model loading succeeds across devices
    
    INTEGRATION TESTS:
    - ✓ End-to-end: PDB → affinity predictions
    - ✓ Batch processing: 100 complexes complete
    - ✓ Format conversion: JSON ↔ CSV ↔ Parquet
    - ✓ Device fallback: GPU → CPU works
    
    REGRESSION TESTS:
    - ✓ Reference complexes produce consistent scores (±0.01 kcal/mol)
    - ✓ Benchmark complexes meet latency targets
    - ✓ Outputs match golden standard results
    
    STRESS TESTS:
    - ✓ Memory leak detection during long runs
    - ✓ OOM recovery with 1000+ large complexes
    - ✓ Concurrent requests handling
    - ✓ Graceful shutdown on interrupt
    """
    
    def test_suite(self):
        """Run all tests with detailed reporting."""
```

### 2. Benchmarking Suite

Track performance over time:

```
BENCHMARK RESULTS
Model: boltz2_aff.ckpt
Date: 2026-02-18
Hardware: A100 GPU / 128GB RAM

Complex Size    | Num Atoms | Time (ms) | Throughput (Hz) | Memory (MB)
Small           | < 2k      | 45        | 22.2            | 125
Medium          | 2k-8k     | 125       | 8.0             | 245
Large           | 8k-20k    | 320       | 3.1             | 512
XLarge          | 20k+      | 1200      | 0.83            | 2048

Mean throughput: 5.2 complexes/sec
Batch efficiency (8x): 1.85x speedup
CPU mode: 12.3x slower than GPU
```

---

## DOCUMENTATION & EXAMPLES

### 1. Comprehensive README

Include:
- Installation (with GPU/CPU variants)
- Quick start (3-line example)
- Full API documentation
- Performance benchmarks
- Troubleshooting guide
- Citing/attribution

### 2. Usage Examples

```python
# Example 1: Single complex
from boltz.affinity_rescoring import AffinityRescorer

rescorer = AffinityRescorer("checkpoint.ckpt")
result = rescorer.rescore_pdb("complex.pdb")
print(f"Affinity: {result.affinity_pred:.2f} ± {result.affinity_std:.2f} kcal/mol")

# Example 2: Batch with customization
results = rescorer.rescore_batch(
    pdb_files=glob("complexes/*.pdb"),
    use_ensemble=True,
    num_seeds=5,
)

# Example 3: Export to multiple formats
rescorer.results.export("results.json", format="json")
rescorer.results.export("results.csv", format="csv")
rescorer.results.export("results.parquet", format="parquet")

# Example 4: Custom validation
results = rescorer.rescore_batch(
    pdb_files=pdb_list,
    validation_level="lenient",  # Accept imperfect structures
)
```

---

## DEPLOYMENT CONSIDERATIONS

### 1. API Server Option

Implement optional FastAPI wrapper:

```python
# endpoints/rescore.py
@app.post("/rescore")
async def rescore_endpoint(pdb_data: bytes):
    """Stream endpoint for single complex."""
    result = rescorer.rescore_structure(parse_pdb(pdb_data))
    return result

@app.post("/rescore_batch")
async def batch_rescore(pdb_list: List[bytes]):
    """Batch endpoint with job tracking."""
    job_id = submit_job(pdb_list)
    return {"job_id": job_id, "status": "queued"}
```

### 2. Configuration Management

Support multiple deployment modes:

```yaml
# config.yaml
mode: development  # or testing, production
model:
  checkpoint: auto  # or explicit path
  device: auto      # auto, cuda, cpu
inference:
  batch_size: 1
validation:
  level: moderate
  fail_on_warnings: false
output:
  format: json
  include_diagnostics: false
logging:
  level: INFO
  file: logs/affinity_rescore.log
```

---

## IMPLEMENTATION CHECKLIST

- [ ] Core AffinityRescorer class with all methods
- [ ] Input validation with 15+ specific checks
- [ ] Structure normalization pipeline
- [ ] Featurization for affinity prediction
- [ ] Model loading with device fallback
- [ ] Inference on affinity module only
- [ ] Results export (JSON, CSV, Parquet, SQLite)
- [ ] CLI with basic options
- [ ] Error handling with clear messages
- [ ] Basic logging
- [ ] Unit tests
- [ ] Integration tests (end-to-end workflows)
- [ ] Documentation (README, examples)
- [ ] Configuration management
- [ ] Timing/benchmarking output

---

## SUCCESS CRITERIA

✅ All input validation catches malformed files with clear errors
✅ Predictions reproducible (identical input = identical output)
✅ User receives actionable error messages
✅ GPU and CPU paths work correctly
✅ Inference completes without diffusion step
✅ Results exportable in JSON, CSV, Parquet formats
✅ Timing information captured for benchmarking
✅ Documentation enables users to run on their own data

