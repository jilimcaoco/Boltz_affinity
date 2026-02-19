# System Prompt for Production Affinity Rescoring Implementation

**Target Model**: Claude Opus 3.6 (or equivalent high-capability model)  
**Task**: Implement production-grade protein-ligand affinity rescoring system  
**Scope**: End-to-end system with CLI, validation, error handling, testing  
**Timeline**: Single implementation session  
**Quality Bar**: Production-ready, deployment-capable

---

## CONTEXT & OBJECTIVES

You are implementing a new computational system as part of the Boltz family of protein structure models. The Boltz stack includes:
- **Boltz-1/2**: Transformer-based structure predictors with diffusion refinement
- **Affinity Module**: SOTA ligand-protein binding affinity predictor (your target)
- **Confidence Module**: Prediction quality estimator

Your specific task: Extract the pre-trained affinity module from the full pipeline and adapt it to rapidly rescore existing protein-ligand complexes without expensive diffusion/transformer steps.

### Key Success Metrics
1. **Performance**: <5 sec/complex GPU, <50 sec/complex CPU
2. **Robustness**: 99%+ input validation coverage, zero data loss
3. **Usability**: Users unfamiliar with ML can get results in 5 minutes
4. **Accuracy**: Predictions match full pipeline reference within ±0.01 kcal/mol
5. **Scalability**: Batch 1000+ complexes without memory leaks

---

## CRITICAL ARCHITECTURAL DECISIONS

### 1. Modular Design Over Monolithic

**Decision**: Implement as separate, composable components rather than monolithic class.

**Rationale**: Enables independent testing, easier debugging, allows users to apply components selectively.

**Components**:
- `StructureValidator`: Input validation and error recovery
- `StructureNormalizer`: Standardization and curation
- `ChainIdentifier`: Smart protein/ligand detection
- `AffinityFeaturizer`: Feature extraction with caching
- `ModelManager`: Checkpoint loading and device management
- `InferenceEngine`: Forward passes with memory management
- `ResultsAggregator`: Output formatting and export
- `CLIBuilder`: User interface

**Implementation Note**: Each component should:
- Have clear input/output contracts
- Raise specific, recoverable exceptions
- Log operations at DEBUG level
- Support dependency injection for testing
- Include configuration dataclasses

### 2. Explicit Error Handling Over Exceptions

**Decision**: Design for graceful degradation wherever possible.

**Examples**:
- Bad coordinate detected → Mark residue, continue (not crash)
- GPU memory exceeded → Reduce batch size, retry (not OOM error)
- Missing chain metadata → Auto-detect, warn user (not fail)
- File permission denied → Clear error suggesting remediation

**Implementation**:
```python
from dataclasses import dataclass
from enum import Enum

class ErrorLevel(Enum):
    FATAL = 0      # abort processing
    SEVERE = 1     # warn user, skip this complex
    WARNING = 2    # log and continue
    INFO = 3       # informational only

@dataclass
class ProcessingResult:
    success: bool
    result: Optional[Any]
    errors: List[tuple[ErrorLevel, str]]  # level, message
    warnings: List[str]
    metadata: Dict[str, Any]  # timing, diagnostics
```

### 3. Configuration Management Over Hard-Coded Defaults

**Decision**: All system parameters externalized to config, environment variables, or CLI flags.

**Hierarchy** (highest to lowest priority):
1. CLI flag `--param-name`
2. Environment variable `BOLTZ_PARAM_NAME`
3. Config file (YAML) parameter
4. Hardcoded default in code

**Example**:
```python
from dataclasses import dataclass

@dataclass
class ValidationConfig:
    """Injected into all validators, sourced from config/CLI/env."""
    level: str = "moderate"  # strict, moderate, lenient
    fail_on_warnings: bool = False
    steric_clash_threshold: float = 1.0  # Angstroms
    coordinate_outlier_zscore: float = 5.0

# Usage in validator
def validate_coordinates(coords, config: ValidationConfig):
    if zscore(coords) > config.coordinate_outlier_zscore:
        # Handle based on config.level
```

### 4. Caching Strategy for Performance

**Decision**: Implement multi-level caching without sacrificing reproducibility.

**Cache Levels**:
1. **Session-level**: In-memory embeddings cache (cleared between runs)
2. **Checkpoint-level**: Pre-computed normalization constants (in checkpoint)
3. **User-level**: Optional persistent cache directory (~/.boltz/cache)

**Implementation**:
```python
class EmbeddingCache:
    """Thread-safe, bounded cache for sequence embeddings."""
    
    def __init__(self, max_size_mb: int = 500, ttl_minutes: int = 60):
        self.cache = {}  # {sequence_hash: embedding}
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.access_times = {}
        
    def get(self, sequence: str, model_hash: str) -> Optional[Tensor]:
        """Returns cached embedding or None. Cache hit invalidates on model change."""
        key = (hash(sequence), model_hash)
        if key in self.cache and not self._is_expired(key):
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, sequence: str, model_hash: str, embedding: Tensor):
        """Stores embedding, evicting LRU item if needed."""
        # Maintain size invariant, age off expired entries
```

**Reproducibility Guarantee**: Same input + same model = deterministic output (even with cache).

### 5. Testing Philosophy: Test Pyramid

**Decision**: Heavy investment in unit tests, many integration tests, few e2e tests.

```
        ╱╲  E2E Tests (3-5)
       ╱──╲ - Full workflow tests
      ╱────╲
    ╱──────╲ Integration Tests (15-20)
   ╱────────╲ - Component interaction
  ╱──────────╲- Mock external deps
 ╱────────────╲
Unit Tests (50+) - Individual functions, isolated
```

**Test Examples**:

```python
# UNIT: StructureValidator._validate_coordinates
def test_validate_coordinates_detects_outliers():
    coords = [[0, 0, 0], [1000, 1000, 1000]]  # Outlier
    result = validator.validate_coordinates(coords, config=ValidationConfig(level="strict"))
    assert result.success == False
    assert "coordinate range" in result.errors[0].message.lower()

# INTEGRATION: StructureNormalizer -> StructureValidator
def test_normalizer_produces_valid_output():
    input_pdb = "test_data/complex_messy.pdb"
    normalized = normalizer.normalize(parse_pdb(input_pdb))
    validation = validator.validate_structure(normalized)
    assert validation.errors == []  # Normalizer fixed all issues

# E2E: Full pipeline
def test_rescore_single_complex():
    rescorer = AffinityRescorer("checkpoint.ckpt")
    result = rescorer.rescore_pdb("test_data/reference_complex.pdb")
    assert 6.0 < result.affinity_pred < 8.0  # Known range
    assert result.affinity_std < 1.0
    assert result.inference_time_ms < 5000  # 5 second budget
```

---

## IMPLEMENTATION REQUIREMENTS

### Phase 1: Core Data Handling (Priority 1)

**Objective**: Robust input processing foundation

#### 1.1 Structure Validation
```python
class StructureValidator:
    """
    Validates PDB/CIF files with detailed error reporting.
    
    MUST check:
    - File format validity (magic bytes, header parsing)
    - Coordinate ranges (-500 to +500 Angstroms typical)
    - Element symbols against periodic table
    - Occupancy values ∈ [0,1]
    - B-factor ranges (flag if >200)
    - Residue continuity (backbone distance < 4.0 Å)
    - Atom name conventions
    - Chain IDs validity
    - Steric clashes (>100 atoms within 1 Å)
    - Non-standard residues (with recovery strategy)
    - Missing atoms/density (REMARK 465)
    - Duplicate atom records (choose by occupancy)
    
    Return: ValidationReport with:
    - success: bool
    - errors: List[Tuple[ErrorLevel, str]]
    - warnings: List[str]
    - recoverable: Dict[str, str]  # applied fixes
    - metadata: Dict  # statistics
    """
    
    def validate(self, pdb_file: Path, config: ValidationConfig) -> ValidationReport:
        """Main validation entry point."""
```

#### 1.2 Chain Intelligence
```python
class ChainIdentifier:
    """
    Identifies protein vs. ligand chains with confidence scoring.
    
    Detection hierarchy:
    1. User-provided mapping (highest priority)
    2. Heuristics (residue types, MW, connectivity)
    3. Sequence homology if supplied
    4. Secondary structure prediction
    
    Must provide:
    - Confidence scores (0-1) per assignment
    - Reasoning (which heuristic triggered)
    - Alternatives if confidence too low
    - Explicit user override capability
    """
    
    def identify(
        self,
        structure,
        protein_chains: Optional[list] = None,
        ligand_chains: Optional[list] = None,
        min_confidence: float = 0.95,
    ) -> ChainAssignmentResult:
        """
        Returns assignment with reasoning and confidence.
        Raises if confidence<min_confidence unless user override.
        """
```

#### 1.3 Structure Normalization
```python
class StructureNormalizer:
    """
    Standardizes structures for model input.
    
    Operations (in order):
    1. Parse and decompose (separate chains)
    2. Remove crystallographic artifacts (water, salts)
    3. Handle alternate conformations (take max occupancy)
    4. Center coordinates (subtract centroid)
    5. Normalize non-standard residues
    6. Detect/flag disulfide bonds
    7. Validate result
    
    Return: NormalizedStructure with:
    - protein/ligand_atoms: List[AtomRecord]
    - protein/ligand_coords: np.ndarray (n, 3)
    - transformation_log: List[str]  # reproducibility
    - metadata: Dict  # statistics
    """
    
    def normalize(
        self,
        structure,
        config: NormalizationConfig,
    ) -> NormalizedStructure:
        """Apply all standardization transformations."""
```

### Phase 2: Featurization & Model Integration (Priority 1)

**Objective**: Efficient feature extraction and model loading

#### 2.1 Affinity Featurizer
```python
class AffinityFeaturizer:
    """
    Extracts features for affinity module.
    
    Critical computations:
    - Distance matrix (n_protein × n_ligand atoms)
    - Distance binning (64 bins from 2-22 Angstroms)
    - Pairwise edge features
    - Token embeddings (from transformer encoder)
    - Atom-to-token mapping
    
    Performance target: <500ms total for typical complex
    - Distance matrix: <100ms (vectorized)
    - Embedding lookup: <50ms (cached)
    - Edge features: <200ms
    
    Must handle:
    - Variable-size complexes
    - Feature normalization matching training
    - NaN/Inf detection and reporting
    - Distribution shift detection vs. training
    """
    
    def featurize(
        self,
        normalized_structure: NormalizedStructure,
        protein_sequence: str,
        cache_embeddings: bool = True,
    ) -> FeatureDict:
        """
        Returns: {
            'distance_matrix': np.ndarray (n_prot, n_lig),
            'edge_features': Tensor (..., 64),  # binned distances
            'protein_embeddings': Tensor (..., 256),
            'ligand_embeddings': Tensor (..., 128),
            'sequence_embeddings': Tensor (n_tokens, 384),
            'atom_to_token_map_protein': np.ndarray,
            'atom_to_token_map_ligand': np.ndarray,
            'validation': FeatureValidationReport,
        }
        """
```

#### 2.2 Model Manager & Device Handling
```python
class ModelManager:
    """
    Manages model lifecycle with robust device detection.
    
    Responsibilities:
    - Auto-detect GPU/CPU/MPS availability
    - Download/cache checkpoints (with integrity checks)
    - Load model on appropriate device
    - Handle device fallback (GPU → CPU)
    - Mixed precision setup
    - Model compilation (torch.compile)
    
    Device selection algorithm:
    1. User override (--device flag)
    2. CUDA if available + memory > 4GB (check nvidia-smi)
    3. MPS on macOS if available
    4. CPU fallback
    
    Checkpoint handling:
    - Verify SHA256 signature
    - Extract model config from checkpoint
    - Handle PyTorch Lightning format
    - Version compatibility check
    """
    
    def load_model(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        compile: bool = True,
        use_amp: bool = True,
    ) -> Tuple[nn.Module, Dict]:
        """
        Load model and return (model, config)
        """
```

### Phase 3: Inference Engine (Priority 1)

**Objective**: Fast, memory-efficient prediction pipeline

#### 3.1 Batch Inference
```python
class InferenceEngine:
    """
    Performs forward passes with memory awareness.
    
    Handles:
    - Variable-size complex batching
    - OOM recovery (auto-reduce batch size)
    - Mixed precision inference (FP16)
    - Inference time tracking
    - Stochastic depth for uncertainty (optional)
    
    Performance targets:
    - Single forward: 50-200ms on GPU
    - Batch of 8: 150-400ms (2-5x parallelism)
    - Memory per complex: ~10-50MB
    
    Must track:
    - Frames per complex
    - Total inference time
    - Per-atom logits (for interpretability)
    - Ensemble variance (if enabled)
    """
    
    def infer_batch(
        self,
        features_list: List[FeatureDict],
        model: nn.Module,
        ensemble_seeds: Optional[int] = None,
        verbose: bool = False,
    ) -> List[AffinityPrediction]:
        """
        Infer batch with automatic GPU memory management.
        
        Returns: [{
            'affinity': float,
            'affinity_logits': np.ndarray,  # for calibration
            'inference_time_ms': float,
            'ensemble_mean': Optional[float],
            'ensemble_std': Optional[float],
            'percentiles': Dict[str, float],
            'warnings': List[str],
        }]
        """
```

#### 3.2 Uncertainty Quantification
```python
class UncertaintyEstimator:
    """
    Estimates prediction confidence (optional).
    
    Methods:
    1. Ensemble (3-5 stochastic forward passes)
    2. Dropout-in-inference (MC Dropout)
    3. Temperature scaling (post-hoc calibration)
    
    Output: calibrated uncertainty with:
    - Mean prediction
    - Standard deviation
    - Percentiles (25th, 50th, 75th, 95th)
    - Calibration curve (if available)
    """
```

### Phase 4: Results & Output (Priority 2)

**Objective**: Flexible export with comprehensive metadata

#### 4.1 Results Formatter
```python
class ResultsFormatter:
    """
    Formats predictions for different output formats.
    
    Supported formats:
    - JSON (nested, human-readable)
    - JSONL (streaming, one per line)
    - CSV (tabular, excel-compatible)
    - Parquet (columnar, efficient)
    - SQLite (queryable with indexes)
    - HDF5 (scientific, with groups)
    
    Must include metadata:
    - Run timestamp
    - Model checkpoint identifiers (path, SHA256)
    - Boltz version
    - Feature extraction date
    - Processing settings
    
    Schema MUST be validated before write.
    """
    
    def format_results(
        self,
        results: List[AffinityPrediction],
        output_path: Path,
        format: str = "json",
        include_diagnostics: bool = False,
        compress: bool = True,
    ) -> ExportReport:
        """Export results with validation."""
```

### Phase 5: User Interface (Priority 2)

**Objective**: Intuitive CLI and API

#### 5.1 Click CLI Commands
```bash
# Commands hierarchy
boltz rescore                    # Main command
  pdb <file>                     # Single PDB
  batch <directory>              # Directory of PDBs
  manifest <yaml>                # Manifest file
  stream <stdin>                 # Streaming input

# Key options
--protein-chain [A]              # Protein chain ID
--ligand-chains [auto]           # Ligand chain IDs
--output-format [json]           # json, csv, parquet, sqlite
--batch-size [8]                 # Adaptive batching
--device [auto]                  # cuda, cpu, auto
--use-ensemble [false]           # Uncertainty quantification
--validation [moderate]          # strict, moderate, lenient
--checkpoint [auto]              # Model path
--max-workers [4]                # Parallelization
--dry-run [false]                # Validate only
--log-level [INFO]               # DEBUG, INFO, WARNING, ERROR
```

**Implementation**:
```python
@click.group()
def rescore():
    """Rescore existing protein-ligand complexes with Boltz affinity module."""

@rescore.command()
@click.argument("pdb_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="result.json")
@click.option("--protein-chain", default="A")
@click.option("--ligand-chains", default=None)
@click.option("--device", type=click.Choice(["cuda", "cpu", "auto"]), default="auto")
@click.option("--use-ensemble", is_flag=True)
@click.option("--validation", type=click.Choice(["strict", "moderate", "lenient"]), default="moderate")
def pdb(pdb_file, output, protein_chain, ligand_chains, device, use_ensemble, validation):
    """Score single complex."""
    rescorer = create_rescorer(device=device)
    result = rescorer.rescore_pdb(pdb_file, protein_chain=protein_chain)
    export_result(result, output)
```

### Phase 6: Testing & Documentation (Priority 2)

**Objective**: Comprehensive validation and user guidance

#### 6.1 Test Structure
```
tests/
  unit/
    test_structure_validator.py      # 15+ tests
    test_chain_identifier.py         # 10+ tests
    test_normalizer.py               # 10+ tests
    test_featurizer.py               # 10+ tests
    test_inference_engine.py         # 8+ tests
  integration/
    test_end_to_end_workflow.py      # 5+ tests
    test_batch_processing.py         # 3+ tests
    test_format_conversion.py        # 4+ tests
    test_device_fallback.py          # 2+ tests
  performance/
    benchmarks.py                    # Latency, memory tracking
    regression_tests.py              # Reference results

test_data/
  reference_complex.pdb             # Known good complex
  malformed_complex.pdb             # Edge case
  large_complex.pdb                 # Stress test
```

#### 6.2 Documentation
```
docs/
  README.md                   # Installation, quick start
  API.md                      # Complete API reference
  EXAMPLES.md                 # Usage scenarios
  TROUBLESHOOTING.md          # Common issues + solutions
  PERFORMANCE.md              # Benchmarks, optimization tips
  ARCHITECTURE.md             # Design decisions
  CONTRIBUTING.md            # Development guide
```

---

## ERROR HANDLING REQUIREMENTS

Every error path MUST satisfy:
1. **Informative**: User understands what went wrong
2. **Actionable**: Clear steps to fix or work around
3. **Logged**: Full stack trace at DEBUG level
4. **Recoverable**: Fail fast but don't crash batch

**Example Error Handling**:
```python
try:
    structure = parse_pdb(pdb_file)
except FileNotFoundError:
    error_msg = f"PDB file not found: {pdb_file}\nChecked: {pdb_file.absolute()}"
    logger.error(error_msg)
    raise UserError(error_msg)

except InvalidPDBFormat as e:
    error_msg = f"Invalid PDB format at line {e.line_num}: {e.reason}\n"
    error_msg += "Suggestions:\n"
    error_msg += "- Validate with: validate.rcsb.org\n"
    error_msg += "- Use PDBFixer to repair: pip install pdbfixer\n"
    logger.error(error_msg)
    return ProcessingResult(success=False, errors=[(ErrorLevel.SEVERE, error_msg)])
```

---

## PERFORMANCE OPTIMIZATION CHECKLIST

- [ ] Distance matrix computed with vectorized operations (einsum)
- [ ] Embedding cache with LRU eviction
- [ ] Batch processing with dynamic sizing (adapt to available GPU memory)
- [ ] Mixed precision inference (FP16 on supported GPUs)
- [ ] Model compilation with torch.compile (2-3x speedup)
- [ ] Dataloader with num_workers > 0 for I/O parallelization
- [ ] Memory profiling to identify bottlenecks
- [ ] Benchmark on reference hardware (A100, RTX 4090, CPU)
- [ ] Profile memory usage during batch inference
- [ ] Profile inference latency per component

---

## CRITICAL CONSTRAINTS

1. **Reproducibility**: Same input + seed → identical output (bit-exact)
2. **Idempotency**: Running twice produces identical results (no accidental state)
3. **Atomicity**: Partial failures don't corrupt output (transactional writes)
4. **Memory Safety**: No leaks during 1000-complex runs (profile with memory_profiler)
5. **Device Compatibility**: Works on GPU, CPU, MPS without code changes
6. **Backward Compatibility**: API stable across minor versions

---

## DELIVERABLES

1. **Core Implementation**
   - `src/boltz/model/inference_affinity_only.py` (refactor placeholder)
   - `src/boltz/data/validation/*` (structure validation)
   - `src/boltz/data/preprocessing/*` (normalization)
   - `src/boltz/postprocessing/*` (results export)

2. **CLI & Integration**
   - `src/boltz/cli/rescore_commands.py`
   - `src/boltz/cli/config.py`

3. **Tests**
   - 50+ unit tests (>90% code coverage)
   - 15+ integration tests
   - 3+ end-to-end tests
   - Benchmark suite

4. **Documentation**
   - README with installation, quick start, examples
   - API documentation (docstrings)
   - Troubleshooting guide
   - Architecture document

5. **Configuration**
   - `config/default_config.yaml`
   - `config/validation_presets.yaml`

---

## SUCCESS CRITERIA (MUST ALL PASS)

✅ Process 100 typical complexes in <10 minutes  
✅ Input validation catches 99%+ of malformed files  
✅ Predictions reproducible (±1e-6 relative error on repeat)  
✅ All error cases surface actionable user message  
✅ No memory growth over 1000-complex batch  
✅ GPU and CPU paths produce identical predictions  
✅ Batch of 8 achieves 6-8x vs. single (or explain bottleneck)  
✅ Users can process their own data in <5 minutes with README  
✅ Unit test coverage >85%  
✅ All docstrings present and complete  
✅ Configuration system allows full customization  
✅ Zero data loss on Ctrl+C (graceful shutdown)

---

## BEGIN IMPLEMENTATION

You are ready to proceed. Structure your implementation as follows:

1. **Start with data handling** (validation, normalization)
2. **Build outward to featurization** (connect to existing model)
3. **Add inference engine** (forward passes, batching)
4. **Create CLI interface** (user-facing commands)
5. **Add tests throughout** (don't batch testing at end)
6. **Document comprehensively** (docstrings, examples)

Each component should be independently functional before integrating the next. Prioritize correctness over performance initially; optimize after validation.

**First commit checkpoint**: Working end-to-end on single simple complex (no batching, no CLI yet).

**Final checkpoint**: Full system passing all tests, documented, deployable.

Good luck! This is achievable in a single focused session. Focus on robustness and user experience—these matter more than absolute peak performance.

