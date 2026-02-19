# Production Implementation Quick Reference

## Phase Breakdown & Time Estimates

```
Phase 1: Data Handling (Est. 40% effort)
├─ StructureValidator (validation checks)
├─ ChainIdentifier (automatic detection)
├─ StructureNormalizer (standardization)
└─ Test suite for data layer

Phase 2: Featurization & Integration (Est. 25% effort)
├─ AffinityFeaturizer (feature extraction)
├─ ModelManager (checkpoint loading)
├─ Caching layer (embeddings)
└─ Test suite for features

Phase 3: Inference (Est. 20% effort)
├─ InferenceEngine (batch forward passes)
├─ UncertaintyEstimator (ensemble optional)
├─ Memory management (OOM recovery)
└─ Performance profiling

Phase 4: UI & Export (Est. 10% effort)
├─ CLI commands
├─ ResultsFormatter (multi-format export)
├─ Configuration management
└─ Documentation

Phase 5: Testing & Polish (Est. 5% effort)
├─ Integration tests
├─ Benchmarks
├─ Documentation
└─ Deployment checklist
```

---

## Critical Decision Points

### Decision 1: Stateless vs. Stateful Rescorer
**Question**: Should AffinityRescorer maintain state between calls?

**Recommendation**: **Stateless preferred**
- Each `rescore_pdb()` loads model fresh (if not cached)
- Enables parallel processing, server scaling
- Less risk of state corruption bugs
- Can add optional session/context manager for stateful variant

**Implementation**:
```python
# Preferred: Stateless
rescorer = AffinityRescorer("checkpoint")
result1 = rescorer.rescore_pdb("complex1.pdb")
result2 = rescorer.rescore_pdb("complex2.pdb")  # Fresh

# Optional: Session wrapper for efficiency
with AffinityRescorerSession("checkpoint") as session:
    results = [session.rescore_pdb(f) for f in files]  # Reuses model
```

### Decision 2: Validation Strategy (Strict vs. Lenient)
**Question**: How permissive should input validation be?

**Recommendation**: **Moderate default, configurable**
- Default: Accept common artifacts (alt locations, missing atoms)
- Strict mode: Reject any deviations
- Lenient mode: Accept anything parseable

**Config-driven**:
```python
validator = StructureValidator(config=ValidationConfig(level="moderate"))
# User can override: --validation strict/moderate/lenient
```

### Decision 3: Cache Management (When to Clear)
**Question**: How long should embedding cache persist?

**Recommendation**: **Session-level cache, optional persistent**
- Memory cache cleared per-process (no cross-run leaks)
- Optional `--cache-dir ~/.boltz/cache` for persistent caching
- Cache invalidation: model version + sequence hash

### Decision 4: Batch Size (Fixed vs. Dynamic)
**Question**: Use fixed batch size or adapt to GPU memory?

**Recommendation**: **Adaptive with fallback**
- Heuristic: Start with `--batch-size 8`
- If OOM: Reduce to 4, then 2, then 1
- Log warnings when falling back

---

## Key Classes to Implement

### 1. StructureValidator (50-100 lines per check)
```python
class StructureValidator:
    def validate_file_format(self, pdb_file) -> ValidationReport
    def validate_coordinates(self, coords, config) -> ValidationReport
    def validate_chain_continuity(self, structure) -> ValidationReport
    def validate_steric_clashes(self, coords, threshold) -> ValidationReport
    def validate_occupancy(self, structure) -> ValidationReport
    def validate_bfactors(self, structure, threshold) -> ValidationReport
    
    # Main entry point
    def validate(self, pdb_file, config) -> ValidationReport
```

### 2. ChainIdentifier (30-50 lines per heuristic)
```python
class ChainIdentifier:
    def _heuristic_residue_types(self, structure) -> Dict[str, float]  # confidence
    def _heuristic_molecular_weight(self, structure) -> Dict[str, float]
    def _heuristic_secondary_structure(self, structure) -> Dict[str, float]
    def _heuristic_connectivity(self, structure) -> Dict[str, float]
    
    def identify(self, structure, user_override) -> ChainAssignmentResult
```

### 3. StructureNormalizer (20-30 lines per operation)
```python
class StructureNormalizer:
    def _remove_waters(self, structure) -> Structure
    def _remove_ions(self, structure) -> Structure
    def _handle_alt_locations(self, structure) -> Structure
    def _center_coordinates(self, structure) -> Structure
    def _detect_disulfides(self, structure) -> List[Tuple]
    
    def normalize(self, structure, config) -> NormalizedStructure
```

### 4. AffinityFeaturizer (100-150 lines)
```python
class AffinityFeaturizer:
    def _compute_distance_matrix(self, coord1, coord2) -> np.ndarray  # vectorized
    def _bin_distances(self, distances) -> Tensor
    def _compute_pairwise_features(self, structure) -> Tensor
    def _get_embeddings(self, sequence, cache=True) -> Tensor
    def _atom_to_token_mapping(self, structure) -> np.ndarray
    
    def featurize(self, structure, sequence) -> FeatureDict
```

### 5. InferenceEngine (80-120 lines)
```python
class InferenceEngine:
    def _adaptive_batch_size(self, complexes, initial_batch=8) -> int
    def _forward_pass(self, batch_features) -> Tensor
    def _extract_predictions(self, model_output) -> List[Prediction]
    
    def infer_batch(
        self, features_list, model, batch_size=8, verbose=False
    ) -> List[AffinityPrediction]
```

### 6. ResultsFormatter (50-80 lines per format)
```python
class ResultsFormatter:
    def _to_json(self, results, include_metadata) -> str
    def _to_csv(self, results, include_metadata) -> str
    def _to_parquet(self, results, include_metadata) -> bytes
    
    def format(self, results, output_path, format) -> ExportReport
```

---

## Key Dataclasses (Keep Simple!)

```python
@dataclass
class ValidationReport:
    success: bool
    errors: List[Tuple[ErrorLevel, str]]
    warnings: List[str]
    metadata: Dict[str, Any]

@dataclass
class NormalizedStructure:
    protein_atoms: List[AtomRecord]
    protein_coords: np.ndarray
    ligand_atoms: List[AtomRecord]
    ligand_coords: np.ndarray
    transformation_log: List[str]
    metadata: Dict[str, Any]

@dataclass
class FeatureDict:
    distance_matrix: np.ndarray
    edge_features: Tensor
    protein_embeddings: Tensor
    ligand_embeddings: Tensor
    sequence_embeddings: Tensor
    validation: FeatureValidationReport

@dataclass
class AffinityPrediction:
    affinity: float
    affinity_std: float
    percentiles: Dict[str, float]
    inference_time_ms: float
    complex_id: str
    warnings: List[str]
```

---

## Testing Checklist

### Unit Tests (By Component)
- [ ] StructureValidator: 15+ tests
  - [ ] Valid coordinate ranges
  - [ ] Invalid coordinates (too large, NaN, Inf)
  - [ ] Occupancy edge cases (0, 1, 0.5)
  - [ ] B-factor outliers
  - [ ] Chain continuity breaks
  - [ ] Steric clashes
  - [ ] Non-standard residues
  - [ ] Alt locations

- [ ] ChainIdentifier: 10+ tests
  - [ ] Protein-only structure
  - [ ] Protein + ligand structure
  - [ ] Multi-chain proteins
  - [ ] User override override
  - [ ] Low-confidence cases

- [ ] Normalizer: 8+ tests
  - [ ] Water removal
  - [ ] Ion removal
  - [ ] Centering
  - [ ] Alt location handling
  - [ ] Coordinate transformation validation

- [ ] Featurizer: 12+ tests
  - [ ] Distance matrix correctness
  - [ ] Distance binning
  - [ ] Embedding shapes
  - [ ] NaN/Inf handling
  - [ ] Variable complex sizes

- [ ] InferenceEngine: 8+ tests
  - [ ] Single forward pass
  - [ ] Batch forward pass
  - [ ] OOM recovery
  - [ ] Timing tracking

### Integration Tests
- [ ] End-to-end: PDB → predictions (reference complex)
- [ ] Batch: 10 complexes without errors
- [ ] Format export: JSON, CSV, Parquet all work
- [ ] Device fallback: GPU → CPU produces same results
- [ ] Cache: Repeated calls use cache correctly
- [ ] Ensemble: Multi-seed predictions reasonable

### Regression Tests
- [ ] Reference complex: ±0.01 kcal/mol vs. expected
- [ ] Latency: Single complex < 5 sec on GPU
- [ ] Memory: Batch of 100 < 4GB growth

---

## CLI Interface

### Single Complex
```bash
boltz rescore pdb complex.pdb \
    --output result.json \
    --protein-chain A \
    --ligand-chains B,C \
    --device auto \
    --validation moderate
```

### Batch Processing
```bash
boltz rescore batch ./pdbs \
    --output results.csv \
    --recursive \
    --batch-size 8 \
    --max-workers 4
```

### Streaming/Manifest
```bash
boltz rescore manifest complexes.yaml \
    --output-dir ./results \
    --use-ensemble \
    --num-seeds 5
```

---

## Performance Profiling Commands

```python
# Memory profiling
from memory_profiler import profile

@profile
def full_rescoring_pipeline(pdb_files):
    rescorer = AffinityRescorer("checkpoint")
    for pdb_file in pdb_files:
        result = rescorer.rescore_pdb(pdb_file)

# Run: mprof run script.py && mprof plot

# Latency profiling
import cProfile
cProfile.run('rescorer.rescore_pdb("complex.pdb")', 'stats')

# Then: pstats stats → sort cumulative → print stats 10

# GPU profiling (NVIDIA)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits -l 1
```

---

## Common Pitfalls to Avoid

❌ **Mistake 1**: Modifying input structure in place
✅ **Fix**: Always deep copy before modifications

❌ **Mistake 2**: Not validating feature shapes before model
✅ **Fix**: Add explicit shape assertions

❌ **Mistake 3**: Leaving GPU tensors on device after inference
✅ **Fix**: Always `.cpu()` before saving

❌ **Mistake 4**: Cache growing unbounded
✅ **Fix**: Implement LRU eviction

❌ **Mistake 5**: Model not in eval mode during inference
✅ **Fix**: model.eval() required

❌ **Mistake 6**: Seeding only numpy, not torch
✅ **Fix**: Set both `np.random.seed()` and `torch.manual_seed()`

❌ **Mistake 7**: Not catching torch.cuda.OutOfMemoryError specifically
✅ **Fix**: Catch specific exception, fallback to CPU

❌ **Mistake 8**: Batch processing fails on one → whole job fails
✅ **Fix**: Process atomically, collect errors, continue

❌ **Mistake 9**: Random file I/O errors on network drives
✅ **Fix**: Implement retry logic with exponential backoff

❌ **Mistake 10**: No logging = no debugging
✅ **Fix**: Full logging at DEBUG/INFO/WARNING/ERROR levels

---

## Definition of Done Checklist

For each feature, verify:

- [ ] Code written and committed
- [ ] Docstrings complete (parameter types, returns, raises)
- [ ] Unit tests written and passing
- [ ] Edge cases handled gracefully
- [ ] Error messages actionable
- [ ] Config option available if applicable
- [ ] Logging statement added (DEBUG level)
- [ ] Performance acceptable (<target latency)
- [ ] No GPU memory leaks (tested with 100+ items)
- [ ] Works on CPU and GPU
- [ ] Reproducible (same seed → same result)

---

## File Structure Template

```
src/boltz/model/
├── inference_affinity_only.py       # Main rescorer class
└── inference_components/            # (NEW)
    ├── __init__.py
    ├── structure_validator.py
    ├── chain_identifier.py
    ├── structure_normalizer.py
    ├── featurizer.py
    ├── model_manager.py
    ├── inference_engine.py
    └── uncertainty.py

src/boltz/data/
└── preprocessing/                   # (NEW)
    ├── __init__.py
    ├── errors.py
    ├── config.py
    └── types.py

src/boltz/cli/
├── rescore_commands.py              # (NEW)
└── utils.py

src/boltz/postprocessing/            # (NEW)
├── __init__.py
├── formatter.py
└── validators.py

tests/
├── unit/
│   ├── test_validator.py
│   ├── test_chain_identifier.py
│   ├── test_normalizer.py
│   ├── test_featurizer.py
│   ├── test_inference_engine.py
│   └── test_formatter.py
├── integration/
│   ├── test_end_to_end.py
│   ├── test_batch_processing.py
│   └── test_device_fallback.py
└── data/
    ├── reference_complex.pdb
    ├── malformed_complex.pdb
    └── large_complex.pdb

docs/
├── README.md
├── API.md
├── EXAMPLES.md
└── TROUBLESHOOTING.md
```

---

## Verification Milestones

**Milestone 1** (Data layer working):
- [ ] StructureValidator passes 15+ unit tests
- [ ] ChainIdentifier correctly identifies 5 test cases
- [ ] StructureNormalizer produces valid output
- **Verification**: `pytest tests/unit/test_validator.py -v`

**Milestone 2** (Features working):
- [ ] Featurizer produces correct shapes
- [ ] ModelManager loads checkpoint successfully
- [ ] Features validated against training distribution
- **Verification**: `pytest tests/unit/test_featurizer.py -v`

**Milestone 3** (Inference working):
- [ ] Single complex inference completes <5 sec
- [ ] Batch of 8 processes without OOM
- [ ] Results match expected ranges
- **Verification**: `pytest tests/integration/test_end_to_end.py -v`

**Milestone 4** (UI working):
- [ ] CLI commands parse arguments correctly
- [ ] Single complex output is valid JSON/CSV
- [ ] Batch processing completes successfully
- **Verification**: `boltz rescore pdb test_data/complex.pdb --dry-run`

**Milestone 5** (Production ready):
- [ ] All tests passing (>100 tests)
- [ ] Documentation complete
- [ ] Performance benchmarks met
- [ ] No memory leaks (1000 items)
- **Verification**: Full test suite + benchmark suite

---

## Quick Start for Implementation

```python
# 1. Begin with simplest validator
class StructureValidator:
    def validate_file_exists(self, pdb_file) -> bool:
        return pdb_file.exists()

# 2. Add one check at a time
    def validate_coordinates_numeric(self, coords) -> bool:
        return np.all(np.isfinite(coords))

# 3. Test each
# pytest tests/unit/test_validator.py::test_file_exists -v

# 4. Then build next component
class StructureNormalizer:
    def normalize(self, structure):
        # Apply each transformation
        return normalized

# 5. Test again
# pytest tests/unit/test_normalizer.py -v

# 6. Connect components
class AffinityRescorer:
    def rescore_pdb(self, pdb_file):
        structure = parse_pdb(pdb_file)
        report = validator.validate(structure)
        normalized = normalizer.normalize(structure)
        features = featurizer.featurize(normalized)
        prediction = engine.infer([features])[0]
        return prediction

# 7. Test end-to-end
# pytest tests/integration/test_end_to_end.py -v
```

---

## Final Checklist Before Submission

- [ ] All code merged to main branch
- [ ] All tests passing (`pytest --cov=src/boltz`)
- [ ] Type hints present for all functions
- [ ] Docstrings complete (Google style)
- [ ] No hardcoded paths (all configurable)
- [ ] No debug print statements
- [ ] Error messages non-technical for users
- [ ] README has quick start example
- [ ] API documentation generated
- [ ] Performance benchmarks documented
- [ ] Backwards compatibility maintained
- [ ] No breaking changes to existing APIs

---

**This is your implementation roadmap. Follow it sequentially, test at each step, and you'll have a production system.**

