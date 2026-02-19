# Rescoring Protein-Ligand Complexes with Boltz Affinity Module

This guide explains how to modify the Boltz codebase to perform rapid rescoring of existing protein-ligand complexes using only the affinity module, bypassing the slow diffusion and transformer steps.

## Architecture Overview

The current Boltz-2 pipeline consists of:
1. **Transformer blocks** → Predicts initial structure and embeddings
2. **Diffusion process** → Iteratively refines the structure (slow, ~200 sampling steps)
3. **Confidence module** → Estimates prediction confidence
4. **Affinity module** → Predicts ligand-protein binding affinity (your target)

### Why This Approach is Faster

- **Current flow**: YAML/FASTA → Featurization → MSA generation → Transformer → Diffusion → Confidence → Affinity (~minutes per complex)
- **Proposed flow**: PDB/CIF (existing structure) → Featurization → Affinity module only (~seconds per complex)

The affinity module operates on:
- Protein-ligand structure coordinates (x_pred)
- Token-level embeddings (z)
- Sequence embeddings (s_inputs)
- Distance features (feats)

## Implementation Steps

### 1. Create a New Data Module for Existing Structures

**File:** `src/boltz/data/module/inference_affinity_only.py`

Create a lightweight data module that:
- Takes PDB/CIF files as input
- Skips MSA computation
- Loads pre-computed coordinates directly
- Generates featurization needed for the affinity head

Key differences from `Boltz2InferenceDataModule`:
```python
# Instead of BoltzInferenceDataModule which includes full pipeline,
# Create AffinityOnlyInferenceDataModule that:
# 1. Loads structure from disk (PDB/CIF)
# 2. Extracts token embeddings from affinity-trained encoder
# 3. Computes distance features
# 4. Skips diffusion entirely
```

### 2. Create Input Parser for PDB/CIF Files

**File:** `src/boltz/data/parse/pdb_structure.py`

Extend existing PDB parser to:
- Parse existing PDB/CIF structures (not just templates)
- Identify protein chains and ligands
- Extract coordinates and connectivity
- Create mock "Structure" objects compatible with the featurization pipeline

### 3. Create Lightweight Featurizer

**File:** `src/boltz/data/feature/featurizer_affinity_only.py`

Create specialized featurizer that:
- **Skips** all-atom modeling, diffusion prep, residue cropping logic
- Directly computes:
  - Distance matrices between atoms
  - Pair features for the affinity module
  - Token embeddings (or cache them)

### 4. Create Affinity-Only Inference Module

**File:** `src/boltz/model/inference_affinity_only.py`

This is the core module:
```python
class AffinityOnlyInferenceModule:
    """
    Wraps the affinity module for standalone rescoring.
    
    Workflow:
    1. Load pre-computed or lightweight features
    2. Forward through feature encoder (if needed)
    3. Forward through affinity module only
    4. Extract binding affinity score
    """
    
    def __init__(self, affinity_checkpoint: str, device: str = "cuda"):
        self.affinity_model = self._load_affinity_weights(affinity_checkpoint)
        
    def forward(self, pdb_file: str) -> dict:
        """
        Args:
            pdb_file: Path to PDB/CIF file
            
        Returns:
            {
                'affinity_pred': float,  # binding affinity prediction
                'affinity_std': float,    # prediction uncertainty
                'ligand_id': str,
                'protein_id': str
            }
        """
        pass
```

### 5. Create CLI Command: `rescore`

**File:** `src/boltz/main.py` (add new command)

Add a new Click command alongside `predict`:
```bash
boltz rescore --input complexes.pdb --output results.json --checkpoint boltz2_aff.ckpt
```

**Features:**
- Batch processing of multiple PDB/CIF files
- For each file:
  - Parse protein and ligand
  - Load coordinates
  - Run through affinity module
  - Write affinity scores to JSON/CSV

### 6. Input Format

Accept directories with PDB files plus metadata YAML:

```yaml
# complexes.yaml
complexes:
  - id: complex_001
    pdb: ./pdbs/complex_001.pdb
    protein_chain: A
    ligand_chain: [B, C]
    
  - id: complex_002
    pdb: ./pdbs/complex_002.pdb
    protein_chain: A
    ligand_chain: L
```

Or simpler: directory with `.pdb` files where:
- Protein chains: A, B, C, etc.
- Ligand: typically chains X, Y, Z or small molecules in HETATM records

### 7. Output Format

Generate CSV/JSON with results:
```json
[
  {
    "id": "complex_001",
    "affinity": -7.2,
    "affinity_std": 0.3,
    "num_atoms_protein": 1245,
    "num_atoms_ligand": 42,
    "timestamp": "2026-02-18T10:30:00"
  },
  ...
]
```

## Key Code Changes Required

### Option A: Minimal Changes (Reuse Existing Components)

1. Load the pre-trained Boltz-2 affinity model weights
2. Create a wrapper that:
   - Takes PDB coordinates + chain IDs
   - Extracts coordinates into tensor format
   - Computes distance features manually
   - Runs through affinity head only

**Pros:** Minimal code changes, leverages existing model
**Cons:** Need to replicate some featurization logic

### Option B: Structural Refactoring (Cleaner)

1. Extract affinity module logic into standalone components
2. Create dedicated `AffinityScorer` class
3. Separate concerns:
   - Structure loading ← Own logic
   - Feature computation ← Subset of existing
   - Affinity prediction ← Existing model

**Pros:** Clean architecture, reusable components
**Cons:** More refactoring upfront

## Recommended Implementation Path

1. **Phase 1** (Quick): Modify existing inference pipeline
   - Add `--skip_diffusion` flag
   - Add `--skip_transformer` flag  
   - Add `--input_coordinates` flag for existing structures
   
2. **Phase 2** (Clean): Create dedicated affinity rescoring module
   - New `AffinityRescorer` class
   - Simplified data pipeline
   - Optimized for batch processing

3. **Phase 3** (Polish): Add utilities
   - PDB/CIF batch processing
   - Results aggregation
   - Performance benchmarking

## Example Usage (After Implementation)

```bash
# Single complex
python -m boltz rescore --pdb complex.pdb --checkpoint boltz2_aff.ckpt

# Batch processing
python -m boltz rescore --input_dir ./complexes --checkpoint boltz2_aff.ckpt --output results.csv

# From YAML manifest
python -m boltz rescore --manifest complexes.yaml --checkpoint boltz2_aff.ckpt
```

## Performance Expectations

- **Current full pipeline**: ~2-5 minutes per complex
- **Transformer only**: ~30-45 seconds
- **Affinity-only (proposed)**: ~2-5 seconds per complex

**Speedup**: 10-50x faster than full pipeline

## Key Files to Examine

- `src/boltz/model/modules/affinity.py` - The AffinityModule implementation
- `src/boltz/model/models/boltz2.py` - How affinity is currently integrated
- `src/boltz/data/module/inferencev2.py` - Current inference pipeline
- `src/boltz/data/feature/featurizerv2.py` - Featurization logic
- `examples/affinity.yaml` - Affinity input format

## Notes

- The affinity module expects structured pair-wise features between protein atoms and ligand atoms
- You'll need access to trained encoder embeddings (already in the checkpoint)
- Distance binning and pairwise conditioning are critical for the affinity head
- The model was trained with augmented coordinate data, so slight coordinate perturbations shouldn't hurt

## Debugging Checklist

- ✓ Affinity model loads from checkpoint
- ✓ Forward pass shape compatibility (batch, atoms, features)
- ✓ Feature dimensions match training (token_s, token_z)
- ✓ Output extraction matches training objective (e.g., regression head)
- ✓ Batch processing handles variable-size complexes
