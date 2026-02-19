"""
Main AffinityRescorer class.

Orchestrates the complete affinity rescoring pipeline:
input validation → parsing → featurization → inference → export.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from boltz.affinity_rescoring.export import ResultsExporter, compute_batch_summary
from boltz.affinity_rescoring.inference import (
    AffinityModelManager,
    create_affinity_yaml,
)
from boltz.affinity_rescoring.models import (
    AffinityResult,
    BatchSummary,
    DeviceOption,
    LigandScore,
    OutputFormat,
    RescoreConfig,
    Timer,
    ValidationLevel,
    ValidationStatus,
    compute_file_sha256,
)
from boltz.affinity_rescoring.mol2_parser import MOL2Parser
from boltz.affinity_rescoring.parsers import (
    count_receptor_contacts,
    get_chain_sequences,
    get_seqres_sequences,
    parse_structure_file,
)
from boltz.affinity_rescoring.smiles_inference import (
    infer_ligand_smiles_from_structure,
)
from boltz.affinity_rescoring.validation import (
    ChainIdentifier,
    StructureNormalizer,
    StructureValidator,
)

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _merge_reference_sequences(
    structure_path: Path,
    user_sequences: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build a merged reference-sequence dict: user overrides > SEQRES.

    Priority order:
    1. User-supplied ``reference_sequences`` (always wins)
    2. SEQRES / entity sequences extracted from the structure file

    Parameters
    ----------
    structure_path : Path
        Path to the PDB / CIF file (used for SEQRES extraction).
    user_sequences : dict, optional
        User-provided chain_id → full sequence overrides.

    Returns
    -------
    dict
        Merged chain_id → reference sequence.  May be empty if the file
        has no SEQRES records and the user didn't provide anything.
    """
    # Start with SEQRES
    merged = get_seqres_sequences(structure_path)

    # Overlay user-provided sequences (they take priority)
    if user_sequences:
        for chain_id, seq in user_sequences.items():
            if chain_id in merged:
                logger.info(
                    f"Chain {chain_id}: user-provided reference sequence "
                    f"({len(seq)} residues) overrides SEQRES "
                    f"({len(merged[chain_id])} residues)."
                )
            else:
                logger.info(
                    f"Chain {chain_id}: using user-provided reference sequence "
                    f"({len(seq)} residues)."
                )
            merged[chain_id] = seq

    return merged


class AffinityRescorer:
    """
    Production-ready protein-ligand affinity rescoring system.

    Leverages the Boltz-2 affinity module to predict binding affinities
    for protein-ligand complexes. Supports single complex scoring,
    batch processing, and receptor-based virtual screening.

    Parameters
    ----------
    checkpoint : str
        Path to affinity model checkpoint, or 'auto' to download.
    device : str
        Device to use: 'auto', 'cuda', 'cpu', 'mps'.
    config : RescoreConfig, optional
        Full configuration object. Overrides individual params.
    cache_dir : str, optional
        Cache directory for checkpoints and processed data.

    Examples
    --------
    >>> rescorer = AffinityRescorer("boltz2_aff.ckpt")
    >>> result = rescorer.rescore_pdb("complex.pdb")
    >>> print(f"Affinity: {result.affinity_pred:.2f}")

    >>> results = rescorer.rescore_batch(["complex1.pdb", "complex2.pdb"])
    >>> rescorer.export_results(results, "scores.csv", format="csv")
    """

    def __init__(
        self,
        checkpoint: str = "auto",
        device: str = "auto",
        config: Optional[RescoreConfig] = None,
        cache_dir: Optional[str] = None,
        validation_level: str = "moderate",
    ):
        self.config = config or RescoreConfig(
            checkpoint=checkpoint,
            device=DeviceOption(device),
            validation_level=ValidationLevel(validation_level),
        )

        self._cache_dir = Path(
            cache_dir or os.environ.get("BOLTZ_CACHE", "~/.boltz")
        ).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._validator = StructureValidator(level=self.config.validation_level)
        self._chain_identifier = ChainIdentifier()
        self._normalizer = StructureNormalizer()
        self._mol2_parser = MOL2Parser()
        self._exporter = ResultsExporter(model_checkpoint=checkpoint)

        # Model manager - lazy initialization
        self._model_manager: Optional[AffinityModelManager] = None
        self._checkpoint = checkpoint

        # Results cache
        self._last_results: List[AffinityResult] = []

    @property
    def results(self) -> ResultsExporter:
        """Access the results exporter with last results."""
        return self._exporter

    # ─── Single Complex Rescoring ─────────────────────────────────────────

    def rescore_pdb(
        self,
        input_path: str | Path,
        protein_chain: Optional[str] = None,
        ligand_chains: Optional[List[str]] = None,
        output_path: Optional[str | Path] = None,
        output_format: str = "json",
        ligand_smiles: Optional[Dict[str, str]] = None,
        use_msa_server: bool = False,
        reference_sequences: Optional[Dict[str, str]] = None,
    ) -> AffinityResult:
        """
        Rescore a single PDB/CIF complex.

        Parameters
        ----------
        input_path : str or Path
            Path to PDB or CIF file
        protein_chain : str, optional
            Protein chain ID (auto-detected if not provided)
        ligand_chains : list of str, optional
            Ligand chain IDs (auto-detected if not provided)
        output_path : str or Path, optional
            Where to save results
        output_format : str
            Output format (json, csv, parquet, sqlite, excel)
        ligand_smiles : dict, optional
            Map of chain_id → SMILES for ligands. If not provided,
            SMILES are automatically inferred from HETATM coordinates
            using RDKit bond perception (rdDetermineBonds). User-provided
            SMILES always take priority over auto-inferred ones.
        use_msa_server : bool
            Whether to use MSA server for sequence search
        reference_sequences : dict, optional
            Map of chain_id → full amino acid sequence. Overrides
            both ATOM-derived and SEQRES-derived sequences when
            provided, ensuring the model sees the complete biological
            sequence even if the PDB has missing loops.

        Returns
        -------
        AffinityResult
            Prediction result with metadata
        """
        path = Path(input_path)
        result = AffinityResult(id=path.stem, source_file=str(path))

        with Timer() as total_timer:
            try:
                # Step 1: Validate file
                file_report = self._validator.validate_file(path)
                if not file_report.is_valid:
                    result.validation_status = ValidationStatus.FAILED
                    result.error_message = "; ".join(str(e) for e in file_report.errors)
                    result.validation_issues = [str(e) for e in file_report.issues]
                    return result

                # Step 2: Parse structure
                with Timer() as parse_timer:
                    atoms, metadata = parse_structure_file(path)

                # Step 3: Validate atoms
                atom_report = self._validator.validate_atoms(atoms)
                chain_report = self._validator.validate_chains(atoms)

                all_issues = atom_report.issues + chain_report.issues
                result.validation_issues = [str(i) for i in all_issues]
                result.warnings = [str(i) for i in all_issues if i.severity == "warning"]

                if not atom_report.is_valid:
                    if self.config.validation_level == ValidationLevel.STRICT:
                        result.validation_status = ValidationStatus.FAILED
                        result.error_message = "; ".join(str(e) for e in atom_report.errors)
                        return result
                    else:
                        result.validation_status = ValidationStatus.WARNING

                # Step 4: Identify chains
                chain_assignment = self._chain_identifier.identify_chains(
                    atoms,
                    protein_chains=[protein_chain] if protein_chain else None,
                    ligand_chains=ligand_chains,
                )

                if not chain_assignment.protein_chains:
                    result.validation_status = ValidationStatus.FAILED
                    result.error_message = (
                        "No protein chains detected. "
                        "Provide explicit --protein-chain or check file."
                    )
                    return result

                if not chain_assignment.ligand_chains:
                    result.validation_status = ValidationStatus.FAILED
                    result.error_message = (
                        "No ligand detected. "
                        "Provide explicit --ligand-chains or check file."
                    )
                    return result

                result.protein_chain = chain_assignment.protein_chains[0]
                result.ligand_chains = chain_assignment.ligand_chains

                # Step 5: Get sequences (with gap-filling from SEQRES + user overrides)
                merged_refs = _merge_reference_sequences(
                    path, reference_sequences
                )
                sequences = get_chain_sequences(atoms, reference_sequences=merged_refs)
                result.protein_residue_count = len(sequences.get(result.protein_chain, ""))

                ligand_atoms = [a for a in atoms if a.chain_id in set(chain_assignment.ligand_chains)]
                result.ligand_atom_count = len(ligand_atoms)

                # Step 6: Resolve ligand SMILES (user-provided or auto-inferred)
                resolved_smiles, smiles_warnings = infer_ligand_smiles_from_structure(
                    atoms=atoms,
                    ligand_chains=chain_assignment.ligand_chains,
                    user_smiles=ligand_smiles,
                )
                result.warnings.extend(smiles_warnings)

                if not resolved_smiles:
                    result.validation_status = ValidationStatus.FAILED
                    result.error_message = (
                        "Could not determine ligand SMILES. Auto-inference from "
                        "HETATM coordinates failed. Provide SMILES manually via "
                        "--ligand-smiles or ligand_smiles parameter. "
                        "Example: {'B': 'CCO'}"
                    )
                    return result

                ligand_smiles = resolved_smiles

                # Step 7: Create temporary YAML and run prediction
                with Timer() as inference_timer:
                    affinity_output = self._run_boltz_prediction(
                        sequences=sequences,
                        protein_chains=chain_assignment.protein_chains,
                        ligand_chains=chain_assignment.ligand_chains,
                        ligand_smiles=ligand_smiles,
                        use_msa_server=use_msa_server,
                    )

                result.inference_time_ms = inference_timer.elapsed_ms

                if affinity_output is None:
                    result.validation_status = ValidationStatus.FAILED
                    result.error_message = "Affinity prediction failed. Check logs."
                    return result

                # Step 8: Parse results
                result.affinity_pred = affinity_output.get("affinity_pred_value", float("nan"))
                result.affinity_probability_binary = affinity_output.get(
                    "affinity_probability_binary", float("nan")
                )

                # Ensemble values
                if "affinity_pred_value1" in affinity_output:
                    result.affinity_pred_value1 = affinity_output["affinity_pred_value1"]
                    result.affinity_pred_value2 = affinity_output.get("affinity_pred_value2")
                    result.affinity_probability_binary1 = affinity_output.get(
                        "affinity_probability_binary1"
                    )
                    result.affinity_probability_binary2 = affinity_output.get(
                        "affinity_probability_binary2"
                    )

                    # Compute std from ensemble
                    if result.affinity_pred_value1 is not None and result.affinity_pred_value2 is not None:
                        vals = [result.affinity_pred_value1, result.affinity_pred_value2]
                        result.affinity_std = float(
                            (sum((v - result.affinity_pred) ** 2 for v in vals) / len(vals)) ** 0.5
                        )

                if result.validation_status != ValidationStatus.WARNING:
                    result.validation_status = ValidationStatus.SUCCESS

            except Exception as e:
                logger.exception(f"Error rescoring {path}: {e}")
                result.validation_status = ValidationStatus.FAILED
                result.error_message = str(e)

        result.processing_time_ms = total_timer.elapsed_ms
        result.model_checkpoint = self._checkpoint

        # Export if output path specified
        if output_path is not None:
            fmt = OutputFormat(output_format)
            self._exporter.export([result], output_path, fmt=fmt)

        self._last_results = [result]
        return result

    # ─── Batch Rescoring ──────────────────────────────────────────────────

    def rescore_batch(
        self,
        pdb_files: List[str | Path],
        protein_chain: Optional[str] = None,
        ligand_chains: Optional[List[str]] = None,
        output_path: Optional[str | Path] = None,
        output_format: str = "csv",
        ligand_smiles: Optional[Dict[str, str]] = None,
        use_msa_server: bool = False,
    ) -> List[AffinityResult]:
        """
        Rescore a batch of PDB/CIF complexes.

        Each complex is processed independently - failures don't cascade.

        Parameters
        ----------
        pdb_files : list of str/Path
            Paths to structure files
        protein_chain, ligand_chains : optional
            Applied to all complexes if provided
        output_path : str/Path, optional
            Where to save aggregated results
        output_format : str
            Output format for aggregated results
        ligand_smiles : dict, optional
            Ligand SMILES mapping (applied to all complexes)
        use_msa_server : bool
            Whether to use MSA server

        Returns
        -------
        list of AffinityResult
        """
        results: List[AffinityResult] = []

        try:
            from tqdm import tqdm
            files_iter = tqdm(pdb_files, desc="Rescoring", unit="complex")
        except ImportError:
            files_iter = pdb_files

        for pdb_file in files_iter:
            try:
                result = self.rescore_pdb(
                    pdb_file,
                    protein_chain=protein_chain,
                    ligand_chains=ligand_chains,
                    ligand_smiles=ligand_smiles,
                    use_msa_server=use_msa_server,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdb_file}: {e}")
                results.append(AffinityResult(
                    id=Path(pdb_file).stem,
                    source_file=str(pdb_file),
                    validation_status=ValidationStatus.FAILED,
                    error_message=str(e),
                ))

        # Export aggregated results
        if output_path is not None:
            fmt = OutputFormat(output_format)
            self._exporter.export(results, output_path, fmt=fmt)

        self._last_results = results
        return results

    # ─── Directory Scanning ───────────────────────────────────────────────

    def rescore_directory(
        self,
        input_dir: str | Path,
        output_path: Optional[str | Path] = None,
        output_format: str = "csv",
        recursive: bool = False,
        ligand_smiles: Optional[Dict[str, str]] = None,
        use_msa_server: bool = False,
    ) -> List[AffinityResult]:
        """
        Rescore all PDB/CIF files in a directory.

        Parameters
        ----------
        input_dir : str or Path
            Directory containing structure files
        output_path : str/Path, optional
            Where to save results
        output_format : str
            Output format
        recursive : bool
            Scan subdirectories
        ligand_smiles : dict, optional
            Ligand SMILES mapping
        use_msa_server : bool
            Whether to use MSA server

        Returns
        -------
        list of AffinityResult
        """
        dir_path = Path(input_dir)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        extensions = ("*.pdb", "*.cif", "*.mmcif", "*.ent", "*.pdbx")
        files: List[Path] = []

        for ext in extensions:
            if recursive:
                files.extend(dir_path.rglob(ext))
            else:
                files.extend(dir_path.glob(ext))

        files.sort()
        logger.info(f"Found {len(files)} structure files in {dir_path}")

        if not files:
            logger.warning(f"No structure files found in {dir_path}")
            return []

        return self.rescore_batch(
            pdb_files=[str(f) for f in files],
            output_path=output_path,
            output_format=output_format,
            ligand_smiles=ligand_smiles,
            use_msa_server=use_msa_server,
        )

    # ─── Receptor-Based Rescoring ─────────────────────────────────────────

    def rescore_receptor(
        self,
        receptor_path: str | Path,
        ligands_path: str | Path,
        protein_chain: Optional[str] = None,
        output_path: str | Path = "scores.csv",
        output_format: str = "csv",
        validation_level: str = "moderate",
        sort_by: str = "affinity_score",
        reference_sequences: Optional[Dict[str, str]] = None,
    ) -> List[LigandScore]:
        """
        Score a fixed receptor against multiple ligands from MOL2 file.

        This is the primary workflow for virtual screening:
        one protein receptor scored against many ligand poses.

        Parameters
        ----------
        receptor_path : str or Path
            Path to receptor PDB/CIF file
        ligands_path : str or Path
            Path to MOL2 file with multiple ligands
        protein_chain : str, optional
            Protein chain ID (auto-detected if not provided)
        output_path : str or Path
            Output file path
        output_format : str
            Output format (csv, excel, parquet, json)
        validation_level : str
            Validation strictness
        sort_by : str
            Column to sort results by
        reference_sequences : dict, optional
            Map of chain_id → full amino acid sequence.
            Overrides ATOM-derived and SEQRES-derived sequences.

        Returns
        -------
        list of LigandScore
        """
        receptor = Path(receptor_path)
        ligands = Path(ligands_path)

        # Validate receptor
        receptor_report = self._validator.validate_file(receptor)
        if not receptor_report.is_valid:
            raise ValueError(
                f"Invalid receptor file: "
                + "; ".join(str(e) for e in receptor_report.errors)
            )

        # Parse receptor
        logger.info(f"Loading receptor: {receptor}")
        receptor_atoms, receptor_meta = parse_structure_file(receptor)

        # Identify protein chain
        chain_assignment = self._chain_identifier.identify_chains(
            receptor_atoms,
            protein_chains=[protein_chain] if protein_chain else None,
            ligand_chains=[],
        )

        if not chain_assignment.protein_chains:
            raise ValueError(
                "No protein chain detected in receptor. "
                "Provide --protein-chain explicitly."
            )

        protein_chain_id = chain_assignment.protein_chains[0]
        protein_atoms = [a for a in receptor_atoms if a.chain_id == protein_chain_id]
        merged_refs = _merge_reference_sequences(receptor, reference_sequences)
        sequences = get_chain_sequences(receptor_atoms, reference_sequences=merged_refs)
        protein_seq = sequences.get(protein_chain_id, "")

        logger.info(
            f"Receptor: chain {protein_chain_id}, "
            f"{len(protein_atoms)} atoms, {len(protein_seq)} residues"
        )

        # Parse ligands
        logger.info(f"Parsing ligands from: {ligands}")
        ligand_structures = self._mol2_parser.extract_ligands_with_names(ligands)
        logger.info(f"Found {len(ligand_structures)} ligands")

        if not ligand_structures:
            logger.warning("No ligands found in MOL2 file")
            return []

        # Score each ligand
        scores: List[LigandScore] = []

        try:
            from tqdm import tqdm
            lig_iter = tqdm(ligand_structures, desc="Scoring ligands", unit="lig")
        except ImportError:
            lig_iter = ligand_structures

        for lig_struct in lig_iter:
            score = self._score_single_ligand(
                lig_struct,
                protein_atoms,
                protein_seq,
                protein_chain_id,
            )
            scores.append(score)

        # Export
        fmt = OutputFormat(output_format)
        self._exporter.export_receptor_results(
            scores,
            output_path,
            fmt=fmt,
            sort_by=sort_by,
        )

        logger.info(
            f"Completed: {len(scores)} ligands, "
            f"{sum(1 for s in scores if s.validation_status == ValidationStatus.SUCCESS)} successful, "
            f"{sum(1 for s in scores if s.validation_status == ValidationStatus.FAILED)} failed"
        )

        return scores

    # ─── Export Convenience ───────────────────────────────────────────────

    def export_results(
        self,
        results: Optional[List[AffinityResult]] = None,
        output_path: str | Path = "results.csv",
        format: str = "csv",
    ) -> Path:
        """Export results to a file."""
        if results is None:
            results = self._last_results
        fmt = OutputFormat(format)
        return self._exporter.export(results, output_path, fmt=fmt)

    # ─── Internal Methods ─────────────────────────────────────────────────

    def _run_boltz_prediction(
        self,
        sequences: Dict[str, str],
        protein_chains: List[str],
        ligand_chains: List[str],
        ligand_smiles: Dict[str, str],
        use_msa_server: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Run Boltz prediction by creating YAML and invoking the predict command.

        Returns parsed affinity JSON or None on failure.
        """
        # Create temp directory
        work_dir = Path(tempfile.mkdtemp(prefix="boltz_rescore_"))

        try:
            # Create YAML
            yaml_path = work_dir / "input.yaml"
            yaml_data = self._build_yaml_data(
                sequences, protein_chains, ligand_chains, ligand_smiles
            )

            import yaml
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_data, f, default_flow_style=False)

            # Run Boltz predict
            cmd = [
                sys.executable, "-m", "boltz", "predict",
                str(yaml_path),
                "--out_dir", str(work_dir / "output"),
                "--recycling_steps", str(self.config.recycling_steps),
                "--diffusion_samples_affinity", str(self.config.diffusion_samples),
                "--sampling_steps", str(self.config.sampling_steps),
            ]

            if self._checkpoint != "auto":
                cmd.extend(["--affinity_checkpoint", self._checkpoint])

            if not self.config.affinity_mw_correction:
                # The CLI flag is --affinity_mw_correction (toggle)
                pass  # Default is True

            if self.config.device != DeviceOption.AUTO:
                cmd.extend(["--accelerator", self.config.device.value])

            if use_msa_server:
                cmd.append("--use_msa_server")

            logger.debug(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
                cwd=str(work_dir),
            )

            if result.returncode != 0:
                logger.error(f"Boltz predict failed (rc={result.returncode}):")
                logger.error(result.stderr[-2000:] if result.stderr else "No stderr")
                return None

            # Find affinity output
            output_base = work_dir / "output"
            for json_file in output_base.rglob("affinity_*.json"):
                with open(json_file) as f:
                    return json.load(f)

            logger.warning("No affinity JSON found in output")
            return None

        except subprocess.TimeoutExpired:
            logger.error("Boltz predict timed out (>2h)")
            return None
        except Exception as e:
            logger.error(f"Error running Boltz predict: {e}")
            return None
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

    def _build_yaml_data(
        self,
        sequences: Dict[str, str],
        protein_chains: List[str],
        ligand_chains: List[str],
        ligand_smiles: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build the YAML data structure for Boltz input."""
        yaml_sequences = []

        for chain_id in protein_chains:
            if chain_id in sequences:
                yaml_sequences.append({
                    "protein": {
                        "id": chain_id,
                        "sequence": sequences[chain_id],
                    }
                })

        binder_chain = None
        for chain_id in ligand_chains:
            if chain_id in ligand_smiles:
                yaml_sequences.append({
                    "ligand": {
                        "id": chain_id,
                        "smiles": ligand_smiles[chain_id],
                    }
                })
                if binder_chain is None:
                    binder_chain = chain_id

        if binder_chain is None:
            raise ValueError(
                f"No SMILES provided for ligand chains {ligand_chains}. "
                f"Available SMILES keys: {list(ligand_smiles.keys())}"
            )

        return {
            "version": 1,
            "sequences": yaml_sequences,
            "properties": [
                {"affinity": {"binder": binder_chain}},
            ],
        }

    def _score_single_ligand(
        self,
        ligand: "LigandStructure",
        protein_atoms: List["AtomInfo"],
        protein_seq: str,
        protein_chain_id: str,
    ) -> LigandScore:
        """
        Score a single ligand against a receptor.

        Note: This currently creates a placeholder score since
        direct MOL2→Boltz integration requires SMILES conversion.
        For full functionality, ligands should provide SMILES strings.
        """
        score = LigandScore(
            ligand_name=ligand.name,
            n_atoms=ligand.num_atoms,
        )

        with Timer() as t:
            try:
                # Validate ligand atoms
                atom_report = self._validator.validate_atoms(ligand.atoms)
                if not atom_report.is_valid:
                    score.validation_status = ValidationStatus.FAILED
                    score.validation_issues = "; ".join(
                        str(e) for e in atom_report.errors
                    )
                    return score

                if atom_report.warnings:
                    score.validation_status = ValidationStatus.WARNING
                    score.validation_issues = "; ".join(
                        str(w) for w in atom_report.warnings
                    )

                # Count receptor contacts
                score.receptor_contacts = count_receptor_contacts(
                    protein_atoms, ligand.atoms
                )

                # Try to convert MOL2 atoms to SMILES for Boltz prediction
                smiles = self._mol2_to_smiles(ligand)
                if smiles is None:
                    # Fallback: infer from 3D coordinates
                    from boltz.affinity_rescoring.smiles_inference import (
                        infer_smiles_from_atoms,
                    )
                    smiles = infer_smiles_from_atoms(ligand.atoms)
                if smiles is None:
                    # Cannot score without SMILES
                    score.validation_status = ValidationStatus.FAILED
                    score.error_message = (
                        "Cannot convert MOL2 to SMILES. "
                        "Provide ligands with SMILES strings for scoring."
                    )
                    return score

                # Run Boltz prediction
                ligand_chain_id = "L"
                affinity_output = self._run_boltz_prediction(
                    sequences={protein_chain_id: protein_seq},
                    protein_chains=[protein_chain_id],
                    ligand_chains=[ligand_chain_id],
                    ligand_smiles={ligand_chain_id: smiles},
                )

                if affinity_output is not None:
                    score.affinity_score = affinity_output.get(
                        "affinity_pred_value", float("nan")
                    )
                    score.confidence = affinity_output.get(
                        "affinity_probability_binary", float("nan")
                    )

                    if "affinity_pred_value1" in affinity_output:
                        v1 = affinity_output["affinity_pred_value1"]
                        v2 = affinity_output.get("affinity_pred_value2", v1)
                        score.affinity_uncertainty = abs(v1 - v2) / 2.0

                    if score.validation_status != ValidationStatus.WARNING:
                        score.validation_status = ValidationStatus.SUCCESS
                else:
                    score.validation_status = ValidationStatus.FAILED
                    score.error_message = "Boltz prediction failed"

            except Exception as e:
                score.validation_status = ValidationStatus.FAILED
                score.error_message = str(e)

        score.processing_ms = t.elapsed_ms
        return score

    def _mol2_to_smiles(self, ligand: "LigandStructure") -> Optional[str]:
        """
        Attempt to convert a MOL2 ligand structure to SMILES.

        Uses RDKit if available.
        """
        try:
            from rdkit import Chem

            # Try to build molecule from atoms and bonds
            mol = Chem.RWMol()
            atom_map: Dict[int, int] = {}  # mol2_atom_id → rdkit_idx

            for atom in ligand.atoms:
                rd_atom = Chem.Atom(atom.element)
                idx = mol.AddAtom(rd_atom)
                atom_map[atom.index] = idx

            for a1, a2, btype in ligand.bonds:
                if a1 in atom_map and a2 in atom_map:
                    bond_type = {
                        1: Chem.BondType.SINGLE,
                        2: Chem.BondType.DOUBLE,
                        3: Chem.BondType.TRIPLE,
                        4: Chem.BondType.AROMATIC,
                        5: Chem.BondType.SINGLE,  # amide
                    }.get(btype, Chem.BondType.SINGLE)
                    try:
                        mol.AddBond(atom_map[a1], atom_map[a2], bond_type)
                    except Exception:
                        pass

            try:
                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                return smiles if smiles else None
            except Exception:
                # Try without sanitization
                smiles = Chem.MolToSmiles(mol)
                return smiles if smiles else None

        except ImportError:
            logger.warning(
                "RDKit not available. Cannot convert MOL2 to SMILES. "
                "Install via: pip install rdkit"
            )
            return None
        except Exception as e:
            logger.debug(f"SMILES conversion failed for {ligand.name}: {e}")
            return None

    # ─── Dry Run ──────────────────────────────────────────────────────────

    def dry_run(
        self,
        input_path: str | Path,
        protein_chain: Optional[str] = None,
        ligand_chains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate inputs without running inference.

        Useful for checking file validity, chain assignment,
        and expected processing before committing to a full run.

        Returns
        -------
        dict
            Validation summary including chains, atoms, issues
        """
        path = Path(input_path)
        report: Dict[str, Any] = {
            "file": str(path),
            "valid": False,
            "chains": {},
            "issues": [],
        }

        # File validation
        file_report = self._validator.validate_file(path)
        if not file_report.is_valid:
            report["issues"] = [str(e) for e in file_report.issues]
            return report

        # Parse
        try:
            atoms, metadata = parse_structure_file(path)
            report["metadata"] = metadata
        except Exception as e:
            report["issues"] = [f"Parse error: {e}"]
            return report

        # Atom validation
        atom_report = self._validator.validate_atoms(atoms)
        chain_report = self._validator.validate_chains(atoms)
        report["issues"] = [str(i) for i in atom_report.issues + chain_report.issues]

        # Chain identification
        assignment = self._chain_identifier.identify_chains(
            atoms,
            protein_chains=[protein_chain] if protein_chain else None,
            ligand_chains=ligand_chains,
        )

        report["chains"] = {
            "protein": assignment.protein_chains,
            "ligand": assignment.ligand_chains,
            "confidence": assignment.confidence,
            "reasoning": assignment.reasoning,
        }

        # Sequences (with SEQRES gap-filling)
        seqres = get_seqres_sequences(path)
        sequences = get_chain_sequences(atoms, reference_sequences=seqres)
        report["sequences"] = {k: len(v) for k, v in sequences.items()}
        if seqres:
            report["seqres_sequences"] = {k: len(v) for k, v in seqres.items()}

        # Counts
        report["total_atoms"] = len(atoms)
        for chain_id in set(a.chain_id for a in atoms):
            chain_atoms = [a for a in atoms if a.chain_id == chain_id]
            report["chains"].setdefault("details", {})[chain_id] = {
                "num_atoms": len(chain_atoms),
                "residue_types": len(set(a.residue_name for a in chain_atoms)),
            }

        report["valid"] = atom_report.is_valid
        return report
