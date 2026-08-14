"""
Multi-pocket structure prediction + affinity rescoring pipeline.

Runs the full Boltz-2 prediction with N copies of a single ligand against a
receptor, then extracts each receptor + ligand-copy pair as an individual
pocket structure and scores it through the AffinityRescorer.

Workflow:
    1. Validate inputs (receptor PDB, ligand SMILES, N).
    2. Identify protein chain & sequence.
    3. Assign N unique chain IDs for ligand copies.
    4. Write a Boltz YAML with protein + N ligand copies (no affinity property).
    5. Run `boltz predict` via subprocess.
    6. Read predicted structure + confidence JSON.
    7. Extract one PDB per ligand chain (protein + that copy).
    8. Rescore each pocket via AffinityRescorer.
    9. Emit CSV + HTML report.
"""

from __future__ import annotations

import json
import logging
import shutil
import string
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from boltz.affinity_rescoring.models import (
    DeviceOption,
    MultiPocketReport,
    PocketResult,
    ValidationStatus,
)
from boltz.affinity_rescoring.parsers import (
    get_chain_sequences,
    parse_structure_file,
)
from boltz.affinity_rescoring.validation import ChainIdentifier, StructureValidator

logger = logging.getLogger(__name__)

# Single-character chain ID pool (uppercase A-Z); used for ligand copies.
_CHAIN_ID_POOL = list(string.ascii_uppercase)
MAX_LIGAND_COPIES = 25  # 26 letters minus protein chain


class MultiPocketPipeline:
    """Predict N binding poses of a ligand against a receptor and score each.

    Parameters
    ----------
    affinity_checkpoint : str
        Path to affinity checkpoint, or 'auto' to auto-download.
    structure_checkpoint : str, optional
        Path to Boltz-2 structure checkpoint, passed via ``--checkpoint`` to
        ``boltz predict``. ``None`` lets Boltz auto-download.
    device : str
        Inference device: 'auto' | 'cuda' | 'cpu' | 'mps'.
    cache_dir : str, optional
        Boltz cache directory (forwarded to both prediction and rescoring).
    """

    def __init__(
        self,
        affinity_checkpoint: str = "auto",
        structure_checkpoint: Optional[str] = None,
        device: str = "auto",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.affinity_checkpoint = affinity_checkpoint
        self.structure_checkpoint = structure_checkpoint
        self.device = device
        self.cache_dir = cache_dir

        self._validator = StructureValidator()
        self._chain_identifier = ChainIdentifier()

    # ─── Public API ───────────────────────────────────────────────────────

    def run(
        self,
        receptor: Optional[str | Path] = None,
        ligand_smiles: str = "",
        n_pockets: int = 5,
        output_dir: str | Path = "./mp_results",
        protein_chain: Optional[str] = None,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        sort_by: str = "affinity_pred",
        ascending: bool = False,
        keep_boltz_outputs: bool = True,
        reference_sequence: Optional[str] = None,
        receptor_sequence: Optional[str] = None,
        msa_path: Optional[str | Path] = None,
    ) -> MultiPocketReport:
        """Run the full multi-pocket pipeline.

        Parameters
        ----------
        receptor : str or Path, optional
            Receptor PDB/CIF file.  Either ``receptor`` or
            ``receptor_sequence`` must be supplied.
        ligand_smiles : str
            SMILES of the ligand to dock in N copies.
        n_pockets : int
            Number of independent ligand copies to predict (1..25).
        output_dir : str or Path
            Directory for results (created if absent).
        protein_chain : str, optional
            Protein chain ID.  Required when using ``receptor_sequence``
            (defaults to ``'A'`` if not provided).  Auto-detected from
            the PDB when ``receptor`` is given.
        recycling_steps, sampling_steps : int
            Boltz prediction hyperparameters.
        sort_by : str
            Column for ranking pockets in the CSV/HTML.
        ascending : bool
            Sort direction (default False = best score first when score
            is e.g. binding probability or pKd).
        keep_boltz_outputs : bool
            If False, the temporary Boltz prediction directory is removed
            after extraction.
        reference_sequence : str, optional
            Full protein sequence override for PDB inputs that have
            missing loops / incomplete SEQRES records.
        receptor_sequence : str, optional
            Plain amino-acid sequence of the receptor.  When provided,
            no PDB file is needed — Boltz will fold the receptor from
            scratch together with the ligand copies.  Ignored if
            ``receptor`` is also given.
        msa_path : str or Path, optional
            Path to a pre-computed MSA file (``.a3m`` or ``.csv``) for
            the receptor.  When provided, the path is injected into
            both the structure-prediction YAML and the per-pocket
            affinity-rescoring YAML so the pipeline never contacts the
            MSA server.  If ``None``, the MSA is generated on-the-fly
            using :func:`~boltz.affinity_rescoring.mmseqs2.precompute_msa`
            (ColabFold MMseqs2 server) and saved to
            ``<output_dir>/receptor_msa.a3m``.
        """
        if receptor is None and receptor_sequence is None:
            raise ValueError(
                "Provide either 'receptor' (PDB/CIF file) or "
                "'receptor_sequence' (amino-acid string)."
            )

        # Validate msa_path early if provided; auto-generation happens
        # after protein_seq is determined below.
        if msa_path is not None:
            msa_path = Path(msa_path).resolve()
            if not msa_path.exists():
                raise FileNotFoundError(f"MSA file not found: {msa_path}")

        t0 = time.time()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        structures_dir = out_dir / "structures"
        structures_dir.mkdir(exist_ok=True)
        boltz_out_dir = out_dir / "boltz_prediction"
        boltz_out_dir.mkdir(exist_ok=True)

        # ── Resolve receptor source ───────────────────────────────────────
        sequence_only = receptor is None
        receptor_path: Optional[Path] = None

        if sequence_only:
            self._validate_smiles_and_n(ligand_smiles, n_pockets)
            protein_chain_id = protein_chain or "A"
            protein_seq = receptor_sequence
            receptor_label = "sequence-only"
            target_id = _sanitize_id(f"seq_{protein_chain_id}") + f"_mp{n_pockets}"
        else:
            receptor_path = Path(receptor).resolve()
            self._validate_inputs(receptor_path, ligand_smiles, n_pockets)
            protein_chain_id, protein_seq = self._extract_protein_info(
                receptor_path, protein_chain, reference_sequence
            )
            receptor_label = str(receptor_path)
            target_id = _sanitize_id(receptor_path.stem) + f"_mp{n_pockets}"

        # ── Auto-generate MSA if not supplied ────────────────────────────────
        if msa_path is None:
            from boltz.affinity_rescoring.mmseqs2 import precompute_msa
            msa_auto_path = out_dir / "receptor_msa.a3m"
            logger.info(
                "No msa_path provided — generating MSA via ColabFold "
                "MMseqs2 server and caching at %s",
                msa_auto_path,
            )
            msa_path = Path(precompute_msa(protein_seq, msa_auto_path))

        # Assign chain IDs for ligand copies (avoiding collision with protein chain)
        ligand_chain_ids = self._assign_ligand_chains(protein_chain_id, n_pockets)
        logger.info(
            "Multi-pocket: protein chain=%s, ligand copies=%s",
            protein_chain_id,
            ligand_chain_ids,
        )

        # Write Boltz YAML (with pre-computed MSA path injected)
        yaml_path = out_dir / f"{target_id}.yaml"
        self._write_boltz_yaml(
            yaml_path=yaml_path,
            protein_chain=protein_chain_id,
            protein_sequence=protein_seq,
            ligand_chain_ids=ligand_chain_ids,
            ligand_smiles=ligand_smiles,
            msa_path=msa_path,
        )

        # Run Boltz prediction (no --use_msa_server — MSA is in the YAML)
        logger.info("Running boltz predict (target=%s)...", target_id)
        self._run_boltz_predict(
            yaml_path=yaml_path,
            out_dir=boltz_out_dir,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
        )

        # Locate prediction outputs
        pred_dir = boltz_out_dir / f"boltz_results_{target_id}" / "predictions" / target_id
        if not pred_dir.exists():
            # Fallback: some boltz versions skip the boltz_results_ prefix
            alt = boltz_out_dir / "predictions" / target_id
            if alt.exists():
                pred_dir = alt
            else:
                raise RuntimeError(
                    f"Boltz prediction directory not found. Expected {pred_dir}"
                )

        model_pdb = self._find_model_pdb(pred_dir, target_id)
        confidence_json = pred_dir / f"confidence_{target_id}_model_0.json"
        confidence_data = (
            json.loads(confidence_json.read_text())
            if confidence_json.exists()
            else {}
        )

        # Extract per-pocket PDBs
        pocket_pdbs = self._extract_pocket_pdbs(
            model_pdb=model_pdb,
            protein_chain=protein_chain_id,
            ligand_chain_ids=ligand_chain_ids,
            output_dir=structures_dir,
            target_id=target_id,
        )

        # Rescore each pocket
        from boltz.affinity_rescoring.rescorer import AffinityRescorer

        rescorer = AffinityRescorer(
            checkpoint=self.affinity_checkpoint,
            device=self.device,
            cache_dir=self.cache_dir,
        )

        pockets: List[PocketResult] = []
        for idx, (chain_id, pdb_path) in enumerate(pocket_pdbs):
            logger.info("Scoring pocket %d (chain %s)...", idx, chain_id)
            pocket = self._score_pocket(
                rescorer=rescorer,
                pocket_id=idx,
                chain_id=chain_id,
                pdb_path=pdb_path,
                protein_chain=protein_chain_id,
                ligand_smiles=ligand_smiles,
                confidence_data=confidence_data,
                confidence_json_path=confidence_json,
                msa_path=msa_path,
            )
            pockets.append(pocket)

        # Sort pockets for the report
        pockets_sorted = sorted(
            pockets,
            key=lambda p: (
                float("inf") if _is_nan(getattr(p, sort_by, float("nan"))) else getattr(p, sort_by)
            ),
            reverse=not ascending,
        )

        # Export CSV + HTML
        csv_path = out_dir / "multipocket_scores.csv"
        html_path = out_dir / "multipocket_report.html"
        _write_csv(pockets_sorted, csv_path)

        from boltz.affinity_rescoring.export import generate_multipocket_html

        generate_multipocket_html(
            html_path=html_path,
            receptor=receptor_label,
            ligand_smiles=ligand_smiles,
            protein_chain=protein_chain_id,
            n_requested=n_pockets,
            pockets=pockets_sorted,
            sort_by=sort_by,
        )

        if not keep_boltz_outputs:
            shutil.rmtree(boltz_out_dir, ignore_errors=True)

        report = MultiPocketReport(
            receptor=receptor_label,
            ligand_smiles=ligand_smiles,
            n_pockets_requested=n_pockets,
            n_pockets_extracted=len(pockets),
            protein_chain=protein_chain_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            pockets=pockets_sorted,
            csv_path=str(csv_path),
            html_path=str(html_path),
            structures_dir=str(structures_dir),
            boltz_prediction_dir=str(boltz_out_dir) if keep_boltz_outputs else "",
            total_time_s=time.time() - t0,
        )
        logger.info(
            "Multi-pocket pipeline complete: %d pockets in %.1fs",
            len(pockets),
            report.total_time_s,
        )
        return report

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _validate_smiles_and_n(self, ligand_smiles: str, n_pockets: int) -> None:
        if not (1 <= n_pockets <= MAX_LIGAND_COPIES):
            raise ValueError(
                f"n_pockets must be in [1, {MAX_LIGAND_COPIES}]; got {n_pockets}"
            )
        try:
            from rdkit import Chem
        except ImportError as e:  # pragma: no cover
            raise ImportError("RDKit is required for SMILES validation") from e
        mol = Chem.MolFromSmiles(ligand_smiles)
        if mol is None:
            raise ValueError(f"Invalid ligand SMILES: {ligand_smiles!r}")

    def _validate_inputs(
        self, receptor: Path, ligand_smiles: str, n_pockets: int
    ) -> None:
        if not receptor.exists():
            raise FileNotFoundError(f"Receptor not found: {receptor}")
        self._validate_smiles_and_n(ligand_smiles, n_pockets)

    def _extract_protein_info(
        self,
        receptor: Path,
        protein_chain: Optional[str],
        reference_sequence: Optional[str],
    ) -> Tuple[str, str]:
        atoms, _ = parse_structure_file(receptor)
        chain_assignment = self._chain_identifier.identify_chains(
            atoms,
            protein_chains=[protein_chain] if protein_chain else None,
            ligand_chains=[],
        )
        if not chain_assignment.protein_chains:
            raise ValueError(
                "No protein chain detected. Provide --protein-chain explicitly."
            )
        chain_id = chain_assignment.protein_chains[0]

        ref = {chain_id: reference_sequence} if reference_sequence else None
        sequences = get_chain_sequences(atoms, reference_sequences=ref)
        seq = sequences.get(chain_id, "")
        if not seq:
            raise ValueError(
                f"Could not extract sequence for protein chain {chain_id}."
            )
        return chain_id, seq

    @staticmethod
    def _assign_ligand_chains(protein_chain: str, n: int) -> List[str]:
        pool = [c for c in _CHAIN_ID_POOL if c != protein_chain]
        if n > len(pool):
            raise ValueError(
                f"Cannot allocate {n} ligand chain IDs (pool size {len(pool)})."
            )
        return pool[:n]

    @staticmethod
    def _write_boltz_yaml(
        yaml_path: Path,
        protein_chain: str,
        protein_sequence: str,
        ligand_chain_ids: List[str],
        ligand_smiles: str,
        msa_path: Optional[Path] = None,
    ) -> None:
        # Hand-write YAML to avoid an extra dependency on PyYAML at import time
        # of this module (Boltz already requires it transitively, but we keep
        # output minimal and predictable).
        lines = [
            "version: 1",
            "sequences:",
            "  - protein:",
            f"      id: {protein_chain}",
            f"      sequence: {protein_sequence}",
        ]
        if msa_path is not None:
            lines.append(f"      msa: {Path(msa_path).resolve()}")
        if len(ligand_chain_ids) == 1:
            lines += [
                "  - ligand:",
                f"      id: {ligand_chain_ids[0]}",
                f"      smiles: '{ligand_smiles}'",
            ]
        else:
            ids_yaml = "[" + ", ".join(ligand_chain_ids) + "]"
            lines += [
                "  - ligand:",
                f"      id: {ids_yaml}",
                f"      smiles: '{ligand_smiles}'",
            ]
        yaml_path.write_text("\n".join(lines) + "\n")
        logger.debug("Wrote multi-pocket YAML: %s", yaml_path)

    def _run_boltz_predict(
        self,
        yaml_path: Path,
        out_dir: Path,
        recycling_steps: int,
        sampling_steps: int,
    ) -> None:
        cmd = [
            sys.executable,
            "-m",
            "boltz.main",
            "predict",
            str(yaml_path),
            "--out_dir",
            str(out_dir),
            "--model",
            "boltz2",
            "--output_format",
            "pdb",
            "--recycling_steps",
            str(recycling_steps),
            "--sampling_steps",
            str(sampling_steps),
            "--diffusion_samples",
            "1",
            "--override",
        ]
        if self.structure_checkpoint:
            cmd += ["--checkpoint", str(self.structure_checkpoint)]
        if self.cache_dir:
            cmd += ["--cache", str(self.cache_dir)]
        if self.device == "cpu":
            cmd += ["--accelerator", "cpu"]
        # NOTE: --use_msa_server is intentionally NEVER added.  The MSA
        # path must be present in the YAML (run() validates this).

        logger.info("Boltz predict command: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error("Boltz predict failed.\nSTDOUT:\n%s\nSTDERR:\n%s",
                         proc.stdout, proc.stderr)
            raise RuntimeError(
                f"boltz predict exited with code {proc.returncode}. "
                f"See logs for details."
            )

    @staticmethod
    def _find_model_pdb(pred_dir: Path, target_id: str) -> Path:
        candidates = [
            pred_dir / f"{target_id}_model_0.pdb",
            pred_dir / f"{target_id}_model_0.cif",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Last resort: any model_0
        for p in pred_dir.glob(f"{target_id}_model_0.*"):
            return p
        raise FileNotFoundError(
            f"No model_0 prediction file found in {pred_dir}"
        )

    @staticmethod
    def _extract_pocket_pdbs(
        model_pdb: Path,
        protein_chain: str,
        ligand_chain_ids: List[str],
        output_dir: Path,
        target_id: str,
    ) -> List[Tuple[str, Path]]:
        """Use gemmi to write one PDB per (protein + single ligand chain)."""
        import gemmi

        structure = gemmi.read_structure(str(model_pdb))
        results: List[Tuple[str, Path]] = []

        for idx, lig_chain in enumerate(ligand_chain_ids):
            new_struct = gemmi.Structure()
            new_struct.name = f"{target_id}_pocket_{idx:02d}_{lig_chain}"
            new_model = gemmi.Model("1")

            kept_chains: List[str] = []
            for orig_model in structure:
                for chain in orig_model:
                    if chain.name in (protein_chain, lig_chain):
                        new_model.add_chain(chain.clone())
                        kept_chains.append(chain.name)
                break  # only first model

            if protein_chain not in kept_chains or lig_chain not in kept_chains:
                logger.warning(
                    "Pocket %s: missing chain (kept=%s, expected=[%s, %s])",
                    idx, kept_chains, protein_chain, lig_chain,
                )

            new_struct.add_model(new_model)
            out_pdb = output_dir / f"pocket_{idx:02d}_{lig_chain}.pdb"
            new_struct.write_pdb(str(out_pdb))
            results.append((lig_chain, out_pdb))

        return results

    def _score_pocket(
        self,
        rescorer,
        pocket_id: int,
        chain_id: str,
        pdb_path: Path,
        protein_chain: str,
        ligand_smiles: str,
        confidence_data: dict,
        confidence_json_path: Path,
        msa_path: Optional[Path] = None,
    ) -> PocketResult:
        pocket = PocketResult(
            pocket_id=pocket_id,
            chain_id=chain_id,
            structure_path=str(pdb_path),
            confidence_json_path=str(confidence_json_path) if confidence_json_path.exists() else "",
        )

        # Pull per-chain confidence numbers from the full-prediction JSON.
        try:
            pair_iptm = confidence_data.get("pair_chains_iptm", {})
            # Keys may be strings or ints depending on serialization
            row = pair_iptm.get(protein_chain) or pair_iptm.get(str(protein_chain)) or {}
            pocket.interface_iptm = float(
                row.get(chain_id, row.get(str(chain_id), float("nan")))
            )
            pocket.boltz_confidence_score = float(
                confidence_data.get("confidence_score", float("nan"))
            )
        except (TypeError, ValueError):
            pass

        try:
            msa_paths = (
                {protein_chain: str(Path(msa_path).resolve())}
                if msa_path is not None
                else None
            )
            result = rescorer.rescore_pdb(
                pdb_path,
                protein_chain=protein_chain,
                ligand_chains=[chain_id],
                ligand_smiles={chain_id: ligand_smiles},
                msa_paths=msa_paths,
            )
            pocket.affinity_pred = result.affinity_pred
            pocket.affinity_std = result.affinity_std
            pocket.affinity_probability_binary = result.affinity_probability_binary
            pocket.n_ligand_atoms = result.ligand_atom_count
            pocket.validation_status = result.validation_status
            if result.validation_status == ValidationStatus.FAILED:
                pocket.error_message = result.error_message
        except Exception as e:  # noqa: BLE001
            logger.exception("Scoring failed for pocket %d (%s): %s", pocket_id, chain_id, e)
            pocket.validation_status = ValidationStatus.FAILED
            pocket.error_message = str(e)

        return pocket


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _sanitize_id(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return safe or "target"


def _is_nan(x) -> bool:
    try:
        return x != x  # NaN-only property
    except Exception:  # noqa: BLE001
        return False


def _write_csv(pockets: List[PocketResult], path: Path) -> None:
    import csv

    if not pockets:
        path.write_text("")
        return
    fields = list(pockets[0].to_dict().keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in pockets:
            writer.writerow(p.to_dict())
