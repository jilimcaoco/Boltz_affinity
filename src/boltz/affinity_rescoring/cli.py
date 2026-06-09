"""
CLI commands for affinity rescoring.

Adds the `boltz rescore` command group to the existing Boltz CLI.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click


def setup_logging(log_level: str) -> None:
    """Configure logging for the rescoring module."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _guard_msa_server_flag(use_msa_server: bool) -> None:
    """Reject ``--use-msa-server`` early with the canonical error message.

    The ColabFold public MSA server is disabled in this fork — see
    :mod:`boltz.affinity_rescoring.msa_cache` for the precompute workflow.
    Each ``rescore *`` subcommand calls this guard so the error surface is
    identical regardless of which subcommand was invoked.
    """
    if not use_msa_server:
        return
    from boltz.affinity_rescoring.msa_cache import raise_msa_server_disabled
    try:
        raise_msa_server_disabled()
    except Exception as exc:  # noqa: BLE001
        raise click.UsageError(str(exc)) from exc


def _guard_lora_finetune_flags(
    use_lora: Optional[str], use_finetune: Optional[str],
) -> None:
    """Reject combining ``--use-lora`` with ``--use-finetune``.

    The two are different fine-tuning regimes (low-rank residual vs full
    weight update) and stacking them is not supported. Surface a clear
    UsageError before any model is loaded.
    """
    if use_lora and use_finetune:
        msg = (
            "--use-lora and --use-finetune are mutually exclusive; pick "
            "one. (Adapter composition is not supported.)"
        )
        raise click.UsageError(msg)


@click.group("rescore")
def rescore_cli():
    """Protein-ligand affinity rescoring using the Boltz-2 affinity module."""
    pass


# ─── PDB Command ─────────────────────────────────────────────────────────────


@rescore_cli.command("pdb")
@click.option(
    "--input", "-i", "input_path", required=True,
    type=click.Path(exists=True),
    help="Path to PDB or CIF structure file.",
)
@click.option(
    "--output", "-o", "output_path", default=None,
    type=click.Path(),
    help="Output file path. Defaults to <input_stem>_affinity.<format>.",
)
@click.option(
    "--protein-chain", default=None,
    help="Protein chain ID (auto-detected if not provided).",
)
@click.option(
    "--ligand-chains", default=None,
    help="Comma-separated ligand chain IDs (auto-detected if not provided).",
)
@click.option(
    "--ligand-smiles", default=None,
    help='Ligand SMILES as JSON: \'{"B": "CCO"}\'. '
         'Optional — SMILES are auto-inferred from coordinates if not provided.',
)
@click.option(
    "--output-format", default="json",
    type=click.Choice(["json", "csv", "parquet", "sqlite", "excel"]),
    help="Output format.",
)
@click.option(
    "--device", default="auto",
    type=click.Choice(["auto", "cuda", "cpu", "mps"]),
    help="Device for inference.",
)
@click.option(
    "--validation", default="moderate",
    type=click.Choice(["strict", "moderate", "lenient"]),
    help="Validation strictness level.",
)
@click.option(
    "--checkpoint", default="auto",
    help="Checkpoint path or 'auto' to download.",
)
@click.option(
    "--dry-run", is_flag=True, default=False,
    help="Validate inputs without running inference.",
)
@click.option(
    "--use-msa-server", is_flag=True, default=False,
    help="[DISABLED in this fork] Pre-compute MSAs with "
         "`python -m boltz.affinity_rescoring.mmseqs2` and point "
         "--msa-directory (or $BOLTZ_MSA_CACHE_DIR) at the cache instead.",
)
@click.option(
    "--msa-directory", default=None,
    type=click.Path(exists=False),
    help="Directory with pre-computed MSA files (.a3m or .csv). "
         "Accepts canonical hash files (<sha256>.a3m) or legacy "
         "<target>_<chain>.a3m / <chain>.a3m names. "
         "$BOLTZ_MSA_CACHE_DIR is also searched.",
)
@click.option(
    "--reference-sequence", default=None,
    help='Full biological sequence(s) as JSON: \'{"A": "MKTL..."}\'. '
         'Overrides SEQRES and ATOM-derived sequences to handle '
         'missing loops / incomplete structures.',
)
@click.option(
    "--recycling-steps", default=None, type=int,
    help="Trunk recycling steps (default: 3). Lower = faster, less accurate.",
)
@click.option(
    "--fast", is_flag=True, default=False,
    help="Ultra-fast mode: 1 recycling step. ~3x faster, minor accuracy trade-off.",
)
@click.option(
    "--use-lora", "use_lora", default=None,
    help="Name of a registered LoRA adapter (or path to an adapter directory) "
         "to apply to the affinity model before scoring.",
)
@click.option(
    "--use-finetune", "use_finetune", default=None,
    help="Name of a registered full fine-tune (or path to its directory) "
         "to apply to the affinity model before scoring. "
         "Mutually exclusive with --use-lora.",
)
@click.option(
    "--log-level", default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level.",
)
def rescore_pdb(
    input_path: str,
    output_path: Optional[str],
    protein_chain: Optional[str],
    ligand_chains: Optional[str],
    ligand_smiles: Optional[str],
    output_format: str,
    device: str,
    validation: str,
    checkpoint: str,
    dry_run: bool,
    use_msa_server: bool,
    msa_directory: Optional[str],
    reference_sequence: Optional[str],
    recycling_steps: Optional[int],
    fast: bool,
    use_lora: Optional[str],
    use_finetune: Optional[str],
    log_level: str,
):
    """Rescore a single PDB/CIF protein-ligand complex."""
    setup_logging(log_level)
    _guard_msa_server_flag(use_msa_server)
    _guard_lora_finetune_flags(use_lora, use_finetune)

    from boltz.affinity_rescoring import AffinityRescorer

    rescorer = AffinityRescorer(
        checkpoint=checkpoint,
        device=device,
        validation_level=validation,
        recycling_steps=recycling_steps,
        fast=fast,
        lora=use_lora,
        finetune=use_finetune,
    )

    # Parse ligand chains
    lig_chains = None
    if ligand_chains:
        lig_chains = [c.strip() for c in ligand_chains.split(",")]

    if dry_run:
        report = rescorer.dry_run(
            input_path,
            protein_chain=protein_chain,
            ligand_chains=lig_chains,
        )
        click.echo(json.dumps(report, indent=2))
        return

    # Parse ligand SMILES
    smiles_dict = None
    if ligand_smiles:
        try:
            smiles_dict = json.loads(ligand_smiles)
        except json.JSONDecodeError:
            click.echo(
                f"Error: --ligand-smiles must be valid JSON. "
                f'Example: \'{{"B": "CCO"}}\'',
                err=True,
            )
            sys.exit(1)

    # Parse reference sequences
    ref_seqs = None
    if reference_sequence:
        try:
            ref_seqs = json.loads(reference_sequence)
        except json.JSONDecodeError:
            click.echo(
                f"Error: --reference-sequence must be valid JSON. "
                f'Example: \'{{"A": "MKTL..."}}\'',
                err=True,
            )
            sys.exit(1)

    # Default output path
    if output_path is None:
        suffix = {"json": ".json", "csv": ".csv", "parquet": ".parquet",
                  "sqlite": ".db", "excel": ".xlsx"}
        output_path = f"{Path(input_path).stem}_affinity{suffix.get(output_format, '.json')}"

    result = rescorer.rescore_pdb(
        input_path,
        protein_chain=protein_chain,
        ligand_chains=lig_chains,
        output_path=output_path,
        output_format=output_format,
        ligand_smiles=smiles_dict,
        use_msa_server=use_msa_server,
        msa_directory=msa_directory,
        reference_sequences=ref_seqs,
    )

    # Print summary
    if result.validation_status.value == "SUCCESS":
        click.echo(
            f"Affinity: {result.affinity_pred:.4f} "
            f"(probability: {result.affinity_probability_binary:.4f})"
        )
        if result.affinity_std and result.affinity_std > 0:
            click.echo(f"Uncertainty: ±{result.affinity_std:.4f}")
    elif result.validation_status.value == "WARNING":
        click.echo(f"Affinity: {result.affinity_pred:.4f} (with warnings)")
        for w in result.warnings:
            click.echo(f"  Warning: {w}", err=True)
    else:
        click.echo(f"FAILED: {result.error_message}", err=True)
        sys.exit(1)

    click.echo(f"Results saved to: {output_path}")


# ─── Batch Command ────────────────────────────────────────────────────────────


@rescore_cli.command("batch")
@click.option(
    "--input-dir", required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing PDB/CIF files.",
)
@click.option(
    "--output", "-o", "output_path", default="batch_results.csv",
    help="Output file path.",
)
@click.option(
    "--output-format", default="csv",
    type=click.Choice(["json", "csv", "parquet", "sqlite", "excel"]),
    help="Output format.",
)
@click.option(
    "--recursive", is_flag=True, default=False,
    help="Scan subdirectories recursively.",
)
@click.option(
    "--device", default="auto",
    type=click.Choice(["auto", "cuda", "cpu", "mps"]),
)
@click.option(
    "--validation", default="moderate",
    type=click.Choice(["strict", "moderate", "lenient"]),
)
@click.option(
    "--checkpoint", default="auto",
)
@click.option(
    "--ligand-smiles", default=None,
    help='Ligand SMILES as JSON (applied to all complexes). '
         'Optional — auto-inferred from coordinates if not provided.',
)
@click.option(
    "--smiles-csv", "smiles_csv", default=None,
    type=click.Path(exists=True),
    help="CSV file with per-compound SMILES (columns: 'name' and 'smiles'). "
         "The 'name' column must match PDB file stems.  Per-compound entries "
         "override --ligand-smiles and bypass coordinate-based SMILES "
         "auto-inference, which is the most common source of rescoring "
         "failures when working with boltz-predicted PDB poses.",
)
@click.option(
    "--use-msa-server", is_flag=True, default=False,
    help="[DISABLED in this fork] See `boltz.affinity_rescoring.msa_cache`.",
)
@click.option(
    "--msa-directory", default=None,
    type=click.Path(exists=False),
    help="Directory with pre-computed MSA files (.a3m or .csv). "
         "Used for every complex in the batch. Accepts canonical "
         "hash files (<sha256>.a3m) or legacy <target>_<chain>.a3m "
         "/ <chain>.a3m names. $BOLTZ_MSA_CACHE_DIR is also searched.",
)
@click.option(
    "--recycling-steps", default=None, type=int,
    help="Trunk recycling steps (default: 3). Lower = faster, less accurate.",
)
@click.option(
    "--fast", is_flag=True, default=False,
    help="Ultra-fast mode: 1 recycling step. ~3x faster, minor accuracy trade-off.",
)
@click.option(
    "--use-lora", "use_lora", default=None,
    help="LoRA adapter name (or path) to apply before scoring.",
)
@click.option(
    "--use-finetune", "use_finetune", default=None,
    help="Full fine-tune name (or path) to apply before scoring. "
         "Mutually exclusive with --use-lora.",
)
@click.option(
    "--log-level", default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
def rescore_batch(
    input_dir: str,
    output_path: str,
    output_format: str,
    recursive: bool,
    device: str,
    validation: str,
    checkpoint: str,
    ligand_smiles: Optional[str],
    smiles_csv: Optional[str],
    use_msa_server: bool,
    msa_directory: Optional[str],
    recycling_steps: Optional[int],
    fast: bool,
    use_lora: Optional[str],
    use_finetune: Optional[str],
    log_level: str,
):
    """Rescore all PDB/CIF files in a directory."""
    setup_logging(log_level)
    _guard_msa_server_flag(use_msa_server)
    _guard_lora_finetune_flags(use_lora, use_finetune)

    from boltz.affinity_rescoring import AffinityRescorer
    from boltz.affinity_rescoring.export import compute_batch_summary

    rescorer = AffinityRescorer(
        checkpoint=checkpoint,
        device=device,
        validation_level=validation,
        recycling_steps=recycling_steps,
        fast=fast,
        lora=use_lora,
        finetune=use_finetune,
    )

    smiles_dict = None
    if ligand_smiles:
        try:
            smiles_dict = json.loads(ligand_smiles)
        except json.JSONDecodeError:
            click.echo("Error: --ligand-smiles must be valid JSON.", err=True)
            sys.exit(1)

    # Build per-compound SMILES lookup from CSV (name→smiles).
    # This is the primary fix for PDB rescoring failures: the labels CSV
    # produced by prepare_validation_inputs.py already has canonical SMILES
    # for every compound.  Passing it here completely bypasses RDKit
    # coordinate-based SMILES auto-inference.
    compound_smiles: Optional[dict] = None
    if smiles_csv:
        import csv as _csv
        compound_smiles = {}
        with open(smiles_csv, newline="") as fh:
            reader = _csv.DictReader(fh)
            if reader.fieldnames is None or "name" not in reader.fieldnames or "smiles" not in reader.fieldnames:
                click.echo(
                    "Error: --smiles-csv must have 'name' and 'smiles' columns. "
                    f"Found: {reader.fieldnames}",
                    err=True,
                )
                sys.exit(1)
            for row in reader:
                name = (row.get("name") or "").strip()
                smi  = (row.get("smiles") or "").strip()
                if name and smi:
                    compound_smiles[name] = smi
        click.echo(f"[smiles-csv] loaded {len(compound_smiles)} SMILES entries from {smiles_csv}")

    results = rescorer.rescore_directory(
        input_dir,
        output_path=output_path,
        output_format=output_format,
        recursive=recursive,
        ligand_smiles=smiles_dict,
        compound_smiles=compound_smiles,
        use_msa_server=use_msa_server,
        msa_directory=msa_directory,
    )

    # Print summary
    summary = compute_batch_summary(results)
    click.echo(f"\n{'='*50}")
    click.echo(f"Batch Rescoring Summary")
    click.echo(f"{'='*50}")
    click.echo(f"Total processed:  {summary.total_processed}")
    click.echo(f"Successful:       {summary.successful}")
    click.echo(f"Failed:           {summary.failed}")
    if not all(
        x != x for x in [summary.mean_affinity]  # NaN check
    ):
        click.echo(f"Mean affinity:    {summary.mean_affinity:.4f}")
        click.echo(f"Std affinity:     {summary.std_affinity:.4f}")
    click.echo(f"Total time:       {summary.inference_time_total_s:.1f}s")
    if summary.throughput_complexes_per_second > 0:
        click.echo(f"Throughput:       {summary.throughput_complexes_per_second:.2f} complexes/s")
    click.echo(f"Results saved to: {output_path}")


# ─── Receptor Command ────────────────────────────────────────────────────────


@rescore_cli.command("receptor")
@click.option(
    "--receptor", required=True,
    type=click.Path(exists=True),
    help="Path to receptor PDB/CIF file.",
)
@click.option(
    "--ligands", required=True,
    type=click.Path(exists=True),
    help="Path to MOL2 file with ligands.",
)
@click.option(
    "--output", "-o", "output_path", default="scores.csv",
    help="Output file path.",
)
@click.option(
    "--protein-chain", default=None,
    help="Protein chain ID in receptor file.",
)
@click.option(
    "--output-format", default="csv",
    type=click.Choice(["json", "csv", "parquet", "excel"]),
    help="Output format.",
)
@click.option(
    "--device", default="auto",
    type=click.Choice(["auto", "cuda", "cpu", "mps"]),
)
@click.option(
    "--validation", default="moderate",
    type=click.Choice(["strict", "moderate", "lenient"]),
)
@click.option(
    "--checkpoint", default="auto",
)
@click.option(
    "--sort-by", default="affinity_score",
    type=click.Choice(["affinity_score", "confidence", "ligand_name", "n_atoms"]),
    help="Sort results by this column.",
)
@click.option(
    "--reference-sequence", default=None,
    help='Full biological sequence(s) as JSON: \'{"A": "MKTL..."}\'. '
         'Overrides SEQRES and ATOM-derived sequences.',
)
@click.option(
    "--use-msa-server", is_flag=True, default=False,
    help="[DISABLED in this fork] Pre-compute MSAs and pass them via "
         "--msa-directory (or $BOLTZ_MSA_CACHE_DIR).",
)
@click.option(
    "--msa-directory", default=None,
    type=click.Path(exists=False),
    help="Directory with pre-computed MSA files (.a3m or .csv), or "
         "a directory in which to cache server-generated MSAs. "
         "Ideal for HPC nodes without internet access. "
         "Pre-compute on a login node, then pass the directory here.",
)
@click.option(
    "--recycling-steps", default=None, type=int,
    help="Trunk recycling steps (default: 3). Lower = faster, less accurate.",
)
@click.option(
    "--fast", is_flag=True, default=False,
    help="Ultra-fast mode: 1 recycling step. ~3x faster. Recommended for "
         "initial large-scale screening; re-score top hits at default.",
)
@click.option(
    "--use-lora", "use_lora", default=None,
    help="LoRA adapter name (or path) to apply before scoring.",
)
@click.option(
    "--use-finetune", "use_finetune", default=None,
    help="Full fine-tune name (or path) to apply before scoring. "
         "Mutually exclusive with --use-lora.",
)
@click.option(
    "--log-level", default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
def rescore_receptor(
    receptor: str,
    ligands: str,
    output_path: str,
    protein_chain: Optional[str],
    output_format: str,
    device: str,
    validation: str,
    checkpoint: str,
    sort_by: str,
    reference_sequence: Optional[str],
    use_msa_server: bool,
    msa_directory: Optional[str],
    recycling_steps: Optional[int],
    fast: bool,
    use_lora: Optional[str],
    use_finetune: Optional[str],
    log_level: str,
):
    """Score a receptor against multiple ligands from MOL2 file."""
    setup_logging(log_level)
    _guard_msa_server_flag(use_msa_server)
    _guard_lora_finetune_flags(use_lora, use_finetune)

    from boltz.affinity_rescoring import AffinityRescorer

    rescorer = AffinityRescorer(
        checkpoint=checkpoint,
        device=device,
        validation_level=validation,
        recycling_steps=recycling_steps,
        fast=fast,
        lora=use_lora,
        finetune=use_finetune,
    )

    # Parse reference sequences
    ref_seqs = None
    if reference_sequence:
        try:
            ref_seqs = json.loads(reference_sequence)
        except json.JSONDecodeError:
            click.echo(
                f"Error: --reference-sequence must be valid JSON. "
                f'Example: \'{{"A": "MKTL..."}}\'',
                err=True,
            )
            sys.exit(1)

    scores = rescorer.rescore_receptor(
        receptor_path=receptor,
        ligands_path=ligands,
        protein_chain=protein_chain,
        output_path=output_path,
        output_format=output_format,
        sort_by=sort_by,
        reference_sequences=ref_seqs,
        use_msa_server=use_msa_server,
        msa_directory=msa_directory,
    )

    # Print summary
    successful = sum(1 for s in scores if s.validation_status.value == "SUCCESS")
    failed = sum(1 for s in scores if s.validation_status.value == "FAILED")

    click.echo(f"\n{'='*50}")
    click.echo(f"Receptor Rescoring Summary")
    click.echo(f"{'='*50}")
    click.echo(f"Total ligands:  {len(scores)}")
    click.echo(f"Successful:     {successful}")
    click.echo(f"Failed:         {failed}")

    # Top hits
    import math
    scored = [s for s in scores if not math.isnan(s.affinity_score)]
    if scored:
        scored.sort(key=lambda s: s.affinity_score)
        click.echo(f"\nTop 5 hits:")
        for s in scored[:5]:
            click.echo(
                f"  {s.ligand_name:30s} "
                f"score={s.affinity_score:8.4f}  "
                f"conf={s.confidence:.4f}"
            )

    click.echo(f"\nResults saved to: {output_path}")


# ─── Manifest Command ────────────────────────────────────────────────────────


@rescore_cli.command("manifest")
@click.option(
    "--manifest", required=True,
    type=click.Path(exists=True),
    help="Path to YAML manifest file listing complexes.",
)
@click.option(
    "--output-dir", default="./scores",
    help="Output directory for results.",
)
@click.option(
    "--output-format", default="json",
    type=click.Choice(["json", "csv", "parquet"]),
)
@click.option("--device", default="auto")
@click.option("--checkpoint", default="auto")
@click.option("--use-msa-server", is_flag=True, default=False,
              help="[DISABLED in this fork] See msa_cache.py policy.")
@click.option("--use-lora", "use_lora", default=None,
              help="LoRA adapter name (or path) to apply before scoring.")
@click.option("--use-finetune", "use_finetune", default=None,
              help="Full fine-tune name (or path) to apply before scoring. "
                   "Mutually exclusive with --use-lora.")
@click.option("--log-level", default="INFO")
def rescore_manifest(
    manifest: str,
    output_dir: str,
    output_format: str,
    device: str,
    checkpoint: str,
    use_msa_server: bool,
    use_lora: Optional[str],
    use_finetune: Optional[str],
    log_level: str,
):
    """Rescore complexes listed in a YAML manifest."""
    setup_logging(log_level)
    _guard_msa_server_flag(use_msa_server)
    _guard_lora_finetune_flags(use_lora, use_finetune)
    import yaml

    with open(manifest) as f:
        manifest_data = yaml.safe_load(f)

    if not isinstance(manifest_data, dict) or "complexes" not in manifest_data:
        click.echo(
            "Error: Manifest must contain a 'complexes' key with a list of entries.",
            err=True,
        )
        sys.exit(1)

    from boltz.affinity_rescoring import AffinityRescorer

    rescorer = AffinityRescorer(
        checkpoint=checkpoint,
        device=device,
        lora=use_lora,
        finetune=use_finetune,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for entry in manifest_data["complexes"]:
        pdb_file = entry.get("pdb")
        smiles = entry.get("ligand_smiles")
        protein_chain = entry.get("protein_chain")
        ligand_chains_str = entry.get("ligand_chains")

        if not pdb_file:
            click.echo(f"Warning: Skipping entry without 'pdb' key: {entry}", err=True)
            continue

        lig_chains = None
        if ligand_chains_str:
            lig_chains = [c.strip() for c in ligand_chains_str.split(",")]

        smiles_dict = None
        if smiles:
            smiles_dict = smiles if isinstance(smiles, dict) else {"L": smiles}

        result = rescorer.rescore_pdb(
            pdb_file,
            protein_chain=protein_chain,
            ligand_chains=lig_chains,
            ligand_smiles=smiles_dict,
            use_msa_server=use_msa_server,
        )
        results.append(result)

    # Export all
    from boltz.affinity_rescoring.models import OutputFormat
    fmt = OutputFormat(output_format)
    output_file = out_dir / f"manifest_results.{output_format}"
    rescorer.export_results(results, output_file, format=output_format)

    click.echo(f"Processed {len(results)} complexes. Results: {output_file}")


# ─── Multi-Pocket Command ────────────────────────────────────────────────────


@rescore_cli.command("multipocket")
@click.option(
    "--receptor", "-r", default=None, type=click.Path(exists=True),
    help="Receptor PDB/CIF file. Mutually exclusive with --receptor-sequence.",
)
@click.option(
    "--receptor-sequence", default=None, type=str,
    help=(
        "Plain amino-acid sequence of the receptor (single-letter codes). "
        "Use instead of --receptor when no PDB is available; Boltz will "
        "fold the receptor from scratch together with the ligand copies. "
        "Mutually exclusive with --receptor."
    ),
)
@click.option(
    "--ligand-smiles", "-s", required=True, type=str,
    help="Ligand SMILES string (one ligand, replicated N times).",
)
@click.option(
    "--n-pockets", "-n", default=5, type=int,
    help="Number of ligand copies / candidate binding pockets (1..25).",
)
@click.option(
    "--output-dir", "-o", required=True, type=click.Path(),
    help="Output directory for structures, CSV, and HTML report.",
)
@click.option(
    "--protein-chain", default=None,
    help=(
        "Protein chain ID. Auto-detected from PDB when --receptor is used. "
        "Defaults to 'A' when using --receptor-sequence."
    ),
)
@click.option(
    "--reference-sequence", default=None,
    help="Full protein sequence override for PDB inputs with missing loops.",
)
@click.option(
    "--device", default="auto",
    type=click.Choice(["auto", "cuda", "cpu", "mps"]),
    help="Device for inference.",
)
@click.option(
    "--checkpoint", "structure_checkpoint", default=None,
    type=click.Path(exists=True),
    help="Boltz-2 structure checkpoint (default: auto-download).",
)
@click.option(
    "--affinity-checkpoint", default="auto",
    help="Affinity checkpoint or 'auto' to download.",
)
@click.option(
    "--cache-dir", default=None, type=click.Path(),
    help="Boltz cache directory (default: ~/.boltz or $BOLTZ_CACHE).",
)
@click.option(
    "--recycling-steps", default=3, type=int,
    help="Trunk recycling steps for Boltz prediction.",
)
@click.option(
    "--sampling-steps", default=200, type=int,
    help="Diffusion sampling steps for Boltz prediction.",
)
@click.option(
    "--msa", "msa_path", default=None, required=False,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a pre-computed MSA file (.a3m or .csv) for the receptor. "
         "If omitted, the MSA is generated automatically via the ColabFold "
         "MMseqs2 server (affinity_rescoring/mmseqs2.py) and cached in "
         "<output-dir>/receptor_msa.a3m.",
)
@click.option(
    "--sort-by", default="affinity_pred",
    type=click.Choice([
        "affinity_pred", "affinity_probability_binary",
        "interface_iptm", "boltz_confidence_score",
    ]),
    help="Column for ranking pockets in report.",
)
@click.option(
    "--ascending/--descending", default=False,
    help="Sort direction (default: descending = best first).",
)
@click.option(
    "--keep-boltz-outputs/--no-keep-boltz-outputs", default=True,
    help="Keep raw boltz prediction directory.",
)
@click.option(
    "--log-level", default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level.",
)
def multipocket(
    receptor: Optional[str],
    receptor_sequence: Optional[str],
    ligand_smiles: str,
    n_pockets: int,
    output_dir: str,
    protein_chain: Optional[str],
    reference_sequence: Optional[str],
    device: str,
    structure_checkpoint: Optional[str],
    affinity_checkpoint: str,
    cache_dir: Optional[str],
    recycling_steps: int,
    sampling_steps: int,
    msa_path: str,
    sort_by: str,
    ascending: bool,
    keep_boltz_outputs: bool,
    log_level: str,
):
    """Predict N binding pockets for a ligand and score each with the affinity head.

    Provide the receptor as either a PDB/CIF file (--receptor) or a plain
    amino-acid sequence (--receptor-sequence); exactly one is required.
    """
    setup_logging(log_level)

    if receptor and receptor_sequence:
        raise click.UsageError(
            "--receptor and --receptor-sequence are mutually exclusive. "
            "Provide exactly one."
        )
    if not receptor and not receptor_sequence:
        raise click.UsageError(
            "Provide either --receptor (PDB/CIF file) or "
            "--receptor-sequence (amino-acid string)."
        )

    from boltz.affinity_rescoring.multipocket import MultiPocketPipeline

    pipeline = MultiPocketPipeline(
        affinity_checkpoint=affinity_checkpoint,
        structure_checkpoint=structure_checkpoint,
        device=device,
        cache_dir=cache_dir,
    )

    report = pipeline.run(
        receptor=receptor,
        receptor_sequence=receptor_sequence,
        ligand_smiles=ligand_smiles,
        n_pockets=n_pockets,
        output_dir=output_dir,
        protein_chain=protein_chain,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        msa_path=msa_path,
        sort_by=sort_by,
        ascending=ascending,
        keep_boltz_outputs=keep_boltz_outputs,
        reference_sequence=reference_sequence,
    )

    click.echo(f"\nMulti-pocket pipeline complete in {report.total_time_s:.1f}s")
    click.echo(f"  Pockets extracted: {report.n_pockets_extracted}")
    click.echo(f"  CSV:  {report.csv_path}")
    click.echo(f"  HTML: {report.html_path}")
    click.echo(f"  Structures: {report.structures_dir}")
