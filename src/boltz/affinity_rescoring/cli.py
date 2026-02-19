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
    help="Use MSA server for sequence search.",
)
@click.option(
    "--reference-sequence", default=None,
    help='Full biological sequence(s) as JSON: \'{"A": "MKTL..."}\'. '
         'Overrides SEQRES and ATOM-derived sequences to handle '
         'missing loops / incomplete structures.',
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
    reference_sequence: Optional[str],
    log_level: str,
):
    """Rescore a single PDB/CIF protein-ligand complex."""
    setup_logging(log_level)

    from boltz.affinity_rescoring import AffinityRescorer

    rescorer = AffinityRescorer(
        checkpoint=checkpoint,
        device=device,
        validation_level=validation,
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
    "--use-msa-server", is_flag=True, default=False,
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
    use_msa_server: bool,
    log_level: str,
):
    """Rescore all PDB/CIF files in a directory."""
    setup_logging(log_level)

    from boltz.affinity_rescoring import AffinityRescorer
    from boltz.affinity_rescoring.export import compute_batch_summary

    rescorer = AffinityRescorer(
        checkpoint=checkpoint,
        device=device,
        validation_level=validation,
    )

    smiles_dict = None
    if ligand_smiles:
        try:
            smiles_dict = json.loads(ligand_smiles)
        except json.JSONDecodeError:
            click.echo("Error: --ligand-smiles must be valid JSON.", err=True)
            sys.exit(1)

    results = rescorer.rescore_directory(
        input_dir,
        output_path=output_path,
        output_format=output_format,
        recursive=recursive,
        ligand_smiles=smiles_dict,
        use_msa_server=use_msa_server,
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
    log_level: str,
):
    """Score a receptor against multiple ligands from MOL2 file."""
    setup_logging(log_level)

    from boltz.affinity_rescoring import AffinityRescorer

    rescorer = AffinityRescorer(
        checkpoint=checkpoint,
        device=device,
        validation_level=validation,
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
@click.option("--use-msa-server", is_flag=True, default=False)
@click.option("--log-level", default="INFO")
def rescore_manifest(
    manifest: str,
    output_dir: str,
    output_format: str,
    device: str,
    checkpoint: str,
    use_msa_server: bool,
    log_level: str,
):
    """Rescore complexes listed in a YAML manifest."""
    setup_logging(log_level)
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
