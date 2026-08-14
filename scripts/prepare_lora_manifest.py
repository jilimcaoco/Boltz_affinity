#!/usr/bin/env python3
"""Prepare LoRA training inputs from a simple CSV + a receptor PDB/CIF.

Given a CSV that only has the columns  name / ligand / target  (the format
you'd typically export from an assay spreadsheet), this script:

  1. Parses the receptor PDB/CIF file to extract protein-chain sequences.
  2. Writes one Boltz YAML per unique receptor (or one shared one if a
     single receptor file is provided).
  3. Expects pre-docked complex PDB/CIF files (one per ligand), and validates
     they exist.
  4. Outputs the full LoRA training manifest CSV that boltz lora train expects:
       name, ligand, receptor, target, structure

Usage
-----
  python scripts/prepare_lora_manifest.py \\
      --input assay_hits.csv \\
      --receptor receptor.pdb \\
      --poses-dir docked_poses/ \\
      --out-dir lora_inputs/ \\
      [--msa /path/to/msa.a3m] \\
      [--chain A] \\
      [--pose-pattern "{name}.pdb"] \\
      [--ligand-col ligand] \\
      [--name-col name] \\
      [--target-col target] \\
      [--use-msa-server]

Input CSV requirements
----------------------
Must contain at minimum three columns (names configurable via flags):
  - name     : unique identifier for each compound
  - ligand   : SMILES string OR path to a MOL2/SDF file
  - target   : numeric affinity value (e.g. pIC50)

Pose files
----------
By default, the script looks for a docked complex PDB in `--poses-dir`
named `{name}.pdb` for each row (configurable via `--pose-pattern`).
Example: if name=mol_001, it looks for `docked_poses/mol_001.pdb`.

If none of the auto-discovered poses exist, the script reports which names
are missing and exits non-zero.

Outputs (all under --out-dir)
------------------------------
  receptor.yaml   – Boltz YAML for the receptor (one per unique receptor file)
  manifest.csv    – Full training manifest ready for `boltz lora train`
  missing.txt     – List of names whose pose file was not found (if any)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional


# ─── YAML generation ─────────────────────────────────────────────────────────


def extract_sequences_from_structure(pdb_path: Path, chain: Optional[str]) -> dict[str, str]:
    """Return {chain_id: sequence} from a PDB/CIF via gemmi + SEQRES fallback."""
    try:
        import gemmi
    except ImportError:
        print(
            "ERROR: gemmi is required to auto-extract sequences from PDB/CIF.\n"
            "Install it with:  pip install gemmi\n"
            "Or pass --sequence manually to skip auto-extraction.",
            file=sys.stderr,
        )
        sys.exit(1)

    st = gemmi.read_structure(str(pdb_path))
    st.setup_entities()

    THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    NONSTANDARD = {
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS",
        "HSE": "HIS", "HSP": "HIS",
        "CYX": "CYS", "CYM": "CYS",
        "ASH": "ASP", "GLH": "GLU",
        "MSE": "MET", "SEC": "CYS", "TPO": "THR", "SEP": "SER",
        "PTR": "TYR", "PYL": "LYS",
        "LYN": "LYS", "LYP": "LYS",
    }

    sequences: dict[str, str] = {}

    # Attempt SEQRES first (more complete for truncated structures)
    for entity in st.entities:
        if entity.entity_type != gemmi.EntityType.Polymer:
            continue
        if entity.polymer_type not in (
            gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD,
        ):
            continue
        full_seq = entity.full_sequence
        if not full_seq:
            continue
        residues_1l = []
        for code in full_seq:
            name = code.strip().upper()
            name = NONSTANDARD.get(name, name)
            info = gemmi.find_tabulated_residue(name)
            if not info.is_amino_acid():
                continue
            olc = info.one_letter_code
            if olc and olc != "?":
                residues_1l.append(olc.upper())
        if not residues_1l:
            continue
        seq_str = "".join(residues_1l)
        # Map entity → chain(s)
        for model in st:
            for ch in model:
                try:
                    ent = st.get_entity_of(ch)
                    if ent and ent.name == entity.name:
                        if chain is None or ch.name == chain:
                            sequences[ch.name] = seq_str
                except Exception:
                    pass
            break

    # Fallback: ATOM-record sequences if SEQRES empty
    if not sequences:
        for model in st:
            for ch in model:
                if chain is not None and ch.name != chain:
                    continue
                seen: dict[tuple, str] = {}
                for residue in ch:
                    if residue.entity_type != gemmi.EntityType.Polymer:
                        continue
                    key = (residue.seqid.num, residue.seqid.icode)
                    if key in seen:
                        continue
                    name = residue.name.strip().upper()
                    name = NONSTANDARD.get(name, name)
                    if name in THREE_TO_ONE:
                        seen[key] = THREE_TO_ONE[name]
                if seen:
                    sequences[ch.name] = "".join(seen[k] for k in sorted(seen))
            break

    return sequences


def build_receptor_yaml(
    sequences: dict[str, str],
    msa: Optional[str] = None,
    use_msa_server: bool = False,
    ligand_chain_id: str = "B",
    msa_for_chain: Optional[dict[str, str]] = None,
) -> str:
    """Render a Boltz YAML string from extracted protein sequences.

    Per-chain MSA paths (``msa_for_chain``) take priority over the
    single ``msa`` argument. ``use_msa_server`` is honoured only for
    backward compatibility — the calling CLI rejects it before reaching
    this function.
    """
    lines = ["version: 1", "sequences:"]

    # Assign sequential chain IDs A, B, C, ... skipping the ligand slot
    assigned_ids = []
    char_iter = iter(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != ligand_chain_id)
    for chain_name in sequences:
        assigned_ids.append((chain_name, next(char_iter)))

    for original_chain, boltz_id in assigned_ids:
        seq = sequences[original_chain]
        lines.append(f"  - protein:")
        lines.append(f"      id: {boltz_id}")
        lines.append(f"      sequence: {seq}")
        per_chain = (msa_for_chain or {}).get(original_chain) \
            or (msa_for_chain or {}).get(boltz_id)
        chosen_msa = per_chain or msa
        if chosen_msa:
            lines.append(f"      msa: {chosen_msa}")
        elif use_msa_server:
            pass  # omitting msa key → Boltz will run the server (disabled here)
        else:
            lines.append(f"      msa: empty")

    # Ligand placeholder — will be replaced per-row at training time,
    # but we include the ligand chain block so the YAML is valid.
    lines.append(f"  - ligand:")
    lines.append(f"      id: {ligand_chain_id}")
    lines.append(f"      smiles: 'C'  # placeholder — overridden per row by the trainer")
    lines.append(f"properties:")
    lines.append(f"  - affinity:")
    lines.append(f"      binder: {ligand_chain_id}")

    return "\n".join(lines) + "\n"


# ─── Manifest construction ───────────────────────────────────────────────────


def discover_pose(name: str, poses_dir: Path, pattern: str) -> Optional[Path]:
    filename = pattern.replace("{name}", name)
    candidate = poses_dir / filename
    return candidate if candidate.exists() else None


def read_input_csv(
    path: Path,
    name_col: str,
    ligand_col: str,
    target_col: str,
) -> list[dict]:
    rows = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for required in (name_col, ligand_col, target_col):
            if required not in fieldnames:
                print(
                    f"ERROR: Column '{required}' not found in {path}.\n"
                    f"Available columns: {fieldnames}\n"
                    "Use --name-col / --ligand-col / --target-col to remap.",
                    file=sys.stderr,
                )
                sys.exit(1)
        for i, row in enumerate(reader):
            try:
                float(row[target_col])
            except (TypeError, ValueError):
                print(
                    f"WARNING: row {i} has non-numeric target "
                    f"'{row.get(target_col)}' — skipped.",
                    file=sys.stderr,
                )
                continue
            rows.append({
                "name": row[name_col].strip(),
                "ligand": row[ligand_col].strip(),
                "target": row[target_col].strip(),
            })
    return rows


def write_manifest(
    rows: list[dict],
    receptor_yaml: Path,
    out_path: Path,
) -> None:
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["name", "ligand", "receptor", "target", "structure"],
        )
        writer.writeheader()
        writer.writerows(rows)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="Input CSV with name / ligand / target columns.")
    p.add_argument("--receptor", "-r", required=True, type=Path,
                   help="Receptor PDB or CIF file.")
    p.add_argument("--poses-dir", "-p", type=Path, default=None,
                   help="Directory containing one docked-complex PDB/CIF per row. "
                        "If omitted the structure column is left empty (only valid "
                        "for mode=full training, which is not yet supported).")
    p.add_argument("--out-dir", "-o", type=Path, default=Path("lora_inputs"),
                   help="Output directory (created if absent). Default: lora_inputs/")
    p.add_argument("--pose-pattern", default="{name}.pdb",
                   help="Filename pattern for pose files. "
                        "Use {name} as placeholder. Default: {name}.pdb")
    p.add_argument("--msa", default=None,
                   help="Path to a pre-computed MSA .a3m file. "
                        "If omitted, the shared cache ($BOLTZ_MSA_CACHE_DIR "
                        "and --msa-dir, if given) is searched for a file "
                        "keyed by the sequence hash, chain id or receptor "
                        "stem. If nothing is found, `msa: empty` is written.")
    p.add_argument("--msa-dir", default=None, type=Path,
                   help="Extra directory of pre-computed .a3m files to search "
                        "before falling back to $BOLTZ_MSA_CACHE_DIR.")
    p.add_argument("--use-msa-server", action="store_true", default=False,
                   help="[DISABLED in this fork] Pre-compute MSAs with "
                        "`python -m boltz.affinity_rescoring.mmseqs2` and "
                        "point --msa-dir / $BOLTZ_MSA_CACHE_DIR at the cache.")
    p.add_argument("--chain", default=None,
                   help="Restrict to a single receptor chain (e.g. A). "
                        "Default: include all protein chains.")
    p.add_argument("--sequence", default=None,
                   help="Provide the protein sequence directly (one-letter codes) "
                        "instead of auto-extracting from the PDB/CIF. "
                        "When set, --chain must also identify the single chain ID "
                        "to use in the YAML (defaults to 'A').")
    p.add_argument("--ligand-chain-id", default="B",
                   help="Chain ID to assign to the ligand in the YAML. Default: B")
    # Column remapping
    p.add_argument("--name-col", default="name",
                   help="Input CSV column that contains the compound name.")
    p.add_argument("--ligand-col", default="ligand",
                   help="Input CSV column that contains the ligand (SMILES or path).")
    p.add_argument("--target-col", default="target",
                   help="Input CSV column that contains the numeric target value.")
    # Target scale
    p.add_argument("--convert-ic50-nm", action="store_true", default=False,
                   help="Convert IC50 values in nM to pIC50 "
                        "(applies to the target column: pIC50 = 9 - log10(IC50_nM)).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import math

    if args.use_msa_server:
        try:
            from boltz.affinity_rescoring.msa_cache import (
                raise_msa_server_disabled,
            )
            raise_msa_server_disabled()
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not args.input.exists():
        print(f"ERROR: Input CSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.receptor.exists():
        print(f"ERROR: Receptor file not found: {args.receptor}", file=sys.stderr)
        sys.exit(1)
    if args.poses_dir and not args.poses_dir.exists():
        print(f"ERROR: Poses directory not found: {args.poses_dir}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Sequence extraction / YAML generation ────────────────────────────────
    if args.sequence:
        chain_id = args.chain or "A"
        sequences = {chain_id: args.sequence}
        print(f"Using user-supplied sequence ({len(args.sequence)} residues, chain {chain_id}).")
    else:
        print(f"Extracting protein sequence(s) from {args.receptor} …")
        sequences = extract_sequences_from_structure(args.receptor, chain=args.chain)
        if not sequences:
            print(
                "ERROR: No protein chains found in receptor file.\n"
                "Try specifying --chain or --sequence manually.",
                file=sys.stderr,
            )
            sys.exit(1)
        for ch, seq in sequences.items():
            print(f"  Chain {ch}: {len(seq)} residues")

    # ── Resolve per-chain MSAs from the shared cache ─────────────────────────
    # Priority: explicit --msa (single file, broadcast to all chains)
    #           > per-chain hit from --msa-dir / $BOLTZ_MSA_CACHE_DIR
    #           > write `msa: empty`.
    msa_for_chain: dict[str, str] = {}
    if not args.msa:
        try:
            from boltz.affinity_rescoring.msa_cache import find_msa
        except Exception:  # pragma: no cover
            find_msa = None  # type: ignore[assignment]
        if find_msa is not None:
            extra_dirs = [args.msa_dir] if args.msa_dir else None
            target_hint = args.receptor.stem
            for ch, seq in sequences.items():
                hit = find_msa(
                    sequence=seq,
                    msa_dirs=extra_dirs,
                    chain_id=ch,
                    target=target_hint,
                )
                if hit is not None:
                    msa_for_chain[ch] = str(hit)
                    print(f"  Resolved MSA for chain {ch} → {hit}")
                else:
                    print(f"  No cached MSA found for chain {ch} "
                          f"(will write `msa: empty`).")

    yaml_text = build_receptor_yaml(
        sequences,
        msa=args.msa,
        use_msa_server=args.use_msa_server,
        ligand_chain_id=args.ligand_chain_id,
        msa_for_chain=msa_for_chain,
    )
    receptor_yaml_path = args.out_dir / "receptor.yaml"
    receptor_yaml_path.write_text(yaml_text)
    print(f"Wrote receptor YAML → {receptor_yaml_path}")

    # ── Read input CSV ───────────────────────────────────────────────────────
    input_rows = read_input_csv(
        args.input,
        name_col=args.name_col,
        ligand_col=args.ligand_col,
        target_col=args.target_col,
    )
    print(f"Read {len(input_rows)} rows from {args.input}.")

    # ── Optional IC50 → pIC50 conversion ────────────────────────────────────
    if args.convert_ic50_nm:
        converted = 0
        for row in input_rows:
            try:
                ic50_nm = float(row["target"])
                row["target"] = f"{9.0 - math.log10(ic50_nm):.4f}"
                converted += 1
            except (ValueError, ZeroDivisionError):
                pass
        print(f"Converted {converted} IC50 values (nM) → pIC50.")

    # ── Discover pose files ──────────────────────────────────────────────────
    manifest_rows: list[dict] = []
    missing: list[str] = []

    for row in input_rows:
        structure_path: Optional[str] = None
        if args.poses_dir:
            pose = discover_pose(row["name"], args.poses_dir, args.pose_pattern)
            if pose is None:
                missing.append(row["name"])
            else:
                structure_path = str(pose.resolve())

        manifest_rows.append({
            "name": row["name"],
            "ligand": row["ligand"],
            "receptor": str(receptor_yaml_path.resolve()),
            "target": row["target"],
            "structure": structure_path or "",
        })

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest_path = args.out_dir / "manifest.csv"
    write_manifest(manifest_rows, receptor_yaml_path, manifest_path)
    print(f"Wrote manifest CSV  → {manifest_path} ({len(manifest_rows)} rows)")

    # ── Report missing poses ─────────────────────────────────────────────────
    if missing:
        missing_path = args.out_dir / "missing.txt"
        missing_path.write_text("\n".join(missing) + "\n")
        print(
            f"\nWARNING: {len(missing)}/{len(input_rows)} pose file(s) not found.\n"
            f"  Pattern: {args.poses_dir}/{args.pose_pattern}\n"
            f"  Missing names written to: {missing_path}\n"
            "  These rows have an empty 'structure' column and will fail during\n"
            "  `boltz lora train --mode rescore` unless you supply the files.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Ready ─────────────────────────────────────────────────────────")
    print(f"  Receptor YAML : {receptor_yaml_path}")
    print(f"  Training CSV  : {manifest_path}")
    print(f"  Rows          : {len(manifest_rows)}")
    if args.poses_dir:
        ok = len(manifest_rows) - len(missing)
        print(f"  Poses found   : {ok}/{len(manifest_rows)}")
    print()
    print("Next step:")
    print(f"  boltz lora train \\")
    print(f"      --name my_adapter \\")
    print(f"      --csv {manifest_path} \\")
    print(f"      --loss huber --rank 8 --epochs 10")


if __name__ == "__main__":
    main()
