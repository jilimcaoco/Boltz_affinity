#!/usr/bin/env python3
"""
fetch_fep_benchmark_structures.py

Downloads and catalogues co-crystal structures for all eight FEP+ benchmark
targets from the RCSB PDB.

Steps
-----
1. Query RCSB Search API for each target (UniProt → X-ray ≤ 2.5 Å, has ligand).
2. (After user confirmation) Download mmCIF files.
3. Parse each mmCIF, extract non-polymer ligands, flag cofactor candidates.
4. Filter by heavy-atom count [10, 45].
5. Save datasets/fep_benchmark_structures.csv.
6. Print per-target and overall summary.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_CONFIG: dict[str, dict[str, str]] = {
    "cdk2":     {"uniprot": "P24941", "name": "cyclin-dependent kinase 2"},
    "tyk2":     {"uniprot": "P29597", "name": "tyrosine kinase 2"},
    "jnk1":     {"uniprot": "P45983", "name": "mitogen-activated protein kinase 8"},
    "p38":      {"uniprot": "Q16539", "name": "mitogen-activated protein kinase 14"},
    "mcl1":     {"uniprot": "Q07820",
                 "name": "induced myeloid leukemia cell differentiation protein"},
    "ptp1b":    {"uniprot": "P18031",
                 "name": "tyrosine-protein phosphatase non-receptor type 1"},
    "thrombin": {"uniprot": "P00734", "name": "prothrombin"},
    "bace":     {"uniprot": "P56817", "name": "beta-secretase 1"},
}

EXCLUDE: set[str] = {
    "HOH", "WAT", "SO4", "GOL", "EDO", "PEG",
    "MPD", "PO4", "EPE", "MES", "DMSO", "ACT",
    "EOH", "FMT", "TRS", "BME", "DTT", "ACE",
    "NH2", "NO3", "CL",  "NA",  "MG",  "ZN",
    "CA",  "K",   "MN",  "FE",  "CU",  "IOD",
    "BR",  "F",   "SEP", "TPO", "PTR", "MLY",
    "MSE", "CSO", "OCS", "CME", "CSD", "SNN",
    "DAN", "PLM", "GDP", "GTP", "ATP", "ADP",
    "AMP", "ANP", "AGS", "GCP", "STU",
}

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.cif"

DATASETS_DIR = Path("datasets")
RAW_DIR = DATASETS_DIR / "raw"
OUTPUT_CSV = DATASETS_DIR / "fep_benchmark_structures.csv"
ERROR_LOG = DATASETS_DIR / "fetch_errors.log"

HEAVY_ATOM_MIN = 10
HEAVY_ATOM_MAX = 45

USER_AGENT = "FEP-benchmark-interpretability-study/1.0"

# ---------------------------------------------------------------------------
# Logging (file + stderr)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def _init_error_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create/append header on first run
    if not path.exists():
        with path.open("w") as fh:
            fh.write("# FEP benchmark fetch error log\n")


def _log_error(path: Path, pdb_id: str, msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with path.open("a") as fh:
        fh.write(f"{ts}\t{pdb_id}\t{msg}\n")


# ---------------------------------------------------------------------------
# STEP 1 — Query RCSB
# ---------------------------------------------------------------------------

def _build_query(uniprot: str) -> dict[str, Any]:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers"
                            ".reference_sequence_identifiers.database_accession"
                        ),
                        "operator": "exact_match",
                        "value": uniprot,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater",
                        "value": 0,
                    },
                },
            ],
        },
        "request_options": {"return_all_hits": True, "results_verbosity": "minimal"},
        "return_type": "entry",
        "request_info": {"src": "ui", "query_id": ""},
    }


def query_rcsb_for_target(target: str, cfg: dict[str, str]) -> list[str]:
    """Return list of PDB IDs matching all four criteria for *target*."""
    uniprot = cfg["uniprot"]
    query = _build_query(uniprot)
    try:
        resp = requests.post(
            RCSB_SEARCH_URL,
            json=query,
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return [hit["identifier"] for hit in data.get("result_set", [])]
    except requests.RequestException as exc:
        logger.error("RCSB query failed for %s (%s): %s", target, uniprot, exc)
        return []


def step1_query_all_targets() -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    print("\nQuerying RCSB PDB for all eight targets …\n")
    header = f"  {'Target':<12}| Hits"
    divider = "  " + "-" * 12 + "|------"
    print(header)
    print(divider)
    total = 0
    for target, cfg in TARGET_CONFIG.items():
        pdb_ids = query_rcsb_for_target(target, cfg)
        hits[target] = pdb_ids
        total += len(pdb_ids)
        print(f"  {target:<12}| {len(pdb_ids)}")
    print(divider)
    print(f"  {'Total':<12}| {total}")
    print()
    return hits


# ---------------------------------------------------------------------------
# STEP 2 — Download mmCIF files
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _download_cif(
    session: requests.Session,
    pdb_id: str,
    dest: Path,
    error_log: Path,
) -> bool:
    """Download a single CIF file; return True on success."""
    url = RCSB_DOWNLOAD_TEMPLATE.format(pdb_id=pdb_id)
    backoff = 2
    for attempt in range(1, 4):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except requests.ConnectionError as exc:
            if attempt == 3:
                msg = f"ConnectionError after 3 attempts: {exc}"
                _log_error(error_log, pdb_id, msg)
                return False
            time.sleep(backoff)
            backoff *= 2
        except requests.HTTPError as exc:
            msg = f"HTTP {exc.response.status_code}: {exc}"
            _log_error(error_log, pdb_id, msg)
            return False
        except requests.RequestException as exc:
            msg = f"RequestException: {exc}"
            _log_error(error_log, pdb_id, msg)
            return False
    return False  # unreachable but keeps type checker happy


def step2_download(
    hits: dict[str, list[str]],
    error_log: Path,
) -> dict[str, dict[str, Path]]:
    """
    Download all CIF files.

    Returns
    -------
    downloaded : dict[target, dict[pdb_id, cif_path]]
    """
    # Try tqdm; fall back to simple counter logging
    try:
        from tqdm import tqdm  # type: ignore[import]
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    total = sum(len(v) for v in hits.values())
    session = _make_session()
    downloaded: dict[str, dict[str, Path]] = {t: {} for t in hits}
    n = 0

    iterator: Any
    if use_tqdm:
        iterator = tqdm(total=total, unit="cif", desc="Downloading")

    for target, pdb_ids in hits.items():
        target_dir = RAW_DIR / target
        target_dir.mkdir(parents=True, exist_ok=True)

        for pdb_id in pdb_ids:
            dest = target_dir / f"{pdb_id}.cif"
            n += 1

            if dest.exists() and dest.stat().st_size > 0:
                downloaded[target][pdb_id] = dest
                if use_tqdm:
                    iterator.update(1)
                else:
                    if n % 25 == 0:
                        print(f"  Downloaded {n}/{total} ({target})")
                continue

            ok = _download_cif(session, pdb_id, dest, error_log)
            if ok:
                downloaded[target][pdb_id] = dest
            elif dest.exists():
                dest.unlink(missing_ok=True)  # remove partial file

            if use_tqdm:
                iterator.update(1)
            else:
                if n % 25 == 0:
                    print(f"  Downloaded {n}/{total} ({target})")

            time.sleep(1)  # polite delay

    if use_tqdm:
        iterator.close()

    return downloaded


# ---------------------------------------------------------------------------
# STEP 3 — Parse mmCIF
# ---------------------------------------------------------------------------

def _parse_with_gemmi(cif_path: Path) -> dict[str, Any] | None:
    """Return parsed data dict using gemmi, or None if unavailable/failed."""
    try:
        import gemmi  # type: ignore[import]
    except ImportError:
        return None
    try:
        doc = gemmi.cif.read(str(cif_path))
        block = doc.sole_block()

        # --- resolution ---
        resolution: float | None = None
        for tag in ("_reflns.d_resolution_high", "_refine.ls_d_res_high"):
            val = block.find_value(tag)
            if val and val not in (".", "?"):
                try:
                    resolution = float(val)
                    break
                except ValueError:
                    pass

        # --- deposition date ---
        dep_date: str | None = None
        val = block.find_value("_pdbx_database_status.recvd_initial_deposition_date")
        if val and val not in (".", "?"):
            dep_date = val.strip("'\"")

        # --- nonpolymer residues ---
        atom_site = block.find(
            "_atom_site.",
            ["group_PDB", "label_asym_id", "label_comp_id",
             "label_seq_id", "type_symbol"],
        )

        hetatm_atoms: dict[tuple[str, str, str], int] = {}  # (chain, resn, seqid) -> count
        for row in atom_site:
            if row[0].strip().upper() != "HETATM":
                continue
            if row[4].strip().upper() == "H":
                continue  # skip hydrogens
            key = (row[1].strip(), row[2].strip(), row[3].strip())
            hetatm_atoms[key] = hetatm_atoms.get(key, 0) + 1

        residues: list[dict[str, Any]] = []
        for (chain_id, resn, seqid), count in hetatm_atoms.items():
            residues.append({
                "chain_id": chain_id,
                "residue_name": resn,
                "residue_number": seqid,
                "heavy_atom_count_mmcif": count,
            })

        return {
            "resolution": resolution,
            "deposition_date": dep_date,
            "residues": residues,
        }
    except Exception as exc:
        logger.warning("gemmi parse failed for %s: %s", cif_path.name, exc)
        return None


def _parse_with_biopython(cif_path: Path) -> dict[str, Any] | None:
    """Return parsed data dict using BioPython, or None if unavailable/failed."""
    try:
        from Bio.PDB import MMCIFParser  # type: ignore[import]
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # type: ignore[import]
    except ImportError:
        return None
    try:
        mmcif_dict = MMCIF2Dict(str(cif_path))

        # --- resolution ---
        resolution: float | None = None
        for key in ("_reflns.d_resolution_high", "_refine.ls_d_res_high"):
            vals = mmcif_dict.get(key, [])
            if isinstance(vals, str):
                vals = [vals]
            for v in vals:
                if v and v not in (".", "?"):
                    try:
                        resolution = float(v)
                        break
                    except ValueError:
                        pass
            if resolution is not None:
                break

        # --- deposition date ---
        dep_date: str | None = None
        vals = mmcif_dict.get(
            "_pdbx_database_status.recvd_initial_deposition_date", []
        )
        if isinstance(vals, str):
            vals = [vals]
        if vals:
            dep_date = vals[0].strip("'\"") if vals[0] not in (".", "?") else None

        # --- nonpolymer residues ---
        groups = mmcif_dict.get("_atom_site.group_PDB", [])
        chains = mmcif_dict.get("_atom_site.label_asym_id", [])
        resnames = mmcif_dict.get("_atom_site.label_comp_id", [])
        seqids = mmcif_dict.get("_atom_site.label_seq_id", [])
        elements = mmcif_dict.get("_atom_site.type_symbol", [])

        if isinstance(groups, str):
            groups = [groups]
            chains = [chains]
            resnames = [resnames]
            seqids = [seqids]
            elements = [elements]

        hetatm_atoms: dict[tuple[str, str, str], int] = {}
        for grp, ch, rn, sq, el in zip(groups, chains, resnames, seqids, elements):
            if grp.strip().upper() != "HETATM":
                continue
            if el.strip().upper() == "H":
                continue
            key = (ch.strip(), rn.strip(), sq.strip())
            hetatm_atoms[key] = hetatm_atoms.get(key, 0) + 1

        residues: list[dict[str, Any]] = []
        for (chain_id, resn, seqid), count in hetatm_atoms.items():
            residues.append({
                "chain_id": chain_id,
                "residue_name": resn,
                "residue_number": seqid,
                "heavy_atom_count_mmcif": count,
            })

        return {
            "resolution": resolution,
            "deposition_date": dep_date,
            "residues": residues,
        }
    except Exception as exc:
        logger.warning("BioPython parse failed for %s: %s", cif_path.name, exc)
        return None


def _parse_cif(cif_path: Path) -> dict[str, Any] | None:
    """Try gemmi first, fall back to BioPython."""
    result = _parse_with_gemmi(cif_path)
    if result is not None:
        return result
    result = _parse_with_biopython(cif_path)
    return result


# ---------------------------------------------------------------------------
# STEP 4 — Heavy-atom count via RDKit
# ---------------------------------------------------------------------------

def _rdkit_heavy_atoms(pdb_id: str, cif_path: Path) -> int | None:
    """
    Attempt to get heavy-atom count from RDKit.
    Returns None if RDKit is unavailable or parsing fails.
    """
    try:
        from rdkit import Chem  # type: ignore[import]
    except ImportError:
        return None
    try:
        mol = Chem.MolFromPDBFile(str(cif_path), removeHs=True, sanitize=False)
        if mol is None:
            return None
        return mol.GetNumHeavyAtoms()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# STEP 3+4 combined — process all structures
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "target", "pdb_id", "chain_id", "ligand_id", "ligand_name",
    "heavy_atom_count", "heavy_atom_source", "resolution",
    "deposition_date", "cif_path", "possible_cofactor", "passed_size_filter",
]


def step3_to_4_process(
    downloaded: dict[str, dict[str, Path]],
    error_log: Path,
) -> list[dict[str, Any]]:
    """
    Parse CIF files, filter ligands, return list of row dicts.
    """
    rows: list[dict[str, Any]] = []
    rdkit_failures: list[str] = []
    cofactor_warnings: list[tuple[str, str]] = []

    for target, cif_map in downloaded.items():
        for pdb_id, cif_path in cif_map.items():
            parsed = _parse_cif(cif_path)
            if parsed is None:
                _log_error(error_log, pdb_id, "CIF parse failure (both gemmi and BioPython)")
                rdkit_failures.append(pdb_id)
                continue

            resolution = parsed["resolution"]
            dep_date = parsed["deposition_date"]
            residues = parsed["residues"]

            # Separate excluded vs candidate
            non_excluded = [r for r in residues if r["residue_name"] not in EXCLUDE]
            excluded = [r for r in residues if r["residue_name"] in EXCLUDE]

            # Cofactor-flag logic: if no non-excluded ligands remain and all
            # excluded ones are in the EXCLUDE set, check for sole ligand in kinase
            if not non_excluded and excluded:
                for r in excluded:
                    cofactor_warnings.append((pdb_id, r["residue_name"]))
                    print(
                        f"  [WARNING] {pdb_id}: only excluded molecule "
                        f"'{r['residue_name']}' found — possible_cofactor flagged"
                    )

            # Process candidates
            for res in non_excluded:
                resn = res["residue_name"]
                mmcif_count = res["heavy_atom_count_mmcif"]
                chain_id = res["chain_id"]
                seq_id = res["residue_number"]

                # RDKit heavy atom count (uses whole-structure PDB — approximate)
                rdkit_count = _rdkit_heavy_atoms(pdb_id, cif_path)
                if rdkit_count is not None:
                    heavy_atom_count = rdkit_count
                    ha_source = "rdkit"
                else:
                    heavy_atom_count = mmcif_count
                    ha_source = "mmcif_count"
                    if pdb_id not in rdkit_failures:
                        rdkit_failures.append(pdb_id)

                passed = HEAVY_ATOM_MIN <= heavy_atom_count <= HEAVY_ATOM_MAX

                rows.append({
                    "target": target,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "ligand_id": resn,
                    "ligand_name": resn,  # CIF has no common name without extra lookup
                    "heavy_atom_count": heavy_atom_count,
                    "heavy_atom_source": ha_source,
                    "resolution": resolution,
                    "deposition_date": dep_date,
                    "cif_path": str(cif_path),
                    "possible_cofactor": False,
                    "passed_size_filter": passed,
                })

            # Also add cofactor-flagged rows
            for r in excluded:
                resn = r["residue_name"]
                mmcif_count = r["heavy_atom_count_mmcif"]
                chain_id = r["chain_id"]

                passed = HEAVY_ATOM_MIN <= mmcif_count <= HEAVY_ATOM_MAX

                rows.append({
                    "target": target,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "ligand_id": resn,
                    "ligand_name": resn,
                    "heavy_atom_count": mmcif_count,
                    "heavy_atom_source": "mmcif_count",
                    "resolution": resolution,
                    "deposition_date": dep_date,
                    "cif_path": str(cif_path),
                    "possible_cofactor": True,
                    "passed_size_filter": passed,
                })

    # Store for summary
    step3_to_4_process._rdkit_failures = rdkit_failures  # type: ignore[attr-defined]
    step3_to_4_process._cofactor_warnings = cofactor_warnings  # type: ignore[attr-defined]
    return rows


# ---------------------------------------------------------------------------
# STEP 5 — Save CSV
# ---------------------------------------------------------------------------

def step5_save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# STEP 6 — Print summary
# ---------------------------------------------------------------------------

def step6_summary(
    hits: dict[str, list[str]],
    rows: list[dict[str, Any]],
    rdkit_failures: list[str],
    cofactor_warnings: list[tuple[str, str]],
) -> None:
    import statistics

    passed_rows = [r for r in rows if r["passed_size_filter"] and not r["possible_cofactor"]]

    print("\n" + "=" * 75)
    print("Per-target summary:")
    hdr = (
        f"  {'Target':<12}| {'Raw hits':>9} | {'After filter':>18} "
        f"| {'Resolution (mean±std)':^23} | {'Date range'}"
    )
    print(hdr)
    print("  " + "-" * 12 + "|" + "-" * 11 + "|" + "-" * 20 + "|" + "-" * 25 + "|" + "-" * 22)

    for target in TARGET_CONFIG:
        raw = len(hits.get(target, []))
        t_passed = [r for r in passed_rows if r["target"] == target]
        after = len(t_passed)

        resolutions = [r["resolution"] for r in t_passed if r["resolution"] is not None]
        if resolutions:
            mean_r = statistics.mean(resolutions)
            std_r = statistics.pstdev(resolutions)
            res_str = f"{mean_r:.2f}±{std_r:.2f}"
        else:
            res_str = "N/A"

        dates = sorted(r["deposition_date"] for r in t_passed if r["deposition_date"])
        date_range = f"{dates[0]} – {dates[-1]}" if dates else "N/A"

        print(
            f"  {target:<12}| {raw:>9} | {after:>18} "
            f"| {res_str:^23} | {date_range}"
        )

    print()
    total_dl = sum(len(v) for v in hits.values())
    all_ha = [r["heavy_atom_count"] for r in passed_rows]

    print(f"  Overall:")
    print(f"    Total PDB entries downloaded       : {total_dl}")
    print(f"    Total ligand instances after filter: {len(passed_rows)}")

    if all_ha:
        mean_ha = statistics.mean(all_ha)
        min_ha = min(all_ha)
        max_ha = max(all_ha)
        print(
            f"    Heavy atom count distribution      : "
            f"mean={mean_ha:.1f}, min={min_ha}, max={max_ha}"
        )
    else:
        print("    Heavy atom count distribution      : N/A")

    print(f"    Possible cofactor warnings         : {len(cofactor_warnings)}")
    for pdb_id, resn in cofactor_warnings:
        print(f"      {pdb_id} — {resn}")

    print(f"    Structures with RDKit parse failures: {len(rdkit_failures)}")
    for pdb_id in rdkit_failures:
        print(f"      {pdb_id}")

    print("=" * 75)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and catalogue FEP+ benchmark co-crystal structures."
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt (for automated pipelines).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    _init_error_log(ERROR_LOG)

    # ---- Step 1 ----
    hits = step1_query_all_targets()
    total = sum(len(v) for v in hits.values())

    # Confirmation prompt
    if not args.yes:
        print(
            f"Found {total} total structures across all 8 targets. "
            "Proceed with download? (yes/no)"
        )
        answer = input().strip().lower()
        if answer != "yes":
            print("Aborted.")
            sys.exit(0)
    else:
        print(f"Found {total} total structures across all 8 targets. "
              "Auto-confirmed (--yes).")

    # ---- Step 2 ----
    print()
    downloaded = step2_download(hits, ERROR_LOG)

    # ---- Steps 3 & 4 ----
    print("\nParsing mmCIF files and extracting ligand information …")
    rows = step3_to_4_process(downloaded, ERROR_LOG)
    rdkit_failures = getattr(step3_to_4_process, "_rdkit_failures", [])
    cofactor_warnings = getattr(step3_to_4_process, "_cofactor_warnings", [])

    # ---- Step 5 ----
    step5_save_csv(rows, OUTPUT_CSV)

    # ---- Step 6 ----
    step6_summary(hits, rows, rdkit_failures, cofactor_warnings)


if __name__ == "__main__":
    main()
