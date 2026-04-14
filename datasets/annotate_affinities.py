#!/usr/bin/env python3
"""
annotate_affinities.py

Looks up experimental binding affinities for every entry in
datasets/fep_benchmark_structures.csv using three sources in priority order:
  1. PDBbind refined set index  (highest quality)
  2. BindingDB REST API
  3. ChEMBL REST API            (best-effort fallback)

Saves datasets/fep_benchmark_annotated.csv and prints a per-target summary
with pIC50 distributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("datasets")
INPUT_CSV = DATASETS_DIR / "fep_benchmark_structures.csv"
OUTPUT_CSV = DATASETS_DIR / "fep_benchmark_annotated.csv"

PDBBIND_DEFAULT_PATHS = [
    Path.home() / "Downloads" / "INDEX_refined_data.2020",
    DATASETS_DIR / "pdbbind_index.csv",
]

BINDINGDB_URL = (
    "https://www.bindingdb.org/axis2/services/BDBService/"
    "getLigandsByPDB?pdbid={pdb_id}&response=json"
)
BINDINGDB_DELAY = 0.5  # seconds between requests

CHEMBL_ACTIVITY_URL = (
    "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    "?target_chembl_id={chembl_id}"
    "&pchembl_value__isnull=false"
    "&standard_type__in=IC50,Ki,Kd"
    "&limit=1000"
)

CHEMBL_TARGET_IDS: dict[str, str] = {
    "cdk2":     "CHEMBL301",
    "tyk2":     "CHEMBL2366",
    "jnk1":     "CHEMBL2276",
    "p38":      "CHEMBL260",
    "mcl1":     "CHEMBL4891",
    "ptp1b":    "CHEMBL1862",
    "thrombin": "CHEMBL204",
    "bace":     "CHEMBL4822",
}

USER_AGENT = "FEP-benchmark-interpretability-study/1.0"

EXTRA_COLUMNS = [
    "pIC50", "affinity_type", "affinity_source",
    "n_measurements", "affinity_notes",
    "has_affinity", "affinity_reliable",
]

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _geometric_mean(values: list[float]) -> float:
    """Geometric mean of positive values."""
    n = len(values)
    if n == 0:
        raise ValueError("Empty list")
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / n)


def _to_pic50(value_nm: float) -> float:
    """Convert nM value to pIC50."""
    return -math.log10(value_nm * 1e-9)


# ---------------------------------------------------------------------------
# SOURCE 1 — PDBbind
# ---------------------------------------------------------------------------

def load_pdbbind_index(path: Path) -> dict[str, dict[str, Any]]:
    """
    Parse PDBbind index file.

    Returns dict keyed by lowercase PDB ID:
      {"pIC50": float, "affinity_type": "pKd"/"pKi", "affinity_source": "pdbbind"}
    """
    index: dict[str, dict[str, Any]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            pdb_id = parts[0].lower()
            try:
                neg_log_kd_ki = float(parts[3])
            except (ValueError, IndexError):
                continue
            # Parse affinity type from column 4, e.g. "Kd=3.8nM" or "Ki=120nM"
            raw_aff = parts[4] if len(parts) > 4 else ""
            if raw_aff.lower().startswith("kd"):
                aff_type = "pKd"
            elif raw_aff.lower().startswith("ki"):
                aff_type = "pKi"
            else:
                aff_type = "pKd"  # default

            index[pdb_id] = {
                "pIC50": neg_log_kd_ki,
                "affinity_type": aff_type,
                "affinity_source": "pdbbind",
                "n_measurements": 1,
                "affinity_notes": raw_aff,
            }
    return index


def find_pdbbind_path(cli_path: str | None) -> Path | None:
    if cli_path:
        p = Path(cli_path)
        return p if p.exists() else None
    for p in PDBBIND_DEFAULT_PATHS:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# SOURCE 2 — BindingDB
# ---------------------------------------------------------------------------

def query_bindingdb(pdb_id: str, session: requests.Session) -> dict[str, Any] | None:
    """
    Query BindingDB for a PDB ID.
    Returns affinity dict or None if not found / parse error.
    """
    url = BINDINGDB_URL.format(pdb_id=pdb_id.upper())
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            return None  # PDB not in BindingDB — expected for many entries
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("BindingDB request failed for %s: %s", pdb_id, exc)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("BindingDB JSON decode failed for %s: %s", pdb_id, exc)
        return None

    # Navigate to the affinities list — BindingDB's structure varies by version
    affinities = []
    try:
        outer = data.get("getLigandsByPDBResponse", {})
        # Could be a list or a single dict
        ligands = outer.get("affinities", outer.get("affinity", []))
        if isinstance(ligands, dict):
            ligands = [ligands]
    except AttributeError:
        return None

    ic50_vals: list[float] = []
    ki_vals: list[float] = []
    n_total = 0

    for entry in ligands:
        if not isinstance(entry, dict):
            continue
        # BindingDB uses different key names across API versions
        for ic50_key in ("IC50", "ic50"):
            raw = entry.get(ic50_key, "")
            if raw and raw not in ("", "N/A", "None"):
                try:
                    relation = entry.get("IC50_Relation", entry.get("relation", "="))
                    if relation.strip() != "=":
                        continue
                    ic50_nm = float(str(raw).replace(",", ""))
                    if ic50_nm > 0:
                        ic50_vals.append(ic50_nm)
                        n_total += 1
                except (ValueError, TypeError):
                    pass
        for ki_key in ("Ki", "ki"):
            raw = entry.get(ki_key, "")
            if raw and raw not in ("", "N/A", "None"):
                try:
                    relation = entry.get("Ki_Relation", entry.get("relation", "="))
                    if relation.strip() != "=":
                        continue
                    ki_nm = float(str(raw).replace(",", ""))
                    if ki_nm > 0:
                        ki_vals.append(ki_nm)
                        n_total += 1
                except (ValueError, TypeError):
                    pass

    # Prefer IC50; fall back to Ki
    if ic50_vals:
        gm = _geometric_mean(ic50_vals)
        pic50 = _to_pic50(gm)
        return {
            "pIC50": round(pic50, 3),
            "affinity_type": "IC50",
            "affinity_source": "bindingdb",
            "n_measurements": len(ic50_vals),
            "affinity_notes": f"geomean_IC50={gm:.2f}nM (n={len(ic50_vals)})",
        }
    if ki_vals:
        gm = _geometric_mean(ki_vals)
        pic50 = _to_pic50(gm)
        return {
            "pIC50": round(pic50, 3),
            "affinity_type": "Ki",
            "affinity_source": "bindingdb",
            "n_measurements": len(ki_vals),
            "affinity_notes": f"geomean_Ki={gm:.2f}nM (n={len(ki_vals)})",
        }
    return None


# ---------------------------------------------------------------------------
# SOURCE 3 — ChEMBL
# ---------------------------------------------------------------------------

def load_chembl_for_target(
    target: str, session: requests.Session
) -> list[dict[str, Any]]:
    """
    Fetch all pChEMBL activities for a target from ChEMBL.
    Returns list of activity dicts (raw from API).
    """
    chembl_id = CHEMBL_TARGET_IDS.get(target)
    if not chembl_id:
        return []

    all_activities: list[dict[str, Any]] = []
    url: str | None = CHEMBL_ACTIVITY_URL.format(chembl_id=chembl_id)

    while url:
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            logger.warning("ChEMBL request failed for %s: %s", target, exc)
            break

        activities = data.get("activities", [])
        all_activities.extend(activities)

        # Pagination
        page_meta = data.get("page_meta", {})
        next_url = page_meta.get("next")
        if next_url:
            # ChEMBL returns relative paths like /chembl/api/data/activity.json?...
            if next_url.startswith("http"):
                url = next_url
            else:
                url = f"https://www.ebi.ac.uk{next_url}"
        else:
            url = None

    return all_activities


def build_chembl_lookup(
    activities: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Build a lookup keyed by molecule_chembl_id → best pChEMBL value.
    Also build a secondary lookup by ligand name for approximate matching.
    Returns dict[molecule_chembl_id, affinity_dict].
    """
    molecule_data: dict[str, list[float]] = {}
    molecule_type: dict[str, str] = {}
    molecule_name: dict[str, str] = {}

    for act in activities:
        mol_id = act.get("molecule_chembl_id", "")
        pchembl = act.get("pchembl_value")
        std_type = act.get("standard_type", "IC50")
        name = act.get("molecule_pref_name") or act.get("compound_name", "")
        if not mol_id or pchembl is None:
            continue
        try:
            val = float(pchembl)
        except (ValueError, TypeError):
            continue
        molecule_data.setdefault(mol_id, []).append(val)
        molecule_type[mol_id] = std_type
        if name:
            molecule_name[mol_id] = str(name).upper()

    result: dict[str, dict[str, Any]] = {}
    for mol_id, vals in molecule_data.items():
        mean_val = sum(vals) / len(vals)
        result[mol_id] = {
            "pIC50": round(mean_val, 3),
            "affinity_type": molecule_type.get(mol_id, "IC50"),
            "affinity_source": "chembl_approximate",
            "n_measurements": len(vals),
            "affinity_notes": f"ChEMBL {mol_id} pchembl_value mean (n={len(vals)})",
            "_name": molecule_name.get(mol_id, ""),
        }
    return result


def match_chembl(
    ligand_id: str,
    chembl_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Match a ligand_id (PDB residue name) to a ChEMBL molecule by name.
    This is approximate and best-effort.
    """
    ligand_upper = ligand_id.upper()
    for mol_id, info in chembl_lookup.items():
        mol_name = info.get("_name", "")
        if mol_name and (ligand_upper in mol_name or mol_name in ligand_upper):
            return info
    return None


# ---------------------------------------------------------------------------
# Main annotation logic
# ---------------------------------------------------------------------------

def annotate(
    rows: list[dict[str, Any]],
    pdbbind_index: dict[str, dict[str, Any]],
    session: requests.Session,
) -> list[dict[str, Any]]:
    """
    Annotate each row with affinity data from the three sources.
    Modifies rows in-place and returns them.
    """
    # Pre-load ChEMBL data per target (one bulk call per target)
    print("\nPre-loading ChEMBL activity data for all targets …")
    chembl_by_target: dict[str, dict[str, dict[str, Any]]] = {}
    for target in CHEMBL_TARGET_IDS:
        activities = load_chembl_for_target(target, session)
        chembl_by_target[target] = build_chembl_lookup(activities)
        print(f"  {target}: {len(activities)} ChEMBL activities loaded")

    # Collect unique PDB IDs that need BindingDB lookup
    all_pdb_ids = [r["pdb_id"].lower() for r in rows]
    pdbbind_hit_ids = {pid for pid in all_pdb_ids if pid in pdbbind_index}
    bindingdb_needed = [pid for pid in dict.fromkeys(all_pdb_ids)
                        if pid not in pdbbind_hit_ids]

    print(f"\nQuerying BindingDB for {len(bindingdb_needed)} PDB IDs not in PDBbind …")
    bindingdb_cache: dict[str, dict[str, Any] | None] = {}
    for i, pdb_id in enumerate(bindingdb_needed, 1):
        result = query_bindingdb(pdb_id, session)
        bindingdb_cache[pdb_id] = result
        if i % 25 == 0 or i == len(bindingdb_needed):
            print(f"  BindingDB: {i}/{len(bindingdb_needed)}")
        time.sleep(BINDINGDB_DELAY)

    print(f"\nAnnotating {len(rows)} rows …")
    for row in rows:
        pdb_id_lower = row["pdb_id"].lower()
        target = row["target"]
        ligand_id = row.get("ligand_id", "")

        affinity: dict[str, Any] | None = None

        # --- Source 1: PDBbind ---
        if pdb_id_lower in pdbbind_index:
            affinity = pdbbind_index[pdb_id_lower]

        # --- Source 2: BindingDB ---
        if affinity is None:
            bdb = bindingdb_cache.get(pdb_id_lower)
            if bdb is not None:
                affinity = bdb

        # --- Source 3: ChEMBL (approximate) ---
        if affinity is None:
            chembl_lookup = chembl_by_target.get(target, {})
            chembl_match = match_chembl(ligand_id, chembl_lookup)
            if chembl_match is not None:
                affinity = chembl_match

        # Apply affinity data (or nulls)
        if affinity is not None:
            row["pIC50"] = affinity.get("pIC50", "")
            row["affinity_type"] = affinity.get("affinity_type", "")
            row["affinity_source"] = affinity.get("affinity_source", "")
            row["n_measurements"] = affinity.get("n_measurements", 1)
            row["affinity_notes"] = affinity.get("affinity_notes", "")
            row["has_affinity"] = True
            src = row["affinity_source"]
            n = row["n_measurements"]
            row["affinity_reliable"] = (
                src == "pdbbind" or (src == "bindingdb" and n >= 2)
            )
        else:
            row["pIC50"] = ""
            row["affinity_type"] = ""
            row["affinity_source"] = "none"
            row["n_measurements"] = 0
            row["affinity_notes"] = ""
            row["has_affinity"] = False
            row["affinity_reliable"] = False

    return rows


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

HIST_BINS = [
    (None, 5,  "<5"),
    (5,    6,  "5-6"),
    (6,    7,  "6-7"),
    (7,    8,  "7-8"),
    (8,    9,  "8-9"),
    (9,   10,  "9-10"),
    (10, None, ">10"),
]


def _text_histogram(values: list[float], width: int = 30) -> str:
    counts = []
    for lo, hi, label in HIST_BINS:
        n = sum(
            1 for v in values
            if (lo is None or v >= lo) and (hi is None or v < hi)
        )
        counts.append((label, n))
    total = sum(c for _, c in counts)
    if total == 0:
        return "(no data)"
    lines = []
    max_n = max(c for _, c in counts) or 1
    for label, n in counts:
        bar_len = int(n / max_n * width)
        bar = "#" * bar_len
        lines.append(f"    {label:>5} [{bar:<{width}}] {n}")
    return "\n".join(lines)


def print_summary(rows: list[dict[str, Any]]) -> None:
    import statistics

    targets = list(dict.fromkeys(r["target"] for r in rows))

    header = (
        f"  {'Target':<12}| {'Total':>7} | {'Has aff.':>9} | "
        f"{'PDBbind':>8} | {'BindingDB':>10} | {'ChEMBL':>7} | {'None':>6}"
    )
    divider = "  " + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 11 + "|" \
              + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 8

    print("\n" + "=" * 80)
    print("Per-target affinity annotation summary:")
    print(header)
    print(divider)

    for target in targets:
        t_rows = [r for r in rows if r["target"] == target]
        total = len(t_rows)
        has = sum(1 for r in t_rows if r["has_affinity"])
        pdb_n = sum(1 for r in t_rows if r["affinity_source"] == "pdbbind")
        bdb_n = sum(1 for r in t_rows if r["affinity_source"] == "bindingdb")
        chm_n = sum(1 for r in t_rows if r["affinity_source"] == "chembl_approximate")
        none_n = sum(1 for r in t_rows if not r["has_affinity"])
        print(
            f"  {target:<12}| {total:>7} | {has:>9} | "
            f"{pdb_n:>8} | {bdb_n:>10} | {chm_n:>7} | {none_n:>6}"
        )

    print()
    print("pIC50 distribution per target:")
    for target in targets:
        pics = [
            float(r["pIC50"]) for r in rows
            if r["target"] == target and r["pIC50"] != ""
        ]
        if pics:
            mean_v = statistics.mean(pics)
            std_v = statistics.pstdev(pics)
            min_v = min(pics)
            max_v = max(pics)
            print(f"\n  {target}: mean={mean_v:.2f}, std={std_v:.2f}, "
                  f"min={min_v:.2f}, max={max_v:.2f}")
            print(_text_histogram(pics))
        else:
            print(f"\n  {target}: no affinity data")

    print("=" * 80)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_input_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (fieldnames, rows)."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def save_output_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    out_fields = fieldnames + [c for c in EXTRA_COLUMNS if c not in fieldnames]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate FEP benchmark structures with experimental affinities."
    )
    parser.add_argument(
        "--pdbbind_index",
        metavar="PATH",
        default=None,
        help="Path to PDBbind index file (INDEX_refined_data.2020). "
             "Defaults to ~/Downloads/INDEX_refined_data.2020 or "
             "datasets/pdbbind_index.csv.",
    )
    parser.add_argument(
        "--input",
        default=str(INPUT_CSV),
        metavar="PATH",
        help=f"Input CSV (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_CSV),
        metavar="PATH",
        help=f"Output CSV (default: {OUTPUT_CSV})",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input CSV not found: {input_path}", file=sys.stderr)
        print(
            "Run datasets/fetch_fep_benchmark_structures.py first to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load input
    print(f"Loading {input_path} …")
    fieldnames, rows = load_input_csv(input_path)
    print(f"  {len(rows)} rows loaded.")

    # Load PDBbind index
    pdbbind_path = find_pdbbind_path(args.pdbbind_index)
    if pdbbind_path is not None:
        print(f"Loading PDBbind index from {pdbbind_path} …")
        pdbbind_index = load_pdbbind_index(pdbbind_path)
        print(f"  {len(pdbbind_index)} entries loaded.")
    else:
        print("PDBbind index not found — skipping source 1.")
        pdbbind_index = {}

    session = _session()

    # Annotate
    rows = annotate(rows, pdbbind_index, session)

    # Save
    save_output_csv(output_path, fieldnames, rows)

    # Summary
    print_summary(rows)


if __name__ == "__main__":
    main()
