#!/usr/bin/env python
"""
pull_chembl_affinity_data.py

Pull and curate ChEMBL bioactivity data for two GPCR targets (DRD4, 5HT2A)
to build a fine-tuning dataset for a structure-based affinity prediction
model (e.g. Boltz-2 LoRA fine-tuning).

Dependencies:
    pip install chembl-webresource-client pandas numpy rdkit
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from chembl_webresource_client.new_client import new_client


# Silence RDKit noisy parser warnings; we handle invalid SMILES explicitly.
RDLogger.DisableLog("rdkit.*")

TARGETS = {
    "CHEMBL219": "DRD4",
    "CHEMBL224": "5HT2A",
}

# ChEMBL field-name reference (verified against the live REST schema):
#
#   activity endpoint  -> per-measurement fields (one row = one data point)
#   assay endpoint     -> per-assay metadata (confidence_score lives here)
#
# IMPORTANT: ChEMBL silently DROPS fields in `.only(...)` that are not valid
# for the queried endpoint.  Putting an invalid name here (e.g. `year` or
# `confidence_score` on the activity endpoint) returns an empty column with
# no warning. Always validate with VITAL_FIELDS below after the fetch.

# Fields fetched from the activity endpoint. Verified present on real records.
ACTIVITY_FIELDS = [
    # identity
    "molecule_chembl_id",
    "canonical_smiles",
    # measurement
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",          # pre-computed -log10(IC50_M); useful sanity check
    # assay link
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    # data-quality flags from the activity endpoint itself
    "data_validity_comment",  # NULL means no flag; non-NULL = curator flagged bad
    "potential_duplicate",    # 0 = ok, 1 = ChEMBL thinks this is a duplicate
    "standard_flag",          # 1 = standardized in canonical units
    # target link
    "target_chembl_id",
    "target_pref_name",
    # provenance
    "document_chembl_id",
    "document_year",          # NB: NOT "year"; activity endpoint uses document_year
]

# Fields fetched separately from the assay endpoint, keyed by assay_chembl_id.
ASSAY_FIELDS = [
    "assay_chembl_id",
    "confidence_score",
    "assay_organism",
    "relationship_type",      # 'D' = direct single protein (matches our filter)
]

# Columns that MUST be present and at least partially populated after fetch.
# If any of these come back missing or 100% NaN we abort with a clear error
# (instead of silently zeroing out the dataset in a later filter).
VITAL_ACTIVITY_FIELDS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "assay_chembl_id",
    "assay_type",
    "target_chembl_id",
]
VITAL_ASSAY_FIELDS = [
    "assay_chembl_id",
    "confidence_score",
]

# Boltz-2 trained the affinity head on Ki, Kd, IC50, AC50, EC50, XC50 with an
# intra-assay pairwise Huber loss that cancels the Cheng-Prusoff offset, so
# mixing types is fine *provided* assay grouping is preserved.
STANDARD_TYPES = ["IC50", "Ki", "Kd", "EC50", "AC50", "XC50"]

# Minimum ChEMBL assay confidence per assay_type.  'B' (binding biochemical)
# maxes at 9; 7+ = direct single protein. 'F' (functional) maxes at 7; 6+ is
# the usual trustworthy cutoff.
MIN_CONFIDENCE = {"B": 7, "F": 6}


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
def _validate_schema(df: pd.DataFrame,
                     vital: List[str],
                     context: str) -> None:
    """Abort loudly if VITAL columns are missing or entirely NaN.

    ChEMBL's `.only()` silently drops unknown field names. Without this check
    a misspelled field name produces an empty column and downstream filters
    quietly drop every row. We catch that here with a precise error message.
    """
    if df.empty:
        raise RuntimeError(
            f"[{context}] DataFrame is empty after fetch; "
            "no records returned from ChEMBL."
        )
    missing = [c for c in vital if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[{context}] missing required columns: {missing}. "
            f"Got: {sorted(df.columns)}. "
            "This usually means a field name in the corresponding *_FIELDS "
            "list is not valid for that ChEMBL endpoint."
        )
    all_null = [c for c in vital if df[c].isna().all()]
    if all_null:
        raise RuntimeError(
            f"[{context}] required columns are entirely NaN: {all_null}. "
            "ChEMBL returned the column header but no values. Check the "
            "field name is correct for this endpoint and delete any stale "
            "cached .pkl files before re-running."
        )
    # Helpful coverage report.
    for c in vital:
        n_null = int(df[c].isna().sum())
        if n_null > 0:
            logging.info("[%s] column '%s' has %d/%d NaN values",
                         context, c, n_null, len(df))


# --------------------------------------------------------------------------- #
# Step 1: data pull
# --------------------------------------------------------------------------- #
def fetch_target_activities(target_chembl_id: str,
                            max_retries: int = 3,
                            backoff_seconds: int = 5) -> pd.DataFrame:
    """Query the ChEMBL activity endpoint for one target with all filters
    applied server-side. Returns a pandas DataFrame."""
    activity = new_client.activity

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logging.info("Querying ChEMBL activities for %s (attempt %d/%d)",
                         target_chembl_id, attempt, max_retries)
            qs = (
                activity
                .filter(target_chembl_id=target_chembl_id)
                .filter(standard_type__in=STANDARD_TYPES)
                .filter(standard_relation__in=["=", "<"])
                .filter(standard_units="nM")
                .filter(assay_type__in=["B", "F"])
                .filter(target_organism="Homo sapiens")
                # 'D' = direct single-protein assay-to-target relationship,
                # matching Boltz-2's "single protein" filter.
                .filter(target_relationship="D")
                .filter(standard_value__isnull=False)
                .filter(canonical_smiles__isnull=False)
                .only(ACTIVITY_FIELDS)
            )
            # Materialize the queryset. The client paginates lazily.
            records = list(qs)
            df = pd.DataFrame.from_records(records)
            logging.info("Retrieved %d raw records for %s", len(df),
                         target_chembl_id)
            _validate_schema(df, VITAL_ACTIVITY_FIELDS,
                             context=f"activity:{target_chembl_id}")
            return df
        except RuntimeError:
            # Schema validation errors should not be retried.
            raise
        except Exception as exc:  # broad: client raises many transient errors
            last_err = exc
            logging.warning("ChEMBL query failed for %s on attempt %d: %s",
                            target_chembl_id, attempt, exc)
            if attempt < max_retries:
                time.sleep(backoff_seconds)
    raise RuntimeError(
        f"Failed to query ChEMBL for {target_chembl_id} "
        f"after {max_retries} attempts: {last_err}"
    )


def fetch_assay_metadata(assay_ids: List[str],
                         use_cache: bool,
                         cache_path: str = "chembl_assays.pkl",
                         batch_size: int = 200) -> pd.DataFrame:
    """Fetch confidence_score and other per-assay metadata from the ChEMBL
    assay endpoint. Returns a DataFrame indexed by assay_chembl_id.

    Cached separately from activity data because activity caches and assay
    caches have independent invalidation lifecycles.
    """
    if use_cache and os.path.exists(cache_path):
        logging.info("Loading cached assay metadata from %s", cache_path)
        with open(cache_path, "rb") as fh:
            df = pickle.load(fh)
        # Top up any missing assays (e.g. if activity cache grew).
        missing = sorted(set(assay_ids) - set(df["assay_chembl_id"]))
        if not missing:
            _validate_schema(df, VITAL_ASSAY_FIELDS, context="assay:cached")
            return df
        logging.info("Cache missing %d assays; fetching incrementally.",
                     len(missing))
        new_df = _fetch_assays_in_batches(missing, batch_size)
        df = pd.concat([df, new_df], ignore_index=True, sort=False)
    else:
        df = _fetch_assays_in_batches(sorted(set(assay_ids)), batch_size)

    _validate_schema(df, VITAL_ASSAY_FIELDS, context="assay:fetched")
    if use_cache:
        logging.info("Caching %d assay records to %s", len(df), cache_path)
        with open(cache_path, "wb") as fh:
            pickle.dump(df, fh)
    return df


def _fetch_assays_in_batches(assay_ids: List[str],
                             batch_size: int) -> pd.DataFrame:
    assay = new_client.assay
    rows = []
    for i in range(0, len(assay_ids), batch_size):
        batch = assay_ids[i:i + batch_size]
        logging.info("Fetching assay metadata batch %d-%d / %d",
                     i, i + len(batch), len(assay_ids))
        qs = (
            assay
            .filter(assay_chembl_id__in=batch)
            .only(ASSAY_FIELDS)
        )
        rows.extend(list(qs))
    df = pd.DataFrame.from_records(rows)
    return df


def load_or_fetch(target_chembl_id: str, use_cache: bool) -> pd.DataFrame:
    label = TARGETS[target_chembl_id]
    cache_path = f"chembl_raw_{label}.pkl"
    if use_cache and os.path.exists(cache_path):
        logging.info("Loading cached raw data from %s", cache_path)
        with open(cache_path, "rb") as fh:
            df = pickle.load(fh)
        # Validate cached data so a stale pickle from an earlier (buggy) run
        # is caught immediately rather than corrupting curation.
        try:
            _validate_schema(df, VITAL_ACTIVITY_FIELDS,
                             context=f"activity:{target_chembl_id}:cached")
        except RuntimeError as e:
            raise RuntimeError(
                f"Cached activity data at {cache_path} is stale or corrupt: "
                f"{e}. Delete the file and re-run without --cache (or with "
                f"--cache to refresh)."
            ) from None
        return df

    df = fetch_target_activities(target_chembl_id)
    if use_cache:
        logging.info("Caching raw data to %s", cache_path)
        with open(cache_path, "wb") as fh:
            pickle.dump(df, fh)
    return df


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _log_filter(step: str, before: int, after: int) -> None:
    logging.info("[%s] rows before=%d  after=%d  dropped=%d",
                 step, before, after, before - after)


def _build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


# --------------------------------------------------------------------------- #
# Curation pipeline
# --------------------------------------------------------------------------- #
def curate(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all expected activity columns are present (some may be missing
    # only when a target returned zero rows; defensive only).
    for col in ACTIVITY_FIELDS:
        if col not in df.columns:
            df[col] = np.nan

    # ----- Step 1.5: data-quality flags from the activity endpoint ---------
    # Drop rows ChEMBL curators flagged as suspicious or duplicates.
    before = len(df)
    df = df[df["data_validity_comment"].isna()].copy()
    _log_filter("data_validity_comment is NULL", before, len(df))

    before = len(df)
    df["potential_duplicate"] = pd.to_numeric(
        df["potential_duplicate"], errors="coerce"
    ).fillna(0).astype(int)
    df = df[df["potential_duplicate"] == 0].copy()
    _log_filter("potential_duplicate == 0", before, len(df))

    # ----- Step 2: confidence score filter (per-assay-type threshold) -------
    # confidence_score lives on the ASSAY endpoint; it should have been merged
    # into this dataframe by main() under the column 'confidence_score'.
    before = len(df)
    if "confidence_score" not in df.columns:
        raise RuntimeError(
            "Column 'confidence_score' missing from curate() input. "
            "It must be merged in from the assay endpoint before curation."
        )
    df["assay_confidence_score"] = pd.to_numeric(
        df["confidence_score"], errors="coerce"
    )
    n_with_conf = int(df["assay_confidence_score"].notna().sum())
    logging.info("confidence_score non-null=%d / %d", n_with_conf, len(df))
    if n_with_conf == 0 and len(df) > 0:
        raise RuntimeError(
            "Every row has a NaN confidence_score after the assay-endpoint "
            "merge. Inspect the assay cache (chembl_assays.pkl) and the "
            "fetch_assay_metadata() function."
        )
    # 'B' (binding) needs >=7, 'F' (functional) needs >=6.  Rows with unknown
    # assay_type fall back to the stricter binding threshold.
    min_conf = df["assay_type"].map(MIN_CONFIDENCE).fillna(MIN_CONFIDENCE["B"])
    df = df[df["assay_confidence_score"] >= min_conf].copy()
    _log_filter("confidence_score (B>=7, F>=6)", before, len(df))

    # ----- Step 3: numeric coercion + range filter --------------------------
    before = len(df)
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["standard_value"].notna()
            & (df["standard_value"] > 0)
            & (df["standard_value"] <= 100_000)].copy()
    _log_filter("standard_value in (0, 100000] nM", before, len(df))

    # ----- Step 4: pIC50 / log10(uM) + censored flag ------------------------
    # pIC50 = -log10(value_M) is human-readable; the Boltz-2 affinity head
    # emits log10(IC50 in uM) = 6 - pIC50, with LOWER = stronger binding.
    # We carry both: pIC50 for inspection, log10_aff_uM for training.
    df["pIC50"] = -np.log10(df["standard_value"].astype(float) * 1e-9)
    df["log10_aff_uM"] = np.log10(df["standard_value"].astype(float) / 1000.0)
    df["is_censored"] = df["standard_relation"].astype(str) == "<"

    # ----- Step 5: PAINS filter ---------------------------------------------
    before = len(df)
    pains = _build_pains_catalog()

    mols: List[Chem.Mol | None] = [
        Chem.MolFromSmiles(s) if isinstance(s, str) else None
        for s in df["canonical_smiles"].tolist()
    ]
    pains_flag = []
    valid_mask = []
    for m in mols:
        if m is None:
            pains_flag.append(False)
            valid_mask.append(False)
        else:
            hit = pains.HasMatch(m)
            pains_flag.append(bool(hit))
            valid_mask.append(True)
    df["pains_flag"] = pains_flag
    df["_mol"] = mols
    df["_valid"] = valid_mask
    df = df[df["_valid"] & (~df["pains_flag"])].copy()
    _log_filter("PAINS + valid SMILES", before, len(df))

    # ----- Step 6: molecular weight -----------------------------------------
    before = len(df)
    df["mol_weight"] = df["_mol"].apply(Descriptors.MolWt)
    df = df[df["mol_weight"] <= 800].copy()
    _log_filter("mol_weight<=800", before, len(df))

    # ----- Step 7: deduplication --------------------------------------------
    # IMPORTANT: we keep `assay_chembl_id` in the dedup key so that the same
    # compound measured in two different assays remains as two separate rows.
    # Boltz-2's intra-assay pairwise Huber loss needs that assay structure to
    # cancel inter-assay offsets.  Only true replicates (same compound, same
    # assay, same readout type) are collapsed.
    before = len(df)
    group_cols = [
        "canonical_smiles", "standard_type",
        "target_chembl_id", "assay_chembl_id",
    ]

    def _agg(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        pic50_med = float(np.median(group["pIC50"]))
        pic50_std = float(np.std(group["pIC50"], ddof=1)) if n > 1 else np.nan
        log10_med = float(np.median(group["log10_aff_uM"]))
        keep = group.iloc[0].copy()
        keep["pIC50"] = pic50_med
        keep["pIC50_std"] = pic50_std
        keep["log10_aff_uM"] = log10_med
        keep["n_measurements"] = n
        # If any replicate was censored, conservatively keep censored flag
        keep["is_censored"] = bool(group["is_censored"].any())
        return keep

    df = (
        df.groupby(group_cols, as_index=False, sort=False, group_keys=False)
          .apply(_agg)
          .reset_index(drop=True)
    )
    _log_filter("dedup by (smiles, type, target, assay)", before, len(df))

    # InChIKey + second dedup pass (still per-assay to preserve grouping).
    def _inchikey(mol: Chem.Mol) -> str | None:
        try:
            return Chem.MolToInchiKey(mol)
        except Exception:
            return None

    df["inchikey"] = df["_mol"].apply(_inchikey)
    before = len(df)
    df = df[df["inchikey"].notna()].copy()
    df = (
        df.sort_values("n_measurements", ascending=False)
          .drop_duplicates(
              subset=[
                  "inchikey", "standard_type",
                  "target_chembl_id", "assay_chembl_id",
              ],
              keep="first",
          )
          .reset_index(drop=True)
    )
    _log_filter("dedup by (inchikey, type, target, assay)", before, len(df))

    # ----- Step 8: assay-level variance filter ------------------------------
    before_rows = len(df)
    assay_stats = (
        df.groupby("assay_chembl_id")["pIC50"]
          .agg(["count", "std"])
          .rename(columns={"count": "n", "std": "pic50_std"})
    )
    keep_assays = assay_stats.index[
        (assay_stats["n"] >= 5) & (assay_stats["pic50_std"] >= 0.5)
    ]
    n_assays_before = assay_stats.shape[0]
    n_assays_after = len(keep_assays)
    df = df[df["assay_chembl_id"].isin(keep_assays)].copy()
    logging.info(
        "[assay variance] assays before=%d after=%d dropped=%d",
        n_assays_before, n_assays_after, n_assays_before - n_assays_after,
    )
    _log_filter("assay variance (n>=5, std>=0.5)", before_rows, len(df))

    # ----- Step 9: canonical SMILES standardization -------------------------
    df["canonical_smiles_std"] = df["_mol"].apply(
        lambda m: Chem.MolToSmiles(m, canonical=True)
    )

    # ----- Step 10: binary labels -------------------------------------------
    df["binary_label"] = (df["pIC50"] >= 5.0).astype(int)
    df["binary_label_strict"] = (df["pIC50"] >= 6.0).astype(int)

    # ----- group_id: the unit of the intra-assay pairwise Huber loss --------
    df["group_id"] = df["assay_chembl_id"].astype(str)

    # Drop helper cols
    df = df.drop(columns=["_mol", "_valid", "pains_flag"])
    return df


# --------------------------------------------------------------------------- #
# Output + summary
# --------------------------------------------------------------------------- #
FINAL_COLUMNS = [
    "inchikey", "canonical_smiles_std", "molecule_chembl_id",
    "source_target", "target_chembl_id",
    "standard_type", "standard_relation", "standard_value", "standard_units",
    # `log10_aff_uM` is the value to feed LoRA training (matches the Boltz-2
    # affinity head convention: log10(IC50 in uM), LOWER = stronger).
    # `pIC50` is retained for human interpretation only.
    "log10_aff_uM",
    "pIC50", "pIC50_std", "n_measurements", "is_censored",
    "mol_weight",
    "binary_label", "binary_label_strict",
    # `group_id` is the intra-assay grouping key for the pairwise loss.
    "group_id",
    "assay_chembl_id", "assay_description", "assay_type",
    "assay_confidence_score",
    "document_chembl_id", "document_year",
]


def write_output(df: pd.DataFrame, output_path: str) -> None:
    # Ensure all final columns exist (some may be missing if df is empty).
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df_out = df[FINAL_COLUMNS].copy()
    df_out.to_csv(output_path, index=False)
    logging.info("Wrote %d rows to %s", len(df_out), output_path)


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("CURATION SUMMARY")
    print("=" * 72)

    if df.empty:
        print("No rows in final dataset.")
        return

    for tgt_label, sub in df.groupby("source_target"):
        print(f"\n--- Target: {tgt_label} ({sub['target_chembl_id'].iloc[0]}) ---")
        print(f"  rows                : {len(sub)}")
        print(f"  unique compounds    : {sub['inchikey'].nunique()}")
        print(f"  assays retained     : {sub['assay_chembl_id'].nunique()}")
        print(f"  censored (< values) : {int(sub['is_censored'].sum())}")

        type_counts = sub["standard_type"].value_counts()
        print("  measurement types:")
        for t in STANDARD_TYPES:
            print(f"    {t:>4} : {int(type_counts.get(t, 0))}")

        atype_counts = sub["assay_type"].value_counts()
        print("  assay types:")
        for at in ("B", "F"):
            print(f"    {at:>4} : {int(atype_counts.get(at, 0))}")

        p = sub["pIC50"]
        print("  pIC50 distribution:")
        print(f"    mean={p.mean():.3f}  std={p.std():.3f}  "
              f"min={p.min():.3f}  max={p.max():.3f}")
        print(f"    q25={p.quantile(0.25):.3f}  "
              f"q50={p.quantile(0.50):.3f}  "
              f"q75={p.quantile(0.75):.3f}")

        y = sub["log10_aff_uM"]
        print("  log10_aff_uM (Boltz-2 target scale, lower=stronger):")
        print(f"    mean={y.mean():.3f}  std={y.std():.3f}  "
              f"min={y.min():.3f}  max={y.max():.3f}")

        n = len(sub)
        f1 = sub["binary_label"].mean()
        f2 = sub["binary_label_strict"].mean()
        print(f"  binary_label (pIC50>=5.0)        : "
              f"{int(sub['binary_label'].sum())}/{n} active "
              f"(frac={f1:.3f})")
        print(f"  binary_label_strict (pIC50>=6.0) : "
              f"{int(sub['binary_label_strict'].sum())}/{n} active "
              f"(frac={f2:.3f})")

    print("\n" + "=" * 72)
    print(f"TOTAL rows: {len(df)}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull and curate ChEMBL affinity data for DRD4 and 5HT2A."
    )
    p.add_argument(
        "--cache", action="store_true",
        help="Cache raw API responses to .pkl files and reuse on rerun."
    )
    p.add_argument(
        "--output", default="chembl_D4_5HT2A_curated.csv",
        help="Output CSV filename (default: chembl_D4_5HT2A_curated.csv)."
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    frames = []
    for tgt_id, tgt_label in TARGETS.items():
        logging.info("=== Target %s (%s) ===", tgt_label, tgt_id)
        raw = load_or_fetch(tgt_id, use_cache=args.cache)
        if raw.empty:
            logging.warning("No records returned for %s", tgt_id)
            continue
        raw["source_target"] = tgt_label
        # Force target_chembl_id (some records may report subunit ids).
        raw["target_chembl_id"] = tgt_id
        frames.append(raw)

    if not frames:
        logging.error("No data retrieved for any target. Exiting.")
        return 1

    df_raw = pd.concat(frames, ignore_index=True, sort=False)
    logging.info("Combined raw rows: %d", len(df_raw))

    # ---- Enrich with assay-endpoint metadata (confidence_score, ...) -----
    unique_assays = sorted(df_raw["assay_chembl_id"].dropna().unique().tolist())
    logging.info("Fetching metadata for %d unique assays", len(unique_assays))
    df_assays = fetch_assay_metadata(unique_assays, use_cache=args.cache)
    # Drop any overlapping columns from df_raw before merge so the assay
    # columns are authoritative.
    overlap = [c for c in df_assays.columns
               if c != "assay_chembl_id" and c in df_raw.columns]
    if overlap:
        df_raw = df_raw.drop(columns=overlap)
    df_raw = df_raw.merge(df_assays, how="left", on="assay_chembl_id")
    n_missing_conf = int(df_raw["confidence_score"].isna().sum())
    if n_missing_conf:
        logging.warning(
            "%d/%d activity rows have no matched assay metadata "
            "(confidence_score will be NaN and they will be dropped).",
            n_missing_conf, len(df_raw),
        )

    df_curated = curate(df_raw)
    write_output(df_curated, args.output)
    print_summary(df_curated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
