"""
Results export for affinity rescoring.

Supports multiple output formats: JSON, JSONL, CSV, Parquet, SQLite, Excel.
All formats use a consistent schema with optional metadata.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from boltz.affinity_rescoring.models import (
    AffinityResult,
    BatchSummary,
    LigandScore,
    OutputFormat,
)

logger = logging.getLogger(__name__)


def compute_batch_summary(results: List[AffinityResult]) -> BatchSummary:
    """Compute summary statistics from a list of results."""
    summary = BatchSummary(total_processed=len(results))

    successful_preds = []
    total_time = 0.0

    for r in results:
        total_time += r.processing_time_ms
        if r.validation_status.value == "SUCCESS" and not math.isnan(r.affinity_pred):
            summary.successful += 1
            successful_preds.append(r.affinity_pred)
        elif r.validation_status.value == "FAILED":
            summary.failed += 1
        else:
            summary.warnings_count += 1

    summary.inference_time_total_s = total_time / 1000.0

    if successful_preds:
        import numpy as np
        arr = np.array(successful_preds)
        summary.mean_affinity = float(np.mean(arr))
        summary.std_affinity = float(np.std(arr))
        summary.min_affinity = float(np.min(arr))
        summary.max_affinity = float(np.max(arr))

    if total_time > 0:
        summary.throughput_complexes_per_second = len(results) / (total_time / 1000.0)

    return summary


class ResultsExporter:
    """
    Exports results in multiple formats with consistent schema.

    Supported formats:
    - JSON (human-readable, nested)
    - JSONL (streaming, one result per line)
    - CSV (tabular, spreadsheet-compatible)
    - Parquet (columnar, efficient storage)
    - SQLite (queryable, with indexes)
    - Excel (spreadsheet with sheets)
    """

    VERSION = "1.0"

    def __init__(
        self,
        model_checkpoint: str = "",
        include_metadata: bool = True,
    ):
        self.model_checkpoint = model_checkpoint
        self.include_metadata = include_metadata

    # ─── Generic Export ───────────────────────────────────────────────────

    def export(
        self,
        results: List[AffinityResult],
        output_path: str | Path,
        fmt: OutputFormat = OutputFormat.CSV,
        include_summary: bool = True,
    ) -> Path:
        """
        Export results in the specified format.

        Returns the actual output path used.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == OutputFormat.JSON:
            return self._export_json(results, path, include_summary)
        elif fmt == OutputFormat.JSONL:
            return self._export_jsonl(results, path)
        elif fmt == OutputFormat.CSV:
            return self._export_csv(results, path)
        elif fmt == OutputFormat.PARQUET:
            return self._export_parquet(results, path)
        elif fmt == OutputFormat.SQLITE:
            return self._export_sqlite(results, path)
        elif fmt == OutputFormat.EXCEL:
            return self._export_excel(results, path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    # ─── JSON ─────────────────────────────────────────────────────────────

    def _export_json(
        self, results: List[AffinityResult], path: Path, include_summary: bool
    ) -> Path:
        """Export as nested JSON."""
        output = {
            "version": self.VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_checkpoint": self.model_checkpoint,
            "affinity_rescoring_version": self.VERSION,
            "results": [self._result_to_json(r) for r in results],
        }

        if include_summary:
            summary = compute_batch_summary(results)
            output["summary"] = {
                "total_processed": summary.total_processed,
                "successful": summary.successful,
                "failed": summary.failed,
                "mean_affinity": _safe_float(summary.mean_affinity),
                "std_affinity": _safe_float(summary.std_affinity),
                "inference_time_total_s": round(summary.inference_time_total_s, 2),
                "throughput_complexes_per_second": round(
                    summary.throughput_complexes_per_second, 4
                ),
            }

        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Exported {len(results)} results to {path} (JSON)")
        return path

    def _result_to_json(self, r: AffinityResult) -> Dict[str, Any]:
        """Convert a single result to JSON-compatible dict."""
        d = {
            "id": r.id,
            "source_file": r.source_file,
            "protein_chain": r.protein_chain,
            "ligand_chains": r.ligand_chains,
            "affinity_pred": _safe_float(r.affinity_pred),
            "affinity_std": _safe_float(r.affinity_std),
            "affinity_probability_binary": _safe_float(r.affinity_probability_binary),
            "num_protein_residues": r.protein_residue_count,
            "num_ligand_atoms": r.ligand_atom_count,
            "inference_time_ms": round(r.inference_time_ms, 2),
            "validation_status": r.validation_status.value,
            "warnings": r.warnings,
        }

        if r.affinity_pred_value1 is not None:
            d["ensemble"] = {
                "affinity_pred_value1": _safe_float(r.affinity_pred_value1),
                "affinity_pred_value2": _safe_float(r.affinity_pred_value2),
                "affinity_probability_binary1": _safe_float(r.affinity_probability_binary1),
                "affinity_probability_binary2": _safe_float(r.affinity_probability_binary2),
            }

        if r.error_message:
            d["error_message"] = r.error_message

        return d

    # ─── JSONL ────────────────────────────────────────────────────────────

    def _export_jsonl(self, results: List[AffinityResult], path: Path) -> Path:
        """Export as JSONL (one JSON object per line)."""
        with open(path, "w") as f:
            for r in results:
                line = json.dumps(self._result_to_json(r), default=str)
                f.write(line + "\n")

        logger.info(f"Exported {len(results)} results to {path} (JSONL)")
        return path

    # ─── CSV ──────────────────────────────────────────────────────────────

    def _export_csv(self, results: List[AffinityResult], path: Path) -> Path:
        """Export as CSV."""
        if not results:
            path.touch()
            return path

        rows = [r.to_dict() for r in results]
        fieldnames = list(rows[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported {len(results)} results to {path} (CSV)")
        return path

    # ─── Parquet ──────────────────────────────────────────────────────────

    def _export_parquet(self, results: List[AffinityResult], path: Path) -> Path:
        """Export as Parquet (requires pyarrow or polars)."""
        try:
            import polars as pl
        except ImportError:
            try:
                import pandas as pd
                rows = [r.to_dict() for r in results]
                df = pd.DataFrame(rows)
                df.to_parquet(str(path), index=False)
                logger.info(f"Exported {len(results)} results to {path} (Parquet via pandas)")
                return path
            except ImportError:
                raise ImportError(
                    "Parquet export requires polars or pandas+pyarrow. "
                    "Install via: pip install polars  OR  pip install pandas pyarrow"
                )

        rows = [r.to_dict() for r in results]
        df = pl.DataFrame(rows)
        df.write_parquet(str(path))
        logger.info(f"Exported {len(results)} results to {path} (Parquet)")
        return path

    # ─── SQLite ───────────────────────────────────────────────────────────

    def _export_sqlite(self, results: List[AffinityResult], path: Path) -> Path:
        """Export as SQLite database."""
        if path.exists():
            path.unlink()

        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE results (
                id TEXT PRIMARY KEY,
                source_file TEXT,
                affinity_pred REAL,
                affinity_std REAL,
                affinity_probability_binary REAL,
                protein_chain TEXT,
                ligand_chains TEXT,
                protein_residue_count INTEGER,
                ligand_atom_count INTEGER,
                processing_time_ms REAL,
                featurization_time_ms REAL,
                inference_time_ms REAL,
                validation_status TEXT,
                validation_issues TEXT,
                warnings TEXT,
                error_message TEXT
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX idx_affinity ON results(affinity_pred)")
        cursor.execute("CREATE INDEX idx_status ON results(validation_status)")

        # Insert results
        for r in results:
            d = r.to_dict()
            cursor.execute(
                "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                tuple(d.values()),
            )

        # Metadata table
        cursor.execute("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("version", self.VERSION),
        )
        cursor.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("timestamp", datetime.now(timezone.utc).isoformat()),
        )
        cursor.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("model_checkpoint", self.model_checkpoint),
        )
        cursor.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("total_results", str(len(results))),
        )

        conn.commit()
        conn.close()

        logger.info(f"Exported {len(results)} results to {path} (SQLite)")
        return path

    # ─── Excel ────────────────────────────────────────────────────────────

    def _export_excel(self, results: List[AffinityResult], path: Path) -> Path:
        """Export as Excel with multiple sheets."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Excel export requires pandas and openpyxl. "
                "Install via: pip install pandas openpyxl"
            )

        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)

        # Separate successful and failed
        success_df = df[df["validation_status"] == "SUCCESS"]
        failed_df = df[df["validation_status"] == "FAILED"]

        with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All Results", index=False)
            if not success_df.empty:
                success_df.to_excel(writer, sheet_name="Successful", index=False)
            if not failed_df.empty:
                failed_df.to_excel(writer, sheet_name="Failed", index=False)

            # Metadata sheet
            summary = compute_batch_summary(results)
            meta_data = {
                "Key": [
                    "Version", "Timestamp", "Model Checkpoint",
                    "Total Processed", "Successful", "Failed",
                    "Mean Affinity", "Std Affinity",
                    "Total Time (s)", "Throughput (complexes/s)",
                ],
                "Value": [
                    self.VERSION,
                    datetime.now(timezone.utc).isoformat(),
                    self.model_checkpoint,
                    summary.total_processed,
                    summary.successful,
                    summary.failed,
                    _safe_float(summary.mean_affinity),
                    _safe_float(summary.std_affinity),
                    round(summary.inference_time_total_s, 2),
                    round(summary.throughput_complexes_per_second, 4),
                ],
            }
            pd.DataFrame(meta_data).to_excel(
                writer, sheet_name="Metadata", index=False
            )

        logger.info(f"Exported {len(results)} results to {path} (Excel)")
        return path

    # ─── Receptor Rescoring Export ────────────────────────────────────────

    def export_receptor_results(
        self,
        results: List[LigandScore],
        output_path: str | Path,
        fmt: OutputFormat = OutputFormat.CSV,
        include_failed: bool = True,
        sort_by: str = "affinity_score",
        descending: bool = False,
    ) -> Path:
        """
        Export receptor-based rescoring results.

        Parameters
        ----------
        results : list of LigandScore
            Scoring results per ligand
        output_path : str or Path
            Output file path
        fmt : OutputFormat
            Export format
        include_failed : bool
            Include failed ligands in output
        sort_by : str
            Column to sort by
        descending : bool
            Sort descending
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Filter
        if not include_failed:
            results = [r for r in results if r.validation_status.value != "FAILED"]

        # Sort
        if sort_by and results:
            reverse = descending
            try:
                results = sorted(
                    results,
                    key=lambda r: (
                        getattr(r, sort_by)
                        if not math.isnan(getattr(r, sort_by, float("nan")))
                        else float("inf")
                    ),
                    reverse=reverse,
                )
            except (AttributeError, TypeError):
                logger.warning(f"Cannot sort by '{sort_by}'. Skipping sort.")

        if fmt == OutputFormat.CSV:
            return self._export_receptor_csv(results, path)
        elif fmt == OutputFormat.EXCEL:
            return self._export_receptor_excel(results, path)
        elif fmt == OutputFormat.PARQUET:
            return self._export_receptor_parquet(results, path)
        elif fmt == OutputFormat.JSON:
            return self._export_receptor_json(results, path)
        else:
            # Default to CSV for unsupported formats
            return self._export_receptor_csv(results, path)

    def _export_receptor_csv(self, results: List[LigandScore], path: Path) -> Path:
        """Export receptor results as CSV."""
        if not results:
            path.touch()
            return path

        rows = [r.to_dict() for r in results]
        fieldnames = list(rows[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported {len(results)} ligand scores to {path} (CSV)")
        return path

    def _export_receptor_excel(self, results: List[LigandScore], path: Path) -> Path:
        """Export receptor results as Excel."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Excel export requires pandas+openpyxl. "
                "Install via: pip install pandas openpyxl"
            )

        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)

        success_df = df[df["validation_status"] == "SUCCESS"]
        failed_df = df[df["validation_status"] == "FAILED"]

        with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Results", index=False)
            if not success_df.empty:
                success_df.to_excel(writer, sheet_name="Hits", index=False)
            if not failed_df.empty:
                failed_df.to_excel(writer, sheet_name="Issues", index=False)

        logger.info(f"Exported {len(results)} ligand scores to {path} (Excel)")
        return path

    def _export_receptor_parquet(self, results: List[LigandScore], path: Path) -> Path:
        """Export receptor results as Parquet."""
        try:
            import polars as pl
            rows = [r.to_dict() for r in results]
            df = pl.DataFrame(rows)
            df.write_parquet(str(path))
        except ImportError:
            import pandas as pd
            rows = [r.to_dict() for r in results]
            df = pd.DataFrame(rows)
            df.to_parquet(str(path), index=False)

        logger.info(f"Exported {len(results)} ligand scores to {path} (Parquet)")
        return path

    def _export_receptor_json(self, results: List[LigandScore], path: Path) -> Path:
        """Export receptor results as JSON."""
        output = {
            "version": self.VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_checkpoint": self.model_checkpoint,
            "total_ligands": len(results),
            "successful": sum(1 for r in results if r.validation_status.value == "SUCCESS"),
            "failed": sum(1 for r in results if r.validation_status.value == "FAILED"),
            "results": [r.to_dict() for r in results],
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Exported {len(results)} ligand scores to {path} (JSON)")
        return path


def _safe_float(v: float) -> Any:
    """Convert float to JSON-safe value (None for NaN/Inf)."""
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, 4)
