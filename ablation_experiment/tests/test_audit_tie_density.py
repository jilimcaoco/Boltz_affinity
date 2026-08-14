"""Unit tests for audit_tie_density.py's classification rule (Task 0b).

The interesting case is small receptors: the spec's absolute "<100 distinct
= suspect" threshold is a good coarseness proxy on big receptors but
misfires on small ones, where having fewer than 100 distinct scores is
forced by the compound count rather than by ties.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis_scripts"))
import audit_tie_density as atd  # noqa: E402


class TestClassify:
    def test_very_few_distinct_is_unusable(self):
        assert atd.classify(distinct_values=3, tied_top1pct_flag=False, n_scores=400) == "unusable"

    def test_unusable_is_absolute_even_on_tiny_receptors(self):
        # 5 distinct out of 5 compounds: no ties, but a 5-level ranking is
        # uninterpretable regardless of why it's coarse.
        assert atd.classify(distinct_values=5, tied_top1pct_flag=False, n_scores=5) == "unusable"

    def test_coarse_ranking_with_real_ties_is_suspect(self):
        # 50 distinct across 400 compounds -> heavy duplication
        assert atd.classify(distinct_values=50, tied_top1pct_flag=False, n_scores=400) == "suspect"

    def test_small_receptor_with_no_ties_is_ok(self):
        # Regression: 80 distinct / 80 compounds is zero ties, must not be
        # flagged just for being under the absolute 100 threshold.
        assert atd.classify(distinct_values=80, tied_top1pct_flag=False, n_scores=80) == "ok"

    def test_small_receptor_with_ties_is_still_suspect(self):
        assert atd.classify(distinct_values=40, tied_top1pct_flag=False, n_scores=80) == "suspect"

    def test_large_receptor_fully_distinct_is_ok(self):
        assert atd.classify(distinct_values=400, tied_top1pct_flag=False, n_scores=400) == "ok"

    def test_tied_top1pct_always_flags(self):
        # plenty of distinct values overall, but the top of the ranking is
        # one tie block -> the exact failure mode of the bug
        assert atd.classify(distinct_values=400, tied_top1pct_flag=True, n_scores=400) == "suspect"


class TestAuditAblationCsv:
    def test_missing_file_raises_with_actionable_message(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            atd.audit_ablation_csv(tmp_path / "nope.csv")
        assert "run_feature_ablation.py" in str(exc.value)

    def test_constant_scores_classified_unusable(self, tmp_path):
        rows = []
        for i in range(60):
            name = f"ZINC{i:06d}" if i % 2 == 0 else f"lig{i}"
            rows.append({"experiment": "bias_only", "receptor_id": "R1", "ligand_name": name,
                          "affinity_pred_value": 5.0, "error": None})
        csv_path = tmp_path / "abl.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        result = atd.audit_ablation_csv(csv_path)
        assert len(result) == 1
        assert result[0]["status"] == "unusable"
        assert result[0]["distinct_values"] == 1
        assert result[0]["condition"] == "bias_only"

    def test_well_separated_scores_classified_ok(self, tmp_path):
        rows = []
        for i in range(60):
            name = f"ZINC{i:06d}" if i % 2 == 0 else f"lig{i}"
            rows.append({"experiment": "baseline", "receptor_id": "R1", "ligand_name": name,
                          "affinity_pred_value": float(i), "error": None})
        csv_path = tmp_path / "abl.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        result = atd.audit_ablation_csv(csv_path)
        assert result[0]["status"] == "ok"
        assert result[0]["distinct_values"] == 60

    def test_rows_with_errors_are_excluded(self, tmp_path):
        rows = []
        for i in range(40):
            name = f"ZINC{i:06d}" if i % 2 == 0 else f"lig{i}"
            rows.append({"experiment": "baseline", "receptor_id": "R1", "ligand_name": name,
                          "affinity_pred_value": float(i), "error": None})
        # 20 failed rows that must not count toward the tie statistics
        for i in range(40, 60):
            rows.append({"experiment": "baseline", "receptor_id": "R1", "ligand_name": f"lig{i}",
                          "affinity_pred_value": None, "error": "boom"})
        csv_path = tmp_path / "abl.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        result = atd.audit_ablation_csv(csv_path)
        assert result[0]["n"] == 40
