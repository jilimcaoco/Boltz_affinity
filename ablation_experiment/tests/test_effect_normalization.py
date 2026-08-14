"""Unit tests for compute_effect_normalization.py (Task 6): NAE math,
receptor-class lookup/override, per-class aggregation, and rank
preservation. Pure pandas/numpy -- no GPU or real ablation data needed.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis_scripts"))
import compute_effect_normalization as en  # noqa: E402


def _summary(rows):
    return pd.DataFrame(rows)


class TestComputeNAE:
    def test_basic_nae_math(self):
        summary = _summary([
            {"receptor": "R1", "experiment": "baseline", "point_estimate_logAUC": 40.0},
            {"receptor": "R1", "experiment": "bias_only", "point_estimate_logAUC": 0.0},
            {"receptor": "R1", "experiment": "no_distogram", "point_estimate_logAUC": 36.0},
            {"receptor": "R1", "experiment": "no_z_trunk", "point_estimate_logAUC": 10.0},
            {"receptor": "R1", "experiment": "no_s_inputs", "point_estimate_logAUC": 20.0},
        ])
        nae_df = en.compute_nae(summary)
        row = nae_df[(nae_df["receptor"] == "R1") & (nae_df["channel"] == "distogram")].iloc[0]
        # denom = 40 - 0 = 40; numer = 40 - 36 = 4; NAE = 4/40 = 0.1
        assert row["NAE"] == pytest.approx(0.1)
        row_z = nae_df[(nae_df["receptor"] == "R1") & (nae_df["channel"] == "z_trunk")].iloc[0]
        assert row_z["NAE"] == pytest.approx(30.0 / 40.0)

    def test_missing_baseline_or_bias_only_raises(self):
        summary = _summary([
            {"receptor": "R1", "experiment": "no_distogram", "point_estimate_logAUC": 36.0},
        ])
        with pytest.raises(ValueError):
            en.compute_nae(summary)

    def test_near_zero_denominator_is_flagged_not_divided(self):
        summary = _summary([
            {"receptor": "R1", "experiment": "baseline", "point_estimate_logAUC": 0.3},
            {"receptor": "R1", "experiment": "bias_only", "point_estimate_logAUC": 0.1},
            {"receptor": "R1", "experiment": "no_distogram", "point_estimate_logAUC": 0.2},
        ])
        nae_df = en.compute_nae(summary)
        row = nae_df.iloc[0]
        assert row["denominator_near_zero_flag"] is True or row["denominator_near_zero_flag"] == True  # noqa: E712
        assert np.isnan(row["NAE"])

    def test_missing_ablation_experiment_is_skipped_not_errored(self):
        summary = _summary([
            {"receptor": "R1", "experiment": "baseline", "point_estimate_logAUC": 40.0},
            {"receptor": "R1", "experiment": "bias_only", "point_estimate_logAUC": 0.0},
            {"receptor": "R1", "experiment": "no_distogram", "point_estimate_logAUC": 36.0},
            # no_z_trunk, no_s_inputs missing entirely
        ])
        nae_df = en.compute_nae(summary)
        assert set(nae_df["channel"]) == {"distogram"}


class TestReceptorClassification:
    def test_known_receptor_from_builtin_table(self):
        class_map = en.load_receptor_class_map(None)
        assert class_map["AA2AR"] == "GPCR"
        assert class_map["CDK2"] == "kinase"
        assert class_map["ESR1"] == "nuclear_receptor"

    def test_unknown_receptor_defaults_to_other_with_warning(self):
        class_map = en.load_receptor_class_map(None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = en.classify_receptors(["TOTALLY_UNKNOWN_TARGET"], class_map)
        assert df.iloc[0]["receptor_class"] == "other"
        assert any("TOTALLY_UNKNOWN_TARGET" in str(w.message) for w in caught)

    def test_override_csv_takes_precedence(self, tmp_path):
        override_path = tmp_path / "override.csv"
        pd.DataFrame([{"receptor": "AA2AR", "receptor_class": "custom_class"}]).to_csv(override_path, index=False)
        class_map = en.load_receptor_class_map(override_path)
        assert class_map["AA2AR"] == "custom_class"
        # unrelated entries from the built-in table survive
        assert class_map["CDK2"] == "kinase"


class TestPerClassBreakdown:
    def test_aggregates_by_class(self):
        nae_df = _summary([
            {"receptor": "AA2AR", "channel": "z_trunk", "NAE": 0.8},   # GPCR
            {"receptor": "CDK2", "channel": "z_trunk", "NAE": 0.6},    # kinase
            {"receptor": "ABL1", "channel": "z_trunk", "NAE": 0.4},    # kinase
        ])
        class_df = _summary([
            {"receptor": "AA2AR", "receptor_class": "GPCR"},
            {"receptor": "CDK2", "receptor_class": "kinase"},
            {"receptor": "ABL1", "receptor_class": "kinase"},
        ])
        result = en.per_class_breakdown(nae_df, "NAE", class_df, ["channel"])
        kinase_row = result[result["receptor_class"] == "kinase"].iloc[0]
        assert kinase_row["NAE_mean"] == pytest.approx(0.5)
        assert kinase_row["n_receptors"] == 2
        gpcr_row = result[result["receptor_class"] == "GPCR"].iloc[0]
        assert gpcr_row["NAE_mean"] == pytest.approx(0.8)


class TestRankPreservation:
    def test_identical_scores_give_rho_one(self):
        rows = []
        for i in range(10):
            rows.append({"receptor_id": "R1", "experiment": "baseline", "ligand_id": f"lig{i}",
                         "affinity_pred_value": float(i)})
            rows.append({"receptor_id": "R1", "experiment": "no_z_trunk", "ligand_id": f"lig{i}",
                         "affinity_pred_value": float(i)})
        df = pd.DataFrame(rows)
        rank_df = en.compute_rank_preservation(df)
        assert rank_df.iloc[0]["spearman_rho"] == pytest.approx(1.0)

    def test_reversed_ranks_give_rho_negative_one(self):
        rows = []
        for i in range(10):
            rows.append({"receptor_id": "R1", "experiment": "baseline", "ligand_id": f"lig{i}",
                         "affinity_pred_value": float(i)})
            rows.append({"receptor_id": "R1", "experiment": "no_z_trunk", "ligand_id": f"lig{i}",
                         "affinity_pred_value": float(9 - i)})
        df = pd.DataFrame(rows)
        rank_df = en.compute_rank_preservation(df)
        assert rank_df.iloc[0]["spearman_rho"] == pytest.approx(-1.0)

    def test_too_few_common_compounds_skipped(self):
        rows = [
            {"receptor_id": "R1", "experiment": "baseline", "ligand_id": "lig0", "affinity_pred_value": 1.0},
            {"receptor_id": "R1", "experiment": "no_z_trunk", "ligand_id": "lig0", "affinity_pred_value": 1.0},
        ]
        df = pd.DataFrame(rows)
        rank_df = en.compute_rank_preservation(df)
        assert rank_df.empty
