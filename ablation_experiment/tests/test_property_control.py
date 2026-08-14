"""Unit tests for compute_property_control.py (Task 5): descriptor/
fingerprint computation, residual regression, and AVE bias. Uses real RDKit
on synthetic SMILES and real scikit-learn -- no boltz/MOL2/GPU dependency,
unlike the data-loading layer (load_ligand_smiles_for_receptor), which needs
the full environment and isn't covered here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
sklearn = pytest.importorskip("sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis_scripts"))
import compute_property_control as pc  # noqa: E402

BENZENE = "c1ccccc1"
ETHANOL = "CCO"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
ACETATE = "CC(=O)[O-]"  # formal charge -1


class TestLigandProperties:
    def test_heavy_atom_count(self):
        props = pc.compute_ligand_properties(BENZENE)
        assert props["heavy_atom_count"] == 6.0

    def test_formal_charge(self):
        props = pc.compute_ligand_properties(ACETATE)
        assert props["formal_charge"] == -1.0

    def test_neutral_molecule_zero_charge(self):
        props = pc.compute_ligand_properties(ETHANOL)
        assert props["formal_charge"] == 0.0

    def test_invalid_smiles_returns_none(self):
        assert pc.compute_ligand_properties("not a smiles!!") is None

    def test_larger_molecule_has_more_heavy_atoms(self):
        p_small = pc.compute_ligand_properties(ETHANOL)
        p_big = pc.compute_ligand_properties(CAFFEINE)
        assert p_big["heavy_atom_count"] > p_small["heavy_atom_count"]


class TestECFP4:
    def test_returns_correct_shape_and_dtype(self):
        fp = pc.compute_ecfp4(BENZENE)
        assert fp.shape == (2048,)
        assert set(np.unique(fp)).issubset({0, 1})

    def test_invalid_smiles_returns_none(self):
        assert pc.compute_ecfp4("garbage!!") is None

    def test_identical_molecules_give_identical_fingerprints(self):
        fp1 = pc.compute_ecfp4(BENZENE)
        fp2 = pc.compute_ecfp4("C1=CC=CC=C1")  # same molecule, different SMILES
        assert np.array_equal(fp1, fp2)

    def test_different_molecules_give_different_fingerprints(self):
        fp1 = pc.compute_ecfp4(BENZENE)
        fp2 = pc.compute_ecfp4(CAFFEINE)
        assert not np.array_equal(fp1, fp2)


class TestTanimotoNN:
    def test_identical_sets_give_similarity_one(self):
        fps = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.int8)
        sim = pc._tanimoto_nn_mean(fps, fps)
        assert sim == pytest.approx(1.0)

    def test_disjoint_bits_give_similarity_zero(self):
        query = np.array([[1, 1, 0, 0]], dtype=np.int8)
        ref = np.array([[0, 0, 1, 1]], dtype=np.int8)
        assert pc._tanimoto_nn_mean(query, ref) == pytest.approx(0.0)

    def test_empty_input_returns_nan(self):
        fps = np.zeros((0, 4), dtype=np.int8)
        ref = np.array([[1, 0, 0, 0]], dtype=np.int8)
        assert np.isnan(pc._tanimoto_nn_mean(fps, ref))
        assert np.isnan(pc._tanimoto_nn_mean(ref, fps))


class TestAveBias:
    def test_unbiased_split_near_zero(self):
        rng = np.random.default_rng(0)
        X = (rng.random((40, 64)) > 0.5).astype(np.int8)
        y = np.array([1, 0] * 20)
        # random split -> train/test similarity structure shouldn't
        # systematically favor within-class matches
        idx = rng.permutation(40)
        train_idx, test_idx = idx[:30], idx[30:]
        bias = pc.ave_bias(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        assert abs(bias) < 0.5

    def test_clustered_split_shows_positive_bias(self):
        # actives cluster tightly around one fingerprint, decoys around a
        # very different one -> train/test split within each cluster should
        # show strong bias (test actives look like train actives, not decoys)
        rng = np.random.default_rng(1)
        base_active = (rng.random(128) > 0.5).astype(np.int8)
        base_decoy = 1 - base_active  # maximally different bit pattern

        def jitter(base, n, flip_frac=0.02):
            out = np.tile(base, (n, 1)).copy()
            for row in out:
                flip_idx = rng.choice(len(base), size=int(len(base) * flip_frac), replace=False)
                row[flip_idx] = 1 - row[flip_idx]
            return out

        A = jitter(base_active, 20)
        D = jitter(base_decoy, 20)
        X = np.vstack([A, D])
        y = np.array([1] * 20 + [0] * 20)
        idx = rng.permutation(40)
        train_idx, test_idx = idx[:30], idx[30:]
        bias = pc.ave_bias(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        assert bias > 0.3


class TestResidualLogauc:
    def test_too_few_compounds_returns_nan(self):
        scores = {"a": 1.0, "b": 2.0}
        properties = {"a": {"heavy_atom_count": 10, "clogp": 1, "tpsa": 20, "formal_charge": 0},
                       "b": {"heavy_atom_count": 12, "clogp": 2, "tpsa": 25, "formal_charge": 0}}
        logauc, r2, n = pc.compute_residual_logauc(scores, properties, {"a"}, {"b"}, min_compounds=8)
        assert np.isnan(logauc)
        assert n == 2

    def test_perfectly_size_explained_score_gives_near_random_residual(self):
        # score is *exactly* a linear function of heavy_atom_count -> after
        # regressing it out, nothing should be left to separate actives from
        # decoys (residuals ~ 0, ties), so residual logAUC should be far
        # below the max the raw (unregressed) score would show.
        rng = np.random.default_rng(2)
        names = [f"lig{i}" for i in range(10)] + [f"ZINC{i:06d}" for i in range(10)]
        lig_set = {n for n in names if not n.startswith("ZINC")}
        dec_set = {n for n in names if n.startswith("ZINC")}
        heavy_atoms = {n: float(rng.integers(10, 40)) for n in names}
        scores = {n: 2.0 * heavy_atoms[n] + 5.0 for n in names}  # pure linear function, no noise
        properties = {n: {"heavy_atom_count": heavy_atoms[n], "clogp": 0.0, "tpsa": 0.0, "formal_charge": 0.0}
                       for n in names}
        logauc, r2, n_used = pc.compute_residual_logauc(scores, properties, lig_set, dec_set, min_compounds=5)
        assert r2 == pytest.approx(1.0, abs=1e-6)
        assert n_used == 20


class Test2DClassifierAndAve:
    def test_returns_nan_for_too_few_compounds(self):
        fingerprints = {"a": np.zeros(8, dtype=np.int8), "b": np.ones(8, dtype=np.int8)}
        result = pc.compute_2d_logauc_and_ave(fingerprints, {"a"}, {"b"}, n_splits=5)
        assert np.isnan(result["logauc_2d"])

    def test_separable_fingerprints_score_above_random(self):
        rng = np.random.default_rng(3)
        base_active = (rng.random(256) > 0.5).astype(np.int8)
        base_decoy = 1 - base_active

        def jitter(base, n, prefix, flip_frac=0.05):
            out = {}
            for i in range(n):
                row = base.copy()
                flip_idx = rng.choice(len(base), size=int(len(base) * flip_frac), replace=False)
                row[flip_idx] = 1 - row[flip_idx]
                out[f"{prefix}{i}"] = row
            return out

        actives = jitter(base_active, 30, "lig")
        decoys = jitter(base_decoy, 30, "ZINC")
        fingerprints = {**actives, **decoys}
        lig_set, dec_set = set(actives), set(decoys)

        result = pc.compute_2d_logauc_and_ave(fingerprints, lig_set, dec_set, n_splits=5, seed=0)
        assert result["logauc_2d"] > 20.0  # clearly separable -> well above random (0)
        assert result["n_compounds"] == 60
