"""
Unit tests for distogram masking in AffinityModule.

Tests the masking logic in isolation — constructs the same masks and
tensor operations used in AffinityModule.forward() and verifies that
each mode zeros the correct region of the embedded distogram tensor.
"""

import pytest
import torch


# ── Helpers that replicate the masking logic from AffinityModule.forward() ──

def apply_distogram_mask(distogram, d, rec, lig, mode, distance_cutoff=None):
    """
    Apply distogram masking exactly as AffinityModule.forward() does.

    Parameters
    ----------
    distogram : Tensor [B, N, N, D]
        Embedded distogram (output of dist_bin_pairwise_embed).
    d : Tensor [B, N, N]
        Raw pairwise distances (only used for distance_cutoff mode).
    rec : Tensor [B, N]
        Boolean receptor mask (True for receptor tokens).
    lig : Tensor [B, N]
        Boolean ligand mask (True for ligand tokens).
    mode : str
        Masking mode.
    distance_cutoff : float, optional
        Cutoff in Å for distance_cutoff mode.

    Returns
    -------
    Tensor [B, N, N, D]
        Masked distogram.
    """
    if mode == "none":
        return distogram
    elif mode == "zero_all":
        return torch.zeros_like(distogram)
    elif mode == "zero_cross":
        cross = (
            lig[:, :, None] * rec[:, None, :]
            + rec[:, :, None] * lig[:, None, :]
        )
        return distogram * (1 - cross.unsqueeze(-1).float())
    elif mode == "zero_ligand":
        any_lig = lig[:, :, None] | lig[:, None, :]
        return distogram * (~any_lig).unsqueeze(-1).float()
    elif mode == "zero_receptor":
        rec_rec = rec[:, :, None] * rec[:, None, :]
        return distogram * (1 - rec_rec.unsqueeze(-1).float())
    elif mode == "distance_cutoff":
        cutoff = distance_cutoff if distance_cutoff is not None else 8.0
        beyond = (d > cutoff).unsqueeze(-1).float()
        return distogram * (1 - beyond)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def setup_tensors():
    """
    Build a small synthetic system:
      - 4 receptor tokens (indices 0-3)
      - 2 ligand tokens (indices 4-5)
      - embedding dim = 8
    All distogram values are non-zero (filled with ones) so any
    zeroing is immediately detectable.
    """
    B, N, D = 1, 6, 8
    n_rec, n_lig = 4, 2

    # Embedded distogram — all ones so any zero is a mask effect
    distogram = torch.ones(B, N, N, D)

    # Raw distances: receptor tokens are close together, ligand tokens
    # are far from receptor but close to each other
    coords = torch.tensor([
        [0.0, 0.0, 0.0],  # rec 0
        [1.0, 0.0, 0.0],  # rec 1
        [2.0, 0.0, 0.0],  # rec 2
        [3.0, 0.0, 0.0],  # rec 3
        [20.0, 0.0, 0.0], # lig 0  (far from receptor)
        [21.0, 0.0, 0.0], # lig 1
    ]).unsqueeze(0)  # [1, 6, 3]
    d = torch.cdist(coords, coords).squeeze(0).unsqueeze(0)  # [1, 6, 6]

    rec = torch.zeros(B, N, dtype=torch.bool)
    rec[0, :n_rec] = True

    lig = torch.zeros(B, N, dtype=torch.bool)
    lig[0, n_rec:n_rec + n_lig] = True

    return distogram, d, rec, lig, n_rec, n_lig, D


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDistogramMaskNone:
    def test_no_change(self, setup_tensors):
        distogram, d, rec, lig, *_ = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "none")
        assert torch.equal(result, distogram)


class TestDistogramMaskZeroAll:
    def test_all_zeros(self, setup_tensors):
        distogram, d, rec, lig, *_ = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_all")
        assert result.abs().sum().item() == 0.0

    def test_shape_preserved(self, setup_tensors):
        distogram, d, rec, lig, *_ = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_all")
        assert result.shape == distogram.shape


class TestDistogramMaskZeroCross:
    def test_cross_pairs_zeroed(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_cross")

        # Receptor-ligand block (rec rows, lig cols) should be zero
        rec_lig_block = result[0, :n_rec, n_rec:n_rec + n_lig, :]
        assert rec_lig_block.abs().sum().item() == 0.0

        # Ligand-receptor block (lig rows, rec cols) should be zero
        lig_rec_block = result[0, n_rec:n_rec + n_lig, :n_rec, :]
        assert lig_rec_block.abs().sum().item() == 0.0

    def test_receptor_receptor_preserved(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_cross")

        # Receptor-receptor block should be unchanged
        rec_rec_orig = distogram[0, :n_rec, :n_rec, :]
        rec_rec_result = result[0, :n_rec, :n_rec, :]
        assert torch.equal(rec_rec_result, rec_rec_orig)

    def test_ligand_ligand_preserved(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_cross")

        # Ligand-ligand block should be unchanged
        lig_lig_orig = distogram[0, n_rec:n_rec + n_lig, n_rec:n_rec + n_lig, :]
        lig_lig_result = result[0, n_rec:n_rec + n_lig, n_rec:n_rec + n_lig, :]
        assert torch.equal(lig_lig_result, lig_lig_orig)

    def test_symmetry(self, setup_tensors):
        """Cross mask should be symmetric: (i,j) zeroed iff (j,i) zeroed."""
        distogram, d, rec, lig, *_ = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_cross")
        zero_mask = (result.abs().sum(-1) == 0)  # [B, N, N]
        assert torch.equal(zero_mask, zero_mask.transpose(1, 2))


class TestDistogramMaskZeroLigand:
    def test_all_ligand_rows_zeroed(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_ligand")

        # Entire rows for ligand tokens should be zero
        lig_rows = result[0, n_rec:n_rec + n_lig, :, :]
        assert lig_rows.abs().sum().item() == 0.0

    def test_all_ligand_cols_zeroed(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_ligand")

        # Entire columns for ligand tokens should be zero
        lig_cols = result[0, :, n_rec:n_rec + n_lig, :]
        assert lig_cols.abs().sum().item() == 0.0

    def test_receptor_receptor_preserved(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_ligand")

        rec_rec_orig = distogram[0, :n_rec, :n_rec, :]
        rec_rec_result = result[0, :n_rec, :n_rec, :]
        assert torch.equal(rec_rec_result, rec_rec_orig)

    def test_superset_of_cross(self, setup_tensors):
        """zero_ligand should zero everything zero_cross zeros, plus more."""
        distogram, d, rec, lig, *_ = setup_tensors
        cross_result = apply_distogram_mask(distogram, d, rec, lig, "zero_cross")
        lig_result = apply_distogram_mask(distogram, d, rec, lig, "zero_ligand")

        cross_zero = (cross_result.abs().sum(-1) == 0)
        lig_zero = (lig_result.abs().sum(-1) == 0)

        # Everything zeroed by cross should also be zeroed by ligand
        assert (cross_zero & ~lig_zero).sum().item() == 0
        # Ligand should zero strictly more (the lig-lig block)
        assert lig_zero.sum().item() > cross_zero.sum().item()


class TestDistogramMaskZeroReceptor:
    def test_receptor_receptor_zeroed(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_receptor")

        rec_rec_block = result[0, :n_rec, :n_rec, :]
        assert rec_rec_block.abs().sum().item() == 0.0

    def test_cross_pairs_preserved(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_receptor")

        # Receptor-ligand cross block should be preserved
        rec_lig_orig = distogram[0, :n_rec, n_rec:n_rec + n_lig, :]
        rec_lig_result = result[0, :n_rec, n_rec:n_rec + n_lig, :]
        assert torch.equal(rec_lig_result, rec_lig_orig)

    def test_ligand_ligand_preserved(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_receptor")

        lig_lig_orig = distogram[0, n_rec:n_rec + n_lig, n_rec:n_rec + n_lig, :]
        lig_lig_result = result[0, n_rec:n_rec + n_lig, n_rec:n_rec + n_lig, :]
        assert torch.equal(lig_lig_result, lig_lig_orig)


class TestDistogramMaskDistanceCutoff:
    def test_close_pairs_preserved(self, setup_tensors):
        """Pairs within cutoff should keep their values."""
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        cutoff = 5.0
        result = apply_distogram_mask(
            distogram, d, rec, lig, "distance_cutoff", distance_cutoff=cutoff,
        )
        # Receptor tokens 0-3 are at 0,1,2,3 Å apart from neighbors
        # Pairs within 5Å: (0,1)=1, (0,2)=2, (0,3)=3, (1,2)=1, (1,3)=2,
        # (2,3)=1, plus self-distances=0. All within cutoff.
        # Check rec 0 vs rec 1 (distance = 1.0 < 5.0) → preserved
        assert result[0, 0, 1, :].abs().sum().item() > 0

    def test_far_pairs_zeroed(self, setup_tensors):
        """Pairs beyond cutoff should be zeroed."""
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        cutoff = 5.0
        result = apply_distogram_mask(
            distogram, d, rec, lig, "distance_cutoff", distance_cutoff=cutoff,
        )
        # Receptor token 0 (at x=0) vs ligand token 0 (at x=20) → d=20 > 5
        assert result[0, 0, 4, :].abs().sum().item() == 0.0

    def test_default_cutoff_is_8(self, setup_tensors):
        """When distance_cutoff=None, default should be 8Å."""
        distogram, d, rec, lig, *_ = setup_tensors
        result_default = apply_distogram_mask(
            distogram, d, rec, lig, "distance_cutoff", distance_cutoff=None,
        )
        result_explicit = apply_distogram_mask(
            distogram, d, rec, lig, "distance_cutoff", distance_cutoff=8.0,
        )
        assert torch.equal(result_default, result_explicit)

    def test_large_cutoff_preserves_all(self, setup_tensors):
        """Cutoff larger than max distance should preserve everything."""
        distogram, d, rec, lig, *_ = setup_tensors
        cutoff = 100.0  # Larger than any pairwise distance
        result = apply_distogram_mask(
            distogram, d, rec, lig, "distance_cutoff", distance_cutoff=cutoff,
        )
        assert torch.equal(result, distogram)

    def test_zero_cutoff_zeros_all_except_self(self, setup_tensors):
        """Cutoff of 0 should zero everything except self-distances (d=0)."""
        distogram, d, rec, lig, *_ = setup_tensors
        result = apply_distogram_mask(
            distogram, d, rec, lig, "distance_cutoff", distance_cutoff=0.0,
        )
        # Only diagonal (self-pairs, d=0) should survive
        B, N, _, D = distogram.shape
        for i in range(N):
            assert result[0, i, i, :].abs().sum().item() > 0
        # Off-diagonal should be zero
        off_diag_mask = ~torch.eye(N, dtype=torch.bool)
        assert result[0, off_diag_mask, :].abs().sum().item() == 0.0


class TestMaskCoverage:
    """Verify that the union of complementary masks covers all pairs."""

    def test_zero_cross_plus_zero_receptor_covers_receptor_involvement(
        self, setup_tensors,
    ):
        """zero_cross + zero_receptor should zero all pairs involving receptor."""
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        cross_result = apply_distogram_mask(distogram, d, rec, lig, "zero_cross")
        rec_result = apply_distogram_mask(distogram, d, rec, lig, "zero_receptor")

        cross_zero = (cross_result.abs().sum(-1) == 0)
        rec_zero = (rec_result.abs().sum(-1) == 0)
        union_zero = cross_zero | rec_zero

        # Every pair involving a receptor token should be zeroed by the union
        for i in range(n_rec):
            for j in range(n_rec + n_lig):
                assert union_zero[0, i, j].item(), f"Pair ({i},{j}) not covered"
                assert union_zero[0, j, i].item(), f"Pair ({j},{i}) not covered"

    def test_zero_ligand_plus_zero_receptor_covers_all(self, setup_tensors):
        """zero_ligand + zero_receptor together should zero everything
        (since every pair involves at least one receptor or ligand token)."""
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        lig_result = apply_distogram_mask(distogram, d, rec, lig, "zero_ligand")
        rec_result = apply_distogram_mask(distogram, d, rec, lig, "zero_receptor")

        lig_zero = (lig_result.abs().sum(-1) == 0)
        rec_zero = (rec_result.abs().sum(-1) == 0)
        union_zero = lig_zero | rec_zero

        # All pairs should be covered
        assert union_zero.all()


class TestMaskCountCorrectness:
    """Verify exact counts of zeroed pairs for each mode."""

    def test_zero_cross_count(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_cross")
        zero_pairs = (result.abs().sum(-1) == 0).sum().item()
        # Cross pairs: n_rec * n_lig * 2 (both directions)
        expected = n_rec * n_lig * 2
        assert zero_pairs == expected

    def test_zero_ligand_count(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        N = n_rec + n_lig
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_ligand")
        zero_pairs = (result.abs().sum(-1) == 0).sum().item()
        # All pairs involving ligand: lig rows (n_lig * N) + lig cols (N * n_lig)
        # minus double-counted lig-lig block (n_lig * n_lig)
        expected = n_lig * N + N * n_lig - n_lig * n_lig
        assert zero_pairs == expected

    def test_zero_receptor_count(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_receptor")
        zero_pairs = (result.abs().sum(-1) == 0).sum().item()
        # Receptor-receptor block
        expected = n_rec * n_rec
        assert zero_pairs == expected

    def test_zero_all_count(self, setup_tensors):
        distogram, d, rec, lig, n_rec, n_lig, D = setup_tensors
        N = n_rec + n_lig
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_all")
        zero_pairs = (result.abs().sum(-1) == 0).sum().item()
        assert zero_pairs == N * N


class TestEdgeCases:
    def test_single_ligand_token(self):
        """System with 1 ligand token and 3 receptor tokens."""
        B, N, D = 1, 4, 4
        distogram = torch.ones(B, N, N, D)
        d = torch.ones(B, N, N) * 5.0
        rec = torch.tensor([[True, True, True, False]])
        lig = torch.tensor([[False, False, False, True]])

        result = apply_distogram_mask(distogram, d, rec, lig, "zero_cross")
        # Cross: (3,0), (3,1), (3,2) and (0,3), (1,3), (2,3) = 6 pairs
        zero_pairs = (result.abs().sum(-1) == 0).sum().item()
        assert zero_pairs == 6

    def test_batch_dimension(self):
        """Masking should work independently per batch element."""
        B, N, D = 2, 4, 4
        distogram = torch.ones(B, N, N, D)
        d = torch.ones(B, N, N) * 5.0
        rec = torch.tensor([
            [True, True, False, False],
            [True, True, True, False],
        ])
        lig = torch.tensor([
            [False, False, True, True],
            [False, False, False, True],
        ])

        result = apply_distogram_mask(distogram, d, rec, lig, "zero_cross")

        # Batch 0: 2 rec × 2 lig × 2 directions = 8 zeroed pairs
        zero_b0 = (result[0].abs().sum(-1) == 0).sum().item()
        assert zero_b0 == 8

        # Batch 1: 3 rec × 1 lig × 2 directions = 6 zeroed pairs
        zero_b1 = (result[1].abs().sum(-1) == 0).sum().item()
        assert zero_b1 == 6

    def test_non_uniform_values(self):
        """Masking should work with varied embedding values, not just ones."""
        B, N, D = 1, 5, 8
        torch.manual_seed(42)
        distogram = torch.randn(B, N, N, D)
        d = torch.ones(B, N, N) * 10.0
        rec = torch.tensor([[True, True, True, False, False]])
        lig = torch.tensor([[False, False, False, True, True]])

        original = distogram.clone()
        result = apply_distogram_mask(distogram, d, rec, lig, "zero_cross")

        # Cross blocks should be exactly zero
        assert result[0, :3, 3:, :].abs().sum().item() == 0.0
        assert result[0, 3:, :3, :].abs().sum().item() == 0.0

        # Non-cross blocks should be exactly unchanged
        assert torch.equal(result[0, :3, :3, :], original[0, :3, :3, :])
        assert torch.equal(result[0, 3:, 3:, :], original[0, 3:, 3:, :])

    def test_unknown_mode_raises(self, setup_tensors):
        distogram, d, rec, lig, *_ = setup_tensors
        with pytest.raises(ValueError, match="Unknown mode"):
            apply_distogram_mask(distogram, d, rec, lig, "invalid_mode")
