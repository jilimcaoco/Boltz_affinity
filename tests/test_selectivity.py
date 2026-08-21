"""Tests for cross-receptor selectivity training.

Covers the three pieces added for selectivity finetuning:

* the manifest / collate plumbing (``pair_id``, ``receptor_id`` and the
  previously-dropped ``is_censored``),
* :class:`~boltz.lora.data.PairedReceptorSampler`, and
* :mod:`boltz.lora.selectivity_losses`.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from boltz.lora.data import (
    OPTIONAL_COLUMNS,
    LoRADataset,
    LoRARow,
    PairedReceptorSampler,
    lora_collate,
    parse_lora_csv,
)
from boltz.lora.losses import censored_boltz2_affinity_loss, load_loss_from_spec
from boltz.lora.selectivity_losses import make_selectivity_loss


# ── helpers ──────────────────────────────────────────────────────────────────


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _rows(specs: list[tuple]) -> list[LoRARow]:
    """Build LoRARows from ``(pair_id, receptor_id, group_id)`` tuples."""
    return [
        LoRARow(
            name=f"r{i}",
            ligand="CCO",
            receptor=f"{rid}.yaml",
            target=0.0,
            structure="s.pdb",
            group_id=gid,
            pair_id=pid,
            receptor_id=rid,
            row_index=i,
        )
        for i, (pid, rid, gid) in enumerate(specs)
    ]


def _pred(values: list[float]) -> dict[str, torch.Tensor]:
    return {"affinity_pred_value": torch.tensor(values).reshape(-1, 1)}


def _batch(targets: list[float], **cols) -> dict:
    out = {"target": torch.tensor(targets, dtype=torch.float32)}
    out.update(cols)
    return out


# ── manifest / collate plumbing ──────────────────────────────────────────────


def test_optional_columns_advertise_pair_fields():
    for col in ("pair_id", "receptor_id", "is_censored"):
        assert col in OPTIONAL_COLUMNS


def test_parse_csv_reads_pair_columns(tmp_path: Path):
    csv_path = tmp_path / "m.csv"
    _write_csv(csv_path, [
        {"ligand": "CCO", "receptor": "mor.yaml", "target": "1.0",
         "structure": "a.pdb", "pair_id": "L1", "receptor_id": "MOR",
         "group_id": "panel1", "is_censored": "0"},
        {"ligand": "CCO", "receptor": "dor.yaml", "target": "3.0",
         "structure": "b.pdb", "pair_id": "L1", "receptor_id": "DOR",
         "group_id": "panel1", "is_censored": "1"},
    ])
    rows, _ = parse_lora_csv(csv_path)
    assert [r.pair_id for r in rows] == ["L1", "L1"]
    assert [r.receptor_id for r in rows] == ["MOR", "DOR"]
    assert [r.is_censored for r in rows] == [0, 1]


def test_collate_forwards_censoring_and_pair_columns(tmp_path: Path):
    """Regression test: lora_collate used to drop is_censored entirely, so
    every censored_* loss silently degraded to a two-sided Huber."""
    csv_path = tmp_path / "m.csv"
    _write_csv(csv_path, [
        {"ligand": "CCO", "receptor": "mor.yaml", "target": "1.0",
         "structure": "a.pdb", "pair_id": "L1", "receptor_id": "MOR",
         "is_censored": "1"},
        {"ligand": "CCO", "receptor": "dor.yaml", "target": "3.0",
         "structure": "b.pdb", "pair_id": "L1", "receptor_id": "DOR",
         "is_censored": "0"},
    ])
    ds = LoRADataset(csv_path)
    batch = lora_collate([ds[0], ds[1]])
    assert batch["is_censored"] == [1, 0]
    assert batch["pair_id"] == ["L1", "L1"]
    assert batch["receptor_id"] == ["MOR", "DOR"]


# ── PairedReceptorSampler ────────────────────────────────────────────────────


def test_sampler_keeps_both_arms_together():
    rows = _rows([
        ("L1", "MOR", "p"), ("L1", "DOR", "p"),
        ("L2", "MOR", "p"), ("L2", "DOR", "p"),
        ("L3", "MOR", "p"), ("L3", "DOR", "p"),
    ])
    sampler = PairedReceptorSampler(rows, batch_size=4, shuffle=True)
    seen: set[int] = set()
    for batch in sampler:
        pairs = {rows[i].pair_id for i in batch}
        for pid in pairs:
            members = {i for i in batch if rows[i].pair_id == pid}
            expected = {i for i, r in enumerate(rows) if r.pair_id == pid}
            assert members == expected, f"pair {pid} split across batches"
        seen.update(batch)
    assert seen == set(range(len(rows))), "every row must be yielded once"


def test_sampler_len_matches_iteration():
    rows = _rows([(f"L{i}", r, "p") for i in range(5) for r in ("MOR", "DOR")])
    sampler = PairedReceptorSampler(rows, batch_size=4, shuffle=True)
    assert len(sampler) == len(list(sampler))


def test_sampler_len_matches_iteration_with_mixed_unit_sizes():
    """Greedy packing of 2-row pairs and 1-row singles gives an order-dependent
    batch count, so a cached unshuffled length would disagree with what
    __iter__ actually yields -- and DataLoader trusts len(sampler)."""
    # One 2-row pair plus two 1-row units at batch_size=2: greedy packing
    # gives 2 batches for the order (1,1,2) and 3 for (1,2,1).
    rows = _rows([
        ("L1", "MOR", "p"), ("L1", "DOR", "p"),
        ("L2", "MOR", "p"),
        ("L3", "DOR", "p"),
    ])
    counts = set()
    for epoch in range(20):
        sampler = PairedReceptorSampler(rows, batch_size=2, shuffle=True)
        sampler.set_epoch(epoch)
        batches = list(sampler)
        assert len(sampler) == len(batches), f"mismatch at epoch {epoch}"
        assert sorted(i for b in batches for i in b) == list(range(len(rows)))
        counts.add(len(batches))
    assert len(counts) > 1, (
        "test is vacuous unless the batch count actually varies with order"
    )


def test_sampler_respects_batch_size_and_blocks():
    rows = _rows([
        ("L1", "MOR", "panelA"), ("L1", "DOR", "panelA"),
        ("L2", "MOR", "panelB"), ("L2", "DOR", "panelB"),
    ])
    sampler = PairedReceptorSampler(rows, batch_size=4, shuffle=False)
    for batch in sampler:
        assert len(batch) <= 4
        # A batch never mixes two panels, so the ligand-axis term stays valid.
        assert len({rows[i].group_id for i in batch}) == 1


def test_sampler_counts_and_unpaired_rows():
    rows = _rows([
        ("L1", "MOR", "p"), ("L1", "DOR", "p"),
        ("L2", "MOR", "p"),                      # incomplete pair
        (None, None, "p"),                       # unpaired row
    ])
    sampler = PairedReceptorSampler(rows, batch_size=8, shuffle=False)
    assert sampler.n_pairs == 2
    assert sampler.n_complete_pairs == 1
    assert sampler.n_unpaired == 1
    assert sorted(i for b in sampler for i in b) == [0, 1, 2, 3]


def test_sampler_epoch_changes_order():
    rows = _rows([(f"L{i}", r, "p") for i in range(6) for r in ("MOR", "DOR")])
    sampler = PairedReceptorSampler(rows, batch_size=2, shuffle=True)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second
    sampler.set_epoch(0)
    assert list(sampler) == first, "same epoch must reproduce the same order"


# ── selectivity loss ─────────────────────────────────────────────────────────

_PAIR_COLS = {
    "pair_id": ["L1", "L1"],
    "receptor_id": ["MOR", "DOR"],
    "group_id": ["p", "p"],
    "is_censored": [0, 0],
}


def test_perfect_prediction_is_zero():
    loss_fn = make_selectivity_loss(bce_weight=0.0, read_env=False)
    t = [1.0, 3.0]
    out = loss_fn(_pred(t), _batch(t, **_PAIR_COLS))
    assert float(out) == pytest.approx(0.0, abs=1e-6)


def test_delta_term_penalises_wrong_difference():
    """Predicting the right mean but the wrong gap must cost something."""
    loss_fn = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_rank=0.0, bce_weight=0.0, read_env=False,
    )
    t = [1.0, 3.0]                       # true Δ = -2 (MOR 100x stronger)
    good = float(loss_fn(_pred([1.0, 3.0]), _batch(t, **_PAIR_COLS)))
    flat = float(loss_fn(_pred([2.0, 2.0]), _batch(t, **_PAIR_COLS)))
    assert good == pytest.approx(0.0, abs=1e-6)
    assert flat > 0.5


def test_delta_term_is_invariant_to_a_shared_shift():
    """A ligand-global offset cancels in Δ — the point of scoring one ligand
    against both receptors."""
    loss_fn = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_rank=0.0, bce_weight=0.0, read_env=False,
    )
    t = [1.0, 3.0]
    base = float(loss_fn(_pred([1.5, 3.2]), _batch(t, **_PAIR_COLS)))
    shifted = float(loss_fn(_pred([6.5, 8.2]), _batch(t, **_PAIR_COLS)))
    assert base == pytest.approx(shifted, abs=1e-6)


def test_arm_swap_leaves_loss_unchanged():
    """Huber is even and soft-label BCE is symmetric, so presenting a pair in
    the other order is a no-op; this is why no swap augmentation is added."""
    loss_fn = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, bce_weight=0.0, read_env=False,
    )
    a = float(loss_fn(_pred([1.4, 3.1]), _batch([1.0, 3.0], **_PAIR_COLS)))
    swapped = dict(_PAIR_COLS, receptor_id=["DOR", "MOR"])
    b = float(loss_fn(_pred([3.1, 1.4]), _batch([3.0, 1.0], **swapped)))
    assert a == pytest.approx(b, abs=1e-6)


def test_rank_term_rewards_correct_ordering():
    loss_fn = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_sel=0.0, bce_weight=0.0, read_env=False,
    )
    t = [1.0, 4.0]
    right = float(loss_fn(_pred([1.0, 4.0]), _batch(t, **_PAIR_COLS)))
    wrong = float(loss_fn(_pred([4.0, 1.0]), _batch(t, **_PAIR_COLS)))
    assert wrong > right


def test_rank_term_is_offset_robust():
    """A constant per-receptor offset changes Δ regression but must leave the
    ordering objective comparatively untouched."""
    rank_only = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_sel=0.0, bce_weight=0.0, read_env=False,
    )
    delta_only = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_rank=0.0, bce_weight=0.0, read_env=False,
    )
    t = [1.0, 4.0]
    # Model reproduces the ordering but inflates the gap by 2 log units.
    p = _pred([0.0, 5.0])
    assert float(delta_only(p, _batch(t, **_PAIR_COLS))) > 1.0
    assert float(rank_only(p, _batch(t, **_PAIR_COLS))) < 0.1


def test_censored_anti_target_is_one_sided():
    """DOR reported as '> 10 uM': predicting *even weaker* must not be
    penalised, predicting stronger must be."""
    loss_fn = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_rank=0.0, bce_weight=0.0, read_env=False,
    )
    cols = dict(_PAIR_COLS, is_censored=[0, 1])
    t = [1.0, 1.0]                        # reported Δ = 0, DOR is a lower bound
    weaker = float(loss_fn(_pred([1.0, 5.0]), _batch(t, **cols)))
    stronger = float(loss_fn(_pred([1.0, -3.0]), _batch(t, **cols)))
    assert weaker == pytest.approx(0.0, abs=1e-6)
    assert stronger > 0.5


def test_both_arms_censored_contributes_nothing():
    loss_fn = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_rank=0.0, bce_weight=0.0, read_env=False,
    )
    cols = dict(_PAIR_COLS, is_censored=[1, 1])
    out = loss_fn(_pred([-5.0, 9.0]), _batch([1.0, 1.0], **cols))
    assert float(out) == pytest.approx(0.0, abs=1e-6)


def test_ligand_axis_term_never_pairs_across_receptors():
    """Keyed on (group_id, receptor_id): two rows in one panel measured against
    *different* receptors are not a ligand pair, and pairing them would
    reintroduce the assay offset the term exists to cancel."""
    loss_fn = make_selectivity_loss(
        w_point=0.0, w_sel=0.0, w_rank=0.0, w_lig=1.0, bce_weight=0.0,
        read_env=False,
    )
    cols = {
        "group_id": ["p", "p"],
        "receptor_id": ["MOR", "DOR"],
        "pair_id": ["L1", "L1"],
        "is_censored": [0, 0],
    }
    # Prediction gets the cross-receptor difference badly wrong, but there is
    # no *same-receptor* pair in this batch, so the ligand-axis term is zero.
    out = loss_fn(_pred([0.0, 0.0]), _batch([1.0, 5.0], **cols))
    assert float(out) == pytest.approx(0.0, abs=1e-6)


def test_ligand_axis_term_fires_within_one_receptor():
    loss_fn = make_selectivity_loss(
        w_point=0.0, w_sel=0.0, w_rank=0.0, w_lig=1.0, bce_weight=0.0,
        read_env=False,
    )
    cols = {
        "group_id": ["p", "p"],
        "receptor_id": ["MOR", "MOR"],
        "pair_id": ["L1", "L2"],
        "is_censored": [0, 0],
    }
    out = loss_fn(_pred([0.0, 0.0]), _batch([1.0, 5.0], **cols))
    assert float(out) > 0.5


def test_tail_weight_emphasises_selective_pairs():
    """sel_tail_gamma reweights examples; it must not move the optimum."""
    plain = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_rank=0.0, bce_weight=0.0, read_env=False,
    )
    tail = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_rank=0.0, bce_weight=0.0,
        sel_tail_gamma=2.0, read_env=False,
    )
    # Two pairs in one batch: L1 is highly selective, L2 is not.  Both are
    # mispredicted by the same 2.0 log units.
    cols = {
        "pair_id": ["L1", "L1", "L2", "L2"],
        "receptor_id": ["MOR", "DOR", "MOR", "DOR"],
        "group_id": ["p"] * 4,
        "is_censored": [0] * 4,
    }
    t = [1.0, 5.0, 1.0, 1.0]          # |delta| = 4 for L1, 0 for L2
    # Optimum is unchanged: a perfect prediction is still exactly zero.
    assert float(tail(_pred(t), _batch(t, **cols))) == pytest.approx(0.0, abs=1e-6)
    # Weighting shifts emphasis onto the selective pair, so the same total
    # error costs more when it lands there.
    on_selective = _pred([1.0, 3.0, 1.0, 1.0])
    on_flat = _pred([1.0, 5.0, 1.0, 3.0])
    assert float(plain(on_selective, _batch(t, **cols))) == pytest.approx(
        float(plain(on_flat, _batch(t, **cols))), abs=1e-6
    )
    assert float(tail(on_selective, _batch(t, **cols))) > float(
        tail(on_flat, _batch(t, **cols))
    )


def test_reference_receptor_fixes_sign_convention():
    mor_ref = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_sel=0.0, bce_weight=0.0,
        reference_receptor="MOR", read_env=False,
    )
    # Rows given in the other manifest order must produce the same value.
    a = float(mor_ref(_pred([1.0, 4.0]), _batch([1.0, 4.0], **_PAIR_COLS)))
    flipped = dict(_PAIR_COLS, receptor_id=["DOR", "MOR"])
    b = float(mor_ref(_pred([4.0, 1.0]), _batch([4.0, 1.0], **flipped)))
    assert a == pytest.approx(b, abs=1e-6)


def test_replicate_rows_under_one_pair_id_are_skipped():
    """Two rows for the same receptor are a replicate, not a cross-receptor
    pair; inventing a direction for them would be wrong."""
    loss_fn = make_selectivity_loss(
        w_point=0.0, w_lig=0.0, w_rank=0.0, bce_weight=0.0, read_env=False,
    )
    cols = dict(_PAIR_COLS, receptor_id=["MOR", "MOR"])
    out = loss_fn(_pred([0.0, 9.0]), _batch([1.0, 1.0], **cols))
    assert float(out) == pytest.approx(0.0, abs=1e-6)


def test_empty_group_id_is_treated_as_missing():
    """Matches the guard in intra_assay_huber_loss: a blank group is not a
    group, so blank-keyed rows must not be paired with each other."""
    loss_fn = make_selectivity_loss(
        w_point=0.0, w_sel=0.0, w_rank=0.0, w_lig=1.0, bce_weight=0.0,
        read_env=False,
    )
    cols = {
        "group_id": ["", ""],
        "receptor_id": ["MOR", "MOR"],
        "pair_id": ["L1", "L2"],
        "is_censored": [0, 0],
    }
    out = loss_fn(_pred([0.0, 0.0]), _batch([1.0, 5.0], **cols))
    assert float(out) == pytest.approx(0.0, abs=1e-6)


def test_missing_pair_columns_degrade_gracefully():
    """A plain potency manifest with no pair_id must still train."""
    loss_fn = make_selectivity_loss(read_env=False)
    out = loss_fn(_pred([1.0, 2.0]), _batch([1.0, 2.0], group_id=["a", "a"]))
    assert torch.isfinite(out)


def test_selectivity_off_matches_censored_boltz2_baseline():
    """With both selectivity weights at zero the loss must reduce *exactly* to
    the existing baseline, so the no-selectivity control carries no
    implementation difference."""
    off = make_selectivity_loss(w_sel=0.0, w_rank=0.0, read_env=False)
    pred = {
        "affinity_pred_value": torch.tensor([0.4, 2.2, 3.1]).reshape(-1, 1),
        "affinity_logits_binary": torch.tensor([0.5, -0.2, -1.0]).reshape(-1, 1),
    }
    batch = _batch(
        [1.0, 2.0, 3.0],
        group_id=["p", "p", "p"],
        is_censored=[0, 1, 0],
        is_binder=[1, 0, 0],
    )
    a = float(off(pred, batch))
    b = float(censored_boltz2_affinity_loss(pred, batch))
    assert a == pytest.approx(b, rel=1e-6)


@pytest.mark.parametrize("name", [
    "selectivity_joint",
    "selectivity_rank_only",
    "selectivity_delta_only",
    "selectivity_off",
])
def test_presets_resolve_and_produce_scalars(name):
    fn = load_loss_from_spec(name)
    out = fn(_pred([1.0, 3.0]), _batch([1.0, 3.5], **_PAIR_COLS), None)
    assert out.ndim == 0
    assert math.isfinite(float(out))


def test_dotted_module_loss_spec():
    fn = load_loss_from_spec(
        "boltz.lora.selectivity_losses:selectivity_joint"
    )
    assert callable(fn)


def test_env_override(monkeypatch):
    monkeypatch.setenv("BOLTZ_SELECTIVITY_W_SEL", "0.0")
    monkeypatch.setenv("BOLTZ_SELECTIVITY_W_RANK", "0.0")
    monkeypatch.setenv("BOLTZ_SELECTIVITY_W_LIG", "0.0")
    monkeypatch.setenv("BOLTZ_SELECTIVITY_BCE_WEIGHT", "0.0")
    fn = make_selectivity_loss()
    # Only the point term survives: mean Huber of a 1.0 error over 2 rows.
    out = float(fn(_pred([1.0, 3.0]), _batch([2.0, 4.0], **_PAIR_COLS)))
    assert out == pytest.approx(0.5, abs=1e-6)


def test_rejects_nonpositive_tau():
    with pytest.raises(ValueError, match="tau"):
        make_selectivity_loss(tau=0.0, read_env=False)


def test_gradients_flow_to_predictions():
    loss_fn = make_selectivity_loss(read_env=False)
    p = torch.tensor([1.0, 3.0], requires_grad=True)
    out = loss_fn({"affinity_pred_value": p.reshape(-1, 1)},
                  _batch([1.5, 4.0], **_PAIR_COLS))
    out.backward()
    assert p.grad is not None
    assert torch.any(p.grad != 0)
