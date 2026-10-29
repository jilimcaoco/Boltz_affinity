"""The ``is_censored`` flag must reach the censored losses intact."""

from __future__ import annotations

import logging

import torch

import boltz.lora.train as T
from boltz.lora.data import (
    LoRARow,
    lora_collate,
    warn_if_censored_loss_without_censored_rows,
)
from boltz.lora.losses import (
    censored_intra_assay_huber_loss,
    intra_assay_huber_loss,
)


def _rows(censored: set[int]) -> list[LoRARow]:
    return [
        LoRARow(
            name=f"r{i}", ligand="C", receptor="r.yaml", target=float(i) / 2,
            structure="s.pdb", group_id="g0", is_censored=int(i in censored),
            row_index=i,
        )
        for i in range(6)
    ]


def test_collate_carries_is_censored():
    batch = lora_collate([r.to_batch_item() for r in _rows({1, 4})])
    assert batch["is_censored"] == [0, 1, 0, 0, 1, 0]


def test_censored_loss_sees_flag_through_collate():
    batch = lora_collate([r.to_batch_item() for r in _rows({0, 3})])
    # Over-predict every row (pred > target): only one-sided terms for the
    # censored rows go to zero, so the censored loss must be smaller.
    pred = {"affinity_pred_value": (batch["target"] + 1.5).unsqueeze(-1)}
    censored = censored_intra_assay_huber_loss(pred, batch)
    plain = intra_assay_huber_loss(pred, batch)
    assert censored < plain
    # Without the flag the two losses coincide (the pre-fix behaviour).
    batch_no_flag = {k: v for k, v in batch.items() if k != "is_censored"}
    assert torch.equal(censored_intra_assay_huber_loss(pred, batch_no_flag), plain)


def test_row_drop_keeps_is_censored_aligned():
    batch = lora_collate([r.to_batch_item() for r in _rows({2, 5})])
    row = T._extract_row_from_batch(batch, 2)
    assert row["is_censored"] == [1]


def test_warns_when_censored_loss_has_no_censored_rows(caplog):
    with caplog.at_level(logging.WARNING, logger="boltz.lora.data"):
        warn_if_censored_loss_without_censored_rows("censored_boltz2_affinity", _rows(set()))
    assert "--keep-censored" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="boltz.lora.data"):
        warn_if_censored_loss_without_censored_rows("censored_boltz2_affinity", _rows({1}))
        warn_if_censored_loss_without_censored_rows("boltz2_affinity", _rows(set()))
    assert caplog.text == ""


def test_trainer_with_censored_rows_and_failed_featurisation(lora_harness):
    """A row failing featurisation is dropped from the batch; is_censored must
    be subset with it or the censored mask no longer lines up with preds."""
    names = lora_harness.write_csv(censored={"m01", "m04", "m07"})
    lora_harness.fail_rows.update({names[0], names[3]})
    _, run = lora_harness(loss_spec="censored_intra_assay_huber", epochs=2)
    assert len(run["metrics"]) == 2
    assert all(torch.isfinite(torch.tensor(m["loss"])) for m in run["metrics"])
