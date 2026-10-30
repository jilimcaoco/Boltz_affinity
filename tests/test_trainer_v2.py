"""Trainer v2: gradient reaches the affinity pairformer stack; presets, coverage
guard, adapter loading and provenance.

Uses real (small) ``AffinityModule`` instances on CPU, mounted the way Boltz-2
mounts them (``affinity_module1`` / ``affinity_module2`` next to a
``confidence_module`` with its own pairformer stack).
"""

from __future__ import annotations

import json
import logging
import re

import pytest
import torch
from torch import nn

import boltz.lora.train as T
from boltz.finetune.targets import resolve_targets as ft_resolve_targets
from boltz.lora.adapter import LoRAAdapter, LoRAConfig
from boltz.lora.apply import load_adapter_into_model
from boltz.lora.inject import apply_lora, lora_state_dict
from boltz.lora.layers import LoRALinear
from boltz.lora.provenance import _git, repo_state
from boltz.lora.registry import LoRARegistry
from boltz.lora.targets import TARGET_PRESETS, resolve_targets
from boltz.model.layers.pairformer import PairformerNoSeqModule
from boltz.model.modules.affinity import AffinityModule

TOKEN_S = TOKEN_Z = 16
N_TOK = 6
LINEARS_PER_BLOCK = 23  # PairformerNoSeqLayer (verified on the real checkpoint)
HEAD_LINEARS = 9


def _affinity_module(num_blocks: int) -> AffinityModule:
    return AffinityModule(
        token_s=TOKEN_S, token_z=TOKEN_Z,
        pairformer_args={
            "num_blocks": num_blocks, "dropout": 0.0,
            "activation_checkpointing": True,  # as in the Boltz-2 checkpoint
            "pairwise_head_width": 8, "pairwise_num_heads": 2,
        },
        transformer_args={
            "num_blocks": 1, "num_heads": 2, "token_s": TOKEN_S,
            "activation_checkpointing": True,
        },
        num_dist_bins=8,
    )


class _Boltz2Like(nn.Module):
    """affinity_module1 (2 blocks) + affinity_module2 (1 block) + confidence."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.affinity_ensemble = True
        self.affinity_module1 = _affinity_module(2)
        self.affinity_module2 = _affinity_module(1)
        self.confidence_module = nn.Module()
        self.confidence_module.pairformer_stack = PairformerNoSeqModule(
            TOKEN_Z, 1, pairwise_head_width=8, pairwise_num_heads=2,
        )


def _inputs():
    g = torch.Generator().manual_seed(1)
    feats = {
        "token_to_rep_atom": torch.eye(N_TOK)[None],
        "token_pad_mask": torch.ones(1, N_TOK),
        "mol_type": torch.tensor([[0, 0, 0, 0, 3, 3]]),
        "affinity_token_mask": torch.tensor([[0, 0, 0, 0, 1, 1]]),
    }
    # Frozen-trunk inputs: nothing here requires grad.
    s = torch.randn(1, N_TOK, TOKEN_S, generator=g)
    z = torch.randn(1, N_TOK, N_TOK, TOKEN_Z, generator=g)
    x = torch.randn(1, 1, N_TOK, 3, generator=g) * 5
    return s, z, x, feats


def _lora_bs(model: nn.Module, scope: str) -> list[nn.Parameter]:
    return [m.lora_B for n, m in model.named_modules()
            if isinstance(m, LoRALinear) and f".{scope}." in f".{n}"]


def _adapt_and_backward(*, legacy: bool, modules=("affinity_module1",)):
    model = _Boltz2Like().train()
    T.set_checkpoint_mode(model, legacy=legacy)
    pats = resolve_targets("affinity_heads_pairformer", modules=modules)
    apply_lora(model, pats, r=2, alpha=4.0)
    s, z, x, feats = _inputs()
    model.affinity_module1(s, z, x, feats)["affinity_pred_value"].sum().backward()
    return model


# ─── A. gradient reaches the stack ───────────────────────────────────────────


def test_v2_trains_affinity_pairformer_stack():
    model = _adapt_and_backward(legacy=False)
    stack = _lora_bs(model.affinity_module1, "pairformer_stack")
    assert len(stack) == 2 * LINEARS_PER_BLOCK
    assert all(b.grad is not None for b in stack)
    assert any(b.grad.abs().sum() > 0 for b in stack)


def test_legacy_reproduces_frozen_stack_bug():
    """Reentrant checkpointing on a no-grad input: no stack gradient at all,
    while the heads still train — exactly what every pre-v2 adapter shows."""
    model = _adapt_and_backward(legacy=True)
    assert all(b.grad is None for b in _lora_bs(model.affinity_module1, "pairformer_stack"))
    assert any(b.grad is not None for b in _lora_bs(model.affinity_module1, "affinity_heads"))


def test_set_checkpoint_mode_covers_every_pairformer():
    model = _Boltz2Like()
    assert T.set_checkpoint_mode(model, legacy=True) == 3
    stacks = [m for m in model.modules() if isinstance(m, PairformerNoSeqModule)]
    assert all(m.checkpoint_use_reentrant for m in stacks)
    T.set_checkpoint_mode(model, legacy=False)
    assert not any(m.checkpoint_use_reentrant for m in stacks)


# ─── B. gradient-coverage guard ──────────────────────────────────────────────


def _coverage(model):
    cov = T.GradCoverage(list(model.named_parameters()))
    cov.update()
    return cov.summary()


def test_coverage_passes_when_stack_trains():
    summary = _coverage(_adapt_and_backward(legacy=False))
    n_grad, n_total = summary["scopes"]["affinity_module1.pairformer_stack"]
    assert n_total == 2 * LINEARS_PER_BLOCK * 2  # lora_A + lora_B per layer
    assert n_grad > 0  # lora_A grads are 0 on step 1 (lora_B starts at 0)
    T.check_grad_coverage(summary, forward_modules=("affinity_module1",), legacy=False)


def test_coverage_raises_on_dead_stack_unless_legacy(caplog):
    summary = _coverage(_adapt_and_backward(legacy=True))
    assert summary["scopes"]["affinity_module1.pairformer_stack"][0] == 0
    with pytest.raises(RuntimeError, match="affinity_module1.pairformer_stack: 0/"):
        T.check_grad_coverage(summary, forward_modules=("affinity_module1",), legacy=False)
    with caplog.at_level(logging.WARNING, logger=T.logger.name):
        T.check_grad_coverage(summary, forward_modules=("affinity_module1",), legacy=True)
    assert "legacy_trainer" in caplog.text


def test_coverage_only_warns_outside_forward(caplog):
    """Deprecated heads_pairformer adapts confidence_module, which the affinity
    forward never runs: a warning, not a failure."""
    model = _Boltz2Like().train()
    T.set_checkpoint_mode(model, legacy=False)
    apply_lora(model, TARGET_PRESETS["heads_pairformer"], r=2, alpha=4.0)
    s, z, x, feats = _inputs()
    model.affinity_module1(s, z, x, feats)["affinity_pred_value"].sum().backward()
    summary = _coverage(model)
    with caplog.at_level(logging.WARNING, logger=T.logger.name):
        T.check_grad_coverage(summary, forward_modules=("affinity_module1",), legacy=False)
    assert "confidence_module.pairformer_stack: 0/" in caplog.text
    assert "affinity_module2" in caplog.text


# ─── C. presets ──────────────────────────────────────────────────────────────


def _linear_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


@pytest.mark.parametrize(("modules", "expected"), [
    (("affinity_module1",), {"affinity_module1": 2 * LINEARS_PER_BLOCK + HEAD_LINEARS}),
    (("affinity_module1", "affinity_module2"), {
        "affinity_module1": 2 * LINEARS_PER_BLOCK + HEAD_LINEARS,
        "affinity_module2": 1 * LINEARS_PER_BLOCK + HEAD_LINEARS,
    }),
])
def test_scope_preset_counts_and_no_leak(modules, expected):
    pats = resolve_targets("affinity_heads_pairformer", modules=modules)
    hits = [n for n in _linear_names(_Boltz2Like()) if any(re.search(p, n) for p in pats)]
    counts: dict[str, int] = {}
    for n in hits:
        counts[n.split(".")[0]] = counts.get(n.split(".")[0], 0) + 1
    assert counts == expected
    assert not any(n.startswith("confidence_module") for n in hits)


def test_deprecated_heads_pairformer_unchanged_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_targets("heads_pairformer") == TARGET_PRESETS["heads_pairformer"]
    assert "deprecated" in caplog.text
    # Legacy resolution: new names are raw regexes, as before they existed.
    assert resolve_targets("affinity_module", legacy=True) == ("affinity_module",)


def test_lora_and_finetune_scopes_cover_the_same_modules():
    model = _Boltz2Like()
    mods = ("affinity_module1", "affinity_module2")
    lora_pats = resolve_targets("affinity_heads_pairformer", modules=mods)
    ft_pats = ft_resolve_targets("affinity_heads_pairformer", modules=mods)
    lora_layers = {n for n in _linear_names(model) if any(re.search(p, n) for p in lora_pats)}
    ft_params = {n for n, _ in model.named_parameters() if any(re.search(p, n) for p in ft_pats)}
    # Every adapted Linear's weight is trained by the full fine-tune...
    assert {f"{n}.weight" for n in lora_layers} <= ft_params
    # ...and full FT stays inside the same subtrees.
    assert all(re.match(r"affinity_module[12]\.(affinity_heads|pairformer_stack)\.", n) for n in ft_params)


# ─── D. adapter loading with incomplete metadata ─────────────────────────────


def test_load_adapter_without_layer_metadata(tmp_path):
    torch.manual_seed(0)
    trained = _Boltz2Like()
    pats = resolve_targets("affinity_heads", modules=("affinity_module1",))
    layers = apply_lora(trained, pats, r=2, alpha=4.0)
    for m in trained.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B)
    registry = LoRARegistry(tmp_path)
    # Mimic the recovered v2 adapters: no target_patterns, no adapted_layers.
    registry.save(
        LoRAAdapter(name="recovered", config=LoRAConfig(rank=2, alpha=4.0, target_patterns=[])),
        lora_state_dict(trained),
    )
    fresh = _Boltz2Like()
    adapter = load_adapter_into_model(fresh, "recovered", registry=registry)
    assert sorted(adapter.adapted_layers) == sorted(layers)
    assert sorted(n for n, m in fresh.named_modules() if isinstance(m, LoRALinear)) == sorted(layers)
    s, z, x, feats = _inputs()
    trained.eval(), fresh.eval()
    assert torch.equal(
        trained.affinity_module1(s, z, x, feats)["affinity_pred_value"],
        fresh.affinity_module1(s, z, x, feats)["affinity_pred_value"],
    )


# ─── E. provenance ───────────────────────────────────────────────────────────


def test_repo_state_records_commit():
    state = repo_state()
    assert set(state) == {"boltz_git_sha", "boltz_git_dirty", "torch_version"}
    assert state["boltz_git_sha"] is None or re.fullmatch(r"[0-9a-f]{40}", state["boltz_git_sha"])
    assert state["torch_version"] == str(torch.__version__)


def test_git_outside_repo_is_none(tmp_path):
    assert _git(["rev-parse", "HEAD"], tmp_path) is None


def test_trainer_records_semantics_and_provenance(lora_harness):
    _, run = lora_harness(epochs=1)
    assert run["trainer_semantics"] == "v2"
    assert run["target_modules"] == ["affinity_module"]
    assert run["grad_coverage"]["n_without_grad"] == 0
    assert "boltz_git_sha" in run and "torch_version" in run


# ─── F. finetune validation ──────────────────────────────────────────────────


def test_finetune_uses_validation_for_early_stopping(lora_harness, monkeypatch, tmp_path):
    import boltz.finetune.train as FT
    from boltz.finetune.registry import FinetuneRegistry

    monkeypatch.setattr(FT, "_load_base_model", lambda ckpt, dev: (lora_harness.base_model(), "", ""))
    registry = FinetuneRegistry(tmp_path / "ft")
    torch.manual_seed(0)
    FT.train_finetune(FT.FinetuneArgs(
        name="ft", csv_path=str(lora_harness.csv), loss_spec="intra_assay_huber",
        target_spec="affinity_module", epochs=3, batch_size=4, learning_rate=1e-2,
        plot_loss_curve=False, device="cpu", val_split=0.25,
        early_stopping_patience=1, early_stopping_min_delta=1e9,
    ), registry=registry)
    meta = json.loads((registry.adapter_dir("ft") / "meta.json").read_text())
    run = meta["history"][-1]
    assert all("val_loss" in m for m in run["metrics"])
    # min_delta so large no epoch can improve on val: stop after patience.
    assert len(run["metrics"]) == 2
    assert run["trainer_semantics"] == "v2" and run["affinity_output"] == "ensemble"
    assert run["grad_coverage"]["n_without_grad"] == 0
