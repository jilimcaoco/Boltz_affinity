"""
tests/test_interpretability.py
===============================
Validate the interpretability pipeline on a synthetic AffinityModule
with random weights.  No real checkpoint or GPU required.

Runs in < 30 seconds on CPU.
"""

from __future__ import annotations

import math

import sys
import traceback
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Ensure src/ and interpretability/ are importable.
# ---------------------------------------------------------------------------
_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TEST_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
_INTERP_DIR = _PROJECT_ROOT / "interpretability"

for p in (_SRC_DIR, _INTERP_DIR, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Constants for the synthetic setup.
# ---------------------------------------------------------------------------
TOKEN_S = 64
TOKEN_Z = 64
N_PROTEIN = 20
N_LIGAND = 10
N_TOKENS = N_PROTEIN + N_LIGAND
N_ATOMS = N_TOKENS  # 1:1 atom-to-token mapping for simplicity
BATCH = 1

PAIRFORMER_ARGS = {
    "num_blocks": 2,
    "pairwise_head_width": 16,
    "pairwise_num_heads": 2,
    "dropout": 0.0,
}

TRANSFORMER_ARGS = {
    "token_s": TOKEN_S,
    "num_blocks": 1,
    "num_heads": 2,
    "activation_checkpointing": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

results: list[tuple[str, bool, str]] = []


def report(name: str, passed: bool, detail: str = "") -> None:
    tag = "PASS" if passed else "FAIL"
    msg = f"[{tag}] {name}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)
    results.append((name, passed, detail))


def _make_synthetic_module():
    """Create a small AffinityModule with random weights."""
    from boltz.model.modules.affinity import AffinityModule

    module = AffinityModule(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        pairformer_args=PAIRFORMER_ARGS,
        transformer_args=TRANSFORMER_ARGS,
    )
    module.eval()
    return module


def _make_synthetic_inputs(device: str = "cpu") -> dict:
    """Build the kwargs dict for AffinityModule.forward().

    Returns a dict with keys: s_inputs, z, x_pred, feats, multiplicity,
    use_kernels.
    """
    s_inputs = torch.randn(BATCH, N_TOKENS, TOKEN_S, device=device)
    z = torch.randn(BATCH, N_TOKENS, N_TOKENS, TOKEN_Z, device=device)
    x_pred = torch.randn(BATCH, N_ATOMS, 3, device=device)

    # token_to_rep_atom: identity mapping (N_TOKENS == N_ATOMS)
    token_to_rep_atom = torch.eye(N_TOKENS, device=device).unsqueeze(0)

    # mol_type: 0=PROTEIN for first 20, 3=NONPOLYMER for last 10
    mol_type = torch.zeros(BATCH, N_TOKENS, dtype=torch.long, device=device)
    mol_type[:, N_PROTEIN:] = 3

    # pad mask: all valid
    token_pad_mask = torch.ones(BATCH, N_TOKENS, dtype=torch.float, device=device)

    # affinity_token_mask: 1 for ligand tokens
    affinity_token_mask = torch.zeros(BATCH, N_TOKENS, dtype=torch.float, device=device)
    affinity_token_mask[:, N_PROTEIN:] = 1.0

    feats = {
        "token_to_rep_atom": token_to_rep_atom,
        "token_pad_mask": token_pad_mask,
        "mol_type": mol_type,
        "affinity_token_mask": affinity_token_mask,
    }

    return {
        "s_inputs": s_inputs,
        "z": z,
        "x_pred": x_pred,
        "feats": feats,
        "multiplicity": 1,
        "use_kernels": False,
    }


def _make_interface_mask(device: str = "cpu") -> torch.Tensor:
    """Boolean mask (1, N, N): True where i < N_PROTEIN and j >= N_PROTEIN
    or vice versa (cross-interface pairs)."""
    mask = torch.zeros(BATCH, N_TOKENS, N_TOKENS, dtype=torch.bool, device=device)
    # protein-ligand
    mask[:, :N_PROTEIN, N_PROTEIN:] = True
    # ligand-protein
    mask[:, N_PROTEIN:, :N_PROTEIN] = True
    return mask


# ---------------------------------------------------------------------------
# Test 1: Forward pass + captured_z
# ---------------------------------------------------------------------------

def test_forward_and_captured_z():
    name = "forward_pass_and_captured_z"
    try:
        from interpretability.hooks import InstrumentedAffinityModule

        module = _make_synthetic_module()
        instrumented = InstrumentedAffinityModule(module)
        kwargs = _make_synthetic_inputs()

        num_layers = len(module.pairformer_stack.layers)

        with torch.no_grad():
            out = instrumented(**kwargs)

        # Check output exists and has expected keys.
        has_value = "affinity_pred_value" in out
        has_binary = "affinity_logits_binary" in out
        num_captured = len(instrumented.captured_z)

        if not has_value or not has_binary:
            report(name, False, f"Missing output keys: value={has_value}, binary={has_binary}")
            return instrumented, kwargs

        # Pre-layer-0 capture at key -1.
        assert -1 in instrumented.captured_z, \
            "Pre-layer-0 z was not captured"

        # Expect num_layers post-layer outputs + 1 pre-layer-0 entry.
        expected_count = num_layers + 1
        if num_captured != expected_count:
            report(
                name, False,
                f"captured_z has {num_captured} entries, expected {expected_count}. "
                f"Keys: {sorted(instrumented.captured_z.keys())}",
            )
            return instrumented, kwargs

        # Check shape of each captured z.
        for layer_idx, cz in instrumented.captured_z.items():
            if cz.shape != (BATCH, N_TOKENS, N_TOKENS, TOKEN_Z):
                report(
                    name, False,
                    f"captured_z[{layer_idx}] shape={list(cz.shape)}, "
                    f"expected [{BATCH}, {N_TOKENS}, {N_TOKENS}, {TOKEN_Z}]",
                )
                return instrumented, kwargs

        report(name, True, f"{num_captured} layers captured, shapes OK")
        return instrumented, kwargs

    except Exception:
        report(name, False, traceback.format_exc())
        return None, None


# ---------------------------------------------------------------------------
# Test 2: Logit lens
# ---------------------------------------------------------------------------

def test_logit_lens(instrumented, kwargs):
    name = "logit_lens"
    try:
        from interpretability.logit_lens import compute_logit_lens

        if instrumented is None:
            report(name, False, "Skipped — forward pass failed")
            return

        interface_mask = _make_interface_mask()
        layer_z = instrumented.captured_z

        pseudo = compute_logit_lens(instrumented, layer_z, interface_mask)

        num_layers = len(instrumented.module.pairformer_stack.layers)
        num_results = len(pseudo)

        # Expected: num_layers post-layer outputs + 1 pre-layer-0 entry.
        num_expected = num_layers + 1

        if num_results != num_expected:
            report(
                name, False,
                f"compute_logit_lens returned {num_results} entries, "
                f"expected {num_expected}. Keys: {sorted(pseudo.keys())}",
            )
            return

        # Each value should be a (B, 1) tensor.
        for layer_idx, val in pseudo.items():
            if val.shape != (BATCH, 1):
                report(
                    name, False,
                    f"pseudo_ic50[{layer_idx}] shape={list(val.shape)}, expected [{BATCH}, 1]",
                )
                return

        sample_val = list(pseudo.values())[0].item()
        report(name, True, f"{num_results} layers, sample pseudo_ic50={sample_val:.4f}")

    except Exception:
        report(name, False, traceback.format_exc())


# ---------------------------------------------------------------------------
# Test 3: SVD extraction
# ---------------------------------------------------------------------------

def test_svd_extraction():
    name = "svd_extraction"
    try:
        from interpretability.svd_extractor import extract_head_circuits

        module = _make_synthetic_module()
        circuits = extract_head_circuits(module, layer_idx=0, head_idx=0)

        expected_keys = {
            "qk_U", "qk_S", "qk_Vh",
            "ov_U", "ov_S", "ov_Vh",
            "spectrum_ratio",
            "top_qk_query_dir", "top_qk_key_dir",
            "top_ov_input_dir", "top_ov_output_dir",
        }
        missing = expected_keys - set(circuits.keys())
        if missing:
            report(name, False, f"Missing keys: {missing}")
            return

        sr = circuits["spectrum_ratio"]
        if not isinstance(sr, float):
            report(name, False, f"spectrum_ratio type={type(sr).__name__}, expected float")
            return

        if sr < 0 or not math.isfinite(sr):
            report(name, False, f"spectrum_ratio={sr}, expected non-negative finite")
            return

        report(name, True, f"spectrum_ratio={sr:.4f}, all keys present")

    except Exception:
        report(name, False, traceback.format_exc())


# ---------------------------------------------------------------------------
# Test 4: Causal patching (zero_pairs)
# ---------------------------------------------------------------------------

def test_patch_zero_pairs():
    name = "patch_zero_pairs"
    try:
        from interpretability.hooks import InstrumentedAffinityModule
        from interpretability.patcher import patch_and_measure

        module = _make_synthetic_module()
        instrumented = InstrumentedAffinityModule(module)
        kwargs = _make_synthetic_inputs()

        # Clean run for baseline.
        with torch.no_grad():
            clean_out = instrumented(**kwargs)
        baseline_ic50 = clean_out["affinity_pred_value"].item()

        # Patch: zero all interface pairs.
        pair_mask = torch.zeros(N_TOKENS, N_TOKENS, dtype=torch.bool)
        pair_mask[:N_PROTEIN, N_PROTEIN:] = True
        pair_mask[N_PROTEIN:, :N_PROTEIN] = True

        patch_spec = {
            "type": "zero_pairs",
            "layer_idx": 0,
            "pair_mask": pair_mask,
        }

        effect = patch_and_measure(instrumented, kwargs, patch_spec, baseline_ic50)

        if not isinstance(effect, float):
            report(name, False, f"Returned type={type(effect).__name__}, expected float")
            return

        report(
            name, True,
            f"baseline={baseline_ic50:.4f}, effect={effect:+.4f}, "
            f"patched={baseline_ic50 + effect:.4f}",
        )

    except Exception:
        report(name, False, traceback.format_exc())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Interpretability Pipeline — Synthetic Validation")
    print("=" * 60)
    print()

    # Test 1
    instrumented, kwargs = test_forward_and_captured_z()

    # Test 2
    test_logit_lens(instrumented, kwargs)

    # Test 3
    test_svd_extraction()

    # Test 4
    test_patch_zero_pairs()

    # Summary
    print()
    print("-" * 60)
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        print("Failed tests:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
    print("-" * 60)


if __name__ == "__main__":
    main()
