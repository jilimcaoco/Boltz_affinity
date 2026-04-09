"""
scripts/interpretability/run_patching_experiment.py
=====================================================
Dataset-type–specific causal patching experiments on cached AffinityModule
inputs.  Implements four experiment types that target different circuit
hypotheses:

Experiments
-----------
head_ablation
    For every (layer, head) pair, zero the softmax weights of tri_att_start
    and measure the IC50 shift.  Use for dataset types 1 (congeneric series)
    and 3 (physicochemical decoupling) to find heads that drive affinity
    differentiation.

    Output: {complex}_head_ablation.csv
            columns: layer, head, causal_effect

alanine_scan
    For every protein token, zero all z[i, lig] and z[lig, i] cross-pairs
    at each layer and measure the IC50 shift.  Mimics alanine scanning:
    which residue–ligand contacts are causally important?  Use for
    dataset type 2.

    Output: {complex}_alanine_scan.csv
            columns: residue_idx, layer, causal_effect

target_shuffle
    At each layer, replace z entering that layer with z from a reference
    complex (a different protein target) using the swap_z patch.  Measures
    which layer encodes target identity.  Use for dataset type 4.

    Output: {complex}_target_shuffle.csv
            columns: layer, reference_complex, causal_effect

    NOTE: Requires the two complexes to have the same number of tokens.
    If they differ, the script will raise ValueError and exit non-zero.

distance_perturb
    Translate ligand representative atoms by a series of displacements
    in random unit-vector directions.  Replays AffinityModule.forward()
    directly on the cached file (no re-caching needed); trunk z is fixed,
    only the distance-bin features change.  Use for dataset type 5.

    Output: {complex}_distance_perturb.csv
            columns: displacement_A, direction_idx, ic50, causal_effect

    IMPORTANT asymmetry: trunk z reflects the original DOCK3.8 pose.
    Distance bins reflect the perturbed coordinates.  This experiment
    isolates the distance circuit, not the full pipeline response.

Usage
-----
    python scripts/interpretability/run_patching_experiment.py \\
        --complex_name  TYK2_inhibitor \\
        --cached_inputs cached/TYK2_inhibitor.pt \\
        --checkpoint    ~/.boltz/boltz2_aff.ckpt \\
        --output_dir    results/patching/ \\
        --experiment    head_ablation

    # alanine scan — all protein tokens:
        --experiment    alanine_scan

    # target shuffle — needs reference complex cached .pt:
        --experiment    target_shuffle \\
        --reference_cached cached/CDK2_inhibitor.pt

    # distance perturbation:
        --experiment    distance_perturb \\
        --displacements 0.5,1.0,1.5,2.0 \\
        --n_directions  8
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
_INTERP_DIR = _PROJECT_ROOT / "interpretability"

for _p in (_SRC_DIR, _INTERP_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse module loading helpers from run_analysis.py
sys.path.insert(0, str(_INTERP_DIR))
from run_analysis import (  # noqa: E402
    _load_affinity_module,
    _build_forward_kwargs,
)
from interpretability.hooks import InstrumentedAffinityModule  # noqa: E402
from interpretability.patcher import patch_and_measure         # noqa: E402


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  WARNING: no rows to write for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Experiment 1: Head ablation
# ---------------------------------------------------------------------------

def run_head_ablation(
    instrumented: InstrumentedAffinityModule,
    forward_kwargs: dict,
    baseline_ic50: float,
    output_dir: Path,
    complex_name: str,
) -> None:
    """Zero each tri_att_start head at each layer and measure IC50 shift."""
    affinity_module = instrumented.module
    num_layers = len(affinity_module.pairformer_stack.layers)
    num_heads = affinity_module.pairformer_stack.layers[0].tri_att_start.mha.no_heads

    rows: list[dict] = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            patch_spec = {
                "type": "zero_head",
                "layer_idx": layer_idx,
                "head_idx": head_idx,
            }
            effect = patch_and_measure(
                instrumented, forward_kwargs, patch_spec, baseline_ic50
            )
            rows.append({
                "layer": layer_idx,
                "head": head_idx,
                "causal_effect": f"{effect:.6f}",
                "baseline_ic50": f"{baseline_ic50:.6f}",
            })
            print(
                f"  layer={layer_idx} head={head_idx}  "
                f"effect={effect:+.4f}"
            )

    _write_csv(rows, output_dir / f"{complex_name}_head_ablation.csv")


# ---------------------------------------------------------------------------
# Experiment 2: Alanine scan
# ---------------------------------------------------------------------------

def run_alanine_scan(
    instrumented: InstrumentedAffinityModule,
    forward_kwargs: dict,
    baseline_ic50: float,
    cached: dict,
    output_dir: Path,
    complex_name: str,
) -> None:
    """Zero cross-pairs for each protein token at each layer."""
    affinity_module = instrumented.module
    num_layers = len(affinity_module.pairformer_stack.layers)

    is_protein: torch.Tensor = cached.get("is_protein_mask")
    ligand_range: tuple[int, int] = cached.get("ligand_token_range")

    if is_protein is None or ligand_range is None:
        raise ValueError(
            "Cached inputs are missing 'is_protein_mask' or 'ligand_token_range'. "
            "Re-run cache_affinity_inputs.py with an up-to-date version."
        )

    N = is_protein.shape[0]
    lig_start, lig_end = ligand_range
    protein_indices = is_protein.nonzero(as_tuple=False).squeeze(-1).tolist()

    print(
        f"  {len(protein_indices)} protein tokens × "
        f"{num_layers} layers = {len(protein_indices) * num_layers} patches"
    )

    rows: list[dict] = []
    for res_idx in protein_indices:
        for layer_idx in range(num_layers):
            pair_mask = torch.zeros(N, N, dtype=torch.bool)
            # Zero all pairs between this residue and every ligand token.
            pair_mask[res_idx, lig_start:lig_end] = True
            pair_mask[lig_start:lig_end, res_idx] = True

            patch_spec = {
                "type": "zero_pairs",
                "layer_idx": layer_idx,
                "pair_mask": pair_mask,
            }
            effect = patch_and_measure(
                instrumented, forward_kwargs, patch_spec, baseline_ic50
            )
            rows.append({
                "residue_idx": res_idx,
                "layer": layer_idx,
                "causal_effect": f"{effect:.6f}",
                "baseline_ic50": f"{baseline_ic50:.6f}",
            })

        if res_idx % 20 == 0:
            print(f"  ... residue {res_idx}/{N}")

    _write_csv(rows, output_dir / f"{complex_name}_alanine_scan.csv")


# ---------------------------------------------------------------------------
# Experiment 3: Target shuffle
# ---------------------------------------------------------------------------

def run_target_shuffle(
    instrumented: InstrumentedAffinityModule,
    forward_kwargs: dict,
    baseline_ic50: float,
    reference_cached_path: Path,
    output_dir: Path,
    complex_name: str,
) -> None:
    """Replace z at each layer with z from a reference complex."""
    affinity_module = instrumented.module
    num_layers = len(affinity_module.pairformer_stack.layers)

    ref_cached = torch.load(reference_cached_path, map_location="cpu", weights_only=False)
    ref_name = reference_cached_path.stem

    # The reference z is the trunk output — not the AffinityModule input.
    # Use the "z" key which is z captured before AffinityModule processes it.
    ref_z = ref_cached["z"]
    current_z = forward_kwargs["z"]

    if ref_z.shape != current_z.shape:
        raise ValueError(
            f"Shape mismatch: current z={list(current_z.shape)}, "
            f"reference z={list(ref_z.shape)}.\n"
            "Target shuffling requires complexes with identical token counts. "
            "Consider trimming or padding to the same sequence length."
        )

    rows: list[dict] = []
    for layer_idx in range(num_layers):
        patch_spec = {
            "type": "swap_z",
            "layer_idx": layer_idx,
            "swap_tensor": ref_z,
        }
        effect = patch_and_measure(
            instrumented, forward_kwargs, patch_spec, baseline_ic50
        )
        rows.append({
            "layer": layer_idx,
            "reference_complex": ref_name,
            "causal_effect": f"{effect:.6f}",
            "baseline_ic50": f"{baseline_ic50:.6f}",
        })
        print(f"  layer={layer_idx}  effect={effect:+.4f}")

    _write_csv(rows, output_dir / f"{complex_name}_target_shuffle.csv")


# ---------------------------------------------------------------------------
# Experiment 4: Distance perturbation
# ---------------------------------------------------------------------------

def run_distance_perturb(
    instrumented: InstrumentedAffinityModule,
    forward_kwargs: dict,
    baseline_ic50: float,
    cached: dict,
    displacements: list[float],
    n_directions: int,
    output_dir: Path,
    complex_name: str,
    seed: int = 42,
) -> None:
    """Translate ligand representative atoms by fixed displacements.

    Trunk z is unchanged (loaded from cache).  Only distance bins change
    because AffinityModule.forward() recomputes them from x_pred internally.
    This isolates the distance circuit from the trunk representation.
    """
    ligand_range: tuple[int, int] = cached.get("ligand_token_range")
    if ligand_range is None:
        raise ValueError("Cached inputs missing 'ligand_token_range'.")

    lig_start, lig_end = ligand_range
    feats = forward_kwargs["feats"]

    # token_to_rep_atom: (B, N_tokens, N_atoms) — one-hot assignment.
    token_to_rep_atom = feats["token_to_rep_atom"]  # (1, N_tokens, N_atoms)

    # Find the representative atom index for each ligand token.
    lig_rep_atoms: list[int] = (
        token_to_rep_atom[0, lig_start:lig_end, :]
        .argmax(dim=-1)
        .tolist()
    )
    lig_rep_atoms_t = torch.tensor(lig_rep_atoms, dtype=torch.long)

    # Generate fixed random unit-vector directions (reproducible).
    rng = torch.Generator()
    rng.manual_seed(seed)
    directions = torch.randn(n_directions, 3, generator=rng)
    directions = directions / directions.norm(dim=-1, keepdim=True)  # (D, 3)

    original_x_pred = forward_kwargs["x_pred"].clone()  # (1, N_atoms, 3)

    rows: list[dict] = []
    for displacement in displacements:
        for dir_idx, direction in enumerate(directions):
            x_pred = original_x_pred.clone()
            # Translate ligand representative atoms.
            x_pred[0, lig_rep_atoms_t, :] += displacement * direction

            patched_kwargs = {**forward_kwargs, "x_pred": x_pred}
            with torch.no_grad():
                out = instrumented(**patched_kwargs)
            patched_ic50 = out["affinity_pred_value"].item()
            effect = patched_ic50 - baseline_ic50

            rows.append({
                "displacement_A": f"{displacement:.3f}",
                "direction_idx": dir_idx,
                "dir_x": f"{direction[0].item():.4f}",
                "dir_y": f"{direction[1].item():.4f}",
                "dir_z": f"{direction[2].item():.4f}",
                "ic50": f"{patched_ic50:.6f}",
                "causal_effect": f"{effect:.6f}",
                "baseline_ic50": f"{baseline_ic50:.6f}",
            })

        print(
            f"  displacement={displacement:.2f}Å  "
            f"mean_effect={sum(float(r['causal_effect']) for r in rows[-n_directions:]) / n_directions:+.4f}"
        )

    _write_csv(rows, output_dir / f"{complex_name}_distance_perturb.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a dataset-type–specific causal patching experiment."
    )
    parser.add_argument("--complex_name",  required=True)
    parser.add_argument("--cached_inputs", required=True,
                        help="Path to .pt file from cache_affinity_inputs.py.")
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to boltz2_aff.ckpt.")
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["head_ablation", "alanine_scan", "target_shuffle", "distance_perturb"],
        help="Which patching experiment to run.",
    )
    # target_shuffle only
    parser.add_argument("--reference_cached", default=None,
                        help="Path to reference complex .pt file (target_shuffle only).")
    # distance_perturb only
    parser.add_argument("--displacements", default="0.5,1.0,1.5,2.0",
                        help="Comma-separated displacement values in Å.")
    parser.add_argument("--n_directions", type=int, default=8,
                        help="Number of random displacement directions.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    cached_path = Path(args.cached_inputs).expanduser()
    output_dir  = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cached inputs from {cached_path} ...")
    cached = torch.load(cached_path, map_location=args.device, weights_only=False)

    print(f"Loading AffinityModule from {args.checkpoint} ...")
    affinity_module = _load_affinity_module(
        Path(args.checkpoint).expanduser(), args.device
    )

    instrumented = InstrumentedAffinityModule(affinity_module)
    forward_kwargs = _build_forward_kwargs(cached, args.device)

    # Baseline forward pass.
    with torch.no_grad():
        clean_out = instrumented(**forward_kwargs)
    baseline_ic50 = clean_out["affinity_pred_value"].item()
    print(f"Baseline IC50: {baseline_ic50:.6f}")

    # Dispatch.
    if args.experiment == "head_ablation":
        run_head_ablation(
            instrumented, forward_kwargs, baseline_ic50,
            output_dir, args.complex_name,
        )

    elif args.experiment == "alanine_scan":
        run_alanine_scan(
            instrumented, forward_kwargs, baseline_ic50,
            cached, output_dir, args.complex_name,
        )

    elif args.experiment == "target_shuffle":
        if not args.reference_cached:
            parser.error("--reference_cached is required for target_shuffle")
        run_target_shuffle(
            instrumented, forward_kwargs, baseline_ic50,
            Path(args.reference_cached).expanduser(),
            output_dir, args.complex_name,
        )

    elif args.experiment == "distance_perturb":
        displacements = [float(x) for x in args.displacements.split(",")]
        run_distance_perturb(
            instrumented, forward_kwargs, baseline_ic50,
            cached, displacements, args.n_directions,
            output_dir, args.complex_name,
        )

    instrumented.remove_hooks()
    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
