"""
interpretability/run_analysis.py
=================================
Command-line script that runs a full interpretability analysis on one
complex and saves all outputs.

Usage
-----
    python interpretability/run_analysis.py \\
        --complex_name TYK2_inhibitor \\
        --cached_inputs cached_inputs_tyk2.pt \\
        --checkpoint ~/.boltz/boltz2_aff.ckpt \\
        --output_dir results/ \\
        --run_logit_lens \\
        --run_svd \\
        --run_patch '{"type": "zero_head", "layer_idx": 0, "head_idx": 2}'

Outputs (all written to --output_dir):
  {complex_name}_logit_lens.csv    — layer, pseudo_ic50
  {complex_name}_svd.csv           — layer, head, circuit, direction_name, ...
  Patch result printed to stdout

Requires a cached .pt file produced by cache_affinity_inputs.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running from the interpretability/ folder.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cached_inputs(path: Path, device: str) -> dict:
    """Load the .pt file produced by cache_affinity_inputs.py."""
    data = torch.load(path, map_location=device, weights_only=False)
    return data


def _load_affinity_module(
    checkpoint: Path,
    device: str,
):
    """Load the full Boltz2 checkpoint and return the AffinityModule.

    Uses ``AffinityModelManager`` from the affinity-rescoring pipeline
    which handles checkpoint patching, device placement, and correct
    predict_args (sampling_steps=0, no diffusion).

    The AffinityModule is stored as an attribute of the Boltz2
    LightningModule:
      - model.affinity_module   (non-ensemble)
      - model.affinity_module1  (ensemble, first head)

    Returns
    -------
    tuple[AffinityModule, dict]
        The AffinityModule and a dict of embedding tables extracted from
        the full model for chemical interpretability projections:
        ``{"residue_embedding": Tensor (num_tokens, token_s) or None,
           "residue_embedding_z": Tensor (num_tokens, token_z) or None}``
    """
    from boltz.affinity_rescoring.inference import (
        AffinityModelManager,
        _get_module,
    )
    from boltz.affinity_rescoring.models import DeviceOption

    device_option = DeviceOption.CUDA if device == "cuda" else DeviceOption.CPU
    manager = AffinityModelManager(device=device_option)
    model = manager.load_model(checkpoint_path=str(checkpoint))

    # Extract the AffinityModule (unwrap torch.compile if needed).
    if hasattr(model, "affinity_module"):
        affinity_mod = _get_module(model, "affinity_module")
    elif hasattr(model, "affinity_module1"):
        affinity_mod = _get_module(model, "affinity_module1")
    else:
        raise RuntimeError(
            "Checkpoint does not contain an affinity module.  "
            "Make sure you use the boltz2_aff.ckpt checkpoint."
        )

    affinity_mod = affinity_mod.to(device)
    affinity_mod.eval()

    # --- Extract embedding tables for chemical interpretability ---
    # res_type_encoding lives in the trunk InputEmbedder (token_s space).
    # SVD directions live in token_z (pair-representation) space.
    # We project via the AffinityModule's s_to_z_prod_in1 to bridge.
    embedding_tables: dict = {
        "residue_embedding": None,
        "residue_embedding_z": None,
    }
    try:
        # model.input_embedder.res_type_encoding.weight: (token_s, num_tokens)
        res_enc_weight = model.input_embedder.res_type_encoding.weight  # (token_s, num_tokens)
        res_emb_s = res_enc_weight.T.detach().to(device)  # (num_tokens, token_s)
        embedding_tables["residue_embedding"] = res_emb_s

        # Project into token_z space via s_to_z_prod_in1.
        # s_to_z_prod_in1.weight: (token_z, token_s)
        s_to_z = affinity_mod.s_to_z_prod_in1.weight  # (token_z, token_s)
        res_emb_z = (res_emb_s @ s_to_z.T).detach()  # (num_tokens, token_z)
        embedding_tables["residue_embedding_z"] = res_emb_z
    except AttributeError:
        print("  WARNING: Could not extract residue embeddings from trunk.")

    return affinity_mod, embedding_tables


def _build_forward_kwargs(
    cached: dict,
    device: str,
) -> dict:
    """Assemble the kwargs expected by AffinityModule.forward() from the
    cached .pt file.

    AffinityModule.forward signature:
        forward(s_inputs, z, x_pred, feats, multiplicity=1, use_kernels=False)

    Required feats keys (accessed in AffinityModule.forward):
        token_to_rep_atom, token_pad_mask, mol_type, affinity_token_mask
    """
    s_inputs = cached["s_inputs"].to(device)
    z = cached["z"].to(device)

    # IMPORTANT — distance perturbation asymmetry (dataset type 5):
    # When x_pred is perturbed from the original DOCK3.8 pose,
    # the two sources of geometric information in the affinity
    # module become desynchronised:
    #   - trunk z : reflects ORIGINAL pose geometry
    #               (fixed, loaded from cached .pt file)
    #   - distance bins : recomputed from PERTURBED x_pred
    #                     coordinates inside AffinityModule.forward()
    # This is intentional for circuit isolation — it isolates
    # how the module's distance circuit responds to coordinate
    # changes independently of the trunk representation.
    # This is NOT equivalent to running the full Boltz-2
    # pipeline on the perturbed pose.
    # Document this distinction in any paper or report
    # using type 5 perturbation data.
    x_pred = cached["x_pred"].to(device)

    # Reconstruct the feats dict from cached sub-keys.
    # cache_affinity_inputs.py saves feats as a sub-dict "feats" containing
    # the necessary keys with their original batch dimensions.
    feats = {k: v.to(device) for k, v in cached["feats"].items()}

    return {
        "s_inputs": s_inputs,
        "z": z,
        "x_pred": x_pred,
        "feats": feats,
        "multiplicity": 1,
        "use_kernels": False,
    }


def _build_interface_mask(cached: dict, device: str) -> torch.Tensor:
    """Return the cross-pair mask as a (B, N, N) boolean tensor."""
    mask = cached["cross_pair_mask"]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return mask.to(device).bool()


# ---------------------------------------------------------------------------
# Analysis steps
# ---------------------------------------------------------------------------

def _run_logit_lens(
    instrumented_module,
    forward_kwargs: dict,
    interface_mask: torch.Tensor,
    output_dir: Path,
    complex_name: str,
) -> dict[int, float]:
    """Run logit lens and save CSV.  Returns {layer: pseudo_ic50}."""
    from interpretability.logit_lens import compute_logit_lens

    # Run the forward pass to populate captured_z.
    instrumented_module(**forward_kwargs)
    layer_z = instrumented_module.captured_z

    pseudo_ic50s = compute_logit_lens(instrumented_module, layer_z, interface_mask)

    # Write CSV.
    csv_path = output_dir / f"{complex_name}_logit_lens.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "pseudo_ic50"])
        trajectory = {}
        for layer_idx in sorted(pseudo_ic50s.keys()):
            val = pseudo_ic50s[layer_idx].item()
            writer.writerow([layer_idx, f"{val:.6f}"])
            trajectory[layer_idx] = val

    print(f"  Logit lens saved to {csv_path}")
    return trajectory


def _run_svd(
    affinity_module,
    output_dir: Path,
    complex_name: str,
    embedding_tables: dict | None = None,
):
    """Run SVD extraction for every layer×head and save CSV."""
    from interpretability.svd_extractor import extract_head_circuits
    from interpretability.chemical_basis import classify_direction

    layers = affinity_module.pairformer_stack.layers
    num_layers = len(layers)
    # All tri_att_start heads have the same count.
    num_heads = layers[0].tri_att_start.mha.no_heads

    # Prepare embedding tables for chemical projection.
    # SVD directions are in token_z space, so use the z-projected table.
    res_emb_z = None
    if embedding_tables is not None:
        res_emb_z = embedding_tables.get("residue_embedding_z")

    csv_path = output_dir / f"{complex_name}_svd.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "head", "circuit", "direction_name",
            "top_residue_types", "spectrum_ratio",
        ])

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                circuits = extract_head_circuits(
                    affinity_module, layer_idx, head_idx,
                )
                spectrum_ratio = circuits["spectrum_ratio"]

                # Four principal directions to classify.
                direction_map = {
                    "qk_query": ("qk", circuits["top_qk_query_dir"]),
                    "qk_key": ("qk", circuits["top_qk_key_dir"]),
                    "ov_input": ("ov", circuits["top_ov_input_dir"]),
                    "ov_output": ("ov", circuits["top_ov_output_dir"]),
                }

                for dir_name, (circuit_type, direction) in direction_map.items():
                    try:
                        label = classify_direction(
                            direction, affinity_module, top_k=3,
                            residue_embedding_table=res_emb_z,
                        )
                    except (ValueError, RuntimeError):
                        label = "<no embedding tables available>"

                    writer.writerow([
                        layer_idx,
                        head_idx,
                        circuit_type,
                        dir_name,
                        label.replace("\n", " | "),
                        f"{spectrum_ratio:.4f}",
                    ])

    print(f"  SVD analysis saved to {csv_path}")


def _run_activation_svd(
    instrumented_module,
    forward_kwargs: dict,
    interface_mask: torch.Tensor,
    output_dir: Path,
    complex_name: str,
    embedding_tables: dict | None = None,
) -> None:
    """Run activation-level SVD on captured z-deltas and attention patterns.

    Produces two CSVs:
      {complex_name}_z_delta_svd.csv   — per-layer z-update decomposition
      {complex_name}_attn_svd.csv      — per-(layer,head) attention pattern SVD

    These complement the weight SVD (which is input-independent) by showing
    what the model *actually computes* on this specific input.
    """
    from interpretability.svd_extractor import (
        extract_attention_svd,
        extract_z_delta_svd,
        extract_head_circuits,
    )
    import torch.nn.functional as F

    # Run forward pass to populate captured_z and captured_attn.
    instrumented_module(**forward_kwargs)
    layer_z = instrumented_module.captured_z
    layer_attn = instrumented_module.captured_attn

    # Get metadata for labelling token positions.
    affinity_module = instrumented_module.module
    num_layers = len(affinity_module.pairformer_stack.layers)
    num_heads = affinity_module.pairformer_stack.layers[0].tri_att_start.mha.no_heads

    # --- Z-delta SVD ---
    z_delta_rows: list[dict] = []
    for layer_idx in range(num_layers):
        try:
            result = extract_z_delta_svd(
                layer_z, layer_idx, interface_mask=interface_mask, top_k=5,
            )
        except ValueError:
            continue

        sv = result["singular_values"]
        channel_dirs = result["channel_dirs"]  # (k, token_z)

        # Compare top channel direction with OV output directions from weight SVD.
        ov_alignments: list[str] = []
        if channel_dirs.shape[0] > 0:
            top_channel = channel_dirs[0]  # (token_z,)
            for head_idx in range(num_heads):
                circuits = extract_head_circuits(affinity_module, layer_idx, head_idx)
                ov_out = circuits["top_ov_output_dir"]
                cos = F.cosine_similarity(
                    top_channel.unsqueeze(0), ov_out.unsqueeze(0)
                ).item()
                ov_alignments.append(f"h{head_idx}:{cos:.3f}")

        for rank_k in range(len(sv)):
            top_i, top_j = result["top_pair_idx"][rank_k] if rank_k < len(result["top_pair_idx"]) else (-1, -1)
            z_delta_rows.append({
                "layer": layer_idx,
                "rank": rank_k,
                "singular_value": f"{sv[rank_k].item():.6f}",
                "effective_rank": f"{result['effective_rank']:.4f}",
                "top_pair_i": top_i,
                "top_pair_j": top_j,
                "ov_alignment": "; ".join(ov_alignments) if rank_k == 0 else "",
            })

    z_csv = output_dir / f"{complex_name}_z_delta_svd.csv"
    if z_delta_rows:
        with open(z_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(z_delta_rows[0].keys()))
            writer.writeheader()
            writer.writerows(z_delta_rows)
        print(f"  Z-delta SVD saved to {z_csv}")

    # --- Attention pattern SVD ---
    attn_rows: list[dict] = []
    for layer_idx in range(num_layers):
        if layer_idx not in layer_attn:
            continue
        for direction_name in ("tri_att_start", "tri_att_end"):
            head_dict = layer_attn[layer_idx].get(direction_name, {})
            for head_idx in sorted(head_dict.keys()):
                attn_w = head_dict[head_idx].unsqueeze(0)  # (1, N, N)
                result = extract_attention_svd(attn_w, top_k=3)

                sv = result["singular_values"]
                for rank_k in range(len(sv)):
                    q_idx = result["top_query_idx"][rank_k] if rank_k < len(result["top_query_idx"]) else -1
                    k_idx = result["top_key_idx"][rank_k] if rank_k < len(result["top_key_idx"]) else -1
                    attn_rows.append({
                        "layer": layer_idx,
                        "direction": direction_name,
                        "head": head_idx,
                        "rank": rank_k,
                        "singular_value": f"{sv[rank_k].item():.6f}",
                        "effective_rank": f"{result['effective_rank']:.4f}",
                        "top_query_token": q_idx,
                        "top_key_token": k_idx,
                    })

    attn_csv = output_dir / f"{complex_name}_attn_svd.csv"
    if attn_rows:
        with open(attn_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(attn_rows[0].keys()))
            writer.writeheader()
            writer.writerows(attn_rows)
        print(f"  Attention SVD saved to {attn_csv}")


def _run_patch(
    instrumented_module,
    forward_kwargs: dict,
    baseline_ic50: float,
    patch_spec: dict,
):
    """Run a causal patch and print the result."""
    from interpretability.patcher import patch_and_measure

    effect = patch_and_measure(
        instrumented_module,
        forward_kwargs,
        patch_spec,
        baseline_ic50,
    )
    print(f"  Patch spec: {json.dumps(patch_spec, default=str)}")
    print(f"  Baseline IC50:  {baseline_ic50:.6f}")
    print(f"  Causal effect:  {effect:+.6f}")
    print(f"  Patched IC50:   {baseline_ic50 + effect:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a full interpretability analysis on one complex using "
            "cached AffinityModule inputs."
        ),
    )
    parser.add_argument(
        "--complex_name",
        required=True,
        help="Human-readable name for the complex (used in output filenames).",
    )
    parser.add_argument(
        "--cached_inputs",
        required=True,
        help="Path to .pt file from cache_affinity_inputs.py.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to Boltz2 affinity checkpoint (e.g. boltz2_aff.ckpt).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where result CSVs are written.",
    )
    parser.add_argument(
        "--run_logit_lens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run logit-lens analysis (default: True).  Use --no-run_logit_lens to skip.",
    )
    parser.add_argument(
        "--run_svd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run SVD circuit analysis (default: True).  Use --no-run_svd to skip.",
    )
    parser.add_argument(
        "--run_activation_svd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run activation-level SVD (default: True).  Use --no-run_activation_svd to skip.",
    )
    parser.add_argument(
        "--run_patch",
        default=None,
        help=(
            "Optional patch specification as a JSON string.  "
            'E.g. \'{"type": "zero_head", "layer_idx": 0, "head_idx": 2}\''
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu).",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------
    warnings.filterwarnings("ignore", ".*that has Tensor Cores.*")
    torch.set_grad_enabled(False)

    cached_inputs_path = Path(args.cached_inputs).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ---------------------------------------------------------------
    # 1. Load cached inputs
    # ---------------------------------------------------------------
    print(f"Loading cached inputs from {cached_inputs_path} ...")
    cached = _load_cached_inputs(cached_inputs_path, device)

    # ---------------------------------------------------------------
    # 2. Load AffinityModule and wrap in InstrumentedAffinityModule
    # ---------------------------------------------------------------
    print(f"Loading AffinityModule from {checkpoint_path} ...")
    affinity_module, embedding_tables = _load_affinity_module(checkpoint_path, device)

    from interpretability.hooks import InstrumentedAffinityModule
    instrumented = InstrumentedAffinityModule(affinity_module)

    # ---------------------------------------------------------------
    # 3. Clean forward pass → baseline IC50
    # ---------------------------------------------------------------
    print("Running clean forward pass ...")
    forward_kwargs = _build_forward_kwargs(cached, device)
    interface_mask = _build_interface_mask(cached, device)

    with torch.no_grad():
        clean_out = instrumented(**forward_kwargs)

    baseline_ic50 = clean_out["affinity_pred_value"].item()
    print(f"  Baseline IC50: {baseline_ic50:.6f}")

    # ---------------------------------------------------------------
    # 4. Logit lens
    # ---------------------------------------------------------------
    logit_lens_trajectory: dict[int, float] = {}
    if args.run_logit_lens:
        print("Running logit lens analysis ...")
        logit_lens_trajectory = _run_logit_lens(
            instrumented, forward_kwargs, interface_mask,
            output_dir, args.complex_name,
        )

    # ---------------------------------------------------------------
    # 5. SVD circuit analysis
    # ---------------------------------------------------------------
    if args.run_svd:
        print("Running SVD circuit analysis ...")
        _run_svd(affinity_module, output_dir, args.complex_name,
                 embedding_tables=embedding_tables)

    # ---------------------------------------------------------------
    # 5b. Activation SVD analysis
    # ---------------------------------------------------------------
    if args.run_activation_svd:
        print("Running activation SVD analysis ...")
        _run_activation_svd(
            instrumented, forward_kwargs, interface_mask,
            output_dir, args.complex_name, embedding_tables,
        )

    # ---------------------------------------------------------------
    # 6. Causal patching (optional)
    # ---------------------------------------------------------------
    if args.run_patch:
        print("Running causal patch ...")
        patch_spec = json.loads(args.run_patch)
        _run_patch(instrumented, forward_kwargs, baseline_ic50, patch_spec)

    # ---------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------
    largest_drop_layer = "N/A"
    if logit_lens_trajectory:
        sorted_layers = sorted(logit_lens_trajectory.keys())
        if len(sorted_layers) >= 2:
            max_drop = 0.0
            for i in range(1, len(sorted_layers)):
                prev = sorted_layers[i - 1]
                curr = sorted_layers[i]
                drop = abs(
                    logit_lens_trajectory[curr] - logit_lens_trajectory[prev]
                )
                if drop > max_drop:
                    max_drop = drop
                    largest_drop_layer = str(curr)

    print(
        f"\nSUMMARY: complex={args.complex_name}  "
        f"baseline_ic50={baseline_ic50:.4f}  "
        f"largest_logit_lens_drop_layer={largest_drop_layer}"
    )

    # Cleanup hooks.
    instrumented.remove_hooks()


if __name__ == "__main__":
    main()
