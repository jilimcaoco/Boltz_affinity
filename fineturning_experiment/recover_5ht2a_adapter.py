#!/usr/bin/env python3
"""Recover meta.json for 5ht2a_v2_rank_64 from checkpoint"""

import sys
import time
import torch
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from boltz.lora.adapter import (
    LoRAAdapter,
    LoRAConfig,
    TrainingRun,
    BOLTZ_LORA_FORMAT_VERSION,
)

adapter_dir = Path(__file__).parent / "adapters" / "5ht2a_v2_rank_64"

# Load the raw training checkpoint from adapter.pt
ckpt = torch.load(adapter_dir / "adapter.pt", map_location="cpu", weights_only=False)

print("Checkpoint keys      :", list(ckpt.keys()))
print("Epochs completed     :", ckpt.get("epoch_completed"))
print("Metrics count        :", len(ckpt.get("metrics", [])))
model_state_keys = list(ckpt.get("model_state", {}).keys())
print("Model state keys (3) :", model_state_keys[:3])

if not model_state_keys:
    sys.exit("ERROR: model_state is empty — this checkpoint may not contain LoRA weights.")

# Reconstruct proper adapter.pt format
config = LoRAConfig(
    rank=32,
    alpha=64.0,
    dropout=0.0,
    target_spec="heads_pairformer",
    target_patterns=[],
)
proper_blob = {
    "state_dict": ckpt["model_state"],
    "config": config.to_dict(),
    "format_version": BOLTZ_LORA_FORMAT_VERSION,
}
torch.save(proper_blob, adapter_dir / "adapter.pt")
print("✓ Wrote adapter.pt (proper format, %d tensors)" % len(ckpt["model_state"]))

# Reconstruct meta.json
manifest_path = str(
    (Path(__file__).parent / "manifests" / "lora_manifest_5HT2A.csv").resolve()
)
run = TrainingRun(
    started_at=ckpt.get("started_at", time.time()),
    finished_at=time.time(),
    data_manifest=manifest_path,
    mode="rescore",
    loss_spec="boltz2_affinity",
    epochs=20,
    learning_rate=1e-4,
    batch_size=32,
    metrics=ckpt.get("metrics", []),
    notes=(
        "5HT2A LoRA rank=32 alpha=64 — meta.json recovered from checkpoint "
        "after FileExistsError bug (checkpoint dir pre-existed registry.save)"
    ),
)
adapter = LoRAAdapter(
    name="5ht2a_v2_rank_64",
    config=config,
    history=[run],
)
(adapter_dir / "meta.json").write_text(adapter.to_json())
print("✓ Wrote meta.json")

# Update registry
from boltz.lora.registry import default_registry
registry = default_registry()
idx = registry._read_index()
idx["5ht2a_v2_rank_64"] = {
    "path": str(adapter_dir.resolve()),
    "created_at": adapter.created_at,
    "updated_at": time.time(),
    "parent_adapter": None,
    "rank": config.rank,
    "target_spec": config.target_spec,
    "num_adapted_layers": len(model_state_keys),
}
registry._write_index(idx)
print("✓ Updated registry.json")

metrics = run.metrics
if metrics:
    print(
        "Recovered metrics:",
        [f"e{int(m['epoch'])+1}={m['loss']:.5f}" for m in metrics],
    )
print("✓ Done.")
