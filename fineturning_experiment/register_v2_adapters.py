#!/usr/bin/env python3
"""Register v2_rank_64 adapters in the local registry.json"""

import json
import time
from pathlib import Path

adapter_dir = Path(__file__).parent / "adapters"
registry_path = adapter_dir / "registry.json"

# Read current registry
with open(registry_path) as f:
    registry = json.load(f)

print("Current registry entries:", list(registry.keys()))

# Import torch to count tensors
import torch

# Add drd4_v2_rank_64
drd4_path = adapter_dir / "drd4_v2_rank_64"
if drd4_path.exists() and (drd4_path / "meta.json").exists():
    # Get metadata from meta.json
    meta = json.loads((drd4_path / "meta.json").read_text())
    # Count LoRA tensors
    ckpt = torch.load(drd4_path / "adapter.pt", map_location="cpu", weights_only=False)
    num_tensors = len(ckpt.get("state_dict", {})) if isinstance(ckpt, dict) and "state_dict" in ckpt else 0
    
    registry["drd4_v2_rank_64"] = {
        "path": str(drd4_path.resolve()),
        "created_at": meta["created_at"],
        "updated_at": time.time(),
        "parent_adapter": None,
        "rank": meta["config"]["rank"],
        "target_spec": meta["config"]["target_spec"],
        "num_adapted_layers": num_tensors,
    }
    print(f"✓ Added drd4_v2_rank_64 ({num_tensors} tensors)")

# Add 5ht2a_v2_rank_64
ht2a_path = adapter_dir / "5ht2a_v2_rank_64"
if ht2a_path.exists() and (ht2a_path / "meta.json").exists():
    meta = json.loads((ht2a_path / "meta.json").read_text())
    ckpt = torch.load(ht2a_path / "adapter.pt", map_location="cpu", weights_only=False)
    num_tensors = len(ckpt.get("state_dict", {})) if isinstance(ckpt, dict) and "state_dict" in ckpt else 0
    
    registry["5ht2a_v2_rank_64"] = {
        "path": str(ht2a_path.resolve()),
        "created_at": meta["created_at"],
        "updated_at": time.time(),
        "parent_adapter": None,
        "rank": meta["config"]["rank"],
        "target_spec": meta["config"]["target_spec"],
        "num_adapted_layers": num_tensors,
    }
    print(f"✓ Added 5ht2a_v2_rank_64 ({num_tensors} tensors)")

# Write updated registry
with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

print("\nUpdated registry entries:", list(registry.keys()))
print("✓ Registry updated successfully.")
