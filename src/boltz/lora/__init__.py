"""LoRA finetuning for the Boltz affinity stack.

Public surface:
    - LoRALinear: low-rank residual wrapper around nn.Linear.
    - apply_lora / remove_lora / merge_lora: injection utilities.
    - LoRAAdapter, LoRARegistry: persistence + provenance.
    - load_adapter_into_model: inference-time application.
"""

from boltz.lora.adapter import LoRAAdapter, LoRAConfig
from boltz.lora.apply import load_adapter_into_model
from boltz.lora.inject import apply_lora, merge_lora, remove_lora
from boltz.lora.layers import LoRALinear
from boltz.lora.registry import LoRARegistry, default_registry

__all__ = [
    "LoRAAdapter",
    "LoRAConfig",
    "LoRALinear",
    "LoRARegistry",
    "apply_lora",
    "default_registry",
    "load_adapter_into_model",
    "merge_lora",
    "remove_lora",
]
