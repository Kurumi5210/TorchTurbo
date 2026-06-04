"""Utilities for normalizing parameter names when a model is wrapped by PEFT.

When a model is wrapped by ``PeftModel``, parameter names gain extra
prefixes and segments that don't exist in the base model checkpoint:

- ``base_model.model.`` prefix from the PeftModel / BaseTuner wrapper
- ``.base_layer.`` segment from LoRA's ``LoraLayer.base_layer``

For example::

    base_model.model.model.layers.1.mlp.experts.base_layer.weight1
    →  model.layers.1.mlp.experts.weight1

    base_model.model.model.layers.1.self_attn.q_proj.base_layer.weight
    →  model.layers.1.self_attn.q_proj.weight

These helper functions strip those PEFT-specific decorations so that
weight loading, saving, FSDP sharding, and parallel-plan matching all
use the original HF checkpoint parameter names regardless of whether
LoRA is active.
"""

from __future__ import annotations

# PeftModel wraps the original model as:
#   PeftModel.base_model  →  BaseTuner(.model = original_model)
# so every parameter name gets a "base_model.model." prefix.
_PEFT_PREFIX = "base_model.model."

# LoRA layers wrap the original module as:
#   LoraLayer.base_layer  →  the original nn.Linear / GroupedMLP
# injecting a ".base_layer." segment before the original leaf name.
_BASE_LAYER_SEGMENT = ".base_layer."

# LoRA adapter parameter name patterns created by PEFT.
_ADAPTER_KEYWORDS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_A2.",
    ".lora_B2.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
)


def strip_peft_prefix(name: str) -> str:
    """Strip PEFT-specific prefixes/segments from a parameter name.

    Removes:

    1. The ``base_model.model.`` prefix added by PeftModel.
    2. The ``.base_layer.`` segment added by LoRA's LoraLayer.

    This restores the original HuggingFace checkpoint name so that
    weight loading, saving, and FSDP plan matching work regardless of
    whether the model has LoRA adapters applied.

    Examples
    --------
    >>> strip_peft_prefix("model.layers.1.mlp.experts.weight1")
    'model.layers.1.mlp.experts.weight1'
    >>> strip_peft_prefix("base_model.model.model.layers.1.mlp.experts.base_layer.weight1")
    'model.layers.1.mlp.experts.weight1'
    >>> strip_peft_prefix("base_model.model.model.layers.1.self_attn.q_proj.base_layer.weight")
    'model.layers.1.self_attn.q_proj.weight'
    """
    if name.startswith(_PEFT_PREFIX):
        name = name[len(_PEFT_PREFIX):]
    name = name.replace(_BASE_LAYER_SEGMENT, ".")
    return name


def is_peft_adapter_param(name: str) -> bool:
    """Return True if *name* is a LoRA adapter parameter.

    LoRA adapter parameters (``lora_A``, ``lora_B``, ``lora_A2``,
    ``lora_B2``, etc.) are created by PEFT and do not exist in the
    base model checkpoint.  They should be skipped during weight loading
    and are excluded from base-model saving.

    Callers should apply :func:`strip_peft_prefix` **before** calling
    this function so the check works on the canonical name.

    Examples
    --------
    >>> is_peft_adapter_param("model.layers.1.self_attn.q_proj.lora_A.default.weight")
    True
    >>> is_peft_adapter_param("model.layers.1.mlp.experts.lora_A2.default.weight")
    True
    >>> is_peft_adapter_param("model.layers.1.mlp.experts.weight1")
    False
    """
    return any(kw in name for kw in _ADAPTER_KEYWORDS)