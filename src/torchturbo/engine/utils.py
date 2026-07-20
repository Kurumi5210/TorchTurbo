"""Small dependency-light helpers shared by engine modules."""

from __future__ import annotations

import torch
import torch.nn as nn

_PEFT_PREFIX = "base_model.model."
_BASE_LAYER_SEGMENT = ".base_layer."
_ADAPTER_KEYWORDS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_A2.",
    ".lora_B2.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
)


def strip_peft_prefix(name: str) -> str:
    if name.startswith(_PEFT_PREFIX):
        name = name[len(_PEFT_PREFIX):]
    return name.replace(_BASE_LAYER_SEGMENT, ".")


def is_peft_adapter_param(name: str) -> bool:
    return any(keyword in name for keyword in _ADAPTER_KEYWORDS)


def _is_meta_tensor(tensor: torch.Tensor) -> bool:
    if getattr(tensor, "is_meta", False):
        return True

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None

    if DTensor is not None and isinstance(tensor, DTensor):
        return tensor.to_local().is_meta
    return False


def has_meta_tensor(model: nn.Module) -> bool:
    """Return True if any parameter or persistent buffer is still on meta."""
    for _, param in model.named_parameters():
        if _is_meta_tensor(param):
            return True

    for module in model.modules():
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        for buf_name, buf in module.named_buffers(recurse=False):
            if buf is None or buf_name in non_persistent:
                continue
            if _is_meta_tensor(buf):
                return True

    return False
