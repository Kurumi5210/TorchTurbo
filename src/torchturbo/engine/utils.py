"""Small dependency-light helpers shared by engine modules."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from itertools import chain
from typing import Any

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


def validate_signature(
    function: Callable[..., Any], /, **expected_defaults: Any
) -> None:
    """Require explicit keyword parameters with the expected defaults."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError) as exc:
        raise TypeError(f"cannot inspect {function!r}") from exc

    keyword_kinds = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    function_name = getattr(function, "__qualname__", repr(function))
    for name, expected_default in expected_defaults.items():
        parameter = parameters.get(name)
        if parameter is None or parameter.kind not in keyword_kinds:
            raise TypeError(
                f"{function_name} must explicitly declare keyword parameter "
                f"{name!r}; **kwargs alone is not sufficient"
            )
        if (
            type(parameter.default) is not type(expected_default)
            or parameter.default != expected_default
        ):
            raise TypeError(
                f"{function_name} parameter {name!r} must default to "
                f"{expected_default!r}"
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


def has_meta_tensor(model: nn.Module) -> list[str]:
    """Return the fully qualified names of parameters and buffers on meta."""
    named_tensors = chain(model.named_parameters(), model.named_buffers())
    return [name for name, tensor in named_tensors if _is_meta_tensor(tensor)]
