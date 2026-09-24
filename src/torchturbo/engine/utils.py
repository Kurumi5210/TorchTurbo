"""Small dependency-light helpers shared by engine modules."""

from __future__ import annotations

import inspect
import re
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


def match_fqn(pattern: str, fqn: str) -> bool:
    """Return whether an FQN matches a pattern.

    The pattern must match the entire FQN. ``*`` matches one or more characters
    other than ``.``, so it stays inside a single segment. ``**`` can also
    match ``.``, so it can span multiple segments.

    Examples:
        ``*`` stays inside one segment, and may match part of it:

        >>> match_fqn("layers.*.weight", "layers.0.weight")
        True

        ``**`` may match across multiple segments:

        >>> match_fqn("layers.**.weight", "layers.0.attention.q_proj.weight")
        True
    """
    regex = re.escape(pattern)
    regex = regex.replace(r"\*\*", r".*")
    regex = regex.replace(r"\*", r"[^.]+")
    return bool(re.match(f"^{regex}$", fqn))


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


def assert_packed_inputs(inputs: dict[str, Any]) -> None:
    """Check the packed SFT input contract on CPU before device transfer.

    A packed row has a valid prefix and an optional masked tail. Position IDs
    retain model-specific coordinates, with explicit self-attention boundaries.
    """
    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Packed input_ids must have shape [1, sequence_length].")
    if input_ids.device.type != "cpu":
        raise ValueError("Packed inputs must be checked on the CPU before device transfer.")
    if inputs.get("attention_mask") is not None:
        raise ValueError("Packed inputs must provide cu/max without attention_mask.")

    seq_len = input_ids.shape[-1]
    position_ids = inputs["position_ids"]
    if position_ids.device.type != "cpu":
        raise ValueError("Packed position_ids must be checked on the CPU.")
    if position_ids.ndim not in (2, 3) or position_ids.shape[-2:] != (1, seq_len):
        raise ValueError("Packed position_ids must align with the single input row.")

    cu_q = inputs["cu_seq_lens_q"]
    cu_k = inputs["cu_seq_lens_k"]
    for cu in (cu_q, cu_k):
        if cu.device.type != "cpu" or cu.dtype != torch.int32:
            raise ValueError("Packed cu_seq_lens_q/k must be CPU int32 tensors.")
        if cu.ndim != 1 or cu.numel() < 2:
            raise ValueError("Packed cu_seq_lens_q/k must contain sample boundaries.")
    if not torch.equal(cu_q, cu_k):
        raise ValueError("Packed SFT requires identical query and key boundaries.")
    lengths = cu_q.diff()
    if cu_q[0] != 0 or torch.any(lengths <= 0) or cu_q[-1] > seq_len:
        raise ValueError("Packed cu must start at zero and increase within the input length.")
    max_length = int(lengths.max())
    for key in ("max_length_q", "max_length_k"):
        if not isinstance(inputs[key], int) or inputs[key] != max_length:
            raise ValueError(f"{key} must be the longest packed sample length as an int.")

    text_positions = inputs.get("text_position_ids")
    if text_positions is None and position_ids.ndim == 2:
        text_positions = position_ids
    if text_positions is not None:
        if text_positions.device.type != "cpu" or text_positions.shape != (1, seq_len):
            raise ValueError("Packed text position IDs must align with the CPU input row.")
        valid_positions = text_positions[0, : int(cu_q[-1])]
        starts = torch.nonzero(valid_positions == 0, as_tuple=True)[0]
        if not torch.equal(starts, cu_q[:-1].long()):
            raise ValueError("Packed text position resets must match cu sample boundaries.")

    labels = inputs.get("labels")
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("Packed labels must align with input_ids.")


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
