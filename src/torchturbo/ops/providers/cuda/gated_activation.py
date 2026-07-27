from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


_SUPPORTED_ACTIVATIONS = (F.silu, F.gelu)


def _torch_gated_activation(
    x: torch.Tensor,
    *,
    activation_func: Callable[[torch.Tensor], torch.Tensor],
    limit: float = 0.0,
) -> torch.Tensor:
    if activation_func not in _SUPPORTED_ACTIVATIONS:
        raise ValueError("gated_activation only supports F.silu and F.gelu.")

    limit = float(limit)
    if limit < 0:
        raise ValueError("limit must be non-negative.")

    gate, up = torch.chunk(x, 2, dim=-1)
    if limit > 0:
        up = torch.clamp(up.float(), min=-limit, max=limit)
        gate = torch.clamp(gate.float(), max=limit)
    return (activation_func(gate) * up).to(x.dtype)


@register_op(
    op_name="gated_activation",
    impl_name="torch_compile_gated_activation",
    kind=OpImplKind.CUSTOM,
    priority=50,
    devices=(TurboDevice.CUDA,),
)
class TorchCompileGatedActivation(OpImpl):
    def __init__(self) -> None:
        self._compiled: dict[tuple[Callable, float], Any] = {}

    def available(self) -> bool:
        return hasattr(torch, "compile")

    def __call__(
        self,
        x: torch.Tensor,
        *,
        activation_func: Callable[[torch.Tensor], torch.Tensor],
        limit: float = 0.0,
    ) -> torch.Tensor:
        key = (activation_func, float(limit))
        compiled = self._compiled.get(key)
        if compiled is None:

            def gated_activation_impl(tensor: torch.Tensor) -> torch.Tensor:
                return _torch_gated_activation(
                    tensor,
                    activation_func=activation_func,
                    limit=limit,
                )

            compiled = torch.compile(gated_activation_impl)
            self._compiled[key] = compiled
        return compiled(x)
