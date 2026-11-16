from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="rms_norm",
    impl_name="npu_rms_norm",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuRMSNorm(OpImpl):
    def available(self) -> bool:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor:
        from torch_npu import npu_rms_norm

        if hidden_states.numel() == 0:
            # Empty SP shards still need zero gradients for input and weight.
            return hidden_states * weight.to(hidden_states.dtype)

        return npu_rms_norm(hidden_states, weight, eps)[0]
