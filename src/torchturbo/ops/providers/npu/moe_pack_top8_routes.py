import torch

from torchturbo.runtime import TurboDevice
from ...base import OpImpl, OpImplKind
from ...registry import register_op


class _PackTop8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, ids, num_experts):
        from ._pack_top8 import pack

        ctx.save_for_backward(ids)
        return pack(weights, ids, num_experts)

    @staticmethod
    def backward(ctx, grad_dense, grad_mask, grad_ordered):
        (ids,) = ctx.saved_tensors
        return grad_dense.gather(1, ids), None, None


@register_op(
    op_name="moe_pack_top8_routes",
    impl_name="triton_npu_pack_top8_routes",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuPackTop8Routes(OpImpl):
    def available(self) -> bool:
        try:
            from triton.backends.ascend import compiler  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self, weights: torch.Tensor, ids: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _PackTop8.apply(weights, ids, num_experts)
