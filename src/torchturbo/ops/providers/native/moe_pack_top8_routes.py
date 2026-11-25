import torch

from ...base import OpImpl, OpImplKind
from ...moe import pack_routes
from ...registry import register_op


@register_op(
    op_name="moe_pack_top8_routes",
    impl_name="torch_moe_pack_top8_routes",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchPackTop8Routes(OpImpl):
    def __call__(
        self, weights: torch.Tensor, ids: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return pack_routes(weights, ids, num_experts)
