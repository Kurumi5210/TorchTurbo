import pytest
import torch

from torchturbo.ops.providers.native.moe_pack_top8_routes import TorchPackTop8Routes
from torchturbo.ops.providers.npu.moe_pack_top8_routes import NpuPackTop8Routes
from torchturbo.runtime import is_npu_available


@pytest.mark.parametrize("rows", [0, 1, 17, 512])
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_top8_zero_routes_and_gradients(rows: int, device: str) -> None:
    if device == "npu" and (
        not is_npu_available() or not NpuPackTop8Routes().available()
    ):
        pytest.skip("Requires Ascend Triton")
    ids = torch.tensor([511, 3, 63, 1, 32, 7, 12, 0], device=device).expand(rows, -1)
    weights = (
        torch.arange(8, device=device, dtype=torch.float32)
        .expand(rows, -1)
        .requires_grad_()
    )
    provider = NpuPackTop8Routes() if device == "npu" else TorchPackTop8Routes()
    dense, mask, ordered = provider(weights, ids, 512)
    torch.testing.assert_close(ordered, ids.sort(-1).values, rtol=0, atol=0)
    torch.testing.assert_close(mask.sum(1), ids.new_full((rows,), 8))
    assert mask.gather(1, ids).all()
    torch.testing.assert_close(
        dense, weights.new_zeros(rows, 512).scatter_(1, ids, weights)
    )
    upstream = torch.randn_like(dense)
    (grad,) = torch.autograd.grad(dense, weights, upstream)
    torch.testing.assert_close(grad, upstream.gather(1, ids), atol=0, rtol=0)
