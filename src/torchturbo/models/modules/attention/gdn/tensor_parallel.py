"""TP-local GatedDeltaNet parameters for local convolution and norm kernels."""

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial


def local_gdn_parameters(
    module: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read live TP parameters inside the active unsharded FSDP forward.

    TP layouts keep conv1d.weight replicated in full [Q, K, V] order and shard
    A_log/dt_bias by value head, while the module metadata holds TP-local head
    counts and widths. FSDP restores each parameter's one-dimensional TP mesh
    before these local kernels. TP admission requires conv1d.bias=None.
    Modules without DTensor parameters return their parameters unchanged.
    """
    weight = module.conv1d.weight
    if not isinstance(weight, DTensor):
        return weight, module.A_log, module.dt_bias
    mesh = weight.device_mesh
    size, rank = mesh.size(), mesh.get_local_rank()
    widths = (module.key_dim * size, module.key_dim * size, module.value_dim * size)
    weight = weight.to_local(grad_placements=[Partial()])
    segments = weight.split(widths, dim=0)
    local_weight = torch.cat([segment.chunk(size, dim=0)[rank] for segment in segments])
    return local_weight, module.A_log.to_local(), module.dt_bias.to_local()


def gated_norm(
    module: nn.Module, core: torch.Tensor, z: torch.Tensor,
) -> torch.Tensor:
    """Call the selected norm inside the active unsharded FSDP forward."""
    if not isinstance(module.norm, nn.Module) or not isinstance(module.norm.weight, DTensor):
        return module.norm(core, z)
    weight = module.norm.weight.to_local(grad_placements=[Partial()])
    return torch.func.functional_call(module.norm, {"weight": weight}, (core, z))
