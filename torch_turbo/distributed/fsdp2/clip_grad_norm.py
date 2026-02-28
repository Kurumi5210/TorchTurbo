import torch
from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

from torch_turbo.distributed.comm import all_gather_tensor
from torch_turbo.distributed.parallel_state import (
    get_edp_mesh, 
    get_sharded_gradient_scaling_factor
)


def scale_moe_gradient(params):
    assert params[0].device_mesh == get_edp_mesh()
    gradient_scaling_factor = get_sharded_gradient_scaling_factor()
    grads = [p.grad for p in params]
    torch._foreach_mul_(grads, gradient_scaling_factor)


@torch.no_grad()
def clip_grad_norm(model, max_norm=1.0, norm_type=2.0):
    params = []
    grads = []
    moe_params = []
    moe_grads = []
    edp = get_edp_mesh()
    for param in model.parameters():
        # Skip parameters without gradients to ensure correctness.
        if param.grad is not None:
            if param.device_mesh == edp:
                moe_params.append(param)
                moe_grads.append(param.grad)
            else:
                params.append(param)
                grads.append(param.grad)

    scale_moe_gradient(moe_params)

    dp_total_norm = _get_total_norm(grads, norm_type=norm_type)
    edp_total_norm = _get_total_norm(moe_grads, norm_type=norm_type)
    dp_total_norm = dp_total_norm.full_tensor().unsqueeze(0)
    edp_total_norm = edp_total_norm.full_tensor().unsqueeze(0)
    edp_total_norm = all_gather_tensor(edp_total_norm, ep_name="ep")
    total_norm = torch.linalg.vector_norm(torch.cat([edp_total_norm, dp_total_norm]), norm_type)
    _clip_grads_with_norm_(params, max_norm, total_norm)
    _clip_grads_with_norm_(moe_params, max_norm, total_norm)

    return total_norm