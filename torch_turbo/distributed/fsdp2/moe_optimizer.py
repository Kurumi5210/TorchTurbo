import torch
from torch_turbo.distributed.parallel_state import get_edp_mesh


def create_fsdp2_moe_optimizer(model, optim_config):
    params = []
    moe_params = []
    edp = get_edp_mesh()
    for param in model.parameters():
        if param.device_mesh != edp:
            params.append(param)
        else:
            moe_params.append(param)

    optimizer_params = [{"params": params}, {"params": moe_params}]
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=optim_config.get("lr", 1e-4),
        betas=optim_config.get("betas", (0.9, 0.999)),
        weight_decay=optim_config.get("weight_decay", 1e-2),
    )

    return optimizer