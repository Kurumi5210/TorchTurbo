import re
import torch

from torch_turbo.utils.torch_utils import torch_version
if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp import fully_shard
else:
    from torch.distributed._composable.fsdp import fully_shard

from torch_turbo.distributed.fsdp.moe_fwd_bwd_prefetch import set_moe_forward_prefetch_for_fsdp2_ep
from torch_turbo.distributed.moe.moe_layer import MoESubmodules, MoELayer
from torch_turbo.distributed.moe.experts import GroupedMLP
from torch_turbo.distributed.sequence_parallel.ulysses import ulysses_flash_attention_forward
from torch_turbo.distributed.parallel_state import (
    init_mesh,
    init_parallel_state,
    get_parallel_state,
    get_edp_mesh,
    get_world_mesh,
)
from torch_turbo.models.registry import (
    get_model_config, 
    get_model_cls,
    get_parallel_plan,
    patch_model_buffer
)
from torch_turbo.models.weight_loader import DistParamLoader
from torch_turbo.transformer.module import ModuleSpec
from torch_turbo.utils.device import is_npu_available

if is_npu_available:
    from torch_npu.contrib import transfer_to_npu as _


def match_module(fqn_pattern: str, fqn: str, prefix: str = None):
    assert isinstance(fqn_pattern, str), f"fqn_pattern must be a str, got {type(fqn_pattern)}"
    assert isinstance(fqn, str), f"fqn must be a str, got {type(fqn)}"

    if prefix:
        fqn_pattern = [".".join([prefix, pattern]) for pattern in fqn_pattern]

    regex_str = re.escape(fqn_pattern).replace(r"\*", r".*")
    regex_str = f"^{regex_str}$"
    regex = re.compile(regex_str)

    match = regex.match(fqn)

    return match


def get_apply_modules(model, apply_modules):
    parallel_modules = []
    layers = model.model.layers
    for target_module in apply_modules:
        for name, module in layers.named_modules():
            if match_module(target_module, name):
                parallel_modules.append(module)
    return parallel_modules


def get_fully_shard_plan(model, apply_modules):
    shard_modules = dict()
    layers = model.model.layers
    for target_module, shard_strategy in apply_modules:
        for name, module in layers.named_modules():
            if target_module == name:
                shard_modules[module] = shard_strategy
    return shard_modules


def apply_sp(model, sp_config):
    # qwen: self_attn
    # bailing: attention
    sp_modules = get_apply_modules(model, sp_config.apply_modules)
    for module in sp_modules:
        module._flash_attention_forward = ulysses_flash_attention_forward


def apply_ep(model, moe_config, ep_config):
    # initilize moe_layer
    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    moe_layer = MoELayer(moe_config, moe_submodules)
    # apply moe_layer
    if hasattr(model.config, "first_k_dense_replace"):
        moe_layer_idx = model.config.first_k_dense_replace
    else:
        moe_layer_idx = 0
    ep_modules = get_apply_modules(model, ep_config.apply_modules)
    for i in range(moe_layer_idx, len(ep_modules)):
        ep_module = ep_modules[i]
        if hasattr(ep_module, "shared_experts"):
            moe_layer.shared_experts = ep_module.shared_experts
        ep_module = moe_layer


def apply_fsdp(model, fsdp_config, fsdp_kwargs):
    fsdp_plan = get_fully_shard_plan(model, fsdp_config.apply_modules)
    for module, shard_strategy in fsdp_plan.items():
        if shard_strategy == 'edp':
            fully_shard(module, mesh=get_edp_mesh(), **fsdp_kwargs)
        elif shard_strategy == 'dp':
            fully_shard(module, mesh=get_world_mesh(), **fsdp_kwargs)

    for layer in model.model.layers:
        fully_shard(layer, mesh = get_world_mesh(), **fsdp_kwargs)
        setattr(layer, "moe_forward_prefetch", True)

    fully_shard(model, mesh = get_world_mesh(), **fsdp_kwargs)
    outer_cls = model.model.layers[0].__class__
    if hasattr(model.config, "first_k_dense_replace"):
        inter_cls = model.model.layers[model.config.first_k_dense_replace].mlp.experts.__class__
    else:
        inter_cls = model.model.layers[0].mlp.experts.__class__
    outermost_cls = model.__class__
    set_moe_forward_prefetch_for_fsdp2_ep(model, outer_cls, inter_cls, outermost_cls, model.config)


def build_fsdp_turbo_model(args, fsdp_kwargs, device):
    init_mesh(args)
    init_parallel_state(args.sp_size, args.ep_size)
    parallel_state = get_parallel_state()

    config = get_model_config(args)
    # DEBUG
    # config.num_hidden_layers = 1
    # TODO: is device meta necessary?
    with torch.device("meta"):
        model = get_model_cls(config)
        # get parallel config
        parallel_plan = get_parallel_plan(config)
        # apply sp modules
        if parallel_state.sp_enabled():
            apply_sp(model, parallel_plan.sp_config)
        # apply ep modules
        if parallel_state.ep_enabled():
            apply_ep(model, parallel_plan.moe_config, parallel_plan.ep_config)
    # TODO: why to bf16?
    model = model.to(dtype=torch.bfloat16)

    # apply fully shard modules
    apply_fsdp(model, parallel_plan.fsdp_config, fsdp_kwargs)
    # TODO: why to empty? need to register buffer after empty
    model.to_empty(device=device)

    # patch model buffer
    patch_model_buffer(config, model, device)
    
    # load distributed weight
    weight_loader = DistParamLoader(args, config, model, device)
    weight_loader.load_weights()
    return model