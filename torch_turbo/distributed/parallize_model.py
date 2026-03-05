import re
import sys
import torch
from typing import Optional

from torch_turbo.utils.torch_utils import torch_version
if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp import fully_shard
else:
    from torch.distributed._composable.fsdp import fully_shard

from torch_turbo.distributed.fsdp2.moe_fwd_bwd_prefetch import set_moe_forward_prefetch_for_fsdp2_ep
from torch_turbo.distributed.moe.moe_layer import MoELayer
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
    register_model_buffer
)
from torch_turbo.models.weight_loader import DistParamLoader
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


def get_sp_modules(model, apply_modules):
    parallel_modules = []
    for target_name in apply_modules:
        for name, module in model.named_modules():
            if match_module(target_name, name):
                parallel_modules.append(module)
    return parallel_modules


def get_fully_shard_modules(model, apply_modules):
    shard_modules = dict()
    for target_name, shard_strategy in apply_modules.items():
        for name, module in model.named_modules():
            if match_module(target_name, name):
                shard_modules[module] = shard_strategy
    return shard_modules


def apply_sp(model, sp_plan):
    # qwen: self_attn
    # bailing: attention
    module = sys.modules[model.__module__]

    sp_modules = get_sp_modules(model, sp_plan.apply_modules)
    print(f"apply sp modules: {sp_modules}")
    for module in sp_modules:
        if hasattr(module, "_flash_attention_forward"):
            module._flash_attention_forward = ulysses_flash_attention_forward
            print(f"Monkey patched ulysses_flash_attention_forward in {module}!")
        else:
            from transformers.integrations import flash_attention
            flash_attention._flash_attention_forward = ulysses_flash_attention_forward
            print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")
            break


def apply_ep(model, moe_config, ep_plan):
    # apply moe_layer
    if hasattr(model.config, "first_k_dense_replace"):
        moe_layer_idx = model.config.first_k_dense_replace
    else:
        moe_layer_idx = 0

    for i, layer in enumerate(model.model.layers):
        if i < moe_layer_idx:
            continue
        
        # initilize moe_layer
        moe_layer = MoELayer(moe_config)
        cur_module = layer.get_submodule(ep_plan.apply_modules[0])
        if hasattr(cur_module, "shared_experts"):
            moe_layer.shared_experts = cur_module.shared_experts

        layer.register_module(ep_plan.apply_modules[0], moe_layer)


def apply_fsdp(model, hf_config, fsdp_config, fsdp_kwargs):
    fsdp_plan = get_fully_shard_modules(model, fsdp_config.apply_modules)
    print(f"apply fsdp sharding plan: {fsdp_plan}")
    for module, shard_strategy in fsdp_plan.items():
        if shard_strategy == 'edp':
            print(f"apply_fsdp edp: {module}")
            fully_shard(module, mesh=get_edp_mesh(), **fsdp_kwargs)
            setattr(module, "expert_forward_prefetch", True)
        elif shard_strategy == 'dp':
            print(f"apply_fsdp dp: {module}")
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
    set_moe_forward_prefetch_for_fsdp2_ep(model, outer_cls, inter_cls, outermost_cls, hf_config)


def apply_parallel_model(
    hf_config, 
    parallel_state, 
    fsdp_kwargs, 
    device, 
    torch_dtype=torch.bfloat16
):
    # used for DEBUG
    # hf_config.num_hidden_layers = 2
    
    # get parallel config
    parallel_plan = get_parallel_plan(hf_config)
    moe_config = parallel_plan.moe_config

    # create parallel model on meta device
    with torch.device("meta"):
        model = get_model_cls(hf_config)
        # apply sp modules
        if parallel_state.sp_enabled():
            apply_sp(model, parallel_plan.sp_plan)
        # apply ep modules
        if parallel_state.ep_enabled():
            apply_ep(model, moe_config, parallel_plan.ep_plan)
    
    model = model.to(dtype=torch_dtype)

    # apply fully shard modules
    apply_fsdp(model, hf_config, parallel_plan.fsdp_plan, fsdp_kwargs)
    # create empty tensor for model params on target device
    model.to_empty(device=device)
    
    return model


def build_fsdp_turbo_model(args, fsdp_kwargs, device):
    init_mesh(args)
    init_parallel_state(args.sp_size, args.ep_size)
    parallel_state = get_parallel_state()
    # get model config
    hf_config = get_model_config(args)

    # create parallized fsdp model
    model = apply_parallel_model(hf_config, parallel_state, fsdp_kwargs, device)

    # patch model buffer
    register_model_buffer(hf_config, model, device)
    
    # load distributed weight
    weight_loader = DistParamLoader(model, hf_config, device, args)
    weight_loader.load_weights()

    return model