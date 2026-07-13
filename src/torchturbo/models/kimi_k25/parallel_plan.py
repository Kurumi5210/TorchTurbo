from sys import modules
from typing import Any
from functools import partial
import torch.distributed as dist

from ...parallel import FSDPPlanSpec, FSDPParallelPlan
from ..modules.moe.moe_layer import MoELayer, MoEConfig, MoESubmodules
from ..modules.moe.module import ModuleSpec
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.config_converter import get_base_moe_config
from ..modules.attention import ulysses


def _is_kimi_moe_module(module: Any) -> bool:
    return hasattr(module, "experts") and hasattr(module, "gate")


def _replace_kimi_moe_only(
    original_module, hf_config, sp_info, ep_info, moe_submodules, **kwargs
):
    if not _is_kimi_moe_module(original_module):
        return original_module

    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    language_config = hf_config.text_config
    kimi_k25_config_kwargs = get_base_moe_config(
        hf_config=language_config,
        ep_size=ep_size,
        num_moe_experts=language_config.n_routed_experts,
        moe_router_num_groups=language_config.n_group,
        moe_router_group_topk=language_config.topk_group,
        moe_router_topk_scaling_factor=language_config.routed_scaling_factor,
        moe_router_score_function=language_config.scoring_func,
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
    )
    moe_config = MoEConfig(**kimi_k25_config_kwargs)

    moe_layer = MoELayer(
        config=moe_config,
        ep_group=ep_group,
        submodules=moe_submodules,
        use_fused_swiglu=False,
    )

    if hasattr(original_module, "shared_experts"):
        moe_layer.shared_experts = original_module.shared_experts

    return moe_layer


def _replace_self_attn(original_module, hf_config, sp_info, ep_info, **kwargs):
    if hasattr(original_module, "_flash_attention_forward"):
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        patched_forward = partial(
            ulysses.ulysses_flash_attention_forward, sp_group=sp_group, sp_size=sp_size
        )
        original_module._flash_attention_forward = patched_forward
    return original_module


def get_parallel_plan() -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard
    from ...parallel import FSDPPlacementSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "language_model.model.layers.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "language_model.model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "language_model.model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_kimi_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "language_model.model.layers.*.self_attn": FSDPPlanSpec(
                replace_module=_replace_self_attn,
            ),
            # "vision_tower": FSDPPlanSpec(
            #     no_shard=True,
            # ),
            "language_model.model.embed_tokens": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
            "language_model.lm_head": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
        },
        sp_size=2,
        ep_size=4,
    )
