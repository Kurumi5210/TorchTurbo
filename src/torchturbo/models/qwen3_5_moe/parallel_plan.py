from typing import Any
import torch.nn.functional as F

from ...parallel import FSDPPlanSpec, FSDPParallelPlan
from ..modules.moe.moe_layer import MoELayer, MoEConfig, MoESubmodules
from ..modules.moe.module import ModuleSpec
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.config_converter import get_base_moe_config


def _is_qwen3_5_moe_module(module: Any) -> bool:
    return hasattr(module, "experts") and hasattr(module, "gate")


def _replace_qwen3_5_moe_only(
    original_module, hf_config, sp_info, ep_info, moe_submodules, **kwargs
):
    if not _is_qwen3_5_moe_module(original_module):
        return original_module

    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
    moe_config_kwargs = get_base_moe_config(
        hf_config=text_config,
        ep_size=ep_size,
        num_moe_experts=text_config.num_experts,
    )
    moe_config = MoEConfig(**moe_config_kwargs)

    moe_layer = MoELayer(
        config=moe_config,
        ep_group=ep_group,
        submodules=moe_submodules,
        use_fused_swiglu=False,
    )

    if hasattr(original_module, "shared_expert"):
        moe_layer.shared_expert = original_module.shared_expert
    if hasattr(original_module, "shared_expert_gate"):
        moe_layer.shared_expert_gate = original_module.shared_expert_gate

    return moe_layer


def get_parallel_plan() -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard
    from ...parallel import FSDPPlacementSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "model.language_model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "model.language_model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_qwen3_5_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "mtp.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "mtp.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_qwen3_5_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
        },
        sp_size=1,
        ep_size=8,
    )
