from torch import nn
from typing import Any

from torch_turbo.utils.torch_utils import torch_version
if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp._fully_shard import FSDPModule
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
else:
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule
    from torch.distributed._composable.fsdp._fsdp_state import FSDPState
    from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup


def set_moe_forward_prefetch_for_fsdp2_ep(fsdp2_model, layer_cls, expert_cls, model_cls, config=None):
    model_modules = []
    layer_modules = []
    expert_modules = []
    first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)

    for _, named_module in fsdp2_model.named_modules():
        if isinstance(named_module, FSDPModule) and isinstance(named_module, model_cls):
            model_modules.append(named_module)

        if isinstance(named_module, FSDPModule) and isinstance(named_module, layer_cls):
            layer_modules.append(named_module)

        if isinstance(named_module, FSDPModule) and isinstance(named_module, expert_cls):
            expert_modules.append(named_module)

    assert len(model_modules) == 1
    num_nextn_predict_layers = getattr(config, "num_nextn_predict_layers", 0)
    # Validate MTP layers in assertion
    assert len(layer_modules) + num_nextn_predict_layers == len(expert_modules) + first_k_dense_replace

    for i in range(len(layer_modules)):
        # layer prefetch layer
        # expert prefetch expert
        cur_layer = layer_modules[i]
        j = i - first_k_dense_replace
        if j >= 0:
            cur_expert = expert_modules[j]

        # deal with forward
        if i == 0:
            if j == 0:
                model_modules[0].set_modules_to_forward_prefetch([cur_layer, cur_expert])
            else:
                model_modules[0].set_modules_to_forward_prefetch([cur_layer])
        else:
            # layer_{i - 1} prefetch layer_i
            if j == 0:
                layer_modules[i - 1].set_modules_to_forward_prefetch([cur_expert, cur_layer])
            else:
                layer_modules[i - 1].set_modules_to_forward_prefetch([cur_layer])
            
            if j > 0:
                expert_modules[j - 1].set_modules_to_forward_prefetch([cur_expert])
        

        # deal with backward
        if i > 0:
            # layer_i prefetch layer_{i - 1}
            layer_backward_prefetch_states = [layer_modules[i - 1]._get_fsdp_state()]
            setattr(layer_modules[i], 'backward_prefetch_states', layer_backward_prefetch_states)

        if j > 0:
            # layer_i prefetch layer_{i - 1}
            expert_backward_prefetch_states = [expert_modules[j - 1]._get_fsdp_state()]
            setattr(expert_modules[j], 'backward_prefetch_states', expert_backward_prefetch_states)


class BackwardPrefetchContext:
    def __init__(self):
        self.in_backward = False

        def pre_forward_hook(state, module, args, kwargs):
            old_prefetch_list = None
            cur_prefetch_list = getattr(module, "backward_prefetch_states", [])
            if self.in_backward and cur_prefetch_list != []:
                old_prefetch_list = state._states_to_forward_prefetch
                state._states_to_forward_prefetch = cur_prefetch_list

            res = self.old_pre_forward(state, module, args, kwargs)

            if old_prefetch_list is not None:
                state._states_to_forward_prefetch = old_prefetch_list

            return res

        self.old_pre_forward = FSDPState._pre_forward
        FSDPState._pre_forward = pre_forward_hook


        def post_forward_hook(state, module: nn.Module, input: Any, output: Any):
            old_mesh_info = None
            cur_prefetch_list = getattr(module, "backward_prefetch_states", [])
            if self.in_backward and cur_prefetch_list != []:
                old_mesh_info = state._fsdp_param_group.post_forward_mesh_info
                state._fsdp_param_group.post_forward_mesh_info = None

            res = self.old_post_forward(state, module, input, output)

            if old_mesh_info != None:
                state._fsdp_param_group.post_forward_mesh_info = old_mesh_info

            return res

        self.old_post_forward = FSDPState._post_forward
        FSDPState._post_forward = post_forward_hook

        def backward_prefetch_hook(group):
            return

        self.old_backward_prefetch = FSDPParamGroup._backward_prefetch
        FSDPParamGroup._backward_prefetch = backward_prefetch_hook


    def __enter__(self):
        self.in_backward = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.in_backward = False

fsdp2_backward_prefetch_manager = BackwardPrefetchContext()