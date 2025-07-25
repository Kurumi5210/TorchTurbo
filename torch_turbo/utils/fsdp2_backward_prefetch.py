from torch_turbo.utils.torch_utils import torch_version
from torch import nn
from typing import Any

if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
else:
    from torch.distributed._composable.fsdp._fsdp_state import FSDPState
    from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup

#   BackwardPrefetchContext is designed to switch prefetch logic between forward & backward.
#   During forward, prefetch next layer
#   During backward, prefetch previous layer
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