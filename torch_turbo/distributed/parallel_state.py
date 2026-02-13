import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torch_turbo.utils.device import get_device_name


# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
_WORLD_MESH = None
_EXPERT_MESH = None
_GRADIENT_SCALING_FACTOR = 1.0


def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    """Creates a ProcessGroup."""
    kwargs = {
        'ranks': ranks,
        'timeout': timeout,
        'backend': backend,
        'pg_options': pg_options,
        'use_local_synchronization': use_local_synchronization,
        'group_desc': group_desc,
    }
    return torch.distributed.new_group(**kwargs)


def initialize_model_parallel(
    expert_model_parallel_size: int = 1,
) -> None:
    """Initialize expert model parallel groups.

    Args:
        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.
    """
    # Get world size and rank.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Build the expert model parallel group
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
    for i in range(world_size // expert_model_parallel_size):
        ranks = list(range(i*expert_model_parallel_size, (i+1)*expert_model_parallel_size))
        group = create_group(
            ranks,
            pg_options=None,
            group_desc='EXPERT_MODEL_PARALLEL_GROUP',
        )
        if rank in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = group


def get_expert_model_parallel_group(check_initialized=True):
    """Get the expert-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_MODEL_PARALLEL_GROUP is not None
        ), 'expert model parallel group is not initialized'
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_model_parallel_world_size():
    """Return world size for the expert-model-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_expert_model_parallel_group())
    else:
        return 0


def get_expert_model_parallel_rank():
    """Return caller's rank in the expert-model-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_expert_model_parallel_group())
    else:
        return 0


_PARALLEL_STATE = None


class ParallelState:
    sp_size: int = 1
    ep_size: int = 1

    def sp_enabled(self) -> bool:
        return self.sp_size > 1
    
    def ep_enabled(self) -> bool:
        return self.ep_size > 1


def init_parallel_state(sp_size, ep_size):
    global _PARALLEL_STATE
    _PARALLEL_STATE = ParallelState(sp_size, ep_size)

def get_parallel_state():
    return _PARALLEL_STATE

# DeviceMesh related funcitons
def init_mesh(args):
    # TODO: change device name to a global variable to switch between cuda and npu
    world_size = args.world_size
    device_name = get_device_name()
    world_mesh = init_device_mesh(device_name, (world_size,))
    ep_mesh = init_device_mesh(device_name, (world_size // args.ep_size, args.ep_size), mesh_dim_names=("edp", "ep"))
    gradient_scaling_factor = (1 / args.ep_size) if args.average_in_collective else 1.0

    def is_same_group(process_group, device_mesh):
        mesh_group = device_mesh.get_group(0)
        mesh_ranks = dist.get_process_group_ranks(mesh_group)
        given_ranks = dist.get_process_group_ranks(process_group)
        return mesh_ranks == given_ranks

    # verify ep group is the same with ep_mesh
    megatron_ep_group = get_expert_model_parallel_group()
    assert is_same_group(megatron_ep_group, ep_mesh['ep'])

    global _WORLD_MESH
    global _EXPERT_MESH
    global _GRADIENT_SCALING_FACTOR
    _WORLD_MESH = world_mesh
    _WORLD_MESH = ep_mesh
    _GRADIENT_SCALING_FACTOR = gradient_scaling_factor

def get_world_mesh():
    return _WORLD_MESH

def get_expert_mesh():
    return _EXPERT_MESH

def get_edp_mesh():
    return _EXPERT_MESH['edp']

def get_sharded_gradient_scaling_factor():
    return _GRADIENT_SCALING_FACTOR