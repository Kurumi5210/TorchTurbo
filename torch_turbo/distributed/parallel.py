from datetime import timedelta
from typing import Optional

import torch


# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None


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
    distributed_timeout_minutes: int = 30,
    nccl_communicator_config_path: Optional[str] = None,
) -> None:
    """Initialize expert model parallel groups.

    Args:
        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes, for operations executed against distributed
            process groups.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
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