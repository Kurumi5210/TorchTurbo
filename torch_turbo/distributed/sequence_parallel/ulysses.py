from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from .utils import (
    repeat_kv,
    unpad_tensor,
    pad_tensor,
)
from torch_turbo.distributed.parallel_state import get_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_world_size


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous() if ctx.async_op else grad_output[0]
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
            None,
            None,
        )


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup = None,
) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = unpad_tensor(x, seq_dim, padding_size)
    return x

def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup = None) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


def ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask,
    query_length,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    ########## AlltoAll for Ulysses ##########
    assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"

    # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
    # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
    # For example:
    # - nheads_k=4, sp=8, repeats=2
    # - nheads_k=8, sp=8, repeats=1
    # - nheads_k=16, sp=8, repeats=1
    repeats = max(ulysses_sp_size // key_states.size(2), 1)
    key_states = repeat_kv(key_states, repeats)
    value_states = repeat_kv(value_states, repeats)

    # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
    query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
    key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
    value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

    # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate
    # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.
    # https://github.com/huggingface/transformers/pull/33932

    # (bsz, seq_len/n) -> (bsz, seq_len)
    position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
    torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
    position_ids = torch.concat(position_ids_list, dim=-1)
    if query_length != query_states.size(1):
        assert query_length * ulysses_sp_size == query_states.size(1), "ulysses fa only pass 1/sp query here"
        query_length = query_states.size(1)

    # (bsz, seq_len, n_head/n, head_dim)
    attn_output = _flash_attention_forward(query_states, key_states, value_states, attention_mask, query_length, *args, position_ids=position_ids, **kwargs)

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output