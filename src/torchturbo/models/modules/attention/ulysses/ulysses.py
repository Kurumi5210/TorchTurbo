import inspect
import math
from functools import partial
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from torchturbo.runtime import get_device_runtime

from ...comm.comm import gather_from_sequence_parallel_region
from ..interface import attention_forward
from ..qk_clip import QKClipState, get_qk_clip_state
from .utils import (
    repeat_kv,
    unpad_tensor,
    pad_tensor,
    slice_input_tensor,
)
from torchturbo.logging import get_logger

logger = get_logger(__name__)


def _all_to_all(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    sp_group: Optional[dist.ProcessGroup],
    sp_size: int,
):
    input_list = [
        t.contiguous() for t in torch.tensor_split(local_input, sp_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(sp_size)]
    comm = dist.all_to_all(output_list, input_list, group=sp_group, async_op=True)
    comm.wait()
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _all_to_all_single(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    sp_group: Optional[dist.ProcessGroup],
    sp_size: int,
    a2a_stream: Optional[Any] = None,
) -> Tensor:
    """Exchange equal tensor splits with ``all_to_all_single``.

    The peer axis is packed in front for the collective, then moved to the
    requested gather dimension. This works for either direction of the
    Ulysses exchange; ``scatter_dim`` and ``gather_dim`` are not hard-coded.
    """
    assert local_input.size(0) == 1, (
        "all_to_all_single optimization requires batch size 1; callers must "
        "route other shapes through list all_to_all"
    )
    scatter_size = local_input.size(scatter_dim)
    if scatter_size % sp_size != 0:
        raise ValueError(
            f"scatter dimension size {scatter_size} must be divisible by "
            f"sequence parallel size {sp_size}"
        )

    split_shape = (
        *local_input.shape[:scatter_dim],
        sp_size,
        scatter_size // sp_size,
        *local_input.shape[scatter_dim + 1 :],
    )
    split_input = local_input.reshape(split_shape)
    peer_dim = scatter_dim
    packed_order = (
        peer_dim,
        *range(peer_dim),
        peer_dim + 1,
        *range(peer_dim + 2, local_input.ndim + 1),
    )
    # Allocate communication buffers on the caller stream. Only the collective
    # is placed on the side stream, so large tensors stay in one allocator pool.
    packed_input = split_input.permute(packed_order).contiguous()
    packed_output = torch.empty_like(packed_input)

    if a2a_stream is None:
        comm = dist.all_to_all_single(
            packed_output,
            packed_input,
            group=sp_group,
            async_op=True,
        )
        comm.wait()
    else:
        device_runtime = get_device_runtime()
        a2a_stream.wait_stream(device_runtime.current_stream())
        with device_runtime.stream(a2a_stream):
            comm = dist.all_to_all_single(
                packed_output,
                packed_input,
                group=sp_group,
                async_op=True,
            )
            comm.wait()
        packed_input.record_stream(a2a_stream)
        packed_output.record_stream(a2a_stream)

    output = packed_output.movedim(0, gather_dim).flatten(gather_dim, gather_dim + 1)
    return output.contiguous()


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    sp_group: Optional[dist.ProcessGroup],
    sp_size: int,
    a2a_stream: Optional[Any] = None,
) -> Tensor:
    """Select the optimized collective from the tensor shape.

    ``all_to_all_single`` is an implementation detail. Batch-one Ulysses
    tensors use it automatically; callers do not need to know which collective
    implements the exchange.
    """
    scatter_dim %= local_input.ndim
    gather_dim %= local_input.ndim
    if scatter_dim == gather_dim:
        raise ValueError("scatter_dim and gather_dim must be different")
    if local_input.size(scatter_dim) % sp_size != 0:
        raise ValueError(
            f"scatter dimension size {local_input.size(scatter_dim)} must be "
            f"divisible by sequence parallel size {sp_size}"
        )
    if local_input.size(0) == 1:
        return _all_to_all_single(
            local_input,
            scatter_dim,
            gather_dim,
            sp_group,
            sp_size,
            a2a_stream,
        )
    return _all_to_all(local_input, scatter_dim, gather_dim, sp_group, sp_size)


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        local_input: Tensor,
        sp_group: dist.ProcessGroup,
        sp_size: int,
        scatter_dim: int,
        gather_dim: int,
        a2a_stream: Optional[Any],
    ) -> Tensor:
        ctx.sp_group = sp_group
        ctx.sp_size = sp_size
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return all_to_all_tensor(
            local_input,
            scatter_dim,
            gather_dim,
            sp_group,
            sp_size,
            a2a_stream,
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_output: Tensor
    ) -> Tuple[Tensor, None, None, None, None, None]:
        return (
            all_to_all_tensor(
                grad_output[0],
                ctx.gather_dim,
                ctx.scatter_dim,
                ctx.sp_group,
                ctx.sp_size,
            ),
            None,
            None,
            None,
            None,
            None,
        )


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    sp_group: ProcessGroup,
    sp_size: int,
    unpadded_dim_size: int = 0,
    a2a_stream: Optional[Any] = None,
) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    if not sp_group:
        return x
    x = SeqAllToAll.apply(
        x,
        sp_group,
        sp_size,
        head_dim,
        seq_dim,
        a2a_stream,
    )
    if unpadded_dim_size and unpadded_dim_size % sp_size != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = unpad_tensor(x, seq_dim, padding_size)
    return x


def gather_heads_scatter_seq(
    x: Tensor, head_dim: int, seq_dim: int, sp_group: ProcessGroup, sp_size: int
) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    if not sp_group:
        return x
    dim_size = x.size(seq_dim)
    if dim_size % sp_size != 0:
        padding_size = sp_size - (dim_size % sp_size)
        x = pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(x, sp_group, sp_size, seq_dim, head_dim, None)


def ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask,
    query_length,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    sp_group: ProcessGroup,
    sp_size: int,
    qk_clip_state: QKClipState | None = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)
        cu_seq_lens_q/k, max_length_q/k: explicit global packed boundaries.
            Each rank holds the same copy, so no position_ids all_gather is
            needed to reconstruct them after the sequence gather.
        qk_clip_state (QKClipState, optional): Accumulator in global query-head order.

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    local_seq_length = query_states.size(1)
    if qk_clip_state is not None:
        num_heads = query_states.size(2)
        if num_heads != qk_clip_state.max_logits.numel() or num_heads % sp_size:
            raise ValueError("QK clipping requires a matching global head count divisible by SP size.")

    ########## AlltoAll for Ulysses ##########
    if sp_size > 1:
        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
        # For example:
        # - nheads_k=4, sp=8, repeats=2
        # - nheads_k=8, sp=8, repeats=1
        # - nheads_k=16, sp=8, repeats=1
        repeats = max(sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(
            query_states,
            seq_dim=1,
            head_dim=2,
            sp_group=sp_group,
            sp_size=sp_size,
        )
        key_states = gather_seq_scatter_heads(
            key_states,
            seq_dim=1,
            head_dim=2,
            sp_group=sp_group,
            sp_size=sp_size,
        )
        value_states = gather_seq_scatter_heads(
            value_states,
            seq_dim=1,
            head_dim=2,
            sp_group=sp_group,
            sp_size=sp_size,
        )

        # Explicit cu/max describe the full gathered sequence: drop the
        # rank-local position_ids so boundaries are not re-derived from the
        # local shard. Legacy callers without cu (e.g. kimi's position-based
        # path) still rebuild boundaries from the gathered positions.
        explicit_varlen = (
            kwargs.get("cu_seq_lens_q") is not None
            and kwargs.get("cu_seq_lens_k") is not None
        )
        if explicit_varlen:
            position_ids = None
        elif position_ids is not None and (
            qk_clip_state is None or position_ids.size(-1) == local_seq_length
        ):
            position_ids_list = [torch.empty_like(position_ids) for _ in range(sp_size)]
            torch.distributed.all_gather(position_ids_list, position_ids, group=sp_group)
            position_ids = torch.concat(position_ids_list, dim=-1)
        if (
            qk_clip_state is not None
            and attention_mask is not None
            and attention_mask.ndim == 2
            and attention_mask.size(-1) == local_seq_length
        ):
            masks = [torch.empty_like(attention_mask) for _ in range(sp_size)]
            dist.all_gather(masks, attention_mask.contiguous(), group=sp_group)
            attention_mask = torch.cat(masks, dim=-1)
        if query_length != query_states.size(1):
            assert query_length * sp_size == query_states.size(
                1
            ), "ulysses fa only pass 1/sp query here"
            query_length = query_states.size(1)

    result = attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        *args,
        position_ids=position_ids,
        return_max_logits=qk_clip_state is not None,
        **kwargs,
    )
    if qk_clip_state is not None:
        attn_output, head_max = result
        sp_rank = dist.get_rank(sp_group) if sp_size > 1 else 0
        qk_clip_state.record(head_max, head_offset=sp_rank * query_states.size(2))
    else:
        attn_output = result

    ########## AlltoAll for Ulysses ##########
    if sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(
            attn_output, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
        )

    return attn_output


TURBO_ULYSSES_ATTENTION = "turbo_ulysses"


def ulysses_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    sp_group: Optional[ProcessGroup] = None,
    sp_size: int = 1,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Ulysses SP attention as an HF attention-interface function.

    Follows the ``ALL_ATTENTION_FUNCTIONS`` contract: inputs use the
    [batch, heads, seq, head_dim] layout, outputs use [batch, seq, heads,
    head_dim] (HF's ``flash_attention_forward`` does not transpose its
    result back), the module supplies ``is_causal`` and dtype context, and
    varlen kwargs (``cu_seq_lens_q/k`` etc.) flow through ``**kwargs``. The
    body splices the DeepSpeed-Ulysses all-to-all pair around
    :func:`ulysses_flash_attention_forward`. Training modules that carry a
    QK-clip state (installed by the QK clipping controller) also record
    their per-head maximum logits through it.
    """
    from transformers.integrations.flash_attention import get_target_dtype

    seq_len = query_states.shape[2]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    target_dtype = get_target_dtype(query_states, module)
    is_causal = is_causal if is_causal is not None else module.is_causal

    # Q heads can be fewer than SP ranks (e.g. 16 heads with SP=32): repeat
    # them so the all-to-all has enough heads to scatter, then dedup on the
    # way back. K/V heads are repeated inside ulysses_flash_attention_forward.
    sp_repeat = 1
    if sp_size > 1:
        num_q_heads = query_states.size(2)
        sp_repeat = math.lcm(sp_size, num_q_heads) // num_q_heads
        if sp_repeat > 1:
            query_states = query_states.repeat_interleave(sp_repeat, dim=2)

    attn_output = ulysses_flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        target_dtype=target_dtype,
        sp_group=sp_group,
        sp_size=sp_size,
        qk_clip_state=get_qk_clip_state(module) if module.training else None,
        **kwargs,
    )

    if sp_repeat > 1:
        attn_output = attn_output[:, :, ::sp_repeat, :]
    return attn_output, None


def register_turbo_ulysses_attention(
    attn_config: Any,
    sp_group: Optional[ProcessGroup],
    sp_size: int,
) -> None:
    """Bind the ``turbo_ulysses`` attention interface to one SP group.

    Registers the SP-bound interface in ``ALL_ATTENTION_FUNCTIONS`` and
    switches ``attn_config._attn_implementation`` to it. Attention modules
    resolve the interface from the registry at forward time, so the switch
    takes effect on models that are already built. Subconfigs that keep
    their own implementation (e.g. VLM vision towers on ``flash_attention_2``)
    stay on the native flash path.

    The name is also mapped in HF's mask registry to the flash-attention
    mask function, so mask factories dispatch for it instead of taking the
    unrecognized-name early exit.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS[TURBO_ULYSSES_ATTENTION] = partial(
        ulysses_attention_forward, sp_group=sp_group, sp_size=sp_size
    )
    # HF checks the global mask registry before dispatching mask creation.
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        TURBO_ULYSSES_ATTENTION,
        ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"],
    )
    attn_config._attn_implementation = TURBO_ULYSSES_ATTENTION

    logger.info_rank0(
        "Registered %s attention interface for SP size=%d",
        TURBO_ULYSSES_ATTENTION,
        sp_size,
    )


def _forward_arg_index(forward, names: tuple[str, ...], skip_first: int):
    try:
        parameters = list(inspect.signature(forward).parameters.values())
    except (TypeError, ValueError):
        return None

    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    positional_index = 0
    for parameter in parameters:
        if parameter.kind in positional_kinds:
            if parameter.name in names:
                arg_index = positional_index - skip_first
                return arg_index if arg_index >= 0 else None
            positional_index += 1
    return None


def patch_kda_attention_forward_for_sp(
    attn_cls: type,
    sp_group: ProcessGroup,
    sp_size: int,
    sp_rank: int,
):
    """Patch KDA/DSA-style sequence-state attention for correctness under SP.

    These layers carry state along the sequence dimension, so local shard-only
    execution would break causality. This wrapper gathers the full sequence,
    runs the original forward, then scatters the output sequence back.
    """

    if sp_size <= 1 or sp_group is None:
        return

    origin_attr = "_torchturbo_origin_forward"
    origin_forward = getattr(attn_cls, origin_attr, None)
    if origin_forward is None:
        origin_forward = attn_cls.forward
        setattr(attn_cls, origin_attr, origin_forward)

    cache_arg_index = _forward_arg_index(
        origin_forward, ("past_key_values", "past_key_value"), skip_first=2
    )

    def sp_forward(self, hidden_states, *args, **kwargs):
        return kda_attention_sp_forward(
            self,
            origin_forward,
            hidden_states,
            args,
            kwargs,
            cache_arg_index,
            sp_group,
            sp_size,
            sp_rank,
        )

    attn_cls.forward = sp_forward


def kda_attention_sp_forward(
    module,
    origin_forward,
    hidden_states: Tensor,
    args,
    kwargs,
    cache_arg_index: Optional[int],
    sp_group: ProcessGroup,
    sp_size: int,
    sp_rank: int,
):
    past = kwargs.get("past_key_values", kwargs.get("past_key_value", None))
    if past is None and cache_arg_index is not None and len(args) > cache_arg_index:
        past = args[cache_arg_index]
    if past is not None:
        raise NotImplementedError(
            "Sequence-parallel KDA/DSA attention does not support KV cache."
        )

    local_seq_len = hidden_states.size(1)
    full_hidden_states = all_gather_sp_with_grad_reduce(
        hidden_states, dim=1, sp_group=sp_group, sp_size=sp_size
    )
    result = origin_forward(module, full_hidden_states, *args, **kwargs)
    is_tuple = isinstance(result, (tuple, list))
    if is_tuple:
        output, *rest = result
    else:
        output = result
    output = slice_input_tensor(
        output,
        dim=1,
        group=sp_group,
        rank=sp_rank,
        padding=False,
    )
    if output.size(1) != local_seq_len:
        output = output[:, :local_seq_len].contiguous()
    if is_tuple:
        return (output, *rest)
    return output


# copied from verl
def patch_vlm_for_ulysses_input_slicing(model_class: type):
    """
    Applies a monkey patch to the forward method of a given model class
    to enable Ulysses sequence parallelism input slicing.
    """

    def _create_ulysses_wrapped_decoder_forward(original_forward):
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            position_ids = kwargs.get("position_ids")
            visual_pos_masks = kwargs.get("visual_pos_masks")
            deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

            slice_now = (
                inputs_embeds is not None
                and current_ulysses_sp_size > 1
                and getattr(self, "_needs_initial_slice", True)
            )
            if slice_now:
                call_kwargs["inputs_embeds"] = slice_input_tensor(
                    inputs_embeds, dim=1, padding=False
                )
                call_kwargs["position_ids"] = slice_input_tensor(
                    position_ids, dim=-1, padding=False
                )
                # Also slice visual_pos_masks and deepstack_visual_embeds for Qwen3 VL models
                if visual_pos_masks is not None:
                    original_visual_mask = visual_pos_masks
                    sliced_visual_mask = slice_input_tensor(
                        visual_pos_masks, dim=1, padding=False
                    )
                    call_kwargs["visual_pos_masks"] = sliced_visual_mask

                    if deepstack_visual_embeds is not None:
                        sliced_embeds = []

                        num_visual_before = original_visual_mask.sum().item()
                        num_visual_in_shard = sliced_visual_mask.sum().item()

                        if num_visual_in_shard > 0 and num_visual_before > 0:
                            # Calculate which visual embeddings belong to this shard
                            # We need to find the offset of visual tokens in this shard
                            rank = get_ulysses_sequence_parallel_rank()
                            seq_len = original_visual_mask.shape[1]
                            local_seq_len = seq_len // current_ulysses_sp_size
                            start_idx = rank * local_seq_len
                            end_idx = start_idx + local_seq_len

                            # Get total visual tokens before and up to the end of the shard's sequence slice
                            # This correctly handles batches by summing across all samples
                            visual_start = (
                                original_visual_mask[:, :start_idx].sum().item()
                                if start_idx > 0
                                else 0
                            )
                            visual_end = original_visual_mask[:, :end_idx].sum().item()

                            # Slice each tensor in deepstack_visual_embeds
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[visual_start:visual_end])
                        else:
                            # No visual tokens in this shard, create empty tensors to maintain gradient flow
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[:0])
                        call_kwargs["deepstack_visual_embeds"] = sliced_embeds

                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    original_forward = model_class.forward
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward
    logger.info_rank0("Monkey patch %s.forward for Ulysses SP input slicing.", model_class.__name__)


def patch_forward_for_sp_input_slicing(
    model_class: type, sp_group: dist.ProcessGroup, sp_size: int
):
    if sp_size <= 1:
        return

    original_forward = model_class.forward
    sp_rank = dist.get_rank(sp_group)

    def _create_ulysses_wrapped_decoder_forward(original_forward):
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            position_ids = kwargs.get("position_ids")
            visual_pos_masks = kwargs.get("visual_pos_masks")
            deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = sp_size

            slice_now = (
                inputs_embeds is not None
                and current_ulysses_sp_size > 1
                and getattr(self, "_needs_initial_slice", True)
            )
            if slice_now:
                call_kwargs["inputs_embeds"] = slice_input_tensor(
                    inputs_embeds, dim=1, group=sp_group, rank=sp_rank, padding=False
                )
                # VLM batches without explicit positions (the model derives
                # mrope coordinates) stay None instead of crashing the slice.
                if position_ids is not None:
                    call_kwargs["position_ids"] = slice_input_tensor(
                        position_ids, dim=-1, group=sp_group, rank=sp_rank, padding=False
                    )
                # Also slice visual_pos_masks and deepstack_visual_embeds for Qwen3 VL models
                if visual_pos_masks is not None:
                    original_visual_mask = visual_pos_masks
                    sliced_visual_mask = slice_input_tensor(
                        visual_pos_masks,
                        dim=1,
                        group=sp_group,
                        rank=sp_rank,
                        padding=False,
                    )
                    call_kwargs["visual_pos_masks"] = sliced_visual_mask

                    if deepstack_visual_embeds is not None:
                        sliced_embeds = []

                        num_visual_before = original_visual_mask.sum().item()
                        num_visual_in_shard = sliced_visual_mask.sum().item()

                        if num_visual_in_shard > 0 and num_visual_before > 0:
                            # Calculate which visual embeddings belong to this shard
                            # We need to find the offset of visual tokens in this shard
                            rank = sp_rank
                            seq_len = original_visual_mask.shape[1]
                            local_seq_len = seq_len // current_ulysses_sp_size
                            start_idx = rank * local_seq_len
                            end_idx = start_idx + local_seq_len

                            # Get total visual tokens before and up to the end of the shard's sequence slice
                            # This correctly handles batches by summing across all samples
                            visual_start = (
                                original_visual_mask[:, :start_idx].sum().item()
                                if start_idx > 0
                                else 0
                            )
                            visual_end = original_visual_mask[:, :end_idx].sum().item()

                            # Slice each tensor in deepstack_visual_embeds
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[visual_start:visual_end])
                        else:
                            # No visual tokens in this shard, create empty tensors to maintain gradient flow
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[:0])
                        call_kwargs["deepstack_visual_embeds"] = sliced_embeds

                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward


def all_gather_sp_with_grad_reduce(
    tensor: Tensor, dim: int, sp_group: ProcessGroup, sp_size: int
) -> Tensor:
    """All-gather tensor and reduce gradients for replicated full-sequence use.

    This is for layers that must run on the full sequence on every SP rank and
    then slice their output back to the local shard. Each rank's local loss may
    still depend on hidden states from other sequence shards, so backward must
    sum gradient contributions for each original shard across all ranks.
    """

    if sp_size <= 1 or sp_group is None:
        return tensor
    if dim < 0:
        dim = tensor.dim() + dim
    tensor = tensor.movedim(dim, 0).contiguous()
    tensor = gather_from_sequence_parallel_region(
        tensor, tensor_parallel_output_grad=True, group=sp_group
    )
    return tensor.movedim(0, dim).contiguous()
