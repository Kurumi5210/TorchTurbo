import functools
from typing_extensions import override

import torch


from transformers import AutoModelForImageTextToText

from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from torchturbo.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

from torchturbo.runtime import get_device_runtime

from ...engine.config import ParameterGroup
from ..base import TurboVLMModel
from ..registry import register_model
from ..weight_converter import FusedExpertWeightConverter


class Qwen3_5MoeWeightConverter(FusedExpertWeightConverter):

    def __init__(self, model, parallel_context):
        super().__init__(model, parallel_context)
        config = self.config
        if hasattr(config, "text_config"):
            self.config = config.text_config
            self.model_type = self.config.model_type
            self.dtype = self.get_save_dtype(self.config)

    def get_hf_alog_name(self):
        return "A_log"

    def get_hf_linear_attn_norm_name(self):
        return "linear_attn.norm"

    @override
    def get_default_hf_dtype(self, param_name):
        if self.get_hf_alog_name() in param_name:
            return torch.float32
        if self.get_hf_linear_attn_norm_name() in param_name:
            return torch.float32
        return self.dtype


@register_model(
    "qwen3_5_moe",
    config_cls=Qwen3_5MoeConfig,
    auto_class=AutoModelForImageTextToText,
)
class TurboQwen3_5MoeForConditionalGeneration(
    TurboVLMModel, Qwen3_5MoeForConditionalGeneration
):

    config_class = Qwen3_5MoeConfig
    base_model_prefix = "model"
    multi_modal_projector_module_names = ("model.visual.merger",)
    language_model_module_names = ("model.language_model",)

    @property
    def visual(self):
        # swift's qwen _post_encode (forward pre-hook on this outer model) calls
        # ``model.visual``; the tower lives on the inner Qwen3_5MoeModel.
        return self.model.visual

    def get_norm(self):
        # Final text-model RMSNorm lives on the inner language model (nested
        # below base_model); the base default ``<base_model>.norm`` does not find
        # it. Expose it so the FSDP2 tail optimizer merges it into the lm_head
        # resident unit instead of the auto root unit (the root unit's lone norm
        # breaks CPU offload).
        return self.model.language_model.norm

    def get_parameter_groups(self):
        groups = super().get_parameter_groups()
        groups[ParameterGroup.VISION_TOWER] = (
            (name, param)
            for name, param in self.model.visual.named_parameters(prefix="model.visual")
            if name != "model.visual.merger"
            and not name.startswith("model.visual.merger.")
        )
        return groups

    def __init__(self, config: Qwen3_5MoeConfig) -> None:
        if hasattr(config, "vision_config"):
            config.vision_config._attn_implementation = "flash_attention_2"
        super().__init__(config)
        self._post_init_patches()

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def _post_init_patches(self):
        import torch
        import functools
        import itertools

        # Wrap get_rope_index:
        # 1. auto-fill mm_token_type_ids when not provided (the open-source LF
        #    collator never passes this arg);
        # 2. support LLaMA-Factory neat packing. The upstream get_rope_index
        #    collapses the attention_mask with .bool(), which merges packed
        #    sub-sequences into one contiguous run and never resets positions.
        #    Under neat packing the mask carries incrementing integers
        #    (1, 2, 3, ... per packed sub-sequence, 0 = pad), so we detect that
        #    (max > 1) and run a packing-aware variant here, keeping
        #    modeling_qwen3_5_moe.py unchanged (upstream-faithful).
        original_get_rope_index = self.model.get_rope_index
        get_vision_position_ids = self.model.get_vision_position_ids
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = getattr(self.config, "image_token_id", None)
        video_token_id = getattr(self.config, "video_token_id", None)

        def _build_mm_token_type_ids(input_ids):
            mm_token_type_ids = torch.zeros_like(input_ids)
            if image_token_id is not None:
                mm_token_type_ids[input_ids == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[input_ids == video_token_id] = 2
            return mm_token_type_ids

        @functools.wraps(original_get_rope_index)
        def _wrapped_get_rope_index(
            input_ids,
            mm_token_type_ids=None,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
            **kwargs,
        ):
            # ``mm_token_type_ids`` is a *positional* parameter of the upstream
            # ``get_rope_index`` (right after ``input_ids``), so a caller may
            # pass it (and ``input_ids``) positionally OR by keyword — and the
            # open-source LF collator never passes ``mm_token_type_ids`` at all.
            # Bind these to real parameters here instead of ``*args`` /
            # ``**kwargs``: reading them from ``kwargs`` breaks when they're
            # passed positionally (which the signature invites — caller 2 in
            # modeling_qwen3_5_moe.py passes ``input_ids`` positionally, so the
            # old ``kwargs["input_ids"]`` would KeyError).
            if mm_token_type_ids is None:
                mm_token_type_ids = _build_mm_token_type_ids(input_ids)

            # Packing-aware path only when the mask actually carries packing
            # labels (neat packing): values > 1 distinguish packed sub-sequences.
            # A plain 0/1 mask (or no mask) falls through to the upstream impl,
            # so non-packed training and inference are byte-identical to before.
            if attention_mask is None or attention_mask.numel() == 0:
                return original_get_rope_index(
                    input_ids,
                    mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    **kwargs,
                )

            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            grid_iters = {
                1: iter(image_grid_thw) if image_grid_thw is not None else None,
                2: iter(video_grid_thw) if video_grid_thw is not None else None,
            }

            mrope_position_deltas = []
            for batch_idx in range(input_ids.shape[0]):
                row_mask = attention_mask[batch_idx]
                input_token_type = mm_token_type_ids[batch_idx]

                # Valid (non-padding) tokens and their packed-block labels.
                nonzero_idx = row_mask != 0
                token_types = input_token_type[nonzero_idx].tolist()
                block_labels = row_mask[nonzero_idx].tolist()

                # Group consecutive tokens by (block_label, modality): a change in
                # EITHER the packed-block label OR the modality starts a new group,
                # so two images in adjacent packed sequences consume two separate
                # grid_thw entries and each block restarts positions from 0.
                input_type_group = []
                for (block_label, modality_type), group in itertools.groupby(
                    range(len(token_types)),
                    key=lambda i: (block_labels[i], token_types[i]),
                ):
                    grp = list(group)
                    input_type_group.append((modality_type, grp[0], grp[-1] + 1, block_label))

                current_pos = 0
                prev_block = None
                llm_pos_ids_list = []
                for modality_type, start_idx, end_idx, block_label in input_type_group:
                    # Reset position at each packed-sequence boundary.
                    if block_label != prev_block:
                        current_pos = 0
                        prev_block = block_label
                    # text == 0
                    if modality_type == 0:
                        text_len = end_idx - start_idx
                        llm_pos_ids_list.append(
                            torch.arange(text_len, device=input_ids.device)
                            .view(1, -1)
                            .expand(3, -1)
                            + current_pos
                        )
                        current_pos += text_len
                    # image == 1, video == 2
                    else:
                        grid_thw = next(grid_iters[modality_type])
                        vision_position_ids = get_vision_position_ids(
                            current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                        )
                        # Packing can truncate the image placeholder span at a block
                        # boundary while image_grid_thw still describes the full image.
                        # Surviving tokens are the patch prefix -> align positions to them.
                        placeholder_len = end_idx - start_idx
                        grid_len = int(vision_position_ids.shape[1])
                        if grid_len != placeholder_len:
                            if grid_len > placeholder_len:
                                vision_position_ids = vision_position_ids[:, :placeholder_len]
                            else:
                                last = vision_position_ids[:, -1:].expand(3, placeholder_len - grid_len)
                                vision_position_ids = torch.cat([vision_position_ids, last], dim=1)
                            import warnings as _warnings

                            _warnings.warn(
                                f"[qwen3_5_mrope packing] batch_idx={batch_idx} group "
                                f"modality={'image' if modality_type == 1 else 'video'}: start_idx {start_idx},  end_idx {end_idx} "
                                f"placeholder_len={placeholder_len} != grid_len={grid_len} "
                                f"(grid_thw={grid_thw.tolist()}, spatial_merge_size="
                                f"{spatial_merge_size}). Truncated/aligned to {placeholder_len}; "
                                f"neat packing likely cut image tokens at a block boundary.",
                                stacklevel=2,
                            )
                        llm_pos_ids_list.append(vision_position_ids)
                        current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

                if llm_pos_ids_list:
                    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                    position_ids[:, batch_idx, nonzero_idx] = llm_positions.to(position_ids.device)
                    mrope_position_deltas.append(int(llm_positions.max()) + 1 - len(token_types))
                else:
                    # All-padding row: nothing to assign, delta is 0.
                    mrope_position_deltas.append(0)

            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas

        self.model.get_rope_index = _wrapped_get_rope_index


    def on_parallelized(self, parallel_context):
        from ..modules.attention.ulysses import patch_forward_for_sp_input_slicing

        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        self.patch_vision_attention_interface()
        if sp_size > 1:
            from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeTextModel,
                Qwen3_5MoeGatedDeltaNet,
                Qwen3_5MoeMultiTokenPredictor,
            )

            patch_forward_for_sp_input_slicing(Qwen3_5MoeTextModel, sp_group, sp_size)

            # Patch full attention for large SP (sp_size > num_attention_heads)
            text_config = (
                self.config.text_config
                if hasattr(self.config, "text_config")
                else self.config
            )
            num_attention_heads = text_config.num_attention_heads
            if sp_size > num_attention_heads:
                self._patch_ulysses_fa_for_sp(
                    num_attention_heads, sp_size
                )

            self.patch_ulysses_fa(sp_info)

            self.patch_gated_delta_net_for_sp(
                Qwen3_5MoeGatedDeltaNet, sp_group, sp_size, sp_rank
            )

            if self.mtp is not None:
                self.patch_mtp_for_sp(
                    Qwen3_5MoeMultiTokenPredictor, sp_group, sp_size, sp_rank
                )

    @classmethod
    def patch_vision_attention_interface(cls):
        from . import modeling_qwen3_5_moe
        from transformers.modeling_flash_attention_utils import _flash_attention_forward

        attn_cls = modeling_qwen3_5_moe.Qwen3_5MoeVisionAttention
        origin_attr = "_torchturbo_qwen3_5_moe_vision_origin_attention_interface"
        if getattr(attn_cls, origin_attr, None) is not None:
            return  # already patched
        setattr(attn_cls, origin_attr, attn_cls.attention_interface)

        def sp_attention_interface(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            *args,
            **kwargs,
        ):
            call_kwargs = kwargs.copy()
            position_ids = call_kwargs.pop("position_ids", None)
            dropout = call_kwargs.pop("dropout", 0.0)
            softmax_scale = call_kwargs.pop("scaling", None)
            is_causal = call_kwargs.pop("is_causal", False)

            seq_len = query_states.shape[2]
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            # Call _flash_attention_forward directly — NOT through
            # ulysses_flash_attention_forward. Vision attention doesn't
            # need SP (sequence parallel), so we bypass the Ulysses
            # all-to-all wrapper entirely. This also avoids interference
            # from _patch_ulysses_fa_for_sp's head-repeat logic and
            # from LlamaFactoryHook's global _flash_attention_forward
            # monkey-patch (which routes through ulysses_fa with
            # sp_group/sp_size from the language model).
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length=seq_len,
                is_causal=is_causal,
                dropout=dropout,
                position_ids=position_ids,
                softmax_scale=softmax_scale,
                **call_kwargs,  # cu_seq_lens_q/k, max_length_q/k flow through
            )
            return attn_output, None

        attn_cls.attention_interface = staticmethod(sp_attention_interface)

    # ------------------------------------------------------------------
    # Full Attention: sp_repeat for Q heads
    # ------------------------------------------------------------------

    @classmethod
    def _patch_ulysses_fa_for_sp(cls, num_attention_heads, sp_size):
        """Patch ulysses_flash_attention_forward to handle sp_size > num_attention_heads.

        When sp_size > num_attention_heads (e.g., SP=32 with 16 Q heads),
        Q heads are fewer than SP groups and cannot be split across ranks.
        Solution: repeat Q heads by sp_repeat_q before all-to-all, then
        dedup after all-to-all back.

        Example (35B, SP=32):
          num_attention_heads=16, sp_repeat_q=sp_size//num_attention_heads=2
          Q: [B, T/32, 16, D] → repeat → [B, T/32, 32, D]
          → all-to-all → [B, T, 1, D] → flash attn → [B, T, 1, D]
          → all-to-all back → [B, T/32, 32, D] → dedup → [B, T/32, 16, D]

        K/V are handled by the existing GQA repeat in ulysses_flash_attention_forward
        (num_key_value_heads=2, repeat by 16 → 32 heads, divisible by sp_size=32).
        """
        from ..modules.attention.ulysses import ulysses as ulysses_module
        from ..modules.attention import ulysses as ulysses_pkg

        sp_repeat_q = sp_size // num_attention_heads
        if sp_repeat_q <= 1:
            return

        _original_ulysses_fa = ulysses_module.ulysses_flash_attention_forward

        @functools.wraps(_original_ulysses_fa)
        def _ulysses_fa_with_sp_repeat(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            *args,
            position_ids=None,
            sp_group=None,
            sp_size=1,
            **kwargs,
        ):
            # Repeat Q heads to match sp_size
            query_states = query_states.repeat_interleave(sp_repeat_q, dim=2)

            # Call original ulysses flash attention
            attn_output = _original_ulysses_fa(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length,
                *args,
                position_ids=position_ids,
                sp_group=sp_group,
                sp_size=sp_size,
                **kwargs,
            )

            # Dedup: after all-to-all back, output has sp_repeat_q * num_q_heads heads.
            # Each group of sp_repeat_q heads are identical copies → take first of each.
            attn_output = attn_output[:, :, ::sp_repeat_q, :]

            return attn_output

        ulysses_module.ulysses_flash_attention_forward = _ulysses_fa_with_sp_repeat
        ulysses_pkg.ulysses_flash_attention_forward = _ulysses_fa_with_sp_repeat

    # ------------------------------------------------------------------
    # MTP: SP input slicing
    # ------------------------------------------------------------------

    @classmethod
    def patch_mtp_for_sp(cls, mtp_cls, sp_group, sp_size, sp_rank):
        """Slice MTP inputs under SP (VLM path keeps input_ids full-length).
        The MTP forward calls ``_roll_left_one(cur_input_ids)`` on
        ``input_ids``, then ``embed_tokens(cur_input_ids)`` to produce
        the "next-token" embedding.  If ``input_ids`` were sliced to
        ``[B, T/sp]`` first, the rolled-in token at the slice boundary
        would be ``pad_token_id`` instead of the real next token.
        """
        from ..modules.attention.ulysses import slice_input_tensor

        original_forward = mtp_cls.forward

        def _mtp_sp_forward(self_mtp, input_ids, hidden_states, embed_tokens, **kwargs):
            # slice position metadata for SP
            position_embeddings = kwargs.pop("position_embeddings", None)
            attention_mask = kwargs.pop("attention_mask", None)
            position_ids = kwargs.pop("position_ids", None)
            pad_token_id = int(kwargs.pop("pad_token_id", 0))

            if position_ids is not None:
                position_ids = slice_input_tensor(
                    position_ids,
                    dim=-1,
                    group=sp_group,
                    rank=sp_rank,
                    padding=False,
                )
            if position_embeddings is not None:
                cos, sin = position_embeddings
                position_embeddings = (
                    slice_input_tensor(cos, dim=1, group=sp_group, rank=sp_rank, padding=False),
                    slice_input_tensor(sin, dim=1, group=sp_group, rank=sp_rank, padding=False),
                )

            # wrap embed_tokens so fc([embeds || hidden]) stays in shape
            def _sp_embed_tokens(ids):
                # ids is full-length [B, T]; embed produces [B, T, H]
                # slice back to [B, T/sp, H] to match hidden_states
                return slice_input_tensor(
                    embed_tokens(ids),
                    dim=1,
                    group=sp_group,
                    rank=sp_rank,
                    padding=False,
                )

            return original_forward(
                self_mtp, input_ids, hidden_states, _sp_embed_tokens,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pad_token_id=pad_token_id,
                **kwargs,
            )

        mtp_cls.forward = _mtp_sp_forward

    # ------------------------------------------------------------------
    # GatedDeltaNet: head-parallel SP with sp_repeat
    # ------------------------------------------------------------------

    @classmethod
    def patch_gated_delta_net_for_sp(cls, attn_cls, sp_group, sp_size, sp_rank):
        """Head-parallel SP for GatedDeltaNet linear attention.

        Each SP rank computes num_heads/sp_size heads on the full sequence.
        Q/K/V are split first then gathered individually.

        When sp_size > num_k_heads (e.g., SP=32 with 16 key heads),
        Q/K heads are repeated by sp_repeat to match sp_size before
        the all-to-all. V already has enough heads and does NOT repeat.

        Shape reference (Qwen3.5-35B-A3B):
          nk_full = linear_num_key_heads = 16
          nv_full = linear_num_value_heads = 32
          dk = linear_key_head_dim = 128
          dv = linear_value_head_dim = 128
          G_orig = nv_full / nk_full = 2

          sp_repeat = max(1, sp_size // nk_full)
          sh = nk_full * sp_repeat      (super-head count)
          G_per_sh = nv_full // sh       (value heads per super-head)

          SP=1:  sp_repeat=1, sh=16, G_per_sh=2, nk=16, nv=32
          SP=2:  sp_repeat=1, sh=16, G_per_sh=2, nk=8,  nv=16
          SP=32: sp_repeat=2, sh=32, G_per_sh=1, nk=1,  nv=1
        """
        from ..modules.attention.ulysses.ulysses import (
            gather_seq_scatter_heads,
            gather_heads_scatter_seq,
        )
        from .modeling_qwen3_5_moe import _get_cu_seqlens
        from torchturbo.ops import ops
        import torch.nn.functional as F

        def _slice_param(weight, dim):
            """Slice a parameter along `dim` for this SP rank.

            e.g., A_log [32] → [16] for SP=2 rank 0/1
                  conv1d.weight [8192, 1, 4] → [4096, 1, 4] for SP=2 rank 0/1
            """
            slices = [slice(None)] * weight.dim()
            d = weight.size(dim)
            slices[dim] = slice(
                sp_rank * d // sp_size, (sp_rank + 1) * d // sp_size
            )
            return weight[tuple(slices)]

        def sp_forward(self_attn, hidden_states, **kwargs):
            # Apply mask & extract shapes
            cu_seqlens = None
            attention_mask = kwargs.get("attention_mask", None)
            cu_seqlens = _get_cu_seqlens(attention_mask)
            B, local_T, _ = hidden_states.shape

            nk_full, nv_full = self_attn.num_k_heads, self_attn.num_v_heads
            dk, dv = self_attn.head_k_dim, self_attn.head_v_dim

            # Projections on local sequence shard [B, T/sp, *]
            mixed_qkv = self_attn.in_proj_qkv(hidden_states)
            z = self_attn.in_proj_z(hidden_states)
            b = self_attn.in_proj_b(hidden_states)
            a = self_attn.in_proj_a(hidden_states)

            # Split fused QKV → Q/K repeat → super-head packing → all-to-all
            #
            # When sp_size > nk_full (e.g., SP=32), Q/K have fewer heads (16)
            # than SP groups (32). Solution: repeat Q/K heads by sp_repeat,
            # so super-head count sh = nk_full * sp_repeat = sp_size.
            #
            # V has nv_full heads which is already >= sp_size, no repeat needed.
            q_raw, k_raw, v_raw = torch.split(
                mixed_qkv,
                [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim],
                dim=-1,
            )

            sp_repeat = max(1, sp_size // nk_full)
            sh = nk_full * sp_repeat  # super-head count
            G_per_sh = nv_full // sh  # value heads per super-head

            # Q/K: reshape to heads, repeat if sp_repeat > 1
            q_shaped = q_raw.reshape(B, local_T, nk_full, dk)
            k_shaped = k_raw.reshape(B, local_T, nk_full, dk)
            if sp_repeat > 1:
                q_shaped = q_shaped.repeat_interleave(sp_repeat, dim=2)
                k_shaped = k_shaped.repeat_interleave(sp_repeat, dim=2)

            # V: reshape to super-head grouping (no repeat)
            v_shaped = v_raw.reshape(B, local_T, sh, G_per_sh * dv)

            # Pack Q/K/V into super-heads
            interleaved_qkv = torch.cat(
                [q_shaped, k_shaped, v_shaped], dim=-1
            )

            # All-to-all #1: gather seq / scatter heads
            interleaved_qkv = gather_seq_scatter_heads(
                interleaved_qkv,
                seq_dim=1,
                head_dim=2,
                sp_group=sp_group,
                sp_size=sp_size,
            )
            T = interleaved_qkv.shape[1]
            nk = nk_full * sp_repeat // sp_size  # key heads per rank
            nv = nv_full // sp_size  # value heads per rank

            # De-interleave super-head → separate Q, K, V
            q_raw = interleaved_qkv[..., :dk]
            k_raw = interleaved_qkv[..., dk : 2 * dk]
            v_raw = interleaved_qkv[..., 2 * dk :].reshape(B, T, nv, dv)

            # Conv1d on local heads
            # Re-fuse Q/K/V → conv1d with sliced+repeated weights → re-split
            local_qkv = torch.cat(
                [
                    q_raw.reshape(B, T, nk * dk),
                    k_raw.reshape(B, T, nk * dk),
                    v_raw.reshape(B, T, nv * dv),
                ],
                dim=-1,
            )

            w = self_attn.conv1d.weight  # [conv_dim, 1, kernel]
            wq, wk, wv = torch.split(
                w,
                [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim],
                dim=0,
            )

            # Q/K weight: head-level repeat, then slice by sp_size.
            # CRITICAL: repeat_interleave on flat dim=0 is channel-level
            # (c0,c0,c1,c1,...), but Q/K data is head-level repeated
            # (H0_all_128ch, H0_all_128ch, ...). Must reshape to [heads, dk, 1, K]
            # first, repeat on head dim, then flatten back.
            K = self_attn.conv_kernel_size
            if sp_repeat > 1:
                wq = wq.reshape(nk_full, dk, 1, K).repeat_interleave(sp_repeat, dim=0).reshape(-1, 1, K)
                wk = wk.reshape(nk_full, dk, 1, K).repeat_interleave(sp_repeat, dim=0).reshape(-1, 1, K)
            # wv: [nv_full * dv, 1, K] = [32*128, 1, 4], no repeat needed

            local_conv_weight = torch.cat(
                [_slice_param(wq, 0), _slice_param(wk, 0), _slice_param(wv, 0)], dim=0
            )

            local_conv_bias = None
            if self_attn.conv1d.bias is not None:
                bq, bk, bv = torch.split(
                    self_attn.conv1d.bias,
                    [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim],
                    dim=0,
                )
                if sp_repeat > 1:
                    bq = bq.reshape(nk_full, dk).repeat_interleave(sp_repeat, dim=0).reshape(-1)
                    bk = bk.reshape(nk_full, dk).repeat_interleave(sp_repeat, dim=0).reshape(-1)
                local_conv_bias = torch.cat(
                    [_slice_param(bq, 0), _slice_param(bk, 0), _slice_param(bv, 0)],
                    dim=0,
                )

            local_qkv, _ = ops.causal_conv1d(
                x=local_qkv,  # [B, T, D]
                weight=local_conv_weight.squeeze(1),  # d, 1, w -> d, w
                bias=local_conv_bias,
                activation=self_attn.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
            )

            q, k, v = torch.split(
                local_qkv, [nk * dk, nk * dk, nv * dv], dim=-1
            )
            q = q.reshape(B, T, nk, dk)
            k = k.reshape(B, T, nk, dk)
            v = v.reshape(B, T, nv, dv)

            # GQA expand
            # When sp_repeat > 1, the GQA ratio is partially absorbed by sp_repeat.
            # Use per-rank nk vs nv (not full) to decide whether to expand.
            if nk < nv:
                r = nv // nk
                q = q.repeat_interleave(r, dim=2)
                k = k.repeat_interleave(r, dim=2)

            # All-to-all for z, b, a (use value-head dimension nv_full)
            z = gather_seq_scatter_heads(
                z.reshape(B, local_T, nv_full, dv),
                seq_dim=1,
                head_dim=2,
                sp_group=sp_group,
                sp_size=sp_size,
            )
            b = gather_seq_scatter_heads(
                b.unsqueeze(-1),
                seq_dim=1,
                head_dim=2,
                sp_group=sp_group,
                sp_size=sp_size,
            ).squeeze(-1)
            a = gather_seq_scatter_heads(
                a.unsqueeze(-1),
                seq_dim=1,
                head_dim=2,
                sp_group=sp_group,
                sp_size=sp_size,
            ).squeeze(-1)

            # Compute gate (beta) and delta (g)
            beta = b.sigmoid()
            A_log_local = _slice_param(self_attn.A_log, 0)
            dt_bias_local = _slice_param(self_attn.dt_bias, 0)
            g = -A_log_local.float().exp() * F.softplus(a.float() + dt_bias_local)

            # Chunk Gated Delta Rule (core linear attention)
            out, _ = ops.gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens = cu_seqlens,
            )

            # All-to-all #2: gather heads / scatter seq back
            out = gather_heads_scatter_seq(
                out, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )
            z = gather_heads_scatter_seq(
                z, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )

            # RMSNorm Gated + output projection
            out = self_attn.norm(out.reshape(-1, dv), z.reshape(-1, dv))
            return self_attn.out_proj(out.reshape(B, local_T, -1))

        attn_cls.forward = sp_forward

    def get_weight_converter(self, parallel_context):
        return Qwen3_5MoeWeightConverter(self, parallel_context)

    def on_materialized(self):
        device = get_device_runtime().device
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        lm = self.model.language_model
        rotary_emb = lm.rotary_emb
        rope_init_fn = rotary_emb.compute_default_rope_parameters
        if rotary_emb.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, rotary_emb.attention_scaling = rope_init_fn(rotary_emb.config, device)
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary_emb.register_buffer(
            "original_inv_freq", inv_freq.clone(), persistent=False
        )

        vis = self.model.visual
        if hasattr(vis, "rotary_pos_emb"):
            vis_rope = vis.rotary_pos_emb
            vis_inv_freq = 1.0 / (
                vis_rope.theta
                ** (
                    torch.arange(0, vis_rope.dim, 2, dtype=torch.float)
                    / vis_rope.dim
                )
            ).to(device)
            vis_rope.register_buffer("inv_freq", vis_inv_freq, persistent=False)
