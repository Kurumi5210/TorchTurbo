import os
import functools

import torch

from transformers import AutoModelForImageTextToText

from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from torchturbo.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

from torchturbo.ops import npu_fused_ops
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.utils.device import is_npu_available, get_device_name

from ..registry import register_model
from ..weight_converter import WeightConverterMixin, FusedExpertWeightConverter


class Qwen3_5MoeWeightConverter(FusedExpertWeightConverter):

    def __init__(self, config, moe_config, mesh):
        if hasattr(config, "text_config"):
            text_config = config.text_config
        else:
            text_config = config
        super().__init__(text_config, moe_config, mesh)
        self.full_config = config


@register_model(
    "qwen3_5_moe",
    config_cls=Qwen3_5MoeConfig,
    auto_class=AutoModelForImageTextToText,
)
class TurboQwen3_5MoeForConditionalGeneration(
    WeightConverterMixin, Qwen3_5MoeForConditionalGeneration
):

    is_vlm = True
    config_class = Qwen3_5MoeConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3_5MoeConfig) -> None:
        self._pre_init_patches()
        # FSDP2 requires uniform param dtype. The engine sets top-level
        # config.dtype=float32 but Qwen3.5's sub-configs may still carry
        # bfloat16, causing mixed dtypes at construction time.
        if hasattr(config, "text_config") and hasattr(config.text_config, "dtype"):
            config.text_config.dtype = torch.float32
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "dtype"):
            config.vision_config.dtype = torch.float32
        # Force vision attention to eager to prevent Ulysses from patching
        # visual encoder's flash attention (which lacks position_ids).
        if hasattr(config, "vision_config"):
            config.vision_config._attn_implementation = "eager"
        super().__init__(config)
        self._post_init_patches()

    @classmethod
    def get_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    @staticmethod
    def _pre_init_patches():
        import torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen3_5_module

        if is_npu_available:
            from torchturbo.ops.npu_fused_ops import npu_rms_norm
            MoEAlltoAllTokenDispatcher.token_permutation = (
                npu_fused_ops.fused_token_permutation_npu
            )
            MoEAlltoAllTokenDispatcher.token_unpermutation = (
                npu_fused_ops.fused_token_unpermutation_npu
            )

    def _post_init_patches(self):
        import torch
        import functools

        # Wrap get_rope_index to auto-fill mm_token_type_ids when not provided.
        # Open-source LF collator doesn't pass this arg.
        original_get_rope_index = self.model.get_rope_index
        image_token_id = getattr(self.config, "image_token_id", None)
        video_token_id = getattr(self.config, "video_token_id", None)

        @functools.wraps(original_get_rope_index)
        def _wrapped_get_rope_index(*args, **kwargs):
            if "mm_token_type_ids" not in kwargs or kwargs["mm_token_type_ids"] is None:
                input_ids = kwargs.get("input_ids")
                if input_ids is not None:
                    kwargs["mm_token_type_ids"] = torch.zeros_like(input_ids)
                    if image_token_id is not None:
                        kwargs["mm_token_type_ids"][input_ids == image_token_id] = 1
                    if video_token_id is not None:
                        kwargs["mm_token_type_ids"][input_ids == video_token_id] = 2
            return original_get_rope_index(*args, **kwargs)

        self.model.get_rope_index = _wrapped_get_rope_index

    def setup_sp(self, sp_info):
        from ..modules.attention.ulysses import patch_forward_for_sp_input_slicing

        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
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

            self.patch_gated_delta_net_for_sp(
                Qwen3_5MoeGatedDeltaNet, sp_group, sp_size, sp_rank
            )

            if self.mtp is not None:
                self.patch_mtp_for_sp(
                    Qwen3_5MoeMultiTokenPredictor, sp_group, sp_size, sp_rank
                )

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
        """Slice MTP inputs under SP (VLM path keeps input_ids full-length)."""
        from ..modules.attention.ulysses import slice_input_tensor

        original_forward = mtp_cls.forward

        def _mtp_sp_forward(self_mtp, input_ids, hidden_states, embed_tokens, **kwargs):
            input_ids = slice_input_tensor(
                input_ids, dim=1, group=sp_group, rank=sp_rank, padding=False
            )
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                kwargs["position_ids"] = slice_input_tensor(
                    kwargs["position_ids"],
                    dim=-1,
                    group=sp_group,
                    rank=sp_rank,
                    padding=False,
                )
            if (
                "position_embeddings" in kwargs
                and kwargs["position_embeddings"] is not None
            ):
                cos, sin = kwargs["position_embeddings"]
                cos = slice_input_tensor(
                    cos, dim=1, group=sp_group, rank=sp_rank, padding=False
                )
                sin = slice_input_tensor(
                    sin, dim=1, group=sp_group, rank=sp_rank, padding=False
                )
                kwargs["position_embeddings"] = (cos, sin)
            return original_forward(
                self_mtp, input_ids, hidden_states, embed_tokens, **kwargs
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
        from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            apply_mask_to_padding_states,
        )
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
            hidden_states = apply_mask_to_padding_states(
                hidden_states, kwargs.get("attention_mask")
            )
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
            ).transpose(1, 2)

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

            local_conv_dim = local_qkv.shape[1]

            if self_attn.causal_conv1d_fn is not None:
                local_qkv = self_attn.causal_conv1d_fn(
                    x=local_qkv,
                    weight=local_conv_weight.squeeze(1),
                    bias=local_conv_bias,
                    activation=self_attn.activation,
                    seq_idx=None,
                )
            else:
                local_qkv = F.silu(
                    F.conv1d(
                        local_qkv,
                        local_conv_weight,
                        bias=local_conv_bias,
                        groups=local_conv_dim,
                        padding=self_attn.conv_kernel_size - 1,
                    )[:, :, :T]
                )

            # Split conv output → reshape to heads
            local_qkv = local_qkv.transpose(1, 2)
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
            out, _ = self_attn.chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
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

    def get_weight_converter(self, mesh):
        return Qwen3_5MoeWeightConverter(
            self.config, self.get_moe_config(), mesh
        )

    def post_weight_load(self):
        device = get_device_name()

        lm = self.model.language_model
        rotary_emb = lm.rotary_emb
        inv_freq, rotary_emb.attention_scaling = (
            rotary_emb.compute_default_rope_parameters(rotary_emb.config, device)
        )
        rotary_emb.original_inv_freq = inv_freq
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
            )
            vis_rope.register_buffer("inv_freq", vis_inv_freq, persistent=False)
