import functools
import math

import torch
from transformers import AutoModelForImageTextToText
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from typing_extensions import override

from torchturbo.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from torchturbo.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ModelOutputWithPast,
)
from torchturbo.runtime import get_device_runtime

from ...engine.config import ParameterGroup
from ..base import TurboVLMModel
from ..registry import register_model
from ..weight_converter import BaseWeightConverter


class Qwen3_5WeightConverter(BaseWeightConverter):
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
    "qwen3_5",
    config_cls=Qwen3_5Config,
    auto_class=AutoModelForImageTextToText,
)
class TurboQwen3_5ForConditionalGeneration(
    TurboVLMModel, Qwen3_5ForConditionalGeneration
):

    config_class = Qwen3_5Config
    base_model_prefix = "model"
    multi_modal_projector_module_names = ("model.visual.merger",)
    language_model_module_names = ("model.language_model",)

    @property
    def visual(self):
        return self.model.visual

    def get_parameter_groups(self):
        groups = super().get_parameter_groups()
        groups[ParameterGroup.VISION_TOWER] = (
            (name, param)
            for name, param in self.model.visual.named_parameters(prefix="model.visual")
            if name != "model.visual.merger"
            and not name.startswith("model.visual.merger.")
        )
        return groups

    def get_norm(self):
        return self.model.language_model.norm

    def __init__(self, config: Qwen3_5Config) -> None:
        self._pre_init_patches()
        if hasattr(config, "vision_config"):
            config.vision_config._attn_implementation = "flash_attention_2"
        super().__init__(config)
        self._post_init_patches()

    @can_return_tuple
    @override
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        skip_lm_head: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5CausalLMOutputWithPast | Qwen3_5ModelOutputWithPast:
        """Optionally return pre-head states for TorchTurbo chunked LM loss."""
        if not skip_lm_head:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                return_dict=True,
                **kwargs,
            )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()

    @staticmethod
    def _pre_init_patches():
        # No MoE token-dispatcher patching for dense (no experts).
        pass

    def _post_init_patches(self):
        pass

    def on_parallelized(self, parallel_context):
        from ..modules.attention.ulysses import patch_forward_for_sp_input_slicing

        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        self.patch_vision_attention_interface()
        if sp_size > 1:
            from torchturbo.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5TextModel,
                Qwen3_5GatedDeltaNet,
            )

            patch_forward_for_sp_input_slicing(Qwen3_5TextModel, sp_group, sp_size)

            # Patch full attention when num_attention_heads is not divisible by
            text_config = (
                self.config.text_config
                if hasattr(self.config, "text_config")
                else self.config
            )
            num_attention_heads = text_config.num_attention_heads
            if num_attention_heads % sp_size != 0:
                self._patch_ulysses_fa_for_sp(num_attention_heads, sp_size)

            self.patch_ulysses_fa(sp_info)

            self.patch_gated_delta_net_for_sp(
                Qwen3_5GatedDeltaNet, sp_group, sp_size, sp_rank
            )

    @classmethod
    def patch_vision_attention_interface(cls):
        from . import modeling_qwen3_5
        from transformers.modeling_flash_attention_utils import _flash_attention_forward

        attn_cls = modeling_qwen3_5.Qwen3_5VisionAttention
        origin_attr = "_torchturbo_qwen3_5_vision_origin_attention_interface"
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

            # Vision attention doesn't need SP; bypass the Ulysses all-to-all wrapper.
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
                **call_kwargs,
            )
            return attn_output, None

        attn_cls.attention_interface = staticmethod(sp_attention_interface)

    # ------------------------------------------------------------------
    # Full Attention: sp_repeat for Q heads
    # ------------------------------------------------------------------

    @classmethod
    def _patch_ulysses_fa_for_sp(cls, num_attention_heads, sp_size):
        from ..modules.attention.ulysses import ulysses as ulysses_module
        from ..modules.attention import ulysses as ulysses_pkg

        sp_repeat_q = math.lcm(sp_size, num_attention_heads) // num_attention_heads
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
            query_states = query_states.repeat_interleave(sp_repeat_q, dim=2)
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
            attn_output = attn_output[:, :, ::sp_repeat_q, :]
            return attn_output

        ulysses_module.ulysses_flash_attention_forward = _ulysses_fa_with_sp_repeat
        ulysses_pkg.ulysses_flash_attention_forward = _ulysses_fa_with_sp_repeat


    @classmethod
    def patch_gated_delta_net_for_sp(cls, attn_cls, sp_group, sp_size, sp_rank):
        from ..modules.attention.ulysses.ulysses import (
            gather_seq_scatter_heads,
            gather_heads_scatter_seq,
        )
        from .modeling_qwen3_5 import _get_cu_seqlens
        from torchturbo.ops import ops
        import torch.nn.functional as F

        def _slice_param(weight, dim):
            slices = [slice(None)] * weight.dim()
            d = weight.size(dim)
            slices[dim] = slice(sp_rank * d // sp_size, (sp_rank + 1) * d // sp_size)
            return weight[tuple(slices)]

        def sp_forward(self_attn, hidden_states, **kwargs):
            cu_seqlens = None
            attention_mask = kwargs.get("attention_mask", None)
            cu_seqlens = _get_cu_seqlens(attention_mask)
            B, local_T, _ = hidden_states.shape

            nk_full, nv_full = self_attn.num_k_heads, self_attn.num_v_heads
            dk, dv = self_attn.head_k_dim, self_attn.head_v_dim

            mixed_qkv = self_attn.in_proj_qkv(hidden_states)
            z = self_attn.in_proj_z(hidden_states)
            b = self_attn.in_proj_b(hidden_states)
            a = self_attn.in_proj_a(hidden_states)

            q_raw, k_raw, v_raw = torch.split(
                mixed_qkv, [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim], dim=-1
            )

            sp_repeat = max(1, sp_size // nk_full)
            sh = nk_full * sp_repeat
            G_per_sh = nv_full // sh

            q_shaped = q_raw.reshape(B, local_T, nk_full, dk)
            k_shaped = k_raw.reshape(B, local_T, nk_full, dk)
            if sp_repeat > 1:
                q_shaped = q_shaped.repeat_interleave(sp_repeat, dim=2)
                k_shaped = k_shaped.repeat_interleave(sp_repeat, dim=2)

            v_shaped = v_raw.reshape(B, local_T, sh, G_per_sh * dv)
            interleaved_qkv = torch.cat([q_shaped, k_shaped, v_shaped], dim=-1)

            interleaved_qkv = gather_seq_scatter_heads(
                interleaved_qkv, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )
            T = interleaved_qkv.shape[1]
            nk = nk_full * sp_repeat // sp_size
            nv = nv_full // sp_size

            q_raw = interleaved_qkv[..., :dk]
            k_raw = interleaved_qkv[..., dk : 2 * dk]
            v_raw = interleaved_qkv[..., 2 * dk :].reshape(B, T, nv, dv)

            local_qkv = torch.cat(
                [
                    q_raw.reshape(B, T, nk * dk),
                    k_raw.reshape(B, T, nk * dk),
                    v_raw.reshape(B, T, nv * dv),
                ],
                dim=-1,
            )

            w = self_attn.conv1d.weight
            wq, wk, wv = torch.split(
                w, [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim], dim=0
            )

            K = self_attn.conv_kernel_size
            if sp_repeat > 1:
                wq = wq.reshape(nk_full, dk, 1, K).repeat_interleave(sp_repeat, dim=0).reshape(-1, 1, K)
                wk = wk.reshape(nk_full, dk, 1, K).repeat_interleave(sp_repeat, dim=0).reshape(-1, 1, K)

            local_conv_weight = torch.cat(
                [_slice_param(wq, 0), _slice_param(wk, 0), _slice_param(wv, 0)], dim=0
            )

            local_conv_bias = None
            if self_attn.conv1d.bias is not None:
                bq, bk, bv = torch.split(
                    self_attn.conv1d.bias, [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim], dim=0
                )
                if sp_repeat > 1:
                    bq = bq.reshape(nk_full, dk).repeat_interleave(sp_repeat, dim=0).reshape(-1)
                    bk = bk.reshape(nk_full, dk).repeat_interleave(sp_repeat, dim=0).reshape(-1)
                local_conv_bias = torch.cat(
                    [_slice_param(bq, 0), _slice_param(bk, 0), _slice_param(bv, 0)], dim=0
                )

            local_qkv, _ = ops.causal_conv1d(
                x=local_qkv,
                weight=local_conv_weight.squeeze(1),
                bias=local_conv_bias,
                activation=self_attn.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
            )

            q, k, v = torch.split(local_qkv, [nk * dk, nk * dk, nv * dv], dim=-1)
            q = q.reshape(B, T, nk, dk)
            k = k.reshape(B, T, nk, dk)
            v = v.reshape(B, T, nv, dv)

            if nk < nv:
                r = nv // nk
                q = q.repeat_interleave(r, dim=2)
                k = k.repeat_interleave(r, dim=2)

            z = gather_seq_scatter_heads(
                z.reshape(B, local_T, nv_full, dv), seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )
            b = gather_seq_scatter_heads(
                b.unsqueeze(-1), seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            ).squeeze(-1)
            a = gather_seq_scatter_heads(
                a.unsqueeze(-1), seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            ).squeeze(-1)

            beta = b.sigmoid()
            A_log_local = _slice_param(self_attn.A_log, 0)
            dt_bias_local = _slice_param(self_attn.dt_bias, 0)
            g = -A_log_local.float().exp() * F.softplus(a.float() + dt_bias_local)

            out, _ = ops.gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

            out = gather_heads_scatter_seq(
                out, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )
            z = gather_heads_scatter_seq(
                z, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )

            out = self_attn.norm(out.reshape(-1, dv), z.reshape(-1, dv))
            return self_attn.out_proj(out.reshape(B, local_T, -1))

        attn_cls.forward = sp_forward

    def get_weight_converter(self, parallel_context):
        return Qwen3_5WeightConverter(self, parallel_context)

    def on_materialized(self):
        if getattr(self.config, "tie_word_embeddings", False):
            self.tie_weights()

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
                ** (torch.arange(0, vis_rope.dim, 2, dtype=torch.float) / vis_rope.dim)
            ).to(device)
            vis_rope.register_buffer("inv_freq", vis_inv_freq, persistent=False)
