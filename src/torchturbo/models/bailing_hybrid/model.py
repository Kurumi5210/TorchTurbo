from typing_extensions import override

import torch
from transformers import AutoModelForCausalLM

from torchturbo.models.modules.attention.ulysses import (
    patch_kda_attention_forward_for_sp,
    ulysses_flash_attention_forward,
)
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.utils.device import get_device_name, is_npu_available
from torchturbo.utils.flash_attn_utils import (
    index_first_axis,
    pad_input,
    unpad_input,
    get_unpad_data_lf,
)
from torchturbo.utils.functools import cache
from torchturbo.models.modules.attention.ulysses.ulysses import (
    gather_seq_scatter_heads,
    gather_heads_scatter_seq,
)

from ..registry import register_model
from ..weight_converter import BaseWeightConverter, WeightConverterMixin
from .configuration_bailing_moe_v3 import BailingMoeV3Config
from .modeling_bailing_moe_v3 import BailingMoeV3ForCausalLM

class BailingV3WeightConverter(BaseWeightConverter):
    def get_hf_kda_alog(self):
        return "A_log"

    def get_hf_kda_dt_bias(self):
        return "dt_bias"

    @override
    def get_hf_gate_bias_name(self):
        return "gate.expert_bias"

    @override
    def get_default_hf_dtype(self, param_name):
        if self.get_hf_gate_bias_name() in param_name:
            return torch.float32
        if self.get_hf_kda_alog() in param_name:
            return torch.float32
        if self.get_hf_kda_dt_bias() in param_name:
            return torch.float32
        return self.dtype


@cache
def _get_cu_seqlens(attention_mask):
    if attention_mask is not None and torch.max(attention_mask) > 1:
        _, cu_seqlens, _ = get_unpad_data_lf(attention_mask)
        if cu_seqlens[-1] != attention_mask.size(1):
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (0, 1), value=attention_mask.size(1))
        return cu_seqlens
    return None

@register_model(
    "bailing_hybrid", config_cls=BailingMoeV3Config, auto_class=AutoModelForCausalLM
)
class TurboBailingMoeV3ForCausalLM(WeightConverterMixin, BailingMoeV3ForCausalLM):
    config_class = BailingMoeV3Config
    base_model_prefix = "model"
    _supports_flash_attn_2 = True

    @override
    def get_weight_converter(self, mesh):
        return BailingV3WeightConverter(self.config, self.get_moe_config(), mesh)

    def __init__(self, config: BailingMoeV3Config) -> None:
        self._pre_init_patches(config)
        super().__init__(config)

    @classmethod
    def get_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def setup_sp(self, sp_info):
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        assert sp_size >= 1, "must be"

        if self.config._attn_implementation != "flash_attention_2":
            raise NotImplementedError(
                "BailingMoeV3 sequence parallelism requires flash_attention_2 "
                "for MLA layers. Pass `--flash_attn fa2` to enable it."
            )

        self.patch_mla_attention_interface_for_sp(sp_group, sp_size, sp_rank)
        self.patch_kda_forward(sp_group, sp_size, sp_rank)

    @classmethod
    def patch_kda_forward(cls, sp_group, sp_size, sp_rank):
        def get_parameter_local_cp(weight, dim):
            if sp_size == 1:
                return weight

            slices = [slice(None)] * weight.dim()
            dim_size = weight.size(dim=dim)
            slices[dim] = slice(
                sp_rank * dim_size // sp_size, (sp_rank + 1) * dim_size // sp_size
            )
            weight = weight[tuple(slices)]
            return weight

        def apply_causal_conv1d_fn(conv1d, tensor):
            # self.conv1d.weight: [D, 1, W] (nn.Conv1d format), squeeze to [D, W] for _causal_conv1d_prefill
            conv1d_weight = get_parameter_local_cp(conv1d.weight, dim=0).squeeze(1)
            conv1d_bias = get_parameter_local_cp(conv1d.bias, dim=0) if conv1d.bias else None
            tensor_conv, _ = conv1d._causal_conv1d_prefill(
                x=tensor,
                weight=conv1d_weight,
                bias=conv1d_bias,
                initial_state=None,
                output_final_state=False
            )
            return tensor_conv

        def naive_causal_conv1d_wrapper(conv1d, tensor, cu_seqlens):
            if cu_seqlens is not None:
                # Variable-length: loop over sequences (naive functions don't support cu_seqlens)
                N = cu_seqlens.shape[0] - 1
                outputs = []
                for i in range(N):
                    s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                    o = apply_causal_conv1d_fn(conv1d, tensor[:, s:e])
                    outputs.append(o)
                o = torch.cat(outputs, dim=1)
                # Pad output back to original sequence length (padding positions excluded by cu_seqlens)
                seq_len = tensor.shape[1]
                if o.shape[1] < seq_len:
                    o = torch.cat([o, o.new_zeros(o.shape[0], seq_len - o.shape[1], *o.shape[2:])], dim=1)
                return o
            else:
                return apply_causal_conv1d_fn(conv1d, tensor)

        from fla.modules.convolution import causal_conv1d
        def triton_conv1d_wrapper(conv1d, tensor, cu_seqlens):
            conv1d_weight = get_parameter_local_cp(conv1d.weight, dim=0).squeeze(1)
            conv1d_bias = get_parameter_local_cp(conv1d.bias, dim=0) if conv1d.bias else None
            result, _ = causal_conv1d(
                x=tensor,
                weight=conv1d_weight,
                bias=conv1d_bias,
                residual=None,
                initial_state=None,
                output_final_state=False,
                activation=conv1d.activation,
                backend='triton',
                cu_seqlens=cu_seqlens,
            )
            return result

        from fla.modules.fused_norm_gate import rms_norm_gated
        def fused_norm_gate_wrapper(fng, x, g):
            return rms_norm_gated(
                x,
                g,
                fng.weight,
                None, # bias
                fng.activation,
                residual=None,
                eps=fng.eps,
                prenorm=False,
                residual_in_fp32=False,
            )

        from .modeling_bailing_moe_v3 import _split_last_dim_heads, _merge_last_two_dims
        from fla.ops.kda import chunk_kda
        def sp_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_value=None,
            **kwargs,
        ):
            use_cache = past_key_value is not None
            batch_size, q_len, _ = hidden_states.shape
            assert past_key_value == None, "must be"
            assert self.mode == 'chunk', "Only chunk mode is supported in training."

            cu_seqlens = _get_cu_seqlens(attention_mask)

            q = self.q_proj(hidden_states)   # [1, valid_local, H*K] or [B, T/sp, H*K]
            k = self.k_proj(hidden_states)   # [1, valid_local, H*K] or [B, T/sp, H*K]
            v = self.v_proj(hidden_states)   # [1, valid_local, H*V] or [B, T/sp, H*V]

            if sp_size > 1:
                q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)
                k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)
                v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)

            # TODO: replace with fused ops, performance of triton version is very low
            conv1d_fn = naive_causal_conv1d_wrapper # triton_conv1d_wrapper
            q = conv1d_fn(self.q_conv1d, q, cu_seqlens)
            k = conv1d_fn(self.k_conv1d, k, cu_seqlens)
            v = conv1d_fn(self.v_conv1d, v, cu_seqlens)

            # Compute g, beta from padded hidden_states (always [B, T/sp, H] format)
            if self.no_kda_lora:
                g = self.f_proj(hidden_states)
            else:
                g = self.f_b_proj(self.f_a_proj(hidden_states))

            beta = self.b_proj(hidden_states).float().sigmoid()

            q, k = map(lambda x: _split_last_dim_heads(x, self.head_k_dim), (q, k))
            v = _split_last_dim_heads(v, self.head_dim)
            g = _split_last_dim_heads(g, self.head_dim)

            if sp_size > 1:
                g = gather_seq_scatter_heads(g, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)
                beta = gather_seq_scatter_heads(beta, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)
                # Slice A_log and dt_bias
                heads_per_rank = self.num_heads // sp_size
                A_log_local = self.A_log[sp_rank * heads_per_rank : (sp_rank + 1) * heads_per_rank]
                channels_per_rank = heads_per_rank * self.head_dim
                start_ch = sp_rank * channels_per_rank
                end_ch = start_ch + channels_per_rank
                dt_bias_local = self.dt_bias[start_ch:end_ch]
            else:
                A_log_local = self.A_log
                dt_bias_local = self.dt_bias

            kda_fn = chunk_kda
            o, _ = kda_fn(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log_local,
                dt_bias=dt_bias_local,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            )

            if sp_size > 1:
                # o: [B, T, H/sp, V] -> [B, T/sp, H, V]
                o = gather_heads_scatter_seq(o, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)

            # o_norm uses g from the original padded hidden_states
            if self.no_kda_lora:
                g = self.g_proj(hidden_states)
            else:
                g = self.g_b_proj(self.g_a_proj(hidden_states))
            g = _split_last_dim_heads(g, self.head_dim)
            # o = self.o_norm(o, g)
            o = fused_norm_gate_wrapper(self.o_norm, o, g)

            o = _merge_last_two_dims(o)
            o = self.o_proj(o)
            return o, None, None

        from . import modeling_bailing_moe_v3 as bailing_moe_v3_model
        bailing_moe_v3_model.BailingMoeV3KimiDeltaAttention.forward = sp_forward

    @classmethod
    def patch_mla_attention_interface_for_sp(cls, sp_group, sp_size, sp_rank):
        from . import modeling_bailing_moe_v3 as bailing_moe_v3_model
        attn_cls = bailing_moe_v3_model.BailingMoeV3MultiLatentAttention
        origin_attr = "_torchturbo_bailing_moe_v3_origin_attention_interface"
        if getattr(attn_cls, origin_attr, None) is None:
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

            if position_ids is None:
                raise ValueError(
                    "position_ids is required for BailingMoeV3 MLA Ulysses SP."
                )
            if query_states.size(1) % sp_size != 0:
                raise ValueError(
                    "BailingMoeV3 MLA requires num_heads to be divisible by sp_size."
                )

            attn_output = ulysses_flash_attention_forward(
                query_states.transpose(1, 2).contiguous(),
                key_states.transpose(1, 2).contiguous(),
                value_states.transpose(1, 2).contiguous(),
                attention_mask,
                query_states.size(2),
                module.is_causal,
                dropout=dropout,
                position_ids=position_ids,
                softmax_scale=softmax_scale,
                sp_group=sp_group,
                sp_size=sp_size,
                **call_kwargs,
            )
            return attn_output, None

        attn_cls.attention_interface = staticmethod(sp_attention_interface)

    @staticmethod
    def _patch_mla_attention_interface_for_config(attn_cls, attn_implementation):
        origin_attr = "_torchturbo_bailing_moe_v3_origin_attention_interface"
        if getattr(attn_cls, origin_attr, None) is None:
            setattr(attn_cls, origin_attr, attn_cls.attention_interface)
        if attn_implementation == "eager":
            attn_cls.attention_interface = staticmethod(getattr(attn_cls, origin_attr))
            return

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        attn_cls.attention_interface = staticmethod(
            ALL_ATTENTION_FUNCTIONS[attn_implementation]
        )

    @staticmethod
    def _pre_init_patches(config):
        from . import modeling_bailing_moe_v3 as bailing_moe_v3_model

        TurboBailingMoeV3ForCausalLM._patch_mla_attention_interface_for_config(
            bailing_moe_v3_model.BailingMoeV3MultiLatentAttention,
            config._attn_implementation,
        )

        if is_npu_available:
            from torchturbo.ops import npu_fused_ops

            bailing_moe_v3_model.BailingMoeV3RMSNorm.forward = (
                npu_fused_ops.fused_rms_norm_forward_npu
            )
            MoEAlltoAllTokenDispatcher.token_permutation = (
                npu_fused_ops.fused_token_permutation_npu
            )
            MoEAlltoAllTokenDispatcher.token_unpermutation = (
                npu_fused_ops.fused_token_unpermutation_npu
            )

    def post_weight_load(self):
        """Initialize runtime buffers after weights are on the real device."""

        device = get_device_name()
        rotary_emb = self.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
            rotary_emb.config, device
        )
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary_emb.original_inv_freq = inv_freq
