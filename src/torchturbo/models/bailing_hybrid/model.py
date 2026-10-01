from typing_extensions import override

import types
import torch
from transformers import AutoModelForCausalLM

from torchturbo.utils.flash_attn_utils import (
    index_first_axis,
    pad_input,
    unpad_input,
)
from torchturbo.ops.packed_params import get_packed_params
from torchturbo.models.modules.attention.ulysses.ulysses import (
    gather_seq_scatter_heads,
    gather_heads_scatter_seq,
)
from torchturbo.engine.optimizers import OptimizerRecipe
from torchturbo.runtime import get_device_runtime

from ..registry import register_model
from ..base import TurboModel
from ..weight_converter import BaseWeightConverter
from .configuration_bailing_moe_v3 import BailingMoeV3Config
from .modeling_bailing_moe_v3 import BailingMoeV3ForCausalLM

class BailingV3WeightConverter(BaseWeightConverter):
    @override
    def get_hf_gate_bias_name(self):
        return "gate.expert_bias"


@register_model(
    "bailing_hybrid", config_cls=BailingMoeV3Config, auto_class=AutoModelForCausalLM
)
class TurboBailingMoeV3ForCausalLM(TurboModel, BailingMoeV3ForCausalLM):
    config_class = BailingMoeV3Config
    base_model_prefix = "model"

    @override
    def get_weight_converter(self, parallel_context, model_path):
        return BailingV3WeightConverter(self, parallel_context, model_path)

    @override
    def get_optimizer_recipe(self, optimizer_name: str) -> OptimizerRecipe:
        if optimizer_name == "muon":
            from .optimizer_recipe import get_muon_recipe

            return get_muon_recipe(self)
        return super().get_optimizer_recipe(optimizer_name)

    def get_flops_class(self):
        """Bailing v3 flops recipe (hybrid MLA+KDA attention, dense+MoE FFN, MTP)."""
        from .mfu import BailingMoeV3Flops

        return BailingMoeV3Flops

    def __init__(self, config: BailingMoeV3Config) -> None:
        self._pre_init_patches(config)
        super().__init__(config)

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def on_parallelized(self, parallel_context):
        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        assert sp_size >= 1, "must be"

        if self.config._attn_implementation != "flash_attention_2":
            raise NotImplementedError(
                "BailingMoeV3 sequence parallelism requires flash_attention_2 "
                "for MLA layers. Pass `--flash_attn fa2` to enable it."
            )

        # MLA layers dispatch through the registered turbo_ulysses interface;
        # KDA layers rewrite their forward instead.
        self.register_sp_attention_interface(sp_info)
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

        from torchturbo.ops import ops

        from fla.modules.fused_norm_gate import rms_norm_gated
        def _patched_o_norm_forward(
            fng,
            x,
            g,
            residual=None,
            prenorm=False,
            residual_in_fp32=False,
        ):
            return rms_norm_gated(
                x,
                g,
                fng.weight,
                getattr(fng, "bias", None),
                fng.activation,
                residual=residual,
                eps=fng.eps,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
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

            # Global packed boundaries ride with the forward kwargs; the
            # a2a gathers below rebuild the full sequence on each rank.
            cu_seqlens = kwargs.get("cu_seqlens")
            packed_params = get_packed_params(cu_seqlens, q_len * sp_size)
            if packed_params is not None:
                cu_seqlens = packed_params.cu_seqlens

            q = self.q_proj(hidden_states)   # [1, valid_local, H*K] or [B, T/sp, H*K]
            k = self.k_proj(hidden_states)   # [1, valid_local, H*K] or [B, T/sp, H*K]
            v = self.v_proj(hidden_states)   # [1, valid_local, H*V] or [B, T/sp, H*V]

            if sp_size > 1:
                q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)
                k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)
                v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size)

            # ops dispatcher picks the AscendC provider (priority 100) on NPU
            # and falls back to fla's triton conv otherwise — same integration
            # as the Qwen3.5 path.
            q, _ = ops.causal_conv1d(
                x=q,
                weight=get_parameter_local_cp(self.q_conv1d.weight, dim=0).squeeze(1),  # d,1,w -> d,w
                bias=get_parameter_local_cp(self.q_conv1d.bias, dim=0) if self.q_conv1d.bias else None,
                activation=self.q_conv1d.activation,
                initial_state=None,
                output_final_state=False,
                packed_params=packed_params,
            )
            k, _ = ops.causal_conv1d(
                x=k,
                weight=get_parameter_local_cp(self.k_conv1d.weight, dim=0).squeeze(1),
                bias=get_parameter_local_cp(self.k_conv1d.bias, dim=0) if self.k_conv1d.bias else None,
                activation=self.k_conv1d.activation,
                initial_state=None,
                output_final_state=False,
                packed_params=packed_params,
            )
            v, _ = ops.causal_conv1d(
                x=v,
                weight=get_parameter_local_cp(self.v_conv1d.weight, dim=0).squeeze(1),
                bias=get_parameter_local_cp(self.v_conv1d.bias, dim=0) if self.v_conv1d.bias else None,
                activation=self.v_conv1d.activation,
                initial_state=None,
                output_final_state=False,
                packed_params=packed_params,
            )

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

            self.o_norm.forward = types.MethodType(_patched_o_norm_forward, self.o_norm)
            o = self.o_norm(o, g)

            o = _merge_last_two_dims(o)
            o = self.o_proj(o)
            return o, None, None

        from . import modeling_bailing_moe_v3 as bailing_moe_v3_model
        bailing_moe_v3_model.BailingMoeV3KimiDeltaAttention.forward = sp_forward

    @staticmethod
    def _pre_init_patches(config):
        from . import modeling_bailing_moe_v3 as bailing_moe_v3_model
        from torchturbo.ops import ops

        def rms_norm_forward(self, hidden_states):
            return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)

        bailing_moe_v3_model.BailingMoeV3RMSNorm.forward = rms_norm_forward

    def on_materialized(self):
        """Initialize runtime buffers after weights are on the real device."""

        device = get_device_runtime().device
        rotary_emb = self.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
            rotary_emb.config, device
        )
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary_emb.original_inv_freq = inv_freq
