from transformers import AutoModelForCausalLM

from torchturbo.models.modules.attention.ulysses import (
    patch_kda_attention_forward_for_sp,
    ulysses_flash_attention_forward,
)
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.utils.device import get_device_name, is_npu_available

from ..registry import register_model
from .configuration_bailing_moe_v3 import BailingMoeV3Config
from .modeling_bailing_moe_v3 import BailingMoeV3ForCausalLM


@register_model(
    "bailing_hybrid", config_cls=BailingMoeV3Config, auto_class=AutoModelForCausalLM
)
class TurboBailingMoeV3ForCausalLM(BailingMoeV3ForCausalLM):
    config_class = BailingMoeV3Config
    base_model_prefix = "model"

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
        if sp_size <= 1 or sp_group is None:
            return
        if self.config.num_nextn_predict_layers > 0:
            raise NotImplementedError(
                "BailingMoeV3 sequence parallelism does not support MTP layers yet."
            )
        if self.config._attn_implementation != "flash_attention_2":
            raise NotImplementedError(
                "BailingMoeV3 sequence parallelism requires flash_attention_2 "
                "for MLA layers. Pass `--flash_attn fa2` to enable it."
            )
        from . import modeling_bailing_moe_v3 as bailing_moe_v3_model

        self._patch_mla_attention_interface_for_sp(
            bailing_moe_v3_model.BailingMoeV3MultiLatentAttention,
            sp_group,
            sp_size,
            sp_rank,
        )
        patch_kda_attention_forward_for_sp(
            bailing_moe_v3_model.BailingMoeV3KimiDeltaAttention,
            sp_group,
            sp_size,
            sp_rank,
        )

    @staticmethod
    def _patch_mla_attention_interface_for_sp(attn_cls, sp_group, sp_size, sp_rank):
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
