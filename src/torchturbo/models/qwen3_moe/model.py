import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeConfig,
)

from torchturbo.ops import npu_fused_ops
from torchturbo.utils.device import is_npu_available, get_device_name
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.models.modules.attention.ulysses.utils import create_causal_mask
from ..registry import register_model
from transformers import AutoModelForCausalLM


class TurboQwen3MoeRotaryEmbedding(Qwen3MoeRotaryEmbedding):
    def __init__(self, config, device=None):
        super().__init__(
            config,
            device,
        )

        inv_freq, attention_scaling = self.rope_init_fn(config, device)
        self.attention_scaling = attention_scaling
        self.register_buffer(
            "inv_freq", inv_freq.to(dtype=torch.bfloat16), persistent=False
        )


@register_model("qwen3_moe", config_cls=Qwen3MoeConfig, auto_class=AutoModelForCausalLM)
class TurboQwen3MoeForCausalLM(Qwen3MoeForCausalLM):

    config_class = Qwen3MoeConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3MoeConfig) -> None:
        self._pre_init_patches()
        super().__init__(config)
        self._post_init_patches()

    @classmethod
    def get_parallel_plan(cls, **kwargs):
        """Return the default parallel plan for this model."""
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    @staticmethod
    def _pre_init_patches():
        import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe_model

        if is_npu_available:
            qwen3_moe_model.apply_rotary_pos_emb = (
                npu_fused_ops.apply_rotary_pos_emb_npu
            )
            qwen3_moe_model.Qwen3MoeRMSNorm.forward = (
                npu_fused_ops.fused_rms_norm_forward_npu
            )
            MoEAlltoAllTokenDispatcher.token_permutation = (
                npu_fused_ops.fused_token_permutation_npu
            )
            MoEAlltoAllTokenDispatcher.token_unpermutation = (
                npu_fused_ops.fused_token_unpermutation_npu
            )

        qwen3_moe_model.create_causal_mask = create_causal_mask

        # qwen3_moe_model.Qwen3MoeRotaryEmbedding = TurboQwen3MoeRotaryEmbedding

        # TODO(zlj)
        # parallel_state = get_parallel_state()
        # if parallel_state.sp_enabled():
        #     qwen3_moe_model.create_causal_mask = create_causal_mask
        # qwen3_moe_model.create_causal_mask = create_causal_mask

    def _post_init_patches(self):
        pass

    def post_weight_load(self):
        """Initialize runtime buffers after weights are on real device."""

        # patch to rotary_emb
        rotary_emb = self.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
            rotary_emb.config, get_device_name()
        )
        rotary_emb.register_buffer(
            "inv_freq", inv_freq.to(dtype=torch.bfloat16), persistent=False
        )
