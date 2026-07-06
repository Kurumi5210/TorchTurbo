from typing import Any

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # Older versions

import torch

from torchturbo.ops import npu_fused_ops
from torchturbo.utils.device import is_npu_available, get_device_name
from .modeling_kimi_k25 import KimiK25ForConditionalGeneration
from .configuration_kimi_k25 import KimiK25Config
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.models.modules.attention.ulysses import (
    patch_vlm_for_ulysses_input_slicing,
)

from ..registry import register_model
from ..weight_converter import BaseWeightConverter, WeightConverterMixin
from transformers import AutoModelForCausalLM


class KimiK25WeightConverter(BaseWeightConverter):
    """Weight converter for Kimi-K2.5.

    Kimi-K2.5 is a VLM whose real transformer dims live under
    ``config.text_config`` (DeepseekV3Config), not on the top-level
    ``KimiK25Config`` wrapper. BaseWeightConverter reads ``hidden_size`` /
    ``moe_intermediate_size`` / ``model_type`` / dtype off ``self.config``,
    so we re-point it at ``text_config``.

    The language model is DeepSeek-V3 with per-expert checkpoint format
    (``language_model.model.layers.N.mlp.experts.{i}.gate_proj/up_proj/
    down_proj.weight``), so the BaseWeightConverter defaults for the
    expert (``gate_proj``/``up_proj``/``down_proj``) and router
    (``router`` -> ``gate``, ``gate.e_score_correction_bias``) names
    already match; no name overrides are needed.
    """

    def __init__(self, config, moe_config, mesh):
        text_config = config.text_config if hasattr(config, "text_config") else config
        super().__init__(text_config, moe_config, mesh)
        self.full_config = config


@register_model("kimi_k25", config_cls=KimiK25Config, auto_class=AutoModelForCausalLM)
class TurboKimiK25ForConditionalGeneration(WeightConverterMixin, KimiK25ForConditionalGeneration):

    is_vlm = True

    # ---- Config dtype set / reset (VLM override) ----

    @classmethod
    def set_config_dtype_to_float32(cls, config):
        """Also set text_config / vision_config dtype to float32.

        FSDP2 uses float32 as main param storage and requires a uniform
        param dtype at construction time. Kimi's sub-configs may still
        carry bfloat16, which would create mixed dtypes on the meta device.
        """
        state = super().set_config_dtype_to_float32(config)
        if hasattr(config, "text_config") and hasattr(config.text_config, "dtype"):
            state["text_config_dtype"] = config.text_config.dtype
            config.text_config.dtype = torch.float32
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "dtype"):
            state["vision_config_dtype"] = config.vision_config.dtype
            config.vision_config.dtype = torch.float32
        return state

    @classmethod
    def reset_config_dtype(cls, config, state):
        """Reset text_config / vision_config dtype after construction."""
        if "text_config_dtype" in state:
            config.text_config.dtype = state["text_config_dtype"]
        if "vision_config_dtype" in state:
            config.vision_config.dtype = state["vision_config_dtype"]
        super().reset_config_dtype(config, state)

    def get_weight_converter(self, mesh):
        return KimiK25WeightConverter(self.config, self.get_moe_config(), mesh)

    def __init__(self, config: KimiK25Config) -> None:
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
        from . import modeling_deepseek as ds

        if is_npu_available:
            ds.apply_rotary_pos_emb = npu_fused_ops.apply_rotary_pos_emb_kimi_moe
            ds.DeepseekV3RMSNorm.forward = npu_fused_ops.fused_rms_norm_forward_npu
            MoEAlltoAllTokenDispatcher.token_permutation = (
                npu_fused_ops.fused_token_permutation_npu
            )
            MoEAlltoAllTokenDispatcher.token_unpermutation = (
                npu_fused_ops.fused_token_unpermutation_npu
            )

        # patch moe_layer forward
        origin_moe_forward = MoELayer.forward

        def deepseek_moe_forward(self, hidden_states: torch.Tensor):
            output, _ = origin_moe_forward(self, hidden_states)
            return output

        MoELayer.forward = deepseek_moe_forward

    def _post_init_patches(self):
        pass

    def setup_sp(self, sp_info):
        from ..modules.attention.ulysses import patch_forward_for_sp_input_slicing

        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        cls = type(self.language_model)
        patch_forward_for_sp_input_slicing(cls, sp_group, sp_size)

    def post_weight_load(self):
        """Initialize runtime buffers after weights are on real device."""

        device = get_device_name()
        self._register_buffer_for_kimi_k25(device)

    def _register_buffer_for_kimi_k25(self, device):
        # deal with vision tower
        self._register_buffer_for_kimi_k25_vision(self.vision_tower, device)
        # deal with language_model
        self._register_buffer_for_kimi_k2(self.language_model, device)

    def _register_buffer_for_kimi_k2(self, model, device):
        for layer in model.model.layers:
            attn_layer = layer.self_attn
            rotary_emb = attn_layer.rotary_emb

            init_kwargs = {
                "dim": rotary_emb.dim,
                "max_position_embeddings": rotary_emb.max_position_embeddings,
                "base": rotary_emb.base,
                "device": device,
            }

            if hasattr(rotary_emb, "scaling_factor"):
                init_kwargs["scaling_factor"] = rotary_emb.scaling_factor
            if hasattr(rotary_emb, "original_max_position_embeddings"):
                init_kwargs["original_max_position_embeddings"] = (
                    rotary_emb.original_max_position_embeddings
                )
            if hasattr(rotary_emb, "beta_fast"):
                init_kwargs["beta_fast"] = rotary_emb.beta_fast
            if hasattr(rotary_emb, "beta_slow"):
                init_kwargs["beta_slow"] = rotary_emb.beta_slow
            if hasattr(rotary_emb, "mscale"):
                init_kwargs["mscale"] = rotary_emb.mscale
            if hasattr(rotary_emb, "mscale_all_dim"):
                init_kwargs["mscale_all_dim"] = rotary_emb.mscale_all_dim

            rotary_emb.__init__(**init_kwargs)

    def _register_buffer_for_kimi_k25_vision(self, vision_tower, device):
        from .modeling_kimi_k25 import get_1d_sincos_pos_embed

        pos_emb = vision_tower.patch_embed.pos_emb
        time_weight = (
            torch.from_numpy(get_1d_sincos_pos_embed(pos_emb.dim, pos_emb.num_frames))
            .float()
            .unsqueeze(1)
        )
        pos_emb.register_buffer("time_weight", time_weight, persistent=False)

    @override
    def _merge_input_ids_with_image_features(
        self,
        image_features: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        flat_image_features = torch.cat(image_features, dim=0)

        image_token_index: int = self.config.media_placeholder_token_id
        ignore_index: int = self.config.ignore_index

        mask = input_ids == image_token_index
        num_placeholders = mask.sum().item()
        num_features = flat_image_features.shape[0]

        if num_placeholders != num_features:
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {num_placeholders} while"
                f" the number of image features given to the model is {num_features}. "
                "This prevents correct indexing and breaks batch generation."
            )

        final_embedding = inputs_embeds.clone()
        final_embedding[mask] = flat_image_features.to(
            final_embedding.device, dtype=final_embedding.dtype
        )

        final_labels = labels
        if labels is not None:
            final_labels = labels.clone()
            final_labels[mask] = ignore_index

        return final_embedding, attention_mask, final_labels
