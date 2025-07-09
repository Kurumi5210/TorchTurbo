from types import SimpleNamespace

import mindspeed.megatron_adaptor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
import megatron.core.parallel_state as ps
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args

class MegatronUtils:
    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        ps.initialize_model_parallel(
            tensor_model_parallel_size = tensor_model_parallel_size,
            pipeline_model_parallel_size = pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size,
            expert_model_parallel_size = expert_model_parallel_size,
            **kwargs,
        )
        
    @staticmethod 
    def initialize_moe_layer_from_hf_config(config, **config_kwargs):
        num_moe_experts = config.num_experts
        expert_model_parallel_size = ps.get_expert_model_parallel_world_size()
        num_local_experts = num_moe_experts // expert_model_parallel_size
        default_config_kwargs = dict(
            num_layers=1,
            expert_model_parallel_size=expert_model_parallel_size,
            moe_ffn_hidden_size=config.moe_intermediate_size,
            moe_token_dispatcher_type="alltoall",
            gated_linear_unit=True,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_moe_experts=num_moe_experts,
            moe_router_topk=config.num_experts_per_tok,
            use_cpu_initialization=True,
            fp8=None,
            add_bias_linear=False,
        )
        default_config_kwargs.update(**config_kwargs)
        set_args(SimpleNamespace(**default_config_kwargs))
        tf_config = TransformerConfig(**default_config_kwargs)
        tf_config.moe_grouped_gemm = True
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_moe_experts,
            moe_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=True,
        )
        grouped_mlp = MoELayer(tf_config, transformer_layer_spec.submodules.mlp.submodules)

        return grouped_mlp