from torch_turbo.moe.megatron_utils import MegatronUtils

class MoeUtils:
    @staticmethod
    def transform_moe_layer(model, hf_config):
        assert "moe" in hf_config.model_type, f'{hf_config.model_type} must be moe model'
        for layer in model.layers:
            moe = MegatronUtils.initialize_moe_layer_from_hf_config(hf_config)
            layer.mlp = moe