import importlib
from typing import Callable, List, MutableMapping, Optional, Type, Union
from transformers import AutoConfig, AutoModelForCausalLM

from .config_converter import (
    hf_to_moe_config_qwen3,
    hf_to_moe_config_bailing
)
from .buffer_patch import (
    register_buffer_for_base_model,
    register_buffer_for_bailing_v2_5
)
from torch_turbo.distributed.parallel_config import ParallelPlanConfig


_SUPPORT_MODELS = {
    "qwen3_moe": "qwen3_moe",
    "bailing_moe": "bailing_moe_v2",
    "bailing_moe_linear": "bailing_moe_v2_5"
}

_CONFIG_CONVERTER_REGISTRY = {
    "qwen3_moe": hf_to_moe_config_qwen3,
    "bailing_moe": hf_to_moe_config_bailing,
    "bailing_moe_linear": hf_to_moe_config_bailing
}

_BUFFER_REGISTRY = {
    "qwen3_moe": register_buffer_for_base_model,
    "bailing_moe": register_buffer_for_base_model,
    "bailing_moe_linear": register_buffer_for_bailing_v2_5
}

class Registry(MutableMapping):
    def __init__(self): 
        self._model_mapping = dict()

    def __getitem__(self, key):
        return self._model_mapping[key]
    
    def __setitem__(self, key, value):
        self._model_mapping[key] = value

    def __delitem__(self, key):
        del self._model_mapping[key]

    def __iter__(self):
        return iter(self._model_mapping)

    def __len__(self):
        return len(self._model_mapping)
    
    def valid_keys(self) -> List[str]:
        return list(self.keys())
        
    def register(self, key: str, model_cls: Optional[Union[Type, Callable]] = None):
        if model_cls is not None:
            self._model_mapping[key] = model_cls
            return model_cls

        def decorator(model_cls):
            if key in self._model_mapping:
                raise ValueError(
                    f"{key} is already registered. Cannot register duplicate models."
                )
            self._model_mapping.update({key: model_cls})
            return model_cls

        return decorator


MODELING_REGISTRY = Registry()
PARALLEL_CONFIG_REGISTRY = Registry()


def get_model_config(args):
    config_path = args.path
    return AutoConfig.from_pretrained(config_path, trust_remote_code=True)

def get_model_cls(model_config):
    model_type = model_config.model_type
    print(f"model_type>>>>>>>>{model_type}")
    module_name = _SUPPORT_MODELS[model_type]
    importlib.import_module(f'torch_turbo.models.{module_name}')
    print(f"get_model_cls>>>>>>>>>{MODELING_REGISTRY[model_type]()}")
    # if model_type not in MODELING_REGISTRY.valid_keys():
    #     return AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    return MODELING_REGISTRY[model_type]()

def get_parallel_plan(model_config):
    model_type = model_config.model_type
    moe_config = _CONFIG_CONVERTER_REGISTRY[model_type](model_config)
    print(f"moe_config>>>>>>>>>>>{moe_config}")
    print(f"get_parallel_plan>>>>>>>>>{PARALLEL_CONFIG_REGISTRY[model_type]()}")
    sp_config, ep_config, fsdp_config = PARALLEL_CONFIG_REGISTRY[model_type]()
    parallel_plan = ParallelPlanConfig(moe_config, sp_config, ep_config, fsdp_config)
    return parallel_plan

def patch_model_buffer(model_config, model, device):
    model_type = model_config.model_type
    _BUFFER_REGISTRY[model_type](model, device)