import importlib
from typing import Callable, List, MutableMapping, Optional, Type, Union
from transformers import AutoConfig

from .buffer_patch import (
    register_buffer_for_qwen3_moe,
    register_buffer_for_bailing_v2,
    register_buffer_for_bailing_v2_5,
    register_buffer_for_kimi_k2
)
from torch_turbo.distributed.parallel_plan import ParallelPlan


_SUPPORT_MODELS = {
    "qwen3_moe": "qwen3_moe",
    "bailing_moe": "bailing_moe_v2",
    "bailing_moe_linear": "bailing_moe_v2_5",
    "kimi_k2": "kimi_k2"
}


_BUFFER_REGISTRY = {
    "qwen3_moe": register_buffer_for_qwen3_moe,
    "bailing_moe": register_buffer_for_bailing_v2,
    "bailing_moe_linear": register_buffer_for_bailing_v2_5,
    "kimi_k2": register_buffer_for_kimi_k2
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
        
    def register(self, key: str, cls_or_func: Optional[Union[Type, Callable]] = None):
        if cls_or_func is not None:
            self._model_mapping[key] = cls_or_func
            return cls_or_func

        def decorator(cls_or_func):
            if key in self._model_mapping:
                raise ValueError(
                    f"{key} is already registered. Cannot register duplicate models."
                )
            self._model_mapping.update({key: cls_or_func})
            return cls_or_func

        return decorator


MODELING_REGISTRY = Registry()
MOE_CONFIG_REGISTRY = Registry()
PARALLEL_PLAN_REGISTRY = Registry()


def get_model_config(args):
    config_path = args.path
    return AutoConfig.from_pretrained(config_path, trust_remote_code=True)


def get_model_cls(hf_config):
    model_type = hf_config.model_type
    return MODELING_REGISTRY[model_type](hf_config)


def get_parallel_plan(hf_config):
    model_type = hf_config.model_type
    
    module_name = _SUPPORT_MODELS[model_type]
    importlib.import_module(f'torch_turbo.models.{module_name}')

    moe_config = MOE_CONFIG_REGISTRY[model_type](hf_config)
    print(f"Loaded model's MoeConfig: {moe_config}")
    
    sp_plan, ep_plan, fsdp_plan = PARALLEL_PLAN_REGISTRY[model_type]()
    parallel_plan = ParallelPlan(moe_config, sp_plan, ep_plan, fsdp_plan)
    print(f"Loaded model's ParallelPlan: {moe_config}")
    
    return parallel_plan


def register_model_buffer(hf_config, model, device):
    model_type = hf_config.model_type
    return _BUFFER_REGISTRY[model_type](model, device)