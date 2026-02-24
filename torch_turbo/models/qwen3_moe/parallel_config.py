from torch_turbo.distributed.parallel_config import (
    SPConfig,
    EPConfig,
    FullyShardConfig
)


def get_parallel_config():
    sp_plan = SPConfig(['self_attn'])
    ep_plan = EPConfig(['mlp'])
    fsdp_plan = FullyShardConfig({
        'mlp': 'edp',
        'word_embeddings': 'dp',
        'layer': 'dp',
        'lm_head': 'dp'
    })
    return (sp_plan, ep_plan, fsdp_plan)