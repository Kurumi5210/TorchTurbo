from torch_turbo.distributed.parallel_plan import (
    SPPlan,
    EPPlan,
    FullyShardPlan
)


def get_parallel_plan():
    sp_plan = SPPlan(['model.layers.*.self_attn'])
    ep_plan = EPPlan(['model.layers.*.mlp'])
    fsdp_plan = FullyShardPlan({
        'model.layers.*.mlp.experts': 'edp',
        'model.layers.\d+': 'dp',
        'model.embed_tokens': 'dp',
        'lm_head': 'dp'
    })
    return (sp_plan, ep_plan, fsdp_plan)