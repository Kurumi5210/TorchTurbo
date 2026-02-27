from torch_turbo.distributed.parallel_plan import (
    SPPlan,
    EPPlan,
    FullyShardPlan
)


def get_parallel_plan():
    sp_plan = SPPlan(['attention'])
    ep_plan = EPPlan(['mlp'])
    fsdp_plan = FullyShardPlan({
        'mlp.experts': 'edp',
        'word_embeddings': 'dp',
        'lm_head': 'dp'
    })
    return (sp_plan, ep_plan, fsdp_plan)