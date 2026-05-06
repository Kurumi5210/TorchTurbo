from torch_turbo.distributed.parallel_plan import (
    SPPlan,
    EPPlan,
    FullyShardPlan
)

def get_parallel_plan():
    sp_plan = SPPlan(['layers.*.attn'])
    ep_plan = EPPlan(['layers.*.ffn'])
    fsdp_plan = FullyShardPlan({
        'layers.*.ffn.experts': 'edp',
        'layers.\d+': 'dp',
        'embed': 'dp',
        'head': 'dp'
    },
    {
        'param_dtype': 'bf16',
        'reduce_dtype': 'bf16',
    })
    return (sp_plan, ep_plan, fsdp_plan)