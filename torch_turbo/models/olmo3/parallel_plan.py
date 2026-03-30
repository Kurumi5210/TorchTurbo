from torch_turbo.distributed.parallel_plan import (
    SPPlan,
    FullyShardPlan
)


def get_parallel_plan():
    sp_plan = SPPlan(['model.layers.*.self_attn'])
    fsdp_plan = FullyShardPlan({
        'model.layers.*.mlp': 'dp', 
        'model.layers.*.self_attn': 'dp',
        'model.layers.*.post_attention_layernorm': 'dp',
        'model.layers.*.post_feedforward_layernorm': 'dp',
        'model.embed_tokens': 'dp',
        'lm_head': 'dp'
    })
    return (sp_plan, None, fsdp_plan)