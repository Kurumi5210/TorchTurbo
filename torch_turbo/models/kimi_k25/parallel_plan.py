from torch_turbo.distributed.parallel_plan import (
    SPPlan,
    EPPlan,
    FullyShardPlan
)

def get_parallel_plan():
    sp_plan = SPPlan(['language_model.model.layers.*.self_attn'])
    ep_plan = EPPlan(['language_model.model.layers.*.mlp'])
    fsdp_plan = FullyShardPlan({
        'vision_tower.encoder.blocks.\d+': 'dp',
        'vision_tower': 'dp',
        'mm_projector': 'dp',
        'language_model.model.layers.*.mlp.experts': 'edp',
        'language_model.model.layers.\d+': 'dp',
        'language_model.model.embed_tokens': 'dp',
        'language_model.lm_head': 'dp',
        'language_model': 'dp',
    })
    return (sp_plan, ep_plan, fsdp_plan)
