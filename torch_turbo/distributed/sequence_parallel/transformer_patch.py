import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe_module


def create_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids):
    # do not split attention mask into chunks under ulyssess sequence parallel's condition
    return attention_mask

if hasattr(qwen3_moe_module, 'create_causal_mask'):
    qwen3_moe_module.create_causal_mask = create_causal_mask
    print(f"Monkey patch create_causal_mask in Qwen3MoeModel")