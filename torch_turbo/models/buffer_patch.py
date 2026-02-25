import torch


def register_buffer_for_base_model(model, device):
    rotary_emb = model.model.rotary_emb
    inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device)
    rotary_emb.register_buffer("inv_freq", inv_freq.to(dtype=torch.bfloat16), persistent=False)


def register_buffer_for_bailing_v2_5(model, device):
    rotary_emb = model.model.rotary_emb
    inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device)
    rotary_emb.original_inv_freq = inv_freq
    rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)

    # patch to rotary_emb_mla
    rotary_emb_mla = model.model.rotary_emb_mla
    inv_freq, rotary_emb_mla.attention_scaling = rotary_emb_mla.rope_init_fn(rotary_emb_mla.config, device)
    rotary_emb_mla.original_inv_freq = inv_freq
    rotary_emb_mla.register_buffer("inv_freq", inv_freq, persistent=False)
    # patch to layer.attention
    for layer in model.model.layers:
        if layer.attention.__class__.__name__ == 'BailingMoeV2_5LinearAttention':
            slope = -layer.attention.build_slope_tensor(layer.attention.num_heads) * (
                1 - (layer.attention.layer_idx - 1) / (layer.attention.config.num_hidden_layers - 1) + 1e-5
            )
            layer.attention.register_buffer('slope', slope, persistent=False)
