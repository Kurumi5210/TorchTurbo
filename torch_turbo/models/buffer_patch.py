import torch


def _fix_rope_config_for_bailing(config):
    rotary_dim = getattr(config, "rotary_dim", None)
    head_dim = getattr(config, "head_dim", None)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    assert head_dim != None, "must be"
    assert rotary_dim != None or partial_rotary_factor != None, "must be"
    if partial_rotary_factor == None:
        # Ring-1T does not set partial_rotary_factor
        partial_rotary_factor = rotary_dim / head_dim
        config.partial_rotary_factor = partial_rotary_factor


def register_buffer_for_olmo3(model, device):
    print("Register buffer for Olmo3...")
    # patch to rotary_emb
    rotary_emb = model.model.rotary_embs
    inv_freq, _ = rotary_emb.sliding_attention.rope_init_fn(model.config, device)
    rotary_emb.sliding_attention.register_buffer("inv_freq", inv_freq, persistent=False)
    inv_freq, _ = rotary_emb.full_attention.rope_init_fn(model.config, device)
    rotary_emb.full_attention.register_buffer("inv_freq", inv_freq, persistent=False)


def register_buffer_for_qwen3_moe(model, device):
    print("Register buffer for Qwen3MoE...")
    # patch to rotary_emb
    rotary_emb = model.model.rotary_emb
    inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device)
    rotary_emb.register_buffer("inv_freq", inv_freq.to(dtype=torch.bfloat16), persistent=False)


def register_buffer_for_bailing_v2(model, device):
    # patch to rotary_emb
    rotary_emb = model.model.rotary_emb
    _fix_rope_config_for_bailing(rotary_emb.config)
    inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device)
    rotary_emb.register_buffer("inv_freq", inv_freq.to(dtype=torch.bfloat16), persistent=False)


def register_buffer_for_bailing_v2_5(model, device):
    # patch to rotary_emb
    rotary_emb = model.model.rotary_emb
    _fix_rope_config_for_bailing(rotary_emb.config)
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


def register_buffer_for_kimi_k2(model, device):
    for layer in model.model.layers:
        attn_layer = layer.self_attn
        rotary_emb = attn_layer.rotary_emb

        init_kwargs = {
            'dim': rotary_emb.dim,
            'max_position_embeddings': rotary_emb.max_position_embeddings,
            'base': rotary_emb.base,
            'device': device,
        }

        if hasattr(rotary_emb, 'scaling_factor'):
            init_kwargs['scaling_factor'] = rotary_emb.scaling_factor
        if hasattr(rotary_emb, 'original_max_position_embeddings'):
            init_kwargs['original_max_position_embeddings'] = rotary_emb.original_max_position_embeddings
        if hasattr(rotary_emb, 'beta_fast'):
            init_kwargs['beta_fast'] = rotary_emb.beta_fast
        if hasattr(rotary_emb, 'beta_slow'):
            init_kwargs['beta_slow'] = rotary_emb.beta_slow
        if hasattr(rotary_emb, 'mscale'):
            init_kwargs['mscale'] = rotary_emb.mscale
        if hasattr(rotary_emb, 'mscale_all_dim'):
            init_kwargs['mscale_all_dim'] = rotary_emb.mscale_all_dim

        rotary_emb.__init__(**init_kwargs)

def register_buffer_for_kimi_k25_vision(vision_tower, device):
    from .kimi_k25.modeling_kimi_k25 import get_1d_sincos_pos_embed
    pos_emb = vision_tower.patch_embed.pos_emb
    time_weight = torch.from_numpy(
        get_1d_sincos_pos_embed(pos_emb.dim, pos_emb.num_frames)
    ).float().unsqueeze(1)
    pos_emb.register_buffer('time_weight', time_weight, persistent=False)


def register_buffer_for_kimi_k25(model, device):
    # deal with vision tower
    register_buffer_for_kimi_k25_vision(model.vision_tower, device)
    # deal with language_model
    register_buffer_for_kimi_k2(model.language_model, device)