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

        dim = rotary_emb.dim
        base = rotary_emb.base
        max_position_embeddings = rotary_emb.max_position_embeddings
        
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        )

        inv_freq = inv_freq.to(dtype=torch.bfloat16)

        rotary_emb.inv_freq = inv_freq
        rotary_emb.register_buffer(
            "inv_freq", 
            rotary_emb.inv_freq, 
            persistent=False
        )

        rotary_emb._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype(),
        )

def register_buffer_for_kimi_k25_vision(vision_tower, device):
    from .kimi_k25.modeling_kimi_k25 import get_1d_sincos_pos_embed
    pos_emb = vision_tower.patch_embed.pos_emb
    pos_emb.time_weight = torch.from_numpy(
        get_1d_sincos_pos_embed(pos_emb.dim, pos_emb.num_frames)).float().unsqueeze(1)

    rope_2d = vision_tower.encoder.rope_2d
    rope_2d.freqs_cis = rope_2d._precompute_freqs_cis(device)

def register_buffer_for_kimi_k25(model, device):
    # deal with vision tower
    register_buffer_for_kimi_k25_vision(model.vision_tower, device)
    # deal with language_model
    register_buffer_for_kimi_k2(model.language_model, device)