def fix_rope_config_for_bailing(config):
    rotary_dim = getattr(config, "rotary_dim", None)
    head_dim = getattr(config, "head_dim", None)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    assert head_dim != None, "must be"
    assert rotary_dim != None or partial_rotary_factor != None, "must be"
    if partial_rotary_factor == None:
        # Ring-1T does not set partial_rotary_factor
        partial_rotary_factor = rotary_dim / head_dim
        config.partial_rotary_factor = partial_rotary_factor