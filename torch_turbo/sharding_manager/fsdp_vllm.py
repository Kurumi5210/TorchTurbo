import re
import torch

from torch_turbo.distributed.comm import all_gather_tensor


def convert_param_name(key, reverse_key_mapping):
    for pattern, replacement in reverse_key_mapping.items():
        replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
        replacement = re.sub(r"\(.*\)", "", replacement)
        key, n_replace = re.subn(pattern, replacement, key)
        # Early exit of the loop
        if n_replace > 0:
            return key
    return key


def per_tensor_generate(model):
    """ per-tensor generator
    """
    if hasattr(model, "_checkpoint_conversion_mapping"):
        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
    else:
        reverse_key_mapping = {}

    for name, param in model.named_parameters():
        if "mlp.experts" in name:
            prefix = name[:-8]
            num_experts = model.config.num_experts
            full_tensor = param.full_tensor()
            if "weight1" in name:
                full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                experts = torch.split(full_tensor, model.config.hidden_size)
                for i in range(num_experts):
                    gate_proj = f"{prefix}.{i}.gate_proj.weight"
                    up_proj = f"{prefix}.{i}.up_proj.weight"
                    expert2 = torch.split(experts[i].t(), model.config.moe_intermediate_size)
                    yield from zip([convert_param_name(gate_proj, reverse_key_mapping)], [expert2[0]])
                    yield from zip([convert_param_name(up_proj, reverse_key_mapping)], [expert2[1]])
            else:
                # weight2
                full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                experts = torch.split(full_tensor, model.config.moe_intermediate_size)
                for i in range(num_experts):
                    down_proj = f"{prefix}.{i}.down_proj.weight"
                    yield from zip([convert_param_name(down_proj, reverse_key_mapping)], [experts[i].t()])

        elif "mlp.router" in name:
            new_name = name.replace("router", "gate")
            yield from zip([convert_param_name(new_name, reverse_key_mapping)], [param.full_tensor()])
        else:
            yield from zip([convert_param_name(name, reverse_key_mapping)], [param.full_tensor()])

def convert_param_name(key, reverse_key_mapping):
    for pattern, replacement in reverse_key_mapping.items():
        replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
        replacement = re.sub(r"\(.*\)", "", replacement)
        key, n_replace = re.subn(pattern, replacement, key)
        # Early exit of the loop
        if n_replace > 0:
            return key
    return key