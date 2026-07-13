import os
import functools
import inspect

import torch

def io_hook_wrapper(io_hooker, hook_name, original_method):
    func = original_method.__func__ if hasattr(original_method, '__func__') else original_method
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            param_values = [bound.arguments[name] for name in list(sig.parameters)]
            if not isinstance(param_values[0], torch.Tensor):
                param_values = param_values[1:]
            input_tuple = tuple(param_values)
        except TypeError as e:
            print(f'[HOOK] Warning: signature bind failure! {e}')
            input_tuple = args

        try:
            output = original_method(*args, **kwargs)
        except Exception as e:
            output = None
            raise

        output_tuple = (output,) if not isinstance(output, tuple) else output
        io_hooker.append(hook_name, (input_tuple, output_tuple))

        return output

    if isinstance(original_method, staticmethod):
        return staticmethod(wrapper)
    if isinstance(original_method, classmethod):
        return classmethod(wrapper)
    return wrapper

class InputOutputHooker():
    def __init__(self, model, dump_name, num_record_forward = 1):
        self.num_record_forward = num_record_forward
        self.model = model
        self.model_inputs_outputs = []
        self.dump_name = dump_name
        self.counter = 0
        self.hook_func_counter = 0

    def append(self, name, content):
        if 'compute_index_scores' in name or \
            'rotate_activation' in name or \
            'apply_rotary_emb' in name:
            name = f'{self.hook_func_counter}_{name}'
            self.hook_func_counter += 1
        self.model_inputs_outputs.append({name: content})

    def record(self, module, input, output):
        if self.counter >= self.num_record_forward:
            return
        input  = (input,)  if isinstance(input,  torch.Tensor) else input
        output = (output,) if isinstance(output, torch.Tensor) else output
        assert isinstance(input, tuple), 'input must be tuple'
        assert isinstance(output, tuple), f'output must be tuple, {output.dtype} {output}'
        if "input_layernorm" in module.record_name or "q_a_layernorm" in module.record_name:
            self.model_inputs_outputs.append({module.record_name: (input, output, (module.weight.cpu(),))})
        else:
            self.model_inputs_outputs.append({module.record_name: (input, output)})

    def __enter__(self):
        if self.counter == 0:
            i = 0
            if hasattr(self.model, "model"):
                layers = self.model.model.layers
            else:
                layers = self.model.layers
            for layer in layers:
                layer.register_forward_hook(self.record)
                layer.record_name = f"layer.{i}"

                if hasattr(layer, "attn_norm"):
                    layer.attn_norm.register_forward_hook(self.record)
                    layer.attn_norm.record_name = f"layer.{i}.input_layernorm"
                if hasattr(layer, "input_layernorm"):
                    layer.input_layernorm.register_forward_hook(self.record)
                    layer.input_layernorm.record_name = f"layer.{i}.input_layernorm"

                if hasattr(layer, "attn"):
                    layer.attn.register_forward_hook(self.record)
                    layer.attn.record_name = f"layer.{i}.attn"
                    layer.attn.wq_a.register_forward_hook(self.record)
                    layer.attn.wq_a.record_name = f"layer.{i}.attn.q_a_proj"
                    layer.attn.q_norm.register_forward_hook(self.record)
                    layer.attn.q_norm.record_name = f"layer.{i}.attn.q_a_layernorm"
                    attn = layer.attn
                if hasattr(layer, "self_attn"):
                    layer.self_attn.register_forward_hook(self.record)
                    layer.self_attn.record_name = f"layer.{i}.attn"
                    #  qr = self.q_a_layernorm(self.q_a_proj(hidden_states))
                    layer.self_attn.q_a_proj.register_forward_hook(self.record)
                    layer.self_attn.q_a_proj.record_name = f"layer.{i}.attn.q_a_proj"
                    layer.self_attn.q_a_layernorm.register_forward_hook(self.record)
                    layer.self_attn.q_a_layernorm.record_name = f"layer.{i}.attn.q_a_layernorm"
                    attn = layer.self_attn

                if hasattr(attn, "indexer"):
                    attn.indexer.register_forward_hook(self.record)
                    attn.indexer.record_name = f"layer.{i}.attn.indexer"

                if hasattr(layer, "ffn_norm"):
                    layer.ffn_norm.register_forward_hook(self.record)
                    layer.ffn_norm.record_name = f"layer.{i}.post_attention_layernorm"
                if hasattr(layer, "post_attention_layernorm"):
                    layer.post_attention_layernorm.register_forward_hook(self.record)
                    layer.post_attention_layernorm.record_name = f"layer.{i}.post_attention_layernorm"

                if hasattr(layer, "ffn"):
                    layer.ffn.register_forward_hook(self.record)
                    layer.ffn.record_name = f"layer.{i}.mlp"
                if hasattr(layer, "mlp"):
                    layer.mlp.register_forward_hook(self.record)
                    layer.mlp.record_name = f"layer.{i}.mlp"

                i += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.counter += 1
        if self.counter == self.num_record_forward:
            self.dump()

    def dump(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            torch.save(self.model_inputs_outputs, self.dump_name)
            self.model_inputs_outputs = []
            print(f'[HOOK DUMP] dump to {self.dump_name}', flush=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
