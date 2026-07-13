#!/usr/bin/env python3
"""
Modified from Xiyun's original code.

io_hooker.py - Input/Output Hook System.

Purpose: Capture and record layer-wise input/output tensors for debugging and alignment.
Supports two modes:
1. Universal hook wrapper (io_hook_wrapper): wrap any function.
2. Context manager (InputOutputHooker): auto-register forward hooks for all layers.
"""

import os
import functools
import inspect
import torch
import torch.distributed as dist

def io_hook_wrapper(io_hooker, hook_name, original_method):
    """
    Universal hook wrapper.

    Wraps any function/method and records its inputs/outputs at call time.
    Supports static, class and instance methods.

    Args:
        io_hooker: hook manager instance.
        hook_name: string tag for the recorded data.
        original_method: the original function/method to wrap.

    Returns:
        Wrapped function/method.
    """
    func = original_method.__func__ if hasattr(original_method, '__func__') else original_method
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            print(f"[HOOK] Entering: {hook_name}")
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


class InputOutputHooker:
    """
    Input/Output Hook Manager.

    Acts as a context manager that automatically registers forward hooks
    for all model layers and records a specified number of forward passes.
    """
    def __init__(self, model, dump_name, num_record_forward=1):
        """
        Initialize hook manager.

        Args:
            model: target model to monitor.
            dump_name: file path for saving data.
            num_record_forward: how many forward passes to record.
        """
        self.num_record_forward = num_record_forward
        self.model = model
        self.model_inputs_outputs = []
        self.dump_name = dump_name
        self.counter = 0
        self.hook_func_counter = 0

    def append(self, name, content):
        """
        Append recorded data.

        Args:
            name: data tag.
            content: tuple of (inputs, outputs).
        """
        if "i dont know how to get func hook" in name:
            name = f'{self.hook_func_counter}_{name}'
            self.hook_func_counter += 1
        self.model_inputs_outputs.append({name: content})

    def record(self, module, input, output):
        """
        Forward hook callback.

        Args:
            module: current module.
            input: module input.
            output: module output.
        """
        if self.counter >= self.num_record_forward:
            return

        input = (input,) if isinstance(input, torch.Tensor) else input
        output = (output,) if isinstance(output, torch.Tensor) else output

        assert isinstance(input, tuple), 'input must be tuple'
        assert isinstance(output, tuple), f'output must be tuple, {output.dtype} {output}'

        if "input_layernorm" in module.record_name or "post_attention_layernorm" in module.record_name:
            self.append(module.record_name, (input, output, (module.weight.cpu(),)))
        else:
            self.append(module.record_name, (input, output))


    def __enter__(self):
        """
        Context manager entry.

        Register forward hooks for all target layers.
        """
        if self.counter != 0:
            return self

        layers = getattr(self.model, "model", self.model).layers
        for layer_idx, layer in enumerate(layers):
            layer.register_forward_hook(self.record)
            layer.record_name = f"layer.{layer_idx}"
            
            # Only adapt for Ling_mini_v2_5_fla model
            
            # LayerNorm
            if hasattr(layer, "input_layernorm"):
                layer.input_layernorm.register_forward_hook(self.record)
                layer.input_layernorm.record_name = f"layer.{layer_idx}.input_layernorm"

            if hasattr(layer, "post_attention_layernorm"):
                layer.post_attention_layernorm.register_forward_hook(self.record)
                layer.post_attention_layernorm.record_name = f"layer.{layer_idx}.post_attention_layernorm"

            # Attention
            attn = getattr(layer, "attention", None)
            if attn:
                attn.register_forward_hook(self.record)
                attn.record_name = f"layer.{layer_idx}.attn"

                for name in ("query_key_value", "query_layernorm", "key_layernorm", "rotary_emb",
                             "dense", "g_proj", "g_norm", "q_proj", "kv_a_proj_with_mqa",
                             "kv_a_layernorm", "kv_b_proj"):
                    mod = getattr(attn, name, None)
                    if mod:
                        mod.register_forward_hook(self.record)
                        mod.record_name = f"layer.{layer_idx}.attn.{name}"

            # MLP
            mlp = getattr(layer, "mlp", None)
            if mlp:
                mlp.register_forward_hook(self.record)
                mlp.record_name = f"layer.{layer_idx}.mlp"

                for name in ("router", "experts", "shared_experts"):
                    mod = getattr(mlp, name, None)
                    if mod:
                        mod.register_forward_hook(self.record)
                        mod.record_name = f"layer.{layer_idx}.mlp.{name}"
                        if name == "shared_experts":
                            for sub in ("gate_proj", "up_proj", "down_proj", "act_fn"):
                                sub_mod = getattr(mod, sub, None)
                                if sub_mod:
                                    sub_mod.register_forward_hook(self.record)
                                    sub_mod.record_name = f"layer.{layer_idx}.mlp.shared_experts.{sub}"

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit.

        Increment counter and dump when limit is reached.
        """
        self.counter += 1
        if self.counter == self.num_record_forward:
            self.dump()

    def dump(self):
        """
        Save recorded data to file.

        Only rank 0 saves in distributed mode.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        os.makedirs(os.path.dirname(self.dump_name), exist_ok=True)

        if rank == 0:
            torch.save(self.model_inputs_outputs, self.dump_name)
            self.model_inputs_outputs.clear()
            print(f'[HOOK] dump to {self.dump_name}', flush=True)

        if dist.is_initialized():
            dist.barrier()