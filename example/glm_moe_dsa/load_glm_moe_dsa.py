"""
Simple example to load GlmMoeDsa model with meta device and safetensors weights.

Usage:
    # Use torch_turbo implementation (default)
    python example/glm_moe_dsa/load_glm_moe_dsa.py --path /path/to/model/weights

    # 1. 先运行 torch_turbo 实现，dump layer I/O
    python example/glm_moe_dsa/load_glm_moe_dsa.py \
        --path /sfs_turbo/xiyun/sft/ckpt/glm5_small/ \
        --impl torch_turbo \
        --device npu \
        --dump_io /tmp/debug_io

    # 2. 再运行 transformers 实现，dump layer I/O 并对比
    python example/glm_moe_dsa/load_glm_moe_dsa.py \
        --path /sfs_turbo/xiyun/sft/ckpt/glm5_small/ \
        --impl transformers \
        --device npu \
        --dump_io /tmp/debug_io \
        --compare_io torch_turbo


The model directory should contain:
    - model.safetensors.index.json
    - model-00001-of-xxxxx.safetensors
    - config.json (optional, can use default config)
"""

import argparse
import inspect
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from transformers.utils import is_torch_npu_available

if is_torch_npu_available:
    import torch_npu.contrib.transfer_to_npu as _


def seed_all(seed=1234):
    import random
    import numpy as np
    import torch
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_torch_npu_available:
        import torch_npu
        os.environ['HCCL_DETERMINISTIC'] = str(True)
        os.environ['LCCL_DETERMINISTIC'] = str(1)
        os.environ['CLOSE_MATMUL_K_SHIFT'] = str(1)
        os.environ['ATB_MATMUL_SHUFFLE_K_ENABLE'] = "0"
        os.environ['ATB_LLM_LCOC_ENABLE'] = "0"
        torch.use_deterministic_algorithms(True)
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)



def get_model_classes(impl: str):
    """
    Get model config and class based on implementation choice.

    Args:
        impl: "torch_turbo" or "transformers"

    Returns:
        Tuple of (ConfigClass, ModelClass)
    """
    if impl == "torch_turbo":
        from torch_turbo.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaConfig,
            GlmMoeDsaForCausalLM,
        )
    elif impl == "transformers":
        import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as glm5
        from torch_turbo.distributed.sequence_parallel.utils import create_causal_mask
        glm5.create_causal_mask = create_causal_mask
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaConfig,
            GlmMoeDsaForCausalLM,
        )
    else:
        raise ValueError(f"Unknown implementation: {impl}. Use 'torch_turbo' or 'transformers'")

    return GlmMoeDsaConfig, GlmMoeDsaForCausalLM


def load_config_from_path(path: str, config_class):
    """Load config from model directory or create default config."""
    config_path = Path(path) / "config.json"
    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            config_dict = json.load(f)
        config = config_class(**config_dict)
    else:
        print("Using default config (smaller model for demo)")
        # 使用一个小型配置用于演示
        config = config_class(
            vocab_size=154880,
            hidden_size=2048,
            intermediate_size=4096,
            moe_intermediate_size=1024,
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=16,
            n_shared_experts=1,
            n_routed_experts=8,
            routed_scaling_factor=2.5,
            num_experts_per_tok=2,
        )
    config.num_hidden_layers = 4
    return config


def create_model_on_meta_device(config, model_class):
    """Create model on meta device for efficient memory usage."""
    print("Creating model on meta device...")
    with torch.device("meta"):
        model = model_class(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters:\n{model}")
    return model


def find_safetensors_index(path: Path) -> Path:
    """
    Find safetensors index file in directory.

    Searches for files ending with 'safetensors.index.json'.
    Examples:
        - model.safetensors.index.json
        - quant_model_weights.safetensors.index.json
    """
    index_files = list(path.glob("*safetensors.index.json"))
    if not index_files:
        raise FileNotFoundError(
            f"No safetensors index file found in {path}. "
            "Expected file ending with 'safetensors.index.json'"
        )
    if len(index_files) > 1:
        print(f"Warning: Multiple index files found, using {index_files[0]}")
    return index_files[0]


def load_weights_from_safetensors(model, path: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16, impl: str = "torch_turbo"):
    """
    Load weights from safetensors files based on model parameters.

    This approach iterates over model parameters and loads matching weights
    from checkpoint, which is convenient for debugging with reduced layers.

    Args:
        model: Model created on meta device
        path: Directory containing safetensors files
        device: Target device to load weights
        dtype: Target dtype for model weights
        impl: Implementation type ("torch_turbo" or "transformers")
    """
    path = Path(path)

    # Step 1: Find and load index file
    index_path = find_safetensors_index(path)

    print(f"Loading index from {index_path}")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]  # param_name -> filename

    # Step 2: 将模型移动到目标设备
    model.to_empty(device=device)
    model = model.to(dtype=dtype)

    # Step 3: 遍历模型参数来加载权重
    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())

    # 缓存已加载的文件，避免重复加载 (在 CPU 上加载)
    loaded_files = {}

    loaded_count = 0
    missing_in_ckpt = []
    shape_mismatch = []

    def get_file_data(filename):
        """Load file cache on CPU to save GPU memory."""
        if filename not in loaded_files:
            file_path = path / filename
            if not file_path.exists():
                return None
            print(f"Loading {filename}...")
            loaded_files[filename] = load_file(str(file_path), device="cpu")
        return loaded_files[filename]

    # 遍历所有模型参数
    for name, param in model_params.items():
        # 特殊处理 transformers 实现的 experts 权重
        if impl == "transformers" and ".mlp.experts.gate_up_proj" in name:
            # 将 expert.{i}.gate_proj.weight 和 expert.{i}.up_proj.weight 合并
            prefix = name.replace(".mlp.experts.gate_up_proj", ".mlp.experts")
            num_experts = param.shape[0]

            gate_weights = []
            up_weights = []

            load_failed = False
            for i in range(num_experts):
                gate_name = f"{prefix}.{i}.gate_proj.weight"
                up_name = f"{prefix}.{i}.up_proj.weight"

                if gate_name not in weight_map or up_name not in weight_map:
                    load_failed = True
                    break

                gate_sd = get_file_data(weight_map[gate_name])
                up_sd = get_file_data(weight_map[up_name])

                if gate_sd is None or up_sd is None:
                    load_failed = True
                    break

                # 在 CPU 上处理
                gate_weights.append(gate_sd[gate_name].to(dtype))
                up_weights.append(up_sd[up_name].to(dtype))

            if not load_failed:
                # 在 CPU 上合并 gate 和 up: [num_experts, 2*intermediate_dim, hidden_dim]
                combined = torch.stack([torch.cat([g, u], dim=0) for g, u in zip(gate_weights, up_weights)], dim=0)
                if combined.shape == param.shape:
                    # 最后才拷贝到目标设备
                    param.data.copy_(combined.to(device))
                    loaded_count += 1
                else:
                    shape_mismatch.append(f"{name}: model={param.shape}, combined={combined.shape}")
                # 释放 CPU 内存
                del combined, gate_weights, up_weights
            else:
                missing_in_ckpt.append(name)
            continue

        if impl == "transformers" and ".mlp.experts.down_proj" in name:
            # 将 expert.{i}.down_proj.weight 合并
            prefix = name.replace(".mlp.experts.down_proj", ".mlp.experts")
            num_experts = param.shape[0]

            down_weights = []

            load_failed = False
            for i in range(num_experts):
                down_name = f"{prefix}.{i}.down_proj.weight"

                if down_name not in weight_map:
                    load_failed = True
                    break

                down_sd = get_file_data(weight_map[down_name])

                if down_sd is None:
                    load_failed = True
                    break

                # 在 CPU 上处理
                down_weights.append(down_sd[down_name].to(dtype))

            if not load_failed:
                # 在 CPU 上合并 down: [num_experts, hidden_dim, intermediate_dim]
                combined = torch.stack(down_weights, dim=0)
                if combined.shape == param.shape:
                    # 最后才拷贝到目标设备
                    param.data.copy_(combined.to(device))
                    loaded_count += 1
                else:
                    shape_mismatch.append(f"{name}: model={param.shape}, combined={combined.shape}")
                # 释放 CPU 内存
                del combined, down_weights
            else:
                missing_in_ckpt.append(name)
            continue

        # 常规参数加载
        if name not in weight_map:
            missing_in_ckpt.append(name)
            continue

        filename = weight_map[name]
        state_dict = get_file_data(filename)

        if state_dict is None:
            continue

        if name in state_dict:
            # 在 CPU 上转换 dtype，然后拷贝到目标设备
            tensor = state_dict[name].to(dtype).to(device)
            if tensor.shape == param.shape:
                param.data.copy_(tensor)
                loaded_count += 1
            else:
                shape_mismatch.append(f"{name}: model={param.shape}, ckpt={tensor.shape}")

    # 遍历所有模型 buffers
    for name, buf in model_buffers.items():
        if name not in weight_map:
            continue

        filename = weight_map[name]
        state_dict = get_file_data(filename)

        if state_dict is None:
            continue

        if name in state_dict:
            tensor = state_dict[name].to(buf.dtype).to(device)
            if tensor.shape == buf.shape:
                buf.copy_(tensor)

    # 释放缓存
    loaded_files.clear()

    # 统计信息
    total_params = len(model_params)
    print(f"\nLoaded {loaded_count}/{total_params} parameters")

    if missing_in_ckpt:
        print(f"\n{len(missing_in_ckpt)} parameters not in checkpoint (may be fine for reduced model):")
        if len(missing_in_ckpt) <= 10:
            for name in missing_in_ckpt:
                print(f"  - {name}")

    if shape_mismatch:
        print(f"\n{len(shape_mismatch)} shape mismatches:")
        for msg in shape_mismatch:
            print(f"  - {msg}")

    return model


class LayerIOHook:
    """Hook to capture and dump layer inputs/outputs for debugging."""

    # 要 hook 的子模块名称
    SUBMODULES = [
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "mlp",
    ]

    @staticmethod
    def _tensor_summary(tensor: torch.Tensor, prefix: str = "", max_elements: int = 10) -> str:
        """Generate a summary string for a tensor."""
        lines = [f"{prefix}shape={list(tensor.shape)}, dtype={tensor.dtype}"]
        lines.append(f"{prefix}min={tensor.min().item():.6e}, max={tensor.max().item():.6e}, mean={tensor.mean().item():.6e}")

        # 打印前几个元素
        flat = tensor.flatten()
        if flat.numel() <= max_elements:
            elements = ", ".join(f"{v:.6e}" for v in flat.tolist())
        else:
            elements = ", ".join(f"{v:.6e}" for v in flat[:max_elements].tolist())
            elements += ", ..."
        lines.append(f"{prefix}[{elements}]")

        return "\n".join(lines)

    @staticmethod
    def _clean_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Replace inf/-inf/nan with 0 for comparison."""
        clean = tensor.clone()
        clean[torch.isinf(clean) | torch.isnan(clean)] = 0.0
        return clean

    def __init__(self, output_dir: str, impl: str):
        self.output_dir = Path(output_dir)
        self.impl = impl
        self.outputs_dir = self.output_dir / impl
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.hooks = []
        self.module_io = {}  # key: "layer_X.submodule", value: {"input": [...], "output": [...]}

    def _extract_tensors_from_value(self, value, name_prefix=""):
        """Extract all tensors from a value, handling nested tuples/lists."""
        tensors = []

        if isinstance(value, torch.Tensor):
            tensors.append((name_prefix, value.detach().cpu().clone()))
        elif isinstance(value, (tuple, list)):
            for i, sub_item in enumerate(value):
                sub_name = f"{name_prefix}_{i}" if name_prefix else str(i)
                tensors.extend(self._extract_tensors_from_value(sub_item, sub_name))
        elif value is None:
            # 保存 None 值用于对比
            tensors.append((name_prefix, None))
        elif isinstance(value, bool):
            # 保存 bool 值用于对比
            tensors.append((name_prefix, value))
        elif isinstance(value, (int, float)):
            # 保存数值
            tensors.append((name_prefix, value))
        elif hasattr(value, 'items'):
            # dict 类型
            for k, v in value.items():
                sub_name = f"{name_prefix}_{k}" if name_prefix else k
                tensors.extend(self._extract_tensors_from_value(v, sub_name))

        return tensors

    def _make_forward_pre_hook(self, module_name: str):
        """Create a forward pre-hook to capture input kwargs with names."""

        def pre_hook(module, args, kwargs):
            # 获取 forward 方法的签名
            sig = inspect.signature(module.forward)
            param_names = list(sig.parameters.keys())

            # 保存所有输入
            input_tensors = []

            # 处理位置参数
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                else:
                    param_name = f"arg_{i}"
                tensors = self._extract_tensors_from_value(arg, param_name)
                input_tensors.extend(tensors)

            # 处理关键字参数
            for kwarg_name, kwarg_value in kwargs.items():
                tensors = self._extract_tensors_from_value(kwarg_value, kwarg_name)
                input_tensors.extend(tensors)

            # 初始化存储（如果没有的话）
            if module_name not in self.module_io:
                self.module_io[module_name] = {}

            self.module_io[module_name]["input"] = input_tensors

        return pre_hook

    def _make_hook(self, module_name: str):
        """Create a forward hook to capture outputs."""

        def hook(module, args, output):
            # 保存所有输出张量
            output_tensors = self._extract_tensors_from_value(output, "output")

            # 确保存储存在
            if module_name not in self.module_io:
                self.module_io[module_name] = {"input": []}

            self.module_io[module_name]["output"] = output_tensors

        return hook

    def register_hooks(self, model):
        """Register forward hooks on all decoder layers and their submodules."""
        # 获取模型层
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            print("Warning: Could not find layers in model")
            return

        print(f"Registering hooks on {len(layers)} layers with submodules: {self.SUBMODULES}...")
        for layer_idx, layer in enumerate(layers):
            # Hook layer 本身
            layer_name = f"layer_{layer_idx}"
            pre_hook = layer.register_forward_pre_hook(self._make_forward_pre_hook(layer_name), with_kwargs=True)
            hook = layer.register_forward_hook(self._make_hook(layer_name))
            self.hooks.extend([pre_hook, hook])

            # Hook 子模块
            for sub_name in self.SUBMODULES:
                if hasattr(layer, sub_name):
                    sub_module = getattr(layer, sub_name)
                    sub_module_name = f"layer_{layer_idx}.{sub_name}"
                    pre_hook = sub_module.register_forward_pre_hook(self._make_forward_pre_hook(sub_module_name), with_kwargs=True)
                    hook = sub_module.register_forward_hook(self._make_hook(sub_module_name))
                    self.hooks.extend([pre_hook, hook])

            # Hook self_attn 的子模块
            if hasattr(layer, "self_attn"):
                self_attn = layer.self_attn
                attn_submodules = [
                    "q_proj", "q_a_proj", "q_a_layernorm", "q_b_proj",
                    "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj",
                    "o_proj", "indexer",
                ]
                for attn_sub_name in attn_submodules:
                    if hasattr(self_attn, attn_sub_name):
                        attn_sub_module = getattr(self_attn, attn_sub_name)
                        full_name = f"layer_{layer_idx}.self_attn.{attn_sub_name}"
                        pre_hook = attn_sub_module.register_forward_pre_hook(self._make_forward_pre_hook(full_name), with_kwargs=True)
                        hook = attn_sub_module.register_forward_hook(self._make_hook(full_name))
                        self.hooks.extend([pre_hook, hook])

                # Hook indexer 的子模块
                if hasattr(self_attn, "indexer"):
                    indexer = self_attn.indexer
                    indexer_submodules = ["wq_b", "wk", "k_norm", "weights_proj"]
                    for indexer_sub_name in indexer_submodules:
                        if hasattr(indexer, indexer_sub_name):
                            indexer_sub_module = getattr(indexer, indexer_sub_name)
                            full_name = f"layer_{layer_idx}.self_attn.indexer.{indexer_sub_name}"
                            pre_hook = indexer_sub_module.register_forward_pre_hook(self._make_forward_pre_hook(full_name), with_kwargs=True)
                            hook = indexer_sub_module.register_forward_hook(self._make_hook(full_name))
                            self.hooks.extend([pre_hook, hook])

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def save(self):
        """Save captured module I/O to safetensors files."""
        print(f"Saving module I/O to {self.outputs_dir}...")

        # 按照 layer 分组保存
        layer_data = {}
        non_tensor_values = {}  # 保存非 tensor 类型的值用于对比

        for module_name, io_dict in self.module_io.items():
            layer_key = module_name.split(".")[0]  # "layer_X" or "layer_X.sub"
            if layer_key not in layer_data:
                layer_data[layer_key] = {}
            if module_name not in non_tensor_values:
                non_tensor_values[module_name] = {"input": [], "output": []}

            # 保存输入张量
            for tensor_name, value in io_dict["input"]:
                if isinstance(value, torch.Tensor):
                    key = f"{module_name}.input.{tensor_name}"
                    layer_data[layer_key][key] = value.to(torch.float32)
                else:
                    # 保存非 tensor 值用于对比
                    non_tensor_values[module_name]["input"].append((tensor_name, value))

            # 保存输出张量
            for tensor_name, value in io_dict["output"]:
                if isinstance(value, torch.Tensor):
                    key = f"{module_name}.output.{tensor_name}"
                    layer_data[layer_key][key] = value.to(torch.float32)
                else:
                    non_tensor_values[module_name]["output"].append((tensor_name, value))

        # 保存 tensor 数据
        for layer_name, save_dict in layer_data.items():
            output_file = self.outputs_dir / f"{layer_name}.safetensors"
            save_file(save_dict, str(output_file))
            keys = list(save_dict.keys())
            print(f"  Saved {layer_name}: {len(keys)} tensors")

        # 保存非 tensor 值到 json
        if non_tensor_values:
            # 转换为可序列化格式
            serializable = {}
            for module_name, values in non_tensor_values.items():
                if values["input"] or values["output"]:
                    serializable[module_name] = {
                        "input": [(k, str(v)) for k, v in values["input"]],
                        "output": [(k, str(v)) for k, v in values["output"]],
                    }
            if serializable:
                json_file = self.outputs_dir / "non_tensor_values.json"
                with open(json_file, "w") as f:
                    json.dump(serializable, f, indent=2)

        # 保存 non_tensor_values 用于 compare_with
        self._non_tensor_values = non_tensor_values

        print(f"Saved {len(layer_data)} layer files with {len(self.module_io)} modules total")

    def compare_with(self, other_impl: str):
        """Compare captured I/O with another implementation's I/O."""
        other_dir = self.output_dir / other_impl
        if not other_dir.exists():
            print(f"Warning: No I/O data found for {other_impl}")
            return

        print(f"\nComparing {self.impl} vs {other_impl}:")

        # 加载另一个实现的非 tensor 值
        other_non_tensor_file = other_dir / "non_tensor_values.json"
        other_non_tensor = {}
        if other_non_tensor_file.exists():
            with open(other_non_tensor_file) as f:
                other_non_tensor = json.load(f)

        # 获取当前实现的非 tensor 值
        my_non_tensor = getattr(self, '_non_tensor_values', {})

        # 按照 layer 分组比较
        layer_modules = {}
        for module_name in sorted(self.module_io.keys()):
            layer_key = module_name.split(".")[0]
            if layer_key not in layer_modules:
                layer_modules[layer_key] = []
            layer_modules[layer_key].append(module_name)

        for layer_name in sorted(layer_modules.keys()):
            other_file = other_dir / f"{layer_name}.safetensors"
            if not other_file.exists():
                print(f"  {layer_name}: NOT FOUND in {other_impl}")
                continue

            # 加载另一个实现的数据
            other_data = load_file(str(other_file))

            print(f"  {layer_name}:")

            for module_name in layer_modules[layer_name]:
                # 简化模块名显示
                short_name = module_name.replace(f"{layer_name}.", "")
                print(f"    {short_name}:")

                # 比较 input tensors
                for tensor_name, value in self.module_io[module_name]["input"]:
                    key = f"{module_name}.input.{tensor_name}"
                    if isinstance(value, torch.Tensor):
                        if key in other_data:
                            other_tensor = other_data[key].float()
                            my_tensor = value.float()
                            if my_tensor.shape == other_tensor.shape:
                                # 清理 inf/-inf/nan
                                my_tensor_clean = self._clean_tensor(my_tensor)
                                other_tensor_clean = self._clean_tensor(other_tensor)
                                diff = (my_tensor_clean - other_tensor_clean).abs().max().item()
                                mean = (my_tensor_clean - other_tensor_clean).abs().mean().item()
                                if diff > 0:
                                    print(f"      input.{tensor_name}: max_diff={diff:.6e}, mean_diff={mean:.6e} (DIFF)")
                                    # 打印两个 tensor 的摘要
                                    print(self._tensor_summary(my_tensor_clean, "        my: "))
                                    print(self._tensor_summary(other_tensor_clean, "        other: "))
                                else:
                                    print(f"      input.{tensor_name}: max_diff={diff:.6e}, mean_diff={mean:.6e}")
                            else:
                                print(f"      input.{tensor_name}: shape mismatch {my_tensor.shape} vs {other_tensor.shape}")
                        else:
                            print(f"      input.{tensor_name}: NOT FOUND in {other_impl}")

                # 比较 input 非 tensors
                if module_name in my_non_tensor:
                    for name, value in my_non_tensor[module_name]["input"]:
                        if module_name in other_non_tensor:
                            other_values = {k: v for k, v in other_non_tensor[module_name]["input"]}
                            if name in other_values:
                                if str(value) == other_values[name]:
                                    print(f"      input.{name}: {value} (match)")
                                else:
                                    print(f"      input.{name}: {value} vs {other_values[name]} (DIFF)")
                            else:
                                print(f"      input.{name}: {value} (not in {other_impl})")
                        else:
                            print(f"      input.{name}: {value}")

                # 比较 output tensors
                for tensor_name, value in self.module_io[module_name]["output"]:
                    key = f"{module_name}.output.{tensor_name}"
                    if isinstance(value, torch.Tensor):
                        if key in other_data:
                            other_tensor = other_data[key].float()
                            my_tensor = value.float()
                            if my_tensor.shape == other_tensor.shape:
                                # 清理 inf/-inf/nan
                                my_tensor_clean = self._clean_tensor(my_tensor)
                                other_tensor_clean = self._clean_tensor(other_tensor)
                                diff = (my_tensor_clean - other_tensor_clean).abs().max().item()
                                mean = (my_tensor_clean - other_tensor_clean).abs().mean().item()
                                if diff > 0:
                                    print(f"      output.{tensor_name}: max_diff={diff:.6e}, mean_diff={mean:.6e} (DIFF)")
                                    # 打印两个 tensor 的摘要
                                    print(self._tensor_summary(my_tensor_clean, "        my: "))
                                    print(self._tensor_summary(other_tensor_clean, "        other: "))
                                else:
                                    print(f"      output.{tensor_name}: max_diff={diff:.6e}, mean_diff={mean:.6e}")
                            else:
                                print(f"      output.{tensor_name}: shape mismatch {my_tensor.shape} vs {other_tensor.shape}")
                        else:
                            print(f"      output.{tensor_name}: NOT FOUND in {other_impl}")

                # 比较 output 非 tensors
                if module_name in my_non_tensor:
                    for name, value in my_non_tensor[module_name]["output"]:
                        if module_name in other_non_tensor:
                            other_values = {k: v for k, v in other_non_tensor[module_name]["output"]}
                            if name in other_values:
                                if str(value) == other_values[name]:
                                    print(f"      output.{name}: {value} (match)")
                                else:
                                    print(f"      output.{name}: {value} vs {other_values[name]} (DIFF)")
                            else:
                                print(f"      output.{name}: {value} (not in {other_impl})")
                        else:
                            print(f"      output.{name}: {value}")


def compare_io_dirs(dir1: str, dir2: str):
    """
    Compare layer I/O from two existing dump directories.

    Args:
        dir1: Path to first implementation's I/O directory
        dir2: Path to second implementation's I/O directory
    """
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    if not dir1.exists():
        print(f"Error: Directory not found: {dir1}")
        return
    if not dir2.exists():
        print(f"Error: Directory not found: {dir2}")
        return

    # 获取所有 layer 文件
    layer_files1 = sorted(dir1.glob("layer_*.safetensors"))
    layer_files2 = sorted(dir2.glob("layer_*.safetensors"))

    if not layer_files1:
        print(f"Error: No layer files found in {dir1}")
        return
    if not layer_files2:
        print(f"Error: No layer files found in {dir2}")
        return

    # 加载非 tensor 值
    non_tensor1 = {}
    non_tensor2 = {}
    non_tensor_file1 = dir1 / "non_tensor_values.json"
    non_tensor_file2 = dir2 / "non_tensor_values.json"
    if non_tensor_file1.exists():
        with open(non_tensor_file1) as f:
            non_tensor1 = json.load(f)
    if non_tensor_file2.exists():
        with open(non_tensor_file2) as f:
            non_tensor2 = json.load(f)

    print(f"\nComparing module I/O:")
    print(f"  Dir1: {dir1}")
    print(f"  Dir2: {dir2}")
    print(f"  Files in dir1: {len(layer_files1)}")
    print(f"  Files in dir2: {len(layer_files2)}")
    print()

    # 获取层名称
    layers1 = {f.stem for f in layer_files1}
    layers2 = {f.stem for f in layer_files2}
    all_layers = sorted(layers1 | layers2)

    total_tensors = 0

    for layer_name in all_layers:
        file1 = dir1 / f"{layer_name}.safetensors"
        file2 = dir2 / f"{layer_name}.safetensors"

        if not file1.exists():
            print(f"  {layer_name}: NOT FOUND in dir1")
            continue
        if not file2.exists():
            print(f"  {layer_name}: NOT FOUND in dir2")
            continue

        # 加载数据
        data1 = load_file(str(file1))
        data2 = load_file(str(file2))

        # 按 module 分组
        # key format: "layer_X.submodule.input.param_name" or "layer_X.submodule.output.param_name"
        modules1 = {}
        modules2 = {}

        for key in data1.keys():
            parts = key.split(".")
            if len(parts) >= 4:
                module_key = ".".join(parts[:2])  # "layer_X.submodule"
                if module_key not in modules1:
                    modules1[module_key] = []
                modules1[module_key].append(key)

        for key in data2.keys():
            parts = key.split(".")
            if len(parts) >= 4:
                module_key = ".".join(parts[:2])
                if module_key not in modules2:
                    modules2[module_key] = []
                modules2[module_key].append(key)

        all_modules = sorted(set(modules1.keys()) | set(modules2.keys()))
        print(f"  {layer_name}:")

        for module_name in all_modules:
            keys1 = modules1.get(module_name, [])
            keys2 = modules2.get(module_name, [])

            short_name = module_name.replace(f"{layer_name}.", "")
            print(f"    {short_name}:")

            # 找出所有 tensor keys
            all_keys = sorted(set(keys1) | set(keys2))

            for key in all_keys:
                # 提取 tensor 名称部分: "layer_X.submodule.input.param_name" -> "input.param_name"
                tensor_name = ".".join(key.split(".")[2:])

                if key in data1 and key in data2:
                    t1 = data1[key].float()
                    t2 = data2[key].float()
                    if t1.shape == t2.shape:
                        # 清理 inf/-inf/nan
                        t1_clean = t1.clone()
                        t1_clean[torch.isinf(t1_clean) | torch.isnan(t1_clean)] = 0.0
                        t2_clean = t2.clone()
                        t2_clean[torch.isinf(t2_clean) | torch.isnan(t2_clean)] = 0.0
                        diff = (t1_clean - t2_clean).abs().max().item()
                        mean = (t1_clean - t2_clean).abs().mean().item()
                        if diff > 0:
                            print(f"      {tensor_name}: max_diff={diff:.6e}, mean_diff={mean:.6e} (DIFF)")
                            # 打印两个 tensor 的摘要
                            print(f"        dir1: shape={list(t1.shape)}, min={t1.min().item():.6e}, max={t1.max().item():.6e}, mean={t1.mean().item():.6e}")
                            print(f"        dir2: shape={list(t2.shape)}, min={t2.min().item():.6e}, max={t2.max().item():.6e}, mean={t2.mean().item():.6e}")
                        else:
                            print(f"      {tensor_name}: max_diff={diff:.6e}, mean_diff={mean:.6e}")
                    else:
                        print(f"      {tensor_name}: shape mismatch {t1.shape} vs {t2.shape}")
                else:
                    if key not in data1:
                        print(f"      {tensor_name}: NOT FOUND in dir1")
                    else:
                        print(f"      {tensor_name}: NOT FOUND in dir2")

                total_tensors += 1

            # 比较非 tensor 值
            if module_name in non_tensor1 or module_name in non_tensor2:
                # 收集所有非 tensor 值名称
                all_non_tensor_inputs = set()
                all_non_tensor_outputs = set()
                if module_name in non_tensor1:
                    all_non_tensor_inputs.update(k for k, v in non_tensor1[module_name].get("input", []))
                    all_non_tensor_outputs.update(k for k, v in non_tensor1[module_name].get("output", []))
                if module_name in non_tensor2:
                    all_non_tensor_inputs.update(k for k, v in non_tensor2[module_name].get("input", []))
                    all_non_tensor_outputs.update(k for k, v in non_tensor2[module_name].get("output", []))

                # 对比 input
                for name in sorted(all_non_tensor_inputs):
                    v1 = next((v for k, v in non_tensor1.get(module_name, {}).get("input", []) if k == name), None)
                    v2 = next((v for k, v in non_tensor2.get(module_name, {}).get("input", []) if k == name), None)
                    if v1 is not None and v2 is not None:
                        if v1 == v2:
                            print(f"      input.{name}: {v1} (match)")
                        else:
                            print(f"      input.{name}: {v1} vs {v2} (DIFF)")
                    elif v1 is not None:
                        print(f"      input.{name}: {v1} (not in dir2)")
                    else:
                        print(f"      input.{name}: {v2} (not in dir1)")

                # 对比 output
                for name in sorted(all_non_tensor_outputs):
                    v1 = next((v for k, v in non_tensor1.get(module_name, {}).get("output", []) if k == name), None)
                    v2 = next((v for k, v in non_tensor2.get(module_name, {}).get("output", []) if k == name), None)
                    if v1 is not None and v2 is not None:
                        if v1 == v2:
                            print(f"      output.{name}: {v1} (match)")
                        else:
                            print(f"      output.{name}: {v1} vs {v2} (DIFF)")
                    elif v1 is not None:
                        print(f"      output.{name}: {v1} (not in dir2)")
                    else:
                        print(f"      output.{name}: {v2} (not in dir1)")

    print(f"\nComparison complete: {len(all_layers)} layers, {total_tensors} tensors compared")


def dump_weights(model, output_path: str):
    """
    Dump model weights to a directory as safetensors.

    Args:
        model: Model with loaded weights
        output_path: Directory to save weights
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all parameters and buffers
    state_dict = {}
    for name, param in model.named_parameters():
        state_dict[name] = param.data.cpu().contiguous()
    for name, buf in model.named_buffers():
        state_dict[name] = buf.data.cpu().contiguous()

    # Save as single safetensors file (suitable for debug models)
    output_file = output_path / "model.safetensors"
    print(f"Saving {len(state_dict)} tensors to {output_file}...")
    save_file(state_dict, str(output_file))

    # Create index file
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())},
        "weight_map": {name: "model.safetensors" for name in state_dict},
    }
    index_file = output_path / "model.safetensors.index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Saved weights to {output_path}")
    print(f"  - {output_file}")
    print(f"  - {index_file}")

    # Save config
    config_file = output_path / "config.json"
    model.config.to_json_file(str(config_file))
    print(f"  - {config_file}")


def main():
    parser = argparse.ArgumentParser(description="Load GlmMoeDsa model")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to model directory containing safetensors files (required unless --compare_dirs is used)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target device (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Target dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--impl",
        type=str,
        default="torch_turbo",
        choices=["torch_turbo", "transformers"],
        help="Implementation to use (default: torch_turbo)",
    )
    parser.add_argument(
        "--dump",
        type=str,
        default=None,
        help="Directory to dump debug weights (optional)",
    )
    parser.add_argument(
        "--dump_io",
        type=str,
        default=None,
        help="Directory to dump layer I/O for debugging (optional)",
    )
    parser.add_argument(
        "--compare_io",
        type=str,
        default=None,
        help="Compare layer I/O with another implementation (e.g., 'torch_turbo' or 'transformers')",
    )
    parser.add_argument(
        "--compare_dirs",
        type=str,
        default=None,
        help="Compare two existing I/O dump directories (format: 'dir1,dir2'). "
             "If specified, only comparison is done, no model loading.",
    )
    args = parser.parse_args()

    # 如果指定了 compare_dirs，只做比较，不加载模型
    if args.compare_dirs:
        dirs = args.compare_dirs.split(",")
        if len(dirs) != 2:
            print("Error: --compare_dirs must be in format 'dir1,dir2'")
            return
        compare_io_dirs(dirs[0], dirs[1])
        return

    # 否则需要 --path
    if not args.path:
        parser.error("--path is required unless --compare_dirs is specified")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    # Get model classes based on implementation
    print(f"Using implementation: {args.impl}")
    config_class, model_class = get_model_classes(args.impl)

    # Step 1: Load or create config
    config = load_config_from_path(args.path, config_class)
    print(f"Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")

    # Step 2: Create model on meta device
    model = create_model_on_meta_device(config, model_class)

    # Step 3: Load weights from safetensors
    model = load_weights_from_safetensors(
        model,
        args.path,
        device=args.device,
        dtype=dtype_map[args.dtype],
        impl=args.impl,
    )
    if args.impl == "torch_turbo":
        model.model._use_flash_attention_2 = True
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = "eager"

    # Step 4: 简单验证
    print("\nModel loaded successfully!")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    # Step 5: Dump weights if specified
    if args.dump:
        dump_weights(model, args.dump)

    # Step 6: Setup I/O hook if specified
    io_hook = None
    if args.dump_io:
        io_hook = LayerIOHook(args.dump_io, args.impl)
        io_hook.register_hooks(model)

    # 简单前向测试
    print("\nRunning simple forward test...")
    model.eval()
    with torch.no_grad():
        seed_all()
        # 创建一个4k长度的输入
        input_ids = torch.randint(1, 1000, (1, 4096), device=args.device)
        attention_mask = None
        outputs = model(input_ids, attention_mask=attention_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output logits shape: {outputs.logits.shape}")

    # Step 7: Save layer I/O and compare if specified
    if io_hook:
        io_hook.save()
        io_hook.remove_hooks()

        if args.compare_io:
            io_hook.compare_with(args.compare_io)


if __name__ == "__main__":
    main()