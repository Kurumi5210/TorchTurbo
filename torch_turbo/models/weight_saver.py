import re
import os

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from .weight_loader import HF_BUFFER_NAME_MAP
from torch_turbo.distributed.comm import all_gather_tensor, all_gather_tensor_chunked
from torch_turbo.distributed.parallel_state import get_expert_mesh
from torch_turbo.models.registry import get_model_cls, get_parallel_plan


class DistParamSaver:
    def __init__(self, model, args, processing_class=None):
        self.model = model
        self.moe_config = get_parallel_plan(model.config).moe_config
        
        ep_mesh = get_expert_mesh()
        self.ep = ep_mesh["ep"]
        ep_group = self.ep.get_group(0)
        self.ep_size = dist.get_world_size(group=ep_group)
        self.rank = dist.get_rank()

        self.use_te = args.use_transformer_engine
        self.processing_class = processing_class

        self.model_type = model.config.model_type

    def _is_rank_in_mesh(self, target_rank, device_mesh):
        return target_rank in device_mesh.mesh.flatten().tolist()

    def _get_full_hf_state_dict(self):
        # Only rank 0 will save hf model and,
        # offload to cpu to save LLMs which may be too large to fit in one GPU
        full_sd = {}
        if hasattr(self.model.config, "vision_config"):
            config = self.model.config.text_config
        else:
            config = self.model.config

        for name, param in self.model.state_dict().items():
            if isinstance(param, DTensor):
                full_tensor = param.full_tensor()
            else:
                full_tensor = param
            if self.rank == 0:
                if "mlp.experts" in name:
                    prefix = name[:-8]
                    num_experts = self.moe_config.num_moe_experts
                    full_tensor = all_gather_tensor_chunked(full_tensor, ep_name="ep", num_chunks=self.ep_size)
                    if "weight1" in name:
                        experts = torch.split(full_tensor, config.hidden_size)
                        for i in range(num_experts):
                            gate_proj = f"{prefix}.{i}.gate_proj.weight"
                            up_proj = f"{prefix}.{i}.up_proj.weight"
                            expert2 = torch.split(experts[i].t(), config.moe_intermediate_size)
                            full_sd[gate_proj] = expert2[0].cpu()
                            full_sd[up_proj] = expert2[1].cpu()
                    else:
                        # weight2
                        experts = torch.split(full_tensor, config.moe_intermediate_size)
                        for i in range(num_experts):
                            down_proj = f"{prefix}.{i}.down_proj.weight"
                            full_sd[down_proj] = experts[i].t().cpu()
                elif "mlp.router" in name:
                    new_name = name.replace("router", "gate")
                    if self.model_type in HF_BUFFER_NAME_MAP.keys():
                        new_name = new_name.replace("expert_bias", HF_BUFFER_NAME_MAP[self.model_type]["expert_bias"])
                    full_sd[new_name] = full_tensor.cpu()
                else:
                    full_sd[name] = full_tensor.cpu()
            elif self._is_rank_in_mesh(0, self.ep) and "mlp.experts" in name:
                full_tensor = all_gather_tensor_chunked(full_tensor, ep_name="ep", num_chunks=self.ep_size)
                del full_tensor
            else:
                del full_tensor

        return full_sd

    def _get_te_full_hf_state_dict(self):
        # Only rank 0 will save hf model and,
        # offload to cpu to save LLMs which may be too large to fit in one GPU
        full_sd = {}
        if hasattr(self.model.config, "vision_config"):
            config = self.model.config.text_config
        else:
            config = self.model.config


        for name, param in self.model.named_parameters():
            if isinstance(param, DTensor):
                full_tensor = param.full_tensor()
            else:
                full_tensor = param
            if self.rank == 0:
                if "mlp.experts" in name:
                    full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                    prefix_lst = name.split(".")
                    prefix = ".".join(name.split(".")[:5])
                    num_local_experts = self.moe_config.num_moe_experts // self.ep_size
                        
                    if "fc1_linear" in name:
                        experts = torch.split(full_tensor, config.moe_intermediate_size * 2)
                        for i in range(self.ep_size):
                            expert_idx = i * num_local_experts + int(re.findall(r"\d+", prefix_lst[-1])[0])
                            gate_proj = f"{prefix}.{expert_idx}.gate_proj.weight"
                            up_proj = f"{prefix}.{expert_idx}.up_proj.weight"
                            expert2 = torch.split(experts[i], config.moe_intermediate_size)
                            full_sd[gate_proj] = expert2[0].cpu()
                            full_sd[up_proj] = expert2[1].cpu()
                    elif "fc2_linear" in name:
                        experts = torch.split(full_tensor, config.hidden_size)
                        for i in range(self.ep_size):
                            expert_idx = i * num_local_experts + int(re.findall(r"\d+", prefix_lst[-1])[0])
                            down_proj = f"{prefix}.{expert_idx}.down_proj.weight"
                            full_sd[down_proj] = experts[i].cpu()
                elif "mlp.router" in name:
                    new_name = name.replace("router", "gate")
                    full_sd[new_name] = full_tensor.cpu()
                else:
                    full_sd[name] = full_tensor.cpu()
            elif self._is_rank_in_mesh(0, self.ep) and "mlp.experts" in name:
                full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                del full_tensor
            else:
                del full_tensor

        return full_sd


    def save_model(self, save_path):
        if self.use_te:
            state_dict = self._get_te_full_hf_state_dict()
        else:
            state_dict = self._get_full_hf_state_dict()

        rank = torch.distributed.get_rank()
        if rank == 0:
            with torch.device("meta"):
                save_model = get_model_cls(self.model.config)
                
            save_model.to_empty(device="cpu")

            for k in list(state_dict.keys()):
                v = state_dict[k]
                if isinstance(v, torch.Tensor):
                    v_bf16 = v.to(torch.bfloat16)
                    if not v_bf16.is_contiguous():
                        v_bf16 = v_bf16.contiguous()
                    state_dict[k] = v_bf16
                    del v, v_bf16

            hf_local_path = os.path.join(save_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)
            save_model.save_pretrained(hf_local_path, state_dict=state_dict, max_shard_size="10GB", safeserialization=True)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_local_path)
            print(f"Rank: {rank} - Saved hf model to {os.path.abspath(hf_local_path)}")

        # wait for rank0 to dump hf_model to local
        torch.distributed.barrier()