import re
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor.placement_types import Shard
from torch.distributed._tensor import DTensor

from torch_turbo.distributed.parallel_state import get_world_mesh
from torch_turbo.utils.device import get_device_name


class CheckpointingManager:
    @staticmethod
    def load_checkpoint_shard_async(model, save_path, ckpt_group, extra={}):
        """
        Load an FSDP checkpoint for this rank.
        Every rank will load its own model shard.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        checkpoint_id = save_path
        optimizer = extra.get("optimizer", None)
        lr_scheduler = extra.get("lr_scheduler", None)
        state_dict = dict()
        model_sd = model.state_dict()
        world_mesh = get_world_mesh()
        # HACK: change device mesh to use dcp
        for name in list(model_sd.keys()):
            if "mlp.experts.weight" in name:
                local_tensor = model_sd[name].to_local()
                model_sd[name] = DTensor.from_local(local_tensor, device_mesh=world_mesh, placements=[Shard(0)])
        state_dict["model"] = model_sd
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            state_dict["lr_scheduler"] = lr_scheduler.state_dict()
        
        dcp.state_dict_loader.load(state_dict, checkpoint_id=checkpoint_id, process_group=ckpt_group)
        # HACK: resume device mesh
        for name in list(state_dict["model"].keys()):
            if "expert_bias" in name:
                model_sd[name] = model.state_dict()[name]
            if "mlp.experts.weight" in name:
                local_tensor = state_dict["model"][name].to_local()
                old_mesh = model.state_dict()[name].device_mesh
                state_dict["model"][name] = DTensor.from_local(local_tensor, device_mesh=old_mesh, placements=[Shard(0)])
        model.load_state_dict(state_dict["model"])
        if optimizer is not None:
            optimizer.load_state_dict(state_dict["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        # Load RNG state
        rng_path = os.path.join(save_path, f"rng_world_size_{world_size}_rank_{rank}.pt")
        if os.path.exists(rng_path):
            rng_states = torch.load(rng_path, weights_only=False)
            import random
            import numpy as np
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.random.set_rng_state(rng_states["cpu"])
            # Load device-specific RNG state based on device type
            device_name = get_device_name()
            if device_name == "npu":
                torch.npu.random.set_rng_state(rng_states["npu"])
            elif device_name == "cuda":
                torch.cuda.random.set_rng_state(rng_states["cuda"])
            print(f"Rank: {rank} - Loaded RNG state from {os.path.abspath(rng_path)}")
        else:
            print(f"Rank: {rank} - RNG checkpoint not found at {rng_path}")

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    @staticmethod
    def load_checkpoint_shard(model, save_path, extra={}):
        """
        Load an FSDP checkpoint for this rank.
        Every rank will load its own model shard.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_path = os.path.join(save_path, f"model_world_size_{world_size}_rank_{rank}.pt")
        model_state_dict = torch.load(model_path, weights_only=False)
        model.load_state_dict(model_state_dict)
        print(f"Rank: {rank} - Loaded model from {os.path.abspath(model_path)}")

        optimizer = extra.get("optimizer", None)
        if optimizer is not None:
            # load optimizer state dict
            optim_path = os.path.join(save_path, f"optim_world_size_{world_size}_rank_{rank}.pt")
            if os.path.exists(optim_path):
                optimizer_state_dict = torch.load(optim_path, weights_only=False)
                optimizer.load_state_dict(optimizer_state_dict)
                print(f"Rank: {rank} - Loaded optimizer from {os.path.abspath(optim_path)}")
            else:
                print(f"Rank: {rank} - Optimizer checkpoint not found at {optim_path}")

        lr_scheduler = extra.get("lr_scheduler", None)
        if lr_scheduler is not None:
            # load lr scheduler state dict
            scheduler_path = os.path.join(save_path, f"scheduler_world_size_{world_size}_rank_{rank}.pt")
            if os.path.exists(scheduler_path):
                lr_scheduler_state_dict = torch.load(scheduler_path, weights_only=False)
                lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                print(f"Rank: {rank} - Loaded lr scheduler from {os.path.abspath(scheduler_path)}")
            else:
                print(f"Rank: {rank} - Lr scheduler checkpoint not found at {scheduler_path}")

        # Load RNG state
        rng_path = os.path.join(save_path, f"rng_world_size_{world_size}_rank_{rank}.pt")
        if os.path.exists(rng_path):
            rng_states = torch.load(rng_path, weights_only=False)
            import random
            import numpy as np
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.random.set_rng_state(rng_states["cpu"])
            # Load device-specific RNG state based on device type
            device_name = get_device_name()
            if device_name == "npu":
                torch.npu.random.set_rng_state(rng_states["npu"])
            elif device_name == "cuda":
                torch.cuda.random.set_rng_state(rng_states["cuda"])
            print(f"Rank: {rank} - Loaded RNG state from {os.path.abspath(rng_path)}")
        else:
            print(f"Rank: {rank} - RNG checkpoint not found at {rng_path}")

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    @staticmethod
    def save_checkpoint_shard_async(model, save_path, ckpt_group, save_only_model=False, extra={}):
        """
        Save an FSDP checkpoint for this rank.
        Every rank will save its own model shard.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        optimizer = extra.get("optimizer", None)
        lr_scheduler = extra.get("lr_scheduler", None)
        state_dict = dict()
        model_sd = model.state_dict()
        world_mesh = get_world_mesh()
        # HACK: change device mesh to use dcp
        for name in list(model_sd.keys()):
            if "mlp.experts.weight" in name:
                local_tensor = model_sd[name].to_local()
                model_sd[name] = DTensor.from_local(local_tensor, device_mesh=world_mesh, placements=[Shard(0)])
        state_dict["model"] =  model_sd
        if not save_only_model:
            if optimizer is not None:
                state_dict["optimizer"] = optimizer.state_dict()
            if lr_scheduler is not None:
                state_dict["lr_scheduler"] = lr_scheduler.state_dict()
        future = dcp.state_dict_saver.async_save(state_dict, checkpoint_id=save_path, process_group=ckpt_group)
        # Save RNG state
        rng_path = os.path.join(save_path, f"rng_world_size_{world_size}_rank_{rank}.pt")
        import random
        import numpy as np
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        # Save device-specific RNG state based on device type
        device_name = get_device_name()
        if device_name == "npu":
            rng_states["npu"] = torch.npu.random.get_rng_state()
        elif device_name == "cuda":
            rng_states["cuda"] = torch.cuda.random.get_rng_state()
        torch.save(rng_states, rng_path)
        print(f"Rank: {rank} - Saved rng to {os.path.abspath(rng_path)}")

        # wait for everyone to save checkpoints
        torch.distributed.barrier()
        
        return future

    @staticmethod
    def save_checkpoint_shard(model, save_path, save_only_model=False, extra={}):
        """
        Save an FSDP checkpoint for this rank.
        Every rank will save its own model shard.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_path = os.path.join(save_path, f"model_world_size_{world_size}_rank_{rank}.pt")
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, model_path)
        print(f"Rank: {rank} - Saved model to {os.path.abspath(model_path)}")

        if not save_only_model:
            optimizer = extra.get("optimizer", None)
            if optimizer is not None:
                # save optimizer state dict
                optim_path = os.path.join(save_path, f"optim_world_size_{world_size}_rank_{rank}.pt")
                optimizer_state_dict = optimizer.state_dict()
                torch.save(optimizer_state_dict, optim_path)
                print(f"Rank: {rank} - Saved optimizer to {os.path.abspath(optim_path)}")

            lr_scheduler = extra.get("lr_scheduler", None)
            if lr_scheduler is not None:
                # save lr scheduler state dict
                scheduler_path = os.path.join(save_path, f"scheduler_world_size_{world_size}_rank_{rank}.pt")
                lr_scheduler_state_dict = lr_scheduler.state_dict()
                torch.save(lr_scheduler_state_dict, scheduler_path)
                print(f"Rank: {rank} - Saved lr scheduler to {os.path.abspath(scheduler_path)}")

            # Save RNG state
            rng_path = os.path.join(save_path, f"rng_world_size_{world_size}_rank_{rank}.pt")
            import random
            import numpy as np
            rng_states = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
            }
            # Save device-specific RNG state based on device type
            device_name = get_device_name()
            if device_name == "npu":
                rng_states["npu"] = torch.npu.random.get_rng_state()
            elif device_name == "cuda":
                rng_states["cuda"] = torch.cuda.random.get_rng_state()
            torch.save(rng_states, rng_path)
            print(f"Rank: {rank} - Saved rng to {os.path.abspath(rng_path)}")

        # wait for everyone to save checkpoints
        torch.distributed.barrier()

