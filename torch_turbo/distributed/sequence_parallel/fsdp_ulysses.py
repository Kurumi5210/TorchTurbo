from typing import Dict
from torch.distributed.device_mesh import DeviceMesh
import torch

from torch_turbo.distributed.parallel_state import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group


class FSDPUlyssesShardingManager:
    """
    Sharding manager to support data resharding when using FSDP + Ulysses
    """

    def __init__(self, device_mesh: DeviceMesh):
        super().__init__()
        self.device_mesh = device_mesh
        self.seed_offset = 12345

    def __enter__(self):
        if self.device_mesh is not None:
            # We have a global SP group
            # so we have to change to use model-specific sp group
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())
            # TODO: check how to set seed for each model

    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        if self.device_mesh is not None:
            # revert to previous sp group
            set_ulysses_sequence_parallel_group(self.prev_sp_group)
            # TODO: check how to set seed for each model

    def preprocess_data(self, data: Dict[str, torch.Tensor]):
        """
        AllGather data from sp region
        This is because the data is first sharded along the FSDP dimension as we utilize the DP_COMPUTE
        In Ulysses, we need to make sure the same data is used across a SP group
        """
        if self.device_mesh is not None:
            sp_group = self.device_mesh["sp"].get_group()
            sp_size = torch.distributed.get_world_size(group=sp_group)
            output = {}
            sorted_keys = sorted(data.keys())
            for key in sorted_keys:
                val = data[key]
                output[key] = [torch.empty_like(val) for _ in range(sp_size)]
                torch.distributed.all_gather(output[key], val, group=sp_group, async_op=False)
                output[key] = torch.cat(output[key], dim=0)
            return output

        return data

    def preprocess_data_with_pad(self, data: Dict[str, torch.Tensor], pad_value: int = 0):
        """
        Synchronizes variable-length sequences across ranks in a sequence parallel group by padding
        to the maximum sequence length found in the group. Handles multiple data types in input dictionary.

        Args:
            data: Dictionary containing:
                'input_ids': Input token IDs [batch_size, seq_len]
                'labels': Target labels [batch_size, seq_len]
                'attention_mask': Attention mast[batch_size, seq_len]
            sp_group: Sequence parallel process group

        Returns:
            gathered_data: Dictionary with padded and gathered tensors from all ranks
        """
        if self.device_mesh is not None:
            sp_group = self.device_mesh["sp"].get_group()
            sp_size = torch.distributed.get_world_size(group=sp_group)
            batch_size, seq_len = data['input_ids'].shape

            device = data['input_ids'].device

            # 1. Determine global max sequence length
            local_max_len = torch.tensor(seq_len, device=device)
            global_max_len = local_max_len.clone()
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX, group=sp_group)
            global_max_len = global_max_len.item()
            padding_needed = global_max_len - seq_len

            # 2. Pad all tensors to global max length
            output = {}

            # Pad 2D tensors (ids and labels)
            sorted_keys = sorted(data.keys())
            for key in sorted_keys:
                if key == 'labels':
                    padded_data = torch.nn.functional.pad(
                        data[key],
                        (0, padding_needed),  # Pad right side of sequence dimension
                        value=-100  # Use -100 for label padding (common in loss ignore)
                    )
                elif key == 'input_ids':
                    padded_data = torch.nn.functional.pad(
                        data[key],
                        (0, padding_needed),
                        value=pad_value
                    )
                else:
                    padded_data = torch.nn.functional.pad(
                        data[key],
                        (0, padding_needed),  # Pad right side of sequence dimension
                        value=0  # Typically mask with 0 in input ids and attention mask
                    )
                output[key] = [torch.empty_like(padded_data) for _ in range(sp_size)]
                torch.distributed.all_gather(output[key], padded_data, group=sp_group, async_op=False)
                output[key] = torch.cat(output[key], dim=0)
            return output

        return data

    def preprocess_data_for_different_tensor_shape(self, data: list):
        if self.device_mesh is not None:
            sp_group = self.device_mesh["sp"].get_group()
            sp_size = torch.distributed.get_world_size(group=sp_group)
            gathered_lists = [None] * sp_size
            torch.distributed.all_gather_object(gathered_lists, data, group=sp_group)

            output = []
            for lst in gathered_lists:
                output.extend(lst)
            return output

        return data

    def postprocess_data(self, data: Dict[str, torch.Tensor]):
        """
        Split the data to follow FSDP partition
        """
        if self.device_mesh is not None:
            sp_size = self.device_mesh["sp"].size()
            sp_rank = self.device_mesh["sp"].get_local_rank()
            output = {}
            sorted_keys = sorted(data.keys())
            for key in sorted_keys:
                val = data[key]
                output[key] = val.chunk(chunks=sp_size)[sp_rank]
            return output

        return data
