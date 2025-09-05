from torch.utils.data import DataLoader, Dataset
import functools
from torch.utils.data.distributed import DistributedSampler
from collections.abc import Mapping
import torch

def collate_sentences_lm(samples, input_name, label_name):
    if len(samples) == 0:
        return {}

    if len(samples[0]) == 0:
        return {}

    src_tokens = torch.stack([s["source"] for s in samples], 0)
    tgt_tokens = torch.stack([s["target"] for s in samples], 0)

    batch = {
        input_name: src_tokens,
        label_name: tgt_tokens,
    }
    return batch

class BenchmarkLMDataset(Dataset):
    def __init__(
        self,
        vocab_size=10000,
        max_source_positions=1024,
        total_samples=10000,
        null_dataset=False,
    ):
        self.vocab_size = vocab_size
        self.max_source_positions = max_source_positions
        self.total_samples = total_samples
        self.null_dataset = null_dataset

    def __getitem__(self, index):
        if self.null_dataset:
            return {}
        length = self.max_source_positions
        source = torch.randint(1, self.vocab_size, (length,))
        # source = torch.full((length,), 12815, dtype=torch.int)  # Use a fixed value for testing
        target = source.clone()
        return {
            "source": source,
            "target": target,
        }

    def __len__(self):
        return self.total_samples

def get_dataset(seq_length=128, vocab_size=32000, data_size=1000, null_dataset=False):
    return BenchmarkLMDataset(
        vocab_size=vocab_size, max_source_positions=seq_length, total_samples=data_size, null_dataset=null_dataset
    )


def get_dataloader(dataset, batch_size, rank, dp_size, use_distributed):
    dataloader_args = {"batch_size": batch_size, "drop_last": True, "shuffle": False, "num_workers": 2}
    input_name = "input_ids"
    label_name = "labels"
    dataloader_args["collate_fn"] = functools.partial(
        collate_sentences_lm, input_name=input_name, label_name=label_name
    )
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=False, num_replicas=dp_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(dataset, sampler=sampler, **dataloader_args)
    return dataloader

def recursively_apply(
    func, data, *args, test_type=lambda t: isinstance(t, torch.Tensor), error_on_other_type=False, **kwargs
):
    if isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for k, v in data.items()
            }
        )
    elif isinstance(data, (tuple, list)):
        return type(data)(
            [
                recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for v in data
            ]
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects "
            f"that satisfy {test_type.__name__}."
        )
    return data


def data_to_device(data, device, non_blocking=False):
    def to_device(data, device, non_blocking=False):
        if device == "pin":
            data = data.pin_memory()
        else:
            data = data.to(device, non_blocking=non_blocking)
        return data

    return recursively_apply(to_device, data, device, non_blocking=non_blocking)

def split_into_micro_batches(batch, micro_batch_size):
    """
    Split a batch dictionary into multiple micro batches.
    
    Args:
        batch: Input dictionary {key: tensor}, where all tensors must have 
               the same size in the first dimension (batch_size).
        micro_batch_size: Number of samples per micro batch.
    
    Returns:
        List[Dict]: List of split micro batches, each being a dictionary 
                   with the same structure as input.
    """
    if not batch:
        return []

    # Get global batch size (assuming all tensors have same first dim)
    batch_size = next(iter(batch.values())).size(0)
    
    # Calculate number of micro batches needed
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    # Split each tensor in the batch dictionary
    micro_batches = []
    for i in range(num_micro_batches):
        start = i * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        
        micro_batch = {
            key: tensor[start:end]  # Handles tensors of any dimension
            for key, tensor in batch.items()
        }
        micro_batches.append(micro_batch)
    
    return micro_batches
