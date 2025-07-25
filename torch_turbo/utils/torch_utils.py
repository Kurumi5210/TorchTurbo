import re
from typing import List, Tuple, Union

import torch

def torch_version(
    version: str = torch.__version__, return_dev: bool = False
) -> Union[Tuple[int, ...], Tuple[Tuple[int, ...], str]]:
    numbering = re.search(r"^(\d+).(\d+).(\d+)([^\+]*)(\+\S*)?$", version)
    if not numbering:
        return tuple()
    # Catch torch version if run against internal pre-releases, like `1.8.0a0fb`,
    if numbering.group(4):
        # Two options here:
        # - either skip this version (minor number check is not relevant)
        # - or check that our codebase is not broken by this ongoing development.

        # Assuming that we're interested in the second use-case more than the first,
        # return the pre-release or dev numbering
        print(f"Pytorch pre-release version {version} - assuming intent to test it")

    ver = tuple(int(numbering.group(n)) for n in range(1, 4))
    if return_dev:
        return ver, numbering.group(4)
    else:
        return ver

def compare_tensors(key, custom, ref, rank, atol=1e-5):
    if rank != 0:
        return
    # check if the shapes match
    if custom.shape != ref.shape:
        print(f" Rank {rank} - Key: {key} - Shape mismatch ")
        print(f" Rank {rank} - Key: {key} - Expected shape: {custom.shape}")
        print(f" Rank {rank} - Key: {key} - Actual shape: {ref.shape}")
        
    if not torch.allclose(custom.float(), ref.float(), atol=atol):
        diff = torch.abs(custom - ref)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        print(f"Rank {rank} - Key: {key} - Max difference: {max_diff.item():.6f}")
        print(f"Rank {rank} - Key: {key} - Mean difference: {mean_diff.item():.6f}")
        print(f"Rank {rank} - Key: {key} - Custom shape: {custom.shape}")
        print(f"Rank {rank} - Key: {key} - Ref shape: {ref.shape}")
        print(f"Rank {rank} - Key: {key} - Custom value: {custom}")
        print(f"Rank {rank} - Key: {key} - Ref value: {ref}")

def load_safetensors_dir(model_dir: str) -> dict:
    from pathlib import Path
    from safetensors.torch import load_file

    state_dict = {}
    model_path = Path(model_dir)

    for safetensor_file in model_path.glob("*.safetensors"):
        file_path = safetensor_file.as_posix()
        state_dict.update(load_file(file_path, device="cpu"))

    return state_dict
