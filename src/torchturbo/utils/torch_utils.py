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
    import os
    from safetensors.torch import load_file as safetensors_load

    state_dict = {}

    # Step 1: Identify if the input is a Hugging Face repo ID or local path
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id

        try:
            validate_repo_id(model_dir)
            is_hf_repo = True
        except HFValidationError:
            is_hf_repo = False
    except ImportError:
        is_hf_repo = False

    if is_hf_repo:
        from huggingface_hub import snapshot_download

        # Step 2: Download the repo if it's a Hugging Face repo ID
        local_path = snapshot_download(
            repo_id=model_dir,
        )
    else:
        # Assume it's a local path
        local_path = model_dir

    # Step 3: Load all .safetensors and .bin files
    file_paths_to_load = []
    if os.path.isdir(local_path):
        for filename in os.listdir(local_path):
            filepath = os.path.join(local_path, filename)
            if filename.endswith(".safetensors") or filename.endswith(".bin"):
                file_paths_to_load.append(filepath)
    elif os.path.isfile(local_path):
        file_paths_to_load.append(local_path)
    else:
        raise ValueError(
            f"Local path {local_path} does not exist or is not a valid path, "
            f"or {local_path} is a huggingface repo id but huggingface_hub is not installed."
        )

    def _load(filepath: str):
        import time

        t0 = time.perf_counter()
        if filepath.endswith(".safetensors"):
            state_dict = safetensors_load(filepath)
            for k in list(state_dict.keys()):
                state_dict[k] = state_dict[k].clone()
        elif filepath.endswith(".bin"):
            state_dict = torch.load(filepath, map_location="cpu")
        else:
            raise ValueError(f"{filepath} is not a torch bin or safetensor file.")
        t1 = time.perf_counter()
        print(f"[Loading] load {filepath} done in {t1 - t0:.2f} s", flush=True)
        return state_dict

    state_dict = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() // 8)) as executor:
        future_to_checkpoint = {
            executor.submit(_load, path): path for path in file_paths_to_load
        }

        for future in as_completed(future_to_checkpoint):
            path = future_to_checkpoint[future]
            try:
                sd = future.result()
                state_dict.update(sd)
            except Exception as e:
                raise RuntimeError(f"Error loading checkpoint from {path}: {e}")
    return state_dict
