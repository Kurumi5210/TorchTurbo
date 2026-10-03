#!/bin/bash
# Internal counterpart of scripts/runs/deps/npu.sh: add internal-only NPU dependencies below.

# Internal-only dependencies, e.g.:
# pip install /path/to/internal/only.whl

# verified in image 'hcr.meta-wulan01.hw-wulan.local/antsys/vllm:AsystemHybirdEngine-0b9af7b5-20260824140737'

pip config set global.index-url https://pypi.antfin-inc.com/simple/

# KDA kernels (fla) and the Ascend triton backend they run on.
pip install /mnt/sfs_turbo/torchturbo/stable/flash_linear_attention-0.6.0-py3-none-any.whl
pip install /mnt/sfs_turbo/torchturbo/stable/triton_ascend-3.2.2-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
