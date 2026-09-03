#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../deps/lf_deps.sh"

pip install /mnt/sfs_turbo/torchturbo/stable/flash_linear_attention-0.6.0-py3-none-any.whl
pip install /mnt/sfs_turbo/torchturbo/stable/triton_ascend-3.2.2-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl

rm -rf ~/.cache/ ~/.triton/
