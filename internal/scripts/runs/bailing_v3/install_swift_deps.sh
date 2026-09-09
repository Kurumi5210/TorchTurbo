#!/bin/bash
# Internal wrapper: delegates to the open-source install script, then sources the
# internal deps chain (where internal-only dependencies are added).
# Further model-specific internal commands can be appended below.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

bash "${repo_root}/scripts/runs/bailing_v3/install_swift_deps.sh"

source "${script_dir}/../deps/swift_deps.sh"

# Internal-only dependencies, e.g.:
# pip install /path/to/internal/only.whl

pip install /mnt/sfs_turbo/torchturbo/stable/flash_linear_attention-0.6.0-py3-none-any.whl
pip install /mnt/sfs_turbo/torchturbo/stable/triton_ascend-3.2.2-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
