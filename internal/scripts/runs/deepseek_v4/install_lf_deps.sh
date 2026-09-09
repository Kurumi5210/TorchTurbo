#!/bin/bash
# Internal wrapper: delegates to the open-source install script, then sources the
# internal deps chain (where internal-only dependencies are added).
# Further model-specific internal commands can be appended below.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

bash "${repo_root}/scripts/runs/deepseek_v4/install_lf_deps.sh"

source "${script_dir}/../deps/lf_deps.sh"

# Internal-only dependencies, e.g.:
# pip install /path/to/internal/only.whl
