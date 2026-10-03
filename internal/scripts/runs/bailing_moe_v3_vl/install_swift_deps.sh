#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

source "${repo_root}/scripts/runs/deps/swift_deps.sh"
source "${script_dir}/../deps/swift_deps.sh"

python -m pip install opencv-python-headless 'av>=10,<=16.0.0' 'qwen-vl-utils>=0.0.14'
