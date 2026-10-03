#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

source "${repo_root}/scripts/runs/deps/lf_deps.sh"
source "${script_dir}/../deps/lf_deps.sh"

# The checkpoint's native media processor requires torchvision, OpenCV,
# PyAV, and torchaudio.
python -m pip install torchvision==0.25.0
python -m pip install opencv-python-headless 'av>=10,<=16.0.0' 'qwen-vl-utils>=0.0.14'
