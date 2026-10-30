#!/bin/bash
# Swift dependency installation for Qwen3.8 Dense.
#
# This is intentionally self-contained so the Dense run does not depend on a
# model-specific qwen3_5_moe installer.

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

# Keep this list aligned with the Swift dependency installer used by the
# existing Qwen3.5 runs, while keeping the model-specific entry point local.
source "${repo_root}/scripts/runs/deps/swift_deps.sh"
pip install qwen-vl-utils==0.0.14

source "${script_dir}/../deps/swift_deps.sh"

# Keep the Transformers dynamic-module cache from mixing model revisions.
rm -rf ~/.cache/huggingface/modules/transformers_modules/
