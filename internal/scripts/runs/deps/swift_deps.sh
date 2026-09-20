#!/bin/bash
# Internal counterpart of scripts/runs/deps/swift_deps.sh: mirrors its sourcing chain.
# Add internal-only dependency commands below.

deps_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${deps_dir}/platform.sh"

# Internal-only dependencies, e.g.:
# pip install /path/to/internal/only.whl
pip install mdatasets==0.5.70 -i https://pypi.antfin-inc.com/simple