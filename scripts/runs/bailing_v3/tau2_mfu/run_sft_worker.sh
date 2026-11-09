#!/usr/bin/env bash
# Give every worker a fresh compile cache before Python imports Triton or Torch.
set -euo pipefail

cache_base=$1
shift
: "${RANK:?This worker must be launched by torchrun.}"
mkdir -p "${cache_base}"
worker_cache=$(mktemp -d "${cache_base}/torchturbo-rank-${RANK}.XXXXXXXX")
export TRITON_CACHE_DIR=${worker_cache}/triton
export TORCHINDUCTOR_CACHE_DIR=${worker_cache}/inductor
export TORCH_EXTENSIONS_DIR=${worker_cache}/extensions
export PYTHONFAULTHANDLER=1
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
printf '[worker] rank=%s pid=%s compile_cache=%s\n' "${RANK}" "$$" "${worker_cache}"
exec "$@"
