#!/bin/bash
# Internal counterpart of scripts/runs/deps/gpu.sh: add internal-only GPU dependencies below.

# Internal-only dependencies, e.g.:
# pip install /path/to/internal/only.whl

# verified in image 'reg.docker.alibaba-inc.com/aii/aistudio:aistudio-311588595-56586242-1781508033657'

pip install torch==2.10.0
pip uninstall flash_attn -y
pip install /input/torchturbo/stable/h20/build/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl --no-deps
pip install /input/torchturbo/stable/h20/build/nv_grouped_gemm-1.1.4.post8-cp312-cp312-linux_x86_64.whl --no-deps
pip install /input/torchturbo/stable/h20/build/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl --no-deps
