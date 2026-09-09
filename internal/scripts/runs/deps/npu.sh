#!/bin/bash
# Internal counterpart of scripts/runs/deps/npu.sh: add internal-only NPU dependencies below.

# Internal-only dependencies, e.g.:
# pip install /path/to/internal/only.whl

# verified in image 'hcr.meta-wulan01.hw-wulan.local/antsys/vllm:AsystemHybirdEngine-0b9af7b5-20260824140737'

pip config set global.index-url https://pypi.antfin-inc.com/simple/
