#!/bin/bash
set -e

# verified in image 'hcr.meta-wulan01.hw-wulan.local/aii/aistudio:17940140-20260427141323_aarch64'


pip config set global.index-url https://pypi.antfin-inc.com/simple/

pip install transformers==4.57.3

# dependency of llama_factory
pip install deepspeed==0.16.0
pip install tensorboard
pip install psutil==7.2.2
pip install omegaconf
pip install pyparsing
pip install cycler
pip install kiwisolver
pip install peft==0.17.1
pip install torchvision==0.25.0
pip install torchaudio==2.10.0
pip install datasets==4.0.0
pip install accelerate==1.11.0
pip install trl==0.24.0
pip install matplotlib

# install packages related to npu triton
pip install /mnt/sfs_turbo/torchturbo/stable/qwen3_5_deps/triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl --no-deps
pip install /mnt/sfs_turbo/torchturbo/stable/qwen3_5_deps/flash_linear_attention-0.5.2-py3-none-any.whl --no-deps

rm -rf ~/.cache/huggingface/modules/transformers_modules/
