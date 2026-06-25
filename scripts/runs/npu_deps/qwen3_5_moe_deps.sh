#!/bin/bash
set -e

# verified in image 'hcr.meta-wulan01.hw-wulan.local/aii/aistudio:15100125-20251212112959_aarch64'


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
pip install torchaudio==2.10.0
pip install datasets==4.0.0
pip install accelerate==1.11.0
pip install trl==0.18.0
pip install matplotlib
pip install torchvision==0.22.1
pip install torchaudio==2.7.1

# install packages related to npu triton
pip install /sfs_turbo/zhisan/envs/triton_ascend-3.2.0.dev20260315-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl --no-deps
pip install /sfs_turbo/zhisan/envs/flash_linear_attention-0.4.2-py3-none-any.whl --no-deps
cp -f /sfs_turbo/zhisan/envs/bishengir/bin/* /usr/local/Ascend/ascend-toolkit/8.3.RC2/bisheng_toolkit/bishengir/bin/
cp -rf /sfs_turbo/zhisan/envs/bishengir/lib /usr/local/Ascend/ascend-toolkit/8.3.RC2/bisheng_toolkit/bishengir

rm -rf ~/.cache/huggingface/modules/transformers_modules/