#!/bin/bash
set -e

# verified in image 'reg.docker.alibaba-inc.com/aii/aistudio:aistudio-311588595-56586242-1781508033657'

pip install torch==2.10.0
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
pip install accelerate==1.11.0
pip install trl==0.24.0
pip install matplotlib

pip uninstall flash_attn -y
pip install /input/torchturbo/stable/h20/build/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl --no-deps
pip install /input/torchturbo/stable/h20/build/nv_grouped_gemm-1.1.4.post8-cp312-cp312-linux_x86_64.whl --no-deps
pip install /input/torchturbo/stable/h20/build/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl --no-deps

pip install ms-swift==4.1.0

pip install transformers==5.7.0
pip install peft==0.18.0
pip install datasets==3.6.0

rm -rf ~/.cache/huggingface/modules/transformers_modules/