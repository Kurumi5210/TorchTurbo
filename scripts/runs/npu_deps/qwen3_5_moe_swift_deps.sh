#!/bin/bash
set -e

pip config set global.index-url https://pypi.antfin-inc.com/simple/

pip install deepspeed==0.16.0
pip install tensorboard
pip install psutil==7.2.2
pip install omegaconf
pip install pyparsing
pip install cycler
pip install kiwisolver
pip install torchvision==0.25.0
pip install torchaudio==2.10.0
pip install accelerate==1.11.0
pip install trl==0.24.0
pip install matplotlib

pip install ms_swift==4.4.2
pip install transformers==5.9.0
pip install peft==0.18.0
pip install datasets==5.0.1
pip install qwen-vl-utils==0.0.14

rm -rf ~/.cache/huggingface/modules/transformers_modules/
