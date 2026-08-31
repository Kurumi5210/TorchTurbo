#!/bin/bash

deps_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${deps_dir}/platform.sh"

pip install transformers==4.57.3

# dependency of llama_factory
pip install safetensors==0.7.0
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
pip install trl==0.24.0
pip install matplotlib
