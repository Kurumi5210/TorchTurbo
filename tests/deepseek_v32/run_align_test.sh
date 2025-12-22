#!/bin/bash

python tests/deepseek_v32/run_tt_deepseek_v32_infer.py
python tests/deepseek_v32/run_hf_deepseek_v32_infer.py
python tests/deepseek_v32/align_compare.py
cat /tmp/align_compare.log