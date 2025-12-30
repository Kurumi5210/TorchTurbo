#!/bin/bash

python tests/ling_mini_v2_5_fla/run_tt_ling_mini_v2_5_fla_infer.py
python tests/ling_mini_v2_5_fla/run_ds_ling_mini_v2_5_fla_infer.py
python tests/ling_mini_v2_5_fla/align_compare.py
cat /input/zhisan/dump/align_compare.log