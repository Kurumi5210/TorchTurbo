#!/bin/bash

python example/glm_moe_dsa/load_glm_moe_dsa.py --path /sfs_turbo/xiyun/sft/ckpt/glm5_small/  --device npu --impl transformers  --dump_io /tmp/debug_io
python example/glm_moe_dsa/load_glm_moe_dsa.py --path /sfs_turbo/xiyun/sft/ckpt/glm5_small/  --device npu --impl torch_turbo  --dump_io /tmp/debug_io --compare_io transformers

#python example/glm_moe_dsa/load_glm_moe_dsa.py --compare_dirs '/tmp/debug_io/torch_turbo,/tmp/debug_io/transformers'