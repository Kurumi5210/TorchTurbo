#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# deps:
# [images]            hcr.meta-wulan01.hw-wulan.local/antsys/vllm:AsystemHybirdEngine-6968dbd2-20260612130916
# [vllm]              pip install /mnt/sfs_turbo/torchturbo/stable/qwen3_5_deps/vllm-0.23.1rc1.dev752+g1f486d96a.d20260717.empty-cp311-cp311-linux_aarch64.whl
# [vllm-ascend]       pip install /mnt/sfs_turbo/torchturbo/stable/qwen3_5_deps/vllm_ascend-0.19.1rc2.dev934+g2e605e544.d20260719-cp311-cp311-linux_aarch64.whl
# [triton-ascend]     pip install /mnt/sfs_turbo/torchturbo/stable/qwen3_5_deps/triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl --no-deps --force-reinstall
# [peft]              pip install peft==0.19.1

import os
import sys
from vllm import LLM, SamplingParams

os.environ['VLLM_USE_V1'] = '1'
os.environ['VLLM_VERSION'] = '0.21.0'
os.environ['VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS'] = '380'
os.environ['NET_CARD_NAME'] = 'eth0'
os.environ['VLLM_ASCEND_ENABLE_NZ'] = '0'
os.environ['HCCL_BUFFSIZE'] = '300'
os.environ['ASCEND_BUFFER_POOL'] = '4:8'
os.environ['USE_MULTI_GROUPS_KV_CACHE'] = '1'
os.environ['USE_MULTI_BLOCK_POOL'] = '1'

print("=" * 60)
print("Environment Variables:")
for key in ['VLLM_USE_V1', 'VLLM_VERSION', 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS',
            'NET_CARD_NAME', 'VLLM_ASCEND_ENABLE_NZ', 'HCCL_BUFFSIZE',
            'ASCEND_BUFFER_POOL', 'USE_MULTI_GROUPS_KV_CACHE', 'USE_MULTI_BLOCK_POOL']:
    print(f"  {key}={os.environ.get(key)}")
print("=" * 60)


def main():
    # Sample prompts.
    prompts = [
        "Hello, who are you?"
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    llm = LLM(
        model="/sfs_turbo/pretrained_models/Qwen3.5-4B",
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        tensor_parallel_size=2,
        # mamba_cache_dtype='float32',
        max_num_seqs=16,
        gpu_memory_utilization=0.65,
        # enable_return_routed_experts=True,
        )

    outputs = llm.generate(prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
