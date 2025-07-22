from torch_turbo.moe.moe_utils import load_safetensors_dir
from torch_turbo.utils.torch_utils import compare_tensors

def main(args):
    sd = load_safetensors_dir(args.save_path)
    golden_sd = load_safetensors_dir(args.golden_path)
    for key, value in sd.items():
        if key not in golden_sd:
            # print(f"Key {key} not found in golden state dict.")
            continue
        golden_value = golden_sd[key]
        compare_tensors(key, value, golden_value, rank=0, atol=0.0)


# command:
#   export PYTHONPATH=/sfs_turbo/xiyun/rl/3d/opensource/Megatron-LM:/sfs_turbo/dsh/MindSpeed:$PYTHONPATH
#   python example/fsdp2/test_save_weight.py
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="/sfs_turbo/qinjue/models/Qwen3-30B-A3B/")
    parser.add_argument("--golden_path", type=str, default="/sfs_turbo/pretrained_models/Qwen3-30B-A3B")
    args = parser.parse_args()

    main(args)