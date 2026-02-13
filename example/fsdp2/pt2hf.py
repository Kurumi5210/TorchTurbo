from torch_turbo.moe.moe_utils import MoeUtils
from torchturbo.torch_turbo.distributed.parallel_state import initialize_model_parallel
import torch.distributed as dist
import torch_npu

def setup(args, backend="nccl"):
    """初始化进程组，必须在所有分布式操作前调用"""
    dist.init_process_group(backend=backend)
    initialize_model_parallel(expert_model_parallel_size=args.ep_size)
    torch_npu.npu.set_device(dist.get_rank() % torch_npu.npu.device_count())
    print(f"Rank {dist.get_rank()} initialized")

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=16)
    parser.add_argument("--ep_size", type=int, default=4)
    parser.add_argument("--save_path", type=str, required=True, help="path to the saved checkpoint")
    parser.add_argument("--hf_save_path", type=str, required=True, help="path to the hf format save path")
    parser.add_argument("--config_path", type=str, required=True, help="path to the model config")
    parser.add_argument("--average_in_collective", default=True, action="store_true")
    parser.add_argument("--use_fused_swiglu", default=False, action="store_true")
    parser.add_argument("--use_fused_rope", default=False, action="store_true")
    parser.add_argument("--use_fused_rms_norm", default=False, action="store_true")
    parser.add_argument("--use_fused_permute_unpermute", default=False, action="store_true")
    args = parser.parse_args()

    setup(args, backend="hccl")

    MoeUtils.pt2hf(args, args.save_path, args.hf_save_path, args.config_path)

    cleanup()
