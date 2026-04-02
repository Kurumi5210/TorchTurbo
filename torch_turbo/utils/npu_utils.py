import torch_npu
from transformers.utils import is_torch_npu_available

def get_profiler():
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=2),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("/tmp/profile"),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config
    )
    return prof

def seed_all(seed=1234):
    import random
    import numpy as np
    import torch
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_torch_npu_available:
        import torch_npu
        os.environ['HCCL_DETERMINISTIC'] = str(True)
        os.environ['LCCL_DETERMINISTIC'] = str(1)
        os.environ['CLOSE_MATMUL_K_SHIFT'] = str(1)
        os.environ['ATB_MATMUL_SHUFFLE_K_ENABLE'] = "0"
        os.environ['ATB_LLM_LCOC_ENABLE'] = "0"
        torch.use_deterministic_algorithms(True)
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
