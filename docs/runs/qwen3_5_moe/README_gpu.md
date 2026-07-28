# Qwen3 MoE

## 运行脚本

`scripts/runs/gpu_lf_qwen3_5_moe_text_sft.sh`

## AIStudio 环境

```python
#!/usr/bin/env python3
"""Launch a distributed training job on the AIStudio/KubeMaker platform."""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ["ENV_AISTUDIO_HOST"] = "aistudio.alipay.com"

from pypai.conf import ExecConf, KMConf
from pypai.job import PythonJobBuilder
from aistudio_common.openapi.models import DataStore


# ── Job Configuration ──────────────────────────────────────────────────

# Hardware
NUM_GPUS = 8
GPU_TYPE = "h20-3e"
GPU_PER_NODE = 8
CPU_PER_NODE = 80
MEMORY_PER_NODE = 1024000      # MiB
SHARED_MEMORY = 1024000        # MiB
DISK_SIZE = 409600             # MiB

# Runtime
IMAGE = "reg.docker.alibaba-inc.com/aii/aistudio:aistudio-311588595-56586242-1781508033657"
RUNTIME = "pytorch"
AIS_RESOURCE_POOL = "thetainfra"
PRIORITY = "high"
HOST_NETWORK = False
RDMA = True


# Debug vs production launch command
REAL_CMD = (
    "cd /input/asystem/xiyun.wc/ && python qwen3_5.py && "
    "cd /input/asystem/xiyun.wc/torchturbo && bash scripts/runs/gpu_deps/qwen3_5_moe_deps.sh && "
    "bash scripts/runs/gpu_lf_qwen3_5_moe_text_sft.sh"
)

run_command = f"rm -rf /usr/local/cuda/compat && sleep infinity || sleep infinity"


# DIMA (data image) — only needed for production
DIMA = "2026052800116393486"

# Tag
TAG = "type=qwen35-gdpo"

# ── Pip Dependencies ───────────────────────────────────────────────────

PIP_PACKAGES = [
    # Framework
    "transformers==4.45.0",
    "deepspeed==0.13.2",
    "accelerate==0.33.0",
    "pytorch-lightning==1.8.3.post1",
    "-U packaging",
    "-U --no-deps dlrover[torch]==0.4.7",
    # Prebuilt wheels (URLs / local files)
    "-U http://alipay-cognition.cn-hangzhou.alipay.aliyun-inc.com/datacube2/packages/debugger/zy267523/lib/6300124-20231218141320/flash_attn-2.3.6%2Bpack.glm.mask-cp38-cp38-linux_x86_64.whl",
    "--no-deps atorch-1.3.2rc2-py3-none-any.whl",
    "alipay_pcache-0.1.8-py2.py3-none-any.whl",
    "http://alipay-kepler-resource.cn-hangzhou.alipay.aliyun-inc.com/pcache/unifile/20240925/unifile_sdk-0.1.19-12.g5c8f246-py3-none-any.whl",
    "funasr-1.0.19-py3-none-any.whl",
    "openai-whisper-20231117.tar.gz",
    "xformers-0.0.22.post7-cp38-cp38-manylinux2014_x86_64.whl",
    "soundfile-0.12.1-py2.py3-none-manylinux_2_17_x86_64.whl",
]


def _build_pip_install_command() -> str:
    """Build the pip install prefix + per-package install commands."""
    pip_index = "pip config set global.index-url https://pypi.antfin-inc.com/simple/"
    ld_path = (
        "export LD_LIBRARY_PATH="
        "/root/ffmpeg/lib:/opt/conda/lib/python3.8/site-packages/aistudio_common/reader/libs/:"
        "/opt/taobao/java/jre/lib/amd64/server/:/usr/local/cuda/lib64:/lib64:/usr/local/lib"
    )
    setup = f"echo '{ld_path}' >> ~/.bashrc && source ~/.bashrc && {pip_index} && ldconfig"
    return setup
    installs = " && \\\n  ".join(f"pip install {pkg}" for pkg in PIP_PACKAGES)
    return f"{setup} && \\\n  {installs} && \\\n"


# ── Data Stores ────────────────────────────────────────────────────────

DATA_STORES = [
    DataStore(
        mount_point="/input",
        store_name="myun-bmcpfs",
        sub_path="",
    ),
    DataStore(
        mount_point="/mnt/bailingmm-chat/ckpt",
        store_name="pcache-bailingmm-chat",
        sub_path="/mnt/bailingmm-chat/ckpt",
    ),
    DataStore(
        mount_point="/mnt/m-multimodel/dataset",
        store_name="m-multimodel-dataset",
        sub_path="/mnt/m-multimodel/dataset",
    ),
]

# ── K8s Labels ─────────────────────────────────────────────────────────

LABELS = {
    '"kubemaker.alipay.com/enable-privilege"': "true",
}

# ── Build & Submit ─────────────────────────────────────────────────────

def _build_command() -> str:
    """Assemble the full command that runs on each pod."""
    tfevent_dir = "/home/admin/logs/tfevent/"
    log_path = "/home/admin/logs/kubemaker_trainer.log"
    pip_cmd = _build_pip_install_command()

    return (
        f"pwd && ls -al && ldconfig && nvidia-smi && printenv | sort && "
        f"mkdir -p {tfevent_dir} && "
        f"{pip_cmd} && "
        f"{run_command} 2>&1 | tee {log_path}"
    )


def main():
    worker_num = NUM_GPUS // GPU_PER_NODE - 1

    master_conf = ExecConf(
        cpu=CPU_PER_NODE,
        memory=MEMORY_PER_NODE,
        gpu_num=NUM_GPUS,
        num=1,
        shared_memory=SHARED_MEMORY,
        gpu_type=GPU_TYPE,
        disk_m=DISK_SIZE,
    )
    worker_conf = ExecConf(
        cpu=CPU_PER_NODE,
        memory=MEMORY_PER_NODE,
        gpu_num=GPU_PER_NODE,
        num=worker_num,
        shared_memory=SHARED_MEMORY,
        gpu_type=GPU_TYPE,
        disk_m=DISK_SIZE,
    )

    builder = PythonJobBuilder(
        source_root="",
        main_file="",
        platform="kubemaker",
        k8s_app_name=AIS_RESOURCE_POOL,
        k8s_priority=PRIORITY,
        km_conf=KMConf(image=IMAGE),
        master=master_conf,
        worker=worker_conf,
        runtime=RUNTIME,
        data_stores=DATA_STORES,
        labels=LABELS,
        rdma=RDMA,
        tag=TAG,
        host_network=HOST_NETWORK,
        name="Qwen35-35B-gdpo",
        command=_build_command(),
    )
    builder.run(enable_wait=False, dima=DIMA)


if __name__ == "__main__":
    main()
```

## 运行方式

GPU上加载模型
```bash
cd /input/asystem/xiyun.wc/
python qwen3_5.py
```
其中`qwen3_5.py`的内容为，主要是直接利用aistudio提供的方式挂载模型权重，
```python
#pip install -U aistudio_checkpoint
from aistudio_checkpoint.supporter.moe_supporter import lazy_load_moe_checkpoint

load_path = lazy_load_moe_checkpoint(resuming_path="aistudio://18265422/rootmodelQwen3.5-35B-A3B-Base-hgf-asys")
print(load_path) # 挂载路径点，可直接读取ckpt
```

### llamafactory运行命令

安装依赖
```bash
cd /input/asystem/xiyun.wc/torchturbo
bash scripts/runs/gpu_deps/qwen3_5_moe_deps.sh
```

运行训练
```bash
cd /input/asystem/xiyun.wc/torchturbo
bash scripts/runs/gpu_lf_qwen3_5_moe_text_sft.sh
```

### swift运行命令

安装依赖
```bash
cd /input/asystem/xiyun.wc/torchturbo
bash scripts/runs/gpu_deps/qwen3_5_moe_swift_deps.sh
```

运行训练
```bash
cd /input/asystem/xiyun.wc/torchturbo
bash scripts/runs/gpu_swift_qwen3_5_moe_mllm_sft_debug.sh
```