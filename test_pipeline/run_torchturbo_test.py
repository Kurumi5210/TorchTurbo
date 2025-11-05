# -*- coding: utf-8 -*-

import subprocess

# subprocess.run("pip install -U aii-pypai --index=https://artifacts.antgroup-inc.cn/simple/", shell=True)
# subprocess.run("pip install -U mdatasets --index=https://artifacts.antgroup-inc.cn/simple/", shell=True)

from pypai.conf import ExecConf, KMConf
from pypai.job import PythonJobBuilder
from aistudio_common.openapi.models import DataStore

# 2. 初始化sidecar挂载点配置 (目前mount_point/sub_path参数字段不支持修改)
wl02_sfs = DataStore(mount_point="/sfs_turbo", store_name="wl02-antfinix-sfs")

data_stores = [wl02_sfs]

NODE_NUM = 1
TOTAL_GPU = NODE_NUM*16 # die
GPUS_PER_NODE = 16

total_worker_nodes = NODE_NUM
cpu_memory = 1000000 # 2T
master = ExecConf(cpu=256, memory=cpu_memory, gpu_num=GPUS_PER_NODE, num=1, shared_memory=20480, gpu_type="hpu910c", disk_m=192000)
worker = ExecConf(cpu=256, memory=cpu_memory, gpu_num=GPUS_PER_NODE, num=total_worker_nodes - 1, shared_memory=20480, gpu_type="hpu910c", disk_m=192000)
km_conf = KMConf(
    # 镜像地址。可以替换为自己的镜像
    image = "hcr.meta-wulan01.hw-wulan.local/aii/aistudio:13610121-20250923112231",  # sft/dpo
)

PATH_TO_MEGA = "/sfs_turbo/dsh/verl_run/Megatron-LM"
PATH_TO_MIND = "/sfs_turbo/dsh/verl_run/MindSpeed"

env_vars = {
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "PYTHONPATH": f"{PATH_TO_MEGA}:{PATH_TO_MIND}:$PYTHONPATH",
    "NCCL_DEBUG": "WARN",
    "VLLM_USE_V1": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "garbage_collection_threshold:0.95",
    "GLOO_SOCKET_IFNAME": "eth0",
    "TP_SOCKET_IFNAME": "eth0",
    "HCCL_SOCKET_IFNAME": "eth0",

    "VLLM_LOGGING_LEVEL": "INFO",
    "ENABLE_MOE_ALLTOALLV": "1",
    "CLOSE_MATMUL_K_SHIFT": "1",
    "HCCL_IF_BASE_PORT": "24999",
    "HCCL_EXEC_TIMEOUT": "7200",
    "HCCL_CONNECT_TIMEOUT": "7200",
    "HCCL_ASYNC_ERROR_HANDLING": "0",

    "HISTSIZE": "1000",
}

env_cmd = " && ".join(f"export {k}={v}" for k, v in env_vars.items())

commands = env_cmd
commands += " && cd /sfs_turbo/dsh/tmp3/torchturbo && "
commands += "python example/fsdp2/dummy_run_qwen3_ep_npu.py  --path /sfs_turbo/pretrained_models/Qwen3-30B-A3B"
# 设置Pod 重启策略，只要出错就重启 Pod，默认最多重启3次.
# edl_job_args = "--relaunch_always"
edl_job_args = "--enable-job-restart=False --max-num-of-job-restart=0"

def npujob():
    job = PythonJobBuilder(
        source_root='./',
        main_file=None,
        command=commands,
        km_conf=km_conf,
        k8s_app_name="antfcutrn",
        k8s_priority="high",
        master=master,
        worker=worker,
        runtime="pytorch",
        data_stores=data_stores,
        host_network=True,
        rdma=True,
        # edl_job_args=edl_job_args,
        tag="type=torchturbo,basemodel=Qwen3"
    )
    #  如果想提交任务后，可以直接退出
    job.run(enable_wait=False, dima=2025110300111764229)


if __name__ == "__main__":
    npujob()
