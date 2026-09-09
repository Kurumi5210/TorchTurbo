# Qwen35 MoE NPU

## AIStudio 环境

```python
import os
from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf
from pypai.conf import KMConf
from pypai.conf import GpuType
from pypai.conf import AlertConf
from aistudio_common.openapi.models import DataStore

def update_pypai():
    # install dependencies
    import subprocess
    try:
        subprocess.run(["pip", "install", "--upgrade", "aii-pypai"], check=True)
        print("pip executes succeeded!")
    except subprocess.CalledProcessError as e:
        print(f"pip executes failed with code {e.returncode}")

def torch_train():
    update_pypai()
    a3_sfs = DataStore(mount_point="/sfs_turbo", store_name="wl02-02-aidc-antfinix-sfs")
    mnt_sfs = DataStore(mount_point="/mnt/sfs_turbo", store_name="wl02-02-aidc-aicompiler-sfs")
    # a3_pcache = DataStore(store_name="ais-pcache-a3-common-antfinix2", mount_point="/mnt/antfinix2-a3/common")
    a3_aic_pcache = DataStore(store_name="ais-pcache-a3-common-aicompiler", mount_point="/mnt/aicompiler-a3/common")
    # ais-pcache-a3-common-antfinix2 	/mnt/antfinix2-a3/common
    km_conf = KMConf(
        image="hcr.meta-wulan01.hw-wulan.local/antsys/vllm:AsystemHybirdEngine-0b9af7b5-20260824140737"
    )

    labels={
        "\"sigma.ali/rack-group-num\"": "1",  # 同超节点约束，确保任务在同一超节点内
        "\"kubemaker.alipay.com/enable-pytorch-coscheduling\"": "true"  # 确保 PyTorchJob Pod 同时创建，避免超节点调度失败
    }
    # edl_job_args = "--critical_worker_index=0:2 --relaunch_always"

    node_gpu_num = 16
    node_num = 1
    
    custom_cmd="sleep infinity"
    command = ''
    command += 'source /usr/local/Ascend/nnal/atb/set_env.sh && source /root/.bashrc && '
    command += f'export NPUS_PER_NODE={node_gpu_num} && export NNODES={node_num} && '
    command += f"pip install peft==0.15.0 && pip install trl && pip install pynvml && \
        pip install grpcio --trusted-host repo.huaweicloud.com && \
        pip list && sleep 1 && \
        cd /mnt/sfs_turbo/haishan.fhs/torchturbo/ && \
        {custom_cmd}"

    master = ExecConf(cpu=256, memory=2000000, gpu_num=node_gpu_num, num=1, shared_memory=1000000, gpu_type="hpu910c", disk_m=192000)
    worker = ExecConf(cpu=256, memory=2000000, gpu_num=node_gpu_num, num=node_num - 1, shared_memory=1000000, gpu_type="hpu910c", disk_m=192000)
    job = PythonJobBuilder(source_root=None,
                           main_file="", #命令模式必填空，详情参考：https://yuque.antfin-inc.com/aii/aistudio/gpu-job#y3tRY
                           km_conf=km_conf,
                           command=command,
                           platform="kubemaker",
                        #    k8s_app_name="deepseekmedameds1",
                           k8s_app_name="hpuopt",
                           k8s_priority="high",  
                           distribution_strategy="AllreduceStrategy",
                           master=master,
                           worker=worker,
                        #    runtime="elasticdl",
                           runtime='pytorch',
                           name="swift_torchturbo_qwen35b_630jsonl_sp2_ep4",
                           tag="type=sft,basemodel=qwen3.5_35b",
                           data_stores = [a3_sfs, mnt_sfs, a3_aic_pcache],
                           host_network=False,
                           labels=labels,
                        #    edl_job_args = edl_job_args,
                           enable_pcache_fuse=True,
                        #    annotations=annotations,
                        #    alert=alert_conf,
                          )

    job.run(enable_wait=False)

if __name__ == "__main__":
    torch_train()
```

## 运行方式

### llamafactory运行命令

// TODO

### swift运行命令

安装依赖
```bash
sh scripts/runs/qwen3_5_moe/install_swift_deps.sh
```

运行训练
```bash
sh scripts/runs/qwen3_5_moe/swift_mllm_sft.sh
```