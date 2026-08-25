# Qwen3.5 Dense
## 支持特性

| 特性 | 项目 |
| --- | --- |
| **SFT** | 基于 transformers 4.57.3 接入 qwen3.5 modeling |
| | 基于 H20/A3，对齐开源 deepspeed 0.19.2 + transformers 5.7.0 精度 |
| | 支持 ulysses SP，最大支持 SP=32/48 |
| | 对齐 turbo infer v.s. vllm-ascend infer |
| | 对齐 turbo v.s. megatron (via mbridge+mindspeed) infer，涵盖 packing 场景 |

## TODO LIST

- [ ] 高 grad_norm 问题定位 @cjs02503376 @执伞
- [ ] 接入 Ascend C 算子后验证性能/精度 @执伞 @cjs02503376
- [ ] 27B dense 模型 scaling 到 256K，更新长跑实验 @cjs02503376

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
    #a3_sfs_old = DataStore(mount_point="/sfs_turbo", store_name="wl02-antfinix-sfs")
    #a3_sfs_new = DataStore(mount_point="/mnt/sfs_turbo", store_name="wl02-aicompiler-sfs")
    a3_sfs_old = DataStore(mount_point="/sfs_turbo", store_name="wl02-02-aidc-antfinix-sfs")
    a3_sfs_new = DataStore(mount_point="/mnt/sfs_turbo", store_name="wl02-02-aidc-aicompiler-sfs")
    a3_aic_pcache = DataStore(mount_point="/mnt/aicompiler-a3/common", store_name="ais-pcache-a3-common-aicompiler")
    km_conf = KMConf(
        # qwen3.5
        image="hcr.meta-wulan01.hw-wulan.local/antsys/vllm:AsystemHybirdEngine-0b9af7b5-20260824140737"
        
    )
    labels={
        "\"sigma.ali/rack-group-num\"": "1",  # 同超节点约束，确保任务在同一超节点内i
        "\"kubemaker.alipay.com/enable-pytorch-coscheduling\"": "true"  # 确保 PyTorchJob Pod 同时创建，避免超节点调度失败
    }
    edl_job_args = "--critical_worker_index=0:2 --relaunch_always=false --relaunch_count 0"

    annotations = {
        "ELASTICDL_MASTER_CONFIGURATION": '{"cpu": 4, "memory": 2048, "image":"hcr.meta-wulan01.hw-wulan.local/kubemaker/elasticdl:test-aarch64-bc4298b59706f305893044af221fa99880ae2210"}'
    }

    node_gpu_num = 16
    node_num = 1

    command = ""
    command += f"export NPUS_PER_NODE={node_gpu_num} && export NNODES={node_num} && "
    command += f"source /usr/local/Ascend/ascend-toolkit/set_env.sh && echo 'cannVersion: ', $ASCEND_TOOLKIT_HOME && "

    command += ' sleep inf'
    name = 'test-glucose180-qwen3_5'

    master= ExecConf(cpu=256, memory=2800000, gpu_num=node_gpu_num, num=1, shared_memory=20480, gpu_type="hpu910c", disk_m=192000)
    worker = ExecConf(cpu=256, memory=2800000, gpu_num=node_gpu_num, num=node_num-1, shared_memory=20480, gpu_type="hpu910c", disk_m=192000)

    job = PythonJobBuilder(# source_root="./",
                           main_file="", #命令模式必填空，详情参考：https://yuque.antfin-inc.com/aii/aistudio/gpu-job#y3tRY
                           km_conf=km_conf,
                           command=command,
                           platform="kubemaker",
                           k8s_app_name="hpuopt",
                           k8s_priority="high",  
                           distribution_strategy="AllreduceStrategy",    
                           master=master,       
                           worker=worker,
                           runtime="pytorch",
                           name=name,
                           tag="type=SFT,basemodel=ling25",
                           data_stores = [a3_sfs_old, a3_sfs_new, a3_aic_pcache],   # a3_sfs_new is not found since 2026-07-14
                           #data_stores = [a3_sfs_old],
                           host_network=False,
                           labels=labels,
                           edl_job_args = edl_job_args,)
                          #  annotations=annotations,

    job.run()

if __name__ == "__main__":
    torch_train()
```

## 运行脚本

安装依赖：

```bash
bash scripts/runs/npu_deps/qwen3_5_moe_deps.sh
```

| 任务 | 脚本 |
| --- | --- |
| TEXT SFT | `scripts/runs/qwen3_5_dense/lf_text_sft.sh` |
| TEXT CPT | `scripts/runs/qwen3_5_dense/lf_text_cpt.sh` |
| MLLM SFT | `scripts/runs/qwen3_5_dense/lf_mllm_sft.sh` |
| MLLM CPT | `scripts/runs/qwen3_5_dense/lf_mllm_cpt.sh` |

---

## 精度对齐报告

基于 Qwen3.5-4B:与 deepspeed 对齐用单机 8 卡,与 vllm_infer / megatron 对齐用单机单卡。

**结论:训练侧 `loss/grad` 精度达标,可交付;前向 logp 差异来自 GDN 算子跨框架实现差异。**

### 与 deepspeed 精度对齐

实验:Qwen3.5-4B,单机,8K 序列长度,science 数据集。loss / grad_norm 对齐曲线:

![loss](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/222456468/1784793830977-c8754d01-5036-4ec3-b582-b1fe9dfc37fd.png)

![grad_norm](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/222456468/1784794420346-e1aaf533-a497-4668-ace0-71d9fde7ec96.png)

### 与 vllm_infer 精度对齐

vllm 是推理引擎,仅做前向图比对(无 loss/grad):

```bash
# turbo 训练侧 vs vllm 推理侧(前向图)
bash scripts/rl/run_turbo_vs_vllm.sh
```

| 指标 | turbo vs vllm (seq=122) |
| --- | --- |
| 前向 logp mean abs diff | 3.08e-2 |
| **前向 logp max abs diff** | **1.36e-1 ✅** |
| 达 ≤1e-3 位置占比 | 24% |

### 与 megatron 精度对齐

```bash
# turbo vs megatron,前向图 + loss/grad
bash scripts/rl/run_turbo_vs_megatron.sh           # 非 packed
bash scripts/rl/run_turbo_vs_megatron.sh --packed  # 3 序列 packed
```

| 阶段 | 产物 | 说明 |
| --- | --- | --- |
| [1/4] gen_input | `input_spec.pt` | 共享输入 |
| [2/4] megatron | `{graph.pt,loss_grad.json,logp.npy}` | MCore via mbridge + mindspeed |
| [3/4] turbo | 同上 | TorchTurboEngine |
| [4/4] compare | `report.json` | 前向逐 token logp + 模块图比对 |
| [cmp] loss_grad_compare | `loss_grad_report.json` | loss/grad 差异 |

| 指标 | turbo vs megatron 非 packed (seq=122) | turbo vs megatron packed (seq=31) |
| --- | --- | --- |
| 前向 logp mean abs diff | 3.58e-2 | 3.71e-2 |
| 前向 logp max abs diff | 1.29e-1 | 2.40e-1 |
| 达 ≤1e-3 位置占比 | 14% | 20% |
| loss abs diff | 4.7e-4 | 4.9e-3 |
| **loss rel diff** | **1.5e-4 ✅** | **1.4e-3 ✅** |
| grad abs diff | 1.96e-2 | 0.93 |
| **grad rel diff** | **3.9e-4 ✅** | **9.1e-3 ✅** |
| grad ratio (t/m) | 0.9996 | 0.991 |
| 有梯度参数 (turbo/megatron) | 426 / 306 | 426 / 306 |
| 有效 token 数 | 121 | 28 |
