# Qwen3.8 Dense 27B

需要复现 Qwen3.8 DPO 模板和逐步数据对齐时，使用
[`run_npu_alignment.sh`](../../../scripts/runs/qwen3_8_dense/run_npu_alignment.sh)，
并阅读 [DPO 对齐说明](../../../scripts/runs/qwen3_8_dense/DPO_DEBUG.md)。
该入口使用 Swift 4.4.2 的 `qwen3_8` backport、SP16/DP2 和实际消费 trace。
历史 GPU v7 与 NPU 的全程输入仍有差异，不能仅凭相同 seed 或 sampler
认定每一步相同；需要两端完整 trace 比较通过。本页下方保留的是原始 DPO 入口。

## 支持范围

| 特性 | 配置 |
| --- | --- |
| 模型 | Qwen3.8-27B Dense |
| 训练方式 | 离线 DPO、全参数训练 |
| 训练平台 | A3 Ascend NPU |
| 上下文长度 | 默认 128K，可通过 `SEQ_LEN` 覆盖 |
| DPO 目标 | 固定 reference、标准 sigmoid DPO loss |
| TorchTurbo 并行 | SP=16、EP=1 |

Qwen3.8-27B 的配置沿用 Qwen3.5 Dense 的 `qwen3_5` 模型实现，因此 Swift
使用 `turbo_qwen3_5` 模型注册和 `qwen3_5` 对话模板。若权重版本或 NAS 路径
不同，运行前覆盖 `MODEL`；若 reference 使用单独的初始权重，覆盖 `REF_MODEL`。

## AIStudio 环境

```python
from pypai.conf import ExecConf, KMConf
from pypai.job import PythonJobBuilder
from aistudio_common.openapi.models import DataStore


def torch_train():
    a3_sfs = DataStore(
        mount_point="/sfs_turbo",
        store_name="wl02-02-aidc-antfinix-sfs",
    )
    mnt_sfs = DataStore(
        mount_point="/mnt/sfs_turbo",
        store_name="wl02-02-aidc-aicompiler-sfs",
    )
    a3_aic_pcache = DataStore(
        mount_point="/mnt/aicompiler-a3/common",
        store_name="ais-pcache-a3-common-aicompiler",
    )
    node_gpu_num = 16
    node_num = 1
    command = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
        f"export NPUS_PER_NODE={node_gpu_num} && "
        f"export NNODES={node_num} && sleep infinity"
    )
    job = PythonJobBuilder(
        source_root=None,
        main_file="",
        km_conf=KMConf(
            image="hcr.meta-wulan01.hw-wulan.local/antsys/"
            "vllm:AsystemHybirdEngine-0b9af7b5-20260824140737"
        ),
        command=command,
        platform="kubemaker",
        k8s_app_name="hpuopt",
        k8s_priority="high",
        distribution_strategy="AllreduceStrategy",
        master=ExecConf(
            cpu=256,
            memory=2000000,
            gpu_num=node_gpu_num,
            num=1,
            shared_memory=1000000,
            gpu_type="hpu910c",
            disk_m=192000,
        ),
        worker=ExecConf(
            cpu=256,
            memory=2000000,
            gpu_num=node_gpu_num,
            num=node_num - 1,
            shared_memory=1000000,
            gpu_type="hpu910c",
            disk_m=192000,
        ),
        runtime="pytorch",
        name="swift_torchturbo_qwen38_dense27b_dpo",
        tag="type=dpo,basemodel=qwen3.8_dense_27b",
        data_stores=[a3_sfs, mnt_sfs, a3_aic_pcache],
        host_network=False,
        labels={
            '"sigma.ali/rack-group-num"': "1",
            '"kubemaker.alipay.com/enable-pytorch-coscheduling"': "true",
        },
    )
    job.run(enable_wait=False)


if __name__ == "__main__":
    torch_train()
```

## 运行方式

安装 Swift 依赖：

```bash
sh internal/scripts/runs/qwen3_8_dense/install_swift_deps.sh
```

启动 Swift DPO：

```bash
sh internal/scripts/runs/qwen3_8_dense/swift_text_dpo.sh
```

运行脚本默认使用：

- `MODEL=/sfs_turbo/pretrained_models/Qwen3.8-27B`
- `REF_MODEL=$MODEL`
- `SEQ_LEN=131072`
- `DATASET=/mnt/sfs_turbo/torchturbo/datasets/orca_dpo_pairs/swift_orca_rlhf.jsonl`
- `BETA=0.1`、`LOSS_TYPE=sigmoid`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`、`GRADIENT_ACCUMULATION_STEPS=1`
- `LEARNING_RATE=1e-6`、`NUM_TRAIN_EPOCHS=1`、`MAX_STEPS=200`

参数通过环境变量覆盖，例如：

```bash
MODEL=/sfs_turbo/pretrained_models/Qwen3.8-27B \
REF_MODEL=/sfs_turbo/pretrained_models/Qwen3.8-27B \
SEQ_LEN=65536 MAX_STEPS=5 LEARNING_RATE=1e-6 \
sh internal/scripts/runs/qwen3_8_dense/swift_text_dpo.sh
```

脚本按 `SAVE_STRATEGY/SAVE_STEPS` 保存训练状态；从完整 checkpoint 续训时设置：

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N \
sh internal/scripts/runs/qwen3_8_dense/swift_text_dpo.sh
```
