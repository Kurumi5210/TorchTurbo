# Qwen3.8 global_step_176 DPO 实验

实验分支：`feat/qwen38-dpo-global176-experiment`。
基于已整理的 Qwen3.8 模板与数据适配提交 `be5e24dd`。

2026-10-29 的实际训练已完成 40/40 步，运行版本为 `0569b218`。
保存的启动脚本、训练脚本及 YAML 已与该版本核对，运行环境、完整微批
记录和模型导出检查见 [NPU_RUN_20261029.md](NPU_RUN_20261029.md)。

参考 NAS 上 `torchturbo-dsv4-sync8-d4f-pr24` 的组织方式：独立代码目录、
根目录启动入口、安装日志和独立训练输出目录。模型、数据、并行设置和
输出目录集中在 [launch_experiment.sh](launch_experiment.sh)。

代码目录：`/mnt/sfs_turbo/cx/workspace/torchturbo-qwen38-dpo-global176`。

在两节点作业中使用同一条命令：

```python
command += 'bash /mnt/sfs_turbo/cx/workspace/torchturbo-qwen38-dpo-global176/launch_experiment.sh '
```

平台需提供主节点 `MASTER_ADDR`，以及节点序号 `NODE_RANK` 或 `RANK`
（分别为 0、1）。入口先安装本仓库的 Qwen3.8 Swift 依赖，再启动 DPO；
安装失败会直接退出。Swift 固定为 4.4.2、binpacking 固定为 1.5.2。
新克隆的仓库还需通过 `git lfs pull` 获取 `docker/npu/pkg/` 中的依赖
wheel；已部署的 NAS 代码目录已包含这些文件。

代码索引：

| 内容 | 文件 |
| --- | --- |
| 固定实验参数、安装与启动日志 | [launch_experiment.sh](launch_experiment.sh) |
| NPU 参数检查入口 | [run_npu_alignment.sh](internal/scripts/runs/qwen3_8_dense/run_npu_alignment.sh) |
| 完整训练命令与启动清单 | [swift_text_dpo_debug.sh](internal/scripts/runs/qwen3_8_dense/swift_text_dpo_debug.sh) |
| SP、重计算及 loss chunk 配置 | [swift_text_dpo_debug.yaml](internal/scripts/runs/qwen3_8_dense/swift_text_dpo_debug.yaml) |
| Qwen3.8 模板适配 | [swift_qwen3_8.py](src/torchturbo/models/qwen3_5/swift_qwen3_8.py) |
| 模板注册及监督范围接入 | [swift_template.py](src/torchturbo/models/qwen3_5/swift_template.py) |
| 保留验证集划分的参数初始化 | [args.py](src/torchturbo/adapters/msswift/args.py) |
| 数据处理流程接入 | [pipeline.py](src/torchturbo/adapters/msswift/pipeline.py) |
| 采样、逻辑序列指纹与 trace | [dpo_alignment.py](src/torchturbo/adapters/msswift/dpo_alignment.py) |
| Trainer 实际消费记录接入 | [trainer.py](src/torchturbo/adapters/msswift/trainer.py) |
| GPU 侧数据 trace 插件 | [megatron_dpo_trace.py](internal/scripts/diagnostics/megatron_dpo_trace.py) |
| 完整步骤的数据比较器 | [compare_dpo_traces.py](internal/scripts/diagnostics/compare_dpo_traces.py) |

模板与采样的回归检查分别位于
`tests/adapters/msswift/test_qwen38_template.py`、
`tests/adapters/msswift/test_dpo_alignment.py`，启动参数检查位于
`tests/scripts/test_qwen38_dpo_debug.py`。

实验配置：

| 项目 | 配置 |
| --- | --- |
| policy / reference | NAS 新传的 18 分片 `global_step_176`，两者相同 |
| 数据 | `business_user_flag_answer.json`，1% 留出，seed/data_seed=42 |
| 训练 | BF16，全参 DPO，MTP 关闭，一轮 |
| 并行 | 两节点，每节点 16 NPU；SP16、DP2 |
| batch | micro=1，GBS=24，框架 GA=1，由引擎得到实际 GA=12 |
| 序列 | max_length=packing_length=128000，padding_free、binpack |
| 目标 | sigmoid，beta=0.1，RPO=0.1，label_smoothing=0.05 |
| 学习率 | cosine，1e-6 至 1e-7，warmup_ratio=0.05 |
| 基座记录 | node 0 记录模型、reference、数据及全部权重分片哈希 |

输出目录：
`/mnt/sfs_turbo/cx02308786/ckpt/qwen38-dpo-global176-aligned-20261029`。
安装及启动日志是 `bootstrap-0.txt` / `bootstrap-1.txt`，训练日志是
`log-0.txt` / `log-1.txt`，实际消费记录在 `data-trace/`。
再次做独立重复实验前，修改入口中的输出目录。

仅检查命令展开时，在已设置节点信息的环境中执行：

```bash
DRY_RUN=1 bash launch_experiment.sh
```

该检查不会安装依赖或启动训练。完整 GPU/NPU 输入对齐尚未验证通过：
历史 v7 与 NPU 的 960 个 pack 中有 244 个长度构成不同。
本实验保留逐微批校验能力，新的两端运行仍须比较完整 trace。
模板、采样和验证边界见
[DPO_DEBUG.md](internal/scripts/runs/qwen3_8_dense/DPO_DEBUG.md)。
