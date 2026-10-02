# TorchTurbo
A supernode-native post-training engine for rapidly adapting and training large language and vision models.

## 总览
当前大模型的后训练面临多重挑战：模型结构持续演进，需支持 MoE、多模态、稀疏注意力等创新架构；RL和SFT/CPT/DPO 等后训练流程既要灵活适配不断变化的模型结构，又要组合运用 N-d 并行能力来支撑高性能训练。为满足模型的快速迭代需求，亟需一套专注于模型与训练逻辑本身、简洁通用且高性能的分布式训练引擎。

与此同时，AI 基础设施正迈入超节点（SuperNode）时代，其强大的通信能力为超大参数 MoE 模型的训练提供了更优的支撑。基于这样的背景，我们推出 TorchTurbo —— 套专注于模型与训练逻辑本身的分布式训练引擎， 通过融合算子的深度优化与超节点亲和性（充分利用超节点内节点间通信损耗低的优势）来提升模型的适配和训练效率。

TorchTurbo 是一个基于PyTorch FSDP2和Huggingface Transformers构建的分布式训练引擎，通过组合 **全分片数据并行 (Fully-Shard)、专家并行 (EP)、和序列并行 (Ulysess-SP) **为全模态模型灵活搭建高效的并行训练方案。其架构如下所示：

```text
                         ┌─────────────────────────────────────┐
                         │              Model                  │
                         │   (Native HuggingFace Extension)    │
                         │                                     │
                         │   - 定义自身结构 (Architecture)       │
                         │   - 声明并行策略 (ParallelPlan)       │
                         │   - 指定计算需求 (Precision, AC)      │
                         └──────────────┬──────────────────────┘
                                        │
                   ┌────────────────────┼────────────────────┐
                   │                    │                    │
                   ▼                    ▼                    ▼
             ┌──────────┐        ┌──────────┐        ┌─────────────┐
             │  Backend │        │ Adapter  │        │ ParallelPlan│
             │  (执行层) │        │ (适配层)  │        │   (策略层)   │
             │          │        │          │        │             │
             │ - FSDP2  │        │ - LF v0  │        │ - SP/EP/DP  │
             │          │        │ - swift  │        │ - Offload   │
             └──────────┘        └──────────┘        └─────────────┘
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────────────┐
                         │            Engine                   │
                         │            (编排层)                  │
                         │                                     │
                         │  prepare_model → prepare_optimizer  │
                         │              → forward_backward     │
                         └─────────────────────────────────────┘
```

核心原则：

1. **忠于模型原生实现 (Model-faithful)：**采用纯净的开源 HuggingFace 模型实现，仅做模块级替换以实现高性能 (e.g. MoE、Attention、FusionOps)，便于维护与扩展。
2. **简洁的并行训练 (Simple parallel-training transformations)：**以简洁的训练并行方案，仅在FSDP2基础上叠加EP和SP，实现高训练吞吐量。
3. **可插拔接入主流训练框架 (Pluggable into training frameworks)：**训练引擎本身聚焦模型与训练逻辑。
4. **超节点原生 (Supernode-native)：**面向超节点优化的训练引擎。

### Model
每个模型目录负责三件事：

+ 注册模型类型
+ 实现 `TurboModel` 或 `TurboVLMModel` 契约及模型特有 hook
+ 通过 `get_default_parallel_plan()` 提供模型默认并行策略

典型结构：

```text
src/torchturbo/models/<model_name>/
├── __init__.py
├── model.py
├── parallel_plan.py
├── configuration_*.py       # 本地 Transformers 配置
├── modeling_*.py            # 本地 Transformers 模型实现
├── lf_template.py            # 可选，LlamaFactory 集成
├── swift_template.py         # 可选，ms-swift 集成
└── npu.py                    # 可选，设备特有 patch
```

模型需要在本地维护上游 Transformers 代码时，在目录中提供对应的 `configuration_*.py` 和 `modeling_*.py`。

### ParallelPlan
模型通过 `FSDPParallelPlan` 声明分布式执行策略：

```python
from torch.distributed._tensor import Shard
from torchturbo.parallel import FSDPParallelPlan, FSDPPlacementSpec, FSDPPlanSpec


def get_parallel_plan() -> FSDPParallelPlan:
    return FSDPParallelPlan(
        plan={
            "model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
        },
        sp_size=1,
        ep_size=1,
    )
```

包装类通过 `get_default_parallel_plan()` 返回该策略。引擎随后将 `EngineConfig.parallel_plan` 合并到默认策略中，其中包括通过环境变量、配置文件或 CLI 参数指定的运行时 `sp_size` 和 `ep_size`。

### Backend
当前 backend：

+ `fsdp2`: 基于 PyTorch FSDP2 / DTensor 的权重加载、分片、保存和优化器准备

后续 backend 应通过 `src/torchturbo/engine/backend_registry.py` 注册。

### Adapter
当前已注册的 adapter：

+ `llamafactory_v0`: hook LlamaFactory 的模型加载、训练和保存流程
+ `msswift`: 以插件形式接入 ms-swift

TorchTurbo 目前只支持训练，不支持评估、预测和生成，无论是否开启序列并行。
LlamaFactory 和 ms-swift adapter 会拒绝不满足以下条件的配置：`eval_strategy="no"`、
`do_eval=False`、`do_predict=False`、`predict_with_generate=False`、`eval_on_start=False`。

SFT 样本始终相互隔离，包括 LlamaFactory 的 `packing=True, neat_packing=False`。
`train_on_prompt=True` 只增加各样本内部的 prompt 监督，不产生跨样本的预测目标。

`src/torchturbo/adapters/rl/` 提供强化学习框架可复用的 packed-sequence / SP 工具；它是工具包，不是已注册的 adapter。

模型如需注册 LlamaFactory template，可在模型目录提供 `lf_template.py`。  
Adapter 激活时会扫描已注册模型并自动调用对应 `register()`。

## 快速开始
### 构建Docker

```bash
cd docker/npu/

docker build \
--platform linux/arm64 \
-f Dockerfile_aarch64 \
-t torchturbo:aarch64 \
```
平台依赖脚本位于 `scripts/runs/deps/`，模型特有的依赖安装脚本位于 `scripts/runs/<model>/`。

### 使用方法
训练从 torchturbo 提供的入口启动，训练框架（LlamaFactory/ms-swift）会作为依赖库被调用。   

`scripts/runs/run_lf.py` 会显式激活 `llamafactory_v0`，再调用 LlamaFactory 的 `run_exp()`；   
`scripts/runs/run_swift.py` 会解析 TorchTurbo 参数并激活 `msswift` 插件，再分发到所选训练 stage。

使用指导详见：[Qwen3.5 MoE 训练指南](docs/runs/qwen3_5_moe/README_NPU.md)。

### 配置说明
torchturbo 自有配置有三种配置方法，优先级从低到高：

```text
adapter 默认值 < TORCH_TURBO_* 环境变量 < --tt_config YAML 配置文件 < 显式 --tt_* CLI 参数
```

`--tt_*` 参数定义见 [`src/torchturbo/engine/args.py`](src/torchturbo/engine/args.py)。

YAML 配置文件示例见 [`scripts/runs/config_example.yaml`](scripts/runs/config_example.yaml)：

```bash
torchrun --nproc_per_node=8 scripts/runs/run_lf.py \
  --tt_config scripts/runs/config_example.yaml \
  <LLaMA-Factory arguments>
```

`--tt_*` 参数在转发给训练框架前会被剥离，不会污染原框架的参数解析。

### 运行脚本
常用训练脚本按模型分目录放在 `scripts/runs/`（`run_lf.py` / `run_swift.py` 为共享入口）：

```text
scripts/runs/
├── run_lf.py                 # LlamaFactory 入口
├── run_swift.py              # ms-swift 入口
├── config_example.yaml       # --tt_config 配置示例
├── deps/                      # 基础依赖
│   ├── gpu.sh                 # GPU 依赖
│   ├── npu.sh                 # NPU 依赖
│   ├── lf_deps.sh             # LlamaFactory 依赖
│   └── swift_deps.sh          # ms-swift 依赖
└── qwen3_5_moe/
    ├── install_lf_deps.sh    # LlamaFactory 模型依赖
    ├── install_swift_deps.sh # ms-swift 模型依赖
    ├── lf_text_sft.sh        # LlamaFactory SFT 训练脚本
    ├── swift_mllm_sft.sh     # ms-swift 多模态 SFT 训练脚本
    └── swift_text_dpo.sh     # ms-swift DPO 训练脚本
```

示例：

```bash
bash scripts/runs/qwen3_5_moe/lf_text_sft.sh
```

运行前通常需要检查：

+ `model_name_or_path`
+ `output_dir`
+ torchturbo `PYTHONPATH`
+ `NNODES` / `NPUS_PER_NODE` / `MASTER_ADDR` / `MASTER_PORT`
+ `TORCH_TURBO_SP_SIZE` / `TORCH_TURBO_EP_SIZE`（或 `--tt_sp_size` / `--tt_ep_size`）
+ `TORCH_TURBO_GRADIENT_CHECKPOINTING` / `--tt_gradient_checkpointing`
+ `--tt_cpu_offload` 或 `--tt_config` 文件中的 `backend_config.cpu_offload`

### 支持模型
内置模型从 `src/torchturbo/models/` 自动发现。模型的公开环境说明和运行文档位于 `docs/runs/`：

```text
docs/runs/
└── qwen3_5_moe/
```

| 模型 | 注册名 | 类型 | 运行文档 |
| --- | --- | --- | --- |
| Qwen3.5 MoE | `qwen3_5_moe` | VLM / MoE |  `docs/runs/qwen3_5_moe/README_NPU.md` |


## 环境变量说明
| 环境变量 | 说明 |
| --- | --- |
| `TORCH_TURBO_DISABLE=1` | 禁用未显式指定 adapter 名称时通过 `activate()` 执行的检测式激活 |
| `TORCH_TURBO_BACKEND=fsdp2` | 选择分布式 backend |
| `TORCH_TURBO_MIXED_PRECISION=bf16` | 混合精度策略 |
| `TORCH_TURBO_OP_POLICY=auto` | 算子分发策略：`auto` / `native` / `custom` |
| `TORCH_TURBO_GRADIENT_CHECKPOINTING=true` | activation checkpointing |
| `TORCH_TURBO_AC_USE_REENTRANT=true` | 使用 reentrant AC，降低 MoE/线性注意力的跨 step 显存峰值 |
| `TORCH_TURBO_USE_DETERMINISTIC_ALGORITHMS=true` | 要求所选 device runtime 使用确定性算法 |
| `TORCH_TURBO_SP_SIZE=2` | Sequence Parallel size |
| `TORCH_TURBO_EP_SIZE=8` | Expert Parallel size |
| `TORCH_TURBO_GBS=64` | 以 micro-batch 为单位设置全局 batch size；引擎根据 GBS、SP size 和 world size 推导梯度累积步数 |
| `TORCH_TURBO_RESHARD_AFTER_FWD=0` | forward 后是否 reshard |
| `TORCH_TURBO_PREFETCH_STRATEGY=auto` | FSDP prefetch 拓扑：`native` / `auto`（默认）/ `expert_lookahead` / `expert_with_layer` |
| `TORCH_TURBO_PREFETCH_DIRECTION=forward_backward` | prefetch 作用的 pass：`forward` / `forward_backward`（默认） |
| `TORCH_TURBO_FORWARD_PREFETCH_DEPTH=1` | forward 每个 source 预取多少层（默认 1） |
| `TORCH_TURBO_BACKWARD_PREFETCH_DEPTH=1` | backward 每个 source 预取多少层（默认 1，仅 forward_backward 生效） |
| `TORCH_TURBO_DEBUG=1` | 开启 MoE 层额外的 rank-0 调试输出 |


## 命令行参数说明
大部分环境变量都有对应的 `--tt_*` CLI 参数（优先级更高），全部定义见  
[`src/torchturbo/engine/args.py`](src/torchturbo/engine/args.py)。按类别：

| 类别 | 参数 |
| --- | --- |
| 基础配置 | `--tt_config` / `--tt_backend` / `--tt_mixed_precision` / `--tt_op_policy` / `--tt_use_deterministic_algorithms` |
| 并行 | `--tt_sp_size` / `--tt_ep_size` |
| Batch size | `--tt_gbs` |
| 显存 | `--tt_gradient_checkpointing` / `--tt_ac_use_reentrant` / `--tt_cpu_offload` |
| 预取 | `--tt_prefetch_strategy` / `--tt_prefetch_direction` / `--tt_forward_prefetch_depth` / `--tt_backward_prefetch_depth` |
| MTP / MoE loss | `--tt_enable_mtp` / `--tt_mtp_loss_scaling_factor` / `--tt_mtp_log_loss` / `--tt_moe_aux_loss_coeff` / `--tt_moe_z_loss_coeff` / `--tt_enable_chunk_loss` / `--tt_loss_chunk_size` |
| VLM 冻结 | `--tt_freeze_vision_tower` / `--tt_freeze_multi_modal_projector` / `--tt_freeze_language_model` |
| 扩散模型（LLaDA2） | `--tt_cap_lambda` / `--tt_bd_mask_low` / `--tt_bd_mask_high` / `--tt_bd_block_size` |
| Checkpoint | `--tt_default_save_format` |
| 性能分析、指标与日志 | `--tt_profiling` / `--tt_memory_snapshot_steps` / `--tt_log_mfu` / `--tt_card_name` / `--tt_log_level` |


## 添加新模型
1. 在 `src/torchturbo/models/` 下创建模型目录。
2. 在 `model.py` 中让语言模型继承 `TurboModel`、多模态模型继承 `TurboVLMModel`，并用 `@register_model()` 注册模型类。
3. 在 `parallel_plan.py` 中定义默认并行策略，并通过类方法 `get_default_parallel_plan()` 返回该策略。
4. 如有需要，添加 `configuration_*.py` 和 `modeling_*.py`。
5. 相应训练框架需要模型特有注册时，添加 `lf_template.py` 或 `swift_template.py`。
6. 仅当模型需要在对应生命周期执行初始化时，覆盖 `on_parallelized()` 或 `on_materialized()` 等 hook。
7. 添加 `scripts/runs/<model>/` 启动脚本和 `docs/runs/<model>/` 运行文档。

最小包装类：

```python
from transformers import AutoModelForCausalLM

from torchturbo.parallel import FSDPParallelPlan

from .configuration_my_model import MyConfig
from .modeling_my_model import MyModelForCausalLM
from ..base import TurboModel
from ..registry import register_model


@register_model("my_model", config_cls=MyConfig, auto_class=AutoModelForCausalLM)
class TurboMyModelForCausalLM(TurboModel, MyModelForCausalLM):
    config_class = MyConfig

    @classmethod
    def get_default_parallel_plan(cls) -> FSDPParallelPlan:
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()

    def on_materialized(self) -> None:
        """Initialize runtime buffers after parameters are materialized."""
```

`on_materialized()` 是可选 hook；模型没有需要重建的运行时 buffer 时可直接省略。该 hook 会在 checkpoint 加载或设备 materialization 完成后执行。模块替换和分片完成后立即需要执行的初始化，应放在 `on_parallelized()` 中。

## 引用
如果您觉得此项目有帮助，请考虑以下列格式引用

```bibtex
@misc{torchturbo2026,
  title        = {TorchTurbo: A supernode-native post-training engine for rapidly adapting and training large language and vision models.},
  author       = {Tai An and Haishan Feng and Xuejin He and Yicheng Shan and Wei Tang and Chao Wang and Yifei Yang and Liangji Zhuang},
  year         = {2026},
  url          = {},
  license      = {Apache-2.0}
}
```

本项目受益于Megatron-LM，感谢以上项目作者的付出。

## 许可证
Apache-2.0
