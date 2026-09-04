# TorchTurbo

**Model-first distributed training for post-training workloads.**

`TorchTurbo` 是一个以模型为中心的分布式训练加速框架，基于FSDP2训练引擎的3D并行方案，使能GPU/NPU高性能算子，提升训练显存/计算效率。它提供Engine和Adapter，可以灵活接入现有的后训练框架（例如：LLaMA-Factory/Asystem）。由模型声明 ParallelPlan，并交给 Engine / Backend
执行分布式训练。

## 架构

torchturbo 的核心约定是 **model-first**：

```text
                         ┌─────────────────────────────────────┐
                         │              Model                  │
                         │         (First-Class Citizen)       │
                         │                                     │
                         │   - 定义自身结构 (Architecture)      │
                         │   - 声明并行策略 (ParallelPlan)      │
                         │   - 指定计算需求 (Precision, AC)    │
                         └──────────────┬──────────────────────┘
                                        │
                   ┌────────────────────┼────────────────────┐
                   │                    │                    │
                   ▼                    ▼                    ▼
             ┌──────────┐        ┌──────────┐        ┌─────────────┐
             │  Backend │        │ Adapter  │        │ ParallelPlan│
             │ (执行层) │        │ (适配层) │        │  (策略层)   │
             │          │        │          │        │             │
             │ - FSDP2  │        │ - LF v0  │        │ - SP/EP/DP  │
             │          │        │          │        │ - Offload   │
             └──────────┘        └──────────┘        └─────────────┘
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────────────┐
                         │            Engine                   │
                         │           (编排层)                  │
                         │                                     │
                         │  prepare_model → prepare_optimizer  │
                         │              → fit                  │
                         └─────────────────────────────────────┘
```

核心原则：

- 模型自描述：模型通过 `get_parallel_plan()` 声明自己如何被分布式执行。
- Backend 执行意图：Backend 负责执行模型声明的策略，不干预模型逻辑。
- Adapter 连接框架：Adapter hook 到现有训练框架，不改变模型行为。
- Engine 编排流程：Engine 协调各组件完成训练生命周期。

Device Runtime 抽象、算子注册与分发的设计和扩展方式见
[`docs/design/runtime_and_ops.md`](docs/design/runtime_and_ops.md)。

### Model

每个模型目录负责三件事：

- 注册模型类型
- 定义模型特有 patch
- 提供 `get_parallel_plan()`

典型结构：

```text
src/torchturbo/models/<model_name>/
├── __init__.py
├── model.py
├── parallel_plan.py
├── lf_template.py
├── configuration_*.py
└── modeling_*.py
```

如果模型已经在 `transformers` 中原生支持，可以直接复用上游配置和模型；
如果没有，则在模型目录中维护对应的 `configuration_*.py` 和
`modeling_*.py`。

### ParallelPlan

模型通过 `FSDPParallelPlan` 声明分布式执行策略：

```python
from torch.distributed._tensor import Shard
from torchturbo.parallel import FSDPParallelPlan, FSDPPlacementSpec, FSDPPlanSpec


def get_parallel_plan(sp_size: int = 2, ep_size: int = 8) -> FSDPParallelPlan:
    return FSDPParallelPlan(
        plan={
            "model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
        },
        sp_size=sp_size,
        ep_size=ep_size,
    )
```

`EngineConfig` 会先读取模型默认 plan，再用环境变量或显式配置覆盖
`sp_size`、`ep_size` 等运行时参数。

### Backend

当前 backend：

- `fsdp2`: 基于 PyTorch FSDP2 / DTensor 的权重加载、分片、保存和优化器准备

后续 backend 应通过 `engine/backend_registry.py` 注册。

### Adapter

当前主要 adapter：

- `llamafactory_v0`: hook LlamaFactory 的模型加载、训练和保存流程

模型如需注册 LlamaFactory template，可在模型目录提供 `lf_template.py`。
Adapter 激活时会扫描已注册模型并自动调用对应 `register()`。

## 安装

```bash
pip install -e .
```

或者直接使用 PYTHONPATH

```bash
export PYTHONPATH=<your_path>/torchturbo/src:$PYTHONPATH
```

运行 LlamaFactory 相关入口前，还需要在环境中准备：

- PyTorch / torch-npu
- Ascend CANN 运行环境
- LLaMA-Factory 及其 requirements
- 模型所需的额外依赖

## 使用入口

torchturbo 有两种使用入口。

### 原训练框架（LlamaFactory 为例）作为入口

如果训练仍然从 LlamaFactory 启动，torchturbo 作为被集成的加速库使用。
在 LlamaFactory 训练进程中导入 `torchturbo` 即可激活 adapter：

```python
import torchturbo
```

这种方式下，LlamaFactory 仍然负责启动训练，torchturbo 负责 hook 模型加载、
并行策略、权重加载和保存流程。

### torchturbo 作为入口

如果训练从 torchturbo 提供的入口启动，LlamaFactory 会作为依赖库被调用。
`scripts/runs/run_lf.py` 会导入 torchturbo，然后调用 LlamaFactory 的
`run_exp()`：

```bash
torchrun --nproc_per_node=8 scripts/runs/run_lf.py \
  --stage sft \
  --model_name_or_path /path/to/model \
  --dataset your_dataset \
  --template qwen3 \
  --output_dir ./output/my_run
```

### 运行脚本

常用训练脚本按模型分目录放在 `scripts/runs/`（`run_lf.py` / `run_swift.py` 为共享入口）：

```text
scripts/runs/
├── run_lf.py
├── run_swift.py
├── qwen3_moe/
│   └── lf_sft.sh
├── glm5/
│   └── lf_moe_dsa_sft.sh
└── bailing_v3/
    └── lf_sft.sh
```

示例：

```bash
bash scripts/runs/qwen3_moe/lf_sft.sh
bash scripts/runs/glm5/lf_moe_dsa_sft.sh
bash scripts/runs/bailing_v3/lf_sft.sh
```

这些脚本是当前 Ascend/NPU 环境下的启动模板。运行前通常需要检查：

- `model_name_or_path`
- `output_dir`
- LLaMA-Factory `PYTHONPATH`
- torchturbo `PYTHONPATH`
- `NNODES` / `NPUS_PER_NODE` / `MASTER_ADDR` / `MASTER_PORT`
- `TORCH_TURBO_SP_SIZE` / `TORCH_TURBO_EP_SIZE`
- `TORCH_TURBO_GRADIENT_CHECKPOINTING` / `--tt_gradient_checkpointing`

### 运行文档

每个模型的环境说明、脚本使用和对比图放在 `docs/runs/`：

```text
docs/runs/
├── qwen3_moe/
├── glm_moe_dsa/
├── bailing_moe_v3/
└── deepseek_v4/
```

## 支持模型

| 模型 | 注册名 | 类型 | 运行文档 |
| --- | --- | --- | --- |
| Qwen3 MoE | `qwen3_moe` | MoE | `docs/runs/qwen3_moe/README.md` |
| GLM MoE DSA | `glm_moe_dsa` | MoE / DSA | `docs/runs/glm_moe_dsa/README.md` |
| Bailing MoE V3 | `bailing_hybrid` | MoE / MLA / KDA | `docs/runs/bailing_moe_v3/README.md` |
| DeepSeekV4 | `deepseek_v4` | MoE / DSA / HC | `docs/runs/deepseek_v4/README.md` |
| Kimi K25 | `kimi_k25` | VLM / MoE | TODO |

## 环境变量

| 环境变量 | 说明 |
| --- | --- |
| `TORCH_TURBO_DISABLE=1` | 禁用 torchturbo 自动激活 |
| `TORCH_TURBO_BACKEND=fsdp2` | 选择分布式 backend |
| `TORCH_TURBO_MIXED_PRECISION=bf16` | 混合精度策略 |
| `TORCH_TURBO_OP_POLICY=auto` | 算子分发策略：`auto` / `native` / `custom` |
| `TORCH_TURBO_GRADIENT_CHECKPOINTING=true` | activation checkpointing |
| `TORCH_TURBO_SP_SIZE=2` | Sequence Parallel size |
| `TORCH_TURBO_EP_SIZE=8` | Expert Parallel size |
| `TORCH_TURBO_RESHARD_AFTER_FWD=0` | forward 后是否 reshard |
| `TORCH_TURBO_PREFETCH_STRATEGY=auto` | FSDP prefetch 拓扑：`native` / `auto`（默认）/ `expert_lookahead` / `expert_with_layer` |
| `TORCH_TURBO_PREFETCH_DIRECTION=forward_backward` | prefetch 作用的 pass：`forward` / `forward_backward`（默认）|
| `TORCH_TURBO_FORWARD_PREFETCH_DEPTH=1` | forward 每个 source 预取多少层（默认 1）|
| `TORCH_TURBO_BACKWARD_PREFETCH_DEPTH=1` | backward 每个 source 预取多少层（默认 1，仅 forward_backward 生效）|
| `TORCH_TURBO_DEBUG=1` | 开启 torchturbo 调试信息，未完全实现 |

## 添加新模型

1. 在 `src/torchturbo/models/` 下创建模型目录。
2. 在 `model.py` 中用 `@register_model()` 注册 Turbo 模型类。
3. 在 `parallel_plan.py` 中定义默认并行策略。
4. 如有需要，添加 `configuration_*.py` 和 `modeling_*.py`。
5. 如需 LlamaFactory template，添加 `lf_template.py`。
6. 添加 `scripts/runs/<model>/` 启动脚本和 `docs/runs/<model>/` 运行文档。

最小包装类：

```python
from transformers import AutoModelForCausalLM

from torchturbo.models.registry import register_model


@register_model("my_model", config_cls=MyConfig, auto_class=AutoModelForCausalLM)
class TurboMyModelForCausalLM(MyModelForCausalLM):
    config_class = MyConfig

    @classmethod
    def get_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def post_weight_load(self):
        """Initialize runtime buffers after weights are loaded."""
```

## 项目结构

```text
torchturbo/
├── docs/
│   ├── design/                # 架构设计和扩展指南
│   └── runs/                  # 模型运行文档和图片
├── scripts/
│   ├── runs/                  # 当前训练入口脚本
│   └── diagnostics/           # 对比和诊断脚本
└── src/torchturbo/
    ├── adapters/
    │   └── llamafactory_v0/   # LlamaFactory adapter
    ├── engine/
    │   ├── core.py            # TorchTurboEngine
    │   ├── config.py          # EngineConfig
    │   └── backends/
    │       └── fsdp2/         # FSDP2 backend
    ├── models/
    │   ├── registry.py        # 模型注册表
    │   ├── qwen3_moe/
    │   ├── glm_moe_dsa/
    │   ├── bailing_hybrid/
    │   ├── kimi_k25/
    │   └── modules/
    │       ├── attention/     # Ulysses SP helpers
    │       ├── comm/          # 通信原语
    │       └── moe/           # MoE 组件
    ├── ops/                   # 算子注册、provider 与分发
    ├── parallel/              # ParallelPlan 定义
    ├── registry/              # Adapter / Patch 注册机制
    ├── runtime/               # Device Runtime 抽象与设备选择
    └── utils/
```

## 模块整理约定

当前模型路径仍以模型目录内的 patch 和 parallel plan 为主。随着模型数量增加，
通用层、attention backend、MoE 替换逻辑应逐步收敛到一致的 factory 或
registry 形式，避免每个模型重复维护类似 patch。

## 诊断脚本

旧的对比和诊断入口放在 `scripts/diagnostics/`。它们主要用于 baseline 与
torchturbo 输出对齐检查，不作为常规训练入口。

## 许可证

Apache-2.0
