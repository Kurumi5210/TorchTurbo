# FSDP2 Distributed Muon 设计

本文介绍 TorchTurbo 的 FSDP2 Distributed Muon 的设计与实现。

模型负责声明哪些参数使用 Muon，以及如何划分独立矩阵；Engine 把这些声明和
训练配置整理成计划；Backend 再根据实际 FSDP 分片安排计算和通信。

QK clip 作为可选的更新后处理接入同一条流程：模型声明需要校正的 attention
投影，forward 收集每个 head 的最大 logit，所有 optimizer 参数更新完成后再
缩放对应权重。它有独立的 spec 和执行状态，不参与 Muon 的矩阵分配。

## 1. 设计架构与接口

### 1.1 总体流程

```text
+------------------------+    +--------------------------------+    +------------------------+
| TurboModel             |    | TorchTurboEngine               |    | FSDP2Backend           |
| get_optimizer_recipe() |    |                                |    |                        |
| returns:               |    |  +--------------------------+  |    |                        |
| OptimizerRecipe        |    |  | ModelFQNMap              |  |    |                        |
|   (MuonRecipe)         |--->|  | model FQN -> exact FQN   |  |    |                        |
+------------------------+    |  +--------------------------+  |    |                        |
| MuonOptimizerConfig    |--->| resolve recipe FQNs            |    |                        |
+------------------------+    | combine recipe and config      |    |                        |
                              | compile OptimizerPlan          |    |                        |
                              | MuonPlan                       |--->| prepare_optimizer()    |
                              |   + optional QKClipPlan        |    |                        |
                              |                                |    | create:                |
                              |                                |    | FSDPDistributedMuon    |
                              |                                |    | fallback AdamW         |
                              |                                |    | OptimizersContainer    |
                              |                                |    | optional QK clip hooks |
                              | receive final optimizer        |<---| child or container     |
+------------------------+    |                                |    +------------------------+
| Adapter / Trainer      |<---| return optimizer               |
+------------------------+    +--------------------------------+
```

这几个模块之间通过两类对象传递信息：

1. Model 与 Engine 之间传递 `OptimizerRecipe`。对于 Muon，具体类型是
   `MuonRecipe`；
2. Engine 与 Backend 之间传递 `OptimizerPlan`。对于 Muon，具体类型是
   `MuonPlan`。

| 层 | 核心对象 | 职责 | 不负责 |
| --- | --- | --- | --- |
| Model | `MuonRecipe`, `MuonSpecProvider`, `QKClipSpec` | 声明参数选择、逻辑矩阵和 QK clip 投影布局 | rank、collective、stream |
| Engine | `ModelFQNMap`, `MuonPlan`, `QKClipPlan` | 合并 recipe 与 config，绑定参数和 attention 模块 | 执行设备通信 |
| Backend | `FSDPDistributedMuon`, `QKClipController` | 读取真实分片，构造完整 optimizer 并安装 step hooks | 匹配 recipe pattern 或猜测投影布局 |
| Runtime | `DeviceRuntime` | 提供 stream、event、allocator lifetime 能力 | 参数选择 |
| Adapter | LlamaFactory / ms-swift hooks | 把训练框架 optimizer 入口路由给 Engine | 构造 recipe 或 plan |

### 1.2 Model 与 Engine：`OptimizerRecipe`

`TurboModel` 提供统一接口：

```python
def get_optimizer_recipe(self, optimizer_name: str) -> OptimizerRecipe:
    ...
```

Engine 根据 `optimizer.name` 调用这个接口。若选择 `muon`，已接入的模型返回
`MuonRecipe`：

```python
MuonRecipe(
    parameter_overrides={
        "model.layers.*.self_attn.q_proj.weight": MuonParameterSpec(...),
        "vision_tower.**": ExcludeFromMuon(),
    }
)
```

`MuonRecipe` 只描述模型层面的差异：哪些语义模型 FQN 使用特殊 matrix view，哪些
参数不能使用 Muon。可选的 `qk_clip_specs` 另外声明 attention 模块的投影布局，
与 `parameter_overrides` 分开；具体契约见第 5 节。

模型模块还可以实现 `MuonSpecProvider`，声明自己参数的 muon spec。

### 1.3 `ModelFQNMap`：连接模型声明和 prepared model

recipe 使用语义模型 FQN，optimizer 最终操作 prepared model 中的 parameter object。
`ModelFQNMap` 保存两张不可变映射：

```python
modules: Mapping[str, str]
parameters: Mapping[str, str]
```

key 是模块替换后的语义模型 FQN，value 是 prepared model 中的精确 FQN。
构建时按 wrapper 类型明确遍历结构边：

- `PeftModel` / `BaseTuner` 的容器边不进入语义模型 FQN；
- `BaseTunerLayer` 映射其 base layer 的模型参数，adapter 参数不伪装成模型参数；
- `ModulesToSaveWrapper` 只映射当前 forward 真正使用的 active copy；
- `TrainableTokensWrapper` 只保留 base embedding 的模型参数；
- `CheckpointWrapper` 的 wrapped-module 边对语义模型 FQN 透明。

这样可以排除 adapter-only 参数和 `ModulesToSaveWrapper` 的 inactive copy，也保证
一个语义模型 FQN 只解析到一个 active prepared FQN。普通 frozen 模型参数仍保留
在映射中，是否进入 optimizer 由 plan 根据 `requires_grad` 决定。

### 1.4 Engine：从 recipe 和 config 编译 `MuonPlan`

Engine 收集四类输入：

- prepared model，用于绑定真实 parameter object；
- Model 返回的 `MuonRecipe`；
- `engine_config.optimizer.muon` 中的 `MuonOptimizerConfig`；
- Backend 在 model prepare 阶段建立的 `ModelFQNMap`。

`MuonPlan` 按从弱到强的顺序合并三层参数规则：

1. Engine default：属于语义模型参数、二维、非 embedding 的 trainable parameter
   默认使用 `Owned()`；
2. Module provider：了解自身 fused storage 的模块通过 `MuonSpecProvider` 声明精确
   matrix view；
3. Model recipe：模型用语义模型 FQN pattern 覆盖 parameter spec，或显式
   `ExcludeFromMuon()`。

recipe pattern 使用与 parallel plan 相同的匹配规则：

- `*` 只跨一个 FQN segment 内的字符；
- `**` 可以跨多个 segment；
- pattern 必须匹配整个 FQN。

pattern 先匹配 `ModelFQNMap` 的 key，再转换成 prepared FQN。`parameter_overrides`
中没有命中的 pattern 当前打印 warning，并留有升级为 error 的 TODO。

编译完成的 `MuonPlan` 保存：

- 每个 Muon parameter 的精确 prepared FQN 和 `MuonParameterSpec`；
- Muon、AdamW decay、AdamW no-decay 的精确参数组；
- 已确定成员的 `MuonBucket`；
- Muon、fallback AdamW 和 overlap 的运行配置；
- 启用 QK clip 时绑定的 `QKClipPlan`，否则为 `None`。

所有 trainable parameter 必须恰好属于一个 optimizer group：

```text
trainable parameters
|-- resolved MuonParameterSpec -> Muon
`-- otherwise                  -> AdamW
    |-- bias or 1-D parameter  -> no weight decay
    `-- other                  -> configured weight decay
```

按默认规则，embedding、output embedding 和 PEFT adapter-only 参数进入 fallback
AdamW；recipe 也可以显式排除其他参数。若没有任何参数进入 Muon，plan 可以退化为
纯 AdamW，但会打印 warning。

### 1.5 Engine 与 Backend：`OptimizerPlan`

Engine 和 Backend 的接口是：

```python
def prepare_optimizer(
    self,
    plan: OptimizerPlan,
) -> torch.optim.Optimizer:
    ...
```

`OptimizerPlan` 是 Backend 的完整构造输入。对于 `MuonPlan`，FSDP2 Backend：

1. 从 plan 取出精确 Muon 和 AdamW 参数组；
2. 根据真实 DTensor mesh、placement 和 local shard 编译 Muon 执行 schedule；
3. 为 Muon 参数构造 `FSDPDistributedMuon`；
4. 为其余 trainable 参数构造 fallback AdamW；
5. 用 `OptimizersContainer` 验证所有 child optimizer 的参数归属和配置；
6. 只有一个 child 时选该 child，否则选 container 作为最终 optimizer；
7. 若有启用且未冻结的 QK clip 层，在最终 optimizer 上安装 hooks，然后返回。

`"Muon"` 和 `"AdamW"` 是 Backend 内部组织 child optimizer 的名字。
container 提供一个 PyTorch `Optimizer` facade：

- `step()` 和 `zero_grad()` 转发给所有 child optimizer；
- `param_groups` 引用 child optimizer 的真实 group dict，使一个 LR scheduler 能更新
  Muon 和 AdamW；
- 普通 `state_dict()` 使用带版本号的 named-child 格式；
- child optimizer load state 后重新绑定 facade 的 `param_groups`。

只有一个 child 时保留该 optimizer 原生的 `state_dict()` 格式。
Engine 直接保存并返回 Backend 给出的完整 optimizer，不再重组 child optimizer。
这保证 QK clip 的 post-hook 在所有 child 完成更新后执行一次，详见第 5.3 节。

## 2. 分布式方案

FSDP 把参数按行切开，存在不同 rank 上；Muon 则需要对完整矩阵做 Newton-Schulz
迭代。因此，参数存在哪里和矩阵在哪里计算，需要分开处理。

模型先声明哪些元素组成一个矩阵，分配器再决定每个矩阵交给哪个 rank 计算。
负责计算的 rank 称为 owner。数据汇聚到 owner，算出更新量后，再送回原来的参数
分片。这个过程分为三层：

| 层 | 解决的问题 | 结果 |
| --- | --- | --- |
| 矩阵定义 | 一个参数里有哪些独立矩阵，各自的数据存在哪里 | 矩阵列表和存储分片 |
| 任务分配 | bucket 中的矩阵分别交给谁算 | 每个矩阵的 owner |
| 布局与执行 | 如何把分片拼成矩阵，再把更新量放回去 | 本地视图或双向传输计划 |

`muon_layout.py` 负责矩阵定义和布局规划，`FSDPDistributedMuon` 负责分配和执行。
每个参数元素只属于一个矩阵，每个矩阵只由一个 owner 计算；动量和参数更新仍然
在原分片上各做一次。owner 的选择只影响计算位置，不改变参数长期存储的位置。

### 2.1 第一层：定义独立矩阵

模型通过 `MuonParameterSpec` 声明矩阵的含义：

- `Owned()`：整个二维参数就是一个矩阵，不能拆开计算，因此不允许再搭配 view。
- `Shard(0) + compute_view`：参数里包含多个独立矩阵，view 指定每个矩阵的边界。
  每个矩阵仍然要完整地交给一个 owner 计算。

普通 Linear 权重通常使用 `Owned()`。attention 和 expert 参数可以使用下面的 view。
其中 `H` 是 head 数，`E` 是当前参数包含的 expert 数，`D` 是 hidden size，`F` 是
FFN size：

| 声明 | 参数形状 | 独立矩阵 |
| --- | --- | --- |
| `AttentionPerHeadComputeView(H)` | `[H * R, C]` | `H` 个 `[R, C]` |
| `ExpertUpComputeView(E)` | `[E * D, F]` | `E` 个 `[D, F]` |
| `ExpertUpComputeView(E, fused_gate_up=True)` | `[E * D, 2 * F]` | `2 * E` 个 `[D, F]` |
| `ExpertDownComputeView(E)` | `[E * F, D]` | `E` 个 `[F, D]` |

这些 view 都借助通用的 `BatchedMatrixComputeView` 解析形状，得到同一种
`ResolvedBatchedMatrixView`。它是一份形状描述：有多少个 batch、每个矩阵多少行、
参数一共多少列、列方向分几组。它不保存数据，也不决定 owner。

例如参数形状为 `[B * R, G * C]`，每个 batch 沿列分成 `G` 组，就有 `B * G` 个
独立的 `[R, C]` 矩阵。它们按下面的顺序编号：

```text
matrix_id = b * G + g
matrix[matrix_id] = storage[b*R:(b+1)*R, g*C:(g+1)*C]
```

两个 gated experts 因此对应四个矩阵：每个 expert 的 gate 和 up 各算一个。
把它们合成两个 `[D, 2 * F]` 矩阵执行 Muon，会改变算法结果。

布局规划器根据这份描述列出完整矩阵，再按实际 FSDP 布局列出各 rank 的行分片。
确认它们描述的是同一个参数后，才进入任务分配。矩阵的编号和边界由模型决定，
owner 由 backend 决定，两者不混在一起。

### 2.2 第二层：分配矩阵任务

一个 bucket 的所有矩阵一起分配，`Owned()` 和 view 参数用同一套规则。已经完整
存于某个 rank 的矩阵也要参加：这样分配其他矩阵时，才能把这部分计算量算进去。
owner 只能从参数实际使用的 FSDP 通信组中选择。

**成本估算**

对 `[R, C]` 矩阵，令 `s = min(R, C)`、`l = max(R, C)`，Newton-Schulz 的计算量
估算为 `2 * ns_steps * s * s * (2 * l + s)` FLOPs，乘法和加法各计一次。
矩阵字节数按参数实际 dtype 计算，即 `R * C * element_size`；通信传的是这个 dtype，
不能因为计算时转成 BF16，就把流量也按 BF16 算。

一个同步 bucket 先通信，再计算，再反向通信。计算要等最忙的 rank，通信也要等
发送或接收量最大的 rank。因此，用下面的成本比较分配方案：

```text
C_peak = max(compute_flops[rank])
T_peak = max(max(send_bytes[rank], receive_bytes[rank]))
weighted_cost = C_peak + 2 * W * T_peak
```

`W` 是有效计算吞吐与有效通信带宽的比值，单位为 FLOP/byte。默认取 2000，也就是
把搬运一个字节的代价按 2000 FLOPs 估算。它让计算和通信能放在一起比较，避免
为了省一点计算就搬走大量数据。这个默认值不按卡的型号切换，也不自动调整。

`T_peak` 只统计正向通信。反向通信的发送和接收互换，峰值相同，所以乘以 2。
若矩阵共有 `M` 字节，owner 已经持有 `H` 字节，需要正向搬运的就是 `M - H` 字节；
本地复制不算远程通信。

比较两个完整方案时，依次看：

1. 当前 bucket 的 `weighted_cost`；
2. 当前 bucket 的双向远程总字节数；
3. 加上之前 bucket 后，同一通信组的累计 `weighted_cost`；
4. owner 编号表，用来打破最后的平局，保证各 rank 得到相同结果。

先看当前 bucket，是因为它直接影响这一批任务的完成时间。成本一样时先减少总
流量，再考虑历史负载，避免仅为了跨 bucket 均衡而增加无收益的搬运。

**分配步骤**

任务按计算量、矩阵字节数从大到小排列；相同任务按参数 FQN 和矩阵编号排序。
每个通信组都计算两个方案：

- 贪心方案：逐个放入任务，每次选择让已分配任务成本最小的 rank。成本相同，
  先比较当前任务的流量，再比较累计成本和 rank 编号。
- 本地优先方案：先把每个任务放到持有它最多数据的 rank，平局时沿用上面的规则。
  全部放好后，再逐个检查能否换 owner。检查时先扣掉该任务原来的计算量和各 rank
  的收发量，再尝试所有 owner，包括原 owner；只有完整方案变好才移动。

两个方案读取同一份历史负载。最后选成本更好的方案，只把它的负载累计一次，
供下一个 bucket 使用。因此，最终结果不会比相同输入下的贪心方案更差。

**为什么先放完，再调整**

假设 16 个 rank 上有 17 个等量矩阵：一个跨 rank 分片的 Owned 矩阵，以及每个
rank 本地各一个 head。Owned 先分到 rank 0，逐任务贪心可能为了躲开 rank 0 的
第二份工作，把后面的 head 依次往后挪。但不管怎么放，总有一个 rank 要算两份：

| 分配方式 | 最大单 rank 计算量 | 被搬走的 head |
| --- | --- | --- |
| 依次向后挪动 | 两个矩阵 | 15 个 |
| head 保持本地 | 两个矩阵 | 0 个 |

两种方案都要汇聚 Owned 的分片。head 保持本地，计算峰值没变，却少了 15 个 head
的搬运。先放好所有任务，才能在调整时看到这个结果。如果搬运确实能降低完整
成本，分配器仍然会移动矩阵，并不会把它锁在本地。

只检查一轮是为了控制构造开销。有些改进需要同时移动多个任务，一次只动一个
可能找不到；保留贪心方案能补上其中一部分情况，但不能保证最优。任务数为 `J`、
rank 数为 `P` 时，排序后的分配时间和存储数据位置所需的空间都是 `O(J * P)`。
分配只在初始化或加载 checkpoint 后执行，训练 step 复用结果。

这个成本模型没有计算通信启动、打包和网络争用的耗时，不能当作实测时间。
同样，累计负载用于打破平局，不代表所有 bucket 的总耗时。

### 2.3 第三层：把数据送到 owner，再送回来

FSDP 存储分片沿用 PyTorch 的 `Shard.local_shard_size_and_offset()` 规则。这里有
三种编号：global rank 标识进程，group rank 表示它在通信组中的顺序，mesh index
决定它持有哪段 FSDP 分片。mesh 中的 rank 顺序不一定递增，所以必须保存编号
之间的映射，不能直接拿 owner 的 group rank 推算分片位置。

owner 确定后，同一参数分给一个 rank 的矩阵按编号升序排列。编号可以不连续，
例如 rank 0 可以负责矩阵 0、3、5。view 参数的计算 tensor 形状是
`[local_num_matrices, R, C]`；`Owned()` 的 owner 则使用单个 `[R, C]` 矩阵。

规划器比较存储位置和计算位置，分三种情况处理：

- 数据在同一个 rank，元素偏移和行间距也相同：直接用本地视图计算。
- 数据在同一个 rank，但排列不同：在本地重新打包。
- 数据分散在其他 rank：先打包传给 owner，再拼成完整矩阵。

例如 8 个 head、每个 128 行，在 16 个 rank 上分片，每个 rank 只有 64 行。
一个 head 横跨两个分片，需要汇聚后才能计算。只有 8 个矩阵，最多就有 8 个 rank
负责计算；其他 rank 仍负责发送自己的分片，并接收算好的更新量。

本地数据也可能需要重排。fused gate/up 的一行有 `2 * F` 列，单个 gate 矩阵的一行
只有 `F` 列。即使两者在同一 rank，计算前仍可能需要把交错存储的数据整理出来。
所以“没有远程流量”和“可以直接 view”要分开判断。

每条 route 描述一个矩阵与一个存储分片相交的矩形区域，记录两端 rank、起始偏移、
行数、列数和行间距。这样一条 route 就能表示整个 gate 或 up 分片，不必为每一行
单独生成一条。打包时跳过同一行中属于其他矩阵的列，只传需要的元素。

所有 routes 合起来必须恰好覆盖矩阵，没有重复，也没有遗漏。反向传输交换每条
route 的起点和终点，沿用同一个矩形区域，就能把更新量放回原位置。FSDP 为通信
补齐的 padding 不属于参数，不参与这些传输，也不参与动量或参数更新。

### 2.4 动量与参数更新

每个参数每一步按下面的顺序更新：

1. 在原 FSDP 分片上更新动量，完成 Nesterov 准备。
2. 按计划把准备好的数据送到 owner；能直接本地计算的参数省去传输。
3. owner 对完整矩阵执行 Muon，得到更新量。
4. 把更新量送回原分片，应用 weight decay 和参数更新。

动量准备是逐元素操作，因此先在分片上做、再汇聚，与先汇聚再做等价。动量可以
一直保存在原 FSDP 布局中，不必让 owner 常驻一份完整副本。一个参数拆成几个
矩阵任务，都只更新一次它的本地动量和参数。

送到 owner 的数据已经排成独立矩阵，计算前不再拆一次 view；算完也由反向 route
恢复顺序。按形状调整学习率时，使用单个矩阵的 `(R, C)`，不使用 fused 参数的
宽度或当前 rank 分到的矩阵总数。

### 2.5 Bucket 决定哪些参数一起通信

`MuonBucket` 保存一组有序的参数 FQN，成员在 plan 中确定：

| 策略 | 行为 |
| --- | --- |
| `parameter` | 每个 Muon 参数一个 bucket |
| `single` | 所有 Muon 参数放入一个 bucket |
| `size` | 按 `bucket_cap_mb` 顺序打包 |

bucket 的数量和容量由这些规则决定，与 mesh 有几个维度无关。任务分配只决定
bucket 内的矩阵交给谁，不改变 bucket 成员。如果用装箱来理解，各个 rank 是装
计算任务的箱子；bucket 则决定这一次拿哪些任务来分。

同一个 bucket 中需要打包的参数共用收发缓冲区，每个方向最多执行一次
`all_to_all_single`。这些参数必须使用相同的通信组、dtype 和 device。可直接本地
计算的参数可以留在同一个 bucket 中，不进入这次通信。只需本地重排的参数也
经过打包和还原，只省去远程传输。

收发双方按相同的参数和矩阵位置顺序打包。同一行中的不同列组也要区分，才能
保证接收方知道每段数据属于哪个矩阵，反向时又该写回哪里。

bucket 越小，通信次数越多，但临时缓冲区更小，也能更早开始计算。bucket 越大，
通信次数更少，但要准备更多数据才能启动下一阶段。

## 3. 执行计划与运行时

### 3.1 初始化时准备计划

Backend 先把参数 FQN 绑定到实际 DTensor，读取分片布局。随后按第 2 节的三层
流程生成矩阵、选择 owner、安排传输，最后为各 bucket 准备缓冲区和执行顺序。
这些工作只在构造 optimizer 或加载 state 后进行，step 不重新规划。

同一个通信组的各 rank 必须对计划达成一致。对于需要布局转换的 bucket，Backend
对矩阵定义、owner、routes、bucket 顺序和影响计划的 optimizer 选项计算 hash，
再在组内比较。纯本地 bucket 不执行这项 collective 检查。hash 描述的是全组计划，
不能只比较各 rank 自己分到的矩阵或缓冲区大小，因为它们本来就不同。

这个检查要求各 rank 已按相同顺序进入相同的 collective。如果各 rank 连参加哪个
通信组都不一致，hash 检查本身无法解决这个问题。

一个 DP bucket 还可能包含只在本地计算的 expert 参数。不同 EP 位置的参数可能
属于不同 EDP 子组，但不参加这次 DP 传输，因此它们的物理子组编号不放进外层
DP hash。真正参加布局转换的参数则必须记录实际通信组。

### 3.2 同步执行

同步路径按 bucket 顺序，依次完成动量准备、正向传输、矩阵计算、反向传输和
参数更新，再处理下一个 bucket。

一个 rank 有没有参数分片，和它有没有分到计算任务，是两回事：

| 本地参数分片 | 分到的矩阵 | 行为 |
| --- | --- | --- |
| 非空 | 非空 | 准备分片、计算矩阵、收回更新量并更新参数 |
| 非空 | 空 | 发送分片，收回更新量，不做矩阵计算 |
| 空 | 非空 | 接收矩阵并计算，不更新本地空参数 |
| 空 | 空 | 按全组计划参加必要的通信，不做矩阵计算 |

因此，是否进入 collective 由全组计划决定，不能因为本 rank 没有数据就跳过。

### 3.3 让通信和计算重叠

异步路径使用一条通信 stream 和一条计算 stream。开启 `overlap_comm_compute` 后，
用两组缓冲区轮流处理相邻 bucket：计算 bucket k 时，提前汇聚 bucket k+1；
bucket k 算完后，再把它的更新量送回。两个相邻通信 bucket 的顺序如下：

```text
transfer: gather k -> gather k+1 -> scatter k -> scatter k+1
compute:              compute k -> compute k+1
```

gather 和 scatter 共用通信 stream，因此通信之间仍然串行。重叠发生在通信和
另一条 stream 的矩阵计算之间。

每组缓冲区用三个 event 控制读写顺序：

- `input_ready`：汇聚完成，可以开始计算。
- `compute_done`：计算完成，可以送回更新量。
- `finished`：送回和参数更新都完成，缓冲区可以交给后面的 bucket。

`record_stream` 保证异步操作结束前内存不会被回收，event 则保证不会提前读取或
覆盖数据。只有本地计算的 bucket 不占这些缓冲区，也会阻断向后预取。关闭重叠
时只用一组缓冲区，按顺序执行。

### 3.4 缓冲区复用

workspace 按 `(device, dtype)` 管理四类缓冲区：

| 缓冲区 | 用途 |
| --- | --- |
| `storage_exchange` | 正向发送本地分片，反向接收本地更新量 |
| `compute_exchange` | 正向接收 owner 所需的数据，反向发送计算结果 |
| `compute_scratch` | 当前参数的完整矩阵计算空间 |
| `storage_scratch` | 当前参数的动量准备或本地更新量 |

通信缓冲区按本 rank 在整个 bucket 中的收发量分配；计算和存储临时空间分别
按单个参数在本 rank 上的计算块、存储分片取最大值，逐参数复用。空间不够时
扩容，后续 step 和加载 state 后可以继续复用。

正向接收区同时作为反向发送区。处理完一个参数，就把它的结果写回同一段空间。
这要求两方向对应参数的区间和顺序一致，仅仅总容量相等还不够，否则可能覆盖
另一个参数尚未读取的数据。缓冲区切片自身的起始偏移也必须计入二维视图。

### 3.5 计划日志

初始化或加载 state 后，每个 FSDP 通信组只由 group rank 0 打印一条 INFO：组标识、
rank 数、bucket 数、矩阵数、正向远程字节数、选中本地优先调整方案的 bucket 数，
以及任务分配耗时。用这些汇总了解计划规模和搬运量，不逐参数或逐 bucket 打印。

远程字节数按最终分配和参数 dtype 计算，一份数据只计一次，不包含本地复制和
反向通信。它是计划流量，不是实测网络流量。分配耗时只统计 CPU 上排序、构造
候选和调整的时间，不含通信校验及缓冲区分配。训练 step 复用计划时不重复打印，
这些统计也不进入 checkpoint。

## 4. 模型接入

Kimi K2.5 和 Bailing hybrid 已提供 Muon recipe；两者的 `flash_attention_2` MLA
路径也声明了可选的 QK clip spec。Bailing 的 linear attention 使用自己的 Muon
参数规则，不参与 MLA QK clip。

### 4.1 Attention per-head

模型 recipe 对 query projection 和 `kv_b_proj` 声明：

```python
MuonParameterSpec(
    compute_placement=Shard(0),
    compute_view=AttentionPerHeadComputeView(num_attention_heads),
)
```

这些投影权重按 head 划分，每个 head 是一个独立矩阵任务。分配好 owner 后，
数据位置和计算布局一致的直接本地计算，其余的按传输计划汇聚成完整 head。

### 4.2 GroupedMLP per-expert

普通 `GroupedMLP` 把多个 expert 的权重合并存放在 `weight1` 和 `weight2` 中。
模块通过 `MuonSpecProvider` 声明这两个参数该如何拆成独立矩阵：

```python
MuonParameterSpec(
    compute_placement=Shard(0),
    compute_view=ExpertUpComputeView(
        num_experts,
        fused_gate_up=config.gated_linear_unit,
    ),
)
MuonParameterSpec(
    compute_placement=Shard(0),
    compute_view=ExpertDownComputeView(num_experts),
)
```

`weight1` 使用 up view。`fused_gate_up` 默认是 `False`，GroupedMLP 将它设为
`config.gated_linear_unit`，因为这项配置同时决定权重是否沿列拼接 gate 和 up。
不拼接时，每个 expert 是一个 `[D, F]` 矩阵；拼接时是两个。`weight2` 使用 down
view，每个 expert 是一个 `[F, D]` 矩阵。

`num_experts` 从实际参数形状推导：up 用 `weight1.shape[0] // D`，down 用
`weight2.shape[0] // F`。这里取的是 EP 划分后，这个参数实际包含的 expert 数。
FSDP 再把参数分片时，仍使用 DTensor 的全局形状，不能拿本地分片的行数来算。

TransformerEngine 使用 `GroupedLinear`，没有这两个合并参数，所以 provider 返回
空映射，由 Engine 按默认规则选择：符合条件的二维模型参数使用 `Owned()`，其余
参数使用 AdamW。

对于合并存储的 expert，EP 先决定每个 EP 位置负责哪些 expert，再由对应的一维
EDP 子组对权重做 FSDP `Shard(0)` 分片。Muon 只在这个 EDP 子组里分配和传输，
不会把任务分到其他 EP 位置。

例如 `world_size=16, ep_size=4`：

```text
global mesh shape: [edp=4, ep=4]

EP groups:  [0,1,2,3], [4,5,6,7], ...
EDP groups: [0,4,8,12], [1,5,9,13], ...
```

rank 0、4、8、12 对应相同的 EP 位置，负责同一组 expert，其权重在这四个 rank
之间分片。每个完整 expert 矩阵的 owner 从这四个 rank 中选择；gate 和 up 也是
各自独立选择 owner。

EP 划分后的 expert 数可以小于 EDP mesh size，也不要求整除它。例如两个 expert
的 down 参数在八个 rank 上分片，只产生两个矩阵任务，最多两个 rank 负责计算，
其余 rank 发送自己的分片并接收更新量。gated up 则有四个任务，可以分给不同
owner。任务少时，部分 rank 没有矩阵可算，并不影响分片的收发和参数更新。

这是 EDP 上的分片规则。EP 划分本身仍要求全局 expert 数能被 EP size 整除，
每个 EP rank 至少有一个 expert。

### 4.3 显式排除

Kimi recipe 将 `vision_tower.**` 和非 identity `mm_projector.**` 显式排除，交给
AdamW。PEFT adapter 参数不在 `ModelFQNMap.parameters` 的 value 集合中，默认也不会
被二维参数规则误选进 Muon。

## 5. QK clip 设计

### 5.1 原理：用实际 logits 校正 Q/K 权重

QK clip 用于控制 attention logits 的增长。它在 forward 中观测每个 head 的
最大 logit，再据此缩放该 head 的 Q/K 投影权重。这是 optimizer 更新后的权重
校正，与梯度裁剪独立；forward 仍按原来的 attention 公式计算，不直接截断 score。

统计值已经乘上模型实际使用的 softmax scale，只包含 mask 允许的有效位置。
取的是有符号最大值，不是绝对值或 softmax 概率。设 head `h` 在本次累积窗口
中的最大值为 `M_h`，目标阈值为 `tau`，缩放系数为：

```text
s_h = tau / max(tau, M_h)
```

未超阈值时系数为 1，超出时按比例缩小。MLA 的 logit 由 NoPE 和 RoPE 两部分
点积组成，两部分需要乘上相同的系数。`alpha` 决定 NoPE 的缩放由 Q/K 两侧如何
分担；RoPE key 是多个 head 共享的，因此 RoPE 部分只缩放 query：

| MLA 投影部分 | 权重乘数 |
| --- | --- |
| Q-NoPE | `s_h ** alpha` |
| K-NoPE | `s_h ** (1 - alpha)` |
| Q-RoPE | `s_h` |
| V 和共享 K-RoPE | 不变 |

NoPE 点积的总乘数为 `s_h ** alpha * s_h ** (1 - alpha) = s_h`，RoPE 点积的
乘数也是 `s_h`。例如最大值为 400、阈值为 100、`alpha=0.5`，Q-NoPE 和 K-NoPE
各乘 0.5，Q-RoPE 乘 0.25；固定输入时，两部分 logits 都乘 0.25。

需要区分公式和训练时序：统计来自更新前的 forward，缩放作用于更新后的权重。
代码不会再次 forward 检查新 logits，上游层的更新也会改变下次输入。因此阈值
是依据观测进行校正的目标，不是下一次 forward 的严格上界。校正只修改参数，
不缩放 Muon momentum 或 AdamW 的一阶、二阶状态。

### 5.2 核心对象：声明、绑定、统计、执行

| 对象 | 职责 |
| --- | --- |
| `QKClipSpec` | 公共基类，声明 head 数和参与校正的投影 |
| `MLAQKClipSpec` | 描述 MLA 的 query/KV 投影及 NoPE、RoPE、V 布局 |
| `QKClipPlan` | 把模型声明绑定到实际 attention 模块和参数，检查是否满足校正条件 |
| `QKClipState` | 保存一层在当前累积窗口内的 head maxima |
| `QKClipController` | 合并全局统计，在所有参数更新后校正权重并清空统计 |

模型通过 `MuonRecipe.qk_clip_specs` 提供声明；只有训练配置启用 QK clip，
`MuonPlan` 才建立相应的 `QKClipPlan`。plan 依赖公共基类；当前 FSDP2 backend
实现 MLA 的缩放规则。扩展其他 attention 结构时，需要对应的 spec、统计接入
和校正规则。

`MuonParameterSpec` 描述独立矩阵如何执行 Muon，`QKClipSpec` 描述 attention
权重如何校正，两者分开。Q/K 可以分别使用 Muon 或 fallback AdamW，controller
在同一个最终 optimizer 上统一校正这些投影。

### 5.3 核心流程：先收集，再完成更新，最后校正

```text
model recipe + training config
              |
              v
bind QKClipPlan and install state/controller
              |
              v
for each microbatch:
    forward -> accumulate per-head maxima -> backward
              |
              v
gradient clipping
              |
              v
optimizer.step()
    |-- finish Muon and fallback AdamW updates
    `-- reduce global maxima -> scale Q/K weights -> reset statistics
```

公共 attention 接口默认调用 HF；需要 QK 统计时，请求算子同时返回 attention
输出和每个 head 的最大 logit。算子只负责计算，模型侧把 detached 统计写入
`QKClipState`，不让统计进入反向图。

gradient accumulation 的多个 microbatch 按 head 取最大值，而不是平均；相同
forward 的 checkpoint 重算不会重复放大统计。当前模型的 eval 路径不收集统计。

controller 的 hooks 安装在最终 optimizer 上。如果存在多个 child optimizer，
必须等它们全部更新后才校正一次，否则 Q/K 分属不同 child 时会使用不完整的
更新结果。step 前还要确认所有启用层都产生过统计，避免某层静默漏掉校正。

一次实际更新完成后清空统计，开始下一个累积窗口；仅执行 `zero_grad()` 不结束
这个窗口。若调用方跳过整个 optimizer step，权重和已有统计都保留。

### 5.4 分布式思路：合并小统计，原地修改权重分片

SP 交换后，每个 rank 在完整 sequence 上计算一组 heads，只得到自己负责 heads
的最大值。因此统计先按全局 head 顺序对齐，再在覆盖全部训练 rank 的通信组上
取最大值，同时合并 SP head 分组和不同数据 worker 的观测。
所有启用 QK clip 的层共用一块统计 buffer，每个实际 step 只做一次全局 MAX 归约。

权重校正在原 FSDP 分片上完成：每个本地行知道自己属于哪个 head、使用哪个
缩放系数，即使一个 head 被 FSDP 切到多个 rank，两边也会使用相同的系数。
不需要为了 QK clip 汇聚完整权重，也不需要改变 Muon 的 owner 分配。

这里的分工是：attention 计算本地统计，controller 统一全局观测，FSDP 各分片
执行相同规则的局部缩放。通信量随 head 数增长，不随完整投影矩阵的大小增长。

## 6. 训练配置

### 6.1 Engine 配置

最小配置为：

```yaml
engine_config:
  optimizer:
    name: muon
    muon:
      lr: 0.02
      weight_decay: 0.01
```

完整配置见 [`scripts/runs/config_example.yaml`](../../../scripts/runs/config_example.yaml)。
`lr` 和 `weight_decay` 必须显式配置；训练框架自己的 optimizer 参数不会传入
engine-managed Muon。Engine 会打印 warning，明确本次运行采用
`optimizer.muon` 中的值。

CLI 可以使用：

```text
--tt_optimizer muon
```

`optimizer.name` 当前只支持 `muon`。

QK clip 默认关闭。需要时将 `optimizer.muon.qk_clip` 设为
`{threshold: 100.0, alpha: 0.5}`；设为 `null` 则关闭。阈值和缩放分配比例由训练
配置决定，投影路径和 head 布局由模型 spec 决定。

### 6.2 Adapter 行为

LlamaFactory 和 ms-swift 的 `create_optimizer` 都遵循同一规则：

- 配置了 `optimizer.name`：调用 `engine.prepare_optimizer()`，拒绝同时传入外部
  optimizer class 或 kwargs；
- 未配置 `optimizer.name`：保留训练框架选择 optimizer class 的旧路径，并由
  `OptimizerClassPlan` 交给 Backend 构造。

这使 engine-managed optimizer 是显式 opt-in，不改变已有 AdamW 训练行为。

## 7. Checkpoint 与 scheduler

Muon 的动量与参数、梯度使用相同的 DTensor 分片。AdamW 则使用 Backend 按普通
参数和 expert 参数分好的参数组。

DCP 按实际存在的 optimizer 保存状态：只有一个 child 时直接处理该 optimizer；
使用 `OptimizersContainer` 时分别处理各 child。两种情况都会将参数对象映射到
FQN，并保存对应的分片状态。恢复时：

1. DCP 把状态写回实际使用的 optimizer；
2. 若有 Muon，根据恢复的配置重新分配矩阵，生成传输计划并准备缓冲区；
3. 若使用 container，重新引用各 child 恢复后的参数组；
4. 恢复 scheduler 状态，继续通过最终 optimizer 的参数组更新学习率。

scheduler 始终绑定最终 optimizer；是否使用 container 不改变训练侧的调用方式。

owner 表、routes 和缓冲区不进入 checkpoint。动量按原参数分片保存，所以恢复后
即使换了 owner，也不需要改变动量的存储方式。

EP expert optimizer state 使用现有 DCP expert FQN stamping 机制，避免不同 EP
coordinate 上相同 prepared FQN 的 shard key 冲突。

QK clip 校正后的权重随模型参数保存，统计和 controller 属于临时运行状态，
不进入 checkpoint。恢复训练时按 recipe 和配置重新建立；在完整 optimizer step
边界恢复后，从新的 forward 开始收集统计，不需要恢复上一窗口的最大值。

## 8. 版本与当前限制

### 8.1 PyTorch 版本

`FSDPDistributedMuon` 要求 PyTorch 2.10 或更高版本。PyTorch 2.9 的
`reset_sharded_param()` 在重新 padding uneven shard 时可能替换 local storage；本
optimizer 捕获 local backing storage identity，因此要求 2.10 中 idempotent 的 reset
行为。空 logical shard 的 identity 使用 backing storage pointer 和 storage offset。

### 8.2 Muon 当前限制

- 不支持 FSDP CPU offload；启用时 Backend 明确报错；
- 只支持一维 `dp`/`edp` mesh、精确 `Shard(0)` placement 和连续二维 storage；
- 全局参数 shape 必须非空，但允许空 local storage 和空 compute partition；
- 不支持一个 Muon parameter 在 step 中没有 gradient；
- direct local compute 要求分配后的完整矩阵布局与原 local storage 元素映射一致；
- 同一 bucket 中需要重分布的参数不能混合不同 process group、dtype 或 device；
- 只在完整矩阵上执行 Muon，不自动改用分片近似算法或 replicated compute；
- owner 在计划编译时确定，step 中不动态迁移，也不为 optimizer 改变 FSDP mesh；
- `parameter_overrides` 中的 pattern 未命中目前是 warning，计划后续升级为配置错误；
- `ModelFQNMap` 不保存模块替换前后的参数来源关系；replacement 产生的 fused 参数
  需要通过 `MuonSpecProvider` 声明；
- FSDP parallel plan 的 `placement` 当前只通过 `mesh_dim` 选择 submesh，
  `fully_shard` 实际依赖 PyTorch 默认 `Shard(0)`；Muon 会对最终 DTensor placement
  再做严格验证。
