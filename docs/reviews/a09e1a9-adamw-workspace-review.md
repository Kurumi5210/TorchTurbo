# Review：a09e1a9 `feat(optimizer): tune chunked AdamW swap workspaces`

- 被评审 commit：`a09e1a9ba68139a1d31568a9b8a68e156f5ace88`（父 commit `cccd6de`）
- 评审日期：2026-11-26
- 涉及范围：`optimizers/adamw.py`（重写）、新增 `optimizers/adamw_workspace.py`、`config.py`、`muon.py`、README 及 3 个新测试文件
- 第二轮：复核 Codex 根据本评审改的版本 `54eeedb`（分支 `dev-chunk-optimizer`），见文末
  [第二轮复核](#第二轮复核)；修复方案见 [建议修复方案](#建议修复方案)，
  补丁文件 [`a09e1a9-adamw-workspace-fixes.patch`](a09e1a9-adamw-workspace-fixes.patch)

下文的行号都以 a09e1a9 为准；没有写目录的文件都在
`src/torchturbo/engine/backends/fsdp2/optimizers/` 下。

## 结论

主体实现可靠，但在 checkpoint resume 场景下有一个会撑爆 host 内存的问题（第 1 条），
建议合入前修掉。第 2 条和第 1 条可以用同一处改动修复。第 3 条是设计层面的取舍，
需要和作者讨论。

| # | 严重程度 | 问题 | 位置 | `54eeedb` 状态 | 修复方案 |
|---|---|---|---|---|---|
| 1 | 高 | resume 时 pinned sink 的大小约等于 95% 空闲显存，和模型状态大小无关，校准后也不释放 | `adamw.py:389-392` | 部分修复：sink 大小已修，pinned 缓存仍不释放 | A、B |
| 2 | 中 | 扫描范围不按 moment 总量截断，产生大量冗余候选 | `adamw.py:265-285` | 未修 | A |
| 3 | 中 | 默认行为改变，方案不再适用时没有自动回退 | `adamw.py:245-253`、`config.py:71` | 未修 | C、D，默认值需作者决定 |
| 4 | 低 | 计时噪声、起始 slot 大小、README 粒度 | 见下文 | 未修 | E，其余需作者决定 |

## 已核实没有问题的部分

- **数学等价**：AdamW 是逐元素更新，把一个参数切成多个 chunk 后，结果和整块计算一致。
  `torch._fused_adamw_` 这个底层 op 本身不给 step 加一（旧代码是在 `_run_adamw` 里手动
  `step.add_(1)`），新代码把加一统一挪到 `_advance_steps`（`adamw.py:559-562`），所以一个参数
  跨多个 chunk 时 step 也只加一次。fused kernel 在不传 `grad_scale` 时不会改写 grad，
  所以"校准不改变梯度"的说法成立。
- **分布式协调**：`_autotune_workspace`、`_benchmark_workspace`、`_acquire_workspace`、
  `_step_on_device` 的所有失败路径都先 all_reduce 再一起抛错。各 rank 调用 collective 的
  次数一致：候选列表由全局归约后的 `shard`、`budget`、`overlap` 算出，各 rank 完全相同；
  空 rank 用同样数量的 `(1, 0)` 方案占位。我没找到会死锁的分支。
- **双槽流水线**（`adamw_workspace.py:376-413`）：h2d 等 `finished`，compute 等 `input_ready`，
  d2h 等 `compute_done`，同一个 slot 复用前一定会等上一轮 D2H 完成。最后对 d2h 做 host 同步
  就隐含了全部 compute 和 h2d 都已完成，所以 `_step_on_device` 里不同步就直接
  `_drop_workspace()` 是安全的（另有 `record_stream` 兜底）。异常时会 drain 三条流，
  不会去等一个还没 record 的事件。
- **零态恢复**：`reset_workspace_moments` 用的 `iter_chunks` 覆盖了所有 bound 参数的全部元素，
  每次 trial 写回的真实 CPU moment 都会被清回零。

## 测试情况

在本机 CPU 版 torch 2.10 的 venv 上跑了：

```
tests/engine/backends/fsdp2/test_adamw_pipeline.py
tests/engine/backends/fsdp2/test_adamw_workspace.py
tests/engine/backends/fsdp2/test_adamw_workspace_distributed.py
tests/engine/backends/fsdp2/test_cpu_offload_optimizer.py
tests/engine/test_config.py
```

结果是 162 个通过，5 个失败。5 个失败都在 `test_cpu_offload_optimizer.py`，报错为
`pin_memory=True requires a CUDA or other accelerator backend`，属于环境问题：父 commit
`cccd6de` 在同一文件失败 8 个，原因相同，所以不是回归。

**GPU/NPU 上的真实路径本机跑不了，没有验证。** 下面关于 pinned 内存和显存的数字是按代码
逻辑推算的，没有在真机上测过。

---

## 1.【高】resume 时 pinned sink 按最大候选分配，且校准后一直占着

### 现象

`adamw.py:386-393`：

```python
if zero_moments:
    stats["output_mode"] = "real_zero_moments"
else:
    sink_bytes = max(plan.capacity_bytes for plan in candidates)
    cpu_sink = torch.empty(
        sink_bytes, dtype=torch.uint8, pin_memory=True
    )
```

moments 非零时（从 checkpoint 恢复就是这种情况），sink 的大小是所有候选里最大的
`capacity_bytes`，也就是单槽的上限 ≈ `0.95 × 各 rank 最小空闲显存`。这个值只和显存有关，
和模型状态大小无关。

用 `_workspace_candidates` 推算两个典型配置（复现脚本见附录）：

| 场景 | 候选数 / trial 数 | 最大 capacity（= sink 大小） |
|---|---|---|
| LoRA（最大分片 16 KiB，moments 共约 64 MiB，60 GiB 空闲） | 45 / 180 | 57 GiB |
| 7B 全参 8 卡（最大分片约 155 MiB，40 GiB 空闲） | 17 / 68 | 38 GiB |

### 为什么实际情况更糟

1. **按 2 的幂取整**：PyTorch 的 `CachingHostAllocator` 会把 pinned 分配向上取整到 2 的幂
   （torch 2.10 的 `ATen/core/CachingHostAllocator.h:251`，
   `roundSize = c10::llvm::PowerOf2Ceil(size)`）。所以上面两种情况实际都是
   **64 GiB/rank**，一台 8 卡机器就是 512 GiB pinned 内存。
2. **释放后不还给系统**：`cpu_sink = None` 之后，这块内存回到 host 缓存的空闲链表里，
   只有 `torch._C._host_emptyCache()` 才会真正释放。`runtime.empty_cache()` 只调用
   `torch.cuda.empty_cache()` / `npu.empty_cache()`，只清显存；`src/` 里也没有任何地方
   清 host 缓存。空闲链表按大小档位复用，之后不会再有 64 GiB 这一档的 pinned 请求，
   所以这块内存实际上会一直占到训练结束。
3. **每次 resume 都会触发**：调好的方案不会存进 checkpoint，新建的优化器在第一步就会重新校准
   （测试 `test_first_step_calibrates_once_and_checkpoint_resume_updates_once`，
   `tests/engine/backends/fsdp2/test_adamw_workspace.py:982` 覆盖了这个流程）。
   而 resume 之后 moments 一定非零，必然走 sink 分支。

### 影响

host 内存紧张的机器上，resume 会在校准的 snapshot 阶段直接失败（这个失败会经 all_reduce
传到所有 rank，整个作业一起退出）。即使分配成功，也会长期多占几十 GB 的 pinned 内存，
挤占 CPU moments 和数据加载所需的内存。

### 建议

- 首选：按第 2 条截断扫描上限，sink 自然被限制在约 2 倍 moments 大小以内。
- 补充：校准结束后显式释放 host 缓存。但这会把其他已缓存的 pinned 块一起释放，
  需要评估对 CPU moments 分配之外其他 pinned 用户的影响。

## 2.【中】扫描范围不按 moment 总量截断

### 现象

`adamw.py:265-285` 的 `_workspace_candidates` 从 M 开始，每次翻倍，一直加到
`budget // slots`。README 第 189 行也明确写了 "moment size does not cap the sweep"。

一旦单个 slot 能装下这个 rank 的全部 moments，`iter_chunks` 就只会产生一个 chunk，
再大的 slot 在传输量、kernel 数和计算量上都完全相同，只是多占显存。按第 1 条的两个场景：

- LoRA：45 个候选中 21 个属于这种情况
- 7B 全参：17 个候选中 7 个属于这种情况

这些 trial 每次仍要跑一遍完整的 step，再加上参数恢复（从 pageable 快照做 H2D）和
moment 清零，还会把显存一路分配到 95%。

### 建议

在 `_autotune_workspace` 里多做一次 MAX 归约，求出"装下本 rank 全部 moments 所需的 slot 大小"，
用它截断每种槽数的 endpoint。示意（未在真机上测试）：

```python
# 和 iter_chunks 的对齐规则保持一致：每段 16 字节对齐，两半各放一份 moment
def _single_chunk_slot_bytes(bound):
    half = 0
    for p in bound:
        if p.numel:
            elem = max(p.exp_avg.element_size(), p.exp_avg_sq.element_size())
            half = DeviceWorkspace.align(half) + p.numel * elem
    half = DeviceWorkspace.align(half)
    return 2 * half  # 一定是 32 的倍数

# _autotune_workspace 中，与 shard 一起归约
full_slot = torch.tensor(_single_chunk_slot_bytes(bound), dtype=torch.int64, device=runtime.device)
self._reduce_workspace_value(full_slot, dist.ReduceOp.MAX)

# _workspace_candidates 中
endpoint = min(budget // slots, full_slot) // alignment * alignment
```

`full_slot ≥ 2M ≥ start`，所以截断后不会出现 `endpoint < start` 的问题。修完之后，sink 上限变成
`2 × full_slot`（双槽），和模型状态大小挂钩，第 1 条随之解决。

另外，当 `slot ≥ full_slot` 时只有一个 chunk，双槽不会产生任何 overlap，这时的双槽候选也可以跳过。

## 3.【中】默认行为改变，方案不再适用时没有自动回退

### 默认值变化

`config.py:71` 把 `optimizer_memory_fraction` 的默认值从 `0.8` 改成 `None`，
`scripts/runs/config_example.yaml` 里的 `0.8` 也被注释掉了。所有没显式写这个字段的现有配置
都会切换到自动校准，每次启动和每次 resume 的第一步都要额外跑几十到上百次 trial。
这一点应该写进 changelog 或者 PR 描述，让用户知道首步耗时会明显变长。

### 不再适用时直接报错

`adamw.py:249-253`：真实 step 时如果调好的方案超过当前预算，会报
`selected AdamW workspace ... no longer fits ...; recalibrate on all ranks`。
但 `src/` 里**没有任何地方调用 `autotune_workspace()`**，引擎不会自动重新校准，
实际效果就是训练直接退出。旧逻辑每一步都按 fraction 重新计算 workspace 大小，是自适应的。

触发概率不高：真实 step 的预算是 `0.95 × (free + cached)`，只要 step 时已分配的显存没有超过
校准时已占用（reserved）的显存，它就不小于校准时的预算 `0.95 × free`；而且选中的方案通常是
"3% 以内最省显存"的那个，不会顶到上限。但非 PyTorch 显存增长（例如后续新建通信组、
RL 场景里推理引擎残留的显存）仍可能触发，而且一旦触发就没有任何恢复手段。

这个错误已经在 `_reserve_workspace` 里经过 all_reduce 统一了，各 rank 可以一起走同一条
回退路径，建议二选一：

- 从已经测过的 `measurements` 里挑"当前能放下的最快一个"（需要把测量结果保存下来）；
- 或者在所有 rank 上自动重新校准一次。

### 校准时为空的 rank

`adamw.py:245-248`、`360-361`：校准时某个 rank 没有非空的本地分片，它的候选全部被替换成
`(1, 0)`，保存下来的方案也是 `(1, 0)`。以后这个 rank 一旦有了非空参数，就会报
`active rank has no calibrated AdamW workspace`。测试
`test_previously_empty_rank_requires_explicit_calibration` 说明这是有意为之。

不过候选列表各 rank 完全相同，只有空 rank 被替换成了 0，因此可以在空 rank 上另外保存全局选中的
那个方案，这样就不需要这个限制。在 FSDP2 下这种情况很少见（需要该 rank 的所有本地分片都为空），
优先级低。

## 4.【低】其他

- **计时噪声**：每个候选只有 1 次预热加 3 次测量，取中位数，容差 3%。对 LoRA 这种很快的 step，
  噪声可能远超 3%，选择结果接近随机。不过平局时会选显存更小的方案，影响有限。
  可以考虑在 step 很短时增加采样次数。
- **起始 slot 大小**：起点 M 是"最大参数分片的字节数"，但一个 slot 要分成两半，分别放 exp_avg 和
  exp_avg_sq（moments 和参数同 dtype），所以最小的候选一定会把最大参数切成两块。
  请确认是有意这样，还是本意是从 2M 起步（正好能完整装下最大参数）。
- **README 粒度**：新增的英文和中文段落写了很多内部实现细节，比如 zero-restore 的具体机制、
  64 MiB 余量、各项日志字段、version counter；英文段落的换行也不规整（有的行三百多字符，
  有的只有几十）。建议用户文档只保留：默认行为、显式 fraction 的含义、`autotune_workspace()`
  的用法、失败后怎么处理、额外需要多少 host 内存。其余内容移到 docstring。

---

## 附录：候选数估算脚本

```python
# PYTHONPATH=<repo>/src python estimate_candidates.py
from torchturbo.engine.backends.fsdp2.optimizers.adamw import AdamWSwapMixin as M

GiB = 1 << 30
cases = [
    ("LoRA", 16 << 10, int(0.95 * 60 * GiB), 64 << 20),
    ("7B full, 8 ranks", 155 << 20, int(0.95 * 40 * GiB), int(3.5 * GiB)),
]
for name, shard, budget, total_moments in cases:
    plans = M._workspace_candidates(shard, budget // 32 * 32, True)
    redundant = sum(1 for p in plans if p.slot_bytes // 2 >= total_moments // 2)
    print(
        name,
        "candidates=", len(plans),
        "trials=", 4 * len(plans),
        "max_capacity=%.1f GiB" % (max(p.capacity_bytes for p in plans) / GiB),
        "redundant=", redundant,
    )
```

输出：

```
LoRA candidates= 45 trials= 180 max_capacity=57.0 GiB redundant= 21
7B full, 8 ranks candidates= 17 trials= 68 max_capacity=38.0 GiB redundant= 7
```

---

## 第二轮复核

- 被评审：`origin/dev-chunk-optimizer` 的 `54eeedb`（Codex 根据本评审修改，同样基于 `cccd6de`）
- 相对 a09e1a9 改了 6 个文件，+30 / -13
- 复核日期：2026-11-26

### 改了什么

只处理了第 1 条中"sink 太大"这一半：

- **sink 大小**：从"最大候选的 capacity"（≈ 95% 空闲显存）改成 `min(最大 capacity, 2H)`，
  其中 `H = Σ align16(每个活跃参数的 moment 字节数)`（`adamw_workspace.py` 新增的
  `calibration_sink_bytes`）。
- **sink 布局**：不再照搬设备 slot 的布局，改为前半放所有槽的 exp_avg，后半放所有槽的
  exp_avg_sq（`_stage_views` 中的 `sink_offset = base // 2 + span.moment_offset`）。
- `run_workspace` 的 sink 大小校验、README 中英文的 host 内存描述，以及两个测试也相应更新。

### 正确性：没问题

- **为什么不越界**：sink 取 2H 时，最可能越界的是双槽方案中槽 1 的写入。`iter_chunks`
  按顺序填满 chunk，第 k 个 chunk 的起点在不分块的连续布局里正好是 `k × half_slot`，
  所以槽 1 的写入终点 ≤ H，不会越界；两个槽同时在途的写入区间也不重叠。
- **随机测试**：2948 组随机配置（参数大小、dtype 混合、单/双槽、sink 取 capacity 或取 2H）
  全部通过：不越界，exp_avg 不越过中线，相邻两个在途 chunk 的写入区间不重叠。
- **单测**：164 个通过（新增参数化后比 a09e1a9 多 2 个）；仍有 5 个失败，还是本机不能
  pin memory 的老问题。新测试用的 slot 大小 64 / 256 / 1024 正好覆盖三种情况：
  sink 取 capacity、sink 取 2H 且双槽多 chunk、单 chunk。

### 效果

| 场景 | a09e1a9 的 sink | 54eeedb 的 sink |
|---|---|---|
| LoRA | 57 GiB，实际占 64 GiB pinned | 约 64 MiB |
| 7B 全参 8 卡 | 38 GiB，实际占 64 GiB pinned | 约 3.5 GiB，实际占 4 GiB |

### 仍存在的问题

1. **第 1 条的另一半**：pinned 块释放后仍留在 host 缓存，会一直占到训练结束。现在体量和
   moments 本身同一量级（7B 8 卡是每卡 4 GiB，整机 32 GiB），问题小了很多，但仍值得释放。
2. **第 2 条完全没动**：扫描仍然一路到 95% 空闲显存。LoRA 的 45 个候选里仍有 21 个是冗余的，
   trial 仍会把显存分配到 57 GiB。README 里也还写着 "moment size does not cap the sweep"。
3. **第 3、4 条没动**：默认值改为自动校准、方案放不下时没有回退、空 rank 的限制、计时噪声、
   README 粒度。

### 代码细节

- **校验条件冗余**：`run_workspace` 里的
  `numel < plan.capacity_bytes and numel < calibration_sink_bytes(bound, plan.capacity_bytes)`
  中，后者不大于前者，整个条件等价于只判断后半句。
- **缺少说明**：docstring 里的 "Its two halves omit unused device capacity." 没说清新布局；
  `base // 2` 以及"槽 1 不会越过 H"这个不显然的性质都应该加注释，否则以后改分块逻辑时
  很容易被破坏。
- **风格**：`adamw.py:390` 一行有 111 列，`calibration_sink_bytes` 的续行缩进也不规整。
  仓库没配 formatter，只是风格问题。
- **README 措辞**：英文 "no larger than the smaller of …" 不准确，实际取的就是两者的较小值。

---

## 建议修复方案

我在 `54eeedb` 上实现了一版原型，补丁见同目录的
[`a09e1a9-adamw-workspace-fixes.patch`](a09e1a9-adamw-workspace-fixes.patch)
（8 个文件，+260 / -46），在 `dev-chunk-optimizer` 上可以直接 `git apply`：

```bash
git fetch origin
git checkout dev-chunk-optimizer
git show origin/dev-workspace-opus:docs/reviews/a09e1a9-adamw-workspace-fixes.patch | git apply
```

| 修复 | 对应问题 | 涉及文件 |
|---|---|---|
| A. 扫描在"单 chunk 槽容量"处截断 | 第 1、2 条 | `adamw_workspace.py`、`adamw.py` |
| B. 校准用过 sink 后释放 pinned host 缓存 | 第 1 条剩余部分 | `runtime/base.py`、`runtime/cuda.py`、`adamw.py` |
| C. 方案放不下时，所有 rank 一起回退到已测候选 | 第 3 条 | `adamw.py` |
| D. 校准时为空的 rank 也保存全局方案 | 第 3 条 | `adamw.py` |
| E. 54eeedb 的代码细节和 README | 第二轮"代码细节" | `adamw_workspace.py`、README |

**验证范围**：本机 CPU 版 torch 2.10 的 venv。`test_adamw_pipeline.py`、
`test_adamw_workspace.py`、`test_adamw_workspace_distributed.py`（真实 Gloo 双进程）、
`test_cpu_offload_optimizer.py`、`tests/engine/test_config.py`、`tests/runtime` 共 180 个通过；
5 个失败仍是 `test_cpu_offload_optimizer.py` 里 pin memory 的环境问题，和 a09e1a9、54eeedb 相同。
上面的 sink 随机测试在补丁后同样通过。**GPU/NPU 上的真实路径、NPU 的 host 缓存接口和真实计时
都没有验证。**

### 修复 A：扫描在单 chunk 槽容量处截断（第 1、2 条）

**思路**：设 `S` 为"能把本 rank 全部活跃 moments 一次装下的最小 slot"。slot ≥ S 时只产生一个
chunk，传输量、kernel 数和计算量都和 slot = S 时相同，只是多占显存。所以：

- 单槽候选的上限取 `min(budget, S_max)`，`S_max` 是各 rank 的 S 取 MAX；
- 双槽候选的上限取 `S_max - 32`：slot ≥ S_max 时最大的 rank 只有一个 chunk，双槽没有任何
  overlap；
- sink 仍用 Codex 的 `min(最大 capacity, 2H)`。截断后最大候选不超过 2·S_max，sink 自然
  受 moment 总量约束，第 1 条的体量问题从根上解决。

**S 的计算**和 `iter_chunks` 的对齐规则完全一致，正好就是 Codex 的 `2H`，所以抽成一个函数，
`calibration_sink_bytes` 也改用它：

```python
def single_chunk_slot_bytes(bound: Sequence[BoundParameter]) -> int:
    """Return the smallest slot that holds every active moment pair in one chunk."""
    half = 0
    for param in bound:
        element_bytes = max(param.exp_avg.element_size(), param.exp_avg_sq.element_size())
        half += DeviceWorkspace.align(param.numel * element_bytes)
    return 2 * half


def calibration_sink_bytes(bound: Sequence[BoundParameter], capacity: int) -> int:
    return min(capacity, single_chunk_slot_bytes(bound))
```

**不增加 collective**：S 和原来的 `shard` 放进同一个 2 元素张量，一次 MAX 归约：

```python
shard = torch.zeros(2, dtype=torch.int64, device=runtime.device)
...
shard[0] = max((param.local_param.nbytes for param in bound), default=0)
shard[1] = single_chunk_slot_bytes(bound)
...
self._reduce_workspace_value(shard, dist.ReduceOp.MAX)
max_shard, full_slot = shard.tolist()
...
shared_candidates = self._workspace_candidates(max_shard, budget, bool(overlap), full_slot)
```

`_workspace_candidates` 增加一个可选参数 `full_slot`（默认 `None`，保持旧行为）：

```python
for slots in (1, 2) if overlap else (1,):
    endpoint = budget // slots // alignment * alignment
    if full_slot is not None:
        limit = max(full_slot, start) if slots == 1 else full_slot - alignment
        endpoint = min(endpoint, limit)
    if endpoint < start:
        continue
```

`full_slot ≥ 2·align16(M) ≥ align32(M) = start`，所以单槽一定至少有一个候选，
`max(full_slot, start)` 只是防御。

**效果**（附录脚本加上 `full_slot` 参数后的结果）：

| 场景 | 修复前候选 / trial | 修复前最大显存 | 修复后候选 / trial | 修复后最大显存 | 修复后 sink |
|---|---|---|---|---|---|
| LoRA（moments 共约 64 MiB） | 45 / 180 | 57 GiB | 26 / 104 | 128 MiB | ≤ 64 MiB |
| 7B 全参 8 卡（moments 每卡约 3.5 GiB） | 17 / 68 | 38 GiB | 12 / 48 | 7 GiB | ≤ 3.5 GiB |

**测试改动**：

- 新增 `test_candidates_stop_at_one_chunk_and_need_two_chunks_for_overlap`：单槽停在 S，
  双槽停在 S 之下；单个参数时双槽候选全部跳过。
- 新增 `test_single_chunk_slot_is_the_smallest_one_chunk_slot`：混合 dtype 下，slot = S 时
  `iter_chunks` 恰好产生 1 个 chunk，slot = S - 32 时产生 2 个。
- `test_calibration_uses_slowest_rank_time` 的期望候选从 32…2048 变为 32 / 64 / 128。
- 分布式用例 `test_workspace_calibration_with_different_rank_demands[empty-peer]` 中，
  rank 0 的最大候选从 640（预算上限）变为 512（rank 0 的 S），并断言两个 rank 保存同一方案。

### 修复 B：校准后释放 pinned host 缓存（第 1 条剩余部分）

`runtime.empty_cache()` 只清显存。`DeviceRuntime` 新增一个默认什么都不做的钩子，
CUDA 实现调用 PyTorch 的 host 缓存释放接口：

```python
# runtime/base.py
def empty_host_cache(self) -> None:
    """Release cached pinned host blocks that are no longer in use."""

# runtime/cuda.py
def empty_host_cache(self) -> None:
    # torch.cuda.empty_cache() leaves the pinned host allocator untouched.
    accelerator = getattr(torch, "accelerator", None)
    release = getattr(accelerator, "empty_host_cache", None) or getattr(
        torch._C, "_host_emptyCache", None
    )
    if self.available() and release is not None:
        release()
```

`_autotune_workspace` 的 `finally` 中，只在确实分配过 sink 时调用：

```python
saved_params.clear()
if cpu_sink is not None:
    # The caching host allocator would otherwise keep the pinned
    # sink until exit. Every trial has synchronized its copies.
    cpu_sink = None
    runtime.empty_host_cache()
```

- **安全性**：host 缓存只释放空闲链表里、且记录的 stream 事件都已完成的块。正在使用的 CPU
  moments 不受影响；每次 trial 结束都做过 `runtime.synchronize()`。
- **代价**：其他已缓存的空闲 pinned 块也会被一并释放，下次分配需要重新 pin。
  校准只发生一次，可以接受。
- **接口**：`torch._C._host_emptyCache()` 是私有接口（2.5+，仅 CUDA）；
  `torch.accelerator.empty_host_cache()` 是跨设备的公开接口（2.12+）。
- **NPU 待确认**：NPU 走基类的空实现。需要确认 torch_npu 是否提供等价接口，或者 2.12+ 的
  `torch.accelerator.empty_host_cache()` 是否覆盖 NPU。确认后在 `runtime/npu.py` 里实现即可。
- 新增测试 `test_nonzero_calibration_releases_pinned_host_cache_once`：零态校准不释放，
  非零态校准只释放一次，之后的正常 step 不释放。

### 修复 C：方案放不下时，所有 rank 一起回退到已测候选（第 3 条）

**思路**：校准时把测量结果 `(全局方案, 最慢 rank 的中位耗时, 全部 rank 的显存总量)` 保存下来，
这份数据在各 rank 上完全相同。真实 step 中如果某个 rank 放不下调好的方案，所有 rank 用同一组
全局约束从测量结果里重新选，规则和校准时相同（最快 3% 以内选显存最小的）。

**正常路径不增加 collective**：利用 `_reserve_workspace` 已有的状态归约，把状态从 0/1 扩成三态：

```python
status.fill_(0 if error is not None else 1 if refit else 2)
self._reduce_workspace_value(status, dist.ReduceOp.MIN)
if status.item() == 1:
    return None            # 有 rank 需要回退；本 rank 已分配的 workspace 直接丢弃
if not status.item():
    raise ...              # 和原来一样，任一 rank 出错则全部报错
```

`_acquire_workspace` 中，放不下不再直接报错，而是标记 `refit`；归约结果为 1 时，所有 rank
一起进入回退：

```python
refit = target.slot_bytes < minimum or target.capacity_bytes > maximum
...
workspace = self._reserve_workspace(plan.capacity_bytes, status=status, error=error, refit=refit)
if status.item() == 1:
    # Rare path: some rank cannot use the tuned plan. Every rank reaches it.
    plan, error = self._refit_workspace_plan(minimum, maximum), None
    try:
        self._get_pipeline(plan)
    except Exception as pipeline_error:
        error = pipeline_error.with_traceback(None)
    workspace = self._reserve_workspace(plan.capacity_bytes, status=status, error=error)
```

`_refit_workspace_plan` 用一次 2 元素 MIN 归约同时得到"各 rank 最小 slot 的最大值"和
"各 rank 显存上限的最小值"，然后在测量结果里选：

```python
bounds = torch.tensor([-minimum, maximum or inactive], dtype=torch.int64, device=...)
self._reduce_workspace_value(bounds, dist.ReduceOp.MIN)
shared_minimum, shared_maximum = -bounds[0].item(), bounds[1].item()
fitting = [
    entry for entry in self._workspace_measurements
    if entry[0].slot_bytes >= shared_minimum and entry[0].capacity_bytes <= shared_maximum
]
if not fitting:
    raise RuntimeError("... no longer fits ... and no measured candidate fits; recalibrate on all ranks")
selected = self._select_measurement(fitting)      # 和校准共用同一个选择规则
self._tuned_workspace_plan = selected[0]          # 粘性降级，后续 step 回到正常路径
return selected[0] if maximum else WorkspacePlan(1, 0)
```

- **一致性**：所有输入都是全局归约值或各 rank 相同的测量结果，所以各 rank 一定选中同一方案，
  或者一起报错。回退发生在 `_advance_steps` 之前，失败时 step 计数和参数都不变。
- **代价**：只有回退那一步多 2 次小 collective 和一次重新分配。
- **取舍：回退是粘性的**：降级后不会自动升回原方案。这样做的好处是之后每一步都走正常路径；
  如果希望显存恢复后升回去，需要保留原方案并每步多判断一次。**请作者决定。**
- **测试**：
  - 新增 `test_tuned_plan_exceeding_live_budget_falls_back_to_measured_plan`：本地放不下时
    选中能放下且最快的测量方案，step 正常完成，下一步直接使用降级后的方案。
  - 新增 `test_fallback_uses_peer_bounds_so_every_rank_selects_one_plan`：模拟另一个 rank 放不下，
    验证本 rank 按对方的上限选方案，collective 顺序为
    "分配状态 → 共享边界 → 回退分配状态 → 更新状态"。
  - 保留 `test_tuned_plan_exceeding_live_budget_is_rejected`：没有任何测量结果时，仍报
    "no longer fits"。

### 修复 D：校准时为空的 rank 也保存全局方案（第 3 条）

空 rank 仍按 `(1, 0)` 参与每次 trial、不分配显存，但测量结果记录全局方案，保存下来的也是
全局选中的方案。返回值和 `_tuned_workspace_bytes` 仍是本 rank 的实际容量（空 rank 为 0）：

```python
candidates = shared_candidates if minimum else (WorkspacePlan(1, 0),) * len(shared_candidates)
for plan, shared_plan in zip(candidates, shared_candidates):
    ...
    measurements.append((shared_plan, elapsed, control.item()))
...
self._tuned_workspace_plan = selected[0]
selected_capacity = selected[0].capacity_bytes if minimum else 0
```

这个 rank 以后有了非空参数时，直接使用全局方案；放不下就走修复 C 的回退，不再要求手动
重新校准。`"active rank has no calibrated AdamW workspace"` 的检查保留，只作为防御。
测试 `test_previously_empty_rank_requires_explicit_calibration` 是手动设置 `(1, 0)` 的，
仍然通过。

### 修复 E：54eeedb 的代码细节和 README

- `run_workspace` 的 sink 校验简化为
  `cpu_sink.numel() % 32 or cpu_sink.numel() < calibration_sink_bytes(bound, plan.capacity_bytes)`。
- `_stage_views` 的 sink 分支加注释，说明布局和"槽 k 从 `k × half_slot` 开始、不越过 S/2"的性质。
- README 中英文：删除 "moment size does not cap the sweep"，改为说明扫描在单 chunk 槽容量处
  停止、双槽停在它之下；补充"空 rank 保存全局方案"；把"不再适用时须显式重新校准"改为
  修复 C 的回退行为。

### 需要作者决定、补丁里没有改的事项

1. **默认值**：`optimizer_memory_fraction` 默认从 `0.8` 改为 `None`（自动校准）。
   修复 A 之后，trial 数和显存峰值都明显下降。我的建议是保留自动校准作为默认，但在 PR 描述或
   changelog 里写明：首步（以及每次 resume 后的首步）会多跑一轮校准；不希望校准时可显式设置
   `optimizer_memory_fraction: 0.8` 恢复旧行为。
2. **计时噪声**：每个候选 1 次预热加 3 次测量，容差 3%。LoRA 这类很快的 step，噪声可能超过 3%。
   建议改为按时长采样：至少 3 次，且累计时长达到某个下限（例如 50 ms）为止。这会改变各 rank 的
   trial 次数，需要先归约一个各 rank 相同的采样次数，所以没有放进补丁。
3. **起始 slot**：起点 M 是最大参数分片的字节数，但一个 slot 要分两半放 exp_avg 和 exp_avg_sq，
   所以最小候选一定会把最大参数切成两块。请确认是有意这样（多测一档更细的切分），还是本意是从
   2M 起步。
4. **README 粒度**：仍建议用户文档只保留默认行为、显式 fraction 的含义、`autotune_workspace()`
   的用法、失败后怎么处理和额外的 host 内存，实现细节移到 docstring。补丁只改了与修复相关的句子。
5. **回退是否粘性**：见修复 C。
