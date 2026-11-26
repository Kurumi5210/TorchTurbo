# Review：a09e1a9 `feat(optimizer): tune chunked AdamW swap workspaces`

- 被评审 commit：`a09e1a9ba68139a1d31568a9b8a68e156f5ace88`（父 commit `cccd6de`）
- 评审日期：2026-11-26
- 涉及范围：`optimizers/adamw.py`（重写）、新增 `optimizers/adamw_workspace.py`、`config.py`、`muon.py`、README 及 3 个新测试文件

下文的行号都以 a09e1a9 为准；没有写目录的文件都在
`src/torchturbo/engine/backends/fsdp2/optimizers/` 下。

## 结论

主体实现可靠，但在 checkpoint resume 场景下有一个会撑爆 host 内存的问题（第 1 条），
建议合入前修掉。第 2 条和第 1 条可以用同一处改动修复。第 3 条是设计层面的取舍，
需要和作者讨论。

| # | 严重程度 | 问题 | 位置 |
|---|---|---|---|
| 1 | 高 | resume 时 pinned sink 的大小约等于 95% 空闲显存，和模型状态大小无关，校准后也不释放 | `adamw.py:389-392` |
| 2 | 中 | 扫描范围不按 moment 总量截断，产生大量冗余候选 | `adamw.py:265-285` |
| 3 | 中 | 默认行为改变，方案不再适用时没有自动回退 | `adamw.py:245-253`、`config.py:71` |
| 4 | 低 | 计时噪声、起始 slot 大小、README 粒度 | 见下文 |

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
