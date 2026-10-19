审查日期：2026-10-19。本文记录 TorchTurbo / NPU 与 Megatron-Swift / GPU 的 DPO 输入差异、对训练目标的影响，以及两个调试分支的实验依据。

**已确认的结论。** 同一条真实偏好对在两端日志中相差 8 个输入 token 和 418 个监督 token。两项差异都可以由公开 Swift 的模板和版本行为解释，并已在 CPU 上逐项复现 chosen/rejected 的完整 input_ids、labels。它们在模型 forward 之前产生；目前还没有证据将全部业务精度下降归因于这两项差异。

按用户确认，初始权重一致，两份 TorchTurbo 代码视为相同。本次对照关注实际编码结果及训练行为。

**日志中的输入事实。**

| 项目 | 上传的 GPU 日志 | NPU 日志 |
| --- | --- | --- |
| Swift 版本 | 4.6.0.dev0 | 4.4.2 |
| template | qwen3_8 | qwen3_5 |
| agent_template | qwen3_5 | qwen3_5 |
| loss_scale | last_round+ignore_empty_think | last_round+ignore_empty_think |
| chosen 输入 token | 77127 | 77119 |
| rejected 输入 token | 77111 | 77103 |
| chosen 有效监督 token | 1877 | 2295 |
| rejected 有效监督 token | 1861 | 2279 |

原始文件为 [gpu_kubemaker_log.csv](gpu_kubemaker_log.csv) 和 [log-0.txt](log-0.txt)。GPU 版本记录在第 792 行，NPU 版本记录在第 2871 行。GPU chosen input_ids/labels 分别位于第 5795/5696 行，NPU 对应数组位于第 3625/6337 行。这里的行号按原文件 LF 计数，解析 CSV 时保留原始行号。

[verify_logged_inputs_cpu.py](verify_logged_inputs_cpu.py) 仅用 Python 标准库即可核对四组完整数组、差异位置及公共前缀：

```bash
python dpo-debug/verify_logged_inputs_cpu.py
```

**8 个输入 token 的根因：历史 thinking 策略不同。**

公开 Swift 在 2026-08-14 的 [ab726e9](https://github.com/modelscope/ms-swift/commit/ab726e9d445a6520a70df2c831177d46adb1f589) 中加入 Qwen3.8 模板。该模板继承 Qwen3.5，但注册元数据设置了 `preserve_thinking=True`，并调整了基类补空 think 的起始范围。

在本次 `last_round`、`add_non_thinking_prefix=True` 配置下，旧版 qwen3_5 删除历史 thinking，只给最后一次 user 提问之后的 assistant 补前缀；qwen3_8 保留历史 thinking，并允许全部 assistant 轮次补前缀。相关代码见 [qwen3_8 注册](https://github.com/modelscope/ms-swift/blob/cf6316ca2b827d849b0ab054e25aa10551413dcc/swift/template/templates/qwen.py#L700) 和 [前缀添加逻辑](https://github.com/modelscope/ms-swift/blob/cf6316ca2b827d849b0ab054e25aa10551413dcc/swift/template/base.py#L1229)。

样本包含两个历史 assistant 消息，各增加 `<think>\n\n</think>\n\n`，对应 `[248068, 271, 248069, 271]` 四个 token。两段 labels 都为 -100。删除这两段后，GPU 与 NPU 的其余 input_ids 完全相同，两分支均成立。

`ignore_empty_think` 控制空 think 是否参与 loss，不会从输入中删除这些 token。被 mask 的 token 仍参与前向计算，并可能改变后续 attention、recurrent state、位置与条件概率。因此，模板能影响训练结果；仅从 token 数量不能推断影响方向和业务精度变化幅度。

**418 个监督 token 的根因：last_round 的边界函数变了。**

公开 Swift 在 2026-08-16 的 [cf6316c](https://github.com/modelscope/ms-swift/commit/cf6316ca2b827d849b0ab054e25aa10551413dcc) 中修改 `get_last_user_round`：

```python
# Swift 4.4.2
if messages[i]['role'] == 'user':
    return i

# Public Swift after the August 16 change
if messages[i]['role'] in ('user', 'tool'):
    return i
```

[LossScale.__call__](https://github.com/modelscope/ms-swift/blob/f48847d23dbcd72ceb15fdbc5a1482cc7eb0359d/swift/loss_scale/base.py#L91) 本身在两个复现快照中相同，但它导入的函数改变后，最后一个 tool 返回也成为监督边界。相同的 `last_round` 参数字符串因而产生不同的 labels。

| 最后一次提问的消息顺序 | 旧版 NPU | 新版 GPU 对应行为 |
| --- | --- | --- |
| user：提问 | 从这里确定最后一轮 | 位于监督边界之前 |
| assistant：工具调用 | 监督，418 个 token | mask |
| tool：工具返回 | mask | 从这里确定最后一轮，返回内容仍 mask |
| assistant：最终答案 | 监督 | 监督 |

按从 0 开始、左闭右开的区间计，NPU 多监督的是 `[71108, 71526)`。该区间包含工具调用正文和结束符，不包括空 think。chosen/rejected 中该段 token 及此前整个输入前缀完全相同。

这个共享函数也参与 thinking 处理和 rejected_response 构造。2026-09-28 的 [38aa3b7](https://github.com/modelscope/ms-swift/commit/38aa3b79aeb31b7b2623f26883eb5c7dcd5c0485) 为 rejected_response 构造显式指定了 `include_tool=False`。为了单独验证 mask，调试分支没有全局替换该函数。

**额外监督对 DPO 和 RPO 的影响不同。**

先固定 input_ids、模型状态和前向条件，只改变 loss mask。对于当前求和形式的标准 sigmoid DPO，这条样本的公共工具段在 chosen-minus-rejected 中理论上抵消，reference 中也一样：

```text
logp(chosen)   = common_tool_logp + chosen_answer_logp
logp(rejected) = common_tool_logp + rejected_answer_logp

logp(chosen) - logp(rejected)
    = chosen_answer_logp - rejected_answer_logp
```

在因果、确定性前向和相同公共前缀的条件下，这个抵消也适用于参数梯度，不要求训练过程中 policy 一直等于 reference。实际不同布局、低精度计算或随机性下是否完全抵消仍需数值验证。其他偏好对如果两分支的工具轨迹不同，就不能沿用该结论；带长度归一化的其他目标也需单独分析。

RPO 是 chosen 上的平均负对数似然，不做 chosen-minus-rejected 抵消。[TorchTurbo 的实现](../src/torchturbo/adapters/msswift/trainer.py) 使用 `nll_loss = -chosen_logps_sum / num_chosen_tokens`，再由 Swift 按 rpo_alpha 加到 DPO loss。

以这条偏好对单独计算、保持输入相同且 alpha 都为 0.1 为例，令 `L_answer`、`L_tool` 分别表示答案和工具调用的平均 NLL：

```text
Only final answer:
    RPO = 0.1 * L_answer

Tool call and final answer:
    RPO = 0.1 * (1877 * L_answer + 418 * L_tool) / 2295
        = 0.08179 * L_answer + 0.01821 * L_tool
```

因此最终答案在 RPO 中的权重下降约 18.2%，同时增加工具调用监督。18.2% 仅指这条样本的 RPO 答案项权重；它不对应整体梯度、学习率或业务精度的相同比例变化。额外监督可能帮助工具调用，也可能改变回答训练的方向，需要通过评测判断。

NPU 的 rpo_alpha=0.1 已由日志确认；原 27B H20 脚本也配置了 0.1，但此次上传 GPU 日志未记录有效 alpha，不能据此认定该日志对应训练也使用了 0.1。packing 时 NLL 在当前 microbatch 的所有 chosen 有效 token 上取均值，全量训练的权重需要按实际包组成计算。

**CPU 已完成的验证与两个分支。**

未经修改的 Swift v4.4.2（f48847d）+ qwen3_5 完整匹配 NPU 四组数组；公开 2026-08-16 快照 cf6316c + qwen3_8 完整匹配 GPU 四组数组。后者版本号为 4.5.0.dev0，已经包含上述两项改动，早于 GPU 训练。该结果复现了编码行为，没有证明 GPU 私有 checkout 的全部源码与公开快照相同。

两个 TorchTurbo 调试分支均独立基于 `docs/dpo-debug-scripts@26d78e5`，继续使用 Swift 4.4.2 和 template=qwen3_5：

| 实验 | chosen / rejected 输入 token | chosen / rejected 监督 token | 单独改变的行为 |
| --- | ---: | ---: | --- |
| 原 NPU 基线 | 77119 / 77103 | 2295 / 2279 | 无 |
| debug/dpo-history-think | 77127 / 77111 | 2295 / 2279 | 保留历史 thinking，并给历史 assistant 补前缀 |
| debug/dpo-last-tool-mask | 77119 / 77103 | 1877 / 1861 | 工具返回只在 loss scaling 阶段被视为轮次边界 |
| GPU 日志，供编码核对 | 77127 / 77111 | 1877 / 1861 | 同时存在两项行为差异 |

历史分支完整匹配 GPU 的 input_ids，有效监督 token 序列与 NPU 一致。mask 分支完整保留 NPU 的 input_ids，只将工具段的 418 个 labels 设为 -100。这两项检查均覆盖 chosen/rejected，且从任一侧日志恢复消息后都成立。

[verify_dpo_debug_cpu.py](verify_dpo_debug_cpu.py) 还检查了多次工具调用、非空历史 thinking、模板注册幂等性、序列化、SFT/推理及其他 loss strategy。mask 分支额外检查 rejected_response 字符串/列表形式的输入不变，并保留显式 loss 覆盖的优先级。运行命令和 NPU 包准备方式见 [DEBUG_BRANCH.md](DEBUG_BRANCH.md)。

CPU 环境为 PyTorch 2.7.0、Transformers 5.9.0，使用公开 Qwen3.5 tokenizer/processor 元数据。只替换了模板文件的包日志导入以避开整个加速引擎，没有加载模型权重、执行 forward 或验证设备算子。重构的是已渲染消息，未恢复原始 JSON 的全部字段，也未核对生产词表的所有 token。

**训练对照与结论边界。**

两个分支分别和原 NPU 基线比较，保持权重、数据、Swift 版本、RPO、beta、优化器、训练时间轴、推理模板及评测设置一致。不要将两个分支叠加后再用结果判断单项影响。

先确认实际日志中的实验标记、模板 SHA256 和完整输入/labels。原 NPU 脚本会解压固定代码归档，两个分支已加入单文件覆盖与实验标记校验；仅切换 Git 分支不足以保证旧提交包使用了新模板。

历史分支可能因输入长度变化影响超长过滤和 packing 分组。严格隔离上下文影响时，需要固定保留的样本 ID、每包的有序样本列表以及每次更新的 batch；相同 seed 不保证这些内容相同。mask 分支不改变输入长度，但仍应核对实际消费的数据。

在相同 batch 上先比较 DPO 项、RPO 项和各自梯度，再按原训练时间轴完成训练，用同一评测比较对应 checkpoint。多监督 token 是否改变业务精度、历史 think 是否解释回退，需要这些实验结果，而不能由编码计数直接推断。

此次上传 GPU 日志对应 Qwen3.8-Flash-Next MoE、GBS4、2 epochs、最大长度 98304，末次 checkpoint 为 581；它与最初业务表对应的 27B H20 / checkpoint-40 场次不一致。因此，上述日志可作为输入差异的证据，不能直接用来归因原业务指标的全部差距。目前尚未执行两个调试分支的 NPU 训练或业务评测。
